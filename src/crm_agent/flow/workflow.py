from __future__ import annotations

from typing import TypedDict, Any, Dict, List
from collections import defaultdict

from langgraph.graph import StateGraph, END

from crm_agent.db.engine import SessionLocal
from crm_agent.db.repo import Repo
from crm_agent.services.targeting import build_target
from crm_agent.rag.retriever import RagRetriever, build_context_text

ST_BRIEF = "BRIEF"
ST_TARGET_INPUT = "TARGET_INPUT"
ST_TARGET_AUDIENCE = "TARGET_AUDIENCE"
ST_TARGET = "TARGET"
ST_RAG = "RAG"
ST_TEMPLATE_CANDIDATES = "TEMPLATE_CANDIDATES"
ST_COMPLIANCE = "COMPLIANCE"
ST_SELECTED_TEMPLATE = "SELECTED_TEMPLATE"
ST_EXECUTION_RESULT = "EXECUTION_RESULT"


try:
    from crm_agent.agents.template_agent import generate_template_candidates
except Exception:
    generate_template_candidates = None

try:
    from crm_agent.agents.compliance import validate_candidates
except Exception:
    validate_candidates = None

try:
    from crm_agent.agents.execution_agent import generate_final_message
except Exception:
    generate_final_message = None


class CRMState(TypedDict, total=False):
    run_id: str
    channel: str
    tone: str

    brief: dict
    target_input: dict
    target_audience: dict

    target: dict
    rag: dict
    candidates: dict
    compliance: dict

    selected_template: dict
    execution_result: dict


def _repo() -> Repo:
    db = SessionLocal()
    return Repo(db)


def _close_repo(repo: Repo) -> None:
    try:
        repo.db.close()
    except Exception:
        pass


def _build_rag_evidence(
        retrieved: Dict[str, Any],
        max_each_source: int = 3,
        max_text_chars: int = 800,
) -> List[Dict[str, Any]]:
    matches = retrieved.get("matches", []) or []
    per_source = defaultdict(int)

    evidence: List[Dict[str, Any]] = []
    for m in matches:
        md = (m.get("metadata") or {})
        source = md.get("source", "UNKNOWN")
        section = md.get("section", "")
        chunk_id = md.get("chunk_id", "")
        text = (md.get("text") or "").strip()

        if not text:
            continue

        if per_source[source] >= max_each_source:
            continue
        per_source[source] += 1

        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "…"

        evidence.append(
            {
                "id": m.get("id", ""),
                "score": float(m.get("score", 0.0)),
                "source": source,
                "section": section,
                "chunk_id": chunk_id,
                "text": text,
            }
        )

    return evidence


def _safe_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _summarize_target_input(target_input: dict) -> str:
    gender = target_input.get("gender") or []
    age_bands = target_input.get("age_bands") or []
    skin_types = target_input.get("skin_types") or []
    concern_keywords = target_input.get("concern_keywords") or []

    parts = []
    if gender:
        parts.append(f"gender={gender}")
    if age_bands:
        parts.append(f"age_bands={age_bands}")
    if skin_types:
        parts.append(f"skin_types={skin_types}")
    if concern_keywords:
        parts.append(f"concern_keywords={concern_keywords}")

    return " / ".join(parts) if parts else "NO_FILTERS(전체 대상)"


def node_load_brief(state: CRMState) -> CRMState:
    repo = _repo()
    try:
        run_id = state["run_id"]
        run = repo.get_run(run_id)
        if not run:
            raise RuntimeError(f"run_id not found: {run_id}")

        brief_h = repo.get_latest_handoff(run_id, ST_BRIEF)
        brief = brief_h["payload_json"] if brief_h else run.get("brief_json", {"goal": run.get("campaign_goal")})

        channel = state.get("channel") or run.get("channel") or "PUSH"
        tone = state.get("tone") or "amoremall"

        ti_h = repo.get_latest_handoff(run_id, ST_TARGET_INPUT)
        ta_h = repo.get_latest_handoff(run_id, ST_TARGET_AUDIENCE)
        target_input = ti_h["payload_json"] if ti_h else {}
        target_audience = ta_h["payload_json"] if ta_h else {}

        return {
            **state,
            "brief": brief,
            "channel": channel,
            "tone": tone,
            "target_input": _safe_dict(target_input),
            "target_audience": _safe_dict(target_audience),
        }
    finally:
        _close_repo(repo)


def node_targeting(state: CRMState) -> CRMState:
    repo = _repo()
    try:
        run_id = state["run_id"]
        brief = state.get("brief") or {}
        channel = state.get("channel") or "PUSH"
        tone = state.get("tone") or "amoremall"

        target_input = _safe_dict(state.get("target_input") or {})
        target_audience = _safe_dict(state.get("target_audience") or {})

        base_target = build_target(repo.db, brief=brief, channel=channel, tone=tone)
        base_target = _safe_dict(base_target)

        resolved = _safe_dict(target_audience.get("resolved") or {})
        audience_count = int(target_audience.get("count") or 0)
        audience_user_ids = target_audience.get("user_ids") or []
        audience_sample = target_audience.get("sample") or []

        target = {
            **base_target,
            "target_input": target_input,
            "audience": {
                "count": audience_count,
                "user_ids": audience_user_ids,
                "sample": audience_sample,
                "resolved": resolved,
            },
            "target_input_summary": _summarize_target_input(target_input),
        }

        repo.create_handoff(run_id, ST_TARGET, target)
        repo.update_run(run_id, channel=channel, step_id="S2_TARGET")
        return {**state, "target": target}
    finally:
        _close_repo(repo)


def node_rag(state: CRMState) -> CRMState:
    repo = _repo()
    try:
        run_id = state["run_id"]
        brief = state.get("brief") or {}
        target = state.get("target") or {}
        channel = state.get("channel") or "PUSH"
        tone = state.get("tone") or "amoremall"

        goal = brief.get("goal", "") or brief.get("campaign_goal", "")

        target_query = target.get("target_query", {}) or {}
        target_summary = target.get("summary", "") or ""
        target_input_summary = target.get("target_input_summary", "") or ""
        audience = _safe_dict(target.get("audience") or {})
        audience_count = audience.get("count", 0)
        resolved = _safe_dict(audience.get("resolved") or {})

        query = (
            "너는 CRM 마케터/카피라이팅 어시스턴트다.\n"
            "아래 조건에 맞는 메시지 템플릿을 만들 때 참고할 근거를 찾아라.\n\n"
            f"[캠페인 목적]\n- {goal}\n\n"
            f"[채널/톤]\n- channel={channel}\n- tone={tone}\n\n"
            f"[타겟]\n"
            f"- base_target_query={target_query}\n"
            f"- base_target_summary={target_summary}\n"
            f"- selected_filters={target_input_summary}\n"
            f"- audience_count={audience_count}\n"
            f"- concern_mapping={resolved}\n\n"
            "[요청]\n"
            "- 브랜드 가이드(톤/문장 규칙)\n"
            "- 채널 정책(길이/구성/CTA 규칙)\n"
            "- 컴플라이언스(금지 표현/완곡 표현)\n"
            "- 유사 캠페인 포맷/베스트 프랙티스\n"
            "주의: 상품/혜택/가격은 확정하지 말고 슬롯으로 남기는 방향의 가이드만 찾아라."
        )

        retriever = RagRetriever()
        retrieved = retriever.retrieve(query=query, filters=None, top_k=10)

        context = build_context_text(retrieved, max_each=3)
        evidence = _build_rag_evidence(retrieved, max_each_source=3, max_text_chars=800)

        rag_payload = {
            "query": query,
            "top_k": 10,
            "channel": channel,
            "tone": tone,
            "goal": goal,
            "base_target_query": target_query,
            "base_target_summary": target_summary,
            "target_input_summary": target_input_summary,
            "audience_count": audience_count,
            "concern_mapping": resolved,
            "evidence": evidence,
            "context": context,
        }

        repo.create_handoff(run_id, ST_RAG, rag_payload)
        repo.update_run(run_id, step_id="S3_RAG")
        return {**state, "rag": rag_payload}
    finally:
        _close_repo(repo)


def node_candidates(state: CRMState) -> CRMState:
    """
    k=5 고정(후보 5개 유지)
    다양성은 template_agent에서 해결
    """
    repo = _repo()
    try:
        run_id = state["run_id"]
        brief = state.get("brief") or {}
        rag = state.get("rag") or {}
        target = state.get("target") or {}
        channel = state.get("channel") or "PUSH"
        tone = state.get("tone") or "amoremall"

        if generate_template_candidates is None:
            candidates = {
                "candidates": [
                    {
                        "template_id": "T001",
                        "title": "A1_초간단_한줄형(LOCAL_FALLBACK)",
                        "body_with_slots": "{customer_name}님, {product_name}\n{offer}\n{cta}",
                        "variants": [],
                    }
                ]
            }
        else:
            candidates = generate_template_candidates(
                brief=brief,
                channel=channel,
                tone=tone,
                rag_context=rag.get("context", ""),
                target=target,
                k=5,  # 후보 5개 유지
            )

        repo.create_handoff(run_id, ST_TEMPLATE_CANDIDATES, candidates)
        repo.update_run(run_id, step_id="S4_CANDS")
        return {**state, "candidates": candidates}
    finally:
        _close_repo(repo)


def node_compliance(state: CRMState) -> CRMState:
    repo = _repo()
    try:
        run_id = state["run_id"]
        cands = (state.get("candidates") or {}).get("candidates", [])

        if validate_candidates is None:
            results = []
            for c in cands:
                body = c.get("body_with_slots", "")
                status = "PASS"
                reasons = []
                if "100% 효과" in body or "완치" in body:
                    status = "FAIL"
                    reasons.append("과장/확정 표현 가능성")
                results.append({"template_id": c.get("template_id"), "status": status, "reasons": reasons})
            compliance = {"results": results}
        else:
            compliance = validate_candidates(cands)

        repo.create_handoff(run_id, ST_COMPLIANCE, compliance)
        repo.update_run(run_id, step_id="S5_COMP")
        return {**state, "compliance": compliance}
    finally:
        _close_repo(repo)


def node_execute(state: CRMState) -> CRMState:
    repo = _repo()
    try:
        run_id = state["run_id"]
        brief = state.get("brief") or {}
        rag = state.get("rag") or {}
        target = state.get("target") or {}
        audience = _safe_dict((target.get("audience") or {}))

        selected = state.get("selected_template")
        if not selected:
            h = repo.get_latest_handoff(run_id, ST_SELECTED_TEMPLATE)
            if not h:
                raise RuntimeError("selected_template missing (state/DB 모두 없음)")
            selected = h["payload_json"]

        if generate_final_message is None:
            final_text = (selected.get("body_with_slots") or "")
            result = {
                "final_message": final_text,
                "used_template_id": selected.get("template_id"),
                "rag_used": rag.get("context", "")[:1500],
                "audience_count": audience.get("count", 0),
            }
        else:
            result = generate_final_message(
                brief=brief,
                selected_template=selected,
                rag_context=rag.get("context", ""),
            )

        repo.create_handoff(run_id, ST_EXECUTION_RESULT, result)
        repo.update_run(
            run_id,
            step_id="S6_EXEC",
            candidate_id=(selected.get("template_id") or "")[:16],
            rendered_text=result.get("final_message", ""),
        )
        return {**state, "execution_result": result}
    finally:
        _close_repo(repo)


def route_after_compliance(state: CRMState) -> str:
    if state.get("selected_template"):
        return "stage_execute"
    return END


def build_graph():
    g = StateGraph(CRMState)

    g.add_node("stage_load_brief", node_load_brief)
    g.add_node("stage_target", node_targeting)
    g.add_node("stage_rag", node_rag)
    g.add_node("stage_candidates", node_candidates)
    g.add_node("stage_compliance", node_compliance)
    g.add_node("stage_execute", node_execute)

    g.set_entry_point("stage_load_brief")
    g.add_edge("stage_load_brief", "stage_target")
    g.add_edge("stage_target", "stage_rag")
    g.add_edge("stage_rag", "stage_candidates")
    g.add_edge("stage_candidates", "stage_compliance")

    g.add_conditional_edges(
        "stage_compliance",
        route_after_compliance,
        {
            "stage_execute": "stage_execute",
            END: END,
        },
    )
    g.add_edge("stage_execute", END)

    return g.compile()


GRAPH = build_graph()


def run_until_candidates(run_id: str, channel: str, tone: str) -> Dict[str, Any]:
    init_state: CRMState = {"run_id": run_id, "channel": channel, "tone": tone}
    return GRAPH.invoke(init_state)


def run_with_selection(run_id: str, selected_template: dict) -> Dict[str, Any]:
    repo = _repo()
    try:
        repo.create_handoff(run_id, ST_SELECTED_TEMPLATE, selected_template)
        repo.update_run(run_id, step_id="S6_EXEC", candidate_id=(selected_template.get("template_id") or "")[:16])
    finally:
        _close_repo(repo)

    init_state: CRMState = {"run_id": run_id, "selected_template": selected_template}
    return GRAPH.invoke(init_state)
