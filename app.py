from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import sys
import time
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

import streamlit as st
from sqlalchemy import text, bindparam

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from components.crm_ui import crm_ui
from crm_agent.db.engine import SessionLocal
from crm_agent.db.repo import Repo
from crm_agent.flow.workflow import run_until_candidates


# -------------------------
# JSON safe
# -------------------------
def make_json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def _json_to_dict(v):
    """payload_json이 dict 또는 JSON string일 수 있어서 항상 dict로 정규화"""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}


def _table_exists(db, table: str) -> bool:
    try:
        q = text(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = DATABASE()
              AND table_name = :t
            """
        )
        return (db.execute(q, {"t": table}).scalar() or 0) > 0
    except Exception:
        return False


def _has_column(db, table: str, column: str) -> bool:
    try:
        row = db.execute(text(f"SHOW COLUMNS FROM {table} LIKE :col"), {"col": column}).fetchone()
        return row is not None
    except Exception:
        return False


# -------------------------
# Target mappings
# -------------------------
CONCERN_KEYWORD_TO_CATEGORY = {
    "민감성": "피부진정",
    "트러블": "피부진정",
    "탄력없음": "주름/탄력",
    "주름": "주름/탄력",
    "칙칙함": "미백/자외선차단",
    "건조함": "영양/보습",
    "모공": "블랙헤드/모공/피지",
    "고민없음": "미백/자외선차단",
}

CATEGORY_TO_DB_CONCERNS = {
    "피부진정": ["sensitivity", "redness", "barrier", "acne"],
    "주름/탄력": ["wrinkles"],
    "미백/자외선차단": ["pigmentation"],
    "영양/보습": ["hydration", "barrier"],
    "블랙헤드/모공/피지": ["pores"],
}


def _age_band_to_birthyear_ranges(age_bands: list[str]) -> list[tuple[int, int]]:
    cur = datetime.now().year
    ranges = []
    for b in (age_bands or []):
        b = str(b).strip()
        if b == "10대":
            ranges.append((cur - 19, cur - 10))
        elif b == "20대":
            ranges.append((cur - 29, cur - 20))
        elif b == "30대":
            ranges.append((cur - 39, cur - 30))
        elif b == "40대":
            ranges.append((cur - 49, cur - 40))
        elif b in ("50대+", "50대"):
            ranges.append((1900, cur - 50))
    return ranges


def resolve_concerns_from_keywords(keywords: list[str]) -> dict:
    keywords = [k for k in (keywords or []) if k in CONCERN_KEYWORD_TO_CATEGORY]
    categories = [CONCERN_KEYWORD_TO_CATEGORY[k] for k in keywords]

    seen = set()
    categories_uniq = []
    for c in categories:
        if c not in seen:
            seen.add(c)
            categories_uniq.append(c)

    db_vals = []
    for c in categories_uniq:
        db_vals.extend(CATEGORY_TO_DB_CONCERNS.get(c, []))

    seen2 = set()
    db_vals_uniq = []
    for v in db_vals:
        if v not in seen2:
            seen2.add(v)
            db_vals_uniq.append(v)

    return {
        "concern_keywords": keywords,
        "concern_categories": categories_uniq,
        "skin_concerns": db_vals_uniq,
    }


def _build_where_and_params(db, target_resolved: dict):
    gender_vals = target_resolved.get("gender") or []
    age_bands = target_resolved.get("age_bands") or []
    skin_vals = target_resolved.get("skin_types") or []
    concern_vals = target_resolved.get("skin_concerns") or []

    has_skin_type = _has_column(db, "user_features", "skin_type")
    has_concern = _has_column(db, "user_features", "skin_concern_primary")

    where_clauses = []
    params = {}
    bp = []

    if gender_vals:
        where_clauses.append("u.gender IN :genders")
        params["genders"] = gender_vals
        bp.append(bindparam("genders", expanding=True))

    ranges = _age_band_to_birthyear_ranges(age_bands)
    if ranges:
        ors = []
        for i, (y_min, y_max) in enumerate(ranges):
            ors.append(f"(u.birth_year BETWEEN :by_min_{i} AND :by_max_{i})")
            params[f"by_min_{i}"] = y_min
            params[f"by_max_{i}"] = y_max
        where_clauses.append("(" + " OR ".join(ors) + ")")

    if has_skin_type and skin_vals:
        where_clauses.append("uf.skin_type IN :skin_types")
        params["skin_types"] = skin_vals
        bp.append(bindparam("skin_types", expanding=True))

    if has_concern and concern_vals:
        where_clauses.append("uf.skin_concern_primary IN :skin_concerns")
        params["skin_concerns"] = concern_vals
        bp.append(bindparam("skin_concerns", expanding=True))

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    return where_sql, params, bp


def preview_target_count(db, target_resolved: dict) -> int:
    where_sql, params, bp = _build_where_and_params(db, target_resolved)
    q = text(f"""
        SELECT COUNT(*) AS cnt
        FROM users u
        LEFT JOIN user_features uf ON uf.user_id = u.user_id
        {where_sql}
    """).bindparams(*bp)
    return int(db.execute(q, params).scalar() or 0)


def fetch_target_user_ids(db, target_resolved: dict, limit_n: int = 500) -> dict:
    where_sql, params, bp = _build_where_and_params(db, target_resolved)

    q = text(f"""
        SELECT u.user_id
        FROM users u
        LEFT JOIN user_features uf ON uf.user_id = u.user_id
        {where_sql}
        ORDER BY u.user_id
        LIMIT :limit_n
    """).bindparams(*bp)

    params2 = dict(params)
    params2["limit_n"] = int(limit_n)
    rows = db.execute(q, params2).mappings().all()
    user_ids = [r["user_id"] for r in rows]

    q_count = text(f"""
        SELECT COUNT(*) AS cnt
        FROM users u
        LEFT JOIN user_features uf ON uf.user_id = u.user_id
        {where_sql}
    """).bindparams(*bp)
    total = int(db.execute(q_count, params).scalar() or 0)

    return {"total_count": total, "limit": int(limit_n), "user_ids": user_ids}


# -------------------------
# Home helpers: latest selected templates for run_ids
# -------------------------
def _fetch_latest_selected_for_runs(db, run_ids: list[str], limit_n: int = 50) -> list[dict]:
    if not run_ids:
        return []

    q = text(
        """
        WITH last_selected AS (
          SELECT run_id, MAX(created_at) AS max_at
          FROM handoffs
          WHERE stage='SELECTED_TEMPLATE' AND run_id IN :run_ids
          GROUP BY run_id
        )
        SELECT h.run_id, h.created_at, h.payload_json
        FROM handoffs h
        JOIN last_selected ls
          ON ls.run_id = h.run_id
         AND ls.max_at = h.created_at
        WHERE h.stage='SELECTED_TEMPLATE'
        ORDER BY h.created_at DESC
        LIMIT :limit_n
        """
    ).bindparams(bindparam("run_ids", expanding=True))

    rows = db.execute(q, {"run_ids": run_ids, "limit_n": int(limit_n)}).mappings().all()
    out = []
    for r in rows:
        out.append(
            {
                "run_id": r.get("run_id"),
                "created_at": make_json_safe(r.get("created_at")),
                "template": _json_to_dict(r.get("payload_json")),
            }
        )
    return out


# -------------------------
# Home: pending/approved/rejected 기준 변경 + 승인 템플릿 최신/전체 추가
# -------------------------
def fetch_home_data(db, show_all_pending: bool, show_all_approved: bool) -> dict:
    kpi_ctr = "4.2%"
    kpi_ctr_trend = "+0.8%"
    kpi_guard_blocked = 12

    if not _table_exists(db, "handoffs"):
        return {
            "kpi": {
                "ctr": kpi_ctr,
                "ctr_trend": kpi_ctr_trend,
                "guard_blocked": kpi_guard_blocked,
                "pending_candidate": 0,  # 승인/반려 대기
                "active_assets": 0,      # 승인된 템플릿
            },
            "registry": {"approved": 0, "deprecated": 0},
            "pending": {"latest": None, "all": []},
            "approved": {"latest": None, "all": []},
            "ui": {
                "show_all_pending": bool(show_all_pending),
                "show_all_approved": bool(show_all_approved),
            },
        }

    # pending = SELECTED_TEMPLATE은 있는데 APPROVAL이 아직 없는 run
    q_pending_cnt = text(
        """
        WITH has_selected AS (
          SELECT DISTINCT run_id
          FROM handoffs
          WHERE stage='SELECTED_TEMPLATE'
        ),
        has_approval AS (
          SELECT DISTINCT run_id
          FROM handoffs
          WHERE stage='APPROVAL'
        )
        SELECT COUNT(*) AS cnt
        FROM has_selected s
        LEFT JOIN has_approval a ON a.run_id = s.run_id
        WHERE a.run_id IS NULL
        """
    )
    pending_cnt = int(db.execute(q_pending_cnt).scalar() or 0)

    # latest pending 1개 = pending run 중 SELECTED_TEMPLATE 최신 1개
    q_pending_latest = text(
        """
        WITH pending_runs AS (
          SELECT s.run_id
          FROM (SELECT DISTINCT run_id FROM handoffs WHERE stage='SELECTED_TEMPLATE') s
          LEFT JOIN (SELECT DISTINCT run_id FROM handoffs WHERE stage='APPROVAL') a
            ON a.run_id = s.run_id
          WHERE a.run_id IS NULL
        ),
        last_selected AS (
          SELECT h.run_id, MAX(h.created_at) AS max_at
          FROM handoffs h
          JOIN pending_runs p ON p.run_id = h.run_id
          WHERE h.stage='SELECTED_TEMPLATE'
          GROUP BY h.run_id
        )
        SELECT h.run_id, h.created_at, h.payload_json
        FROM handoffs h
        JOIN last_selected ls
          ON ls.run_id = h.run_id
         AND ls.max_at = h.created_at
        WHERE h.stage='SELECTED_TEMPLATE'
        ORDER BY h.created_at DESC
        LIMIT 1
        """
    )
    r_latest = db.execute(q_pending_latest).mappings().first()

    pending_latest = None
    if r_latest:
        pending_latest = {
            "run_id": r_latest.get("run_id"),
            "created_at": make_json_safe(r_latest.get("created_at")),
            "template": _json_to_dict(r_latest.get("payload_json")),  # selected template
        }

    pending_all = []
    if show_all_pending:
        q_pending_all = text(
            """
            WITH pending_runs AS (
              SELECT s.run_id
              FROM (SELECT DISTINCT run_id FROM handoffs WHERE stage='SELECTED_TEMPLATE') s
              LEFT JOIN (SELECT DISTINCT run_id FROM handoffs WHERE stage='APPROVAL') a
                ON a.run_id = s.run_id
              WHERE a.run_id IS NULL
            ),
            last_selected AS (
              SELECT h.run_id, MAX(h.created_at) AS max_at
              FROM handoffs h
              JOIN pending_runs p ON p.run_id = h.run_id
              WHERE h.stage='SELECTED_TEMPLATE'
              GROUP BY h.run_id
            )
            SELECT h.run_id, h.created_at, h.payload_json
            FROM handoffs h
            JOIN last_selected ls
              ON ls.run_id = h.run_id
             AND ls.max_at = h.created_at
            WHERE h.stage='SELECTED_TEMPLATE'
            ORDER BY h.created_at DESC
            LIMIT 50
            """
        )
        rows = db.execute(q_pending_all).mappings().all()
        for r in rows:
            pending_all.append(
                {
                    "run_id": r.get("run_id"),
                    "created_at": make_json_safe(r.get("created_at")),
                    "template": _json_to_dict(r.get("payload_json")),
                }
            )

    # registry counts: latest APPROVAL per run 기준
    q_latest_approval = text(
        """
        WITH last_appr AS (
          SELECT run_id, MAX(created_at) AS max_at
          FROM handoffs
          WHERE stage='APPROVAL'
          GROUP BY run_id
        )
        SELECT h.run_id, h.payload_json
        FROM handoffs h
        JOIN last_appr la
          ON la.run_id = h.run_id
         AND la.max_at = h.created_at
        WHERE h.stage='APPROVAL'
        """
    )
    rows = db.execute(q_latest_approval).mappings().all()
    approved_cnt = 0
    rejected_cnt = 0
    approved_run_ids: list[str] = []

    for r in rows:
        p = _json_to_dict(r.get("payload_json"))
        d = (p.get("decision") or "").strip().upper()
        if d == "APPROVED":
            approved_cnt += 1
            rid = (r.get("run_id") or "").strip()
            if rid:
                approved_run_ids.append(rid)
        elif d == "REJECTED":
            rejected_cnt += 1

    # 승인 템플릿 최신 1개 + (전체보기) 승인 템플릿 리스트
    approved_latest = None
    approved_all = []

    # 최신 1개는 approved_run_ids 기준으로 SELECTED_TEMPLATE 최신을 1개만 뽑아서 사용
    if approved_run_ids:
        latest_rows = _fetch_latest_selected_for_runs(db, approved_run_ids, limit_n=1)
        if latest_rows:
            approved_latest = latest_rows[0]

    if show_all_approved and approved_run_ids:
        approved_all = _fetch_latest_selected_for_runs(db, approved_run_ids, limit_n=50)

    return {
        "kpi": {
            "ctr": kpi_ctr,
            "ctr_trend": kpi_ctr_trend,
            "guard_blocked": kpi_guard_blocked,
            "pending_candidate": pending_cnt,      # 승인/반려 대기함
            "active_assets": approved_cnt,         # 승인 템플릿 자산
        },
        "registry": {
            "approved": approved_cnt,              # 승인
            "deprecated": rejected_cnt,            # 반려
        },
        "pending": {"latest": pending_latest, "all": pending_all},
        "approved": {"latest": approved_latest, "all": approved_all},
        "ui": {
            "show_all_pending": bool(show_all_pending),
            "show_all_approved": bool(show_all_approved),
        },
    }


def _convert_target_payload_to_resolved(payload: dict) -> tuple[dict, dict]:
    age_sel = str(payload.get("age") or "all").strip().lower()
    gender_sel = str(payload.get("gender") or "ALL").strip().upper()
    skin_sel = str(payload.get("skin_type") or "ALL").strip().lower()

    age_map = {
        "10": ["10대"],
        "20": ["20대"],
        "30": ["30대"],
        "40": ["40대"],
        "50": ["50대+"],
        "2030": ["20대", "30대"],
        "4050": ["40대", "50대+"],
        "all": [],
    }
    age_bands = age_map.get(age_sel, [])

    genders_db = [gender_sel] if gender_sel in ("F", "M") else []

    skin_types_db = []
    if skin_sel in ("dry", "oily"):
        skin_types_db = [skin_sel]
    elif skin_sel in ("complex", "combination"):
        skin_types_db = ["combination"]
    elif skin_sel in ("normal",):
        skin_types_db = ["normal"]
    else:
        skin_types_db = []

    ck = payload.get("concern_keywords")
    if ck is None:
        concern_keywords = []
    elif isinstance(ck, str):
        s = ck.strip()
        concern_keywords = [s] if s else []
    elif isinstance(ck, list):
        concern_keywords = [str(x).strip() for x in ck if str(x).strip()]
    else:
        concern_keywords = []

    resolved = resolve_concerns_from_keywords(concern_keywords)

    target_input = {
        "gender": genders_db,
        "age_bands": age_bands,
        "skin_types": skin_types_db,
        "concern_keywords": concern_keywords,
    }
    target_resolved = {**target_input, **resolved}
    return target_input, target_resolved


def fetch_step3_data(db, repo: Repo, run_id: str) -> dict:
    out = {
        "ok": False,
        "run_id": run_id,
        "selected_template": None,
        "selected_candidate_no": None,
        "approvals": [],
        "errors": [],
    }
    if not run_id:
        out["errors"].append("run_id_missing")
        return out

    h_sel = repo.get_latest_handoff(run_id, "SELECTED_TEMPLATE")
    if not h_sel:
        out["errors"].append("no_SELECTED_TEMPLATE")
        return out

    sel = _json_to_dict(h_sel.get("payload_json"))
    out["selected_template"] = sel

    try:
        tid = (sel.get("template_id") or "").strip()
        h_cands = repo.get_latest_handoff(run_id, "TEMPLATE_CANDIDATES")
        if tid and h_cands:
            cands_payload = _json_to_dict(h_cands.get("payload_json"))
            cands = cands_payload.get("candidates") or []
            for i, c in enumerate(cands):
                if (c.get("template_id") or "") == tid:
                    out["selected_candidate_no"] = i + 1
                    break
    except Exception:
        pass

    approvals = []
    try:
        q = text("""
            SELECT created_at, payload_json
            FROM handoffs
            WHERE run_id = :rid AND stage = 'APPROVAL'
            ORDER BY created_at DESC
            LIMIT 20
        """)
        rows = db.execute(q, {"rid": run_id}).mappings().all()
        for r in rows:
            p = _json_to_dict(r.get("payload_json"))
            approvals.append({
                "created_at": make_json_safe(r.get("created_at")),
                "decision": (p.get("decision") or "").upper(),
                "comment": p.get("comment") or "",
                "marketer_id": p.get("marketer_id") or "marketer_001",
            })
    except Exception:
        pass

    out["approvals"] = approvals
    out["ok"] = True
    return out


# -------------------------
# UI -> Streamlit 이벤트 처리
# -------------------------
def handle_component_event(evt: dict, db, repo: Repo) -> None:
    if not evt or not isinstance(evt, dict):
        return

    action = evt.get("action")
    event_id = evt.get("event_id")

    if event_id:
        last = st.session_state.get("_last_event_id")
        if last == event_id:
            return
        st.session_state["_last_event_id"] = event_id

    if action == "HOME_TOGGLE_VIEW_ALL_PENDING":
        st.session_state["show_all_pending"] = not bool(st.session_state.get("show_all_pending", False))
        st.rerun()

    if action == "HOME_TOGGLE_VIEW_ALL_APPROVED":
        st.session_state["show_all_approved"] = not bool(st.session_state.get("show_all_approved", False))
        st.rerun()

    if action == "NAVIGATE_STEP1":
        st.session_state["requested_page"] = "Step1(타겟 설정) → 후보 생성"
        st.rerun()

    # Home에서도 승인/반려 저장
    if action == "HOME_SAVE_APPROVAL":
        payload = evt.get("payload") or {}
        run_id = (payload.get("run_id") or "").strip()
        decision = (payload.get("decision") or "").strip().upper()
        marketer_id = (payload.get("marketer_id") or "marketer_001").strip()
        comment = (payload.get("comment") or "").strip()

        if run_id and decision in ("APPROVED", "REJECTED"):
            repo.create_handoff(
                run_id,
                "APPROVAL",
                {"decision": decision, "comment": comment, "marketer_id": marketer_id},
            )
            try:
                db.commit()
            except Exception:
                pass
            try:
                repo.update_run(run_id, step_id="S4_APPROVAL", status=decision)
                db.commit()
            except Exception:
                pass

        st.session_state["requested_page"] = "Home(UI)"
        st.rerun()

    if action == "STEP1_PREVIEW_TARGET":
        payload = evt.get("payload") or {}
        _, target_resolved = _convert_target_payload_to_resolved(payload)
        cnt = preview_target_count(db, target_resolved)
        st.session_state["step1_result"] = make_json_safe({"preview_ok": True, "target_count": cnt})
        st.rerun()

    if action == "STEP1_CANCEL":
        for k in ["step1_result", "run_id", "_last_event_id", "step2_selected_template_id", "step3_result"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["requested_page"] = "Home(UI)"
        st.rerun()

    if action == "STEP1_SUBMIT":
        payload = evt.get("payload") or {}

        created_by = (payload.get("created_by") or "marketer_001").strip()
        goal_ui = (payload.get("goal") or "cart").strip()
        channel = (payload.get("channel") or "PUSH").strip().upper()
        tone = (payload.get("tone") or "amoremall").strip().lower()
        campaign_text = (payload.get("campaign_text") or "").strip()

        goal_code_map = {
            "cart": "cart_abandon",
            "repurchase": "reorder_top",
            "counseling": "feature_reco",
            "promotion": "feature_reco",
        }
        campaign_goal_code = goal_code_map.get(goal_ui, "feature_reco")

        brief = {
            "goal": goal_ui,
            "campaign_goal": campaign_goal_code,
            "campaign_text": campaign_text,
            "channel_hint": channel,
            "tone_hint": tone,
        }

        target_input, target_resolved = _convert_target_payload_to_resolved(payload)

        rid = repo.create_run(created_by, brief, channel=channel)
        repo.create_handoff(rid, "BRIEF", brief)
        repo.update_run(rid, step_id="S1_BRIEF")

        repo.create_handoff(rid, "TARGET_INPUT", target_input)

        cnt = preview_target_count(db, target_resolved)
        users_pack = fetch_target_user_ids(db, target_resolved, limit_n=500)

        repo.create_handoff(
            rid,
            "TARGET_AUDIENCE",
            {
                "count": int(users_pack["total_count"]),
                "user_ids": users_pack["user_ids"],
                "sample": [],
                "resolved": {
                    "concern_keywords": target_resolved.get("concern_keywords", []),
                    "concern_categories": target_resolved.get("concern_categories", []),
                    "skin_concerns": target_resolved.get("skin_concerns", []),
                },
            },
        )
        repo.update_run(rid, step_id="S2_READY")

        run_until_candidates(rid, channel=channel, tone=tone)

        st.session_state["run_id"] = rid
        st.session_state["step1_result"] = make_json_safe(
            {
                "ok": True,
                "run_id": rid,
                "target_count": cnt,
                "channel": channel,
                "tone": tone,
                "brief": brief,
                "target_input": target_input,
            }
        )
        st.session_state["requested_page"] = "Step2(후보 선택 확정)"
        st.rerun()

    if action == "STEP2_CONFIRM":
        payload = evt.get("payload") or {}
        run_id = (payload.get("run_id") or st.session_state.get("run_id") or "").strip()
        tid = (payload.get("template_id") or st.session_state.get("step2_selected_template_id") or "").strip()

        if not run_id or not tid:
            st.session_state["step2_error"] = {"ok": False, "msg": "run_id 또는 template_id가 없습니다."}
            st.rerun()

        h_cands = repo.get_latest_handoff(run_id, "TEMPLATE_CANDIDATES")
        if not h_cands:
            st.session_state["step2_error"] = {"ok": False, "msg": "TEMPLATE_CANDIDATES가 없습니다."}
            st.rerun()

        candidates = _json_to_dict(h_cands.get("payload_json")).get("candidates") or []
        picked = None
        for c in candidates:
            if (c.get("template_id") or "") == tid:
                picked = c
                break

        if not picked:
            st.session_state["step2_error"] = {"ok": False, "msg": f"후보에서 template_id={tid} 를 찾지 못했습니다."}
            st.rerun()

        repo.create_handoff(run_id, "SELECTED_TEMPLATE", picked)
        try:
            repo.update_run(run_id, step_id="S3_SELECTED", candidate_id=(tid or "")[:16], status="SELECTED")
        except Exception:
            pass

        st.session_state["run_id"] = run_id
        st.session_state["requested_page"] = "Step3(승인/반려 저장)"
        st.rerun()

    if action == "STEP2_REGENERATE":
        payload = evt.get("payload") or {}
        run_id = (payload.get("run_id") or st.session_state.get("run_id") or "").strip()
        if not run_id:
            st.rerun()

        run = repo.get_run(run_id) or {}
        brief = _json_to_dict(run.get("brief_json"))
        channel = (run.get("channel") or brief.get("channel_hint") or "PUSH").strip().upper()
        tone = (brief.get("tone_hint") or "amoremall").strip().lower()

        try:
            run_until_candidates(run_id, channel=channel, tone=tone)
            st.session_state["step2_error"] = {"ok": True, "msg": "재생성 완료"}
        except Exception as e:
            st.session_state["step2_error"] = {"ok": False, "msg": f"재생성 실패: {e}"}
        st.rerun()

    if action == "STEP3_SAVE_APPROVAL":
        payload = evt.get("payload") or {}
        run_id = (payload.get("run_id") or st.session_state.get("run_id") or "").strip()
        decision = (payload.get("decision") or "").strip().upper()
        comment = (payload.get("comment") or "").strip()
        marketer_id = (payload.get("marketer_id") or "marketer_001").strip()

        toast_id = int(time.time() * 1000)

        if not run_id:
            st.session_state["step3_result"] = {"ok": False, "msg": "run_id가 없습니다.", "toast_id": toast_id}
            st.session_state["requested_page"] = "Step3(승인/반려 저장)"
            st.rerun()

        if decision not in ("APPROVED", "REJECTED"):
            st.session_state["step3_result"] = {"ok": False, "msg": "decision은 APPROVED/REJECTED 여야 합니다.", "toast_id": toast_id}
            st.session_state["requested_page"] = "Step3(승인/반려 저장)"
            st.rerun()

        repo.create_handoff(
            run_id,
            "APPROVAL",
            {
                "decision": decision,
                "comment": comment,
                "marketer_id": marketer_id,
            },
        )

        try:
            db.commit()
        except Exception:
            pass

        try:
            repo.update_run(run_id, step_id="S4_APPROVAL", status=decision)
            db.commit()
        except Exception:
            pass

        st.session_state["step3_result"] = {"ok": True, "msg": "저장 완료", "toast_id": toast_id}
        st.session_state["requested_page"] = "Step3(승인/반려 저장)"
        st.rerun()


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="CRM Agent Ops", layout="wide")

if "show_all_pending" not in st.session_state:
    st.session_state["show_all_pending"] = False

if "show_all_approved" not in st.session_state:
    st.session_state["show_all_approved"] = False

if "requested_page" in st.session_state:
    st.session_state["page_selector"] = st.session_state.pop("requested_page")

PAGES = [
    "Home(UI)",
    "Step1(타겟 설정) → 후보 생성",
    "Step2(후보 선택 확정)",
    "Step3(승인/반려 저장)",
    "Run 타임라인",
]
page = st.sidebar.radio("페이지", PAGES, key="page_selector")

st.sidebar.markdown("---")
run_id_in = st.sidebar.text_input("run_id(선택)", value=st.session_state.get("run_id", "")).strip()
if run_id_in:
    st.session_state["run_id"] = run_id_in

db = SessionLocal()
repo = Repo(db)

page_to_ui = {
    "Home(UI)": "index",
    "Step1(타겟 설정) → 후보 생성": "first",
    "Step2(후보 선택 확정)": "second",
    "Step3(승인/반려 저장)": "third",
    "Run 타임라인": "timeline",
}
ui_page = page_to_ui.get(page, "index")

result = {}

if ui_page == "index":
    result = fetch_home_data(
        db,
        show_all_pending=bool(st.session_state["show_all_pending"]),
        show_all_approved=bool(st.session_state["show_all_approved"]),
    )

elif ui_page == "first":
    result = st.session_state.get("step1_result") or {}

elif ui_page == "second":
    rid = (st.session_state.get("run_id") or "").strip()
    result = {"run_id": rid, "candidates": []}
    if rid:
        h_cands = repo.get_latest_handoff(rid, "TEMPLATE_CANDIDATES")
        if h_cands:
            payload = _json_to_dict(h_cands.get("payload_json"))
            result["candidates"] = payload.get("candidates", []) or []

elif ui_page == "third":
    rid = (st.session_state.get("run_id") or "").strip()
    result = fetch_step3_data(db, repo, rid)

    if "step3_result" in st.session_state:
        result["save_result"] = st.session_state.get("step3_result")
    else:
        result["save_result"] = None

else:
    result = {}

evt = crm_ui(page=ui_page, ui_state={}, result=make_json_safe(result), height=900, key="crm_ui")
handle_component_event(evt, db, repo)
