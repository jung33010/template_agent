from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import re
from difflib import SequenceMatcher

from crm_agent.services.tone_guide import load_tone_guide
from crm_agent.agents.brief_normalizer import normalize_campaign_text


REQUIRED_SLOTS_BY_CHANNEL = {
    "PUSH": ["customer_name", "product_name", "offer", "cta"],
    "SMS": ["customer_name", "product_name", "offer", "cta", "unsubscribe"],
    "KAKAO": ["customer_name", "product_name", "offer", "cta"],
    "EMAIL": ["customer_name", "product_name", "offer", "cta", "subject"],
}

OPTIONAL_SLOTS = [
    "coupon_code",
    "expiry_date",
    "deep_link",
    "brand_name",
    "support_contact",
]

DEFAULT_NUM_CANDIDATES = 5


# -----------------------------
# helpers
# -----------------------------
def _normalize_channel(channel: str) -> str:
    c = (channel or "").strip().upper()
    if c in ("PUSH", "SMS", "KAKAO", "EMAIL"):
        return c
    return "PUSH"


def _slot_placeholders_in_text(text: str) -> set[str]:
    return set(re.findall(r"\{([a-zA-Z0-9_]+)\}", text or ""))


def _ensure_required_slots_in_text(text: str, required_slots: List[str]) -> Tuple[str, List[str]]:
    """
    본문에 필수 슬롯이 빠졌으면 마지막에 강제로 붙임.
    returns: (fixed_text, missing_list)
    """
    present = _slot_placeholders_in_text(text or "")
    missing = [s for s in required_slots if s not in present]
    if not missing:
        return (text or "").strip(), []
    fixed = ((text or "").strip() + "\n" + "\n".join([f"{{{m}}}" for m in missing])).strip()
    return fixed, missing


def _format_normalized_campaign_text(normalized: Dict[str, Any], raw_campaign_text: str) -> str:
    keywords = normalized.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []

    normalized_text = (normalized.get("normalized_text") or "").strip()
    category = (normalized.get("category") or "").strip()
    occasion = (normalized.get("occasion") or "").strip()

    finish = normalized.get("finish_or_texture") or []
    style = normalized.get("mood_or_style") or []
    negative = normalized.get("negative") or []

    parts = []
    if normalized_text:
        parts.append(f"- 요약: {normalized_text}")
    if keywords:
        parts.append(f"- 키워드: {', '.join([str(k) for k in keywords[:12]])}")
    if category:
        parts.append(f"- 카테고리(추정): {category}")
    if occasion:
        parts.append(f"- 상황/목적(추정): {occasion}")
    if finish:
        parts.append(f"- 제형/피니시: {', '.join([str(x) for x in finish[:8]])}")
    if style:
        parts.append(f"- 무드/스타일: {', '.join([str(x) for x in style[:8]])}")
    if negative:
        parts.append(f"- 제외조건: {', '.join([str(x) for x in negative[:8]])}")

    parts.append(f"- 원문: {raw_campaign_text}")
    return "\n".join(parts).strip()


def _format_target_context(target: Optional[Dict[str, Any]]) -> str:
    target = target or {}
    base_target_query = target.get("target_query", {}) or {}
    base_target_summary = (target.get("summary", "") or "").strip()
    target_input_summary = (target.get("target_input_summary", "") or "").strip()

    audience = target.get("audience", {}) or {}
    audience_count = audience.get("count", 0)
    resolved = audience.get("resolved", {}) or {}

    lines = [
        f"- base_target_summary: {base_target_summary}" if base_target_summary else "- base_target_summary: (없음)",
        f"- base_target_query: {base_target_query}" if base_target_query else "- base_target_query: (없음)",
        f"- selected_filters: {target_input_summary}" if target_input_summary else "- selected_filters: (없음)",
        f"- audience_count: {audience_count}",
        f"- concern_mapping: {resolved}" if resolved else "- concern_mapping: (없음)",
    ]
    return "\n".join(lines).strip()


def _call_openai(prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    text = getattr(resp, "output_text", None)
    if not text:
        try:
            text = json.dumps(resp.model_dump(), ensure_ascii=False)
        except Exception:
            text = str(resp)

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"LLM did not return JSON. RAW:\n{text[:1500]}")
    return json.loads(m.group(0))


# -----------------------------
# title/headline handling
# -----------------------------
def _is_angle_title(title: str) -> bool:
    t = (title or "").strip()
    return bool(re.match(r"^A[1-5][\-_ ]", t, flags=re.IGNORECASE))


def _clean_title(title: str) -> str:
    """
    제목에서 A1_... 같은 내부 태그가 들어오면 제거.
    """
    t = (title or "").strip()
    t = re.sub(r"^A[1-5][\-_ ]+", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _pick_keywords(normalized: Dict[str, Any]) -> List[str]:
    kws = normalized.get("keywords") or []
    if not isinstance(kws, list):
        return []
    # 너무 긴 단어/중복 제거
    out = []
    seen = set()
    for k in kws:
        s = str(k).strip()
        if not s or len(s) > 12:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[:6]


def _make_headline(angle: str, normalized: Dict[str, Any], campaign_goal: str) -> str:
    """
    title은 실제 SMS/푸시에서 보여도 어색하지 않은 헤드라인(짧고 자연스럽게)
    - angle은 내부적으로만 사용(외부 노출 X)
    """
    angle = (angle or "").upper().strip()
    kws = _pick_keywords(normalized)
    # 대표 키워드로 "겨울/보습/건조" 같은 걸 얻을 수 있으면 활용
    kw1 = kws[0] if len(kws) > 0 else ""
    kw2 = kws[1] if len(kws) > 1 else ""

    # 캠페인 골이 길면 쓰지 않음(헤드라인은 짧아야)
    goal_hint = (campaign_goal or "").strip()
    if len(goal_hint) > 12:
        goal_hint = ""

    # angle별 헤드라인 템플릿(과장/단정 금지, 슬롯 없이도 자연스러움)
    if angle == "A1":
        base = "오늘의 보습 루틴"
    elif angle == "A2":
        base = "건조함 케어 포인트"
    elif angle == "A3":
        base = "겨울 루틴 추천"
    elif angle == "A4":
        base = "재구매 리마인드"
    else:  # A5
        base = "보습 안내"

    # 키워드가 있으면 살짝 섞기(너무 길면 컷)
    if kw1 and kw1 not in base and len(base) <= 10:
        base = f"{kw1} {base}".strip()
    if kw2 and len(base) <= 10 and kw2 not in base:
        base = f"{base} · {kw2}".strip()

    # goal 힌트가 있으면 뒤에 짧게 덧붙이기
    if goal_hint and len(base) <= 10:
        base = f"{base} {goal_hint}".strip()

    # 최종 길이 제한(대략 SMS 제목 느낌)
    if len(base) > 18:
        base = base[:18].rstrip()

    return base


# -----------------------------
# prompt (NO variants)
# -----------------------------
def _build_prompt(
        *,
        channel: str,
        tone_id: str,
        tone_guide_md: str,
        campaign_goal: str,
        campaign_text_normalized: str,
        rag_context: str,
        target_context: str,
        required_slots: List[str],
        k: int,
) -> str:
    """
    title은 "내부 태그(A1..)"가 아니라 "실제 헤드라인"으로 생성하도록 강제
    angle은 JSON에 따로 넣지 말고, 본문 구조만 다르게 하라고 지시
    """
    channel_guide = {
        "SMS": "SMS는 짧고 명확하게(가능하면 90자 내외), 수신거부 슬롯({unsubscribe})을 포함.",
        "PUSH": "PUSH는 1~2문장 + CTA 중심으로 간결하게.",
        "KAKAO": "KAKAO는 친근/가독성(줄바꿈) + CTA 명확.",
        "EMAIL": "EMAIL은 body는 짧게, subject는 슬롯 템플릿 형태로 제공.",
    }.get(channel, "")

    tone_guide_block = tone_guide_md.strip() if tone_guide_md else "(없음: 기본 톤 가이드를 따르세요.)"

    diversity_rules = """
[다양성 규칙(매우 중요)]
- candidates 5개는 서로 '구조/길이/줄바꿈/CTA 위치'가 확실히 달라야 한다.
- 다섯 후보는 아래 5가지 각도를 각각 하나씩 사용하되, 각도 라벨(A1/A2...)은 title에 절대 쓰지 마라.
  1) 초간단(1~2줄) 
  2) 문제-해결(고민→제안→CTA)
  3) 루틴제안(1/2 step 형태)
  4) 리마인드(다시/놓치지 않게 등 완곡)
  5) 안심/문의(문의/확인 유도)
- title은 실제 발송에서 보이는 "짧은 헤드라인"으로 8~18자 내외.
- title에 슬롯({product_name})을 넣지 말고, 내부 태그(A1_...)도 넣지 마라.
- 같은 단어/문장/패턴 반복 금지.
""".strip()

    return f"""
너는 화장품/뷰티 CRM 마케터를 돕는 "Template Agent"다.
중요 원칙:
- 절대 상품/혜택/가격/쿠폰을 확정하지 마라. 모든 변수는 반드시 슬롯(예: {{product_name}}, {{offer}})으로 남겨라.
- 사실 단정/의학적 효능 단정/과장 표현 금지.
- 출력은 반드시 JSON만. 다른 문장/설명 금지.

[입력]
- channel: {channel}
- tone_id(brand): {tone_id}
- campaign_goal: {campaign_goal}
- campaign_text (normalized):
{campaign_text_normalized}

[사용자 선택/타겟 컨텍스트]
{target_context}

[브랜드 톤 가이드(md)]
{tone_guide_block}

[RAG 컨텍스트(근거)]
{rag_context}

[슬롯 규칙]
- 필수 슬롯(required): {required_slots}
- 옵션 슬롯(optional): {OPTIONAL_SLOTS}
- body_with_slots에는 필수 슬롯이 모두 등장해야 한다.
- 슬롯 표기는 반드시 {{slot_name}}

[채널 가이드]
- {channel_guide}

{diversity_rules}

[출력 JSON 스키마]
{{
  "candidates": [
    {{
      "title": "짧은 헤드라인(8~18자)",
      "body_with_slots": "슬롯 포함 본문",
      "default_slot_values": {{
        "cta": "{{deep_link}}",
        "subject": "{{campaign_goal}} 안내 | {{product_name}} {{offer}}"
      }}
    }}
  ]
}}

요청: candidates를 정확히 {k}개 생성하라.
반드시 JSON만 출력.
""".strip()


# -----------------------------
# diversity postprocess
# -----------------------------
def _similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _diversify_body_by_angle(*, angle: str, channel: str) -> str:
    """
    LLM 결과가 너무 비슷할 때 후처리로 구조를 강제 분리.
    - 의미를 확정하지 않고 구조만 다르게.
    - 슬롯 유지.
    """
    angle = (angle or "").strip().upper()

    if angle == "A1":
        b = "\n".join([
            "{customer_name}님, {product_name}",
            "{offer}",
            "바로가기: {cta}",
        ])
    elif angle == "A2":
        b = "\n".join([
            "{customer_name}님, 요즘 루틴 고민 있으셨나요?",
            "{product_name}로 가볍게 점검해보세요.",
            "{offer}",
            "확인하러 가기: {cta}",
        ])
    elif angle == "A3":
        b = "\n".join([
            "{customer_name}님 루틴에 이렇게 더해보세요.",
            "1) {product_name}",
            "2) {offer}",
            "루틴 보러가기: {cta}",
        ])
    elif angle == "A4":
        b = "\n".join([
            "{customer_name}님, 지난 루틴 이어가실 타이밍이에요.",
            "{product_name} 다시 확인해보세요.",
            "{offer}",
            "놓치기 전 확인: {cta}",
        ])
    else:  # A5
        b = "\n".join([
            "{customer_name}님께 안내드려요.",
            "{product_name} 관련 정보예요.",
            "{offer}",
            "궁금한 점은 {support_contact}로 확인해 주세요.",
            "자세히 보기: {cta}",
        ])

    if channel == "SMS" and "{unsubscribe}" not in b:
        b = (b + "\n수신거부: {unsubscribe}").strip()

    return b.strip()


def _postprocess_diversity(
        *,
        candidates: List[Dict[str, Any]],
        channel: str,
        required: List[str],
        normalized: Dict[str, Any],
        campaign_goal: str,
        similarity_threshold: float = 0.86,
) -> List[Dict[str, Any]]:
    """
    title은 헤드라인으로 유지/보정
    다양성 angle은 "index 기반"으로 내부에서만 적용 (title에 의존 X)
    """
    angles = ["A1", "A2", "A3", "A4", "A5"]

    fixed: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        angle = angles[i % len(angles)]

        # 1) title 보정: A1_... 같은 게 들어오면 제거하고 헤드라인 재생성
        title = (c.get("title") or "").strip()
        if not title or _is_angle_title(title):
            title = _clean_title(title)
            # 제거했는데도 비었거나 너무 애매하면 자동 생성
            if not title or len(title) < 4:
                title = _make_headline(angle=angle, normalized=normalized, campaign_goal=campaign_goal)
        # 너무 길면 컷
        if len(title) > 18:
            title = title[:18].rstrip()
        c["title"] = title

        # 2) body 다양성 보정
        body = (c.get("body_with_slots") or "").strip()
        too_similar = any(_similarity(body, prev.get("body_with_slots", "")) >= similarity_threshold for prev in fixed)
        if too_similar:
            body = _diversify_body_by_angle(angle=angle, channel=channel)

        body_fixed, missing = _ensure_required_slots_in_text(body, required)
        c["body_with_slots"] = body_fixed
        c.setdefault("notes", {})
        c["notes"]["missing_slots_fixed"] = missing
        # 내부 angle을 notes에만 남겨 디버깅/분석 가능
        c["notes"]["angle"] = angle

        # variants 키가 있으면 제거
        c.pop("variants", None)

        fixed.append(c)

    return fixed


# -----------------------------
# fallback (NO variants)
# -----------------------------
def _fallback_candidates(*, channel: str, tone_id: str, required: List[str], normalized: Dict[str, Any], campaign_goal: str) -> Dict[str, Any]:
    """
    폴백도 title은 헤드라인 형태로
    """
    channel = _normalize_channel(channel)
    angles = ["A1", "A2", "A3", "A4", "A5"]

    bodies = {
        "A1": "{customer_name}님, {product_name}\n{offer}\n바로가기: {cta}",
        "A2": "{customer_name}님, 요즘 루틴 고민 있으셨나요?\n{product_name}로 가볍게 점검해보세요.\n{offer}\n확인하러 가기: {cta}",
        "A3": "{customer_name}님 루틴 제안!\n1) {product_name}\n2) {offer}\n루틴 보러가기: {cta}",
        "A4": "{customer_name}님, 지난 루틴 이어가실 타이밍이에요.\n{product_name} 다시 확인해보세요.\n{offer}\n놓치기 전 확인: {cta}",
        "A5": "{customer_name}님께 안내드려요.\n{product_name} 관련 정보예요.\n{offer}\n문의: {support_contact}\n자세히 보기: {cta}",
    }

    cands: List[Dict[str, Any]] = []
    for i, angle in enumerate(angles, start=1):
        title = _make_headline(angle=angle, normalized=normalized, campaign_goal=campaign_goal)
        body = bodies[angle]

        if channel == "SMS" and "{unsubscribe}" not in body:
            body = (body + "\n수신거부: {unsubscribe}").strip()

        body_fixed, missing = _ensure_required_slots_in_text(body, required)

        dsv = {
            "cta": "{deep_link}",
            "subject": "{campaign_goal} 안내 | {product_name} {offer}" if channel == "EMAIL" else ""
        }

        cands.append(
            {
                "template_id": f"T{i:03d}",
                "title": title,
                "slot_schema": {"required": required, "optional": OPTIONAL_SLOTS},
                "body_with_slots": body_fixed,
                "channel": channel,
                "tone": tone_id,
                "notes": {"fallback": True, "missing_slots_fixed": missing, "angle": angle},
                "default_slot_values": dsv,
            }
        )

    return {"candidates": cands}


# -----------------------------
# main entry
# -----------------------------
def generate_template_candidates(
        *,
        brief: dict,
        channel: str,
        tone: str,
        rag_context: str,
        target: Optional[Dict[str, Any]] = None,
        k: int = DEFAULT_NUM_CANDIDATES,
) -> Dict[str, Any]:
    """
    후보는 5개(k=5)
    title은 실제 헤드라인(발송 제목)로 생성/보정
    내부 angle은 notes에만 보관(다양성 유지용)
    """
    channel = _normalize_channel(channel)
    required = REQUIRED_SLOTS_BY_CHANNEL[channel]

    tone_id = (tone or "amoremall").strip().lower()
    tone_guide_md = load_tone_guide(tone_id)

    raw_campaign_text = (brief or {}).get("campaign_text", "").strip()
    campaign_goal = (brief or {}).get("goal", "").strip() or (brief or {}).get("campaign_goal", "").strip()

    rag_context = (rag_context or "").strip()[:2500]

    normalized = normalize_campaign_text(raw_campaign_text)
    normalized_prompt_text = _format_normalized_campaign_text(normalized, raw_campaign_text)
    target_context_text = _format_target_context(target)

    notes_common = {
        "campaign_goal": campaign_goal,
        "brand_tone_id": tone_id,
        "principle": "Template agent must not decide product/offer. Keep as slots.",
        "campaign_text_normalized": normalized,
        "target_context": target_context_text,
    }

    try:
        max_k = max(1, min(int(k), DEFAULT_NUM_CANDIDATES))

        prompt = _build_prompt(
            channel=channel,
            tone_id=tone_id,
            tone_guide_md=tone_guide_md,
            campaign_goal=campaign_goal,
            campaign_text_normalized=normalized_prompt_text,
            rag_context=rag_context,
            target_context=target_context_text,
            required_slots=required,
            k=max_k,
        )

        out = _call_openai(prompt)

        raw_cands = (out or {}).get("candidates", []) or []
        if not isinstance(raw_cands, list) or len(raw_cands) < 1:
            fb = _fallback_candidates(
                channel=channel,
                tone_id=tone_id,
                required=required,
                normalized=normalized,
                campaign_goal=campaign_goal,
            )
            for c in fb["candidates"]:
                c.setdefault("notes", {})
                c["notes"].update({**notes_common, "llm_error": "empty_candidates"})
            return fb

        final: List[Dict[str, Any]] = []
        for idx, rc in enumerate(raw_cands[:max_k], start=1):
            title = (rc.get("title") or "").strip()
            body = (rc.get("body_with_slots") or "").strip()

            body_fixed, missing = _ensure_required_slots_in_text(body, required)

            dsv = rc.get("default_slot_values") if isinstance(rc.get("default_slot_values"), dict) else {}
            dsv = dsv or {}
            dsv.setdefault("cta", "{deep_link}")
            if channel == "EMAIL":
                dsv.setdefault("subject", "{campaign_goal} 안내 | {product_name} {offer}")
            else:
                dsv.setdefault("subject", "")

            cand = {
                "template_id": f"T{idx:03d}",
                "title": title,
                "slot_schema": {"required": required, "optional": OPTIONAL_SLOTS},
                "body_with_slots": body_fixed,
                "channel": channel,
                "tone": tone_id,
                "notes": {**notes_common, "missing_slots_fixed": missing, "fallback": False},
                "default_slot_values": dsv,
            }

            # 혹시 LLM이 variants를 끼워 넣어도 제거
            cand.pop("variants", None)
            final.append(cand)

        # title 헤드라인 보정 + 다양성 후처리
        final = _postprocess_diversity(
            candidates=final,
            channel=channel,
            required=required,
            normalized=normalized,
            campaign_goal=campaign_goal,
            similarity_threshold=0.86,
        )

        return {"candidates": final[:max_k]}

    except Exception as e:
        fb = _fallback_candidates(
            channel=channel,
            tone_id=tone_id,
            required=required,
            normalized=normalized,
            campaign_goal=campaign_goal,
        )
        for c in fb["candidates"]:
            c.setdefault("notes", {})
            c["notes"].update({**notes_common, "llm_error": repr(e)})
        return fb
