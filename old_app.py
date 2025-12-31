# old_app.py
import os
import sys
from pathlib import Path
import json
import re
import streamlit as st
import streamlit.components.v1 as components

from crm_agent.product_agent.workflow import run_product_agent

from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import text, bindparam

# ROOT = Path(__file__).resolve().parent
# SRC = ROOT / "src"
# if str(SRC) not in sys.path:
#     sys.path.insert(0, str(SRC))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from crm_agent.db.engine import SessionLocal
from crm_agent.db.repo import Repo
from crm_agent.flow.workflow import run_until_candidates  # 후보 생성까지만 사용


# =========================
# v2 UI embed helpers
# =========================
ROOT = Path(__file__).resolve().parent

UI_ROOT = ROOT / "ui" / "v2"

def render_v2(page: str, height: int = 900):
    """
    v2.zip의 정적 HTML/CSS를 Streamlit에 임베드(데모용).
    - css 상대경로는 인라인로 주입
    - html 내부 페이지 이동 링크는 막고, 페이지 전환은 streamlit이 담당
    """
    html_path = UI_ROOT / f"{page}.html"
    css_path = UI_ROOT / "css" / f"{page}.css"

    if not html_path.exists():
        st.warning(f"UI 파일이 없습니다: {html_path}")
        return
    if not css_path.exists():
        st.warning(f"CSS 파일이 없습니다: {css_path}")
        return

    html_text = html_path.read_text(encoding="utf-8")
    css_text = css_path.read_text(encoding="utf-8")

    # link 태그 제거(상대경로 css는 streamlit iframe에서 안 먹는 경우가 많음)
    html_text = re.sub(r'<link[^>]+href="\./css/[^"]+"[^>]*>\s*', '', html_text)
    # a href 페이지 이동 막기(페이지 전환은 streamlit이 담당)
    html_text = re.sub(r'href="\./(index|first|second|third)\.html"', 'href="#"', html_text)

    # CSS 인라인 주입
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", f"<style>{css_text}</style></head>")
    else:
        html_text = f"<style>{css_text}</style>\n{html_text}"

    components.html(html_text, height=height, scrolling=True)


# =========================
# JSON pretty helper
# =========================
def j(obj):
    """
    Streamlit JSON 출력용 helper
    - datetime/date/Decimal 등 json.dumps가 못 직렬화하는 타입을 문자열로 변환
    """
    def _default(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        return str(o)

    st.code(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), language="json")


st.set_page_config(page_title="AMORE Template Agent", layout="wide")


# -------------------------
# Target mappings
# -------------------------
GENDER_LABEL_TO_DB = {"여": "F", "남": "M"}
SKIN_LABEL_TO_DB = {"건성": "dry", "지성": "oily", "복합성": "combination", "중성": "normal"}

# ✅ 너가 말한 "키워드 → 카테고리" 매핑 (UI/기획 용어)
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

# ✅ "카테고리 → DB enum(user_features.skin_concern_primary)" 매핑
# DB enum: sensitivity, acne, pigmentation, wrinkles, pores, redness, hydration, barrier, unknown
CATEGORY_TO_DB_CONCERNS = {
    "피부진정": ["sensitivity", "redness", "barrier", "acne"],
    "주름/탄력": ["wrinkles"],
    "미백/자외선차단": ["pigmentation"],
    "영양/보습": ["hydration", "barrier"],
    "블랙헤드/모공/피지": ["pores"],
}

CONCERN_KEYWORDS_UI = list(CONCERN_KEYWORD_TO_CATEGORY.keys())
CONCERN_CATEGORIES_UI = sorted(set(CONCERN_KEYWORD_TO_CATEGORY.values()))


# -------------------------
# Target helpers (DB preview)
# -------------------------
def _has_column(db, table: str, column: str) -> bool:
    row = db.execute(text(f"SHOW COLUMNS FROM {table} LIKE :col"), {"col": column}).fetchone()
    return row is not None


def _age_band_to_birthyear_ranges(age_bands: list[str]) -> list[tuple[int, int]]:
    """
    만 나이 근사:
    - 20대(20~29) => birth_year in [cur-29, cur-20]
    """
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
    """
    UI 키워드(민감성/트러블/...) -> 카테고리(피부진정/...) -> DB enum 리스트로 변환
    """
    keywords = [k for k in (keywords or []) if k in CONCERN_KEYWORD_TO_CATEGORY]
    categories = []
    for k in keywords:
        categories.append(CONCERN_KEYWORD_TO_CATEGORY[k])

    # unique preserve order
    seen = set()
    categories_uniq = []
    for c in categories:
        if c not in seen:
            seen.add(c)
            categories_uniq.append(c)

    db_vals = []
    for c in categories_uniq:
        db_vals.extend(CATEGORY_TO_DB_CONCERNS.get(c, []))

    # unique preserve order
    seen2 = set()
    db_vals_uniq = []
    for v in db_vals:
        if v not in seen2:
            seen2.add(v)
            db_vals_uniq.append(v)

    return {
        "concern_keywords": keywords,
        "concern_categories": categories_uniq,
        "skin_concerns": db_vals_uniq,  # DB enum values to filter
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

    return where_sql, params, bp, has_skin_type, has_concern


def preview_target_users_local(db, target_resolved: dict, sample_size: int = 5) -> dict:
    where_sql, params, bp, has_skin_type, has_concern = _build_where_and_params(db, target_resolved)

    q_count = text(f"""
        SELECT COUNT(*) AS cnt
        FROM users u
        LEFT JOIN user_features uf ON uf.user_id = u.user_id
        {where_sql}
    """).bindparams(*bp)

    cnt = db.execute(q_count, params).scalar() or 0

    cols = ["u.user_id", "u.gender", "u.birth_year"]
    if has_skin_type:
        cols.append("uf.skin_type")
    if has_concern:
        cols.append("uf.skin_concern_primary")

    q_sample = text(f"""
        SELECT {", ".join(cols)}
        FROM users u
        LEFT JOIN user_features uf ON uf.user_id = u.user_id
        {where_sql}
        ORDER BY u.user_id
        LIMIT :limit_n
    """).bindparams(*bp)

    params2 = dict(params)
    params2["limit_n"] = int(sample_size)
    rows = db.execute(q_sample, params2).mappings().all()

    cur = datetime.now().year
    sample = []
    for r in rows:
        by = r.get("birth_year")
        age = (cur - int(by)) if by else None
        item = {
            "user_id": r.get("user_id"),
            "gender": r.get("gender"),
            "birth_year": by,
            "age": age,
        }
        if has_skin_type:
            item["skin_type"] = r.get("skin_type")
        if has_concern:
            item["skin_concern_primary"] = r.get("skin_concern_primary")
        sample.append(item)

    return {
        "count": int(cnt),
        "sample": sample,
        "has_skin_type": has_skin_type,
        "has_skin_concern_primary": has_concern,
    }


def fetch_target_user_ids(db, target_resolved: dict, limit_n: int = 500) -> dict:
    """
    뒤쪽 agent로 넘길 "대상 user_id 리스트" 생성 (너무 커지지 않게 limit)
    """
    where_sql, params, bp, _, _ = _build_where_and_params(db, target_resolved)

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
    total = db.execute(q_count, params).scalar() or 0

    return {
        "total_count": int(total),
        "limit": int(limit_n),
        "user_ids": user_ids,
    }


def main():
    st.title("AMORE Template Agent")

    st.caption(
        "✅ 목표: 마케터는 캠페인 자연어 입력 + 톤/타겟 선택만 하고, "
        "Template Agent는 '슬롯 템플릿(뼈대)'만 생성합니다. "
        "상품/혜택/쿠폰/링크 채우기는 뒤 Execution Agent가 담당합니다."
    )

    page = st.sidebar.radio(
        "페이지",
        [
            "Home(UI Preview)",
            "Step1 브리프(캠페인 입력)",
            "Step2 후보 생성(톤/타겟)",
            "Step3 후보/선택(템플릿 확정)",
            "Step4 승인(템플릿 승인/반려)",
            "Step5 Product Agent(슬롯 채우기/발송 payload)",
            "Run 타임라인",
        ],
    )

    run_id = st.sidebar.text_input("run_id", value=st.session_state.get("run_id", ""))
    if run_id:
        st.session_state["run_id"] = run_id

    with SessionLocal() as db:
        repo = Repo(db)

        # -------------------------
        # Home: UI only preview
        # -------------------------
        if page == "Home(UI Preview)":
            st.subheader("UI 미리보기(프론트 v2)")
            colL, colR = st.columns([1.3, 1.0], gap="large")
            with colL:
                render_v2("index", height=950)
            with colR:
                st.info(
                    "이 화면은 **프론트팀 정적 UI(HTML/CSS)** 를 Streamlit에 임베드한 것입니다.\n\n"
                    "- 실제 입력/실행은 좌측 메뉴의 Step1~Step4에서 진행됩니다.\n"
                    "- 데모 단계에서는 UI를 '보여주기용'으로 사용합니다."
                )
                st.write("현재 run_id:", st.session_state.get("run_id", ""))

        # -------------------------
        # Step1
        # -------------------------
        elif page == "Step1 브리프(캠페인 입력)":
            colL, colR = st.columns([1.3, 1.0], gap="large")
            with colL:
                render_v2("first", height=950)

            with colR:
                st.subheader("Step1) 캠페인 브리프 입력 → run 생성")

                created_by = st.text_input("created_by(=user_id)", value="marketer_001")

                scenario_labels = [
                    "1) 화장품 특징 기반 추천 CRM(자연어 입력 필요)",
                    "2) 장바구니 미구매 상품 CRM",
                    "3) 재구매율 높은 상품 CRM",
                ]
                scenario_codes = {
                    scenario_labels[0]: "feature_reco",
                    scenario_labels[1]: "cart_abandon",
                    scenario_labels[2]: "reorder_top",
                }

                goal_label = st.radio("goal(캠페인 목표)", scenario_labels, index=0, horizontal=True)
                campaign_goal_code = scenario_codes[goal_label]

                col1, col2 = st.columns(2)
                with col1:
                    channel_hint = st.selectbox("채널 힌트(선택)", ["PUSH", "SMS", "KAKAO", "EMAIL"], index=0)
                with col2:
                    tone_hint = st.selectbox("브랜드 톤(선택)", ["amoremall", "innisfree"], index=0)

                disable_campaign_text = (goal_label != scenario_labels[0])
                campaign_text_value = "" if disable_campaign_text else (
                    "겨울철 보습 루틴을 강조하면서, 기존 구매 고객의 재구매를 유도하는 캠페인. 톤은 친근하게."
                )

                campaign_text = st.text_area(
                    "campaign_text(자연어 캠페인 설명)",
                    value=campaign_text_value,
                    height=120,
                    disabled=disable_campaign_text,
                    placeholder="(1번 시나리오에서만 입력 가능)",
                )

                brief = {
                    "goal": goal_label,
                    "campaign_goal": campaign_goal_code,
                    "campaign_text": campaign_text,
                    "channel_hint": channel_hint,
                    "tone_hint": tone_hint,
                }

                if st.button("run 생성"):
                    rid = repo.create_run(created_by, brief, channel=channel_hint)
                    repo.create_handoff(rid, "BRIEF", brief)
                    repo.update_run(rid, step_id="S1_BRIEF")

                    st.session_state["run_id"] = rid
                    st.success(f"run_id = {rid}")

                if st.session_state.get("run_id"):
                    run = repo.get_run(st.session_state["run_id"])
                    if run:
                        st.markdown("### 현재 Run")
                        st.write(
                            f"status: {run.get('status')} / step_id: {run.get('step_id')} / channel: {run.get('channel')}"
                        )
                        j(run.get("brief_json", {}))

        # -------------------------
        # Step2
        # -------------------------
        elif page == "Step2 후보 생성(톤/타겟)":
            colL, colR = st.columns([1.3, 1.0], gap="large")
            with colL:
                render_v2("second", height=950)

            with colR:
                st.subheader("Step2) 톤/타겟 선택 → LangGraph 실행(템플릿 후보 생성까지)")
                st.caption("✅ Step2에서 channel/tone 선택 UI는 제거했습니다. Step1 힌트를 사용합니다.")

                if not run_id:
                    st.warning("좌측 run_id 입력 또는 Step1에서 생성하세요.")
                    return

                run = repo.get_run(run_id)
                if not run:
                    st.error("run_id가 유효하지 않습니다.")
                    return

                brief = run.get("brief_json", {}) or {}
                channel = run.get("channel") or brief.get("channel_hint") or "PUSH"
                tone = (brief.get("tone_hint") or "amoremall").strip().lower()

                st.markdown("### 현재 채널/톤(=Step1 힌트)")
                st.write({"channel": channel, "tone": tone})

                st.markdown("### 타겟 선택(키워드 멀티선택)")
                st.caption("ℹ️ 아무 것도 선택하지 않으면 '전체(필터 없음)'로 동작합니다.")

                gender_labels = ["여", "남"]
                age_band_labels = ["10대", "20대", "30대", "40대", "50대+"]
                skin_labels = ["건성", "지성", "복합성", "중성"]

                sel_genders_label = st.multiselect("성별 (미선택=전체)", gender_labels, default=[])
                sel_age_bands = st.multiselect("나이대 (미선택=전체)", age_band_labels, default=[])
                sel_skin_label = st.multiselect("피부타입 (미선택=전체)", skin_labels, default=[])

                sel_concern_keywords = st.multiselect(
                    "피부고민(키워드) (미선택=전체)",
                    CONCERN_KEYWORDS_UI,
                    default=[],
                    help="키워드는 내부적으로 카테고리로 매핑된 뒤, DB skin_concern_primary(enum) 조건으로 변환됩니다.",
                )

                sel_genders_db = [GENDER_LABEL_TO_DB[x] for x in sel_genders_label if x in GENDER_LABEL_TO_DB]
                sel_skin_db = [SKIN_LABEL_TO_DB[x] for x in sel_skin_label if x in SKIN_LABEL_TO_DB]

                target_input = {
                    "gender": sel_genders_db,                 # F/M
                    "age_bands": sel_age_bands,               # 라벨 유지
                    "skin_types": sel_skin_db,                # dry/oily/...
                    "concern_keywords": sel_concern_keywords, # 민감성/트러블/...
                }

                resolved = resolve_concerns_from_keywords(sel_concern_keywords)
                target_resolved = {**target_input, **resolved}

                preview = preview_target_users_local(repo.db, target_resolved, sample_size=5)

                st.markdown("### 타겟 미리보기")
                st.write(f"대상 수: **{preview['count']}명**")

                if not preview.get("has_skin_type"):
                    st.info("user_features.skin_type 컬럼이 없어 피부타입 필터/표시는 현재 무시됩니다. (추가하면 자동 활성화)")
                if not preview.get("has_skin_concern_primary"):
                    st.info("user_features.skin_concern_primary 컬럼이 없어 피부고민 필터/표시는 현재 무시됩니다. (추가하면 자동 활성화)")

                st.markdown("#### TARGET_INPUT (UI 선택값)")
                j(target_input)

                st.markdown("#### TARGET_RESOLVED (키워드→카테고리→DB필터 변환 결과)")
                j({
                    "concern_keywords": target_resolved.get("concern_keywords", []),
                    "concern_categories": target_resolved.get("concern_categories", []),
                    "skin_concerns(DB enum)": target_resolved.get("skin_concerns", []),
                })

                st.markdown("#### 샘플 유저(최대 5)")
                j(preview)

                # ✅ 핵심 수정: workflow가 기대하는 TARGET_AUDIENCE로 저장
                if st.button("LangGraph 실행(후보 생성까지)"):
                    repo.create_handoff(run_id, "TARGET_INPUT", target_input)

                    users_pack = fetch_target_user_ids(repo.db, target_resolved, limit_n=500)
                    target_audience = {
                        "count": int(users_pack["total_count"]),   # 전체 타겟 수(리밋과 무관)
                        "user_ids": users_pack["user_ids"],        # downstream 용
                        "sample": preview.get("sample", []),
                        "resolved": {
                            "concern_keywords": target_resolved.get("concern_keywords", []),
                            "concern_categories": target_resolved.get("concern_categories", []),
                            "skin_concerns": target_resolved.get("skin_concerns", []),
                        },
                    }
                    repo.create_handoff(run_id, "TARGET_AUDIENCE", target_audience)

                    repo.update_run(run_id, step_id="S2_READY")
                    run_until_candidates(run_id, channel=channel, tone=tone)

                    st.success("완료: TARGET/RAG/후보/컴플라이언스 생성됨 (템플릿 후보 단계까지)")

                # ---- handoff view ----
                st.markdown("### TARGET_INPUT")
                h = repo.get_latest_handoff(run_id, "TARGET_INPUT")
                if h:
                    j(h["payload_json"])
                else:
                    st.info("TARGET_INPUT handoff가 아직 없습니다. (Step2 실행 필요)")

                st.markdown("### TARGET_AUDIENCE")
                h = repo.get_latest_handoff(run_id, "TARGET_AUDIENCE")
                if h:
                    j(h["payload_json"])
                else:
                    st.info("TARGET_AUDIENCE handoff가 아직 없습니다. (Step2 실행 필요)")

                st.markdown("### TARGET (workflow가 생성한 값)")
                h = repo.get_latest_handoff(run_id, "TARGET")
                if h:
                    j(h["payload_json"])
                else:
                    st.info("TARGET handoff가 아직 없습니다. (Step2 실행 필요)")

                st.markdown("### RAG")
                h = repo.get_latest_handoff(run_id, "RAG")
                if h:
                    j(h["payload_json"])
                else:
                    st.info("RAG handoff가 아직 없습니다.")

                st.markdown("### TEMPLATE_CANDIDATES (슬롯 템플릿 후보)")
                h = repo.get_latest_handoff(run_id, "TEMPLATE_CANDIDATES")
                if h:
                    j(h["payload_json"])
                else:
                    st.info("후보가 아직 없습니다. Step2 실행 필요")

                st.markdown("### COMPLIANCE (후보별 PASS/WARN/FAIL)")
                h = repo.get_latest_handoff(run_id, "COMPLIANCE")
                if h:
                    j(h["payload_json"])
                else:
                    st.info("컴플라이언스 결과가 아직 없습니다.")

        # -------------------------
        # Step3
        # -------------------------
        elif page == "Step3 후보/선택(템플릿 확정)":
            colL, colR = st.columns([1.3, 1.0], gap="large")
            with colL:
                render_v2("third", height=950)

            with colR:
                st.subheader("Step3) 후보 확인 → 템플릿 선택(확정)")
                if not run_id:
                    st.warning("좌측 run_id 입력 후 진행하세요.")
                    return

                h_cands = repo.get_latest_handoff(run_id, "TEMPLATE_CANDIDATES")
                h_comp = repo.get_latest_handoff(run_id, "COMPLIANCE")
                h_sel = repo.get_latest_handoff(run_id, "SELECTED_TEMPLATE")

                if not h_cands:
                    st.warning("TEMPLATE_CANDIDATES가 없습니다. Step2에서 'LangGraph 실행(후보 생성까지)'를 먼저 실행하세요.")
                    return

                cands_payload = h_cands["payload_json"] or {}
                candidates = cands_payload.get("candidates", []) or []

                if not candidates:
                    st.warning("후보 리스트가 비어 있습니다. Step2 실행 로그를 확인하세요.")
                    j(cands_payload)
                    return

                comp_map = {}
                if h_comp:
                    comp_payload = h_comp["payload_json"] or {}
                    for r in (comp_payload.get("results") or []):
                        comp_map[r.get("template_id")] = r

                selected_id = None
                if h_sel:
                    selected_id = (h_sel["payload_json"] or {}).get("template_id")

                labels = []
                for c in candidates:
                    tid = c.get("template_id")
                    status = (comp_map.get(tid) or {}).get("status", "N/A")
                    title = c.get("title") or ""
                    labels.append((tid, f"[{status}] {tid}  {title}"))

                # default selection
                default_idx = 0
                if selected_id:
                    for i, (tid, _) in enumerate(labels):
                        if tid == selected_id:
                            default_idx = i
                            break

                idx = st.radio(
                    "후보 선택",
                    list(range(len(labels))),
                    format_func=lambda i: labels[i][1],
                    index=default_idx,
                )

                picked = candidates[idx]
                tid = picked.get("template_id")

                st.markdown("### 선택 후보 미리보기")
                j({
                    "template_id": tid,
                    "title": picked.get("title"),
                    "body_with_slots": picked.get("body_with_slots"),
                    "compliance": comp_map.get(tid, {}),
                })

                colA, colB = st.columns(2)
                with colA:
                    if st.button("✅ 이 후보로 확정(SELECTED_TEMPLATE 저장)"):
                        repo.create_handoff(run_id, "SELECTED_TEMPLATE", picked)
                        try:
                            repo.update_run(run_id, step_id="S3_SELECTED", candidate_id=(tid or "")[:16], status="SELECTED")
                        except Exception:
                            pass
                        st.success("SELECTED_TEMPLATE 저장 완료. Step4에서 승인 진행하세요.")

                with colB:
                    if h_sel:
                        st.info("현재 DB에 저장된 SELECTED_TEMPLATE가 있습니다.")
                        j(h_sel["payload_json"])

        # -------------------------
        # Step4
        # -------------------------
        elif page == "Step4 승인(템플릿 승인/반려)":
            st.subheader("Step4) 마케터 승인/반려")
            if not run_id:
                st.warning("좌측 run_id 입력 후 진행하세요.")
                return

            h_sel = repo.get_latest_handoff(run_id, "SELECTED_TEMPLATE")
            if not h_sel:
                st.warning("SELECTED_TEMPLATE가 없습니다. Step3에서 템플릿을 먼저 확정하세요.")
                return

            st.markdown("### 선택된 템플릿(SELECTED_TEMPLATE)")
            j(h_sel["payload_json"])

            marketer_id = st.text_input("marketer_id", value="marketer_001")
            decision = st.radio("결정", ["APPROVED", "REJECTED"], horizontal=True)
            comment = st.text_area("코멘트(선택)", value="", height=100)

            if st.button("결정 저장(APPROVAL handoff)"):
                repo.add_approval(run_id, marketer_id=marketer_id, decision=decision, comment=comment)
                try:
                    repo.update_run(run_id, step_id="S4_APPROVAL", status=decision)
                except Exception:
                    pass
                st.success("승인/반려 저장 완료")

            st.markdown("### 승인 이력")
            approvals = repo.list_approvals(run_id)
            if approvals:
                for a in approvals:
                    st.write(f"- {a.get('created_at')} | {a['payload_json'].get('marketer_id')} | {a['payload_json'].get('decision')}")
                    if a["payload_json"].get("comment"):
                        st.caption(a["payload_json"]["comment"])
            else:
                st.info("승인 이력이 없습니다.")
        

        # -------------------------
# Step5
# -------------------------
        elif page == "Step5 Product Agent(슬롯 채우기/발송 payload)":
            st.subheader("Step5) Product Agent 실행 (유저별 상품 추천 + 슬롯 채움 + send_logs 저장)")

            if not run_id:
                st.warning("좌측 run_id 입력 후 진행하세요.")
                st.stop()

            # ✅ 옵션은 버튼보다 먼저 선언(버튼 클릭 시 값이 반영되게)
            top_k = st.number_input("상품 추천 Top-K", min_value=1, max_value=10, value=3, step=1)
            ignore_opt_in = st.checkbox("테스트 모드(Opt-in 무시하고 렌더링)", value=True)
            max_preview = st.number_input("미리보기 개수", min_value=1, max_value=30, value=10, step=1)

            if st.button("▶ Product Agent 실행"):
                # ✅ 옵션을 run_product_agent에 넘겨야 함
                out = run_product_agent(
                    run_id,
                    top_k_products=int(top_k),
                    ignore_opt_in=bool(ignore_opt_in),
                    max_preview=int(max_preview),
                )
                st.success("완료: campaign_send_logs 저장 + EXECUTION_RESULT 생성")

                st.markdown("### 요약(Product Agent summary)")
                st.json(out.get("summary", {}), expanded=True)

                # ✅ 2-3) DB에서 실제 rendered_text 가져와 출력
                st.markdown("### 유저별 완성 메시지 미리보기 (campaign_send_logs)")

                from sqlalchemy import text
                import pandas as pd

                rows = repo.db.execute(
                    text("""
                        SELECT user_id, status, rendered_text, error_code, error_message
                        FROM campaign_send_logs
                        WHERE run_id = :run_id
                        ORDER BY created_at DESC
                        LIMIT 50
                    """),
                    {"run_id": run_id},
                ).mappings().all()

                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("campaign_send_logs에 데이터가 없습니다.")

            st.markdown("### 최신 EXECUTION_RESULT")
            h = repo.get_latest_handoff(run_id, "EXECUTION_RESULT")
            if h:
                st.json(h["payload_json"], expanded=True)
            else:
                st.info("아직 EXECUTION_RESULT가 없습니다.")



        # -------------------------
        # Run Timeline
        # -------------------------
        elif page == "Run 타임라인":
            st.subheader("Run 타임라인(handoffs)")
            if not run_id:
                st.warning("좌측 run_id 입력 후 진행하세요.")
                return

            run = repo.get_run(run_id)
            st.markdown("### campaign_runs")
            j(run or {})

            st.markdown("### handoffs")
            rows = repo.list_handoffs(run_id)
            if not rows:
                st.info("handoff가 없습니다.")
                return

            import pandas as pd
            df = pd.DataFrame([{
                "created_at": r.get("created_at"),
                "stage": r.get("stage"),
                "payload_version": r.get("payload_version"),
                "handoff_id": r.get("handoff_id"),
            } for r in rows])
            st.dataframe(df, use_container_width=True)

            st.markdown("### 상세 payload(최근 1개)")
            j(rows[-1].get("payload_json"))


if __name__ == "__main__":
    main()
