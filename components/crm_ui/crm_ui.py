# components/crm_ui/crm_ui.py
from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR = _COMPONENT_DIR / "frontend"

_PROJECT_ROOT = _COMPONENT_DIR.parent.parent
_UI_ROOT = _FRONTEND_DIR / "public" / "ui" / "v2"


def _extract_body_inner(html_text: str) -> str:
    """
    full html 문서에서 <body>...</body> 내부만 추출해서 fragment로 반환.
    (div.innerHTML에 넣기 안전)
    """
    m = re.search(r"<body[^>]*>(.*)</body>", html_text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return ""  # 실패 시 빈 문자열로 반환(상위에서 fallback 처리)

    body_inner = m.group(1)

    # body 안의 <script>는 충돌 가능성이 있어 제거 (필요 시 나중에 허용)
    body_inner = re.sub(
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
        "",
        body_inner,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return body_inner


def _build_page_html(page: str) -> str:
    """
    ui/v2/{page}.html + ui/v2/css/{page}.css 를 읽어서
    - link 태그 제거
    - CSS 인라인 주입
    - href 이동 막기
    - body 내부만 추출해서 fragment로 반환
    """
    html_path = _UI_ROOT / f"{page}.html"
    css_path = _UI_ROOT / "css" / f"{page}.css"

    if not html_path.exists():
        return f"<div style='padding:16px'>UI 파일이 없습니다: {html_path}</div>"
    if not css_path.exists():
        return f"<div style='padding:16px'>CSS 파일이 없습니다: {css_path}</div>"

    html_text = html_path.read_text(encoding="utf-8")
    css_text = css_path.read_text(encoding="utf-8")

    # 외부 css link 제거 (인라인로 대체)
    html_text = re.sub(r'<link[^>]+href="\./css/[^"]+"[^>]*>\s*', "", html_text)

    # 기존 파일에 ./first.html 같은 페이지 이동 링크가 있으면 막기
    html_text = re.sub(r'href="\./(index|first|second|third)\.html"', 'href="#"', html_text)

    body_inner = _extract_body_inner(html_text).strip()

    # 핵심: body 추출 실패하면 통째로 반환(최소 화면이라도 뜨게)
    if not body_inner:
        if os.environ.get("CRM_UI_DEBUG") == "1":
            st.sidebar.warning("[crm_ui] <body> extraction failed. Falling back to full HTML.")
            st.sidebar.write(html_text[:500])
        # HTML 전체에서 script 제거만 하고 반환
        safe_full = re.sub(
            r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
            "",
            html_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return f"<style>{css_text}</style>\n{safe_full}"

    fragment = f"<style>{css_text}</style>\n{body_inner}"
    return fragment


_crm_ui = components.declare_component("crm_ui", path=str(_FRONTEND_DIR))


def crm_ui(page: str, ui_state=None, result=None, height=900, key=None):
    ui_state = ui_state or {}
    result = result or {}
    page_html = _build_page_html(page)

    return _crm_ui(
        page=page,
        page_html=page_html,
        ui_state=ui_state,
        result=result,
        height=height,
        key=key,
        default=None,
    )
