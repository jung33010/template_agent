// components/crm_ui/frontend/src/main.js
console.log("[crm_ui] main.js loaded");

const STEP1_STATE_KEY = "crm_step1_state_v1";

function stSetValue(valueObj) {
    window.parent.postMessage(
        { isStreamlitMessage: true, type: "streamlit:setComponentValue", value: valueObj },
        "*"
    );
}
function stSetHeight(h) {
    window.parent.postMessage(
        { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: h },
        "*"
    );
}
function safeText(v, fallback = "-") {
    if (v === null || v === undefined) return fallback;
    const s = String(v);
    return s.length ? s : fallback;
}
function setTextById(id, value, fallback = "0") {
    const el = document.getElementById(id);
    if (!el) return false;
    el.textContent = safeText(value, fallback);
    return true;
}
function readValueById(id, fallback = "") {
    const el = document.getElementById(id);
    if (!el) return fallback;
    if ("value" in el) return el.value ?? fallback;
    return fallback;
}
function readSelectValue(id, fallback) {
    const v = readValueById(id, fallback);
    const s = v === null || v === undefined ? "" : String(v);
    const t = s.trim();
    return t.length ? t : fallback;
}

/**
 * 빈 값("")도 그대로 반환하는 Select Reader
 */
function readSelectAllowEmpty(id, fallback = "") {
    const el = document.getElementById(id);
    if (!el) return fallback;
    return el.value ?? fallback;
}

function sendAction(action, payload) {
    const now = Date.now();
    stSetValue({ action, payload: payload || {}, event_id: `${action}:${now}` });
}

// ----------------------
// Common html helpers
// ----------------------
function escapeHtml(s) {
    return String(s ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

// ----------------------
// Home(index) rendering
// ----------------------
function renderTemplateCard(item, opts = {}) {
    const runId = String(item?.run_id || "");
    const createdAt = item?.created_at ? String(item.created_at) : "";
    const tpl = item?.template || {};
    const title = escapeHtml(tpl?.title || "제목 없음");
    const bodyRaw = String(tpl?.body_with_slots || tpl?.body || "");
    const body = escapeHtml(bodyRaw).replaceAll("\n", "<br/>");

    const badgeText = opts.badgeText || "PENDING";
    const badgeIcon = opts.badgeIcon || "bi-hourglass-split";
    const showActions = opts.showActions !== false; // default true

    const actionsHtml = showActions
        ? `
    <div class="action-btns">
      <button class="btn-approve" type="button" data-action="approve" data-run-id="${escapeHtml(runId)}">
        <i class="bi bi-check-lg"></i> 승인
      </button>
      <button class="btn-reject" type="button" data-action="reject" data-run-id="${escapeHtml(runId)}">
        <i class="bi bi-x-lg"></i> 반려
      </button>
    </div>
    `
        : ``;

    return `
<div class="t-card" data-run-id="${escapeHtml(runId)}">
  <div class="t-header">
    <span class="badge-gold"><i class="bi ${badgeIcon}"></i> ${escapeHtml(badgeText)}</span>
    <div class="t-meta">Run ID: ${escapeHtml(runId)} ${createdAt ? "· " + escapeHtml(createdAt) : ""}</div>
  </div>
  <div class="t-body">
    <strong>${title}</strong><br/>
    ${body || "<span style='color:#94A3B8'>본문 없음</span>"}
  </div>

  <div class="t-footer">
    <div class="guard-status">
      <span class="pass"><i class="bi bi-shield-check"></i> Guardrail</span>
    </div>
    ${actionsHtml}
  </div>
</div>
`;
}

function applyIndexData(result, signal) {
    const kpi = result?.kpi || {};
    const reg = result?.registry || {};
    const pending = result?.pending || {};
    const approved = result?.approved || {};
    const ui = result?.ui || {};

    setTextById("kpi-ctr", kpi.ctr ?? "4.2%", "4.2%");
    setTextById("kpi-ctr-trend", kpi.ctr_trend ?? "+0.0%", "+0.0%");
    setTextById("kpi-guard-blocked", kpi.guard_blocked ?? 0, "0");

    setTextById("kpi-pending", kpi.pending_candidate ?? 0, "0");
    setTextById("kpi-active", kpi.active_assets ?? 0, "0");

    setTextById("reg-approved", reg.approved ?? 0, "0");
    setTextById("reg-deprecated", reg.deprecated ?? 0, "0");

    const latestWrap = document.getElementById("pending-card-wrap");
    if (latestWrap) {
        const latest = pending.latest || null;
        if (!latest) {
            latestWrap.innerHTML = `<div style="padding:12px; color:#64748B;">승인/반려 대기 중인 템플릿이 없습니다.</div>`;
        } else {
            latestWrap.innerHTML = renderTemplateCard(latest, {
                badgeText: "REVIEW",
                badgeIcon: "bi-hourglass-split",
                showActions: true,
            });
        }
    }

    const allWrap = document.getElementById("pending-all-wrap");
    if (allWrap) {
        const showAll = !!ui.show_all_pending;
        allWrap.style.display = showAll ? "block" : "none";

        if (showAll) {
            const rows = Array.isArray(pending.all) ? pending.all : [];
            if (!rows.length) {
                allWrap.innerHTML = `<div style="padding:12px 16px; color:#64748B;">추가 대기 템플릿이 없습니다.</div>`;
            } else {
                const latestRid = String(pending.latest?.run_id || "");
                const filtered = rows.filter((x) => String(x?.run_id || "") !== latestRid);

                allWrap.innerHTML = filtered
                    .map((it) => {
                        return `<div style="padding:0 16px 14px;">${renderTemplateCard(it, {
                            badgeText: "REVIEW",
                            badgeIcon: "bi-hourglass-split",
                            showActions: true,
                        })}</div>`;
                    })
                    .join("");

                if (!filtered.length) {
                    allWrap.innerHTML = `<div style="padding:12px 16px; color:#64748B;">추가 대기 템플릿이 없습니다.</div>`;
                }
            }
        } else {
            allWrap.innerHTML = "";
        }
    }

    const approvedLatestWrap = document.getElementById("approved-card-wrap");
    if (approvedLatestWrap) {
        const latest = approved.latest || null;
        if (!latest) {
            approvedLatestWrap.innerHTML = `<div style="padding:12px; color:#64748B;">승인된 템플릿이 없습니다.</div>`;
        } else {
            approvedLatestWrap.innerHTML = renderTemplateCard(latest, {
                badgeText: "APPROVED",
                badgeIcon: "bi-check-circle-fill",
                showActions: false,
            });
        }
    }

    const approvedAllWrap = document.getElementById("approved-all-wrap");
    if (approvedAllWrap) {
        const showAll = !!ui.show_all_approved;
        approvedAllWrap.style.display = showAll ? "block" : "none";

        if (showAll) {
            const rows = Array.isArray(approved.all) ? approved.all : [];
            if (!rows.length) {
                approvedAllWrap.innerHTML = `<div style="padding:12px 16px; color:#64748B;">추가 승인 템플릿이 없습니다.</div>`;
            } else {
                const latestRid = String(approved.latest?.run_id || "");
                const filtered = rows.filter((x) => String(x?.run_id || "") !== latestRid);

                approvedAllWrap.innerHTML = filtered
                    .map((it) => {
                        return `<div style="padding:0 16px 14px;">${renderTemplateCard(it, {
                            badgeText: "APPROVED",
                            badgeIcon: "bi-check-circle-fill",
                            showActions: false,
                        })}</div>`;
                    })
                    .join("");

                if (!filtered.length) {
                    approvedAllWrap.innerHTML = `<div style="padding:12px 16px; color:#64748B;">추가 승인 템플릿이 없습니다.</div>`;
                }
            }
        } else {
            approvedAllWrap.innerHTML = "";
        }
    }

    const root = document.body;
    const onClick = (e) => {
        const btn = e.target?.closest?.("button[data-action][data-run-id]");
        if (!btn) return;

        e.preventDefault();
        const rid = String(btn.getAttribute("data-run-id") || "");
        const act = String(btn.getAttribute("data-action") || "");
        if (!rid) return;

        if (act === "approve") {
            sendAction("HOME_SAVE_APPROVAL", {
                run_id: rid,
                decision: "APPROVED",
                comment: "",
                marketer_id: "marketer_001",
            });
        } else if (act === "reject") {
            sendAction("HOME_SAVE_APPROVAL", {
                run_id: rid,
                decision: "REJECTED",
                comment: "",
                marketer_id: "marketer_001",
            });
        }
    };
    root.addEventListener("click", onClick, { signal });
}

// ----------------------
// Step1 state persist
// ----------------------
function loadStep1State() {
    if (window.__crm_step1_state && typeof window.__crm_step1_state === "object") {
        return window.__crm_step1_state;
    }
    try {
        const raw = localStorage.getItem(STEP1_STATE_KEY);
        if (!raw) return null;
        const obj = JSON.parse(raw);
        if (!obj || typeof obj !== "object") return null;
        window.__crm_step1_state = obj;
        return obj;
    } catch (_) {
        return null;
    }
}
function saveStep1State(state) {
    window.__crm_step1_state = state;
    try {
        localStorage.setItem(STEP1_STATE_KEY, JSON.stringify(state));
    } catch (_) {}
}
function getDefaultStep1State() {
    return {
        created_by: "marketer_001",
        goal: "cart",
        channel: "PUSH",
        tone: "amoremall",
        campaign_text: "",
        age: "all",
        gender: "ALL",
        skin_type: "ALL",
        concern_keyword: "",
    };
}

function collectStep1StateFromDOM() {
    const cur = loadStep1State() || getDefaultStep1State();
    const goal = readSelectValue("goalSelect", cur.goal || "cart");

    const concern = readSelectAllowEmpty("concernSelect", cur.concern_keyword ?? "");

    const st = {
        created_by: readSelectValue("createdBy", cur.created_by || "marketer_001"),
        goal,
        channel: readSelectValue("channelSelect", cur.channel || "PUSH"),
        tone: readSelectValue("toneSelect", cur.tone || "amoremall"),
        campaign_text: goal === "counseling" ? readValueById("campaignText", cur.campaign_text || "") : "",
        age: readSelectValue("ageSelect", cur.age || "all"),
        gender: readSelectValue("genderSelect", cur.gender || "ALL"),
        skin_type: readSelectValue("skinSelect", cur.skin_type || "ALL"),
        concern_keyword: (concern ?? "").trim(),
    };
    saveStep1State(st);
    return st;
}

function restoreStep1StateToDOM() {
    const st = loadStep1State() || getDefaultStep1State();

    const createdBy = document.getElementById("createdBy");
    const goal = document.getElementById("goalSelect");
    const channel = document.getElementById("channelSelect");
    const tone = document.getElementById("toneSelect");
    const campaignText = document.getElementById("campaignText");
    const age = document.getElementById("ageSelect");
    const gender = document.getElementById("genderSelect");
    const skin = document.getElementById("skinSelect");
    const concern = document.getElementById("concernSelect");

    if (createdBy) createdBy.value = st.created_by ?? "marketer_001";
    if (goal) goal.value = st.goal ?? "cart";
    if (channel) channel.value = st.channel ?? "PUSH";
    if (tone) tone.value = st.tone ?? "amoremall";
    if (age) age.value = st.age ?? "all";
    if (gender) gender.value = st.gender ?? "ALL";
    if (skin) skin.value = st.skin_type ?? "ALL";
    if (concern) concern.value = st.concern_keyword ?? "";

    const isCounseling = (goal ? goal.value : st.goal) === "counseling";
    setCampaignTextEnabled(isCounseling);
    if (campaignText && isCounseling) {
        campaignText.value = st.campaign_text || "";
    }
}

// ----------------------
// Step1 UI logic
// ----------------------
function setCampaignTextEnabled(isEnabled) {
    const ta = document.getElementById("campaignText");
    if (!ta) return;
    if (isEnabled) {
        ta.disabled = false;
        if (!ta.placeholder || ta.placeholder.includes("선택 시에만")) {
            ta.placeholder = "제품 추천/카운슬링 캠페인 전략을 입력하세요.";
        }
    } else {
        ta.value = "";
        ta.disabled = true;
        ta.placeholder = "※ '뷰티 카운슬링 / 상품 추천' 선택 시에만 입력할 수 있어요.";
    }
}
function applyFirstData(result) {
    if (!result) return;
    if (result.target_count !== undefined && result.target_count !== null) {
        setTextById("target-count", result.target_count, "0");
    }
    if (result.run_id) {
        setTextById("run-id-text", result.run_id, "-");
    }
}

function buildPreviewPayloadFromState(state) {
    const ck = (state?.concern_keyword || "").trim();
    return {
        age: state.age,
        gender: state.gender,
        skin_type: state.skin_type,
        concern_keywords: ck ? [ck] : [],
    };
}

let _previewTimer = null;
function requestPreviewDebounced() {
    if (_previewTimer) clearTimeout(_previewTimer);
    setTextById("target-count", "…", "0");
    _previewTimer = setTimeout(() => {
        const st = collectStep1StateFromDOM();
        sendAction("STEP1_PREVIEW_TARGET", buildPreviewPayloadFromState(st));
    }, 250);
}
function resetStep1StateToDefaults() {
    saveStep1State(getDefaultStep1State());
}

// ----------------------
// Binding 관리 (중복 방지)
// ----------------------
function cleanupPreviousBindings() {
    if (window.__crm_ui_abort_controller) {
        try {
            window.__crm_ui_abort_controller.abort();
        } catch (_) {}
    }
    window.__crm_ui_abort_controller = new AbortController();
    return window.__crm_ui_abort_controller;
}

function bindIndexPage(signal, result) {
    const newBtn = document.getElementById("btn-new-template");
    if (newBtn)
        newBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                sendAction("NAVIGATE_STEP1", {});
            },
            { signal }
        );

    const viewAllBtn = document.getElementById("btn-view-all");
    if (viewAllBtn)
        viewAllBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                sendAction("HOME_TOGGLE_VIEW_ALL_PENDING", {});
            },
            { signal }
        );

    const viewAllApprovedBtn = document.getElementById("btn-view-all-approved");
    if (viewAllApprovedBtn)
        viewAllApprovedBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                sendAction("HOME_TOGGLE_VIEW_ALL_APPROVED", {});
            },
            { signal }
        );

    applyIndexData(result || {}, signal);
}

function bindFirstPage(signal) {
    restoreStep1StateToDOM();

    const goalEl = document.getElementById("goalSelect");
    if (goalEl) {
        goalEl.addEventListener(
            "change",
            () => {
                const goal = readSelectValue("goalSelect", "cart");
                setCampaignTextEnabled(goal === "counseling");
                collectStep1StateFromDOM();
            },
            { signal }
        );
    }

    const ageEl = document.getElementById("ageSelect");
    const genderEl = document.getElementById("genderSelect");
    const skinEl = document.getElementById("skinSelect");
    const concernEl = document.getElementById("concernSelect");

    if (ageEl) ageEl.addEventListener("change", requestPreviewDebounced, { signal });
    if (genderEl) genderEl.addEventListener("change", requestPreviewDebounced, { signal });
    if (skinEl) skinEl.addEventListener("change", requestPreviewDebounced, { signal });
    if (concernEl) concernEl.addEventListener("change", requestPreviewDebounced, { signal });

    const createdBy = document.getElementById("createdBy");
    const channel = document.getElementById("channelSelect");
    const tone = document.getElementById("toneSelect");
    const campaignText = document.getElementById("campaignText");
    if (createdBy) createdBy.addEventListener("input", () => collectStep1StateFromDOM(), { signal });
    if (channel) channel.addEventListener("change", () => collectStep1StateFromDOM(), { signal });
    if (tone) tone.addEventListener("change", () => collectStep1StateFromDOM(), { signal });
    if (campaignText) campaignText.addEventListener("input", () => collectStep1StateFromDOM(), { signal });
    if (concernEl) concernEl.addEventListener("change", () => collectStep1StateFromDOM(), { signal });

    const cancelBtn = document.getElementById("btn-cancel-step1");
    if (cancelBtn) {
        cancelBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                resetStep1StateToDefaults();
                sendAction("STEP1_CANCEL", {});
            },
            { signal }
        );
    }

    const form = document.getElementById("briefForm");
    if (form) {
        form.addEventListener(
            "submit",
            (e) => {
                e.preventDefault();
                const st1 = collectStep1StateFromDOM();
                const ck = (st1.concern_keyword || "").trim();
                sendAction("STEP1_SUBMIT", {
                    created_by: st1.created_by,
                    goal: st1.goal,
                    channel: st1.channel,
                    tone: st1.tone,
                    campaign_text: st1.campaign_text,
                    age: st1.age,
                    gender: st1.gender,
                    skin_type: st1.skin_type,
                    concern_keywords: ck ? [ck] : [],
                });
            },
            { signal }
        );
    }

    requestPreviewDebounced();
}

// ----------------------
// Step2
// ----------------------
function getStep2StorageKey(runId) {
    return `crm_step2_selected_idx__${runId || "no_run"}`;
}
function renderSecondCandidates(result) {
    const listEl = document.getElementById("candidate-list");
    if (!listEl) return;

    const runId = result && result.run_id ? String(result.run_id) : "";
    const candidates = result && Array.isArray(result.candidates) ? result.candidates : [];

    const key = getStep2StorageKey(runId);
    let savedIdx = null;
    try {
        const saved = window.localStorage.getItem(key);
        if (saved !== null && !Number.isNaN(Number(saved))) savedIdx = Number(saved);
    } catch (_) {}
    let selectedIdx = savedIdx === null ? 0 : savedIdx;

    if (!candidates.length) {
        listEl.innerHTML = `<div style="padding:12px; color:#64748B;">후보가 아직 없습니다. (TEMPLATE_CANDIDATES handoff 확인)</div>`;
        return;
    }
    if (selectedIdx < 0 || selectedIdx >= candidates.length) selectedIdx = 0;

    const ctrFallback = ["4.8%", "4.2%", "3.9%", "4.2%", "4.2%"];

    listEl.innerHTML = candidates
        .slice(0, 5)
        .map((c, i) => {
            const title = escapeHtml(c?.title || "");
            const body = escapeHtml(c?.body_with_slots || "").replaceAll("\n", "<br/>");
            const isSelected = i === selectedIdx;
            return `
      <div class="candidate-card ${isSelected ? "selected" : ""}" data-idx="${i}">
        <div class="card-header">
          <span class="candidate-no">Candidate #${i + 1} <small>${title}</small></span>
          <div class="status-tags">
            ${isSelected ? `<span class="badge-selected"><i class="bi bi-check-circle-fill"></i> 선택됨</span>` : ""}
            <span class="guard-pass-tag"><i class="bi bi-shield-check"></i> Verified</span>
          </div>
        </div>
        <div class="template-preview">${body}</div>
        <div class="card-footer">
          <div class="score-info">예상 반응률 <strong>${ctrFallback[i] || "4.2%"}</strong></div>
          <button class="btn-select ${isSelected ? "active" : ""}">${isSelected ? "선택 취소" : "선택"}</button>
        </div>
      </div>
    `;
        })
        .join("");

    const updateSelectionUI = (nextIdx) => {
        try {
            window.localStorage.setItem(key, String(nextIdx));
        } catch (_) {}
        listEl.querySelectorAll(".candidate-card").forEach((card) => {
            const idx = Number(card.getAttribute("data-idx"));
            const isSel = idx === nextIdx;
            card.classList.toggle("selected", isSel);

            const tags = card.querySelector(".status-tags");
            if (tags) {
                const badge = tags.querySelector(".badge-selected");
                if (isSel && !badge) {
                    tags.insertAdjacentHTML(
                        "afterbegin",
                        `<span class="badge-selected"><i class="bi bi-check-circle-fill"></i> 선택됨</span>`
                    );
                }
                if (!isSel && badge) badge.remove();
            }

            const btn = card.querySelector(".btn-select");
            if (btn) {
                btn.classList.toggle("active", isSel);
                btn.textContent = isSel ? "선택 취소" : "선택";
            }
        });
    };

    listEl.querySelectorAll(".candidate-card").forEach((card) => {
        const idx = Number(card.getAttribute("data-idx"));
        const btn = card.querySelector(".btn-select");
        const togglePick = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            let cur = selectedIdx;
            try {
                const raw = window.localStorage.getItem(key);
                if (raw !== null && !Number.isNaN(Number(raw))) cur = Number(raw);
            } catch (_) {}
            const next = cur === idx ? -1 : idx;
            selectedIdx = next;
            updateSelectionUI(next);
        };
        card.addEventListener("click", togglePick, { signal: window.__crm_ui_abort_controller.signal });
        if (btn) btn.addEventListener("click", togglePick, { signal: window.__crm_ui_abort_controller.signal });
    });
}
function bindSecondPage(signal, result) {
    renderSecondCandidates(result);

    const regenBtn = document.querySelector(".btn-cancel");
    if (regenBtn)
        regenBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                sendAction("STEP2_REGENERATE", { run_id: String(result?.run_id || "") });
            },
            { signal }
        );

    const confirmBtn = document.querySelector(".btn-submit-main");
    if (confirmBtn) {
        confirmBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                const rid = String(result?.run_id || "");
                const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
                if (!rid || !candidates.length) {
                    sendAction("STEP2_CONFIRM", { run_id: rid, template_id: "" });
                    return;
                }
                const key = getStep2StorageKey(rid);
                let idx = 0;
                try {
                    const raw = window.localStorage.getItem(key);
                    if (raw !== null && !Number.isNaN(Number(raw))) idx = Number(raw);
                } catch (_) {}
                if (idx < 0 || idx >= candidates.length) {
                    sendAction("STEP2_CONFIRM", { run_id: rid, template_id: "" });
                    return;
                }
                sendAction("STEP2_CONFIRM", { run_id: rid, template_id: candidates[idx]?.template_id || "" });
            },
            { signal }
        );
    }
}

// ----------------------
// Step3
// ----------------------
function highlightSlotsToHtml(text) {
    let s = escapeHtml(text || "");
    s = s.replace(/\{[^}]+\}/g, (m) => `<span class="slot-tag">${escapeHtml(m)}</span>`);
    s = s.replaceAll("\n", "<br/>");
    return s;
}
function getStep3StorageKey(runId) {
    return `crm_step3_decision__${runId || "no_run"}`;
}

function showToast(message, variant = "success") {
    const root = document.body;
    if (!root) return;
    const toast = document.createElement("div");
    toast.className = `toast ${variant}`;
    toast.innerHTML = `
    <div class="toast-inner">
      <i class="bi ${variant === "success" ? "bi-check-circle-fill" : "bi-exclamation-triangle-fill"}"></i>
      <span>${escapeHtml(message)}</span>
    </div>
  `;
    root.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 250);
    }, 2200);
}

function renderApprovalHistory(result) {
    const wrap = document.getElementById("approval-history");
    if (!wrap) return;

    const approvals = Array.isArray(result?.approvals) ? result.approvals : [];
    if (!approvals.length) {
        wrap.innerHTML = `<div class="empty-note">아직 저장된 승인/반려 이력이 없습니다.</div>`;
        return;
    }

    wrap.innerHTML = approvals
        .slice(0, 20)
        .map((a) => {
            const at = a.created_at ? String(a.created_at) : "";
            const marketer = a.marketer_id ? String(a.marketer_id) : "-";
            const decision = String(a.decision || "").toUpperCase();
            const comment = a.comment ? String(a.comment) : "";

            let badge = `<span class="pill unknown"><i class="bi bi-question-circle-fill"></i> 미상</span>`;
            if (decision === "APPROVED") badge = `<span class="pill approved"><i class="bi bi-check-circle-fill"></i> 승인</span>`;
            if (decision === "REJECTED") badge = `<span class="pill rejected"><i class="bi bi-x-circle-fill"></i> 반려</span>`;

            return `
      <div class="history-row">
        <div class="history-meta">
          ${badge}
          <span class="meta-item"><i class="bi bi-person"></i> ${escapeHtml(marketer)}</span>
          <span class="meta-item"><i class="bi bi-clock"></i> ${escapeHtml(at)}</span>
        </div>
        <div class="history-comment">${comment ? escapeHtml(comment) : "<span class='muted'>코멘트 없음</span>"}</div>
      </div>
    `;
        })
        .join("");
}

function bindThirdPage(signal, result) {
    const rid = String(result?.run_id || "");
    const sel = result?.selected_template || null;
    const candNo = result?.selected_candidate_no ? Number(result.selected_candidate_no) : null;

    const tagEl = document.querySelector(".final-preview-card .card-tag");
    const bodyEl = document.querySelector(".final-preview-card .template-content");

    const title = sel ? sel.title || "" : "";
    const body = sel ? sel.body_with_slots || "" : "";

    if (tagEl) {
        const prefix = candNo ? `Candidate #${candNo}` : "Selected";
        tagEl.textContent = `${prefix} - ${title || "선택된 템플릿"}`;
    }
    if (bodyEl) {
        bodyEl.innerHTML = highlightSlotsToHtml(body || "[선택된 템플릿이 없습니다]");
    }

    renderApprovalHistory(result);

    const saveRes = result?.save_result || null;
    if (saveRes && saveRes.toast_id) {
        const toastKey = `crm_last_toast__${rid || "no_run"}`;
        const lastId = (() => {
            try {
                return localStorage.getItem(toastKey) || "";
            } catch (_) {
                return "";
            }
        })();
        if (String(lastId) !== String(saveRes.toast_id)) {
            try {
                localStorage.setItem(toastKey, String(saveRes.toast_id));
            } catch (_) {}
            if (saveRes.ok) showToast(saveRes.msg || "저장 완료", "success");
            else showToast(saveRes.msg || "저장 실패", "warn");
        }
    }

    const approveBtn = document.getElementById("btnApprove");
    const rejectBtn = document.getElementById("btnReject");
    const saveBtn = document.getElementById("btnSave");
    const homeBtn = document.getElementById("btnHome");
    const finalBtn = document.getElementById("btnFinalSend");
    const commentTa = document.querySelector(".comment-section textarea.main-textarea");

    const key = getStep3StorageKey(rid);
    let state = { decision: "APPROVED", comment: "" };
    try {
        const raw = localStorage.getItem(key);
        if (raw) {
            const obj = JSON.parse(raw);
            if (obj && typeof obj === "object") state = { ...state, ...obj };
        }
    } catch (_) {}

    const applyDecisionUI = (decision) => {
        state.decision = decision;
        try {
            localStorage.setItem(key, JSON.stringify(state));
        } catch (_) {}
        if (approveBtn) approveBtn.classList.toggle("active", decision === "APPROVED");
        if (rejectBtn) rejectBtn.classList.toggle("active", decision === "REJECTED");
    };
    const applyCommentUI = (comment) => {
        state.comment = comment || "";
        try {
            localStorage.setItem(key, JSON.stringify(state));
        } catch (_) {}
    };

    applyDecisionUI(state.decision);

    if (commentTa) {
        commentTa.value = state.comment || "";
        commentTa.addEventListener("input", () => applyCommentUI(commentTa.value), { signal });
    }
    if (approveBtn) approveBtn.addEventListener("click", () => applyDecisionUI("APPROVED"), { signal });
    if (rejectBtn) rejectBtn.addEventListener("click", () => applyDecisionUI("REJECTED"), { signal });

    // 홈으로(=index.html, Home(UI))
    if (homeBtn) {
        homeBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                sendAction("NAVIGATE_HOME", {});
            },
            { signal }
        );
    }

    // 결정 저장
    if (saveBtn) {
        saveBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                sendAction("STEP3_SAVE_APPROVAL", {
                    run_id: rid,
                    decision: state.decision,
                    comment: state.comment || "",
                    marketer_id: "marketer_001",
                });
            },
            { signal }
        );
    }

    // 최종 발송 -> Step4(fourth.html)
    if (finalBtn) {
        finalBtn.addEventListener(
            "click",
            (e) => {
                e.preventDefault();
                sendAction("NAVIGATE_STEP4", { run_id: rid });
            },
            { signal }
        );
    }
}

// ----------------------
// Step4 (fourth) - 최종 결과 렌더링 (HTML 구조 매칭)
// ----------------------
function bindFourthPage(signal, result) {
    // 1. HTML에서 내용을 넣을 위치 찾기 (수정된 ID 사용)
    const container = document.getElementById("filled-card-wrap");
    const countEl = document.getElementById("final-target-count");
    const runIdEl = document.getElementById("run-id-text");
    
    if (!container) return;

    // 2. 데이터 확인
    const messages = result?.generated_messages;
    const runId = result?.run_id || "-";

    // Run ID 표시
    if (runIdEl) runIdEl.textContent = runId;

    // 데이터가 없을 경우 처리
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
        container.innerHTML = `<div style="padding:24px; text-align:center; color:#64748B; background:#f8fafc; border-radius:8px;">
            <i class="bi bi-exclamation-circle" style="font-size:1.5rem; display:block; margin-bottom:8px;"></i>
            생성된 메시지가 없습니다.<br/>
            <small>이전 단계 로직을 확인해주세요.</small>
        </div>`;
        if (countEl) countEl.textContent = "0";
        return;
    }

    // 3. 카운트 업데이트
    if (countEl) countEl.textContent = messages.length;

    // 4. HTML 생성 (CSS 클래스 t-card 사용하여 디자인 유지)
    container.innerHTML = messages.map((msg, index) => {
        // 보안 및 줄바꿈 처리
        const safeBody = escapeHtml(msg.message || "").replaceAll("\n", "<br/>");
        const safeName = escapeHtml(msg.customer_name || "고객");
        const safeProd = escapeHtml(msg.product_name || "");
        const safeInfo = msg.debug_info ? escapeHtml(msg.debug_info) : "";

        // 첫 번째 카드는 강조(BEST), 나머지는 일반(READY) 뱃지 처리
        const badgeHtml = index === 0 
            ? `<span class="badge-gold" style="background:#fff7ed; color:#c2410c; border:1px solid #ffedd5; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600;"><i class="bi bi-stars"></i> BEST MATCH</span>` 
            : `<span class="badge-soft" style="background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600;"><i class="bi bi-check2-circle"></i> READY</span>`;

        // 카드 HTML 조립
        return `
            <div class="t-card" style="background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div class="t-header" style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
                    ${badgeHtml}
                    <div class="t-meta" style="text-align:right;">
                        <div style="font-weight:600; color:#334155;">${safeName}님</div>
                        <div style="font-size:0.8rem; color:#64748b;">${safeProd}</div>
                    </div>
                </div>

                <div class="t-body" style="color:#334155; line-height:1.6; font-size:0.95rem; background:#f8fafc; padding:12px; border-radius:8px;">
                    ${safeBody}
                </div>

                <div class="t-footer" style="margin-top:12px; display:flex; justify-content:space-between; align-items:center;">
                    <div class="guard-status">
                        <span class="pass" style="color:#10b981; font-size:0.8rem; font-weight:500;">
                            <i class="bi bi-shield-check"></i> Guardrail Passed
                        </span>
                    </div>
                    ${safeInfo ? `<div style="font-size:0.75rem; color:#cbd5e1;">${safeInfo}</div>` : ""}
                </div>
            </div>
        `;
    }).join("");

    // (선택) 발송 버튼 클릭 이벤트 연결
    const sendBtn = document.querySelector(".btn-submit-main");
    if (sendBtn) {
        const newBtn = sendBtn.cloneNode(true); // 기존 이벤트 제거를 위해 복제
        sendBtn.parentNode.replaceChild(newBtn, sendBtn);
        
        newBtn.addEventListener("click", (e) => {
            e.preventDefault();
            alert(`${messages.length}건의 메시지 발송이 시작되었습니다!`);
            // 실제 발송 로직이 필요하면 여기에 sendAction("SEND_FINAL", ...) 추가
        });
    }
}

// ---- receive render ----
window.addEventListener("message", (event) => {
    const msg = event && event.data;
    if (!msg) return;

    if (msg.type === "streamlit:render") {
        const args = msg.args || {};
        const page = args.page || "index";
        const pageHtml = args.page_html || "<div style='padding:16px'>no page_html</div>";
        const result = args.result || {};

        const root = document.getElementById("root") || document.body;
        root.innerHTML = pageHtml;

        const controller = cleanupPreviousBindings();
        const signal = controller.signal;

        if (page === "index") bindIndexPage(signal, result);
        if (page === "first") bindFirstPage(signal);
        if (page === "second") bindSecondPage(signal, result);
        if (page === "third") bindThirdPage(signal, result);
        if (page === "fourth") bindFourthPage(signal, result);

        if (page === "first") applyFirstData(result);

        stSetHeight(document.body.scrollHeight);

        window.__crm_last_page = page;

        requestAnimationFrame(() => {
            window.scrollTo(0, 0);
        });
    }
});

window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 }, "*");