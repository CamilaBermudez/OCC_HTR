// AlbucE viewer — vanilla JS SPA.
//
// Two tabs share one page-payload fetch: when the user picks a page
// we hit `/api/pages/{key}` once, then render tab 1 and tab 2 from the
// same JSON. Selection is a shared {seg, sch} pair so clicking a model line
// (or polygon) highlights its aligned scholarly line and vice-versa. Both full
// transcriptions are always shown; the alignment only drives the highlight.

const state = {
    pages: [],
    currentPageKey: null,
    currentPage: null,       // full payload from /api/pages/{key}
    // Cross-highlight selection: a segmentation-line index (model side) and the
    // scholarly line NO it aligns to. Selecting on either side sets both.
    sel: { seg: null, sch: null },
    // Per-image-pane zoom level (1.0 = fit-to-pane at load time).
    // Two independent images (one per tab) so users can zoom on tab 1
    // and independently zoom on tab 2 without losing their place.
    zoom: { "1": 1, "2": 1 },
};

const ZOOM_MIN = 0.25;
const ZOOM_MAX = 8;
const ZOOM_STEP = 1.25;

// ---------- DOM helpers ----------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function setStatus(msg) {
    $("#status").textContent = msg;
}

// ---------- Fetching ----------
async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
}

async function loadPageList() {
    const { pages } = await fetchJson("/api/pages");
    state.pages = pages;
    const sel = $("#page-select");
    sel.innerHTML = "";
    for (const p of pages) {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        sel.appendChild(opt);
    }
    if (pages.length > 0) {
        await selectPage(pages[0]);
    } else {
        setStatus("No pages available — check server logs for path config.");
    }
}

async function selectPage(pageKey) {
    setStatus(`Loading ${pageKey}…`);
    state.currentPageKey = pageKey;
    state.sel = { seg: null, sch: null };
    // Reset zoom on page change — otherwise a previously-cranked-up
    // zoom persists silently and the new page opens at the wrong
    // magnification.
    state.zoom = { "1": 1, "2": 1 };
    state.currentPage = await fetchJson(`/api/pages/${encodeURIComponent(pageKey)}`);
    $("#page-select").value = pageKey;
    renderAll();
    const nLines = state.currentPage.lines.length;
    const nOurs = state.currentPage.lines.filter(l => l.our_text).length;
    const nScholarly = state.currentPage.scholarly_lines.length;
    setStatus(
        `${pageKey} · ${nLines} segmented lines · ${nOurs} with model transcription · ${nScholarly} scholarly lines`
    );
}

// ---------- Rendering ----------
// The image + SVG overlay is created twice — once for each tab. They
// share the same page data but each has its own DOM subtree so tabs
// can be flipped independently without re-rendering.
function renderImageInto(container, page, target) {
    container.innerHTML = "";
    const img = document.createElement("img");
    img.src = `/api/pages/${encodeURIComponent(page.page_key)}/image`;
    img.alt = page.page_key;
    // Once the image is decoded we know its natural dimensions and can
    // compute the fit-to-pane baseline; setZoom uses that + state.zoom.
    img.addEventListener("load", () => applyZoom(target));
    container.appendChild(img);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${page.image_width} ${page.image_height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    for (const line of page.lines) {
        if (!line.polygon || line.polygon.length < 3) continue;
        const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        poly.setAttribute("points", line.polygon.map(([x, y]) => `${x},${y}`).join(" "));
        poly.dataset.idx = String(line.idx);
        poly.dataset.tab = target;
        poly.addEventListener("click", () => setSelectedSeg(line.idx));
        svg.appendChild(poly);
    }
    container.appendChild(svg);
}

// Compact colored chips for one OCR line's differences vs the scholarly base
// (spec §6.7): each chip is "OCR span → scholarly span", colored by category,
// with the TEI encoding on hover.
function renderDiffChips(line) {
    const box = document.createElement("div");
    box.className = "diff-chips";
    for (const d of line.diffs || []) {
        const chip = document.createElement("span");
        // group = substantive | editorial | scramble (spec §6.7.2/6.7.3). Editorial
        // + scramble are hidden by CSS unless the "show editorial" toggle is on.
        const group = d.group || "substantive";
        chip.className = `diff-chip diff-${d.type} diff-grp-${group}`;
        const ocr = d.ocr_text || "∅";
        const base = d.base_text || "∅";
        chip.textContent = `${ocr}→${base}`;
        chip.title = `${d.type} (${group}): ${d.tei}`;
        box.appendChild(chip);
    }
    return box;
}

function renderLineList(container, page, textField, showDiffs = false) {
    container.innerHTML = "";
    for (const line of page.lines) {
        const row = document.createElement("div");
        row.className = "line-row";
        row.dataset.idx = String(line.idx);

        const idxCell = document.createElement("span");
        idxCell.className = "line-idx";
        idxCell.textContent = line.idx;
        row.appendChild(idxCell);

        const textCell = document.createElement("span");
        textCell.className = "line-text";
        const text = line[textField];
        if (text) {
            textCell.textContent = text;
        } else {
            textCell.classList.add("missing");
            textCell.textContent = "— no transcription —";
        }
        row.appendChild(textCell);

        if (showDiffs) {
            // Flag a transcribed model line that got no confident scholarly
            // alignment even after gap recovery — usually a scholarly-edition
            // error (e.g. several manuscript lines merged into one entry).
            const aligned = page.align && String(line.idx) in page.align;
            if (line.our_text && !aligned) {
                row.classList.add("unaligned");
                row.title = "no confident scholarly alignment (likely a scholarly-transcription merge)";
            }
            if (line.diffs && line.diffs.length) {
                row.appendChild(renderDiffChips(line));
            }
        }

        row.addEventListener("click", () => setSelectedSeg(line.idx));
        container.appendChild(row);
    }
}

// The FULL scholarly edition for the page, its own line numbering. Never
// filtered by the alignment — clicking a scholarly line cross-highlights the
// model line(s) it aligns to (spec §6.6).
function renderScholarlyList(container, page) {
    container.innerHTML = "";
    for (const line of page.scholarly_lines) {
        const row = document.createElement("div");
        row.className = "line-row";
        row.dataset.schno = String(line.no);

        const idxCell = document.createElement("span");
        idxCell.className = "line-idx";
        idxCell.textContent = line.no;
        row.appendChild(idxCell);

        const textCell = document.createElement("span");
        textCell.className = "line-text";
        textCell.textContent = line.text;
        row.appendChild(textCell);

        row.addEventListener("click", () => setSelectedSch(line.no));
        container.appendChild(row);
    }
}

function renderAll() {
    if (!state.currentPage) return;
    const p = state.currentPage;
    renderImageInto($("#image-wrap-1"), p, "1");
    renderImageInto($("#image-wrap-2"), p, "2");
    renderLineList($("#lines-ours-1"), p, "our_text");
    renderScholarlyList($("#lines-scholarly"), p);          // 3-way tab: FULL scholarly edition
    renderLineList($("#lines-ours-2"), p, "our_text", true); // 3-way tab: model + diff chips
    applySelection();
}

// ---------- Selection (cross-highlight between the two columns) ----------
// Selecting a model (segmentation) line highlights it + its aligned scholarly
// line; selecting a scholarly line highlights it + every model line aligned to
// it. Clicking the same thing again clears.
function setSelectedSeg(seg) {
    if (state.sel.seg === seg) {
        state.sel = { seg: null, sch: null };
    } else {
        const align = (state.currentPage && state.currentPage.align) || {};
        const sch = align[String(seg)];
        state.sel = { seg, sch: sch === undefined ? null : sch };
    }
    applySelection();
}

function setSelectedSch(no) {
    if (state.sel.sch === no) {
        state.sel = { seg: null, sch: null };
    } else {
        const align = (state.currentPage && state.currentPage.align) || {};
        // reverse map: first model line that aligns to this scholarly line
        let seg = null;
        for (const [s, n] of Object.entries(align)) {
            if (n === no) { seg = Number(s); break; }
        }
        state.sel = { seg, sch: no };
    }
    applySelection();
}

function applySelection() {
    const { seg, sch } = state.sel;
    const align = (state.currentPage && state.currentPage.align) || {};
    // Polygons + model rows highlight on the segmentation index.
    $$(".image-wrap polygon").forEach((poly) => {
        poly.classList.toggle("selected", seg !== null && Number(poly.dataset.idx) === seg);
    });
    $$(".line-row[data-idx]").forEach((row) => {
        // a model row is selected if it's the chosen seg, OR (when a scholarly
        // line is selected) if it aligns to that scholarly line — so 1:many
        // scholarly→model highlights all matching model lines.
        const idx = Number(row.dataset.idx);
        const isSel =
            (seg !== null && idx === seg) ||
            (sch !== null && align[String(idx)] === sch);
        row.classList.toggle("selected", isSel);
    });
    // Scholarly rows highlight on their line number.
    $$(".line-row[data-schno]").forEach((row) => {
        row.classList.toggle("selected", sch !== null && Number(row.dataset.schno) === sch);
    });
    if (seg !== null || sch !== null) {
        const activeTab = $(".tab-panel.active");
        if (activeTab) {
            activeTab.querySelectorAll(".line-row.selected").forEach((row) => {
                row.scrollIntoView({ block: "nearest", behavior: "smooth" });
            });
        }
    }
}

// ---------- Zoom ----------
// The image-wrap's explicit `width` is the single source of truth for
// on-screen size (with height: auto on the img and SVG width/height
// 100%, the SVG follows automatically). applyZoom computes
//    width = fit_to_pane_width * state.zoom[target]
// where the fit width is derived from the currently-visible image-scroll
// container so window resizes / tab switches recompute cleanly.
function applyZoom(target) {
    const wrap = $(`#image-wrap-${target}`);
    if (!wrap) return;
    const img = wrap.querySelector("img");
    if (!img || !img.naturalWidth) return;
    const scroll = wrap.closest(".image-scroll");
    if (!scroll) return;
    const paneW = scroll.clientWidth - 16;   // padding: 8px on each side
    const paneH = scroll.clientHeight - 16;
    // fit factor = whichever axis constrains us; at zoom=1 the image
    // is exactly contained within the pane.
    const fitFactor = Math.min(
        paneW / img.naturalWidth,
        paneH / img.naturalHeight,
    );
    const fitWidth = img.naturalWidth * Math.max(fitFactor, 0.01);
    const width = fitWidth * state.zoom[target];
    wrap.style.width = `${width}px`;

    // Update the readout in the same pane's toolbar.
    const pane = scroll.closest(".pane-image");
    const label = pane && pane.querySelector(".zoom-level");
    if (label) label.textContent = `${Math.round(state.zoom[target] * 100)}%`;
}

function setZoom(target, newZoom) {
    state.zoom[target] = Math.max(ZOOM_MIN, Math.min(newZoom, ZOOM_MAX));
    applyZoom(target);
}

function bindZoomHandlers() {
    // Toolbar buttons: −, +, reset. Each pane-image carries data-target
    // so one bound function handles both tabs.
    $$(".pane-image").forEach((pane) => {
        const target = pane.dataset.target;
        pane.querySelectorAll("[data-zoom-action]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const action = btn.dataset.zoomAction;
                if (action === "in") setZoom(target, state.zoom[target] * ZOOM_STEP);
                else if (action === "out") setZoom(target, state.zoom[target] / ZOOM_STEP);
                else if (action === "reset") setZoom(target, 1);
            });
        });
        // Cmd/Ctrl + wheel to zoom (Google Maps convention). Without a
        // modifier the wheel scrolls the pane normally, so users can
        // still pan by scrolling.
        const scroll = pane.querySelector(".image-scroll");
        if (scroll) {
            scroll.addEventListener("wheel", (e) => {
                if (!(e.ctrlKey || e.metaKey)) return;
                e.preventDefault();
                const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
                setZoom(target, state.zoom[target] * factor);
            }, { passive: false });
        }
    });
    // Re-fit on window resize so the "fit-to-pane" baseline stays honest.
    window.addEventListener("resize", () => {
        applyZoom("1");
        applyZoom("2");
    });
}

// ---------- Pan (click-and-drag on the image) ----------
// Google-Maps-style pan: mousedown anywhere in the image area (empty
// space OR on a polygon), drag, release. A short click without meaningful
// motion still fires the polygon's normal click handler so line selection
// keeps working — we only intercept once movement crosses PAN_THRESHOLD.
const PAN_THRESHOLD_PX = 4;

function bindPanHandlers() {
    $$(".pane-image").forEach((pane) => {
        const scroll = pane.querySelector(".image-scroll");
        const wrap = pane.querySelector(".image-wrap");
        if (!scroll || !wrap) return;

        let start = null;        // { x, y } — clientX/Y at mousedown
        let startScroll = null;  // pane's scrollLeft/Top at mousedown
        let panning = false;

        scroll.addEventListener("mousedown", (e) => {
            // Left button only. Ignore drags that begin on the zoom
            // toolbar buttons.
            if (e.button !== 0) return;
            if (e.target.closest(".zoom-toolbar")) return;
            start = { x: e.clientX, y: e.clientY };
            startScroll = { x: scroll.scrollLeft, y: scroll.scrollTop };
            panning = false;
        });

        // Use window-level listeners so a drag that leaves the pane's
        // bounds still updates smoothly and always terminates on mouseup.
        window.addEventListener("mousemove", (e) => {
            if (!start) return;
            const dx = e.clientX - start.x;
            const dy = e.clientY - start.y;
            if (!panning && Math.hypot(dx, dy) > PAN_THRESHOLD_PX) {
                panning = true;
                wrap.classList.add("panning");
            }
            if (panning) {
                scroll.scrollLeft = startScroll.x - dx;
                scroll.scrollTop = startScroll.y - dy;
                e.preventDefault();
            }
        });

        window.addEventListener("mouseup", () => {
            if (!start) return;
            const wasPanning = panning;
            start = null;
            startScroll = null;
            panning = false;
            wrap.classList.remove("panning");
            if (wasPanning) {
                // A drag has just ended. The polygon under the cursor
                // would otherwise fire its normal click handler and
                // toggle line selection — swallow that pending click.
                const suppress = (ev) => {
                    ev.stopPropagation();
                    ev.preventDefault();
                };
                window.addEventListener("click", suppress, {
                    capture: true,
                    once: true,
                });
            }
        });
    });
}

// ---------- Tabs ----------
function activateTab(tabId) {
    $$(".tab-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    $$(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === tabId);
    });
    // Copy/Download only make sense on tab 1.
    $("#tab-actions").style.display = tabId === "tab-transcription" ? "flex" : "none";
    // A pane that was previously ``display: none`` had clientWidth = 0
    // when its image loaded, so applyZoom used a bogus fit. Recompute
    // now that the pane is measurable again.
    const target = tabId === "tab-transcription" ? "1" : "2";
    applyZoom(target);
    applySelection();
}

// ---------- Copy / Download (Tab 1) ----------
function currentModelTranscription() {
    if (!state.currentPage) return "";
    return state.currentPage.lines
        .map((l) => (l.our_text || ""))
        .join("\n");
}

async function copyToClipboard() {
    const text = currentModelTranscription();
    try {
        await navigator.clipboard.writeText(text);
        setStatus(`Copied ${text.split("\n").length} lines to clipboard.`);
    } catch (e) {
        setStatus(`Copy failed: ${e.message}`);
    }
}

function downloadTranscription() {
    const text = currentModelTranscription();
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.currentPageKey}_model.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// ---------- Wire up ----------
function bindEvents() {
    $$(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => activateTab(btn.dataset.tab));
    });
    $("#page-select").addEventListener("change", (e) => selectPage(e.target.value));
    $("#copy-btn").addEventListener("click", copyToClipboard);
    $("#download-btn").addEventListener("click", downloadTranscription);
    const editorialToggle = $("#toggle-editorial");
    if (editorialToggle) {
        editorialToggle.addEventListener("change", () => {
            $("#tab-alignment").classList.toggle("show-editorial", editorialToggle.checked);
        });
    }
}

async function init() {
    bindEvents();
    bindZoomHandlers();
    bindPanHandlers();
    try {
        await loadPageList();
    } catch (e) {
        setStatus(`Failed to load page list: ${e.message}`);
    }
}

init();
