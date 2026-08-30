"""Human-in-the-loop review queue + correction persistence (spec §6.13 HITL triage).

Builds a review queue over every line that has a model comparison (the ``line_compare`` JSONs that
drive Tab 4), ranked **worst-first** by kraken raw-CTC line **min-confidence** combined with the
**kraken↔TrOCR disagreement** — so the annotator's time goes to the lines most likely wrong (the
analysis behind this is spec §6.13: kraken's confidence generalises, min-conf is the best per-line
signal, and disagreement is an independent lift). Corrections are appended to a JSONL, one record per
save (crash-safe); the queue marks already-done lines from that file on every load. No pre-fill of the
edit field (avoids anchoring bias) — the model transcriptions are shown read-only for reference only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from frontend.config import REPO_ROOT, Config


def _min_conf(chars: list) -> float | None:
    """Line min-confidence = the least-confident emitted char (items are [char, prob, ...])."""
    vals = [c[1] for c in chars if len(c) > 1 and isinstance(c[1], int | float)]
    return min(vals) if vals else None


def _disagreement(a: str, b: str) -> float:
    """Normalised edit distance between the two models' text (0 = identical, 1 = fully disjoint)."""
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(1, len(a), len(b))


@dataclass(frozen=True)
class QueueLine:
    line_id: str
    page: str
    stem: str
    image: str  # "<page>/<stem>" — the /api/compare/<page>/image/<stem> crop
    kraken_text: str
    trocr_text: str
    catmus_text: str
    scholarly_text: str
    n_chars: int
    min_conf: float
    disagreement: float
    score: float  # sort key, ascending = review first (low conf and/or high disagreement)


def build_queue(cfg: Config) -> list[QueueLine]:
    """Read every line_compare page JSON → one ranked QueueLine per line with a confidence."""
    out: list[QueueLine] = []
    d = cfg.line_compare_dir
    if not d.is_dir():
        return out
    for jf in sorted(d.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a bad JSON file shouldn't kill the whole queue
            continue
        page = data.get("page", jf.stem)
        for ln in data.get("lines", []):
            k = ln.get("kraken") or {}
            t = ln.get("trocr") or {}
            c = ln.get("catmus") or {}
            s = ln.get("scholarly") or {}
            mc = _min_conf(k.get("chars") or c.get("chars") or [])
            if mc is None:  # no confidence for this line → nothing to rank on
                continue
            ktext, ttext = k.get("text", ""), t.get("text", "")
            dis = _disagreement(ktext, ttext)
            out.append(
                QueueLine(
                    line_id=ln["stem"],
                    page=page,
                    stem=ln["stem"],
                    image=f"{page}/{ln['stem']}",
                    kraken_text=ktext,
                    trocr_text=ttext,
                    catmus_text=c.get("text", ""),
                    scholarly_text=s.get("text", ""),
                    n_chars=len(ktext),
                    min_conf=round(float(mc), 4),
                    disagreement=round(dis, 4),
                    score=round(float(mc) - dis, 4),  # low conf OR high disagreement → review first
                )
            )
    out.sort(key=lambda q: q.score)
    return out


# ---- module cache: the queue is static per run; done-status is read live from the JSONL ----
_QUEUE: list[QueueLine] | None = None
_INDEX: dict[str, QueueLine] = {}


def get_queue(cfg: Config) -> list[QueueLine]:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = build_queue(cfg)
        _INDEX.clear()
        _INDEX.update({q.line_id: q for q in _QUEUE})
    return _QUEUE


def corrections_path() -> Path:
    """Where human corrections are appended (override with HITL_CORRECTIONS)."""
    return Path(
        os.environ.get(
            "HITL_CORRECTIONS",
            REPO_ROOT / "data/processed/human_annotations/corrections.jsonl",
        )
    )


def load_done(path: Path) -> dict[str, dict]:
    """line_id → the last saved record (later saves for the same line win)."""
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            try:
                r = json.loads(raw)
                done[r["line_id"]] = r
            except Exception:  # noqa: BLE001
                continue
    return done


def append_correction(path: Path, line_id: str, corrected_text: str, cfg: Config) -> dict:
    """Append one correction record for line_id and return it. No pre-fill, so a save with empty
    text is a legitimate 'reviewed, nothing to change is not assumed' — we still store what was typed.
    """
    get_queue(cfg)
    ql = _INDEX.get(line_id)
    record = {
        "line_id": line_id,
        "page": ql.page if ql else None,
        "image": ql.image if ql else None,
        "kraken_text": ql.kraken_text if ql else None,
        "trocr_text": ql.trocr_text if ql else None,
        "corrected_text": corrected_text,
        "changed_vs_kraken": bool(ql and corrected_text.strip() != (ql.kraken_text or "").strip()),
        "min_conf": ql.min_conf if ql else None,
        "disagreement": ql.disagreement if ql else None,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def queue_payload(cfg: Config, limit: int, skip_done: bool) -> dict:
    """The ranked queue for the frontend: pending (or all) lines + done-status + any saved text."""
    q = get_queue(cfg)
    done = load_done(corrections_path())
    rows = []
    for ql in q:
        is_done = ql.line_id in done
        if skip_done and is_done:
            continue
        row = asdict(ql)
        row["done"] = is_done
        row["corrected_text"] = done.get(ql.line_id, {}).get("corrected_text", "")
        rows.append(row)
        if len(rows) >= limit:
            break
    return {"total": len(q), "done": len(done), "pending": len(q) - len(done), "lines": rows}
