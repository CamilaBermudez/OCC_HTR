"""Calibrated cross-model confidence router (+ bleed prior) — spec §6.13.

The ensemble oracle (per-line best of kraken/TrOCR) reaches 0.9788 char, beating the deployed
0.9743 — but a NAIVE max-confidence router only gets 0.9723 because kraken's CTC posterior and
TrOCR's token prob live on non-comparable scales. Fix: fit a per-model calibrator mapping each
model's raw per-line mean-confidence -> predicted per-line char accuracy on the 600 dev
(isotonic), then on the 300-val route each line to the model with the higher *calibrated* predicted
accuracy. Optional bleed prior: ink-bleed hurts kraken (ρ 0.21) and TrOCR is bleed-robust, so nudge
high-bleed lines toward TrOCR.

Protocol: calibrators fit on the 600 annotated (dev), reported on the full 300-val (matches the LM
λ-tuning). The kraken branch emits the LM-rescored text (deployed 0.9743) via --kraken-rescored so
the comparison against 0.9743 is fair; the routing DECISION still uses raw-CTC confidence (the
rescore has no per-char confidence — realistic, it is available pre-rescore at inference).

    PROJECT_ROOT=. uv run python scripts/ocr/confidence_router.py \
        --dev-dir tests/ocr/evaluations/router_dev600 \
        --val-dir tests/ocr/evaluations/longtail_confidence \
        --bleed tests/ocr/evaluations/ink_bleed_val300_20260718/ink_bleed_20260718_180817.json \
        --kraken-rescored tests/ocr/evaluations/longtail_confidence/kraken_rescored_val.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.isotonic import IsotonicRegression


def load(p):
    return {r["stem"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def corpus(rows):
    """rows = list of (pred, gt) -> (char_acc, word_acc)."""
    cd = cn = wd = wn = 0
    for pred, gt in rows:
        cd += Levenshtein.distance(pred, gt)
        cn += max(1, len(gt))
        wd += Levenshtein.distance(pred.split(), gt.split())
        wn += max(1, len(gt.split()))
    return 1 - cd / cn, 1 - wd / wn


def fit_calibrator(dev):
    """dev per_line rows -> isotonic map raw mean_conf -> per-line char accuracy."""
    x, y = [], []
    for r in dev.values():
        mc = float(r["mean_conf"])
        if not np.isfinite(mc):
            continue
        x.append(mc)
        y.append(min(1.0, max(0.0, 1.0 - float(r["cer"]))))
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, y)
    return iso


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-dir", type=Path, default=Path("tests/ocr/evaluations/router_dev600"))
    ap.add_argument(
        "--val-dir", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence")
    )
    ap.add_argument("--bleed", type=Path, required=True)
    ap.add_argument("--kraken-rescored", type=Path, default=None)
    ap.add_argument("--betas", default="0,0.01,0.02,0.05,0.1")
    a = ap.parse_args()

    dev_k, dev_t = load(a.dev_dir / "per_line_kraken.csv"), load(a.dev_dir / "per_line_trocr.csv")
    val_k, val_t = load(a.val_dir / "per_line_kraken.csv"), load(a.val_dir / "per_line_trocr.csv")
    bleed = json.loads(a.bleed.read_text())["images"]
    resc = load(a.kraken_rescored) if a.kraken_rescored else None

    iso_k, iso_t = fit_calibrator(dev_k), fit_calibrator(dev_t)

    stems = [s for s in val_k if s in val_t]
    # kraken OUTPUT text (rescored if available) vs its raw; TrOCR text; GT
    ktext = {
        s: (resc[s]["pred_rescored"] if resc and s in resc else val_k[s]["pred"]) for s in stems
    }
    kraw = {s: val_k[s]["pred"] for s in stems}
    ttext = {s: val_t[s]["pred"] for s in stems}
    gt = {s: val_k[s]["gt"] for s in stems}

    ak = {s: float(iso_k.predict([float(val_k[s]["mean_conf"])])[0]) for s in stems}
    at = {s: float(iso_t.predict([float(val_t[s]["mean_conf"])])[0]) for s in stems}
    bl = np.array([bleed.get(s + ".png", {}).get("bleed_score", np.nan) for s in stems])
    bmean, bstd = np.nanmean(bl), np.nanstd(bl)
    zbleed = {
        s: (0.0 if not np.isfinite(bl[i]) else (bl[i] - bmean) / (bstd + 1e-9))
        for i, s in enumerate(stems)
    }

    def route(chooser):
        """chooser(s) -> 'k' or 't'; returns (rows, n_trocr)."""
        rows, nt = [], 0
        for s in stems:
            if chooser(s) == "t":
                rows.append((ttext[s], gt[s]))
                nt += 1
            else:
                rows.append((ktext[s], gt[s]))
        return rows, nt

    def dist_k(s):
        return Levenshtein.distance(ktext[s], gt[s])

    def dist_t(s):
        return Levenshtein.distance(ttext[s], gt[s])

    # references
    refs = [
        ("all kraken raw (0.9710)", [(kraw[s], gt[s]) for s in stems], None),
        ("all kraken+LM (deployed)", [(ktext[s], gt[s]) for s in stems], None),
        ("all TrOCR (0.9617)", [(ttext[s], gt[s]) for s in stems], None),
    ]
    # oracle over the emitted texts (kraken+LM vs TrOCR)
    oracle_rows = [(ktext[s] if dist_k(s) <= dist_t(s) else ttext[s], gt[s]) for s in stems]
    n_or_t = sum(1 for s in stems if dist_t(s) < dist_k(s))
    # naive uncalibrated max mean-conf (for reference)
    naive_rows, n_naive_t = route(
        lambda s: "k" if float(val_k[s]["mean_conf"]) >= float(val_t[s]["mean_conf"]) else "t"
    )
    # calibrated confidence router
    conf_rows, n_conf_t = route(lambda s: "k" if ak[s] >= at[s] else "t")

    dep = corpus([(ktext[s], gt[s]) for s in stems])[0]
    print(f"{len(stems)} val lines  |  deployed (kraken+LM) char-acc = {dep:.4f}\n")
    print(
        f"{'strategy':<30} | {'char acc':>8} | {'word acc':>8} | {'→TrOCR':>6} | {'Δchar vs dep':>12}"
    )

    def row(name, rows, nt):
        ca, wa = corpus(rows)
        ntxt = "" if nt is None else f"{nt:>6}"
        print(f"{name:<30} | {ca:>8.4f} | {wa:>8.4f} | {ntxt:>6} | {100 * (ca - dep):>+11.2f}pp")
        return ca

    for name, rows, nt in refs:
        row(name, rows, nt)
    row("ORACLE (kraken+LM vs TrOCR)", oracle_rows, n_or_t)
    row("naive max mean-conf", naive_rows, n_naive_t)
    print("-" * 78)
    row("router: calibrated conf", conf_rows, n_conf_t)

    # + bleed prior (fixed β, not dev-tuned — val bleed only): route t if (at-ak)+β·z > 0
    for beta in [float(x) for x in a.betas.split(",")]:
        if beta == 0:
            continue
        rows, nt = route(lambda s, b=beta: "t" if (at[s] - ak[s]) + b * zbleed[s] > 0 else "k")
        row(f"router: calib conf + bleed β={beta}", rows, nt)

    # margin sweep: route to TrOCR only when it wins by margin τ (τ→∞ ⇒ all-kraken=deployed).
    # If NO τ beats deployed, the confidence signal is too weak to exploit the complementarity.
    print("\nmargin sweep — route→TrOCR iff (ât-âk) > τ:")
    best = (-1, None)
    for tau in [-0.02, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        rows, nt = route(lambda s, t=tau: "t" if (at[s] - ak[s]) > t else "k")
        ca = corpus(rows)[0]
        best = max(best, (ca, tau))
        print(f"  τ={tau:>5} | char {ca:.4f} ({100 * (ca - dep):+.2f}pp) | →TrOCR {nt:>3}")
    print(
        f"best operating point: char {best[0]:.4f} at τ={best[1]} "
        f"({'BEATS' if best[0] > dep else 'does NOT beat'} deployed 0.9743)"
    )

    # paired bootstrap: is (router − deployed) char-acc delta significant? (resample lines)
    def per_line_dist(rows):
        return np.array([[Levenshtein.distance(p, g), max(1, len(g))] for p, g in rows])

    dep_d = per_line_dist([(ktext[s], gt[s]) for s in stems])
    argmax_d = per_line_dist(route(lambda s: "k" if ak[s] >= at[s] else "t")[0])
    tau02_d = per_line_dist(route(lambda s: "t" if (at[s] - ak[s]) > 0.02 else "k")[0])
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(stems), size=(2000, len(stems)))

    def boot_delta(a_d):
        base_acc = 1 - dep_d[:, 0].sum() / dep_d[:, 1].sum()
        d = []
        for row_i in idx:
            ad, dd = a_d[row_i], dep_d[row_i]
            d.append((1 - ad[:, 0].sum() / ad[:, 1].sum()) - (1 - dd[:, 0].sum() / dd[:, 1].sum()))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return base_acc, 100 * np.mean(d), 100 * lo, 100 * hi

    print("\npaired bootstrap Δchar vs deployed (2000 resamples, 95% CI):")
    for nm, a_d in [("argmax router", argmax_d), ("router τ=0.02 (test-picked)", tau02_d)]:
        _, m, lo, hi = boot_delta(a_d)
        sig = "significant" if (lo > 0 or hi < 0) else "NOT significant (CI spans 0)"
        print(f"  {nm:<28} Δ {m:+.3f}pp  [{lo:+.3f}, {hi:+.3f}]  {sig}")

    # diagnostics
    agree = sum(1 for s in stems if (ak[s] >= at[s]) == (dist_k(s) <= dist_t(s)))
    base = sum(1 for s in stems if dist_k(s) <= dist_t(s))  # "always kraken" agreement
    print(
        f"\ncalibrated-conf router agrees with oracle pick on {agree}/{len(stems)} "
        f"({100 * agree / len(stems):.0f}%)  [naive 57%; always-kraken base rate "
        f"{100 * base / len(stems):.0f}%]"
    )


if __name__ == "__main__":
    main()
