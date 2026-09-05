"""Build a merged ViT augmentation pool = all light-aug-real crops + first N
medical renders, as a *symlink* folder + merged ``labels.json`` (spec §6.5.30
medical-volume sweep).

``run_trocr_finetune.py`` globs one ``--augmented-folder`` for ``*.png`` and
requires every png to be keyed in ``--labels-json``. This builder makes such a
folder cheaply (symlinks, no image copies) by unioning:

  * ALL light-augmented real crops (held fixed across the sweep), and
  * the first ``--n-medical`` medical renders in sorted-filename order.

Sorted order makes the sweep **nested**: n-medical 1000 is a subset of 4000 is a
subset of the full medical pool, so the only thing that changes between runs is
the medical volume. Fails loudly if the medical pool has fewer than N usable
pairs (so a too-small pool can never silently shrink the sweep).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_labels(p: Path) -> dict[str, str]:
    return json.loads(p.read_text(encoding="utf-8"))


def link_subset(
    folder: Path, labels: dict[str, str], out_folder: Path, names: list[str]
) -> dict[str, str]:
    """Symlink ``names`` from ``folder`` into ``out_folder``; return their labels."""
    picked: dict[str, str] = {}
    for name in names:
        src = (folder / name).resolve()
        if not src.exists():
            raise FileNotFoundError(f"labelled png missing on disk: {src}")
        dst = out_folder / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        picked[name] = labels[name]
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lightreal-folder", type=Path, required=True)
    ap.add_argument("--lightreal-labels", type=Path, required=True)
    ap.add_argument("--medical-folder", type=Path, required=True)
    ap.add_argument("--medical-labels", type=Path, required=True)
    ap.add_argument("--n-medical", type=int, required=True)
    ap.add_argument("--out-folder", type=Path, required=True)
    ap.add_argument("--out-labels", type=Path, required=True)
    args = ap.parse_args()

    args.out_folder.mkdir(parents=True, exist_ok=True)

    lr_labels = load_labels(args.lightreal_labels)
    med_labels = load_labels(args.medical_labels)

    # light-real: take every labelled png (fixed component)
    lr_names = sorted(lr_labels)
    # medical: first N in sorted order (nested subset)
    med_names_all = sorted(med_labels)
    if len(med_names_all) < args.n_medical:
        raise ValueError(
            f"medical pool has only {len(med_names_all)} pairs < requested "
            f"{args.n_medical} (need a bigger medical render pool)"
        )
    med_names = med_names_all[: args.n_medical]

    merged: dict[str, str] = {}
    merged.update(link_subset(args.lightreal_folder, lr_labels, args.out_folder, lr_names))
    merged.update(link_subset(args.medical_folder, med_labels, args.out_folder, med_names))

    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print(
        f"pool={args.out_folder}  light-real={len(lr_names)}  "
        f"medical={len(med_names)}  total={len(merged)}  labels={args.out_labels}"
    )


if __name__ == "__main__":
    main()
