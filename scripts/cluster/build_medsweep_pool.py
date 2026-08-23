"""Build one merged ViT pool (symlinks + merged ``labels.json``) for the
light-real + medical-volume sweep (spec §6.5.30 follow-up).

A pool = the fixed light-augmented real crops + one or more medical slots, each
contributing its first-N labelled pngs in sorted-filename order (so slots stay
nested: med1k's 1000 are the first 1000 of med4k's 4000). Multiple ``--medical``
groups let med7k = the leader's 4000 (from the med4k pool) + 3000 freshly
rendered lines, in one folder.

Every png is symlinked (no image copies) and keyed in the output labels.json;
fails loudly if a requested slot has fewer labelled pairs than asked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_labels(p: Path) -> dict[str, str]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def link_names(folder: Path, labels: dict[str, str], out: Path, names: list[str]) -> dict[str, str]:
    picked: dict[str, str] = {}
    for name in names:
        src = (folder / name).resolve()
        if not src.exists():
            raise FileNotFoundError(f"labelled png missing on disk: {src}")
        dst = out / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        picked[name] = labels[name]
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lightreal-folder", type=Path, required=True)
    ap.add_argument("--lightreal-labels", type=Path, required=True)
    ap.add_argument(
        "--medical",
        nargs=3,
        action="append",
        metavar=("FOLDER", "LABELS", "N"),
        required=True,
        help="A medical slot: image folder, labels.json, and how many (first-N "
        "sorted labelled pngs) to take. Repeatable.",
    )
    ap.add_argument("--out-folder", type=Path, required=True)
    ap.add_argument("--out-labels", type=Path, required=True)
    args = ap.parse_args()

    args.out_folder.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}

    lr_labels = load_labels(args.lightreal_labels)
    lr_names = sorted(lr_labels)
    merged.update(link_names(args.lightreal_folder, lr_labels, args.out_folder, lr_names))
    print(f"light-real: {len(lr_names)}")

    total_med = 0
    for folder, labels_path, n_str in args.medical:
        folder, labels_path, n = Path(folder), Path(labels_path), int(n_str)
        labels = load_labels(labels_path)
        names_all = sorted(labels)
        if len(names_all) < n:
            raise ValueError(f"{folder} has {len(names_all)} pairs < requested {n}")
        names = names_all[:n]
        dup = set(names) & set(merged)
        if dup:
            raise ValueError(f"filename collision across slots: {sorted(dup)[:3]} ... ({len(dup)})")
        merged.update(link_names(folder, labels, args.out_folder, names))
        total_med += n
        print(f"medical slot: +{n} from {folder}")

    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print(
        f"POOL {args.out_folder}  light-real={len(lr_names)}  medical={total_med}  "
        f"total={len(merged)}  labels={args.out_labels}"
    )


if __name__ == "__main__":
    main()
