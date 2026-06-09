import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_augmentation.label_correction import (
    DEFAULT_SUBSTITUTIONS,
    correct_labels,
)


def _parse_substitutions(spec: str) -> dict[str, str]:
    """Parse ``"v:u,V:U,j:i,J:I"`` style strings into a dict.

    Each comma-separated entry must be ``"from:to"``. Whitespace around
    keys/values is stripped. Duplicates use the last write.
    """
    out: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Substitution entry must be 'from:to'; got {pair!r}. "
                "Example: --substitutions 'v:u,V:U,j:i,J:I'"
            )
        k, v = pair.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    default_subs_str = ",".join(f"{k}:{v}" for k, v in DEFAULT_SUBSTITUTIONS.items())

    parser = argparse.ArgumentParser(
        description=(
            "Build a {augmented_image_name: corrected_label} JSON for a "
            "finished augmentation run. The augmented images directory is "
            "expected to be named aug_<timestamp>/; the output is written to "
            "<output-base-dir>/labels_<timestamp>/labels.json so the two are "
            "paired by timestamp."
        )
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Medieval-text-generation labels.json (the file produced by "
        "run_medieval_text_generation.py — keyed by source image name).",
    )
    parser.add_argument(
        "--augmented-folder",
        required=True,
        help="Augmented images directory (typically aug_<timestamp>/). The "
        "timestamp from this folder name is reused for the output dir.",
    )
    parser.add_argument(
        "--output-base-dir",
        required=False,
        help="Parent directory under which labels_<timestamp>/ is created. "
        "Default: data/processed/synthetic_samples/img_labels",
    )
    parser.add_argument(
        "--substitutions",
        required=False,
        help=(
            "Comma-separated character substitutions in 'from:to' form. "
            f"Default: '{default_subs_str}'"
        ),
    )
    parser.add_argument(
        "--text-field",
        required=False,
        default="original_text",
        help="Which field of each source-labels entry to use as the base "
        "text (default: original_text). Use 'medieval_text' to keep long s "
        "and rotunda r and only apply the additional substitutions on top.",
    )
    parser.add_argument(
        "--logs-dir",
        required=False,
        help="Directory for the run text log. Default: logs/label_correction",
    )

    args = parser.parse_args()

    output_base_dir = (
        Path(args.output_base_dir)
        if args.output_base_dir
        else project_root / "data/processed/synthetic_samples/img_labels"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "label_correction"
    logs_dir.mkdir(parents=True, exist_ok=True)

    substitutions = _parse_substitutions(args.substitutions) if args.substitutions else None

    correct_labels(
        input_json=Path(args.input_json),
        augmented_folder=Path(args.augmented_folder),
        output_base_dir=output_base_dir,
        substitutions=substitutions,
        text_field=args.text_field,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    main()
