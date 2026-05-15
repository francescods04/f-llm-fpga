"""Download a TinyStories text corpus through Hugging Face datasets.

This script is intentionally optional. It requires network access and the
`datasets` package, so it is not part of local smoke tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--out", default="datasets/tinystories.txt")
    parser.add_argument("--max-examples", type=int, default=100_000)
    return parser


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install the research extra.") from exc

    args = build_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    count = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in dataset:
            text = str(row.get(args.text_field, "")).strip()
            if not text:
                continue
            out.write(text)
            out.write("\n\n")
            count += 1
            chars += len(text)
            if count >= args.max_examples:
                break

    print(f"examples={count:,}")
    print(f"chars={chars:,}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()

