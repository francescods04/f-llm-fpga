"""Build a plain-text training corpus from local text/markdown files."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SUFFIXES = {".txt", ".md"}


def iter_text_files(inputs: list[Path], suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
        elif path.is_dir():
            files.extend(
                child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in suffixes
            )
    return sorted(set(files))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input files or directories.")
    parser.add_argument("--out", default="datasets/corpus.txt")
    parser.add_argument("--suffix", action="append", default=None)
    parser.add_argument("--min-chars", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suffixes = set(args.suffix or DEFAULT_SUFFIXES)
    inputs = [Path(item) for item in args.inputs]
    files = iter_text_files(inputs, suffixes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as out:
        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) < args.min_chars:
                continue
            out.write(f"\n\n<doc path=\"{file_path}\">\n")
            out.write(text)
            out.write("\n</doc>\n")
            written += 1
            chars += len(text)

    print(f"files={written}")
    print(f"chars={chars:,}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()

