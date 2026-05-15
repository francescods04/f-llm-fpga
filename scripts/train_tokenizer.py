"""Train a byte-level BPE tokenizer for F-LLM experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from fllm.tokenizer import SPECIAL_TOKENS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="datasets/corpus.txt")
    parser.add_argument("--out", default="datasets/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser


def main() -> None:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise SystemExit("Missing dependency: tokenizers. Install the research extra.") from exc

    args = build_parser().parse_args()
    corpus = Path(args.corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk"]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["bos"],
            SPECIAL_TOKENS["eos"],
            SPECIAL_TOKENS["unk"],
        ],
    )
    tokenizer.train([str(corpus)], trainer)
    tokenizer.save(str(out_path))

    print(f"vocab_size={tokenizer.get_vocab_size()}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()
