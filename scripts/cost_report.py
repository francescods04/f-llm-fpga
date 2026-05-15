"""Print FPGA-vs-GPU cost/energy projection for Qwen3.6-35B-A3B."""

from __future__ import annotations

import argparse
import json

from fllm.cost_model import MODELS, compare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.6-35b-a3b-int4", choices=sorted(MODELS))
    parser.add_argument("--fpga-efficiency", type=float, default=0.70)
    parser.add_argument("--gpu-efficiency", type=float, default=0.55)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = compare(
        args.model,
        fpga_efficiency=args.fpga_efficiency,
        gpu_efficiency=args.gpu_efficiency,
        avg_seq_len=args.seq_len,
    )
    payload = [r.__dict__ for r in results]
    if args.format == "json" or args.json_out:
        encoded = json.dumps(payload, indent=2)
        print(encoded)
        if args.json_out:
            from pathlib import Path
            Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")
        return

    header = f"{'hardware':28} {'fits':>5} {'tok/s':>9} {'ms/tok':>8} {'J/tok':>8} {'$/1M':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.hardware:28} {('y' if r.fits_in_memory else 'n'):>5} "
            f"{r.realistic_tokens_per_s:9.1f} {r.ms_per_token:8.2f} "
            f"{r.joule_per_token:8.3f} {r.usd_per_million_tokens:10.2f}"
        )
    print()
    for r in results:
        if r.notes:
            print(f"note [{r.hardware}]: {r.notes}")


if __name__ == "__main__":
    main()
