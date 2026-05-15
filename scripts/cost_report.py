"""Print FPGA-vs-GPU cost/energy projection for Qwen3.6-35B-A3B."""

from __future__ import annotations

import argparse
import json

from fllm.cost_model import MODELS, FPGAOptimizations, compare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.6-35b-a3b-int4", choices=sorted(MODELS))
    parser.add_argument("--fpga-efficiency", type=float, default=0.75)
    parser.add_argument("--gpu-efficiency", type=float, default=0.55)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--fpga-shard", type=int, default=2,
                        help="number of FPGAs in the shard (default 2 for f2.12xlarge)")
    parser.add_argument("--expert-cache-hit", type=float, default=0.60)
    parser.add_argument("--kv-byte-factor", type=float, default=0.51,
                        help="BFP8 KV bytes / FP16 KV bytes")
    parser.add_argument("--dataflow-savings", type=float, default=0.15)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    opts = FPGAOptimizations(
        expert_cache_hit_rate=args.expert_cache_hit,
        kv_byte_factor=args.kv_byte_factor,
        dataflow_overhead_savings=args.dataflow_savings,
    )
    results = compare(
        args.model,
        fpga_efficiency=args.fpga_efficiency,
        gpu_efficiency=args.gpu_efficiency,
        avg_seq_len=args.seq_len,
        fpga_opts=opts,
        fpga_shard_devices=args.fpga_shard,
    )
    payload = [r.__dict__ for r in results]
    if args.format == "json" or args.json_out:
        encoded = json.dumps(payload, indent=2)
        print(encoded)
        if args.json_out:
            from pathlib import Path
            Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")
        return

    header = f"{'hardware':40} {'fits':>5} {'tok/s':>9} {'ms/tok':>8} {'J/tok':>8} {'$/1M':>10} {'B/tok':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.hardware:40} {('y' if r.fits_in_memory else 'n'):>5} "
            f"{r.realistic_tokens_per_s:9.1f} {r.ms_per_token:8.2f} "
            f"{r.joule_per_token:8.3f} {r.usd_per_million_tokens:10.2f} "
            f"{r.bytes_per_token/1e6:10.2f}"
        )
    print()
    for r in results:
        if r.notes:
            print(f"note [{r.hardware}]: {r.notes}")


if __name__ == "__main__":
    main()
