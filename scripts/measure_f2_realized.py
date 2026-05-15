#!/usr/bin/env python3
"""Measure realized tok/s, power, and HBM BW on AWS f2 FPGA instances.

Requires:
  - An f2.6xlarge or f2.12xlarge instance with AFI loaded.
  - The FPGA runtime (XRT) and xbutil installed.
  - A compiled host executable that drives the FPGA token loop.

This script wraps the host executable, parses xbutil examine output for
power/temperature, and writes a structured JSON report.

Without a real FPGA this script writes a stub.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_xbutil_power() -> dict:
    """Parse xbutil examine output for power draw."""
    # Placeholder — real implementation would subprocess xbutil and regex
    return {
        "fpga_power_w": None,
        "hbm_temp_c": None,
        "status": "stub",
    }


def run_fpga_benchmark(afi_id: str, prompt_len: int, gen_len: int) -> dict:
    """Run the FPGA host executable and parse tok/s."""
    return {
        "afi_id": afi_id,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "tok_s": None,
        "ms_per_token_p50": None,
        "status": "stub",
        "notes": "Run on a real f2 instance with AFI loaded.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure FPGA realized performance")
    parser.add_argument("--afi", required=True, help="AWS AFI ID")
    parser.add_argument("--instance", default="f2.12xlarge", choices=["f2.6xlarge", "f2.12xlarge"])
    parser.add_argument("--output", default="benchmarks/measured_fpga_baseline.json")
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--gen-len", type=int, default=128)
    args = parser.parse_args()

    print(f"[measure_f2_realized] Target: {args.instance} AFI={args.afi}")
    print("[measure_f2_realized] NOTE: this script is a stub on non-FPGA environments.")

    fpga_result = run_fpga_benchmark(args.afi, args.prompt_len, args.gen_len)
    power = parse_xbutil_power()

    report = {
        "hardware": args.instance,
        "afi_id": args.afi,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fpga": fpga_result,
        "power": power,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[measure_f2_realized] Wrote stub to {out_path}")


if __name__ == "__main__":
    main()
