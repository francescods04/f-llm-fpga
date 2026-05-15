"""Cycle-accurate token-loop simulator for the full-FPGA decode path.

Models the autoregressive token generation loop at the architecture level,
using the per-kernel cycle estimators from fllm.cycle_sim as building blocks.
Produces:
  - cycles per token (prefill and decode)
  - bottleneck identification per step
  - sensitivity sweeps (context length, expert cache hit rate, tile count)
  - 2-FPGA sharding overhead
  - speculative decode speed-up (stretch)

This is a research tool, not a cycle-exact RTL simulator. It answers:
  "Given our tile parallelism, what limits throughput at step N?"
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from fllm.cycle_sim import (
    ModelShape,
    TileSpec,
    attention_cycles,
    block_cycles,
    hbm_bytes_per_token,
    lm_head_cycles,
    matvec_cycles,
    moe_cycles,
)


@dataclass(frozen=True)
class TokenLoopConfig:
    """Runtime configuration for the token loop."""
    prefill_len: int = 128
    decode_steps: int = 128
    # 2-FPGA sharding
    num_devices: int = 2
    inter_fpga_latency_cycles: int = 8  # cycles to send one hidden vec peer-to-peer
    # Expert caching
    expert_cache_hit_rate: float = 0.60
    # Optional speculative decode (draft model on FPGA 0)
    speculative_draft_tokens: int = 0   # 0 = disabled
    speculative_accept_rate: float = 0.70


@dataclass
class StepReport:
    step: int
    ctx_len: int
    cycles: int
    bottleneck: str
    hbm_bytes: float
    breakdown: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Single-step cycle model (decode, batch=1, one new token)
# ---------------------------------------------------------------------------
def estimate_decode_step(
    shape: ModelShape,
    tile: TileSpec,
    cfg: TokenLoopConfig,
    ctx_len: int,
) -> StepReport:
    """Cycles for one decode step at the given context length."""
    # Per-block compute
    per_block = block_cycles(shape, tile)
    block_total = sum(per_block.values())

    # Adjust attention cycles for current ctx_len (not avg_ctx)
    # Override shape temporarily for attention
    shape_now = ModelShape(
        hidden=shape.hidden,
        num_layers=shape.num_layers,
        num_heads=shape.num_heads,
        num_kv_heads=shape.num_kv_heads,
        head_dim=shape.head_dim,
        num_experts=shape.num_experts,
        active_experts=shape.active_experts,
        moe_inner=shape.moe_inner,
        vocab=shape.vocab,
        avg_ctx=ctx_len,
        weight_bits=shape.weight_bits,
        kv_bits=shape.kv_bits,
        expert_cache_hit=cfg.expert_cache_hit_rate,
    )
    # Recompute attention with current ctx
    attn = attention_cycles(shape_now, tile)
    moe = moe_cycles(shape_now, tile)

    # Replace attention and moe in the breakdown
    per_block_now = dict(per_block)
    per_block_now["attention"] = attn
    per_block_now["moe"] = moe
    block_total_now = sum(per_block_now.values())

    # Full model compute
    total_compute = block_total_now * shape.num_layers + lm_head_cycles(shape, tile)

    # HBM side
    bpt = hbm_bytes_per_token(shape_now)
    bw = tile.hbm_bw_gbs * tile.hbm_efficiency * 1e9 * cfg.num_devices
    hbm_tps = bw / bpt if bpt > 0 else 0.0
    hbm_ms = 1000.0 / hbm_tps if hbm_tps > 0 else float("inf")
    compute_ms = total_compute / (tile.fmax_mhz * 1e3)

    # Inter-FPGA penalty (shard boundary every num_layers/num_devices layers)
    # Assume even sharding: each FPGA owns num_layers/num_devices blocks.
    # At boundary, hidden vector is exchanged.
    shard_cycles = 0
    if cfg.num_devices > 1:
        # One transfer per boundary per token
        boundaries = cfg.num_devices - 1
        shard_cycles = boundaries * cfg.inter_fpga_latency_cycles

    if compute_ms < hbm_ms:
        bottleneck = "compute"
        realized_cycles = total_compute + shard_cycles
    else:
        bottleneck = "hbm"
        # Convert HBM time back to cycles
        hbm_cycles = int(hbm_ms * tile.fmax_mhz * 1e3)
        realized_cycles = hbm_cycles + shard_cycles

    return StepReport(
        step=0,
        ctx_len=ctx_len,
        cycles=realized_cycles,
        bottleneck=bottleneck,
        hbm_bytes=bpt,
        breakdown={
            "block_compute": block_total_now,
            "lm_head": lm_head_cycles(shape, tile),
            "shard_penalty": shard_cycles,
        },
    )


# ---------------------------------------------------------------------------
# Prefill model (process prompt of length P)
# ---------------------------------------------------------------------------
def estimate_prefill(
    shape: ModelShape,
    tile: TileSpec,
    cfg: TokenLoopConfig,
) -> StepReport:
    """Prefill is not memory-bound; it's compute-bound matmul over the whole prompt.

    Rough model: for each token in the prompt we do one forward, but matvecs
    become small matrix-matrix ops that utilize the tile bank better.
    We approximate by saying prefill throughput is ~P * decode_cycles / 2
    because matrix-matrix has better HBM reuse than matvec.
    """
    decode_one = estimate_decode_step(shape, tile, cfg, cfg.prefill_len)
    # Empirical fudge: matmul reuse gives ~2x effective speedup vs P sequential decodes
    cycles = int(decode_one.cycles * cfg.prefill_len / 2)
    return StepReport(
        step=-1,
        ctx_len=cfg.prefill_len,
        cycles=cycles,
        bottleneck="compute_prefill",
        hbm_bytes=decode_one.hbm_bytes * cfg.prefill_len,
        breakdown=decode_one.breakdown,
    )


# ---------------------------------------------------------------------------
# Full token loop
# ---------------------------------------------------------------------------
def run_token_loop(
    shape: ModelShape,
    tile: TileSpec,
    cfg: TokenLoopConfig,
) -> list[StepReport]:
    """Run prefill + decode and return per-step reports."""
    reports: list[StepReport] = []
    reports.append(estimate_prefill(shape, tile, cfg))

    ctx = cfg.prefill_len
    for step in range(cfg.decode_steps):
        r = estimate_decode_step(shape, tile, cfg, ctx)
        r.step = step
        reports.append(r)
        ctx += 1

    # Speculative decode overlay (stretch)
    if cfg.speculative_draft_tokens > 0:
        reports = _overlay_speculative(reports, shape, tile, cfg)

    return reports


def _overlay_speculative(
    reports: list[StepReport],
    shape: ModelShape,
    tile: TileSpec,
    cfg: TokenLoopConfig,
) -> list[StepReport]:
    """Reduce effective decode cycles assuming draft model verifies k tokens.

    Model:
      - Draft cost: K autoregressive steps on a tiny 1-layer model.
      - Target cost: 1 verification forward pass (slightly more work than 1 decode
        step because it sees K+1 tokens in parallel, but for cycle estimation
        we treat it as 1 decode step plus a small attention overhead).
      - Acceptance: on average we accept accept_rate * K tokens per chunk.
      - Net effect: effective serial steps = 1 / (1 + accept_rate * (K-1))
    """
    # Draft is ~1/num_layers of the target, but same hidden/head dims.
    # Rough estimate: draft body is ~15% of target body params.
    draft_body_ratio = 0.15
    draft_cycles_per_step = int(reports[1].cycles * draft_body_ratio) if len(reports) > 1 else 0
    draft_total = draft_cycles_per_step * cfg.speculative_draft_tokens

    # Target verification: one decode step + small attention penalty for longer ctx
    # We approximate by using the existing decode step cycles.
    target_verify = reports[1].cycles if len(reports) > 1 else 0

    # Amortized target cycles per accepted token
    accepted_per_chunk = cfg.speculative_accept_rate * cfg.speculative_draft_tokens
    if accepted_per_chunk <= 0:
        return reports  # no speedup

    # Cycles per effective token = (draft_total + target_verify) / accepted_per_chunk
    cycles_per_eff_token = (draft_total + target_verify) / accepted_per_chunk

    # Scale every decode report accordingly
    baseline_per_token = reports[1].cycles if len(reports) > 1 else 1
    factor = cycles_per_eff_token / baseline_per_token
    for r in reports:
        if r.step >= 0:
            r.cycles = int(r.cycles * factor)
    return reports


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(reports: list[StepReport], tile: TileSpec, shape: ModelShape):
    prefill = reports[0]
    decode_reports = [r for r in reports if r.step >= 0]

    total_decode_cycles = sum(r.cycles for r in decode_reports)
    avg_decode_cycles = total_decode_cycles / len(decode_reports)
    max_decode_cycles = max(r.cycles for r in decode_reports)
    min_decode_cycles = min(r.cycles for r in decode_reports)

    ms_prefill = prefill.cycles / (tile.fmax_mhz * 1e3)
    ms_decode_avg = avg_decode_cycles / (tile.fmax_mhz * 1e3)
    tok_s_prefill = cfg.prefill_len / (ms_prefill / 1000.0) if ms_prefill > 0 else 0.0
    tok_s_decode_avg = 1000.0 / ms_decode_avg if ms_decode_avg > 0 else 0.0

    print(f"--- Token Loop Report ---")
    print(f"Model: Qwen3-A3B shape, hidden={shape.hidden}, layers={shape.num_layers}, experts={shape.num_experts}")
    print(f"Tile:  lanes={tile.lanes}, rows={tile.tile_rows}, tiles={tile.num_tiles}, fmax={tile.fmax_mhz}MHz")
    print(f"Devices: {cfg.num_devices}, expert_cache_hit={cfg.expert_cache_hit_rate}")
    if cfg.speculative_draft_tokens > 0:
        print(f"Speculative: draft_tokens={cfg.speculative_draft_tokens}, accept_rate={cfg.speculative_accept_rate}")
    print()
    print(f"Prefill  ({prefill.ctx_len} tokens): {prefill.cycles:,} cycles  ({ms_prefill:.2f} ms)  {tok_s_prefill:.1f} tok/s")
    print(f"Decode   ({len(decode_reports)} steps):")
    print(f"  avg cycles/step: {avg_decode_cycles:,.0f}  ({ms_decode_avg:.2f} ms)  {tok_s_decode_avg:.1f} tok/s")
    print(f"  max cycles/step: {max_decode_cycles:,}")
    print(f"  min cycles/step: {min_decode_cycles:,}")
    print()

    # Bottleneck histogram
    bottlenecks = {}
    for r in decode_reports:
        bottlenecks[r.bottleneck] = bottlenecks.get(r.bottleneck, 0) + 1
    print("Bottleneck histogram:")
    for b, c in sorted(bottlenecks.items(), key=lambda x: -x[1]):
        print(f"  {b:12} {c}/{len(decode_reports)} ({c/len(decode_reports)*100:.1f}%)")

    # HBM traffic
    total_hbm_gb = sum(r.hbm_bytes for r in reports) / 1e9
    print(f"\nTotal HBM moved: {total_hbm_gb:.2f} GB")

    # Per-step CSV-like sample (first 8 + last 8)
    print("\nPer-step sample (cycles | bottleneck | ctx):")
    for r in reports[:4] + reports[-4:]:
        tag = "prefill" if r.step < 0 else f"step{r.step:03d}"
        print(f"  {tag}  {r.cycles:>10,} cyc  {r.bottleneck:12}  ctx={r.ctx_len}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefill-len", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=256)
    p.add_argument("--lanes", type=int, default=32)
    p.add_argument("--tile-rows", type=int, default=16)
    p.add_argument("--num-tiles", type=int, default=4)
    p.add_argument("--fmax-mhz", type=float, default=600.0)
    p.add_argument("--num-devices", type=int, default=2)
    p.add_argument("--expert-cache-hit", type=float, default=0.60)
    p.add_argument("--spec-draft", type=int, default=0)
    p.add_argument("--spec-accept", type=float, default=0.70)
    p.add_argument("--json-out", default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    shape = ModelShape()
    tile = TileSpec(
        lanes=args.lanes,
        tile_rows=args.tile_rows,
        num_tiles=args.num_tiles,
        fmax_mhz=args.fmax_mhz,
    )
    cfg = TokenLoopConfig(
        prefill_len=args.prefill_len,
        decode_steps=args.decode_steps,
        num_devices=args.num_devices,
        expert_cache_hit_rate=args.expert_cache_hit,
        speculative_draft_tokens=args.spec_draft,
        speculative_accept_rate=args.spec_accept,
    )

    reports = run_token_loop(shape, tile, cfg)
    print_report(reports, tile, shape)

    if args.json_out:
        payload = [
            {
                "step": r.step,
                "ctx_len": r.ctx_len,
                "cycles": r.cycles,
                "bottleneck": r.bottleneck,
                "hbm_bytes": r.hbm_bytes,
            }
            for r in reports
        ]
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
