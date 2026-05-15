"""Per-kernel cycle estimator for the FPGA decode datapath.

Replaces the HBM-only roofline in cost_model.py with a structural model:
each kernel has a parametric cycle count under its planned parallelism, then
the schedule sums them up over one decoded token. This is the simulator the
Talos V2 comparison doc asked for.

All counts assume the time-multiplexed matvec tile (LANES, TILE_ROWS) is the
single primitive replicated `num_tiles` times in parallel (one per HBM
channel cluster). Per-block schedule for a Qwen3-A3B layer:

    rms_norm(hidden)
    qkv_proj: matvec(hidden -> q_dim + 2*kv_dim)
    rope_apply(q, k)
    kv_pack (BFP8)
    attention: q*k^T over ctx + softmax + p*v
    o_proj: matvec(hidden -> hidden)
    rms_norm(hidden)
    router: matvec(hidden -> num_experts)
    moe_dispatch + active_experts * (up + gate + down)
    moe_combine

LM head runs once per token, not per layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileSpec:
    lanes: int = 32        # SIMD lanes inside a row (INT4 packed)
    tile_rows: int = 16    # output rows in parallel
    num_tiles: int = 4     # replicas (one per HBM channel cluster)
    fmax_mhz: float = 600.0


@dataclass(frozen=True)
class ModelShape:
    hidden: int = 4096
    num_layers: int = 48
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    num_experts: int = 128
    active_experts: int = 8
    moe_inner: int = 1408
    vocab: int = 152064
    avg_ctx: int = 1024


def matvec_cycles(out_features: int, in_features: int, tile: TileSpec) -> int:
    """One matvec on the tile bank. II=1 pipelined."""
    parallel_rows = tile.tile_rows * tile.num_tiles
    rows = (out_features + parallel_rows - 1) // parallel_rows
    beats_per_row = (in_features + tile.lanes - 1) // tile.lanes
    pipeline_fill = beats_per_row + tile.tile_rows  # warm-up
    return rows * beats_per_row + pipeline_fill


def attention_cycles(shape: ModelShape, tile: TileSpec) -> int:
    """Q*K^T scan + softmax + P*V on the same tile bank.

    For batch=1 decode the Q is one row, K and V are ctx rows. Two matvec-like
    scans plus a softmax pass.
    """
    ctx = shape.avg_ctx
    head_dim = shape.head_dim
    qk = matvec_cycles(ctx, head_dim, tile) * shape.num_heads
    # softmax: 2-pass (max-track folded with QK -> only one extra exp+sum pass)
    softmax = ctx + 8  # LUT exp pipelined, +8 cycles divider tail
    pv = matvec_cycles(head_dim, ctx, tile) * shape.num_heads
    return qk + softmax + pv


def moe_cycles(shape: ModelShape, tile: TileSpec) -> int:
    router = matvec_cycles(shape.num_experts, shape.hidden, tile)
    topk = 8  # bitonic stage latency
    per_expert = (
        matvec_cycles(shape.moe_inner, shape.hidden, tile)        # up
        + matvec_cycles(shape.moe_inner, shape.hidden, tile)      # gate
        + matvec_cycles(shape.hidden, shape.moe_inner, tile)      # down
    )
    return router + topk + shape.active_experts * per_expert


def block_cycles(shape: ModelShape, tile: TileSpec) -> dict[str, int]:
    qkv_dim = shape.hidden + 2 * (shape.num_kv_heads * shape.head_dim)
    return {
        "rms_in": shape.hidden // 4,                              # streaming reduce
        "qkv_proj": matvec_cycles(qkv_dim, shape.hidden, tile),
        "rope": shape.num_heads * shape.head_dim // tile.lanes,
        "kv_pack": shape.num_kv_heads * shape.head_dim // tile.lanes,
        "attention": attention_cycles(shape, tile),
        "o_proj": matvec_cycles(shape.hidden, shape.hidden, tile),
        "rms_mid": shape.hidden // 4,
        "moe": moe_cycles(shape, tile),
    }


def lm_head_cycles(shape: ModelShape, tile: TileSpec) -> int:
    return matvec_cycles(shape.vocab, shape.hidden, tile)


@dataclass(frozen=True)
class CycleReport:
    cycles_per_token: int
    ms_per_token: float
    tokens_per_s: float
    breakdown_per_block: dict[str, int]
    block_cycles: int
    lm_head: int


def estimate_token(shape: ModelShape, tile: TileSpec) -> CycleReport:
    per_block = block_cycles(shape, tile)
    block_total = sum(per_block.values())
    lm = lm_head_cycles(shape, tile)
    total = block_total * shape.num_layers + lm
    ms = total / (tile.fmax_mhz * 1e3)
    tps = 1000.0 / ms if ms > 0 else 0.0
    return CycleReport(
        cycles_per_token=total,
        ms_per_token=ms,
        tokens_per_s=tps,
        breakdown_per_block=per_block,
        block_cycles=block_total,
        lm_head=lm,
    )
