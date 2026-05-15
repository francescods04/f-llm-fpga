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
    hbm_bw_gbs: float = 460.0     # per-FPGA HBM bandwidth (VU47P HBM2)
    hbm_efficiency: float = 0.75  # realized fraction with channel-aware layout
    num_devices: int = 2          # shard count for 35B model


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
    weight_bits: int = 4
    kv_bits: int = 8           # BFP8 mantissa width
    expert_cache_hit: float = 0.60
    use_sparse_skip: bool = False   # B4: skip zero rows in matvec
    sparse_density: float = 1.0     # fraction of non-zero rows
    cross_fpga_dispatch_us: float = 0.0  # C4: peer latency
    speculative_effective_factor: float = 1.0  # B6: speed-up from spec decode


def matvec_cycles(out_features: int, in_features: int, tile: TileSpec, density: float = 1.0) -> int:
    """One matvec on the tile bank. II=1 pipelined.

    If density < 1.0 (structured sparsity), the row count is reduced
    proportionally because zero rows are skipped by the controller.
    """
    parallel_rows = tile.tile_rows * tile.num_tiles
    active_rows = int((out_features * density + parallel_rows - 1) // parallel_rows * parallel_rows)
    rows = (active_rows + parallel_rows - 1) // parallel_rows
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
    density = shape.sparse_density if shape.use_sparse_skip else 1.0
    router = matvec_cycles(shape.num_experts, shape.hidden, tile, density)
    topk = 8  # bitonic stage latency
    per_expert = (
        matvec_cycles(shape.moe_inner, shape.hidden, tile, density)        # up
        + matvec_cycles(shape.moe_inner, shape.hidden, tile, density)      # gate
        + matvec_cycles(shape.hidden, shape.moe_inner, tile, density)      # down
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


def hbm_bytes_per_token(shape: ModelShape) -> float:
    """Bytes streamed from HBM per generated token (active path only).

    Three streams: active weights (after expert cache hit), KV read of the
    current context, KV write of the new step. Shared (non-expert) weights
    count fully; expert weights count only the cache-miss fraction.
    """
    # shared weights per layer: QKV proj + O proj + router (no MLP at all
    # because MoE replaces it). Cheap relative to experts.
    q_dim = shape.hidden * shape.hidden
    kv_dim = shape.num_kv_heads * shape.head_dim * shape.hidden
    shared_params = (
        q_dim + 2 * kv_dim                  # QKV proj
        + shape.hidden * shape.hidden       # O proj
        + shape.hidden * shape.num_experts  # router
    ) * shape.num_layers + shape.hidden * shape.vocab  # lm head
    expert_params_per_token = (
        shape.active_experts * shape.num_layers
        * 3 * shape.moe_inner * shape.hidden  # up, gate, down
    )
    weight_bytes = (
        shared_params
        + expert_params_per_token * (1.0 - shape.expert_cache_hit)
    ) * (shape.weight_bits / 8.0)

    kv_per_layer = 2 * shape.num_kv_heads * shape.head_dim * (shape.kv_bits / 8.0)
    kv_read = kv_per_layer * shape.num_layers * shape.avg_ctx
    kv_write = kv_per_layer * shape.num_layers
    return weight_bytes + kv_read + kv_write


def cross_fpga_ms(shape: ModelShape) -> float:
    """Extra ms/token from peer-FPGA dispatch (Gate G9)."""
    return shape.cross_fpga_dispatch_us / 1000.0


@dataclass(frozen=True)
class CycleReport:
    cycles_per_token: int
    ms_per_token: float
    tokens_per_s: float
    breakdown_per_block: dict[str, int]
    block_cycles: int
    lm_head: int
    # HBM side
    bytes_per_token: float
    hbm_tokens_per_s: float
    hbm_ms_per_token: float
    # Realized = min(compute, hbm)
    bottleneck: str


def estimate_token(shape: ModelShape, tile: TileSpec) -> CycleReport:
    per_block = block_cycles(shape, tile)
    block_total = sum(per_block.values())
    lm = lm_head_cycles(shape, tile)
    total = block_total * shape.num_layers + lm
    compute_ms = total / (tile.fmax_mhz * 1e3)
    compute_tps = 1000.0 / compute_ms if compute_ms > 0 else 0.0

    bpt = hbm_bytes_per_token(shape)
    bw = tile.hbm_bw_gbs * tile.hbm_efficiency * 1e9 * tile.num_devices
    hbm_tps = bw / bpt if bpt > 0 else 0.0
    hbm_ms = 1000.0 / hbm_tps if hbm_tps > 0 else float("inf")

    if compute_tps < hbm_tps:
        bottleneck = "compute"
        realized_tps = compute_tps
        realized_ms = compute_ms
    else:
        bottleneck = "hbm"
        realized_tps = hbm_tps
        realized_ms = hbm_ms

    # Apply cross-FPGA dispatch latency (C4)
    realized_ms += cross_fpga_ms(shape)
    realized_tps = 1000.0 / realized_ms if realized_ms > 0 else 0.0

    # Apply speculative decode speed-up (B6)
    realized_tps *= shape.speculative_effective_factor
    realized_ms = 1000.0 / realized_tps if realized_tps > 0 else float("inf")

    return CycleReport(
        cycles_per_token=total,
        ms_per_token=realized_ms,
        tokens_per_s=realized_tps,
        breakdown_per_block=per_block,
        block_cycles=block_total,
        lm_head=lm,
        bytes_per_token=bpt,
        hbm_tokens_per_s=hbm_tps,
        hbm_ms_per_token=hbm_ms,
        bottleneck=bottleneck,
    )
