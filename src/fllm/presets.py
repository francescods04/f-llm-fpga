"""Named operating presets for the FPGA-vs-GPU cost model.

Each preset is a complete configuration of:
  - quantization (INT4 / INT3 / INT2 cold)
  - KV cache format (BFP8 / BFP4)
  - sparsity (dense / 2:4 / 1:4 on cold experts)
  - cache-aware routing (resident set size + λ)
  - speculative decode (single / tree / cascade)

The presets map directly to the concrete projection table in
`docs/H100_BEAT_PLAN.md` §6.
"""

from __future__ import annotations

from dataclasses import dataclass

from fllm.quant import QuantConfig
from fllm.bfp import BFPConfig
from fllm.sparsity import NMSparsity
from fllm.cache_aware_router import CacheAwareConfig


@dataclass(frozen=True)
class Preset:
    name: str
    quant: QuantConfig
    kv_bfp: BFPConfig
    nm_sparsity: NMSparsity | None
    cache_aware: CacheAwareConfig | None
    speculative_gamma: int = 1           # 1 = disabled
    speculative_tree_depth: int = 1
    description: str = ""


# ---------------------------------------------------------------------------
# Conservative — safest quality, smallest win
# ---------------------------------------------------------------------------
CONSERVATIVE = Preset(
    name="conservative",
    quant=QuantConfig(weight_bits=4, activation_bits=8, weight_group_size=-1),
    kv_bfp=BFPConfig(mantissa_bits=8, block_size=32),
    nm_sparsity=None,
    cache_aware=None,                     # no cache-aware routing
    speculative_gamma=1,
    description=(
        "INT4 global + BFP8 KV + no sparsity + no cache-aware routing. "
        "Minimal quality risk; FPGA win comes from dataflow + HBM efficiency alone."
    ),
)

# ---------------------------------------------------------------------------
# Moderate — the paper claim preset
# ---------------------------------------------------------------------------
MODERATE = Preset(
    name="moderate",
    quant=QuantConfig(weight_bits=4, activation_bits=8, weight_group_size=-1),
    kv_bfp=BFPConfig(mantissa_bits=4, block_size=32),
    nm_sparsity=NMSparsity(n=2, m=4),    # 2:4 on cold experts only (applied selectively)
    cache_aware=CacheAwareConfig(
        num_experts=128,
        top_k=8,
        alpha=2.0,
        lambda_kl=0.1,
        lambda_lb=0.01,
        fine_tune_router=True,
    ),
    speculative_gamma=4,
    speculative_tree_depth=1,
    description=(
        "INT4 global + BFP4 KV + 2:4 sparse on cold experts + cache-aware routing "
        "(75% hit target) + single-draft speculative decode γ=4. "
        "Target: Δppl ≤ +0.7, realistic tok/s ~2500 on 2× VU47P."
    ),
)

# ---------------------------------------------------------------------------
# Aggressive — maximum compression, stretch goal
# ---------------------------------------------------------------------------
AGGRESSIVE = Preset(
    name="aggressive",
    quant=QuantConfig(
        weight_bits=4,
        activation_bits=8,
        weight_group_size=64,
        nm_sparsity=NMSparsity(n=1, m=4),
        use_int2=True,                    # ternary on coldest 50% experts
    ),
    kv_bfp=BFPConfig(mantissa_bits=4, block_size=16),
    nm_sparsity=NMSparsity(n=1, m=4),   # 1:4 on cold experts
    cache_aware=CacheAwareConfig(
        num_experts=128,
        top_k=8,
        alpha=3.0,
        lambda_kl=0.3,
        lambda_lb=0.01,
        fine_tune_router=True,
    ),
    speculative_gamma=4,
    speculative_tree_depth=2,             # tree speculation
    description=(
        "INT4 base + INT2 ternary on coldest 50% experts + 1:4 sparse + BFP4 KV "
        "+ cache-aware routing + tree speculation K=4. "
        "Stretch: Δppl ≤ +1.2, realistic tok/s ~9800 on 2× VU47P."
    ),
)

PRESETS: dict[str, Preset] = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
}


def preset_to_fpga_optimizations(preset: Preset) -> dict:
    """Return a kwargs dict compatible with `cost_model.estimate()`."""
    from fllm.cost_model import FPGAOptimizations
    hit = 0.0 if preset.cache_aware is None else 0.75  # target hit rate
    sparsity = preset.nm_sparsity.saved_byte_factor if preset.nm_sparsity else 1.0
    kv_factor = preset.kv_bfp.mantissa_bits / 8.0 + 1.0 / preset.kv_bfp.block_size
    # Normalise KV factor against FP16 = 2 bytes
    kv_byte_factor = kv_factor / 2.0
    return {
        "fpga_opts": FPGAOptimizations(
            expert_cache_hit_rate=hit,
            kv_byte_factor=kv_byte_factor,
            dataflow_overhead_savings=0.15,
            host_loop_us_per_token=0.0,
            hbm_efficiency=0.75,
            weight_sparsity_factor=sparsity,
        )
    }
