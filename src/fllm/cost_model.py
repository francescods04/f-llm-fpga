"""Roofline cost/energy model for FPGA-vs-GPU decode at batch=1.

Purpose: decide whether the full-FPGA path can beat GPU on $/1M tokens and
joule/token *before* writing RTL. All numbers below are upper-bound peak specs
from public datasheets and cloud price sheets. Real implementations land below
peak; the model returns "theoretical-best" plus an efficiency knob to project
realistic ranges.

Targets considered:

- AWS f2.6xlarge / f2.12xlarge: AMD/Xilinx Virtex UltraScale+ VU47P (HBM2 16GB,
  ~460 GB/s, ~9k DSPs). Used as the FPGA target.
- AWS g6.xlarge:   NVIDIA L4 (FP16 ~120 TFLOPS, 300 GB/s, 24 GB).
- AWS g6e.xlarge:  NVIDIA L40S (FP16 ~362 TFLOPS, 864 GB/s, 48 GB).
- AWS p5.48xlarge: 8x NVIDIA H100 SXM (per-GPU HBM3 ~3.35 TB/s, 80 GB).

For batch=1 decode the limiter is *weight-bytes-streamed-per-token*. So:

    ms/token  >=  active_weight_bytes / mem_bandwidth
    tokens/s  <=  mem_bandwidth / active_weight_bytes

This is the ceiling. Real systems hit a fraction of this due to kernel launch,
KV cache traffic, attention compute, and host overhead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    hbm_bandwidth_gbs: float       # GB/s
    hbm_capacity_gb: float
    peak_tops_int8: float          # TOPS at INT8 (per device)
    tdp_watts: float
    aws_hourly_usd: float          # on-demand pricing per device


@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_params_b: float
    active_params_b: float         # MoE: params actually streamed per token
    weight_bits: int               # 4 = INT4 weights
    kv_bytes_per_token: float      # bytes of KV cache appended per token


HARDWARES: dict[str, Hardware] = {
    "fpga_vu47p":  Hardware("AWS f2 VU47P (1 FPGA)", 460,  16,  90,  225, 2.10),
    "gpu_l4":      Hardware("AWS g6 L4",             300,  24, 242,   72, 1.00),
    "gpu_l40s":    Hardware("AWS g6e L40S",          864,  48, 733,  350, 2.30),
    "gpu_h100":    Hardware("AWS p5 H100 SXM",      3350,  80, 1979, 700, 12.30 / 8),
}


MODELS: dict[str, ModelSpec] = {
    "qwen3.6-35b-a3b-int4": ModelSpec(
        name="Qwen3.6-35B-A3B INT4",
        total_params_b=35.0,
        active_params_b=3.0,
        weight_bits=4,
        kv_bytes_per_token=4 * 2 * 48 * 8 * 128,
    ),
    "qwen3.6-35b-a3b-fp16": ModelSpec(
        name="Qwen3.6-35B-A3B FP16",
        total_params_b=35.0,
        active_params_b=3.0,
        weight_bits=16,
        kv_bytes_per_token=4 * 2 * 48 * 8 * 128,
    ),
}


@dataclass(frozen=True)
class FPGAOptimizations:
    """Stackable FPGA-only wins. Each is a fraction in [0, 1] applied to a
    relevant traffic term, not the final tok/s. See docs/FPGA_OPTIMIZATIONS.md
    for the engineering justification of every default value.
    """
    expert_cache_hit_rate: float = 0.60   # fraction of expert reads served on-chip
    kv_byte_factor: float = 0.51          # BFP8 KV vs FP16 KV
    dataflow_overhead_savings: float = 0.15  # fraction of HBM round-trips removed
    host_loop_us_per_token: float = 0.0   # FPGA hardware token loop = 0us host overhead
    hbm_efficiency: float = 0.75          # channel-aware layout vs ~0.55 on GPU
    weight_sparsity_factor: float = 1.0   # N:M sparsity bytes-read factor (1.0 = dense)
    cross_fpga_dispatch_us: float = 0.0    # per-token peer-FPGA latency (µs), Gate G9
    speculative_effective_factor: float = 1.0  # multiply tok/s by this (1.0 = disabled)


@dataclass(frozen=True)
class CostResult:
    hardware: str
    model: str
    fits_in_memory: bool
    peak_tokens_per_s: float
    realistic_tokens_per_s: float
    ms_per_token: float
    joule_per_token: float
    usd_per_million_tokens: float
    bytes_per_token: float
    notes: str


def active_weight_bytes(model: ModelSpec) -> float:
    return model.active_params_b * 1e9 * model.weight_bits / 8.0


def total_weight_bytes(model: ModelSpec) -> float:
    return model.total_params_b * 1e9 * model.weight_bits / 8.0


def estimate(
    hw: Hardware,
    model: ModelSpec,
    *,
    efficiency: float = 0.6,
    avg_seq_len: int = 1024,
    fpga_opts: FPGAOptimizations | None = None,
    is_fpga: bool = False,
    num_devices: int = 1,
) -> CostResult:
    bw_bytes = hw.hbm_bandwidth_gbs * 1e9 * num_devices
    weight_b = active_weight_bytes(model)
    kv_b = model.kv_bytes_per_token * avg_seq_len

    if is_fpga and fpga_opts is not None:
        weight_b = weight_b * (1.0 - fpga_opts.expert_cache_hit_rate) * fpga_opts.weight_sparsity_factor
        kv_b = kv_b * fpga_opts.kv_byte_factor
        bytes_per_tok = (weight_b + kv_b) * (1.0 - fpga_opts.dataflow_overhead_savings)
        eff = max(efficiency, fpga_opts.hbm_efficiency)
        host_us = fpga_opts.host_loop_us_per_token + fpga_opts.cross_fpga_dispatch_us
    else:
        bytes_per_tok = weight_b + kv_b
        eff = efficiency
        host_us = 60.0  # typical CUDA stream cycle + sampling roundtrip at batch=1

    peak_tps = bw_bytes / bytes_per_tok if bytes_per_tok > 0 else 0.0
    real_tps_no_host = peak_tps * eff
    ms_no_host = 1000.0 / real_tps_no_host if real_tps_no_host > 0 else float("inf")
    ms = ms_no_host + host_us / 1000.0
    real_tps = 1000.0 / ms if ms > 0 else 0.0
    if is_fpga and fpga_opts is not None:
        real_tps *= fpga_opts.speculative_effective_factor
    total_w = hw.tdp_watts * num_devices
    joules = total_w / real_tps if real_tps > 0 else float("inf")
    hourly = hw.aws_hourly_usd * num_devices
    usd_per_1m = (hourly / 3600.0) / real_tps * 1e6 if real_tps > 0 else float("inf")
    fits = total_weight_bytes(model) <= hw.hbm_capacity_gb * 1e9 * num_devices
    note = ""
    if not fits:
        note = (
            f"model weights {total_weight_bytes(model)/1e9:.1f}GB "
            f"exceed device HBM {hw.hbm_capacity_gb*num_devices}GB across {num_devices} device(s)"
        )
    return CostResult(
        hardware=hw.name + (f" x{num_devices}" if num_devices > 1 else ""),
        model=model.name,
        fits_in_memory=fits,
        peak_tokens_per_s=peak_tps,
        realistic_tokens_per_s=real_tps,
        ms_per_token=ms,
        joule_per_token=joules,
        usd_per_million_tokens=usd_per_1m,
        bytes_per_token=bytes_per_tok,
        notes=note,
    )


def compare(
    model_key: str,
    *,
    fpga_efficiency: float = 0.70,
    gpu_efficiency: float = 0.55,
    avg_seq_len: int = 1024,
    fpga_opts: FPGAOptimizations | None = None,
    fpga_shard_devices: int = 2,
) -> list[CostResult]:
    """Compare a model across all hardware targets.

    For FPGA paths, runs both:
      - "naive port": GPU-style traffic with `gpu_efficiency`
      - "FPGA-tuned": applies `fpga_opts` (expert cache, BFP KV, dataflow, host)
    The naive number lets us read the *delta* attributable to FPGA-specific work.
    """
    model = MODELS[model_key]
    opts = fpga_opts or FPGAOptimizations()
    results: list[CostResult] = []
    for hw_key, hw in HARDWARES.items():
        if hw_key.startswith("fpga"):
            naive = estimate(
                hw, model, efficiency=gpu_efficiency, avg_seq_len=avg_seq_len,
                is_fpga=False, num_devices=fpga_shard_devices,
            )
            results.append(
                CostResult(
                    hardware=naive.hardware + " [naive port]",
                    model=naive.model, fits_in_memory=naive.fits_in_memory,
                    peak_tokens_per_s=naive.peak_tokens_per_s,
                    realistic_tokens_per_s=naive.realistic_tokens_per_s,
                    ms_per_token=naive.ms_per_token,
                    joule_per_token=naive.joule_per_token,
                    usd_per_million_tokens=naive.usd_per_million_tokens,
                    bytes_per_token=naive.bytes_per_token,
                    notes=naive.notes,
                )
            )
            tuned = estimate(
                hw, model, efficiency=fpga_efficiency, avg_seq_len=avg_seq_len,
                is_fpga=True, fpga_opts=opts, num_devices=fpga_shard_devices,
            )
            results.append(
                CostResult(
                    hardware=tuned.hardware + " [tuned]",
                    model=tuned.model, fits_in_memory=tuned.fits_in_memory,
                    peak_tokens_per_s=tuned.peak_tokens_per_s,
                    realistic_tokens_per_s=tuned.realistic_tokens_per_s,
                    ms_per_token=tuned.ms_per_token,
                    joule_per_token=tuned.joule_per_token,
                    usd_per_million_tokens=tuned.usd_per_million_tokens,
                    bytes_per_token=tuned.bytes_per_token,
                    notes=tuned.notes,
                )
            )
        else:
            results.append(estimate(
                hw, model, efficiency=gpu_efficiency, avg_seq_len=avg_seq_len,
                is_fpga=False,
            ))
    return results
