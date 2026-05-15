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
class CostResult:
    hardware: str
    model: str
    fits_in_memory: bool
    peak_tokens_per_s: float
    realistic_tokens_per_s: float
    ms_per_token: float
    joule_per_token: float
    usd_per_million_tokens: float
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
) -> CostResult:
    bw_bytes = hw.hbm_bandwidth_gbs * 1e9
    weight_b = active_weight_bytes(model)
    kv_b = model.kv_bytes_per_token * avg_seq_len
    bytes_per_tok = weight_b + kv_b
    peak_tps = bw_bytes / bytes_per_tok
    real_tps = peak_tps * efficiency
    ms = 1000.0 / real_tps if real_tps > 0 else float("inf")
    joules = hw.tdp_watts / real_tps if real_tps > 0 else float("inf")
    usd_per_1m = (hw.aws_hourly_usd / 3600.0) / real_tps * 1e6 if real_tps > 0 else float("inf")
    fits = total_weight_bytes(model) <= hw.hbm_capacity_gb * 1e9
    note = ""
    if not fits:
        note = (
            f"model weights {total_weight_bytes(model)/1e9:.1f}GB "
            f"exceed device HBM {hw.hbm_capacity_gb}GB; would need multi-device shard"
        )
    return CostResult(
        hardware=hw.name,
        model=model.name,
        fits_in_memory=fits,
        peak_tokens_per_s=peak_tps,
        realistic_tokens_per_s=real_tps,
        ms_per_token=ms,
        joule_per_token=joules,
        usd_per_million_tokens=usd_per_1m,
        notes=note,
    )


def compare(
    model_key: str,
    *,
    fpga_efficiency: float = 0.70,
    gpu_efficiency: float = 0.55,
    avg_seq_len: int = 1024,
) -> list[CostResult]:
    model = MODELS[model_key]
    results = []
    for hw_key, hw in HARDWARES.items():
        eff = fpga_efficiency if hw_key.startswith("fpga") else gpu_efficiency
        results.append(estimate(hw, model, efficiency=eff, avg_seq_len=avg_seq_len))
    return results
