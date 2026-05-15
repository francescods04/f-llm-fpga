"""Load a Hugging Face Qwen3-style config.json and translate to FLLMConfig.

This is a *shape* import. Weights are not loaded here — the goal is to validate
that our reference model can instantiate the target architecture, run a dummy
forward, and report parameter counts that match the HF model's declared
specifications.

Expected HF config fields (Qwen3-MoE flavor):
    hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
    head_dim (optional), intermediate_size, num_experts, num_experts_per_tok,
    moe_intermediate_size (optional), max_position_embeddings, rope_theta,
    vocab_size, tie_word_embeddings.

Anything missing falls back to a safe default; the function returns the
mapping report alongside the FLLMConfig so the caller can see what was
inferred vs taken verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fllm.config import FLLMConfig


@dataclass
class HFImportReport:
    source: str
    matched: dict[str, Any] = field(default_factory=dict)
    inferred: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_hf_config(path: str | Path) -> tuple[FLLMConfig, HFImportReport]:
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    report = HFImportReport(source=str(p))

    def take(key: str, default: Any, *, mark: str = "matched") -> Any:
        if key in data:
            getattr(report, mark)[key] = data[key]
            return data[key]
        report.inferred[key] = default
        return default

    hidden = take("hidden_size", 4096)
    n_layers = take("num_hidden_layers", 48)
    n_heads = take("num_attention_heads", 32)
    n_kv = take("num_key_value_heads", n_heads)
    head_dim = data.get("head_dim")
    if head_dim is not None:
        report.matched["head_dim"] = head_dim
        if head_dim * n_heads != hidden:
            report.warnings.append(
                f"head_dim*num_heads={head_dim*n_heads} != hidden_size={hidden}; "
                "FLLM ties head_dim = hidden/num_heads"
            )

    inter = take("intermediate_size", hidden * 2)
    n_experts = take("num_experts", 1)
    active = take("num_experts_per_tok", min(8, n_experts))
    moe_inter = data.get("moe_intermediate_size", inter)
    if "moe_intermediate_size" in data:
        report.matched["moe_intermediate_size"] = moe_inter

    ctx = take("max_position_embeddings", 4096)
    rope_theta = take("rope_theta", 10_000_000.0)
    vocab = take("vocab_size", 152064)
    tie = take("tie_word_embeddings", False)

    if hidden % n_heads:
        report.warnings.append(f"hidden_size={hidden} not divisible by num_heads={n_heads}")
    if n_heads % max(n_kv, 1):
        report.warnings.append(f"num_heads={n_heads} not multiple of num_kv_heads={n_kv}")

    if n_experts > 1:
        moe_inner = int(moe_inter)
        mlp_ratio = max(1, int(round(inter / hidden)))
        report.inferred["mlp_ratio"] = mlp_ratio
        report.inferred["moe_inner_size"] = moe_inner
    else:
        moe_inner = 0
        mlp_ratio = max(1, int(round(inter / hidden)))
        report.inferred["mlp_ratio"] = mlp_ratio

    cfg = FLLMConfig(
        vocab_size=vocab,
        context_length=min(ctx, 4096),  # FLLM ctx is software-configurable; FPGA caps lower
        hidden_size=hidden,
        num_layers=n_layers,
        num_heads=n_heads,
        num_kv_heads=n_kv,
        local_window=ctx,
        mlp_ratio=mlp_ratio,
        dropout=0.0,
        tie_embeddings=bool(tie),
        use_moe=n_experts > 1,
        num_experts=n_experts,
        active_experts=active,
        moe_inner_size=moe_inner,
        use_rope=True,
        rope_theta=float(rope_theta),
        use_gqa=True,
    )
    return cfg, report


def write_qwen3_a3b_template(path: str | Path) -> Path:
    """Emit an assumed Qwen3.6-35B-A3B config.json for offline shape testing.

    Numbers are placeholders matching the public Qwen3 MoE pattern. Replace
    with the real config.json once the HF repo is accessible.
    """
    template = {
        "model_type": "qwen3_moe",
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1408,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "max_position_embeddings": 32768,
        "rope_theta": 10000000.0,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return out
