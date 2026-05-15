"""Named model-size presets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    hidden_size: int
    num_layers: int
    num_heads: int
    local_window: int
    mlp_ratio: int = 4
    use_moe: bool = False
    num_experts: int = 1
    active_experts: int = 1


PRESETS = {
    "custom": None,
    "tiny": ModelPreset(hidden_size=64, num_layers=1, num_heads=4, local_window=16),
    "small": ModelPreset(hidden_size=128, num_layers=2, num_heads=4, local_window=32),
    "usable-25m": ModelPreset(hidden_size=384, num_layers=8, num_heads=6, local_window=256),
    "usable-60m": ModelPreset(hidden_size=512, num_layers=12, num_heads=8, local_window=256),
    "paper-150m": ModelPreset(hidden_size=768, num_layers=16, num_heads=12, local_window=512),
    "moe-smoke": ModelPreset(
        hidden_size=128, num_layers=2, num_heads=4, local_window=32,
        mlp_ratio=2, use_moe=True, num_experts=4, active_experts=2,
    ),
    "moe-a3b-proxy": ModelPreset(
        hidden_size=384, num_layers=8, num_heads=8, local_window=256,
        mlp_ratio=2, use_moe=True, num_experts=32, active_experts=4,
    ),
}

