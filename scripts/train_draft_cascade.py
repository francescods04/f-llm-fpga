#!/usr/bin/env python3
"""Train the tiny draft model for speculative decoding cascade (Novelty N3).

Placeholder for the distillation pipeline.  In a real run it would:
  1. Load target Qwen3.6-A3B checkpoint.
  2. Build a 4-layer dense 100 M param draft (or 50 M / 500 M cascade levels).
  3. Distill draft outputs to match target logits on a diverse corpus.
  4. Write trained draft weights to `checkpoints/draft/...`.

Without GPU / dataset this script only validates the draft architecture
and produces a dummy checkpoint + synthetic acceptance sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import random

import torch

from fllm.config import FLLMConfig
from fllm.speculative import DraftModel


def dummy_distill(config: FLLMConfig, num_layers: int = 1) -> DraftModel:
    """Return an untrained draft model (weights initialised)."""
    draft = DraftModel(config, num_layers=num_layers)
    return draft


def synthetic_acceptance_sweep() -> list[dict]:
    """Synthetic acceptance-rate table for gate G8."""
    results = []
    for gamma in [1, 2, 4]:
        for tree_depth in [1, 2]:
            # Acceptance rate drops slightly with deeper tree
            base_accept = 0.82 if gamma == 1 else (0.72 if gamma == 2 else 0.60)
            accept = base_accept * (0.95 ** (tree_depth - 1))
            effective = gamma * accept
            results.append({
                "gamma": gamma,
                "tree_depth": tree_depth,
                "accept_rate": round(accept, 3),
                "effective_tok_per_forward": round(effective, 3),
                "pass_g8": effective >= 2.5,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train draft cascade")
    parser.add_argument("--config", default="Qwen3-A3B", help="Model config name")
    parser.add_argument("--layers", type=int, default=1, help="Draft layers")
    parser.add_argument("--output", default="experiments/draft_cascade_sweep.csv")
    args = parser.parse_args()

    print(f"[train_draft_cascade] Building draft ({args.layers} layers)...")
    cfg = FLLMConfig()
    draft = dummy_distill(cfg, args.layers)
    param_count = sum(p.numel() for p in draft.parameters())
    print(f"  Draft params: {param_count / 1e6:.1f} M")

    print("[train_draft_cascade] Synthetic acceptance sweep...")
    rows = synthetic_acceptance_sweep()
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gamma", "tree_depth", "accept_rate", "effective_tok_per_forward", "pass_g8"])
        writer.writeheader()
        writer.writerows(rows)

    passes = sum(1 for r in rows if r["pass_g8"])
    print(f"Wrote {len(rows)} rows to {args.output}; {passes} passed G8.")


if __name__ == "__main__":
    main()
