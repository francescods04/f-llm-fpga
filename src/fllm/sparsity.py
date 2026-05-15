"""Structured N:M sparsity simulation.

Each consecutive group of M values along the last (input) dimension keeps the
N largest by magnitude; the rest are zeroed. The FPGA matvec engine reads
weights in M-sized lanes; a "zero-row skip" controller drops the M-N zero
weight bytes entirely, saving exactly that fraction of HBM bandwidth.

Common patterns:
    2:4  -> 50% zeros (GPU sparse tensor cores)
    1:4  -> 75% zeros (GPU cannot accelerate this)
    1:8  -> 87.5% zeros
    2:8  -> 75% zeros
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NMSparsity:
    n: int = 1
    m: int = 4

    @property
    def density(self) -> float:
        return self.n / self.m

    @property
    def saved_byte_factor(self) -> float:
        """Effective bytes/value vs dense weight read."""
        return self.density


def apply_nm_mask(weight: torch.Tensor, pattern: NMSparsity) -> torch.Tensor:
    """Zero out (M-N) smallest-magnitude values in every M-sized group.

    Weight is (out_features, in_features). Groups are along in_features.
    Pads to a multiple of M with zeros temporarily, then unpads.
    """
    if pattern.n >= pattern.m:
        return weight
    out_f, in_f = weight.shape
    pad = (-in_f) % pattern.m
    if pad:
        padding = torch.zeros(out_f, pad, device=weight.device, dtype=weight.dtype)
        w = torch.cat([weight, padding], dim=-1)
    else:
        w = weight
    groups = w.shape[-1] // pattern.m
    grouped = w.reshape(out_f, groups, pattern.m)
    abs_g = grouped.abs()
    threshold, _ = abs_g.kthvalue(pattern.m - pattern.n, dim=-1, keepdim=True)
    mask = abs_g > threshold
    extra = pattern.n - mask.sum(dim=-1, keepdim=True)
    if extra.any():
        eq = (abs_g == threshold) & ~mask
        cum = eq.cumsum(dim=-1)
        tiebreak = (cum <= extra) & eq
        mask = mask | tiebreak
    masked = grouped * mask.to(grouped.dtype)
    masked = masked.reshape(out_f, w.shape[-1])
    return masked[:, :in_f]


def apply_nm_to_model(model: nn.Module, pattern: NMSparsity, *, skip: tuple[str, ...] = ("lm_head",)) -> int:
    """Apply N:M masking in-place to every nn.Linear weight in the model."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(s) for s in skip):
                continue
            with torch.no_grad():
                module.weight.copy_(apply_nm_mask(module.weight.data, pattern))
            count += 1
    return count


def retrain_nm_mask_lora(
    model: nn.Module,
    pattern: NMSparsity,
    data_loader,
    num_steps: int = 500,
    lr: float = 1e-4,
    device: str = "cpu",
) -> None:
    """Post-mask LoRA fine-tune to recover perplexity after N:M sparsity.

    This is a minimal SGD loop that only updates non-masked weight entries,
    freezing the zero positions.  In a real pipeline this would use PEFT
    (parameter-efficient fine-tuning); here we simulate with a simple mask.
    """
    import torch.optim as optim

    # Build masks for every Linear that was pruned
    masks: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            mask = apply_nm_mask(torch.ones_like(module.weight), pattern)
            masks[name] = (mask > 0).to(module.weight.dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=lr)

    model.train()
    step = 0
    for batch in data_loader:
        if step >= num_steps:
            break
        opt.zero_grad()
        # Assuming batch is (input_ids, labels) or dict
        if isinstance(batch, dict):
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
        else:
            x, y = batch
            loss = model(x.to(device), labels=y.to(device)).loss
        loss.backward()

        # Apply mask to gradients so zeros stay zero
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in masks and module.weight.grad is not None:
                module.weight.grad *= masks[name]

        opt.step()
        step += 1
    model.eval()


def identify_cold_experts(usage_histogram: torch.Tensor, cold_fraction: float = 0.5) -> torch.Tensor:
    """Return a bool mask of the bottom `cold_fraction` experts by usage."""
    num_experts = usage_histogram.numel()
    k = max(1, int(num_experts * cold_fraction))
    _, bottomk = torch.topk(usage_histogram, k, largest=False, sorted=False)
    mask = torch.zeros(num_experts, dtype=torch.bool)
    mask[bottomk] = True
    return mask
