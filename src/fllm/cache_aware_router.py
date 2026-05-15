"""Cache-aware MoE router — Novelty N1.

Problem: In a standard MoE router each token picks top-k experts via softmax.
The FPGA can only keep a small subset of experts in on-chip URAM/HBM at once.
If the router sends a token to a non-resident (cold) expert, the FPGA must
stream that expert's weights from host DDR or via PCIe peer — hundreds of µs.

Solution: bias the router logits so that resident experts are preferred,
while keeping the quality degradation bounded.  We do this by:

  1. Maintaining a resident_mask per layer (which experts are on-chip).
  2. Adding a learned bias term α·resident_mask to the router logits.
  3. Fine-tuning (LoRA on router or full-router FT) with a composite loss:
       L_route = L_lb + λ · KL(p_router || softmax(logits + α·resident_mask))
     where L_lb is the standard load-balancing auxiliary loss.

The gate G3 decides whether the hit rate justifies the quality cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CacheAwareConfig:
    num_experts: int = 128
    top_k: int = 8
    alpha: float = 2.0          # resident-bias temperature
    lambda_kl: float = 0.1    # KL regularisation weight
    lambda_lb: float = 0.01   # load-balancing loss weight
    fine_tune_router: bool = True


class CacheAwareRouter(nn.Module):
    """Router that biases toward on-chip resident experts.

    Forward accepts hidden states and a resident_mask BoolTensor
    (True = expert currently resident in FPGA URAM/HBM).
    """

    def __init__(self, hidden_size: int, cfg: CacheAwareConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(hidden_size, cfg.num_experts, bias=False)

    def forward(
        self, hidden: torch.Tensor, *, resident_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (expert_weights, expert_indices) both shape (B*T, top_k).

        Args:
            hidden: (batch, seq_len, hidden)
            resident_mask: (num_experts,) bool, True = on-chip.
        """
        B, T, H = hidden.shape
        x = hidden.view(B * T, H)
        logits = self.gate(x)                       # (B*T, num_experts)

        if resident_mask is not None and resident_mask.any():
            bias = resident_mask.float().unsqueeze(0) * self.cfg.alpha  # (1, E)
            biased_logits = logits + bias
        else:
            biased_logits = logits

        weights, indices = torch.topk(
            F.softmax(biased_logits, dim=-1), self.cfg.top_k, dim=-1
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices

    def compute_loss(
        self,
        hidden: torch.Tensor,
        resident_mask: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Composite training loss for cache-aware routing.

        Returns dict with 'total', 'lb', 'kl', 'hit_rate'.
        """
        B, T, H = hidden.shape
        x = hidden.view(B * T, H)
        logits = self.gate(x)

        # Load-balancing loss (standard MoE auxiliary loss)
        router_prob = F.softmax(logits, dim=-1).mean(dim=0)   # (E,)
        lb_loss = self.cfg.num_experts * (router_prob ** 2).sum()

        # KL loss: encourage router to stay close to biased distribution
        biased_logits = logits + resident_mask.float().unsqueeze(0) * self.cfg.alpha
        p_biased = F.softmax(biased_logits, dim=-1)
        p_router = F.softmax(logits, dim=-1)
        kl_loss = F.kl_div(p_router.log(), p_biased, reduction="batchmean")

        total = self.cfg.lambda_lb * lb_loss + self.cfg.lambda_kl * kl_loss

        # Hit rate: fraction of selected experts that are resident
        hits = resident_mask[expert_indices].float().mean()

        return {
            "total": total,
            "lb": lb_loss,
            "kl": kl_loss,
            "hit_rate": hits,
        }


def pick_resident_set(
    usage_histogram: torch.Tensor, capacity: int
) -> torch.Tensor:
    """Given per-expert usage counts, pick the top-`capacity` experts to keep
    resident on-chip.  Returns a bool mask of length num_experts."""
    _, topk = torch.topk(usage_histogram, capacity, largest=True, sorted=False)
    mask = torch.zeros(usage_histogram.numel(), dtype=torch.bool)
    mask[topk] = True
    return mask
