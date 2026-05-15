"""Top-k mixture-of-experts MLP for FPGA-native reference model.

Mirrors the Qwen3-MoE / DeepSeek-style routing pattern that the FPGA target
architecture must support: a small router picks `active_experts` of
`num_experts` SwiGLU expert MLPs per token. The reference path is dense and
correctness-focused; the FPGA path will replace the dense per-expert linear
with packed INT4 matvec kernels driven by the router output.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fllm.config import FLLMConfig


class Expert(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, inner_size, bias=False)
        self.gate = nn.Linear(hidden_size, inner_size, bias=False)
        self.down = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEMLP(nn.Module):
    """Top-k token-choice MoE.

    Token routing is decided by a tiny linear router. Each token gets routed to
    `active_experts` experts; expert outputs are weighted-summed using the
    softmax of the selected router logits (renormalized to top-k, matching
    Qwen3-MoE behavior).
    """

    def __init__(self, config: FLLMConfig) -> None:
        super().__init__()
        if config.active_experts <= 0 or config.active_experts > config.num_experts:
            raise ValueError("active_experts must be in (0, num_experts]")
        inner = config.moe_inner_size if config.moe_inner_size > 0 else config.hidden_size * config.mlp_ratio
        self.num_experts = config.num_experts
        self.active_experts = config.active_experts
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(config.hidden_size, inner) for _ in range(config.num_experts)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        logits = self.router(flat)
        probs_all = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(logits, self.active_experts, dim=-1)
        gates = F.softmax(top_vals, dim=-1)

        out = torch.zeros_like(flat)
        for slot in range(self.active_experts):
            expert_ids = top_idx[:, slot]
            slot_gate = gates[:, slot].unsqueeze(-1)
            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if not torch.any(mask):
                    continue
                tokens = flat[mask]
                out[mask] = out[mask] + slot_gate[mask] * self.experts[expert_id](tokens)

        out = out.reshape(batch, seq_len, hidden)
        self._last_aux_loss = self._compute_aux_loss(probs_all, top_idx)
        return self.dropout(out)

    def _compute_aux_loss(self, probs_all: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """Switch Transformer auxiliary load-balance loss.

        Penalty = num_experts * sum_i (f_i * P_i), where:
          f_i = fraction of tokens routed to expert i (over top-k slots),
          P_i = mean router probability for expert i.
        Scale so a perfectly balanced router yields aux_loss = 1.
        """
        num_tokens = top_idx.size(0)
        mask = F.one_hot(top_idx, num_classes=self.num_experts).float()
        f_i = mask.sum(dim=(0, 1)) / (num_tokens * self.active_experts)
        p_i = probs_all.mean(dim=0)
        return self.num_experts * (f_i * p_i).sum()

    def last_aux_loss(self) -> torch.Tensor:
        return getattr(self, "_last_aux_loss", torch.tensor(0.0))

    def routing_stats(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostic: mean expert load and entropy of routing distribution."""
        flat = x.reshape(-1, x.size(-1))
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(logits, self.active_experts, dim=-1)
        load = torch.zeros(self.num_experts, device=x.device)
        for slot in range(self.active_experts):
            load.scatter_add_(
                0, top_idx[:, slot], torch.ones_like(top_idx[:, slot], dtype=torch.float)
            )
        load = load / load.sum()
        entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1).mean()
        return {"expert_load": load, "router_entropy": entropy}
