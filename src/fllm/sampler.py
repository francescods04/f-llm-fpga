"""Hardware-style categorical sampler with xorshift PRNG.

Bit-exact Python reference for the FPGA hardware sampler. Mirrors the Talos V2
methodology: PRNG + cumulative-sum sampler entirely on-chip, no host roundtrip.

The FPGA implementation:
    1. computes per-token logits (already in fixed-point);
    2. applies a fixed-point softmax via the LUT engine (see lut_activations);
    3. multiplies the PRNG output by the cumulative-sum total to produce a
       sampling threshold;
    4. scans the cumulative distribution and selects the first index that
       exceeds the threshold.

Steps 3-4 fuse into a single pass: stream cum-sum, compare against threshold,
emit the first crossing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class XorshiftPRNG:
    """32-bit xorshift. State must be nonzero. Bit-exact with HDL implementation.

    Reference: Marsaglia, "Xorshift RNGs", 2003.
    """
    state: int = 0xDEADBEEF

    def next_u32(self) -> int:
        s = self.state & 0xFFFFFFFF
        if s == 0:
            s = 1
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        self.state = s
        return s

    def next_unit(self) -> float:
        return self.next_u32() / 0x1_0000_0000


def hardware_sample(
    logits: torch.Tensor,
    *,
    prng: XorshiftPRNG,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> int:
    """One-token categorical sample mimicking the FPGA cumulative-scan path.

    Returns the selected token id as a Python int (a single value flows back
    to the host stream in the RTL).
    """
    if logits.dim() != 1:
        raise ValueError("expected 1-D logits")
    x = logits.to(torch.float32)
    if temperature > 0:
        x = x / max(temperature, 1e-6)
    if top_k is not None and 0 < top_k < x.numel():
        values, _ = torch.topk(x, k=top_k)
        cutoff = values[-1]
        x = torch.where(x < cutoff, torch.full_like(x, float("-inf")), x)
    probs = torch.softmax(x, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    threshold = prng.next_unit() * float(cum[-1].item())
    for idx in range(cum.numel()):
        if float(cum[idx].item()) >= threshold:
            return idx
    return int(cum.numel() - 1)


def hardware_argmax(logits: torch.Tensor) -> int:
    """Greedy path = streaming max-and-index reduction in RTL."""
    if logits.dim() != 1:
        raise ValueError("expected 1-D logits")
    return int(torch.argmax(logits).item())


def streaming_max_index(values: torch.Tensor) -> tuple[float, int]:
    """Cycle-accurate model of the LM-head streaming argmax engine.

    The FPGA scans the vocabulary one beat at a time, holding (max, idx) in
    two registers. This function returns the same result and serves as the
    test oracle for the RTL reduction tree.
    """
    best = float("-inf")
    best_idx = -1
    for idx in range(values.numel()):
        v = float(values[idx].item())
        if v > best:
            best = v
            best_idx = idx
    return best, best_idx
