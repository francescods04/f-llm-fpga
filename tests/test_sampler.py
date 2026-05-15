"""Sanity tests for hardware-style sampler + xorshift PRNG."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fllm.sampler import XorshiftPRNG, hardware_argmax, hardware_sample, streaming_max_index


def test_xorshift_deterministic_and_full_period():
    a = XorshiftPRNG(0xDEADBEEF)
    b = XorshiftPRNG(0xDEADBEEF)
    seq_a = [a.next_u32() for _ in range(1000)]
    seq_b = [b.next_u32() for _ in range(1000)]
    assert seq_a == seq_b
    assert all(0 <= v < (1 << 32) for v in seq_a)
    assert len(set(seq_a)) == 1000  # no near-period collisions in 1k draws


def test_hardware_argmax_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(1024)
    assert hardware_argmax(x) == int(torch.argmax(x).item())


def test_streaming_max_index_matches_torch():
    torch.manual_seed(1)
    x = torch.randn(2048)
    val, idx = streaming_max_index(x)
    assert idx == int(torch.argmax(x).item())
    assert abs(val - float(x.max().item())) < 1e-6


def test_hardware_sample_distribution_close_to_softmax():
    torch.manual_seed(2)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    prng = XorshiftPRNG(123456789)
    counts = Counter(hardware_sample(logits, prng=prng) for _ in range(20_000))
    probs = torch.softmax(logits, dim=-1).tolist()
    for idx, expected in enumerate(probs):
        observed = counts[idx] / 20_000
        assert abs(observed - expected) < 0.03, f"idx={idx} obs={observed} exp={expected}"


def test_hardware_sample_top_k_excludes_low_prob():
    logits = torch.tensor([0.1, 0.2, 5.0, 5.1])
    prng = XorshiftPRNG(987654321)
    seen = {hardware_sample(logits, prng=prng, top_k=2) for _ in range(2000)}
    assert seen <= {2, 3}, f"top-k violated: {seen}"
