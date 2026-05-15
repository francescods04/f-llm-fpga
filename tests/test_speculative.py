"""Smoke tests for speculative decoding.

Validates that speculative_generate_simple produces identical greedy output
when the draft is a perfect copy of the target (acceptance rate = 100%).
"""

import torch
from fllm.config import FLLMConfig
from fllm.model import FLLMForCausalLM
from fllm.speculative import DraftModel, speculative_generate_simple


def _make_config():
    return FLLMConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        mlp_ratio=2,
        context_length=128,
        local_window=64,
        dropout=0.0,
        use_gqa=True,
        use_rope=True,
        use_moe=False,
    )


def test_draft_forward_shape():
    cfg = _make_config()
    draft = DraftModel(cfg, num_layers=1)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits = draft(x)
    assert logits.shape == (1, 8, cfg.vocab_size)
    print("Draft forward shape OK")


def test_speculative_with_perfect_draft():
    """When draft == target (same weights), all tokens should be accepted."""
    torch.manual_seed(42)
    cfg = _make_config()
    target = FLLMForCausalLM(cfg)

    # Perfect draft: same architecture, copy weights
    draft = DraftModel(cfg, num_layers=cfg.num_layers)
    draft.load_state_dict(target.state_dict(), strict=False)

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    # Baseline greedy
    target.eval()
    baseline = target.generate(prompt.clone(), max_new_tokens=8, temperature=0.0)

    # Speculative with gamma=3
    speculative = speculative_generate_simple(
        target, draft, prompt.clone(), max_new_tokens=8, gamma=3
    )

    print(f"Baseline:    {baseline[0].tolist()}")
    print(f"Speculative: {speculative[0].tolist()}")

    assert torch.equal(baseline, speculative), "Speculative decode diverged from baseline!"
    print("Perfect draft acceptance: 100% — PASS")


def test_speculative_with_random_draft():
    """With a random draft, should still produce valid tokens (no crash)."""
    torch.manual_seed(7)
    cfg = _make_config()
    target = FLLMForCausalLM(cfg)
    draft = DraftModel(cfg, num_layers=1)

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    result = speculative_generate_simple(
        target, draft, prompt.clone(), max_new_tokens=6, gamma=2
    )
    assert result.shape[0] == 1
    assert result.shape[1] == 4 + 6  # prompt + 6 new tokens
    print("Random draft speculative decode: no crash — PASS")


def test_draft_is_smaller():
    cfg = _make_config()
    target = FLLMForCausalLM(cfg)
    draft = DraftModel(cfg, num_layers=1)

    # Exclude embeddings and head from count (they are fixed overhead)
    def body_params(m):
        return sum(p.numel() for n, p in m.named_parameters() if "embedding" not in n and "lm_head" not in n)

    target_body = body_params(target)
    draft_body = body_params(draft)
    print(f"Target body params: {target_body:,}")
    print(f"Draft body params:  {draft_body:,}")
    assert draft_body < target_body * 0.6, "Draft body should be significantly smaller"
    print("Draft size check PASS")


if __name__ == "__main__":
    test_draft_forward_shape()
    test_draft_is_smaller()
    test_speculative_with_perfect_draft()
    test_speculative_with_random_draft()
    print("All speculative decode tests passed!")
