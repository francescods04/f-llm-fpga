"""End-to-end integration test: quant + sparsity + vocab cache + speculative decode.

Validates that the full FPGA-targeted inference pipeline can be composed
in software without crashes and produces valid token sequences.
"""

import torch
from fllm.config import FLLMConfig
from fllm.model import FLLMForCausalLM
from fllm.quant import QuantConfig, quantize_model_
from fllm.sparsity import NMSparsity
from fllm.speculative import DraftModel, speculative_generate_simple


def _make_config():
    # A mini Qwen-like config that fits in CPU RAM
    return FLLMConfig(
        vocab_size=256,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        mlp_ratio=2,
        context_length=256,
        local_window=128,
        dropout=0.0,
        use_gqa=True,
        use_rope=True,
        use_moe=False,
        tie_embeddings=False,
    )


def test_end_to_end_int4_sparse():
    """Full pipeline: INT4 + 2:4 sparsity + vocab cache + speculative decode."""
    torch.manual_seed(42)
    cfg = _make_config()
    model = FLLMForCausalLM(cfg)

    # 1. Quantize to INT4 with 2:4 sparsity
    qcfg = QuantConfig(
        weight_bits=4,
        activation_bits=8,
        nm_sparsity=NMSparsity(n=2, m=4),
    )
    replaced = quantize_model_(model, qcfg)
    print(f"Replaced {replaced} Linear layers with QuantLinear")

    # 2. Vocab cache on LM head
    cache_ids = torch.arange(0, cfg.vocab_size, 2)  # every other token
    model.set_vocab_cache(cache_ids)

    # 3. Build draft model
    draft = DraftModel(cfg, num_layers=1)

    # 4. Generate with speculative decode
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    output = speculative_generate_simple(
        model, draft, prompt.clone(), max_new_tokens=12, gamma=3
    )

    assert output.shape[0] == 1
    assert output.shape[1] == 8 + 12  # prompt + 12 new tokens
    print(f"Generated sequence length: {output.shape[1]}")
    print(f"Generated tokens: {output[0].tolist()}")
    print("End-to-end INT4+sparse+speculative: PASS")


def test_end_to_end_int2():
    """Full pipeline: INT2 ternary + speculative decode."""
    torch.manual_seed(7)
    cfg = _make_config()
    model = FLLMForCausalLM(cfg)

    qcfg = QuantConfig(use_int2=True)
    replaced = quantize_model_(model, qcfg)
    print(f"Replaced {replaced} Linear layers with TernaryLinear")

    draft = DraftModel(cfg, num_layers=1)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    output = speculative_generate_simple(
        model, draft, prompt.clone(), max_new_tokens=8, gamma=2
    )

    assert output.shape[1] == 8 + 8
    print(f"Generated tokens: {output[0].tolist()}")
    print("End-to-end INT2+speculative: PASS")


def test_end_to_end_baseline_vs_speculative():
    """Verify that speculative with perfect draft matches baseline greedy."""
    torch.manual_seed(99)
    cfg = _make_config()
    target = FLLMForCausalLM(cfg)

    # Perfect draft
    draft = DraftModel(cfg, num_layers=cfg.num_layers)
    draft.load_state_dict(target.state_dict(), strict=False)

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    baseline = target.generate(prompt.clone(), max_new_tokens=6, temperature=0.0)
    spec = speculative_generate_simple(
        target, draft, prompt.clone(), max_new_tokens=6, gamma=2
    )

    assert torch.equal(baseline, spec), "Perfect draft should produce identical output"
    print("Baseline vs speculative equivalence: PASS")


if __name__ == "__main__":
    test_end_to_end_int4_sparse()
    test_end_to_end_int2()
    test_end_to_end_baseline_vs_speculative()
    print("\nAll end-to-end integration tests passed!")
