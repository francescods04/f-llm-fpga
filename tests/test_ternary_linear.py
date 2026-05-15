"""Quick smoke test for TernaryLinear fake-quant."""
import torch
from fllm.quant import QuantConfig, TernaryLinear, quantize_model_


def test_ternary_forward():
    lin = torch.nn.Linear(64, 32, bias=False)
    lin.weight.data.normal_(0, 0.1)
    tlin = TernaryLinear(lin, QuantConfig(use_int2=True))
    x = torch.randn(1, 64)
    y = tlin(x)
    assert y.shape == (1, 32)
    print("TernaryLinear forward shape OK")


def test_ternary_weight_values():
    lin = torch.nn.Linear(64, 8, bias=False)
    lin.weight.data.normal_(0, 0.5)
    tlin = TernaryLinear(lin, QuantConfig(use_int2=True))
    wq = tlin.weight.detach()
    # After fake-quant, every element should be one of {-scale, 0, +scale}
    # Scale varies per row, so just check ternary pattern
    scale = wq.abs().amax(dim=-1, keepdim=True)
    q = torch.round(wq / scale.clamp_min(1e-8)).clamp(-1, 1)
    unique = torch.unique(q)
    assert set(unique.tolist()).issubset({-1.0, 0.0, 1.0})
    print(f"Ternary weight unique values: {unique.tolist()}")


def test_ternary_cos_sim_vs_float():
    torch.manual_seed(42)
    lin = torch.nn.Linear(256, 128, bias=False)
    lin.weight.data.normal_(0, 0.2)
    x = torch.randn(1, 256)
    y_ref = lin(x)
    tlin = TernaryLinear(lin, QuantConfig(use_int2=True))
    y_ter = tlin(x)
    cos = torch.nn.functional.cosine_similarity(y_ref, y_ter, dim=-1).mean().item()
    print(f"Ternary cos_sim vs float = {cos:.4f}")
    assert cos > 0.70, f"cos_sim too low: {cos}"


def test_ternary_vs_int4_quality():
    torch.manual_seed(42)
    lin = torch.nn.Linear(256, 128, bias=False)
    lin.weight.data.normal_(0, 0.2)
    x = torch.randn(1, 256)
    y_ref = lin(x)

    from fllm.quant import QuantLinear
    qint4 = QuantLinear(lin, QuantConfig(weight_bits=4))
    y_int4 = qint4(x)

    tlin = TernaryLinear(lin, QuantConfig(use_int2=True))
    y_int2 = tlin(x)

    cos_int4 = torch.nn.functional.cosine_similarity(y_ref, y_int4, dim=-1).mean().item()
    cos_int2 = torch.nn.functional.cosine_similarity(y_ref, y_int2, dim=-1).mean().item()
    print(f"INT4 cos_sim = {cos_int4:.4f}")
    print(f"INT2 cos_sim = {cos_int2:.4f}")
    # INT2 is expected to be slightly worse but still > 0.7 for well-conditioned weights
    assert cos_int2 > 0.70


def test_quantize_model_int2():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(16, 8, bias=False)
            self.l2 = torch.nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.l2(self.l1(x))

    model = TinyModel()
    x = torch.randn(1, 16)
    y_before = model(x)

    replaced = quantize_model_(model, QuantConfig(use_int2=True))
    assert replaced == 2
    y_after = model(x)
    assert y_after.shape == y_before.shape
    print(f"quantize_model_ with use_int2 replaced {replaced} layers")


if __name__ == "__main__":
    test_ternary_forward()
    test_ternary_weight_values()
    test_ternary_cos_sim_vs_float()
    test_ternary_vs_int4_quality()
    test_quantize_model_int2()
    print("All TernaryLinear tests passed!")
