"""Compare BF16 vs INT4-weight INT8-act quality on a trained checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fllm.checkpoint import load_training_checkpoint
from fllm.data import encode_text, load_text, sample_batch
from fllm.quant import QuantConfig, quantize_model_


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def loss_on(model, tokens, *, batch_size, seq_len, iters, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    model.eval()
    for _ in range(iters):
        torch.manual_seed(int(torch.randint(0, 1 << 30, (1,), generator=g).item()))
        x, y = sample_batch(tokens, batch_size=batch_size, seq_len=seq_len, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/bpe-smoke/model.pt")
    p.add_argument("--corpus", default="docs/sample_corpus.txt")
    p.add_argument("--device", default="auto")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--activation-bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=-1)
    p.add_argument("--nm-n", type=int, default=0, help="N:M sparsity N (0 = disabled)")
    p.add_argument("--nm-m", type=int, default=4, help="N:M sparsity M")
    p.add_argument("--json-out", default=None)
    return p


def main():
    args = build_parser().parse_args()
    device = choose_device(args.device)
    model, tokenizer, _ = load_training_checkpoint(args.checkpoint, device=device)
    tokens = encode_text(load_text(args.corpus), tokenizer)
    seq_len = min(args.seq_len, model.config.context_length)

    bf16_loss = loss_on(
        model, tokens,
        batch_size=args.batch_size, seq_len=seq_len, iters=args.iters,
        device=device, seed=1,
    )

    qmodel = copy.deepcopy(model)
    nm = None
    if args.nm_n > 0 and args.nm_n < args.nm_m:
        from fllm.sparsity import NMSparsity
        nm = NMSparsity(n=args.nm_n, m=args.nm_m)
    qcfg = QuantConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        weight_group_size=args.group_size,
        nm_sparsity=nm,
    )
    replaced = quantize_model_(qmodel, qcfg, skip=("lm_head",))
    quant_loss = loss_on(
        qmodel, tokens,
        batch_size=args.batch_size, seq_len=seq_len, iters=args.iters,
        device=device, seed=1,
    )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "replaced_linears": replaced,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "weight_group_size": args.group_size,
        "nm_sparsity": f"{args.nm_n}:{args.nm_m}" if nm else "none",
        "bf16_loss": bf16_loss,
        "bf16_ppl": math.exp(min(bf16_loss, 20.0)),
        "quant_loss": quant_loss,
        "quant_ppl": math.exp(min(quant_loss, 20.0)),
        "delta_ppl": math.exp(min(quant_loss, 20.0)) - math.exp(min(bf16_loss, 20.0)),
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
