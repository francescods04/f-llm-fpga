"""Measure quality delta when running model through the FPGA-equivalent path.

Compares BF16 forward vs FPGA-sim forward (INT4 weights, INT8 acts, BFP8 KV,
LUT SiLU/softmax/rsqrt). Reports loss/ppl on a small corpus chunk.
"""

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
from fllm.fpga_sim import fpga_sim_mode


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def eval_loss(model, tokens, *, batch_size, seq_len, iters, device, seed):
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--json-out", default=None)
    return p


def main():
    args = build_parser().parse_args()
    device = choose_device(args.device)
    model, tokenizer, _ = load_training_checkpoint(args.checkpoint, device=device)
    tokens = encode_text(load_text(args.corpus), tokenizer)
    seq_len = min(args.seq_len, model.config.context_length)

    bf16_loss = eval_loss(
        model, tokens, batch_size=args.batch_size, seq_len=seq_len,
        iters=args.iters, device=device, seed=1,
    )

    sim_model = copy.deepcopy(model)
    with fpga_sim_mode(sim_model):
        sim_loss = eval_loss(
            sim_model, tokens, batch_size=args.batch_size, seq_len=seq_len,
            iters=args.iters, device=device, seed=1,
        )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "iters": args.iters,
        "seq_len": seq_len,
        "bf16_loss": bf16_loss,
        "bf16_ppl": math.exp(min(bf16_loss, 20.0)),
        "fpga_sim_loss": sim_loss,
        "fpga_sim_ppl": math.exp(min(sim_loss, 20.0)),
        "delta_ppl": math.exp(min(sim_loss, 20.0)) - math.exp(min(bf16_loss, 20.0)),
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
