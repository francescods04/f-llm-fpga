"""Evaluate checkpoint quality on text loss and simple generation metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fllm.checkpoint import load_training_checkpoint
from fllm.data import encode_text, load_text, sample_batch


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    *,
    model,
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = sample_batch(tokens, batch_size=batch_size, seq_len=seq_len, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def distinct_n(token_ids: list[int], n: int) -> float:
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[idx : idx + n]) for idx in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tiny/model.pt")
    parser.add_argument("--corpus", default="docs/sample_corpus.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--prompt", action="append", default=["Full FPGA inference"])
    parser.add_argument("--new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    model, tokenizer, _ = load_training_checkpoint(args.checkpoint, device=device)

    tokens = encode_text(load_text(args.corpus), tokenizer)
    loss = estimate_loss(
        model=model,
        tokens=tokens,
        batch_size=args.batch_size,
        seq_len=min(args.seq_len, model.config.context_length),
        iters=args.iters,
        device=device,
    )

    samples = []
    for prompt in args.prompt:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
        output = model.generate(
            input_ids,
            max_new_tokens=args.new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=1.1,
        )[0].tolist()
        generated = output[input_ids.numel() :]
        samples.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(output),
                "distinct_1": distinct_n(generated, 1),
                "distinct_2": distinct_n(generated, 2),
            }
        )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backend": device.type,
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "samples": samples,
    }

    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

