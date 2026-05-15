"""Train a tiny F-LLM reference model on a local text corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from fllm.config import FLLMConfig
from fllm.data import encode_text, load_text, sample_batch, split_tokens
from fllm.model import FLLMForCausalLM, count_parameters
from fllm.tokenizer import ByteTokenizer


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: FLLMForCausalLM,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = sample_batch(tokens, batch_size=batch_size, seq_len=seq_len, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="docs/sample_corpus.txt")
    parser.add_argument("--out-dir", default="checkpoints/tiny")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--generate-tokens", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    tokenizer = ByteTokenizer()
    text = load_text(args.corpus)
    tokens = encode_text(text, tokenizer)
    train_tokens, val_tokens = split_tokens(tokens)

    config = FLLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        local_window=args.local_window,
    )
    model = FLLMForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,} train={train_tokens.numel():,} val={val_tokens.numel():,}")
    print(f"parameters={count_parameters(model):,}")

    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(
            train_tokens,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
        )
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(
                model,
                val_tokens,
                batch_size=max(1, min(args.batch_size, 4)),
                seq_len=args.seq_len,
                device=device,
                eval_iters=args.eval_iters,
            )
            elapsed = time.perf_counter() - start_time
            print(
                f"step={step:04d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} elapsed_s={elapsed:.1f}"
            )

    prompt = "Full FPGA inference"
    prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    generated = model.generate(prompt_ids, max_new_tokens=args.generate_tokens)[0].tolist()
    generated_text = tokenizer.decode(generated)
    print("--- sample ---")
    print(generated_text)

    checkpoint = {
        "config": asdict(config),
        "model_state": model.state_dict(),
        "tokenizer": {"type": "byte"},
    }
    torch.save(checkpoint, out_dir / "model.pt")
    (out_dir / "sample.txt").write_text(generated_text, encoding="utf-8")
    print(f"saved={out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()

