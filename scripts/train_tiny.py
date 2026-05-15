"""Train a tiny F-LLM reference model on a local text corpus."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from fllm.checkpoint import save_training_checkpoint
from fllm.config import FLLMConfig
from fllm.data import encode_text, load_text, sample_batch, split_tokens
from fllm.model import FLLMForCausalLM, count_parameters
from fllm.presets import PRESETS
from fllm.tokenizer import load_tokenizer


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
    parser.add_argument("--tokenizer", default="byte", help="`byte` or path to tokenizer.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="custom")
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
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-tie-embeddings", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--generate-tokens", type=int, default=120)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    text = load_text(args.corpus)
    tokens = encode_text(text, tokenizer)
    train_tokens, val_tokens = split_tokens(tokens)

    preset = PRESETS[args.preset]
    hidden_size = preset.hidden_size if preset else args.hidden_size
    num_layers = preset.num_layers if preset else args.num_layers
    num_heads = preset.num_heads if preset else args.num_heads
    local_window = preset.local_window if preset else args.local_window
    mlp_ratio = preset.mlp_ratio if preset else args.mlp_ratio
    use_moe = preset.use_moe if preset else False
    num_experts = preset.num_experts if preset else 1
    active_experts = preset.active_experts if preset else 1

    config = FLLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        local_window=local_window,
        mlp_ratio=mlp_ratio,
        dropout=args.dropout,
        tie_embeddings=not args.no_tie_embeddings,
        use_moe=use_moe,
        num_experts=num_experts,
        active_experts=active_experts,
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
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                f"val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 20.0)):.2f} "
                f"elapsed_s={elapsed:.1f}"
            )

    prompt = "Full FPGA inference"
    prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.generate_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.sample_temperature,
        top_k=args.sample_top_k,
        repetition_penalty=args.repetition_penalty,
    )[0].tolist()
    generated_text = tokenizer.decode(generated)
    print("--- sample ---")
    print(generated_text)

    save_training_checkpoint(
        path=out_dir / "model.pt",
        model=model,
        tokenizer=tokenizer,
        extra={"corpus": args.corpus, "steps": args.steps, "preset": args.preset},
    )
    (out_dir / "sample.txt").write_text(generated_text, encoding="utf-8")
    print(f"saved={out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
