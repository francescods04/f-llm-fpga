"""Generate text from a saved F-LLM checkpoint."""

from __future__ import annotations

import argparse

import torch

from fllm.checkpoint import load_training_checkpoint


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tiny/model.pt")
    parser.add_argument("--prompt", default="Full FPGA inference")
    parser.add_argument("--new-tokens", type=int, default=120)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    model, tokenizer, _ = load_training_checkpoint(args.checkpoint, device=device)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)
    output = model.generate(
        prompt_ids,
        max_new_tokens=args.new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )[0].tolist()
    print(tokenizer.decode(output))


if __name__ == "__main__":
    main()
