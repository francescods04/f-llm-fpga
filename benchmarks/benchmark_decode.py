"""Benchmark greedy decode throughput for the reference model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fllm.config import FLLMConfig
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


def load_model(checkpoint_path: str | None, device: torch.device) -> FLLMForCausalLM:
    if checkpoint_path is None:
        config = FLLMConfig(
            vocab_size=ByteTokenizer().vocab_size,
            context_length=64,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            local_window=32,
        )
        return FLLMForCausalLM(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = FLLMConfig(**checkpoint["config"])
    model = FLLMForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default="Full FPGA inference")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    tokenizer = ByteTokenizer()
    model = load_model(args.checkpoint, device)
    model.eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)

    if args.warmup_tokens > 0:
        _ = model.generate(prompt_ids, max_new_tokens=args.warmup_tokens)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    output = model.generate(prompt_ids, max_new_tokens=args.new_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_second = args.new_tokens / elapsed
    result = {
        "model": "f-llm-reference",
        "backend": device.type,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "parameters": count_parameters(model),
        "prompt_tokens": int(prompt_ids.numel()),
        "generated_tokens": args.new_tokens,
        "elapsed_s": elapsed,
        "tokens_per_second": tokens_per_second,
        "ms_per_token": 1000.0 / tokens_per_second,
        "watts": None,
        "joule_per_token": None,
        "sample": tokenizer.decode(output[0].tolist()),
    }

    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

