"""Instantiate the reference model and run one greedy decode step."""

from __future__ import annotations

import torch

from fllm.config import FLLMConfig
from fllm.model import FLLMForCausalLM, count_parameters


def main() -> None:
    config = FLLMConfig(
        vocab_size=1024,
        context_length=64,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        local_window=32,
    )
    model = FLLMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    logits = model(input_ids)
    generated = model.generate(input_ids, max_new_tokens=4)

    print(f"parameters={count_parameters(model):,}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"generated_shape={tuple(generated.shape)}")


if __name__ == "__main__":
    main()

