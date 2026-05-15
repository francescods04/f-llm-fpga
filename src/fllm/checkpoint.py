"""Checkpoint helpers for models and tokenizers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from shutil import copyfile
from typing import Any

import torch

from fllm.config import FLLMConfig
from fllm.model import FLLMForCausalLM
from fllm.tokenizer import BPETokenizer, ByteTokenizer, TextTokenizer


def save_training_checkpoint(
    *,
    path: str | Path,
    model: FLLMForCausalLM,
    tokenizer: TextTokenizer,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_info = _save_tokenizer_metadata(tokenizer, checkpoint_path.parent)
    payload = {
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "tokenizer": tokenizer_info,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, checkpoint_path)


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[FLLMForCausalLM, TextTokenizer, dict[str, Any]]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device)
    config = FLLMConfig(**checkpoint["config"])
    model = FLLMForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = _load_tokenizer_metadata(checkpoint.get("tokenizer", {"type": "byte"}), path.parent)
    return model, tokenizer, checkpoint


def _save_tokenizer_metadata(tokenizer: TextTokenizer, checkpoint_dir: Path) -> dict[str, str]:
    if isinstance(tokenizer, ByteTokenizer):
        return {"type": "byte"}

    if isinstance(tokenizer, BPETokenizer):
        out_name = "tokenizer.json"
        out_path = checkpoint_dir / out_name
        source = tokenizer.path.resolve()
        if source != out_path.resolve():
            copyfile(source, out_path)
        return {"type": "bpe", "path": out_name}

    raise TypeError(f"unsupported tokenizer type: {type(tokenizer)!r}")


def _load_tokenizer_metadata(metadata: dict[str, str], checkpoint_dir: Path) -> TextTokenizer:
    tokenizer_type = metadata.get("type", "byte")
    if tokenizer_type == "byte":
        return ByteTokenizer()
    if tokenizer_type == "bpe":
        raw_path = Path(metadata["path"])
        path = raw_path if raw_path.is_absolute() else checkpoint_dir / raw_path
        return BPETokenizer(path)
    raise ValueError(f"unsupported tokenizer type: {tokenizer_type!r}")

