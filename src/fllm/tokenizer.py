"""Tokenizer utilities.

The byte tokenizer is useful for early FPGA smoke tests because it has a tiny
vocabulary. The BPE wrapper is the default path for a minimally usable model.
Both expose the same small interface so training, generation, and benchmarking can
switch tokenizers without touching model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


class TextTokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_token_id(self) -> int: ...

    @property
    def bos_token_id(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, ids: list[int] | tuple[int, ...]) -> str: ...


@dataclass(frozen=True)
class ByteTokenizer:
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    byte_offset: int = 3

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        raw = bytearray()
        for token_id in ids:
            if token_id < self.byte_offset:
                continue
            value = token_id - self.byte_offset
            if 0 <= value <= 255:
                raw.append(value)
        return raw.decode("utf-8", errors="replace")


class BPETokenizer:
    """Wrapper around a Hugging Face `tokenizers` JSON file."""

    def __init__(self, path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BPETokenizer requires the `tokenizers` package. "
                "Install the research extra or use ByteTokenizer."
            ) from exc

        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self._pad_token_id = self._required_id(SPECIAL_TOKENS["pad"])
        self._bos_token_id = self._required_id(SPECIAL_TOKENS["bos"])
        self._eos_token_id = self._required_id(SPECIAL_TOKENS["eos"])

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def _required_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token {token!r}")
        return token_id


def load_tokenizer(path: str | Path | None) -> TextTokenizer:
    if path is None or str(path).lower() == "byte":
        return ByteTokenizer()
    return BPETokenizer(path)


def tokenizer_metadata(tokenizer: TextTokenizer) -> dict[str, str]:
    if isinstance(tokenizer, ByteTokenizer):
        return {"type": "byte"}
    if isinstance(tokenizer, BPETokenizer):
        return {"type": "bpe", "path": str(tokenizer.path)}
    raise TypeError(f"unsupported tokenizer type: {type(tokenizer)!r}")

