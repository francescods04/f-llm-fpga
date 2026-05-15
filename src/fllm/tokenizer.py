"""Small tokenizer utilities.

The first implementation uses bytes instead of BPE. This keeps the dependency
surface small and gives the FPGA path a tiny output vocabulary for early tests.
Later experiments can replace this with BPE while preserving the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass


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

