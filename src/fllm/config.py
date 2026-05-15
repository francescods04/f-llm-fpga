"""Model configuration objects for the software reference path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FLLMConfig:
    """Small, hardware-visible decoder configuration.

    These defaults are intentionally small. The first goal is not quality; it is a
    stable model shape that can be trained, quantized, benchmarked, and then lowered
    into FPGA kernels.
    """

    vocab_size: int = 8192
    context_length: int = 512
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 0  # 0 -> MHA (num_kv_heads=num_heads); >0 -> GQA
    local_window: int = 256
    compressed_block_size: int = 16
    compressed_top_k: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    use_moe: bool = False
    num_experts: int = 16
    active_experts: int = 2
    moe_inner_size: int = 0  # 0 -> use hidden * mlp_ratio; >0 -> use this per-expert
    use_rope: bool = False
    rope_theta: float = 10000.0
    use_gqa: bool = False

    def kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads > 0 else self.num_heads
