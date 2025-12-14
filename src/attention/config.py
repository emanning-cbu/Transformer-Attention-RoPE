"""
Configuration for Transformer Multi-Head Attention (+ optional RoPE + optional causal masking).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    # Core dimensions
    d_model: int = 512
    num_heads: int = 8
    max_seq_len: int = 2048

    # Feature toggles
    masked: bool = True
    use_rope: bool = True    # True = apply RoPE to Q/K inside attention

    # RoPE hyperparameter
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
