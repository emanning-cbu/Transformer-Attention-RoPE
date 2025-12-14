import pytest

from src.attention.config import AttentionConfig


def test_config_valid_divisible():
    cfg = AttentionConfig(d_model=512, num_heads=8, max_seq_len=2048)
    cfg.validate()
    assert cfg.head_dim == 64


def test_config_invalid_not_divisible():
    cfg = AttentionConfig(d_model=513, num_heads=8, max_seq_len=2048)
    with pytest.raises(ValueError):
        cfg.validate()
