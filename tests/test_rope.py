import pytest
import torch
from src.attention.rope import build_rope_cache, apply_rope, rotate_half


def test_rope_shapes_and_position0_identity():
    B, H, T, D = 2, 8, 5, 64
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    cos, sin = build_rope_cache(
        seq_len=T, head_dim=D, base=10000.0, device=q.device, dtype=q.dtype)
    q2, k2 = apply_rope(q, k, cos, sin)

    # shape preserved
    assert q2.shape == q.shape
    assert k2.shape == k.shape

    # position 0 should be unchanged because cos(0)=1 and sin(0)=0
    assert torch.allclose(q2[:, :, 0, :], q[:, :, 0, :], atol=1e-6)
    assert torch.allclose(k2[:, :, 0, :], k[:, :, 0, :], atol=1e-6)


def test_rotate_half_swaps_pairs():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    out = rotate_half(x)
    expected = torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]])
    assert torch.equal(out, expected)


def test_rope_cache_dtype_and_shape():
    cos, sin = build_rope_cache(seq_len=3, head_dim=6, dtype=torch.float16)
    assert cos.shape == (3, 6)
    assert sin.shape == (3, 6)
    assert cos.dtype == torch.float16
    assert sin.dtype == torch.float16
    assert torch.allclose(cos[0], torch.ones(6, dtype=torch.float16))
    assert torch.allclose(sin[0], torch.zeros(6, dtype=torch.float16))


def test_apply_rope_offset_matches_manual_slice():
    B, H, T, D = 1, 2, 2, 4
    offset = 3
    cache_len = 8

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    cos, sin = build_rope_cache(
        seq_len=cache_len, head_dim=D, base=10000.0, device=q.device, dtype=q.dtype
    )

    cos_slice = cos[offset:offset + T][None, None, :, :]
    sin_slice = sin[offset:offset + T][None, None, :, :]

    expected_q = (q * cos_slice) + (rotate_half(q) * sin_slice)
    expected_k = (k * cos_slice) + (rotate_half(k) * sin_slice)

    q2, k2 = apply_rope(q, k, cos, sin, position_offset=offset)
    assert torch.allclose(q2, expected_q, atol=1e-6)
    assert torch.allclose(k2, expected_k, atol=1e-6)


def test_apply_rope_raises_when_cache_too_small():
    q = torch.zeros(1, 1, 2, 4)
    cos, sin = build_rope_cache(seq_len=1, head_dim=4)
    with pytest.raises(ValueError):
        apply_rope(q, q, cos, sin, position_offset=1)
