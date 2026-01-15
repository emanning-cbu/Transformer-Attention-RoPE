import torch
from src.attention.rope import build_rope_cache, apply_rope


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
