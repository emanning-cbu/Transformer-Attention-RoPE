import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split last dim into pairs (even, odd) and rotate:
    (a, b) -> (-b, a)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype=None,
):
    """
    Creates cos and sin caches used by RoPE.
    Returns:
      cos: [seq_len, head_dim]
      sin: [seq_len, head_dim]
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires head_dim to be even.")

    # freqs for half-dim (pairs)
    half = head_dim // 2
    i = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (i / half))  # [half]

    pos = torch.arange(seq_len, device=device,
                       dtype=torch.float32)  # [seq_len]
    angles = pos[:, None] * inv_freq[None, :]  # [seq_len, half]

    # expand to full dim by repeating each angle twice (for pairs)
    angles2 = torch.repeat_interleave(
        angles, repeats=2, dim=-1)  # [seq_len, head_dim]

    cos = torch.cos(angles2)
    sin = torch.sin(angles2)

    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)

    return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply RoPE to Q and K.

    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    # reshape caches so they broadcast across batch and heads
    cos = cos[None, None, :, :]  # [1,1,seq,dim]
    sin = sin[None, None, :, :]

    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
