import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split last dim into pairs (even, odd) and rotate in the complex plane:
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


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
):
    """
    Apply RoPE to Q and K.

    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [cache_seq_len, head_dim]
    position_offset: start position inside the cache (useful for KV-caching).
    """
    if position_offset < 0:
        raise ValueError("position_offset must be >= 0")

    seq_len = q.size(-2)
    end_pos = position_offset + seq_len

    if cos.shape != sin.shape:
        raise ValueError("cos and sin caches must have the same shape")
    if end_pos > cos.size(0):
        raise ValueError(
            f"cos/sin cache too small for sequence length {seq_len} starting at {position_offset}"
        )

    # slice caches so they align with the tokens we're rotating
    cos_slice = cos[position_offset:end_pos]
    sin_slice = sin[position_offset:end_pos]

    # reshape so caches broadcast across batch and heads and match q/k dtype/device
    cos_broadcast = cos_slice.to(device=q.device, dtype=q.dtype)[None, None, :, :]
    sin_broadcast = sin_slice.to(device=q.device, dtype=q.dtype)[None, None, :, :]

    q_out = (q * cos_broadcast) + (rotate_half(q) * sin_broadcast)
    k_out = (k * cos_broadcast) + (rotate_half(k) * sin_broadcast)
    return q_out, k_out
