"""
Multi-Head Self-Attention (with optional causal masking + optional RoPE).

Input/Output shape contract:
- Input:  x of shape [batch, seq_len, d_model]
- Output: y of shape [batch, seq_len, d_model]

Optional masks:
- attention_mask (padding mask): [batch, seq_len] where 1=keep, 0=pad
- causal mask (generated internally when config.causal=True):
  prevents attending to future tokens.
"""
