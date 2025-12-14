import torch


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Boolean mask [seq_len, seq_len].
    True = blocked (future tokens), False = allowed.
    """
    # Start with an upper-triangular matrix above the diagonal:
    # positions where j > i are True (blocked)
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )