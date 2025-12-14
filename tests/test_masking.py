import torch
from src.attention.masking import causal_mask


def test_causal_mask_shape_and_values():
    m = causal_mask(4)
    assert m.shape == (4, 4)

    # Above diagonal should be True (blocked)
    assert m[0, 1] == True
    assert m[0, 3] == True
    assert m[2, 3] == True

    # Diagonal and below should be False (allowed)
    assert m[0, 0] == False
    assert m[2, 1] == False
    assert m[3, 0] == False
