import torch

from src.attention.rope import build_rope_cache, apply_rope


def main():
    # tiny example: "I love pizza"
    B, H, T, D = 1, 1, 3, 4  # batch, heads, seq_len, head_dim

    q = torch.tensor([[
        [[1.0,  0.0,  0.5, -0.5],   # pos 0: "I"
         [0.2,  0.8, -0.3,  0.1],   # pos 1: "love"
         [0.9, -0.4,  0.7,  0.2]]   # pos 2: "pizza"
    ]])

    k = q.clone()  # for debugging, keep it simple

    cos, sin = build_rope_cache(
        seq_len=T, head_dim=D, base=10.0, device=q.device, dtype=q.dtype)
    q2, k2 = apply_rope(q, k, cos, sin)

    print("cos:\n", cos)
    print("sin:\n", sin)
    print("q2:\n", q2)


if __name__ == "__main__":
    main()
