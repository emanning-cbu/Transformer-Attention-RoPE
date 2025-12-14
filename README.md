# Transformer-Attention-RoPE

A clean, educational implementation of Transformer multi-head attention supporting causal masking and Rotary Positional Embeddings (RoPE).

This repository focuses on the core attention mechanism used in modern large language models (LLMs), designed to be readable, modular, and easy to reason about.

--------------------------------------------------
Overview
--------------------------------------------------

Transformers rely on attention to allow tokens (words, subwords, symbols) to exchange information with one another.

This repository implements the attention component commonly used in Transformer-based language models, with support for both masked (causal) and unmasked attention, as well as modern positional encoding via RoPE.

The goal of this project is clarity and understanding, not just performance.

--------------------------------------------------
What This Repository Covers
--------------------------------------------------

This repository implements only the attention layer, not a full Transformer.

Specifically, it provides:

- Multi-head self-attention
- Optional causal masking (no peeking at future tokens)
- Rotary Positional Embeddings (RoPE) applied to queries and keys
- Clean separation of concerns for attention math, masking, and positional encoding

Other Transformer components (tokenization, embeddings, feed-forward networks, full Transformer blocks) are intentionally left to other repositories.

--------------------------------------------------
Why RoPE?
--------------------------------------------------

Traditional positional encodings (such as sinusoidal embeddings) add position information before attention.

RoPE works differently.

Instead of adding position vectors to token embeddings, RoPE rotates the query and key vectors inside the attention mechanism based on token position. This allows attention to naturally capture relative distance between tokens, which scales better to longer contexts.

Key intuition:
- Sinusoidal encoding answers: “What absolute position is this token at?”
- RoPE answers: “How far apart are these two tokens?”

RoPE is widely used in modern large language models due to its effectiveness and simplicity.

--------------------------------------------------
Masked vs Unmasked Attention
--------------------------------------------------

This implementation supports two attention modes.

Masked (Causal) Attention:
Used for autoregressive generation.

Rule:
A token may only attend to tokens at the same or earlier positions.

This prevents the model from accessing future information during next-token prediction and is required for GPT-style language models.

Unmasked Attention:
Used when full context is allowed.

Rule:
Every token may attend to every other token.

This mode is common in encoder-style models or when processing complete sequences.

--------------------------------------------------
High-Level Attention Flow
--------------------------------------------------

At a conceptual level, attention works as follows:

1. Input token embeddings are projected into queries (Q), keys (K), and values (V)
2. Q and K are optionally rotated using RoPE
3. Attention scores are computed using dot products between Q and K
4. A causal mask is applied if enabled
5. Scores are normalized using softmax
6. Values (V) are mixed using the attention weights
7. Multiple attention heads are combined into a single output

This repository focuses on implementing these steps clearly and correctly.

--------------------------------------------------
Project Structure (Planned)
--------------------------------------------------

* `transformer-attention-rope/`
    * `src/`
        * `attention/`
            * `multihead_attention.py`
            * `masking.py`
            * `rope.py`
            * `config.py`
    * `tests/`
    * `README.md`

Each component is intentionally separated to keep the implementation understandable and extensible.

--------------------------------------------------
Intended Audience
--------------------------------------------------

This repository is designed for:
- Students learning how Transformers work internally
- Engineers seeking a readable reference implementation
- Employers reviewing understanding of modern LLM internals
- Anyone interested in attention mechanisms beyond black-box usage

--------------------------------------------------
Status
--------------------------------------------------

Work in progress.

This repository is under active development and will be expanded incrementally, with clarity prioritized over brevity.

--------------------------------------------------
License
--------------------------------------------------

MIT License
