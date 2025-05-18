# Transformer Encoder Implementation

A PyTorch implementation of the encoder component from the Transformer architecture introduced in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Architecture Overview

The encoder consists of:
- Input embedding layer
- Positional encoding
- N identical encoder layers, each containing:
  - Multi-head self-attention mechanism
  - Position-wise feed-forward network
  - Residual connections and layer normalization

![Encoder Architecture](https://miro.medium.com/v2/resize:fit:1400/1*BHzGVskWGS_3jEcYYi6miQ.png)

## Key Features

- **Multi-Head Attention**: Parallel attention heads that learn different attention patterns
- **Positional Encoding**: Sinusoidal embeddings to capture sequence order
- **Residual Connections**: Facilitate gradient flow in deep networks
- **Layer Normalization**: Stabilizes training

## Requirements

- Python 3.7+
- PyTorch 1.10+
- NumPy

## Installation

```bash
pip install torch numpy torch
