#Implementation of a transformer block

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import FeedForward
import EmbeddingLayer
import MultiHeadAttention
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_head, dropout):
        super(TransformerBlock, self).__init__()
        self.feed_forward = FeedForward.FeedForward(d_model, d_ff)
        self.attention = MultiHeadAttention.MultiHeadAttention(d_model, d_k, d_v,n_head=n_head)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.WO = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        #x is input of size (max_seq_len,d_model)
        #output is a tensor of size (max_seq_len,d_model)
        attention = self.attention(x, mask)
        x = self.WO(attention)
        attention = self.dropout(attention)
        x = self.layer_norm1(x + attention)
        feed_forward = self.feed_forward(x)
        feed_forward = self.dropout(feed_forward)
        x = self.layer_norm2(x + feed_forward)
        print("Passed through Transformer Block")
        return x

#input example
#input_vector = torch.rand(10, 5)
#block = TransformerBlock(5, 7, 5, 7, 5, 0.1)
#block.forward(input_vector)