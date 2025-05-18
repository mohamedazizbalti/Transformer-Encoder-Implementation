#Implementation of the MultiHeadAttention using my implemented AttentionHead

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
import AttentionHead
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.heads = nn.ModuleList([AttentionHead.AttentionHead(d_model, d_k, d_v) for _ in range(n_head)])
        self.WO = nn.Linear(n_head*d_v, d_model)
    def forward(self, x, mask=None):
        #x is a tensor of size (max_seq_len, d_model)
        #output is a tensor of size (max_seq_len, d_model)
        scores = torch.cat([head(x, mask) for head in self.heads], dim=1)
        scores = self.WO(scores)
        return scores
