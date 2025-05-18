#Attention Head Implementation
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(AttentionHead, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
    def forward(self,x,mask=None):
        #x is a tensor of size (max_seq_len, d_model)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.matmul(q, torch.transpose(k, 0, 1)) / np.sqrt(self.d_k)
        if mask is not None:
            # Assume mask is of shape (seq_len, seq_len), with 0 where masked
            scores = scores.masked_fill(mask == 0, float('-inf'))
        #print(scores.shape)
        scores = torch.softmax(scores, dim=1)
        scores = torch.matmul(scores,v)
        #print(scores.shape)
        return scores

#input example
#input_vector = torch.rand(10, 5)
#attention_head = AttentionHead(5, 5, 7)
#attention_head.forward(input_vector)