#implementation of Encoder Part of Transformer Architecture
#by Azizz

import torch
import torch.nn as nn
import EmbeddingLayer
import TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_head, dropout, vocab, max_seq_len,num_layers):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.dropout = dropout
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.embedding = EmbeddingLayer.Embeddor(d_model, vocab)
        self.transformerBlocks = nn.ModuleList([TransformerBlock.TransformerBlock(d_model, d_ff, d_k, d_v, n_head, dropout) for _ in range(num_layers)])

    def forward(self, x):
        # x is a tensor of size (max_seq_len, vocab)
        x = self.embedding.forward(x)
        #print(x.shape)
        for i in range(self.num_layers):
            x = self.transformerBlocks[i](x)
        print(x.shape)
        return x

#input example
#generate tensor of size (max_seq_len, vocab)
input_vector = torch.tensor([0,1,2,3,4,5,6,7,8,9])
Encoder = Transformer(d_model=256, d_ff=7, d_k=5, d_v=7, n_head=5, dropout=0.1, vocab=10, max_seq_len=10, num_layers=6)
Encoder.forward(input_vector)