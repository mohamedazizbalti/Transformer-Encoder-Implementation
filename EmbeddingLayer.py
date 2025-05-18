#Implementation of embedding layer
import numpy as np
import torch.nn as nn
import torch


def add_positional_encoding(x, d_model):
    #input is a tensor of size (max_seq_len,d_model)
    #output is a tensor of size (max_seq_len,d_model)
    max_seq_len = x.shape[0]
    for i in range(max_seq_len):
        for j in range(d_model):
            if j%2 == 0 :
                x[i,j] += np.sin(i/(10000**(j/d_model)))
            else:
                x[i,j] += np.cos(i/(10000**(j/d_model)))
    return x

#EMBEDDING LAYER :
class Embeddor(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddor, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model)
    def forward(self, x):
        #x is a tensor of size (max_seq_len)
        embedding = self.embedding(x)
        #print("original embedding : ",embedding)
        embedding = add_positional_encoding(embedding, self.d_model)
        #print("positional embedding : ",embedding)
        return embedding
#input example
#input_vector = torch.tensor([0,1,2,3,4,5,6,7,8,9])
#print(input_vector.shape)
#embedding = Embeddor(5, 10)
#embedding.forward(input_vector)