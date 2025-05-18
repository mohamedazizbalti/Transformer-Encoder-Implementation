#Implementation of Tokenizer
import numpy as np
import torch.nn as nn
import torch

#TOKENIZER :
def get_tokenize_dict(vocab):
    #input is a list of sentences and output is a dictionary of tokens
    tokenize_dict = {}
    index = 1
    for i in vocab:
        # i is sentence number i
        for j in i.split():
            #j is word number j
            if j not in tokenize_dict:
                tokenize_dict[j] = index
                index += 1
    tokenize_dict["<unk>"] = index
    tokenize_dict["<pad>"] = 0
    return tokenize_dict

#vocab = ["I am an NLPer", "I was an NLPer", "I am not an NLPer", "I was not an NLPer", "I love an NLPer"]
#print(get_tokenize_dict(vocab))

def tokenize(text, tokenize_dict):
    #input is sentence and output is a list of tokens
    array = np.zeros(len(tokenize_dict))
    index = 0
    for i in text.split():
        if i in tokenize_dict:
            array[index] = tokenize_dict[i]
        else:
            array[index] = tokenize_dict["<unk>"]
        index += 1
    for i in range(index, len(tokenize_dict)):
        array[i] = tokenize_dict["<pad>"]
    return array
#print(tokenize("I am in love NLPer", get_tokenize_dict(vocab)))