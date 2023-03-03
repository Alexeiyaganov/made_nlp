
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
#         print(input.size())
        n, c, w = input.size()
        input = input.view(n, c, w).permute(0, 2, 1)
        pooled = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
#         print(pooled.shape)
        return pooled.view(n, c, w)

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        one_concat = concat_number_of_features//3
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_сonv = nn.Conv1d(hid_size, one_concat, kernel_size=3, padding=1)        
        self.title_pool = ChannelPool(3)
        self.title_dense = nn.Linear(one_concat, one_concat)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_сonv = nn.Conv1d(hid_size, one_concat, kernel_size=3, padding=1)        
        self.full_pool = ChannelPool(3)
        self.full_dense = nn.Linear(one_concat, one_concat)
                
        self.category_out = nn.Linear(n_cat_features, one_concat) # <YOUR CODE HERE>


        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_сonv(title_beg)
        title = self.title_pool(title)
        title = self.title_dense(title)# <YOUR CODE HERE>
        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_сonv(full_beg)# <YOUR CODE HERE>        
        full = self.full_pool(full)
        full = self.full_dense(full)
        
        category = self.category_out(input3)# <YOUR CODE HERE>        
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.inter_dense(concatenated)
        out = self.final_dense(out)# <YOUR CODE HERE>
        
        return out