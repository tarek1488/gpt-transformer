import torch
import torch.nn as nn
import torch.nn.functional as F

#hyper parameters
embed_dim =  32 #vecto dim representing each token
dropout = 0.2
block_size = 8
#---------------------------------------------------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key_layer = nn.Linear(embed_dim, self.head_size, bias=False)
        self.query_layer = nn.Linear(embed_dim, self.head_size)
        self.value_layer = nn.Linear(embed_dim, self.head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #(T, T)
        self.drop_out = nn.Dropout(dropout)
    def forward(self, x):
        _,T,_ = x.shape
        # x -----------> (B, T, C)
        #calculating needed data for attention
        key = self.key_layer(x) # (B, T, HEAD_SIZE)
        query = self.query_layer(x) # (B, T, HEAD_SIZE)
        value = self.query_layer(x) # (B, T, HEAD_SIZE)
        
        #calculating the attention scores
        wei =   query @ key.permute(0,2,1) * self.head_size ** 0.5
        #the masked fill for decoder attention block 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim =-1)
        wei = self.drop_out(wei)
        #making the weight aggregation of values
        out = wei @ value # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_heads, head_size):
        super().__init__()
        #sub attention layers in each layer of out transformer network
        self.heads([Head(head_size) for i  in range(num_of_heads)])
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) #----------- B, T, num_of_heads × head_size
            