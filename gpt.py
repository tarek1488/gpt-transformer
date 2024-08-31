import torch
import torch.nn as nn
import torch.nn.functional as F

#hyper parameters
batch_size = 4
block_size = 8
epochs = 5000
lr = 3e-4
eval_iter = 200
num_of_head = 6
embed_dim =  32 #vecto dim representing each token
#head size = embed_dim / num_of_head
dropout = 0.2
vocab_size = 65
n_layer = 6
#---------------------------------------------------------------------------------------------------------------------------
#input in shape Batch_size, Block_size, Embed_dim 
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
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_of_heads)])
        self.proj = nn.Linear(num_of_head * head_size, embed_dim)
        self.drop_out = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #----------- B, T, num_of_heads × head_size
        out = self.proj(out)
        out = self.drop_out(out)
        return 
    
    
class FeedForward(nn.Module):
    def __init__(self, dim):
        self.net = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)   
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        head_size  = embed_dim // n_head
        self.MultiAttention = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(embed_dim)
        self.l1 = nn.LayerNorm(embed_dim)
        self.l2 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x += self.MultiAttention(self.l1(x))
        out = x + self.ffw(self.l2(x))
        return out

class GPT_LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.token_positional_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_of_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.linear_f = nn.Linear(embed_dim, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

        
    

    
            