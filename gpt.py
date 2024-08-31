import torch
import torch.nn as nn
import torch.nn.functional as F

#hyper parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
block_size = 256
epochs = 5000
lr = 3e-4
eval_iter = 200
num_of_head = 6
eval_interval = 500
embed_dim =  384 #vecto dim representing each token
#head size = embed_dim / num_of_head
dropout = 0.2
n_layer = 6
#---------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as File:
    text = File.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c : i for i, c in enumerate(chars)}
itos = {i: c for c , i in stoi.items()}

#defining the encoding and decoding functions
def encode(s):
    code = [stoi[i] for i in s]
    return code
def decode(code):
    return ''.join([itos[i] for i in code])

#encoding the whole data
data = torch.tensor(encode(text), dtype=torch.long)

#splitting dataset
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#defining the function to get batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#---------------------------------------------------------------------------------------------------------------------------
#input in shape Batch_size, Block_size, Embed_dim 
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key_layer = nn.Linear(embed_dim, head_size, bias=False)
        self.query_layer = nn.Linear(embed_dim, head_size, bias=False)
        self.value_layer = nn.Linear(embed_dim, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #(T, T)
        self.drop_out = nn.Dropout(dropout)
    
    def forward(self, x):
        _,T,_ = x.shape
        # x -----------> (B, T, C)
        #calculating needed data for attention
        key = self.key_layer(x) # (B, T, HEAD_SIZE)
        query = self.query_layer(x) # (B, T, HEAD_SIZE)
        value = self.value_layer(x) # (B, T, HEAD_SIZE)
        
        #calculating the attention scores
        wei = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5
        #wei =   query @ key.permute(0,2,1) * self.head_size ** 0.5
        #the masked fill for decoder attention block 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim =-1)
        wei = self.drop_out(wei)
        #making the weight aggregation of values
        out = wei @ value # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_head, head_size):
        super().__init__()
        #sub attention layers in each layer of out transformer network
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_of_head)])
        self.proj = nn.Linear(num_of_head * head_size, embed_dim)
        self.drop_out = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #----------- B, T, num_of_heads × head_size
        out = self.proj(out)
        out = self.drop_out(out)
        return out
    
    
class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,  4*dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),   
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
        x = x + self.MultiAttention(self.l1(x))
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
        pos_emb = self.token_positional_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.linear_f(x) # (B,T,vocab_size)

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

        
    

model = GPT_LLM()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(epochs):

    # # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == epochs - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
            