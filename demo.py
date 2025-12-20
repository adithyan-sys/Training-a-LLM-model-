import torch
import torch.nn as nn
import torch.nn.functional as F
import random 

from transformers_blocks import Block

print("Torch version:",torch.__version__)
print("CUDA available:",torch.cuda.is_available())
print("GPU name:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no")

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Adi",
    "the roads of Bangalore are busy always",
    "it is raining in Indranagar",
    "the train is late again",
    "i like having cars",
    "onam is my favorite festival",
    "diwali brings light and sweets",
    "india won cricket match"
]

corpus = [s + "<END> \n " for s in corpus]
text = " ".join(corpus)
print(text)

words = list(set(text.split()))
print(words)

vocab_size = len(words)
print(vocab_size)

word2dx = {w: i for i,w in enumerate(words)}
print(word2dx)

idx2word = {i: w for w, i in word2dx.items()} 
print("idx2word : ", idx2word) 

data = torch.tensor([word2dx[w] for w in text.split()], dtype= torch.long)
print(data)
print(len(data))

block_size = 6   #contexts length
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500

def get_batch(batch_size=16):
  ix = torch.randint(len(data) - block_size, (batch_size,))   # 49-6=43 (0-42) , 16  , so it gives 16 sentences sequences
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

class TinyGPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size,embedding_dim)

    self.position_embedding = nn.Embedding(block_size, embedding_dim)  #6,32
    self.blocks = nn.Sequential(*[Block(embedding_dim,block_size,n_heads) for _  in range(n_layers)])    # check this

    self.ln_f = nn.LayerNorm(embedding_dim)
    self.head = nn.Linear(embedding_dim, vocab_size)

  def forward(self,idx,targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding(idx)
    pos_emb = self.position_embedding(torch.arange(T , device = idx.device)) # Corrected: torch.arange
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logists = self.head(x)
    loss = None
    if targets is not None:
      B,T,C = logists.shape   #16,6,42
      loss = F.cross_entropy(logists.view(B*T,C), targets.view(B*T))
    return logists, loss
  def generate(self,idx,max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logists, _ = self(idx_cond)
      logists = logists[:, -1,:]   # raw output
      probs = F.softmax(logists, dim=-1)
      next_idx = torch.multinomial(probs,1)
      idx = torch.cat((idx, next_idx), dim=1)
      return idx
model = TinyGPT()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(epochs):
  xb,yb = get_batch()
  logists, loss = model(xb,yb)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if step % 300 == 0:
    print(f"step {step}, loss= {loss.item():.4f}")

context = torch.tensor([[word2dx["raining"]]], dtype = torch.long)
out = model.generate(context, max_new_tokens=32)
print("Generated output:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))