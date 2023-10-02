import torch
import torch.nn as nn
import params
import math


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        # query, key, value
        self.ql = nn.Linear(params.n_embeddings, head_size, bias=False)
        self.kl = nn.Linear(params.n_embeddings, head_size, bias=False)
        self.vl = nn.Linear(params.n_embeddings, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(params.block_size,  params.block_size)))
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.kl(x)
        q = self.ql(x)
        
        weights = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(C))
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.vl(x)
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.fc = nn.Linear(head_size * num_heads, params.n_embeddings)
        self.dropout = nn.Dropout(params.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # print("Attention shape1 ", out.shape)
        out = self.fc(out)
        out = self.dropout(out)
        # print("Attention shape ", out.shape)
        return out
        
class FeedForward(nn.Module):
    def __init__(self, n_embeddings) -> None:
        super().__init__()
        self.the_net = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings * 4),
            nn.GELU(),
            nn.Linear(n_embeddings * 4, n_embeddings),
            nn.Dropout(params.dropout)
        )
    
    def forward(self, x):
        return self.the_net(x)
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self,n_embed, n_head) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.self_attenion = MultiHeadAttention(head_size, n_head)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.ff = FeedForward(n_embeddings=n_embed)

    def forward(self, x):
        # residual connection
        print(x.shape)
        x = x + self.self_attenion(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Shakespeare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(params.vocab_size, params.n_embeddings)
        self.pos_embedding = nn.Embedding(params.block_size, params.n_embeddings)
        self.tdb = nn.Sequential(*[TransformerDecoderBlock(params.n_embeddings, params.nhead) for _ in range(params.num_decoder_layers)])
        self.ln1 = nn.LayerNorm(params.n_embeddings)
        self.out = nn.Linear(params.n_embeddings, params.vocab_size)

    def forward(self, x, targets=None):
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(params.block_size, device=params.device))
        x = tok_emb + pos_emb
        x = self.tdb(x)
        logits = self.out(self.ln1(x)) # gives logits

        if targets is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(x, targets)
        
        return logits, loss
    
    def generate(self, x, max_output=100, temperature=1.0):
        for i in range(max_output):
            # crop till context
            x = x[:, -params.block_size:]
            logits, _ = self(x)
            logits = logits[:, -1, :] 
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            x = torch.cat([x, next_token], dim=-1)
        return x

