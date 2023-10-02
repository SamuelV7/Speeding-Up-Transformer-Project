import torch
import torch.nn as nn
import params
import math


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        # query, key, value
        self.ql = nn.Linear(head_size, bias=False)
        self.kl = nn.Linear(head_size, bias=False)
        self.vl = nn.Linear(head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(params.block_size,  params.block_size)))
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.kl(x)
        q = self.ql(x)
        
        weights = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(C))
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(params.dropout)
        v = self.vl(x)
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.fc = nn.Linear(head_size * num_heads, head_size)
        self.dropout = nn.Dropout(params.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.fc(out)
        out = self.dropout(out)
        return out
        
class FeedForward(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.the_net = nn.Sequential(
            nn.Linear(params.d_model, params.d_model * 4),
            nn.GELU(),
            nn.Linear(params.d_model * 4, params.d_model),
            nn.Dropout(params.dropout)
        )
    
    def forward(self, x):
        return self.the_net(x)
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        head_size = params.d_model // params.nhead
        self.self_attenion = MultiHeadAttention(head_size, params.nhead)
        self.norm1 = nn.LayerNorm(params.d_model)
        self.norm2 = nn.LayerNorm(params.d_model)
        self.ff = FeedForward(d_model=params.d_model)

    def forward(self, x):
        # residual connection
        x = x + self.self_attenion(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Shakespeare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(params.vocab_size, params.d_model)
        self.pos_embedding = nn.Embedding(params.block_size, params.d_model)
        self.tdb = nn.Sequential(*[TransformerDecoderBlock() for _ in range(params.num_decoder_layers)])
        self.ln1 = nn.LayerNorm(params.d_model)
        self.out = nn.Linear(params.d_model, params.vocab_size)

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

