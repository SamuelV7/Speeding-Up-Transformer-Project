import torch
import torch.nn as nn
import torch.nn.functional as F
import params
import lang_tokenizer as lt


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads) -> None:
        super().__init__()
        self.expand = nn.Linear(params.n_embeddings, 3* params.n_embeddings, bias=False)
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.fc = nn.Linear(params.n_embeddings, params.n_embeddings)

        self.dropout_attention = nn.Dropout(params.dropout)
        self.residual_dropout = nn.Dropout(params.dropout)
        self.flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash_attention:
        #     print("Using custom attention")
        self.register_buffer('mask', torch.tril(torch.ones(params.block_size, params.block_size)).view(1, 1, params.block_size, params.block_size))
    
    def forward(self, x):
        # batch size, sequence length, embedding dimensions
        B, T, C = x.shape
        dropout = params.dropout if self.training else 0.0
        q, k, v = self.expand(x).split(params.n_embeddings, dim=-1)
        div_reshape = lambda x : x.view(B, T, params.nhead, C // params.nhead).transpose(1, 2)
        q, k, v = map(div_reshape, (q, k, v))
        # flash attention
        # this will make it faster
        # if self.flash_attention:
        #     out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, attn_mask=None, is_causal=True)
        # else:
            # manual attention
        attention = (q @ k.transpose(-2, -1)) * k.size(-1) ** -0.5
        attention = attention.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout_attention(attention)
        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.residual_dropout(self.fc(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embeddings) -> None:
        super().__init__()
        self.the_net = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings * 4),
            nn.ReLU(),
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
        # print(x.shape)
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=params.device))
        x = tok_emb + pos_emb
        x = self.tdb(x)
        logits = self.out(self.ln1(x)) # gives logits

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -params.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx