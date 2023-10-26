import torch
from torch import nn 
from torch.nn import functional as f
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # tokens: (batch_size, seq_len) -> (batch_size, seq_len, dim_embed)
        
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int) -> None:
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim_embed)

        residue = x

        ## Self- Attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask = True)
        x += residue

        ## Feed Forward layer
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function

        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len) -> (batch_size, seq_len, dim_embed)
        state = self.embedding(tokens)

        for layer in self.layers:
            x = layer(state)

        # (batch_size, seq_len, dim_embed) 
        output = self.layernorm(state)

        return output
    
