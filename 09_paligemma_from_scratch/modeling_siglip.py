from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size = 768, 
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels = 3,
        image_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e-6,
        attention_dropout = 0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        """
        Visual Flow

        Input: Original Image (1, 3, 224, 224)  
            ↓
        Divided into patches (16x16x3 each)
            ↓
        (14x14) = 196 patches of size 768 (16x16x3 flattened)
            ↓
        Linear/Conv projection to embedding dimension. (1, 768, 14, 14) -flatten-> (1, 768, 196) 
        ( we can't use nn.Embedding(), its impossible to create a infinitely large dict)
            ↓
        Output: 196 embedding vectors, each of dim 768 (1, 196, 768)
        [768 is chosen arbitarily but divisible by num of attention heads 
        (eg. 768 / 12 = 64 dim per head)
        Larger dim can capture more information but requires more computation and memory.]
        """
        super.__init__()

        self.hidden_size = hidden_size # size of the embedding vector of ViT
        self.intermediate_size = intermediate_size # size of linear layer in FFN
        self.num_hidden_layers = num_hidden_layers # num of layers of ViT
        self.num_attention_heads = num_attention_heads # num of attention heads in the multi-head attention
        self.num_channels = num_channels # num of channel of an image (RGB, so it's 3)
        self.image_size = image_size # input image size (it comes with 3 sizes = 224, 448, 896 | We use 224 version)
        self.patch_size = patch_size # image divided into pataches of size 16 x 16
        self.layer_norm_eps = layer_norm_eps # for numerical stability of layer norm
        self.attention_dropout = attention_dropout # 
        self.num_image_tokens = num_image_tokens # during output: num of image embedding for each image
  
  
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
                              in_channels=config.num_channels,
                              out_channels=self.embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size,
                              padding="valid") # No padding
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # (196,768)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), # (1, 196) = [[0, 1, 2, ... , 195]]
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        #
        _, _, height, width = pixel_values.shape
        # [batch_size, channels, height, width] -> [batch_size, embed_dim, num_patches_h, num_patches_w]
        # [1,3,224,224] --> [1,768,14,14]
        patch_embeds = self.patch_embedding(pixel_values) # its just 2D conv
        # [1,768,14,14] --> [1,768,196]
        embeddings = patch_embeds.flatten(2)
        # [1,768,196] --> [1,196,768]
        embeddings = embeddings.transpose(1,2)

        # [1, 196,768]
        embeddings= embeddings + self.position_embedding(self.position_ids)

        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # For parallel computation:
        # [batch_size, num_patches, embed_dim] --> [batch_size, num_patches, num_heads, head_dim] --> [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)


        # Calculate the attention using the formula Q * K^T / sqrt(d_k).
        # attn_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # Apply softmax row-wise
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Attn_weights * value_states: [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}" 
            )
        # (batch_size, num_heads, num_patches, head_dim) --> (batch_size, num_patches, num_heads, head_dim)
        attn_output = attn_output.transpose(1,2).contiguous()
        # (batch_size, num_patches, num_heads, head_dim) --> (batch_size, num_patches, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
        
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        #hidden_states: (batch_size, num_patches, embed_dim)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] --> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim] --> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        # [batch_size, num_patches, embed_dim] --> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, embed_dim)
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states
    
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size # 768

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, channels, height, weight] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(input_embeds = hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config) # ViT

    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, Channels, height, width] -> [Batch_size, Num_patches, Embed_Dim]
        # [10,3,224,224] -> [10,196,768]
        return self.vision_model(pixel_values = pixel_values)   
    

    
