import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    """
    Caches key and value states for efficient autoregressive generation.
    
    During text generation, instead of recomputing attention for the entire sequence,
    we cache previous key/value pairs and only compute attention for the new token.
    This dramatically speeds up generation by avoiding redundant computations.
    """
    def __init__(self) -> None:
        # Lists to store cached keys and values for each transformer layer
        self.key_cache: List[torch.Tensor] = []    # Stores key states per layer
        self.value_cache: List[torch.Tensor] = []  # Stores value states per layer

    def num_items(self) -> int:
        """Returns the current sequence length in the cache (number of tokens processed)."""
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is (batch_size, num_heads_KV, seq_len, head_dim)
            # seq_len dimension tells us how many tokens we've processed
            return self.key_cache[0].shape[-2]
    
    def update(
        self,
        key_states: torch.Tensor,      # New key states to add
        value_states: torch.Tensor,    # New value states to add
        layer_idx: int,                # Which transformer layer these states belong to
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key-value pairs and returns concatenated states.
        
        During generation:
        1. First token: Initialize cache with first token's k/v
        2. Subsequent tokens: Concatenate new k/v with cached k/v
        3. Return full sequence for attention computation
        """
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: (batch_size, num_heads_KV, seq_len, head_dim)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim = 256,
        max_position_embeddings = 8192,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        attention_bias = False,
        attention_dropout = 0.0,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads # For query
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads # for key and value
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

    def __init__(
        self,
        vision_config = None,
        text_config = None,
        ignore_index = -100,
        image_token_index = 256000,
        vocab_size = 257152,
        projection_dim = 2048,
        hidden_size = 2048,
        pad_token_id = None,
        **kwargs,
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.image_token_index = image_token_index

        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size 

        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = vision_config
        self.vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) used in Gemma.
    A variant of Layer Normalization that only uses root mean square for normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        
        # Small constant for numerical stability
        self.eps = eps
        # Learnable scale parameter (gamma parameter) initialized to zeros
        # Shape: (dim,)
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # Calculate RMS normalization:
        # 1. Square all elements
        # 2. Take mean across last dimension (keeping dims for broadcasting)
        # 3. Add epsilon for numerical stability
        # 4. Take inverse square root
        # 5. Multiply input by this scaling factor
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # Ensure normalization happens in float32 for numerical stability
        output = self._norm(x.float())

        # Scale the normalized values with a learned parameter
        # Note: Gemma uses (1 + weight) scaling factor instead of just weight
        # This initialization ensures the layer starts as identity function
        # llamma does x.to(float16) * w whilst Gemma is (x * (1+w)).to(float16)
        output = output * (1.0 + self.weight.float())
        
        # Convert back to input dtype (e.g., float16 or bfloat16)
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) for the Gemma model.
    
    RoPE encodes position information by rotating vectors in a frequency-dependent manner.
    Each dimension pair is rotated by a different frequency, allowing the model to 
    capture both absolute and relative positions effectively.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim  # Dimension of the embeddings (set to head_dim)
        self.max_position_embeddings = max_position_embeddings  # Maximum sequence length
        self.base = base  # Base for the geometric sequence of frequencies

        # Calculate frequencies for each dimension pair using a geometric sequence
        # Formula: θᵢ = base^(-2i/dim) where i = 0,1,2,...,dim//2
        # Lower i -> Higher frequency -> Captures fine-grained relative positions
        # Higher i -> Lower frequency -> Captures long-range dependencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        # Register buffer (not a parameter) for the inverse frequencies
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Compute the cosine and sine components for rotary embeddings.
        
        Args:
            x: Input tensor (batch_size, num_attention_heads, seq_len, head_size)
            position_ids: Position indices for the sequence
            seq_len: Optional sequence length
        """
        # Move inverse frequencies to the same device as input
        self.inv_freq.to(x.device)
        
        # Expand inverse frequencies for batch processing
        # Shape: (1, dim//2, 1) -> (batch_size, dim//2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        
        # Prepare position IDs for matrix multiplication
        # Shape: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].float()

        # Handle device-specific computations (especially for Apple M1/M2 chips)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        # Compute rotation angles with higher precision (float32)
        with torch.autocast(device_type=device_type, enabled=False):
            # Compute mθ for each position and frequency pair
            # Shape: (batch_size, dim//2, 1) @ (batch_size, 1, seq_len) -> (batch_size, dim//2, seq_len)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            
            # Duplicate frequencies for both sin and cos components
            # Shape: (batch_size, seq_len, dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Compute rotation matrices components
            cos = emb.cos()  # For rotation component that preserves norm
            sin = emb.sin()  # For rotation component that changes direction

        # Return components in the same dtype as input
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """
    Helper function that performs rotation on half of the features.
    Used as part of the rotary position embedding (RoPE) calculation.
    
    For a vector [x1, x2, x3, x4], creates [-x2, x1, -x4, x3].
    This rotation helps encode relative positions in the attention mechanism.
    """
    # Split the input tensor along the last dimension (feature dimension)
    x1 = x[..., : x.shape[-1] // 2]  # First half:  [x1, x3]
    x2 = x[..., x.shape[-1] // 2 :]  # Second half: [x2, x4]
    # Concatenate [-x2, x1] to create the rotated vector
    return torch.cat((-x2, x1), dim=-1)  # Returns: [-x2, x1, -x4, x3]

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies rotary positional embeddings to the query (q) and key (k) tensors.
    
    RoPE allows the model to learn relative positions by encoding them through
    rotation in vector space. This creates position-dependent query and key vectors
    that maintain translation equivariance.

    Args:
        q, k: Query and key tensors
        cos, sin: Precomputed cosine and sine values for rotary embeddings
        unsqueeze_dim: Dimension to add for proper broadcasting (usually head dim)
    """
    # Add head dimension to cos and sin for proper broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)  # Shape: (..., 1, seq_len, head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)  # Shape: (..., 1, seq_len, head_dim)

    # Apply rotary embeddings using the formula from the RoPE paper:
    # [cos_θ * x - sin_θ * rotate_half(x), cos_θ * y - sin_θ * rotate_half(y)]
    q_embed = (q * cos) + (rotate_half(q) * sin)  # Rotate query vectors
    k_embed = (k * cos) + (rotate_half(k) * sin)  # Rotate key vectors
    return q_embed, k_embed

class GemmaMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in Gemma transformer blocks.
    Implements a gated feed-forward network that selectively controls information flow.

    The gating mechanism works like a learnable filter/dimmer switch:
    1. Gate path (GELU activation) produces values between 0-1 acting as "switches"
    2. These switches control how much of each feature from the up-projection passes through
    3. Values close to 0 block features, close to 1 allow features, in-between partial passage
    4. The network learns which features are important for different inputs

    The expansion to higher dimension before gating is crucial because:
    1. It provides more "working space" for learning complex feature relationships
    2. Creates an information bottleneck (expand→filter→compress) forcing the model to be selective
    3. Follows empirically successful patterns from other architectures
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size          # Size of input/output tensors
        self.intermediate_size = config.intermediate_size  # Size of internal projections
        
        # Three parallel projections without bias terms:
        # 1. Gate projection: Creates learnable "switches" to control information flow
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 2. Up projection: Expands to higher dim for more expressive feature relationships
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 3. Down projection: Compresses filtered information back to original dimension
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Gated feed-forward computation:
        # 1. Expand input to higher dim for better feature processing (up_proj)
        # 2. Create feature-wise gates between 0-1 (gate_proj + GELU)
        # 3. Filter expanded features using gates (multiplication)
        # 4. Compress back to original dimension (down_proj)
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x)
        )

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Helper function to repeat key/value states to match number of query heads.
    Used for grouped query attention where we have fewer k/v heads than query heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Add new dimension and expand k/v heads to match number of query heads
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # Reshape to merge the k/v heads with their repetitions
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta,
        )

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #
        # (batch_size, seq_len, hidden_size)
        bsz, q_len, _ = hidden_states.size()

        # Project input hidden states to query, key and value spaces
        # Note: num_heads_Q >= num_heads_KV (grouped query attention)
         # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)  # Q projection: hidden_size -> num_heads_Q * head_dim
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)    # K projection: hidden_size -> num_heads_KV * head_dim
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)  # V projection: hidden_size -> num_heads_KV * head_dim
        
        # Reshape projections to separate head dimension and transpose for attention computation
        # Shape: (batch, seq_len, num_heads * head_dim) -> (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        # (batch_size, num_heads_KV, seq_len, head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).tranpose(1,2)
        # (batch_size, num_heads_KV, seq_len, head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)


        # # Apply rotary positional embeddings (RoPE) to queries and key
        # (batch_size, seq_len, head_dim), (batch_size, seq_len, head_dim)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # (batch_size, num_heads_Q, seq_len, head_dim), (batch_size, num_heads_KV, seq_len, head_dim)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update key/value states with cached values during generation
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # For grouped query attention: repeat k/v heads to match number of query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores (scaled dot-product attention)
        # (batch, num_heads_Q, seq_len_Q, head_dim) @ (batch, num_heads_KV, head_dim, seq_len_KV) 
        # -> (batch, num_heads_Q, seq_len_Q, seq_len_KV)
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        # Add attention mask (e.g., causal mask for decoder-only architecture)
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Normalize attention weights with softmax and apply dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout
        attn_weights = nn.function.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute weighted sum of values using attention weights
        # (batch, num_heads_Q, seq_len_Q, seq_len_KV) @ (batch, num_heads_KV, seq_len_KV, head_dim)
        # -> (batch, num_heads_Q, seq_len_Q, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        # Validate output shape
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Reshape attention output to combine heads and project back to hidden_size
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1,2).contiguous()  
        attn_output = attn_output.view(bsz, q_len, -1)        # (batch, seq_len_Q, num_heads_Q * head_dim)
        attn_output = self.o_proj(attn_output)                # (batch, seq_len, hidden_size)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config = config, layer_idx = layer_idx)
        
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps= config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        # (batch_size, seq_len, hidden_size)
        hidden_states = self.input_layernorm(hidden_states)

        # (batch_size, seq_len, hidden_size)
        hidden_states, _ = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache,
        )

        # (batch_size, seq_len, hidden_size)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        
        # Store configuration and basic parameters
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embedding layer: converts token IDs to vectors
        # Shape: (vocab_size, hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Stack of transformer decoder layers that process the embeddings
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization (RMSNorm variant used in Gemma)
        self.norm = GemmaRMSNorm(config.hidden_size, ep=config.rms_norm_eps)

    def get_input_embeddings(self):
        # Helper method to access embedding layer (used for weight tying)
        return self.embed_tokens
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,      # Masks padding tokens
        position_ids: Optional[torch.LongTensor] = None,    # Used for rotary position encoding
        inputs_embeds: Optional[torch.FloatTensor] = None,  # Pre-computed embeddings
        kv_cache: Optional[KVCache] = None,                 # Cache for faster generation
    ) -> torch.FloatTensor:
        # Use pre-computed embeddings as initial hidden states
        # (batch_size, seq_len, hidden_size)
        hidden_states = inputs_embeds
        
        # Scale embeddings by sqrt(hidden_size) for better training dynamics
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # Pass through each transformer decoder layer sequentially
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache,
            )       

        # Final layer normalization
        hidden_states = self.norm(hidden_states)

        # Return the transformed sequence representations
        # Shape: (batch_size, seq_len, hidden_size)
        return hidden_states

class GemmaForCausalLM(nn.Module):
    """
    Gemma model for causal (auto-regressive) language modeling.
    Converts token embeddings to next-token predictions.

    Causal LM = Transformer model + LM head (embeddings ---> logits)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config 
        # Core transformer model that processes token embeddings
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        # Linear layer to project hidden states to vocabulary distribution
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        # Returns the embedding layer that converts token IDs to vectors
        return self.model.embed_tokens
    
    def tie_weights(self):
        # Share 'weights' between 'input embeddings' and 'output projection (logits)'
        # This reduces parameters and ensures consistency
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,      # Masks padding tokens
        position_ids: Optional[torch.LongTensor] = None,    # Position encoding for RoPE
        inputs_embeds: Optional[torch.FloatTensor] = None,  # Token embeddings
        kv_cache: Optional[KVCache] = None,                 # Cache for faster generation
    ) -> Tuple:
        
        # Process sequence through transformer layers
        # Shape: (batch_size, seq_len, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,    # Controls which tokens can attend to each other
            position_ids=position_ids,        # Used for rotary position encoding
            inputs_embeds=inputs_embeds,      # Token embeddings to process
            kv_cache=kv_cache,               # Cached key/values for efficient generation
        )

        # Project hidden states to vocabulary distribution
        hidden_states = outputs
        # Convert hidden states to logits over vocabulary
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(hidden_states)
        # Ensure logits are in float32 for numerical stability
        logits = logits.float()

        # Prepare return dictionary
        return_data = {
            "logits": logits,  # Probability distribution over next tokens
        }

        if kv_cache is not None:
            # Include updated key-value cache for next generation step
            return_data["kv_cache"] = kv_cache

        return return_data 

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias= True) 

    def forward(self, image_features):
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, projection_dim)
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape # same shape as text tokens
        batch_size, sequence_length = input_ids.shape # input_ids tells us how many tokens do we have
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: (batch_size, seq_len, hidden_size)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Shape: (batch_size, seq_len). True for text tokens
        #
        # input_ids: (image_tokens, bos_token, text_token, \n)
        # eg. input_ids: (500, 500, 500, 500, 500, 1, 70, 71, 75, 76, 2)
        # text_mask: (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: (batch_size, seq_len). True for image tokens
        # image_mask: (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        image_mask = input_ids == self.config.image_token_index
        # Shape: (batch_size, seq_len). True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where [add batch_size and embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        # So far, we have now final embedding ready as input for LLM.


        ### CREATE THE ATTENTION MASK ###

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min 
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device 
            )
        else: 
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len 

            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens
            # This only works when we have no padding
            causal_mask =  torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device   
            )
        
        # Add the head dimension
        # (batch_size, q_len, kv_len) --> (batch_size, num_heads_q, q_len, kv_len)
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0: # Prefill part
            # The position of the query is just the last position
            #
            # (256_image_token, 3_text_token)
            # example: (0, 1, ..., 255, 256, 257, 258)
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            # This is to apply rotary_positional_encoding to each token

        else: # Generation part
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    	

    def forward(  
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Step 1: Text Processing
        # Convert token IDs to embeddings
        # Input:  (batch_size, seq_len) token IDs: <image> <bos> <text> <\n>
        # Output: (batch_size, seq_len, hidden_size) embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Step 2: Image Processing
        # Convert image patches to embeddings and project to text space
        # Vision Tower:
        #   Input:  (batch_size, channels, height, width)
        #   Output: (batch_size, num_patches, embed_dim)
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # Project to match text embedding space
        image_features = self.multi_modal_projector(selected_image_feature)

        # Step 3: Multimodal Fusion
        # Combine text and image embeddings for joint processing
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache
        )

        # Final language model processing
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs


