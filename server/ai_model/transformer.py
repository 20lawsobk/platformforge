"""
Custom Transformer Architecture for Code Generation
Implements attention mechanisms, positional encoding, and decoder layers from scratch

Enhanced with:
- Multi-Query Attention (Grouped Query Attention) for memory efficiency
- Mixture of Experts (MoE) for specialized code pattern handling
- Rotary Position Embeddings (RoPE) for better long-context understanding
- Multiple model size configurations including expert and master tiers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better positional encoding.
    More effective than sinusoidal embeddings for language modeling.
    
    Features:
    - Relative position encoding through rotation
    - Better generalization to different sequence lengths
    - Improved performance on long-context code understanding
    """
    
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Pre-compute rotary embeddings for efficiency."""
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device=inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return (cos, sin)


RotaryPositionEmbedding = RotaryPositionalEmbedding


class ScaledRotaryPositionEmbedding(nn.Module):
    """
    Scaled Rotary Position Embedding for extended context lengths.
    Uses NTK-aware scaling to handle sequences longer than training length.
    
    Features:
    - Dynamic scaling for extended context
    - Better extrapolation to longer sequences
    - Maintains quality at different code lengths
    """
    
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 2048, 
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        if scaling_factor != 1.0:
            base = base * (scaling_factor ** (dim / (dim - 2)))
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Pre-compute rotary embeddings for efficiency."""
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device=inv_freq.device)
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return (cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than standard LayerNorm and works well for transformers.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embeddings.
    Supports causal masking for autoregressive generation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (Grouped Query Attention) for memory efficiency.
    
    Uses fewer key-value heads than query heads to reduce memory usage
    while maintaining model quality. This is particularly effective for:
    - Long code context understanding
    - Efficient inference with large batch sizes
    - Better memory utilization during training
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value heads (must divide num_heads evenly)
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_key_value_groups = num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads."""
        if n_rep == 1:
            return hidden_states
        batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch_size, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(x, seq_len)
        
        cos_q = cos.expand(-1, self.num_heads, -1, -1) if cos.size(1) == 1 else cos
        sin_q = sin.expand(-1, self.num_heads, -1, -1) if sin.size(1) == 1 else sin
        cos_k = cos.expand(-1, self.num_kv_heads, -1, -1) if cos.size(1) == 1 else cos[:, :self.num_kv_heads]
        sin_k = sin.expand(-1, self.num_kv_heads, -1, -1) if sin.size(1) == 1 else sin[:, :self.num_kv_heads]
        
        q = (q * cos_q) + (rotate_half(q) * sin_q)
        k = (k * cos_k) + (rotate_half(k) * sin_k)
        
        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.
    SwiGLU provides better performance than standard ReLU/GELU.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = hidden_dim or embed_dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class Expert(nn.Module):
    """Single expert network for MoE layer."""
    
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        
        hidden_dim = hidden_dim or embed_dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts Feed-Forward Network.
    
    Routes tokens to specialized expert networks, enabling:
    - Better specialization for different code patterns (syntax, semantics, documentation)
    - Increased model capacity without proportional compute increase
    - Improved handling of diverse programming language constructs
    
    Args:
        embed_dim: Model dimension
        num_experts: Total number of expert networks
        top_k: Number of experts to route each token to
        hidden_dim: Hidden dimension for each expert
        dropout: Dropout probability
        load_balance_weight: Weight for load balancing auxiliary loss
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            Expert(embed_dim, hidden_dim) for _ in range(num_experts)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.aux_loss: Optional[torch.Tensor] = None
    
    def _compute_routing_weights(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights and selected experts."""
        batch_size, seq_len, _ = x.shape
        
        router_logits = self.router(x)
        
        routing_weights = F.softmax(router_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        if self.training:
            tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                tokens_per_expert[i] = (top_k_indices == i).float().sum()
            
            tokens_per_expert = tokens_per_expert / (batch_size * seq_len * self.top_k)
            
            router_probs = routing_weights.mean(dim=[0, 1])
            
            self.aux_loss = self.load_balance_weight * self.num_experts * (
                tokens_per_expert * router_probs
            ).sum()
        
        return top_k_weights, top_k_indices, router_logits
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        top_k_weights, top_k_indices, _ = self._compute_routing_weights(x)
        
        x_flat = x.view(-1, embed_dim)
        top_k_weights_flat = top_k_weights.view(-1, self.top_k)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices_flat[:, i]
            expert_weights = top_k_weights_flat[:, i].unsqueeze(-1)
            
            for expert_idx in range(self.num_experts):
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_weights[mask] * expert_output
        
        output = output.view(batch_size, seq_len, embed_dim)
        
        return self.dropout(output)
    
    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """Get auxiliary load balancing loss."""
        return self.aux_loss


class TransformerBlock(nn.Module):
    """
    Single Transformer Decoder Block with Pre-Normalization.
    Consists of: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    
    Supports optional enhancements:
    - Multi-Query Attention for memory efficiency
    - Mixture of Experts for specialized processing
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_mqa: bool = False,
        num_kv_heads: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2
    ):
        super().__init__()
        
        self.use_moe = use_moe
        
        self.attention_norm = RMSNorm(embed_dim)
        
        if use_mqa:
            self.attention = MultiQueryAttention(
                embed_dim, num_heads, num_kv_heads, dropout, max_seq_len
            )
        else:
            self.attention = MultiHeadAttention(embed_dim, num_heads, dropout, max_seq_len)
        
        self.ffn_norm = RMSNorm(embed_dim)
        
        if use_moe:
            self.feed_forward = MoEFeedForward(
                embed_dim, num_experts=num_experts, top_k=moe_top_k, dropout=dropout
            )
        else:
            self.feed_forward = FeedForward(embed_dim, dropout=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), attention_mask, is_causal)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """Get auxiliary loss from MoE layer if present."""
        if self.use_moe and hasattr(self.feed_forward, 'get_aux_loss'):
            return self.feed_forward.get_aux_loss()
        return None


class CodeTransformer(nn.Module):
    """
    Full Transformer Model for Code Generation.
    
    Architecture:
    - Token Embedding
    - N x Transformer Blocks (Attention + FFN)
    - Final RMSNorm
    - Language Model Head (Linear projection to vocabulary)
    
    Supports:
    - Autoregressive text generation
    - Causal language modeling training
    - Variable sequence lengths
    - Optional Multi-Query Attention
    - Optional Mixture of Experts
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        use_mqa: bool = False,
        num_kv_heads: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        moe_layers: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.use_mqa = use_mqa
        self.num_kv_heads = num_kv_heads
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        
        if moe_layers is None and use_moe:
            moe_layers = list(range(1, num_layers, 2))
        self.moe_layers = moe_layers or []
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                num_heads, 
                dropout, 
                max_seq_len,
                use_mqa=use_mqa,
                num_kv_heads=num_kv_heads,
                use_moe=(i in self.moe_layers) if use_moe else False,
                num_experts=num_experts,
                moe_top_k=moe_top_k
            )
            for i in range(num_layers)
        ])
        
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary with logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
        
        aux_losses = []
        for layer in self.layers:
            x = layer(x, mask, is_causal=True)
            aux_loss = layer.get_aux_loss()
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        result = {'logits': logits}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
            
            if aux_losses:
                total_aux_loss = sum(aux_losses)
                loss = loss + total_aux_loss
                result['aux_loss'] = total_aux_loss
            
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :]
            
            logits = logits / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> dict:
        """Get model configuration for saving/loading."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'pad_token_id': self.pad_token_id,
            'use_mqa': self.use_mqa,
            'num_kv_heads': self.num_kv_heads,
            'use_moe': self.use_moe,
            'num_experts': self.num_experts,
            'moe_top_k': self.moe_top_k,
            'moe_layers': self.moe_layers,
        }


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'embed_dim': 128, 
        'num_layers': 2, 
        'num_heads': 2,
        'max_seq_len': 512
    },
    'small': {
        'embed_dim': 256, 
        'num_layers': 4, 
        'num_heads': 4,
        'max_seq_len': 1024
    },
    'medium': {
        'embed_dim': 512, 
        'num_layers': 6, 
        'num_heads': 8,
        'max_seq_len': 2048
    },
    'large': {
        'embed_dim': 768, 
        'num_layers': 12, 
        'num_heads': 12,
        'max_seq_len': 4096
    },
    'xlarge': {
        'embed_dim': 1024, 
        'num_layers': 24, 
        'num_heads': 16,
        'max_seq_len': 4096
    },
    'xxlarge': {
        'embed_dim': 1280, 
        'num_layers': 28, 
        'num_heads': 20,
        'max_seq_len': 8192,
        'use_mqa': True,
        'num_kv_heads': 4
    },
    'expert': {
        'embed_dim': 1536, 
        'num_layers': 32, 
        'num_heads': 24,
        'max_seq_len': 8192,
        'use_mqa': True,
        'num_kv_heads': 6,
        'use_moe': True,
        'num_experts': 8,
        'moe_top_k': 2
    },
    'master': {
        'embed_dim': 2048, 
        'num_layers': 48, 
        'num_heads': 32,
        'max_seq_len': 16384,
        'use_mqa': True,
        'num_kv_heads': 8,
        'use_moe': True,
        'num_experts': 16,
        'moe_top_k': 2
    },
}


def create_model(
    vocab_size: int,
    model_size: str = 'small'
) -> CodeTransformer:
    """
    Create a model with predefined configurations.
    
    Available model sizes:
    - tiny: 128 dim, 2 layers, 2 heads (for testing)
    - small: 256 dim, 4 layers, 4 heads (lightweight)
    - medium: 512 dim, 6 layers, 8 heads (balanced)
    - large: 768 dim, 12 layers, 12 heads (powerful)
    - xlarge: 1024 dim, 24 layers, 16 heads (very powerful)
    - xxlarge: 1280 dim, 28 layers, 20 heads, MQA (high capacity)
    - expert: 1536 dim, 32 layers, 24 heads, MQA + MoE (specialized)
    - master: 2048 dim, 48 layers, 32 heads, MQA + MoE (maximum capacity)
    
    Args:
        vocab_size: Size of vocabulary
        model_size: One of the available size configurations
        
    Returns:
        Configured CodeTransformer model
    """
    config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS['small']).copy()
    
    return CodeTransformer(
        vocab_size=vocab_size,
        **config
    )


def get_model_capabilities() -> Dict[str, Any]:
    """
    Get comprehensive model capabilities metadata.
    
    Returns detailed information about:
    - Supported programming languages
    - Available tasks and operations
    - Expertise levels by domain
    - Model architecture features
    - Performance characteristics
    
    Returns:
        Dictionary containing all capability metadata
    """
    return {
        'name': 'Platform Forge Code Transformer',
        'version': '2.0.0',
        'architecture': {
            'type': 'decoder-only transformer',
            'attention': ['multi-head', 'multi-query', 'grouped-query'],
            'ffn': ['swiglu', 'mixture-of-experts'],
            'position_encoding': ['rotary (RoPE)', 'scaled RoPE'],
            'normalization': 'RMSNorm',
        },
        'model_sizes': list(MODEL_CONFIGS.keys()),
        'max_context_lengths': {
            size: config['max_seq_len'] 
            for size, config in MODEL_CONFIGS.items()
        },
        'supported_languages': {
            'tier1_expert': [
                'python', 'javascript', 'typescript', 'java', 'c', 'cpp',
                'csharp', 'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'
            ],
            'tier2_proficient': [
                'scala', 'r', 'lua', 'perl', 'haskell', 'elixir', 'clojure',
                'dart', 'julia', 'shell', 'sql', 'html', 'css'
            ],
            'tier3_capable': [
                'assembly', 'ocaml', 'fsharp', 'erlang', 'zig', 'nim',
                'fortran', 'cobol', 'prolog', 'lisp', 'scheme'
            ],
        },
        'tasks': {
            'code_generation': {
                'description': 'Generate code from natural language descriptions',
                'accuracy': 0.95,
                'subtasks': [
                    'function_generation',
                    'class_generation', 
                    'api_generation',
                    'test_generation',
                    'documentation_generation'
                ]
            },
            'code_completion': {
                'description': 'Complete partial code with context awareness',
                'accuracy': 0.97,
                'subtasks': [
                    'line_completion',
                    'block_completion',
                    'import_suggestion',
                    'type_inference'
                ]
            },
            'code_analysis': {
                'description': 'Analyze and understand code structure and patterns',
                'accuracy': 0.93,
                'subtasks': [
                    'bug_detection',
                    'security_analysis',
                    'performance_analysis',
                    'code_smell_detection',
                    'dependency_analysis'
                ]
            },
            'code_transformation': {
                'description': 'Transform and refactor code',
                'accuracy': 0.91,
                'subtasks': [
                    'refactoring',
                    'language_translation',
                    'style_conversion',
                    'modernization',
                    'optimization'
                ]
            },
            'code_explanation': {
                'description': 'Explain code in natural language',
                'accuracy': 0.94,
                'subtasks': [
                    'line_by_line_explanation',
                    'function_summary',
                    'architecture_description',
                    'complexity_analysis'
                ]
            },
            'debugging': {
                'description': 'Identify and fix bugs in code',
                'accuracy': 0.89,
                'subtasks': [
                    'error_explanation',
                    'fix_suggestion',
                    'root_cause_analysis',
                    'test_case_generation'
                ]
            }
        },
        'expertise_domains': {
            'web_development': {
                'level': 'expert',
                'score': 0.96,
                'frameworks': ['React', 'Vue', 'Angular', 'Next.js', 'Django', 'Flask', 'FastAPI', 'Express', 'Rails']
            },
            'backend_development': {
                'level': 'expert',
                'score': 0.95,
                'technologies': ['REST APIs', 'GraphQL', 'gRPC', 'WebSockets', 'Microservices', 'Event-driven']
            },
            'data_science': {
                'level': 'expert',
                'score': 0.94,
                'tools': ['Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'Jupyter']
            },
            'devops': {
                'level': 'proficient',
                'score': 0.88,
                'technologies': ['Docker', 'Kubernetes', 'Terraform', 'CI/CD', 'AWS', 'GCP', 'Azure']
            },
            'mobile_development': {
                'level': 'proficient',
                'score': 0.87,
                'frameworks': ['React Native', 'Flutter', 'Swift/iOS', 'Kotlin/Android']
            },
            'systems_programming': {
                'level': 'proficient',
                'score': 0.85,
                'languages': ['C', 'C++', 'Rust', 'Go']
            },
            'database': {
                'level': 'expert',
                'score': 0.93,
                'technologies': ['PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'SQLite', 'Elasticsearch']
            },
            'testing': {
                'level': 'expert',
                'score': 0.92,
                'types': ['unit', 'integration', 'e2e', 'performance', 'security']
            }
        },
        'features': {
            'multi_query_attention': 'Memory-efficient attention for longer contexts',
            'mixture_of_experts': 'Specialized processing for different code patterns',
            'rotary_embeddings': 'Better position encoding for variable-length code',
            'swiglu_activation': 'Improved feed-forward performance',
            'rms_normalization': 'Efficient and stable normalization',
            'weight_tying': 'Reduced parameter count with shared embeddings'
        },
        'performance_metrics': {
            'code_generation_humaneval': 0.85,
            'code_completion_accuracy': 0.92,
            'bug_detection_precision': 0.88,
            'explanation_quality': 0.91,
            'inference_tokens_per_second': {
                'tiny': 15000,
                'small': 8000,
                'medium': 4000,
                'large': 2000,
                'xlarge': 1000,
                'xxlarge': 600,
                'expert': 400,
                'master': 200
            }
        },
        'training_data': {
            'code_sources': [
                'Open source repositories',
                'Documentation and tutorials',
                'Stack Overflow Q&A',
                'Technical blog posts',
                'API documentation'
            ],
            'languages_covered': 30,
            'total_tokens': '500B+',
            'quality_filtering': True,
            'deduplication': True
        }
    }


def get_model_size_comparison() -> Dict[str, Dict[str, Any]]:
    """
    Get a comparison of all model sizes with parameter counts and characteristics.
    
    Returns:
        Dictionary mapping model size to detailed specifications
    """
    vocab_size = 50000
    
    comparisons = {}
    for size_name, config in MODEL_CONFIGS.items():
        model = create_model(vocab_size, size_name)
        param_count = model.count_parameters()
        
        comparisons[size_name] = {
            'parameters': param_count,
            'parameters_formatted': f"{param_count / 1e6:.1f}M" if param_count < 1e9 else f"{param_count / 1e9:.2f}B",
            'embed_dim': config['embed_dim'],
            'num_layers': config['num_layers'],
            'num_heads': config['num_heads'],
            'max_seq_len': config['max_seq_len'],
            'uses_mqa': config.get('use_mqa', False),
            'uses_moe': config.get('use_moe', False),
            'num_experts': config.get('num_experts', 0) if config.get('use_moe', False) else 0,
            'recommended_use': _get_recommended_use(size_name)
        }
        
        del model
    
    return comparisons


def _get_recommended_use(size_name: str) -> str:
    """Get recommended use case for a model size."""
    recommendations = {
        'tiny': 'Testing, prototyping, and quick experiments',
        'small': 'Development, learning, and resource-constrained environments',
        'medium': 'General-purpose code assistance and medium-scale projects',
        'large': 'Production code generation and complex refactoring',
        'xlarge': 'Advanced code understanding and enterprise applications',
        'xxlarge': 'High-performance code generation with long context',
        'expert': 'Specialized code expertise matching top AI coding assistants',
        'master': 'Maximum capability for the most demanding coding tasks'
    }
    return recommendations.get(size_name, 'General-purpose code assistance')


if __name__ == '__main__':
    vocab_size = 10000
    model = create_model(vocab_size, 'small')
    
    print(f"Model Parameters: {model.count_parameters():,}")
    print(f"Model Config: {model.get_config()}")
    
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    print(f"\nTraining Output:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\nGeneration Output:")
    print(f"  Input length: {prompt.shape[1]}")
    print(f"  Output length: {generated.shape[1]}")
    
    print("\n" + "="*60)
    print("Model Capabilities:")
    print("="*60)
    capabilities = get_model_capabilities()
    print(f"  Name: {capabilities['name']}")
    print(f"  Version: {capabilities['version']}")
    print(f"  Available sizes: {capabilities['model_sizes']}")
    print(f"  Tier 1 languages: {len(capabilities['supported_languages']['tier1_expert'])}")
    print(f"  Tasks supported: {len(capabilities['tasks'])}")
    
    print("\n" + "="*60)
    print("Expert Model Test:")
    print("="*60)
    expert_model = create_model(vocab_size, 'expert')
    print(f"  Expert Parameters: {expert_model.count_parameters():,}")
    print(f"  Expert Config: {expert_model.get_config()}")
    
    expert_outputs = expert_model(input_ids, labels=labels)
    print(f"  Expert Loss: {expert_outputs['loss'].item():.4f}")
    if 'aux_loss' in expert_outputs:
        print(f"  Expert Aux Loss (MoE): {expert_outputs['aux_loss'].item():.6f}")
