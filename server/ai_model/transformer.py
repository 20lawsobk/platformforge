"""
Custom Transformer Architecture for Code Generation
Implements attention mechanisms, positional encoding, and decoder layers from scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better positional encoding.
    More effective than sinusoidal embeddings for language modeling.
    """
    
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
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


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


class TransformerBlock(nn.Module):
    """
    Single Transformer Decoder Block with Pre-Normalization.
    Consists of: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        self.attention_norm = RMSNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout, max_seq_len)
        
        self.ffn_norm = RMSNorm(embed_dim)
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
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
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
        
        for layer in self.layers:
            x = layer(x, mask, is_causal=True)
        
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
        }


def create_model(
    vocab_size: int,
    model_size: str = 'small'
) -> CodeTransformer:
    """
    Create a model with predefined configurations.
    
    Args:
        vocab_size: Size of vocabulary
        model_size: One of 'tiny', 'small', 'medium', 'large'
        
    Returns:
        Configured CodeTransformer model
    """
    configs = {
        'tiny': {'embed_dim': 128, 'num_layers': 2, 'num_heads': 2},
        'small': {'embed_dim': 256, 'num_layers': 4, 'num_heads': 4},
        'medium': {'embed_dim': 512, 'num_layers': 6, 'num_heads': 8},
        'large': {'embed_dim': 768, 'num_layers': 12, 'num_heads': 12},
        'xlarge': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16},
    }
    
    config = configs.get(model_size, configs['small'])
    
    return CodeTransformer(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
    )


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
