#!/usr/bin/env python3
"""
Diffusion Transformer (DiT) for sequence prediction on encoded video frames.
Uses Rotary Positional Encodings (RoPE) and operates on latent representations.
Designed for flow matching (no noise scheduling).
"""
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict
import math
from einops import rearrange, repeat
from torchtune.modules import RotaryPositionalEmbeddings


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for improved stability"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm


class VectorQuantizer(nn.Module):
    """Vector Quantizer for discretizing actions into 8 codes"""
    
    def __init__(self, num_codes: int = 8, code_dim: int = 256):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        
        # Learnable codebook
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
    
    def forward(self, x: Tensor, compute_loss: bool = False) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            x: Input tensor (..., code_dim)
            compute_loss: Whether to compute VQ loss
        
        Returns:
            quantized: Quantized tensor with same shape as x
            indices: Codebook indices (...,)
            vq_loss: VQ loss (commitment + codebook) if compute_loss=True, else None
        """
        # Flatten input for distance computation
        input_shape = x.shape
        flat_x = x.reshape(-1, self.code_dim)
        
        # Calculate distances to codebook vectors
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook.weight ** 2, dim=1) -
            2 * torch.matmul(flat_x, self.codebook.weight.t())
        )
        
        # Get nearest codebook indices
        indices = torch.argmin(distances, dim=1)
        
        # Quantize
        quantized_flat = self.codebook(indices)
        quantized = quantized_flat.reshape(input_shape)
        
        # Compute VQ loss if requested
        vq_loss = None
        if compute_loss:
            # Commitment loss: ||sg[z_e] - e||^2
            commitment_loss = F.mse_loss(quantized.detach(), x)
            # Codebook loss: ||z_e - sg[e]||^2
            codebook_loss = F.mse_loss(quantized, x.detach())
            vq_loss = codebook_loss + 0.25 * commitment_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        indices = indices.reshape(input_shape[:-1])
        
        return quantized, indices, vq_loss


class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-head spatio-temporal attention with rotary positional embeddings using torchtune's RoPE"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 1024, n_patches: int = 220, num_frames: int = 24):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # RoPE requires even head_dim
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE, got {self.head_dim} (embed_dim={embed_dim}, num_heads={num_heads})"
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.n_patches = n_patches
        self.num_frames = num_frames
        
        # Spatial attention (within frames)
        self.spatial_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.spatial_proj = nn.Linear(embed_dim, embed_dim)
        
        # Temporal attention (across frames)
        self.temporal_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # QKNorm: Normalize queries and keys for stability
        self.spatial_q_norm = RMSNorm(self.head_dim)
        self.spatial_k_norm = RMSNorm(self.head_dim)
        self.temporal_q_norm = RMSNorm(self.head_dim)
        self.temporal_k_norm = RMSNorm(self.head_dim)
        
        # RoPE for spatial dimension (patches)
        self.spatial_rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=n_patches,
            base=10000
        )
        
        # RoPE for temporal dimension (frames)
        self.temporal_rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=num_frames,
            base=10000
        )
    
    def forward(self, x: Tensor, use_causal: bool = False) -> Tensor:
        """
        Spatio-temporal attention: spatial attention per frame, then temporal attention across frames
        
        Args:
            x: (batch_size, seq_len, embed_dim) where seq_len = num_frames * n_patches
            use_causal: Whether to use temporal causal masking
        """
        B, T, C = x.shape
        assert T % self.n_patches == 0, f"seq_len must be divisible by n_patches={self.n_patches}, got {T}"
        
        # Infer actual number of frames from input
        num_frames = T // self.n_patches
        
        # Reshape to (B, num_frames, n_patches, C) for spatial attention
        x = rearrange(x, 'b (f p) c -> b f p c', f=num_frames, p=self.n_patches)
        
        # 1. SPATIAL ATTENTION (within each frame, across patches)
        # Reshape to (B*num_frames, n_patches, C)
        x_spatial = rearrange(x, 'b f p c -> (b f) p c')
        
        # Compute Q, K, V for spatial attention
        qkv_spatial = self.spatial_qkv(x_spatial).reshape(B * num_frames, self.n_patches, 3, self.num_heads, self.head_dim)
        qkv_spatial = qkv_spatial.permute(2, 0, 3, 1, 4)  # (3, B*F, num_heads, P, head_dim)
        q_s, k_s, v_s = qkv_spatial[0], qkv_spatial[1], qkv_spatial[2]
        
        # Apply QKNorm
        q_s = rearrange(q_s, 'bf h p d -> (bf h p) d')
        k_s = rearrange(k_s, 'bf h p d -> (bf h p) d')
        q_s = self.spatial_q_norm(q_s)
        k_s = self.spatial_k_norm(k_s)
        q_s = rearrange(q_s, '(bf h p) d -> bf h p d', bf=B*num_frames, h=self.num_heads, p=self.n_patches)
        k_s = rearrange(k_s, '(bf h p) d -> bf h p d', bf=B*num_frames, h=self.num_heads, p=self.n_patches)
        
        # Apply spatial RoPE
        q_s = rearrange(q_s, 'bf h p d -> bf p h d')
        k_s = rearrange(k_s, 'bf h p d -> bf p h d')
        q_s = self.spatial_rope(q_s)
        k_s = self.spatial_rope(k_s)
        q_s = rearrange(q_s, 'bf p h d -> bf h p d')
        k_s = rearrange(k_s, 'bf p h d -> bf h p d')
        
        # Spatial attention (no causal mask - all patches in a frame attend to each other)
        out_spatial = F.scaled_dot_product_attention(
            q_s, k_s, v_s,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Project and reshape back
        out_spatial = out_spatial.transpose(1, 2).reshape(B * num_frames, self.n_patches, C)
        out_spatial = self.spatial_proj(out_spatial)
        
        # Reshape to (B, num_frames, n_patches, C)
        x = rearrange(out_spatial, '(b f) p c -> b f p c', b=B, f=num_frames)
        
        # 2. TEMPORAL ATTENTION (across frames, for each patch)
        # Reshape to (B*n_patches, num_frames, C)
        x_temporal = rearrange(x, 'b f p c -> (b p) f c')
        
        # Compute Q, K, V for temporal attention
        qkv_temporal = self.temporal_qkv(x_temporal).reshape(B * self.n_patches, num_frames, 3, self.num_heads, self.head_dim)
        qkv_temporal = qkv_temporal.permute(2, 0, 3, 1, 4)  # (3, B*P, num_heads, F, head_dim)
        q_t, k_t, v_t = qkv_temporal[0], qkv_temporal[1], qkv_temporal[2]
        
        # Apply QKNorm
        q_t = rearrange(q_t, 'bp h f d -> (bp h f) d')
        k_t = rearrange(k_t, 'bp h f d -> (bp h f) d')
        q_t = self.temporal_q_norm(q_t)
        k_t = self.temporal_k_norm(k_t)
        q_t = rearrange(q_t, '(bp h f) d -> bp h f d', bp=B*self.n_patches, h=self.num_heads, f=num_frames)
        k_t = rearrange(k_t, '(bp h f) d -> bp h f d', bp=B*self.n_patches, h=self.num_heads, f=num_frames)
        
        # Apply temporal RoPE
        q_t = rearrange(q_t, 'bp h f d -> bp f h d')
        k_t = rearrange(k_t, 'bp h f d -> bp f h d')
        q_t = self.temporal_rope(q_t)
        k_t = self.temporal_rope(k_t)
        q_t = rearrange(q_t, 'bp f h d -> bp h f d')
        k_t = rearrange(k_t, 'bp f h d -> bp h f d')
        
        # Temporal attention with optional causal mask
        out_temporal = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=use_causal,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Project and reshape back
        out_temporal = out_temporal.transpose(1, 2).reshape(B * self.n_patches, num_frames, C)
        out_temporal = self.temporal_proj(out_temporal)
        
        # Reshape back to (B, num_frames * n_patches, C)
        out = rearrange(out_temporal, '(b p) f c -> b (f p) c', b=B, p=self.n_patches)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with RoPE attention, feed-forward, and FiLM conditioning"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0, max_seq_len: int = 1024, n_patches: int = 220, num_frames: int = 24):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttentionWithRoPE(embed_dim, num_heads, dropout, max_seq_len, n_patches, num_frames)
        self.norm2 = RMSNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
        # FiLM: Feature-wise Linear Modulation for time conditioning
        self.time_film_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4)  # 2 * embed_dim for each of 2 norms
        )
        
        # FiLM: Feature-wise Linear Modulation for action conditioning
        self.action_film_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4)  # 2 * embed_dim for each of 2 norms
        )
    
    def forward(self, x: Tensor, t_emb: Tensor, action_emb: Optional[Tensor] = None, use_causal: bool = False) -> Tensor:
        """
        Forward pass with FiLM conditioning from time and actions
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            t_emb: Time embeddings (batch_size, seq_len, embed_dim) - per-patch time embeddings
            action_emb: Action embeddings (batch_size, seq_len, embed_dim) - optional per-patch action embeddings
            use_causal: Whether to use causal masking
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Get FiLM parameters from time embedding
        time_film_params = self.time_film_mlp(t_emb)  # (batch_size, seq_len, embed_dim * 4)
        t_scale1, t_shift1, t_scale2, t_shift2 = time_film_params.chunk(4, dim=-1)
        
        # Get FiLM parameters from action embedding if provided
        if action_emb is not None:
            action_film_params = self.action_film_mlp(action_emb)  # (batch_size, seq_len, embed_dim * 4)
            a_scale1, a_shift1, a_scale2, a_shift2 = action_film_params.chunk(4, dim=-1)
        else:
            # No action conditioning, use zeros
            a_scale1 = a_shift1 = a_scale2 = a_shift2 = 0
        
        # Attention block with FiLM conditioning from both time and action
        h1 = self.norm1(x)
        h1 = h1 * (1 + t_scale1 + a_scale1) + (t_shift1 + a_shift1)
        h1 = self.attn(h1, use_causal)
        x = x + h1
        
        # Feed-forward block with FiLM conditioning from both time and action
        h2 = self.norm2(x)
        h2 = h2 * (1 + t_scale2 + a_scale2) + (t_shift2 + a_shift2)
        h2 = self.ff(h2)
        x = x + h2
        
        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for sequence prediction on encoded video frames.
    Predicts the next frame given a sequence of encoded frames.
    """
    
    def __init__(self, 
                 latent_dim: int,
                 n_patches: int,
                 embed_dim: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 max_seq_len: int = 24,
                 dropout: float = 0.1,
                 action_dim: int = 16,
                 num_action_codes: int = 8):
        """
        Args:
            latent_dim: Dimension of latent space per patch
            n_patches: Number of patches per frame
            embed_dim: Transformer embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            action_dim: Dimension of latent action space
            num_action_codes: Number of VQ codes for actions
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim
        self.num_action_codes = num_action_codes
        
        # Input projection: project each patch to embed_dim
        self.input_proj = nn.Linear(latent_dim, embed_dim)
        
        # Time embedding layers
        self.time_embed_dim = embed_dim // 4  # Use 1/4 of embed_dim for time
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Vector quantizer for actions (shared codebook for input and output)
        self.action_vq = VectorQuantizer(num_codes=num_action_codes, code_dim=action_dim)
        
        # Action conditioning: project codebook embeddings to model dimension
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Encoder: project frame deltas from latent_dim to action_dim for ground truth
        self.action_encoder = nn.Linear(latent_dim, action_dim)
        
        # Learnable action tokens (like CLS tokens in ViT)
        self.prev_action_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, ff_dim, dropout, max_seq_len, n_patches + 1, max_seq_len)  # +1 for prev action token
            for _ in range(num_layers)
        ])
        
        # Output projection: back to latent space (per patch, not for action token)
        self.output_proj = nn.Linear(embed_dim, latent_dim)
        
        # Previous action prediction head: predict previous actions from action token
        self.prev_action_pred_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, action_dim)
        )
        
        # Previous action logits head: converts quantized actions to logits
        self.prev_action_logits_mlp = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, num_action_codes)
        )
        
        # Next action prediction removed
        self.next_action_pred_mlp = None
        self.next_action_logits_mlp = None
        
        # Final RMS norm
        self.norm = RMSNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Initialize output projection with moderate std for flow matching
        # Not too small (would take forever to scale up) or too large (unstable)
        torch.nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        if self.output_proj.bias is not None:
            torch.nn.init.zeros_(self.output_proj.bias)
            
        
        print(f"DiT initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small values for better convergence"""
        if isinstance(module, nn.Linear):
            # Small initialization for better learning of small residuals
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
    
    def get_timestep_embedding(self, timesteps: Tensor) -> Tensor:
        """
        Create sinusoidal timestep embeddings (improved implementation)
        
        Args:
            timesteps: Tensor of shape (batch_size,) with timestep values in [0, 1]
            
        Returns:
            Timestep embeddings of shape (batch_size, time_embed_dim)
        """
        # Scale timesteps to [0, 1000] for better numerical properties
        # Keep dtype/device consistent with inputs to avoid Half/Float mismatches under compile
        scale = torch.tensor(1000.0, device=timesteps.device, dtype=timesteps.dtype)
        timesteps = timesteps * scale
        
        half_dim = self.time_embed_dim // 2
        # Pre-compute frequency scaling
        base = torch.tensor(10000.0, device=timesteps.device, dtype=timesteps.dtype)
        arange = torch.arange(0, half_dim, device=timesteps.device, dtype=timesteps.dtype)
        freqs = torch.exp(-torch.log(base) * arange / half_dim)
        
        # Compute sinusoidal embeddings
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions by padding with zeros
        if self.time_embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            
        return emb
    
    def forward(self, x: Tensor, t: Tensor, actions: Optional[Tensor] = None, use_causal_mask: bool = True):
        """
        Forward pass for sequence prediction
        
        Args:
            x: Input sequence (batch_size, seq_len, n_patches, latent_dim)
            t: Time parameter (batch_size, seq_len) for flow matching - per-frame time values
            actions: Action indices (batch_size, seq_len) - discrete action codes [0, num_action_codes)
            use_causal_mask: Whether to use causal masking for autoregressive prediction
            
        Returns:
            Tuple of (velocity_field, prev_action_logits, next_action_logits, vq_loss)
            - velocity_field: (batch_size, seq_len, n_patches, latent_dim)
            - prev_action_logits: (batch_size, seq_len, num_codes) - logits for previous action prediction
            - next_action_logits: (batch_size, seq_len, num_codes) - logits for next action prediction
            - vq_loss: VQ loss (commitment + codebook) - includes both prev and next
        """
        batch_size, seq_len, n_patches, latent_dim = x.shape
        
        # Get time embeddings for each frame
        # Flatten to (batch_size * seq_len,) for embedding lookup
        t_flat = t.reshape(-1)  # (batch_size * seq_len,)
        t_emb = self.get_timestep_embedding(t_flat)  # (batch_size * seq_len, time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # (batch_size * seq_len, embed_dim)
        
        # Reshape to have per-frame embeddings and broadcast across patches
        # (batch_size * seq_len, embed_dim) -> (batch_size, seq_len, embed_dim) -> (batch_size, seq_len * n_patches, embed_dim)
        t_emb = t_emb.reshape(batch_size, seq_len, self.embed_dim)
        
        # Get action embeddings if provided (using shared VQ codebook)
        if actions is not None:
            # actions: (batch_size, seq_len) - discrete indices
            action_emb = self.action_vq.codebook(actions)  # (batch_size, seq_len, action_dim)
            action_emb = self.action_mlp(action_emb)  # (batch_size, seq_len, embed_dim)
        else:
            action_emb = None
        
        # Flatten spatial and temporal dimensions: (batch_size, seq_len * n_patches, latent_dim)
        x_flat = x.reshape(batch_size, seq_len * n_patches, latent_dim)
        
        # Project each patch to embedding space
        x_emb = self.input_proj(x_flat)  # (batch_size, seq_len * n_patches, embed_dim)
        
        # Reshape to (batch_size, seq_len, n_patches, embed_dim) to insert action tokens
        x_emb = x_emb.reshape(batch_size, seq_len, n_patches, self.embed_dim)
        
        # Add only prev action token
        prev_action_tokens = self.prev_action_token.expand(batch_size, seq_len, 1, self.embed_dim)
        
        # Concatenate patches with prev action token: (batch_size, seq_len, n_patches + 1, embed_dim)
        x_emb = torch.cat([x_emb, prev_action_tokens], dim=2)
        
        # Flatten back: (batch_size, seq_len * (n_patches + 1), embed_dim)
        x_emb = x_emb.reshape(batch_size, seq_len * (n_patches + 1), self.embed_dim)
        
        # Broadcast time embeddings across patches + prev action token
        t_emb_broadcast = t_emb.unsqueeze(2).expand(batch_size, seq_len, n_patches + 1, self.embed_dim)
        t_emb_broadcast = t_emb_broadcast.reshape(batch_size, seq_len * (n_patches + 1), self.embed_dim)
        
        # Broadcast action embeddings across patches + prev action token if provided
        if action_emb is not None:
            action_emb_broadcast = action_emb.unsqueeze(2).expand(batch_size, seq_len, n_patches + 1, self.embed_dim)
            action_emb_broadcast = action_emb_broadcast.reshape(batch_size, seq_len * (n_patches + 1), self.embed_dim)
        else:
            action_emb_broadcast = None
        
        # Apply transformer blocks with separate time and action conditioning
        for block in self.blocks:
            x_emb = block(x_emb, t_emb_broadcast, action_emb_broadcast, use_causal_mask)
        
        # Layer norm (let's try to remove this)
        # x_emb = self.norm(x_emb)
        
        # Reshape to separate patches and action tokens
        # (batch_size, seq_len * (n_patches + 1), embed_dim) -> (batch_size, seq_len, n_patches + 1, embed_dim)
        x_emb = x_emb.reshape(batch_size, seq_len, n_patches + 1, self.embed_dim)
        
        # Split into regular patches and action tokens
        x_patches = x_emb[:, :, :n_patches, :]  # (batch_size, seq_len, n_patches, embed_dim)
        prev_action_token_embs = x_emb[:, :, n_patches, :]  # (batch_size, seq_len, embed_dim)
        
        # Project patches back to latent space
        x_patches_flat = x_patches.reshape(batch_size, seq_len * n_patches, self.embed_dim)
        output = self.output_proj(x_patches_flat)  # (batch_size, seq_len * n_patches, latent_dim)
        
        # Reshape back to (batch_size, seq_len, n_patches, latent_dim)
        output = output.reshape(batch_size, seq_len, n_patches, latent_dim)
        
        # Predict previous actions from prev action token embeddings
        prev_actions_continuous = self.prev_action_pred_mlp(prev_action_token_embs)  # (batch_size, seq_len, action_dim)
        
        # Quantize previous actions to discrete codes (always compute VQ loss)
        prev_actions_quantized, _, prev_vq_loss = self.action_vq(prev_actions_continuous, compute_loss=True)
        
        # Convert quantized previous actions to logits
        prev_logits = self.prev_action_logits_mlp(prev_actions_quantized)  # (batch_size, seq_len, num_action_codes)
        
        # VQ loss (only previous)
        vq_loss = prev_vq_loss
        
        return output, prev_logits, vq_loss
    


def create_dit(latent_dim: int, 
               n_patches: int,
               embed_dim: int = 768,
               num_layers: int = 12,
               num_heads: int = 8,
               max_seq_len: int = 24,
               action_dim: int = 16,
               num_action_codes: int = 8) -> DiffusionTransformer:
    """
    Create a Diffusion Transformer model
    
    Args:
        latent_dim: Latent dimension per patch (from VAE)
        n_patches: Number of patches per frame
        embed_dim: Transformer embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        action_dim: Dimension of latent action space
        num_action_codes: Number of VQ codes for actions
        
    Returns:
        DiffusionTransformer model
    """
    model = DiffusionTransformer(
        latent_dim=latent_dim,
        n_patches=n_patches,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=embed_dim * 4,  # Standard 4x expansion
        max_seq_len=max_seq_len,
        dropout=0.1,
        action_dim=action_dim,
        num_action_codes=num_action_codes
    )
    
    return model
    
def test_dit():
    """Test the DiT model with dummy data"""
    print("Testing Diffusion Transformer...")
    
    # Model parameters (matching encoded dataset format)
    latent_dim = 48  # From VAE
    n_patches = 220
    
    seq_len = 8
    batch_size = 2
    
    # Create model
    dit = create_dit(
        latent_dim=latent_dim,
        n_patches=n_patches,
        embed_dim=512,
        num_layers=6,  # Smaller for testing
        num_heads=8,
        max_seq_len=24
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, n_patches, latent_dim)
    t = torch.rand(batch_size, seq_len)  # Random time values per frame
    print(f"Input shape: {x.shape}")
    print(f"Time shape: {t.shape}")
    
    # Forward pass with time conditioning (no actions)
    output, prev_action_logits, next_action_logits, vq_loss = dit(x, t)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {n_patches}, {latent_dim})")
    print(f"Previous action logits shape: {prev_action_logits.shape}")
    print(f"Next action logits shape: {next_action_logits.shape}")
    print(f"Expected action logits shape: ({batch_size}, {seq_len}, 8)")
    print(f"VQ loss: {vq_loss.item():.6f}")
    
    # Forward pass with action conditioning
    print("\nTesting with action conditioning...")
    actions = torch.randint(0, 8, (batch_size, seq_len))
    output_cond, prev_logits_cond, next_logits_cond, _ = dit(x, t, actions)
    print(f"Output with actions shape: {output_cond.shape}")
    print(f"Previous action logits with conditioning shape: {prev_logits_cond.shape}")
    print(f"Next action logits with conditioning shape: {next_logits_cond.shape}")
    
    # Test RoPE
    print("\nTesting TorchTune RoPE implementation...")
    # RoPE is now integrated into the attention mechanism via TorchTune
    print("RoPE is built into MultiHeadAttentionWithRoPE using torchtune.modules.RotaryPositionalEmbeddings")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dit()
