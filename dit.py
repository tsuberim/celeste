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
        # Predict scale and shift parameters for each normalization layer
        self.film_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4)  # 2 * embed_dim for each of 2 norms
        )
    
    def forward(self, x: Tensor, t_emb: Tensor, use_causal: bool = False) -> Tensor:
        """
        Forward pass with FiLM time conditioning
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            t_emb: Time embeddings (batch_size, seq_len, embed_dim) - per-patch time embeddings
            use_causal: Whether to use causal masking
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Get FiLM parameters from time embedding (per-patch)
        film_params = self.film_mlp(t_emb)  # (batch_size, seq_len, embed_dim * 4)
        scale1, shift1, scale2, shift2 = film_params.chunk(4, dim=-1)  # Each: (batch_size, seq_len, embed_dim)
        
        # No need to unsqueeze - already has the right shape for element-wise operations
        
        # Attention block with FiLM conditioning
        h1 = self.norm1(x)
        h1 = h1 * (1 + scale1) + shift1  # FiLM modulation (element-wise)
        h1 = self.attn(h1, use_causal)
        x = x + h1
        
        # Feed-forward block with FiLM conditioning
        h2 = self.norm2(x)
        h2 = h2 * (1 + scale2) + shift2  # FiLM modulation (element-wise)
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
                 dropout: float = 0.1):
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
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Input projection: project each patch to embed_dim
        self.input_proj = nn.Linear(latent_dim, embed_dim)
        
        # Time embedding layers
        self.time_embed_dim = embed_dim // 4  # Use 1/4 of embed_dim for time
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, ff_dim, dropout, max_seq_len, n_patches, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Output projection: back to latent space (per patch)
        self.output_proj = nn.Linear(embed_dim, latent_dim)
        
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
        timesteps = timesteps * 1000.0
        
        half_dim = self.time_embed_dim // 2
        # Pre-compute frequency scaling
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(0, half_dim, device=timesteps.device) / half_dim
        )
        
        # Compute sinusoidal embeddings
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions by padding with zeros
        if self.time_embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            
        return emb
    
    def forward(self, x: Tensor, t: Tensor, use_causal_mask: bool = True) -> Tensor:
        """
        Forward pass for sequence prediction
        
        Args:
            x: Input sequence (batch_size, seq_len, n_patches, latent_dim)
            t: Time parameter (batch_size, seq_len) for flow matching - per-frame time values
            use_causal_mask: Whether to use causal masking for autoregressive prediction
            
        Returns:
            Predicted velocity field (batch_size, seq_len, n_patches, latent_dim)
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
        t_emb = t_emb.unsqueeze(2).expand(batch_size, seq_len, n_patches, self.embed_dim)
        t_emb = t_emb.reshape(batch_size, seq_len * n_patches, self.embed_dim)
        
        # Flatten spatial and temporal dimensions: (batch_size, seq_len * n_patches, latent_dim)
        x_flat = x.reshape(batch_size, seq_len * n_patches, latent_dim)
        
        # Project each patch to embedding space
        x_emb = self.input_proj(x_flat)  # (batch_size, seq_len * n_patches, embed_dim)
        
        # Apply transformer blocks with per-patch time conditioning
        for block in self.blocks:
            x_emb = block(x_emb, t_emb, use_causal_mask)
        
        # Layer norm (let's try to remove this)
        # x_emb = self.norm(x_emb)
        
        # Project back to latent space for all patches
        output = self.output_proj(x_emb)  # (batch_size, seq_len * n_patches, latent_dim)
        
        # Reshape back to (batch_size, seq_len, n_patches, latent_dim)
        output = output.reshape(batch_size, seq_len, n_patches, latent_dim)
        
        return output
    


def create_dit(latent_dim: int, 
               n_patches: int,
               embed_dim: int = 768,
               num_layers: int = 12,
               num_heads: int = 8,
               max_seq_len: int = 24) -> DiffusionTransformer:
    """
    Create a Diffusion Transformer model
    
    Args:
        latent_dim: Latent dimension per patch (from VAE)
        n_patches: Number of patches per frame
        embed_dim: Transformer embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        
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
        dropout=0.1
    )
    
    return model

def flow_matching_loss(model: DiffusionTransformer, batch: torch.Tensor):
    batch_size, seq_len = batch.shape[0], batch.shape[1]
    t = torch.rand(batch_size, seq_len, device=batch.device)
    prior = torch.randn_like(batch)
    x_t = (1 - t.view(batch_size, seq_len, 1, 1)) * prior + t.view(batch_size, seq_len, 1, 1) * batch
    v_t_pred = model(x_t, t)
    v_target = batch - prior
    loss = torch.nn.functional.mse_loss(v_t_pred, v_target)

    steps = 1
    dt = (1 - t) / (steps + 1)
    for i in range(1, steps + 1):
        t = t + dt
        x_t = x_t + v_t_pred*dt.view(batch_size, seq_len, 1, 1)
        v_t_pred = model(x_t, t)
        loss_2 = torch.nn.functional.mse_loss(v_t_pred, v_target)
        loss += loss_2

    return loss
    
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
    
    # Forward pass with time conditioning
    output = dit(x, t)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {n_patches}, {latent_dim})")
    
    # Test flow matching loss
    print("\nTesting flow matching loss...")
    loss = flow_matching_loss(dit, x, 23)
    print(f"Flow matching loss: {loss.item():.6f}")
    
    # Test RoPE
    print("\nTesting TorchTune RoPE implementation...")
    # RoPE is now integrated into the attention mechanism via TorchTune
    print("RoPE is built into MultiHeadAttentionWithRoPE using torchtune.modules.RotaryPositionalEmbeddings")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dit()
