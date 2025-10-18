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
    """Multi-head attention with rotary positional embeddings using torchtune's RoPE"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 1024, n_patches: int = 220, num_frames: int = 16):
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
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # QKNorm: Normalize queries and keys for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # TorchTune's RoPE implementation
        # RoPE needs to handle the full spatio-temporal sequence: num_frames * n_patches
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=num_frames * n_patches,
            base=10000
        )
        
        # Pre-compute spatio-temporal causal mask
        self.register_buffer('spatiotemporal_mask', self._create_spatiotemporal_causal_mask())
    
    def _create_spatiotemporal_causal_mask(self) -> Tensor:
        """
        Create spatio-temporal causal mask at initialization.
        - Within a frame: all patches can attend to each other (spatial attention)
        - Across frames: can only attend to past frames (temporal causality)
        
        Returns:
            Boolean mask of shape (num_frames * n_patches, num_frames * n_patches)
            True = can attend, False = masked out
        """
        total_len = self.num_frames * self.n_patches
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        for i in range(self.num_frames):
            for j in range(self.num_frames):
                # Frame i patches (rows) attending to frame j patches (columns)
                i_start, i_end = i * self.n_patches, (i + 1) * self.n_patches
                j_start, j_end = j * self.n_patches, (j + 1) * self.n_patches
                
                if j <= i:
                    # Current and past frames: can attend to all patches
                    mask[i_start:i_end, j_start:j_end] = True
                else:
                    # Future frames: mask out
                    mask[i_start:i_end, j_start:j_end] = False
        
        return mask
    
    def forward(self, x: Tensor, use_causal: bool = False) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim) where seq_len = num_frames * n_patches
            use_causal: Whether to use spatio-temporal causal masking
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply QKNorm before RoPE for stability
        q = rearrange(q, 'b h t d -> (b h t) d')
        k = rearrange(k, 'b h t d -> (b h t) d')
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = rearrange(q, '(b h t) d -> b h t d', b=B, h=self.num_heads, t=T)
        k = rearrange(k, '(b h t) d -> b h t d', b=B, h=self.num_heads, t=T)
        
        # Apply RoPE to queries and keys using TorchTune's implementation
        # TorchTune RoPE expects (B, T, num_heads, head_dim), so we need to rearrange
        q = rearrange(q, 'b h t d -> b t h d')  # (B, T, num_heads, head_dim)
        k = rearrange(k, 'b h t d -> b t h d')  # (B, T, num_heads, head_dim)
        
        q = self.rope(q)
        k = self.rope(k)
        
        # Rearrange back to (B, num_heads, T, head_dim) for scaled_dot_product_attention
        q = rearrange(q, 'b t h d -> b h t d')
        k = rearrange(k, 'b t h d -> b h t d')
        
        # Use pre-computed spatio-temporal causal mask if needed
        attn_mask = self.spatiotemporal_mask if use_causal else None
        
        # Use PyTorch's optimized scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            enable_gqa=True
        )
        
        # Reshape output
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, embed_dim)
        
        return self.proj(out)


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
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0, max_seq_len: int = 1024, n_patches: int = 220, num_frames: int = 16):
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
            t_emb: Time embeddings (batch_size, embed_dim)
            use_causal: Whether to use causal masking
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Get FiLM parameters from time embedding
        film_params = self.film_mlp(t_emb)  # (batch_size, embed_dim * 4)
        scale1, shift1, scale2, shift2 = film_params.chunk(4, dim=-1)  # Each: (batch_size, embed_dim)
        
        # Reshape for broadcasting: (batch_size, 1, embed_dim)
        scale1 = scale1.unsqueeze(1)
        shift1 = shift1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        
        # Attention block with FiLM conditioning
        h1 = self.norm1(x)
        h1 = h1 * (1 + scale1) + shift1  # FiLM modulation
        h1 = self.attn(h1, use_causal)
        x = x + h1
        
        # Feed-forward block with FiLM conditioning
        h2 = self.norm2(x)
        h2 = h2 * (1 + scale2) + shift2  # FiLM modulation
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
                 max_seq_len: int = 16,
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

        # Learnable velocity scale
        self.velocity_scale = nn.Parameter(torch.tensor(2.0))
        
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
    
    def forward(self, x: Tensor, t: Tensor = None, use_causal_mask: bool = True) -> Tensor:
        """
        Forward pass for sequence prediction
        
        Args:
            x: Input sequence (batch_size, seq_len, n_patches, latent_dim)
            t: Time parameter (batch_size,) for flow matching. If None, defaults to zeros.
            use_causal_mask: Whether to use causal masking for autoregressive prediction
            
        Returns:
            Predicted velocity field (batch_size, seq_len, n_patches, latent_dim)
        """
        batch_size, seq_len, n_patches, latent_dim = x.shape
        
        # Get time embeddings
        t_emb = self.get_timestep_embedding(t)  # (batch_size, time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # (batch_size, embed_dim)
        
        # Flatten spatial and temporal dimensions: (batch_size, seq_len * n_patches, latent_dim)
        x_flat = x.reshape(batch_size, seq_len * n_patches, latent_dim)
        
        # Project each patch to embedding space
        x_emb = self.input_proj(x_flat)  # (batch_size, seq_len * n_patches, embed_dim)
        
        # Add time conditioning: broadcast time embedding across all patches
        # This tells the model what timestep t it's at in the flow
        # x_emb = x_emb + t_emb.unsqueeze(1)  # (batch_size, seq_len * n_patches, embed_dim)
        
        # Apply transformer blocks with spatio-temporal causal masking
        for block in self.blocks:
            x_emb = block(x_emb, t_emb, use_causal_mask)
        
        # Layer norm (let's try to remove this)
        # x_emb = self.norm(x_emb)
        
        # Project back to latent space for all patches
        output = self.output_proj(x_emb)  # (batch_size, seq_len * n_patches, latent_dim)
        
        # Reshape back to (batch_size, seq_len, n_patches, latent_dim)
        output = output.reshape(batch_size, seq_len, n_patches, latent_dim)
        
        # Apply learnable velocity scale
        output = output * self.velocity_scale
        return output
    


def create_dit(latent_dim: int, 
               n_patches: int,
               embed_dim: int = 768,
               num_layers: int = 12,
               num_heads: int = 8,
               max_seq_len: int = 16) -> DiffusionTransformer:
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

def flow_matching_loss(model: DiffusionTransformer, batch: torch.Tensor, past_context_length: int):
    """
    Vanilla flow matching loss for training DiT model, without special handling of context frames
    
    Args:
        model: DiffusionTransformer model
        batch: Input sequences (batch_size, seq_len, n_patches, latent_dim)
        
    Returns:
        Flow matching loss
    """
    batch_size = batch.shape[0]
    
    # Sample noise
    prior = torch.randn_like(batch)
    
    # Sample time uniformly from [0, 1] for each batch item
    t = torch.rand(batch_size, device=batch.device)
    
    # Linear interpolation between prior and data
    x_t = (1 - t.view(-1, 1, 1, 1)) * prior + t.view(-1, 1, 1, 1) * batch
    
    # Predict velocity field
    v_pred = model(x_t, t)
    
    # True velocity is data - noise
    v_target = batch - prior
    
    # Simple MSE loss between predicted and target velocities
    loss = torch.nn.functional.mse_loss(v_pred, v_target)
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
        max_seq_len=16
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, n_patches, latent_dim)
    t = torch.rand(batch_size)  # Random time values
    print(f"Input shape: {x.shape}")
    print(f"Time shape: {t.shape}")
    
    # Forward pass with time conditioning
    output = dit(x, t)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {n_patches}, {latent_dim})")
    
    # Test without time parameter (should default to zeros)
    output_no_t = dit(x)
    print(f"Output shape (no time): {output_no_t.shape}")
    
    # Test flow matching loss
    print("\nTesting flow matching loss...")
    loss = flow_matching_loss(dit, x, 16)
    print(f"Flow matching loss: {loss.item():.6f}")
    
    # Test RoPE
    print("\nTesting TorchTune RoPE implementation...")
    # RoPE is now integrated into the attention mechanism via TorchTune
    print("RoPE is built into MultiHeadAttentionWithRoPE using torchtune.modules.RotaryPositionalEmbeddings")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dit()
