#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch.nn.parallel import DataParallel
from safetensors.torch import save_file, load_file
import os
from utils import get_device
from einops import rearrange, repeat
import math

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Reparameterization trick"""
    logvar_clamped = torch.clamp(logvar, min=-20, max=20)
    std = torch.exp(0.5 * logvar_clamped)
    eps = torch.randn_like(std)
    return mu + eps * std

class PatchEmbed(nn.Module):
    """Convert image to patches and embed them"""
    def __init__(self, img_height: int = 320, img_width: int = 180, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.patches_h = img_height // patch_size
        self.patches_w = img_width // patch_size
        self.n_patches = self.patches_h * self.patches_w
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, n_patches, embed_dim)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x

class PatchUnembed(nn.Module):
    """Convert patches back to image"""
    def __init__(self, img_height: int = 320, img_width: int = 180, patch_size: int = 16, embed_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patches_h = img_height // patch_size
        self.patches_w = img_width // patch_size
        self.n_patches = self.patches_h * self.patches_w
        
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_patches, embed_dim)
        x = self.proj(x)  # (B, n_patches, patch_size*patch_size*out_channels)
        x = rearrange(x, 'b (n_h n_w) (h w c) -> b c (n_h h) (n_w w)', 
                     h=self.patch_size, w=self.patch_size, 
                     n_h=self.patches_h, n_w=self.patches_w)
        return x

class VAE2(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 16, size: int = 1, 
                 patch_size: int = 16, embed_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        
        # Store parameters for save/load
        self.size = size
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim * size
        
        # Calculate image dimensions (assuming 320x180)
        self.img_height, self.img_width = 320, 180
        # Ensure patch size divides evenly
        self.patch_size = patch_size
        self.patches_h = self.img_height // patch_size
        self.patches_w = self.img_width // patch_size
        self.n_patches = self.patches_h * self.patches_w
        
        # Adjust image dimensions to be divisible by patch size
        self.actual_height = self.patches_h * patch_size
        self.actual_width = self.patches_w * patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_height=self.actual_height,
            img_width=self.actual_width,
            patch_size=patch_size, 
            in_channels=input_channels, 
            embed_dim=self.embed_dim
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, self.embed_dim))
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_ratio) 
            for _ in range(num_layers)
        ])
        
        # Latent space projection (per-patch)
        self.latent_proj_mu = nn.Linear(self.embed_dim, latent_dim)
        self.latent_proj_logvar = nn.Linear(self.embed_dim, latent_dim)
        
        # Decoder (per-patch)
        self.decoder_proj = nn.Linear(latent_dim, self.embed_dim)
        
        # Transformer decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_ratio) 
            for _ in range(num_layers * 2)  # Fewer decoder layers
        ])
        
        # Patch unembedding
        self.patch_unembed = PatchUnembed(
            img_height=self.actual_height,
            img_width=self.actual_width,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            out_channels=input_channels
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Count and print parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VAE2 initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    def _initialize_weights(self):
        """Initialize network weights using proper initialization schemes"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: (batch, channels, 320, 180)
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Keep spatial structure - don't average
        # x: (B, n_patches, embed_dim)
        
        # Project to latent space (per-patch)
        mu = self.latent_proj_mu(x)      # (B, n_patches, latent_dim)
        logvar = self.latent_proj_logvar(x)  # (B, n_patches, latent_dim)
        
        return mu, logvar
    
    def decode(self, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        
        # z: (B, n_patches, latent_dim) - per-patch latent
        # Project from latent space
        x = self.decoder_proj(z)  # (B, n_patches, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply decoder transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Convert back to image
        x = self.patch_unembed(x)  # (B, C, H, W)
        
        # Resize to match input dimensions if needed
        if x.shape[2:] != (self.img_height, self.img_width):
            x = F.interpolate(x, size=(self.img_height, self.img_width), mode='bilinear', align_corners=False)
        
        # Apply tanh activation
        x = torch.tanh(x)
        
        return x
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)  
        z = reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def save(self, path: str):
        """Save VAE2 model using safetensors"""
        # Append model size and latent dim to path
        name, ext = os.path.splitext(path)
        path_with_info = f"{name}_size{self.size}_latent{self.latent_dim}{ext}"
        
        os.makedirs(os.path.dirname(path_with_info), exist_ok=True)
        
        # Get state dict
        state_dict = self.state_dict()
        
        # Save using safetensors
        save_file(state_dict, path_with_info)
        print(f"VAE2 saved to {path_with_info}")
    
    def load(self, path: str):
        name, ext = os.path.splitext(path)
        path_with_info = f"{name}_size{self.size}_latent{self.latent_dim}{ext}"

        """Load VAE2 model from safetensors"""
        if not os.path.exists(path_with_info):
            raise FileNotFoundError(f"Model file not found: {path_with_info}")
        
        # Load state dict
        state_dict = load_file(path_with_info)
        
        # Load into model
        self.load_state_dict(state_dict)
        print(f"VAE2 loaded from {path_with_info}")

def create_vae2(input_channels: int = 3, latent_dim: int = 128, size: int = 1) -> VAE2:
    """Create VAE2 and wrap in DataParallel if multiple GPUs available"""
    
    # Create VAE2
    vae = VAE2(input_channels=input_channels, latent_dim=latent_dim, size=size)
    
    # Wrap in DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        vae = DataParallel(vae)
    
    return vae

def vae_loss(recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float) -> Tensor:
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    logvar_clamped = torch.clamp(logvar, min=-20, max=20)
    kl_loss = -0.5 * beta * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss

if __name__ == "__main__":
    import sys
    import os
    from video_dataset import VideoDataset
    
    video_path = "./videos/z8r255LoVJc.mp4"  # Update path as needed
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Please download a video first using download.py")
        sys.exit(1)
    
    dataset = VideoDataset(video_path, sequence_length=8, max_frames=100)  # Small dataset for testing
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Frame shape: {dataset.frames.shape}")
    
    if len(dataset) > 0:
        frames = dataset[0]  # Get first sequence
        print(f"Input frames shape: {frames.shape}")
        
        x = torch.from_numpy(frames).unsqueeze(0).float()  # (1, 8, 3, 320, 180)
        
        x = x[0, 0]  # (3, 320, 180)
        x = x.unsqueeze(0)  # (1, 3, 320, 180)
        
        print(f"Single frame shape: {x.shape}")
        
        latent_dim = 16
        vae = create_vae2(input_channels=3, latent_dim=latent_dim, size=2)
        
        device = get_device()
        vae = vae.to(device)
        x = x.to(device)
        print(f"Moved model and input to device: {device}")
        
        recon_x, mu, logvar = vae(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {recon_x.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"Logvar shape: {logvar.shape}")
        
        total_loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta=1.0)
        print(f"Total loss: {total_loss:.4f}")
        print(f"Reconstruction loss: {recon_loss:.4f}")
        print(f"KL loss: {kl_loss:.4f}")
        
        with torch.no_grad():
            mu, logvar = vae.encode(x)
            print(f"Latent shape: {mu.shape}")  # Should be (1, n_patches, latent_dim)
            print(f"Logvar shape: {logvar.shape}")  # Should be (1, n_patches, latent_dim)
            
    else:
        print("No sequences in dataset")
