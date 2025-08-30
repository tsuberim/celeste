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
from einops import rearrange

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Reparameterization trick"""
    logvar_clamped = torch.clamp(logvar, min=-20, max=20)
    std = torch.exp(0.5 * logvar_clamped)
    eps = torch.randn_like(std)
    return mu + eps * std

class VAE(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 16, size: int=1):
        super().__init__()
        
        # Store parameters for save/load
        self.size = size
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(input_channels, 16 * size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16 * size, 32 * size, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32 * size, 64 * size, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64 * size, 128 * size, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128 * size, 256 * size, kernel_size=4, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256 * size, 512 * size, kernel_size=(6, 4), stride=1, padding=1)
        
        self.conv_batch_norm1 = nn.BatchNorm2d(16 * size)
        self.conv_batch_norm2 = nn.BatchNorm2d(32 * size)
        self.conv_batch_norm3 = nn.BatchNorm2d(64 * size)
        self.conv_batch_norm4 = nn.BatchNorm2d(128 * size)
        self.conv_batch_norm5 = nn.BatchNorm2d(256 * size)
        self.conv_batch_norm6 = nn.BatchNorm2d(512 * size)

        self.dec_conv1 = nn.Conv2d(latent_dim, 512 * size, kernel_size=(6, 4), stride=1, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(512 * size, 256 * size, kernel_size=(3,4), stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(256 * size, 128 * size, kernel_size=(3,3), stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(128 * size, 64 * size, kernel_size=(3,3), stride=2, padding=1)
        self.dec_conv5 = nn.ConvTranspose2d(64 * size, 32 * size, kernel_size=(2,3), stride=2, padding=1)
        self.dec_conv6 = nn.ConvTranspose2d(32 * size, 16 * size, kernel_size=(2,4), stride=2, padding=1)
        self.dec_conv7 = nn.ConvTranspose2d(16 * size, input_channels, kernel_size=(3,3), stride=1)
        
        # dec conv Batch norms
        self.dec_batch_norm1 = nn.BatchNorm2d(512 * size)
        self.dec_batch_norm2 = nn.BatchNorm2d(256 * size)
        self.dec_batch_norm3 = nn.BatchNorm2d(128 * size)
        self.dec_batch_norm4 = nn.BatchNorm2d(64 * size)
        self.dec_batch_norm5 = nn.BatchNorm2d(32 * size)
        self.dec_batch_norm6 = nn.BatchNorm2d(16 * size)
        self.dec_batch_norm7 = nn.BatchNorm2d(input_channels)
        
        # Latent space parameters
        self.final_conv_mu = nn.Conv2d(size*512, latent_dim, 1)
        self.final_conv_logvar = nn.Conv2d(size*512, latent_dim, 1)
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Count and print parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VAE initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    def _initialize_weights(self):
        """Initialize network weights using proper initialization schemes"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def encode(self, x: Tensor) -> Tensor:
        # x: (batch, channels, 320, 180)
        x = F.relu(self.conv_batch_norm1(self.conv1(x)))  
        x = F.relu(self.conv_batch_norm2(self.conv2(x)))  
        x = F.relu(self.conv_batch_norm3(self.conv3(x)))  
        x = F.relu(self.conv_batch_norm4(self.conv4(x)))  
        x = F.relu(self.conv_batch_norm5(self.conv5(x)))  
        x = F.relu(self.conv_batch_norm6(self.conv6(x))) 
        mu = self.final_conv_mu(x)
        logvar = self.final_conv_logvar(x)
        return mu, logvar
    
    def decode(self, z: Tensor) -> Tensor:
        x = F.relu(self.dec_batch_norm1(self.dec_conv1(z)))  
        x = F.relu(self.dec_batch_norm2(self.dec_conv2(x)))  
        x = F.relu(self.dec_batch_norm3(self.dec_conv3(x)))  
        x = F.relu(self.dec_batch_norm4(self.dec_conv4(x)))  
        x = F.relu(self.dec_batch_norm5(self.dec_conv5(x)))  
        x = F.relu(self.dec_batch_norm6(self.dec_conv6(x)))  
        x = self.dec_batch_norm7(self.dec_conv7(x)) 
        x = torch.tanh(x)
        return x
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)  
        z = reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def save(self, path: str):
        """Save VAE model using safetensors"""
        # Append model size and latent dim to path
        name, ext = os.path.splitext(path)
        path_with_info = f"{name}_size{self.size}_latent{self.latent_dim}{ext}"
        
        os.makedirs(os.path.dirname(path_with_info), exist_ok=True)
        
        # Get state dict
        state_dict = self.state_dict()
        
        # Save using safetensors
        save_file(state_dict, path_with_info)
        print(f"VAE saved to {path_with_info}")
    
    def load(self, path: str):
        name, ext = os.path.splitext(path)
        path_with_info = f"{name}_size{self.size}_latent{self.latent_dim}{ext}"

        """Load VAE model from safetensors"""
        if not os.path.exists(path_with_info):
            raise FileNotFoundError(f"Model file not found: {path_with_info}")
        
        # Load state dict
        state_dict = load_file(path_with_info)
        
        # Load into model
        self.load_state_dict(state_dict)
        print(f"VAE loaded from {path_with_info}")

def create_vae(input_channels: int = 3, latent_dim: int = 128, size: int = 1) -> VAE:
    """Create VAE and wrap in DataParallel if multiple GPUs available"""
    
    # Create VAE
    vae = VAE(input_channels=input_channels, latent_dim=latent_dim, size=size)
    
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
        vae = create_vae(input_channels=3, latent_dim=latent_dim, size=2)
        
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
            print(f"Latent shape: {mu.shape}")  # Should be (1, latent_dim, 16, 9)
            print(f"Logvar shape: {logvar.shape}")  # Should be (1, latent_dim, 16, 9)
            
    else:
        print("No sequences in dataset")
