#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import wandb
from einops import rearrange

from video_dataset import VideoDataset
from vae import create_vae, vae_loss
from utils import get_device

def save_model(vae, save_dir: str, epoch: int, x=None, recon_x=None, batch_idx=None):
    """Save VAE model checkpoint and log reconstructions"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vae.safetensors")
    
    # Handle DataParallel wrapper
    if hasattr(vae, 'module'):
        vae.module.save(save_path)
    else:
        vae.save(save_path)

    # Log reconstructions to wandb if provided
    if x is not None and recon_x is not None and batch_idx is not None:
        x_sample = x[:6]
        recon_sample = recon_x[:6, 0]
        x_sample = rearrange(x_sample, 'b c w h -> b c h w')
        recon_sample = rearrange(recon_sample, 'b c w h -> b c h w')
        
        # Move to CPU and denormalize from [-1, 1] to [0, 255] for wandb
        x_sample = ((x_sample.cpu() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        recon_sample = ((recon_sample.cpu() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        
        comparison = torch.cat([x_sample, recon_sample], dim=3)
        wandb.log({
            f"reconstructions": [
                wandb.Image(img) for img in comparison
            ]
        })

    
def train_vae(video_path: str, 
              sequence_length: int = 8,
              max_frames: int = 1000,
              latent_dim: int = 16,
              size: int = 1,
              batch_size: int = 8,
              num_epochs: int = 100,
              learning_rate: float = 1e-4,
              beta: float = 1.0,
              save_dir: str = "./models"):
    """Train VAE on video dataset"""
    
    # Initialize wandb
    wandb.init(
        project="vae-video-training",
        config={
            "video_path": video_path,
            "sequence_length": sequence_length,
            "max_frames": max_frames,
            "latent_dim": latent_dim,
            "size": size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "beta": beta
        }
    )
    
    # Adjust batch size for multi-GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        batch_size *= num_gpus
        print(f"Using {num_gpus} GPUs, batch size: {batch_size}")
    
    # Setup dataset and model
    dataset = VideoDataset(video_path, sequence_length=sequence_length, max_frames=max_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = get_device()
    vae = create_vae(input_channels=3, latent_dim=latent_dim, size=size).to(device)
    
    # Load checkpoint if exists
    try:
        if hasattr(vae, 'module'):
            vae.module.load(os.path.join(save_dir, "vae.safetensors"))
        else:
            vae.load(os.path.join(save_dir, "vae.safetensors"))
        print("Checkpoint loaded")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
    
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    print(f"Training on {device}, {len(dataset)} sequences, {num_epochs} epochs")
    
    global_batch_idx = 0
    
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, frames in enumerate(pbar):
            b, s, c, h, w = frames.shape
            frames = frames.to(device)
            x = rearrange(frames, 'b s c h w -> (b s) c h w')
            recon_x, mu, logvar = vae(x)
            recon_x = rearrange(recon_x, '(b s) c h w -> b s c h w', b=b)
            mu = rearrange(mu, '(b s) c -> b s c', b=b)
            logvar = rearrange(logvar, '(b s) c -> b s c', b=b)
    
            loss, recon_loss, kl_loss = vae_loss(recon_x, frames, mu, logvar, beta)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            wandb.log({
                "batch_loss": loss.item(),
                "batch_recon_loss": recon_loss.item(),
                "batch_kl_loss": kl_loss.item(),
                "epoch": epoch,
                "batch": batch_idx,
                "global_batch": global_batch_idx
            }, step=global_batch_idx)
            
            # Save every 100 batches
            if (batch_idx + 0) % 100 == 0:
                save_model(vae, save_dir, epoch + 1, x, recon_x, global_batch_idx + 1)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Recon Loss': f'{recon_loss.item():.4f}', 
                'KL Loss': f'{kl_loss.item():.4f}'
            })
            
            global_batch_idx += 1
        
        # Save after each epoch
        save_model(vae, save_dir, epoch + 1, x, recon_x)
    
    wandb.finish()
    return vae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on video dataset")
    parser.add_argument("--video_path", type=str, default="./videos/z8r255LoVJc.mp4")
    parser.add_argument("--sequence_length", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="./models")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Video not found: {args.video_path}")
        exit(1)
    
    print(f"Training: {args.video_path}, {args.num_epochs} epochs, batch {args.batch_size}")
    
    vae = train_vae(
        video_path=args.video_path,
        sequence_length=args.sequence_length,
        max_frames=args.max_frames,
        latent_dim=args.latent_dim,
        size=args.size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        save_dir=args.save_dir
    )
