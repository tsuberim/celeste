#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import wandb
from einops import rearrange
import cv2
from video_dataset import VideoDataset
from vae2 import create_vae2, vae_loss
from utils import get_device
import numpy as np
from einops import rearrange
import atexit

def cleanup_memory():
    """Clean up PyTorch memory and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("Memory cleanup completed via atexit")

def transform_frame_batch(frame):
    frame = rearrange(frame, 'b c w h -> b h w c')
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    frame = torch.clamp(frame, 0, 1)  # Ensure values are in [0, 1]
    return frame

def save_model(vae, save_dir: str, epoch: int, optimizer=None, scheduler=None, 
               global_batch_idx=None, x=None, recon_x=None, batch_idx=None):
    """Save VAE model checkpoint and training state"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model_save_path = os.path.join(save_dir, "vae.safetensors")
    if hasattr(vae, 'module'):
        vae.module.save(model_save_path)
    else:
        vae.save(model_save_path)
    
    # Save training state
    training_state = {
        'epoch': epoch,
        'global_batch_idx': global_batch_idx if global_batch_idx is not None else 0,
        'model_state_dict': vae.state_dict(),
    }
    
    if optimizer is not None:
        training_state['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        training_state['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save training state
    training_save_path = os.path.join(save_dir, "training_state.pt")
    torch.save(training_state, training_save_path)
    
    print(f"Checkpoint saved: model to {model_save_path}, training state to {training_save_path}")

    # Log reconstructions to wandb if provided
    if x is not None and recon_x is not None and batch_idx is not None:
        x_sample = x[:6]
        recon_sample = recon_x[:6, 0]

        x_sample = transform_frame_batch(x_sample)
        recon_sample = transform_frame_batch(recon_sample)

        comparison = torch.cat([x_sample, recon_sample], dim=2)
        comparison = rearrange(comparison, 'b h w c -> b c h w')
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
    
    # Adjust batch size and learning rate for multi-GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        batch_size *= num_gpus
        learning_rate *= num_gpus  # Scale LR by number of GPUs
        print(f"Using {num_gpus} GPUs, batch size: {batch_size}, scaled LR: {learning_rate:.2e}")
    
    # Setup dataset and model
    dataset = VideoDataset(video_path, sequence_length=sequence_length, max_frames=max_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = get_device()
    vae = create_vae2(input_channels=3, latent_dim=latent_dim, size=size).to(device)
    
    # if torch.cuda.is_available():
    #     # Compile the model for faster training
    #     print("Compiling VAE2 with torch.compile...")
    #     vae = torch.compile(vae, mode="max-autotune", fullgraph=True)
    #     print("VAE2 compiled successfully!")
    
    # Register cleanup function with atexit
    atexit.register(cleanup_memory)
    
    optimizer = optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Load checkpoint and training state if exists
    start_epoch = 0
    global_batch_idx = 0
    
    try:
        # Load model weights
        if hasattr(vae, 'module'):
            vae.module.load(os.path.join(save_dir, "vae.safetensors"))
        else:
            vae.load(os.path.join(save_dir, "vae.safetensors"))
        print("Model weights loaded")
        
        # Load training state
        training_state_path = os.path.join(save_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            if 'scheduler_state_dict' in training_state:
                scheduler.load_state_dict(training_state['scheduler_state_dict'])
            start_epoch = training_state['epoch']
            global_batch_idx = training_state['global_batch_idx']
            print(f"Training state loaded: epoch {start_epoch}, batch {global_batch_idx}")
        else:
            print("No training state found, starting from scratch")
            
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Starting training from scratch")
    
    print(f"Training on {device}, {len(dataset)} sequences, {num_epochs} epochs")
    print(f"Starting from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, num_epochs):
        vae.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, frames in enumerate(pbar):
            optimizer.zero_grad()

            b, s, c, h, w = frames.shape
            frames = frames.to(device)
            x = rearrange(frames, 'b s c w h -> (b s) c w h')
            recon_x, mu, logvar = vae(x)
            recon_x = rearrange(recon_x, '(b s) c w h -> b s c w h', b=b)
            
            # Handle VAE2 per-patch latents: (b*s, n_patches, latent_dim) -> (b, s, n_patches, latent_dim)
            if len(mu.shape) == 3:  # VAE2: (batch, n_patches, latent_dim)
                mu = rearrange(mu, '(b s) n c -> b s n c', b=b)
                logvar = rearrange(logvar, '(b s) n c -> b s n c', b=b)
            else:  # Original VAE: (batch, latent_dim)
                mu = rearrange(mu, '(b s) c -> b s c', b=b)
                logvar = rearrange(logvar, '(b s) c -> b s c', b=b)
            loss, recon_loss, kl_loss = vae_loss(recon_x, frames, mu, logvar, beta)
            loss.backward()
            
            # Apply gradient clipping and get gradient norm
            max_grad_norm = 5.0  # You can adjust this value
            grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            wandb.log({
                "batch_loss": loss.item(),
                "batch_recon_loss": recon_loss.item(),
                "batch_kl_loss": kl_loss.item(),
                "grad_norm": grad_norm,
                "grad_clipped": grad_norm > max_grad_norm,
                "epoch": epoch,
                "batch": batch_idx,
                "global_batch": global_batch_idx
            }, step=global_batch_idx)
            
            # Save every 100 batches
            if (batch_idx + 0) % 100 == 0:
                save_model(vae, save_dir, epoch + 1, optimizer=optimizer, scheduler=scheduler,
                          global_batch_idx=global_batch_idx + 1, x=x, recon_x=recon_x, batch_idx=global_batch_idx + 1)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Recon Loss': f'{recon_loss.item():.4f}', 
                'KL Loss': f'{kl_loss.item():.4f}'
            })
            
            global_batch_idx += 1
    
    # Step the scheduler based on epoch loss
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    
    # Log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"learning_rate": current_lr, "epoch": epoch})
    
    # Save after each epoch
    save_model(vae, save_dir, epoch + 1, optimizer=optimizer, scheduler=scheduler,
              global_batch_idx=global_batch_idx, x=x, recon_x=recon_x)
    
    wandb.finish()
    
    # Register cleanup function with atexit
    atexit.register(cleanup_memory)
    
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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
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
