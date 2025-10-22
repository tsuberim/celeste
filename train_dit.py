#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import wandb
import numpy as np
from einops import rearrange
import atexit
from safetensors.torch import save_file, load_file
import tempfile
import time
import math

from encoded_dataset import EncodedDataset
from dit import create_dit, DiffusionTransformer, flow_matching_loss
from utils import get_device
from vae2 import VAE2
from generate import generate_and_save_video


def cleanup_memory():
    """Clean up PyTorch memory and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("Memory cleanup completed via atexit")


def save_model(dit_model, save_dir: str, epoch: int, optimizer=None,
               global_batch_idx=None, hyperparams=None):
    """Save DiT model checkpoint and training state"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with hyperparameters
    if hyperparams:
        filename = f"dit_seq{hyperparams['sequence_length']}_dim{hyperparams['latent_dim']}_embed{hyperparams['embed_dim']}_layers{hyperparams['num_layers']}_heads{hyperparams['num_heads']}.safetensors"
    else:
        filename = "dit.safetensors"
    
    # Save model weights using safetensors
    model_save_path = os.path.join(save_dir, filename)
    
    # Get state dict from model (handle DataParallel)
    if hasattr(dit_model, 'module'):
        state_dict = dit_model.module.state_dict()
    else:
        state_dict = dit_model.state_dict()
    
    save_file(state_dict, model_save_path)
    
    # Save training state
    training_state = {
        'epoch': epoch,
        'global_batch_idx': global_batch_idx if global_batch_idx is not None else 0,
        'model_state_dict': state_dict,
    }
    
    if optimizer is not None:
        training_state['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save training state with same naming convention
    training_filename = filename.replace('.safetensors', '_training_state.pt')
    training_save_path = os.path.join(save_dir, training_filename)
    torch.save(training_state, training_save_path)
    
    print(f"Checkpoint saved: model to {model_save_path}, training state to {training_save_path}")


def generate_and_log_videos(dit_model, vae_model, dataset, past_context_length, 
                            max_seq_len, n_patches, latent_dim, device, 
                            log_prefix="video", step_key="train/global_step", step_value=0):
    """Generate and log video samples to wandb"""
    try:
        print("Generating video sample...")
        
        # Sample 4 random prompt sequences from the dataset
        prompt_batch_size = 4
        sampled_prompts = []
        for _ in range(prompt_batch_size):
            idx = torch.randint(0, len(dataset), (1,)).item()
            prompt_seq = dataset[idx]  # (seq_len, n_patches, latent_dim)
            # Take first past_context_length frames as prompt
            sampled_prompts.append(prompt_seq[:past_context_length])
        
        # Stack prompts: (batch_size, past_context_length, n_patches, latent_dim)
        prompt_sequences = torch.stack(sampled_prompts, dim=0)
        
        # Generate videos and get as numpy array directly (batch, frames, C, H, W)
        video_array = generate_and_save_video(
            dit_model=dit_model,
            vae_model=vae_model,
            video_path=None,  # Not used when return_arrays=True
            num_frames=16,
            past_context_length=past_context_length,
            max_seq_len=max_seq_len,
            n_patches=n_patches,
            latent_dim=latent_dim,
            fps=12,
            device=device,
            batch_size=prompt_batch_size,
            prompt_sequences=prompt_sequences,
            return_arrays=True
        )
        
        if video_array is not None:
            # Log each video separately to wandb
            for i in range(video_array.shape[0]):
                wandb.log({
                    f'{log_prefix}_{i}': wandb.Video(video_array[i], fps=12, format="mp4"),
                    step_key: step_value
                })
            print(f"Logged {video_array.shape[0]} video samples to wandb")
    except Exception as e:
        print(f"Failed to generate video sample: {e}")


def train_dit(dataset_path: str,
              sequence_length: int = 16,
              latent_dim: int = 48,
              n_patches: int = 220,
              embed_dim: int = 512,
              num_layers: int = 12,
              num_heads: int = 8,
              max_seq_len: int = 16,
              batch_size: int = 4,
              num_epochs: int = 1000000,
              learning_rate: float = 1e-3,
              save_dir: str = "./models",
              max_frames: int = None,
              past_context_length: int = 15,
              gradient_accumulation_steps: int = 4):
    """Train DiT on encoded video dataset"""
    
    # Initialize wandb
    wandb.init(
        project="dit-video-training",
        config={
            "dataset_path": dataset_path,
            "sequence_length": sequence_length,
            "latent_dim": latent_dim,
            "n_patches": n_patches,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "max_frames": max_frames,
            "past_context_length": past_context_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
        }
    )
    
    # Adjust batch size and learning rate for multi-GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        batch_size *= num_gpus
        learning_rate *= num_gpus  # Scale LR by number of GPUs
        print(f"Using {num_gpus} GPUs, batch size: {batch_size}, scaled LR: {learning_rate:.2e}")
    
    # Setup dataset
    if os.path.isdir(dataset_path):
        # For directory of H5 files, just use the first one for now
        h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
        if not h5_files:
            raise ValueError(f"No H5 files found in directory: {dataset_path}")
        dataset_file = os.path.join(dataset_path, h5_files[0])
        print(f"Using H5 file: {dataset_file}")
    else:
        # Single H5 file
        dataset_file = dataset_path
    
    dataset = EncodedDataset(dataset_file, sequence_length=sequence_length, max_frames=max_frames)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Setup model
    device = get_device()
    dit_model = create_dit(
        latent_dim=latent_dim,
        n_patches=n_patches,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Load VAE for video generation
    vae_model = VAE2(size=2, latent_dim=latent_dim).to(device)
    vae_checkpoint_path = "./models/vae_size2_latent48.safetensors"
    if os.path.exists(vae_checkpoint_path):
        vae_state_dict = load_file(vae_checkpoint_path)
        vae_model.load_state_dict(vae_state_dict)
        print(f"Loaded VAE from {vae_checkpoint_path}")
    else:
        print(f"Warning: VAE checkpoint not found at {vae_checkpoint_path}, skipping video generation")
        vae_model = None
    
    # Register cleanup function with atexit
    atexit.register(cleanup_memory)
    
    # Setup optimizer with weight decay for regularization
    optimizer = optim.AdamW(dit_model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler("cuda")
    
    # Create hyperparameters dict for filename
    hyperparams = {
        'sequence_length': sequence_length,
        'latent_dim': latent_dim,
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'num_heads': num_heads
    }
    
    # Load checkpoint and training state if exists
    start_epoch = 0
    global_batch_idx = 0
    
    filename = f"dit_seq{sequence_length}_dim{latent_dim}_embed{embed_dim}_layers{num_layers}_heads{num_heads}.safetensors"
    model_save_path = os.path.join(save_dir, filename)
    training_save_path = os.path.join(save_dir, filename.replace('.safetensors', '_training_state.pt'))
    
    if os.path.exists(model_save_path) and os.path.exists(training_save_path):
        print("Loading checkpoint...")
        try:
            # Load model weights
            state_dict = load_file(model_save_path)
            if hasattr(dit_model, 'module'):
                dit_model.module.load_state_dict(state_dict)
            else:
                dit_model.load_state_dict(state_dict)
            
            # Load training state
            training_state = torch.load(training_save_path, map_location=device)
            start_epoch = training_state.get('epoch', 0)
            global_batch_idx = training_state.get('global_batch_idx', 0)
            
            if 'optimizer_state_dict' in training_state:
                optimizer.load_state_dict(training_state['optimizer_state_dict'])
            
            print(f"Checkpoint loaded: resuming from epoch {start_epoch}, batch {global_batch_idx}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
            global_batch_idx = 0
    
    print(f"Starting DiT training from epoch {start_epoch}")
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Mixed precision training: enabled")
    
    # Training loop
    dit_model.train()
    running_loss = 0.0
    best_loss = float('inf')
    
    # Time-based checkpoint and video generation (every 15 minutes)
    checkpoint_interval_seconds = 30 * 60  # 15 minutes
    last_checkpoint_time = time.time() - checkpoint_interval_seconds  # Force initial checkpoint
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                sequences = batch_data.to(device)
                
                # Forward pass with mixed precision
                with torch.amp.autocast("cuda"):
                    loss = flow_matching_loss(dit_model, sequences, past_context_length)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}!")
                    print(f"Skipping batch...")
                    continue
                
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update metrics (unscaled loss for logging)
                running_loss += loss.item() * gradient_accumulation_steps
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # Only step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                    # Gradient clipping and get grad norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=0.5)
                    
                    # Step optimizer with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    global_batch_idx += 1
                else:
                    grad_norm = 0.0
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                    'avg_loss': f"{running_loss / (batch_idx + 1):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'grad_norm': f"{grad_norm:.3f}",
                    'accum': f"{(batch_idx + 1) % gradient_accumulation_steps}/{gradient_accumulation_steps}"
                })
                
                # Log to wandb every batch
                wandb.log({
                    'train/loss': loss.item() * gradient_accumulation_steps,
                    'train/avg_loss': running_loss / (batch_idx + 1),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                    'train/grad_scaler_scale': scaler.get_scale(),
                    'train/velocity_scale': dit_model.velocity_scale.item(),
                    'train/epoch': epoch,
                    'train/global_step': global_batch_idx
                })
                
                # Check if 15 minutes have elapsed since last checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
                    print(f"\n15 minutes elapsed - saving checkpoint and generating video...")
                    
                    # Save checkpoint
                    save_model(dit_model, save_dir, epoch + 1, optimizer, global_batch_idx, hyperparams)
                    
                    # Generate video sample
                    if vae_model is not None:
                        generate_and_log_videos(
                            dit_model, vae_model, dataset, past_context_length,
                            max_seq_len, n_patches, latent_dim, device,
                            log_prefix="generated_video",
                            step_key="train/global_step",
                            step_value=global_batch_idx
                        )
                    
                    last_checkpoint_time = current_time
        
        # Calculate epoch averages
        avg_epoch_loss = epoch_loss / len(dataloader)
        
        print(f"Epoch {epoch+1} completed:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        
        # Log epoch metrics
        wandb.log({
            'epoch/loss': avg_epoch_loss,
            'epoch/number': epoch + 1
        })
        
        # Save model checkpoint after each epoch
        print(f"\nSaving checkpoint after epoch {epoch+1}...")
        save_model(dit_model, save_dir, epoch + 1, optimizer, global_batch_idx, hyperparams)
        
        # Generate and log video after each epoch
        if vae_model is not None:
            generate_and_log_videos(
                dit_model, vae_model, dataset, past_context_length,
                max_seq_len, n_patches, latent_dim, device,
                log_prefix="epoch_video",
                step_key="epoch/number",
                step_value=epoch + 1
            )
        
        # No LR scheduling - stability from architecture (RMSNorm + QKNorm)
        
        # Reset running loss
        running_loss = 0.0
    
    # Final save
    save_model(dit_model, save_dir, num_epochs, optimizer, global_batch_idx, hyperparams)
    
    print("Training completed!")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train DiT model on encoded video dataset")
    parser.add_argument("dataset_path", type=str, 
                       help="Path to encoded dataset (H5 file or directory)")
    parser.add_argument("--sequence_length", type=int, default=16,
                       help="Sequence length for training")
    parser.add_argument("--latent_dim", type=int, default=48,
                       help="Latent dimension per patch")
    parser.add_argument("--n_patches", type=int, default=220,
                       help="Number of patches per frame")
    parser.add_argument("--embed_dim", type=int, default=768,
                       help="Transformer embedding dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16,
                       help="Number of attention heads")
    parser.add_argument("--max_seq_len", type=int, default=16,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1000000,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./models",
                       help="Directory to save model checkpoints")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to load from dataset (None for all)")
    parser.add_argument("--past_context_length", type=int, default=15,
                       help="Past context length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of gradient accumulation steps")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Dataset path not found: {args.dataset_path}")
        return
    
    train_dit(
        dataset_path=args.dataset_path,
        sequence_length=args.sequence_length,
        latent_dim=args.latent_dim,
        n_patches=args.n_patches,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        max_frames=args.max_frames,
        past_context_length=args.past_context_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


if __name__ == "__main__":
    main()
