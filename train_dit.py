#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from dit import create_dit, DiffusionTransformer
from utils import get_device
from vae2 import VAE2


def cleanup_memory():
    """Clean up PyTorch memory and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("Memory cleanup completed via atexit")


def save_model(dit_model, save_dir: str, epoch: int, optimizer_muon=None,
               optimizer_adamw=None, global_batch_idx=None, hyperparams=None, sampler_weights=None):
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
    
    if optimizer_muon is not None:
        training_state['optimizer_state_dict_muon'] = optimizer_muon.state_dict()
    if optimizer_adamw is not None:
        training_state['optimizer_state_dict_adamw'] = optimizer_adamw.state_dict()
    if sampler_weights is not None:
        training_state['sample_grad_ema'] = sampler_weights.float()
    
    # Save training state with same naming convention
    training_filename = filename.replace('.safetensors', '_training_state.pt')
    training_save_path = os.path.join(save_dir, training_filename)
    torch.save(training_state, training_save_path)
    
    print(f"Checkpoint saved: model to {model_save_path}, training state to {training_save_path}")


def generate_and_log_videos(dit_model, vae_model, dataset, 
                            max_seq_len, n_patches, latent_dim, device, 
                            log_prefix="video", step_key="train/global_step", step_value=0):
    return
    """Generate and log video samples to wandb"""
    try:
        print("Generating video sample...")
        
        # Set models to eval mode for generation
        dit_model.eval()
        if vae_model is not None:
            vae_model.eval()
        
        with torch.no_grad():
            # Sample 4 random prompt sequences from the dataset
            prompt_batch_size = 4
            context_length = max_seq_len - 1  # Reserve 1 slot for prediction
            sampled_prompts = []
            for _ in range(prompt_batch_size):
                idx = torch.randint(0, len(dataset), (1,)).item()
                _, prompt_seq = dataset[idx]  # (seq_len, n_patches, latent_dim)
                # Take first context_length frames as prompt
                sampled_prompts.append(prompt_seq[:context_length])
            
            # Stack prompts: (batch_size, context_length, n_patches, latent_dim)
            prompt_sequences = torch.stack(sampled_prompts, dim=0)
            
            # Generate videos and get as numpy array directly (batch, frames, H, W, C)
            from generate import generate_video
            video_array = generate_video(
                dit_model=dit_model,
                vae_model=vae_model,
                num_frames=24,
                max_seq_len=max_seq_len,
                n_patches=n_patches,
                latent_dim=latent_dim,
                device=device,
                batch_size=prompt_batch_size,
                prompt_sequences=prompt_sequences,
                ode_steps=3
            )
            
            if video_array is not None:
                # Log each video separately to wandb
                for i in range(video_array.shape[0]):
                    vid = video_array[i]  # (T, H, W, C), uint8
                    # Reorder to (T, C, H, W) as preferred by wandb to avoid unintended conversions
                    if vid.ndim == 4 and vid.shape[-1] in (1, 3, 4):
                        vid_chw = vid.transpose(0, 3, 1, 2)
                    else:
                        vid_chw = vid
                    wandb.log({
                        f'{log_prefix}_{i}': wandb.Video(vid_chw, fps=12, format="mp4"),
                        step_key: step_value
                    })
                print(f"Logged {video_array.shape[0]} video samples to wandb")
        
        # Restore train mode
        dit_model.train()
        if vae_model is not None:
            vae_model.train()
            
    except Exception as e:
        raise e
        print(f"Failed to generate video sample: {e}")
        # Ensure we restore train mode even on error
        dit_model.train()
        if vae_model is not None:
            vae_model.train()

def interpolate(x_0, x_1, t):
    batch_size, seq_len = x_0.shape[0], x_0.shape[1]
    x_t = (1 - t.view(batch_size, seq_len, 1, 1)) * x_0 + t.view(batch_size, seq_len, 1, 1) * x_1
    return x_t

def train_dit(dataset_path: str,
              sequence_length: int = 24,
              latent_dim: int = 48,
              n_patches: int = 220,
              embed_dim: int = 512,
              num_layers: int = 12,
              num_heads: int = 8,
              max_seq_len: int = 24,
              batch_size: int = 4,
              num_epochs: int = 1000000,
              learning_rate: float = 1e-3,
              save_dir: str = "./models",
              max_frames: int = None):
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
    
    # Initialize per-sample weights for weighted sampling
    # Start with equal weights, will be updated based on losses
    ema_alpha = 0.8  # EMA decay factor
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=torch.ones(len(dataset)),
        num_samples=len(dataset),
        replacement=True
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
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
    
    # Load VAE for video generations
    vae_checkpoint_path = os.path.join(save_dir, f"vae_size2_latent{latent_dim}.safetensors")
    if os.path.exists(vae_checkpoint_path):
        vae_model = VAE2(size=2, latent_dim=latent_dim).to(device)
        vae_state_dict = load_file(vae_checkpoint_path)
        vae_model.load_state_dict(vae_state_dict)
        vae_model.eval()  # Set to eval mode by default
        print(f"Loaded VAE from {vae_checkpoint_path}")
    else:
        print(f"Warning: VAE checkpoint not found at {vae_checkpoint_path}, skipping video generation")
        vae_model = None
    
    # Register cleanup function with atexit
    atexit.register(cleanup_memory)
    
    # Setup optimizer with weight decay for regularization
    params_2d = [param for param in dit_model.parameters() if param.ndim == 2]
    params_other = [param for param in dit_model.parameters() if param.ndim != 2]
    optimizer_muon = optim.Muon(params_2d, lr=learning_rate, weight_decay=0.01)
    optimizer_adamw = optim.AdamW(params_other, lr=learning_rate, weight_decay=0.01)
    
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
                dit_model.module.load_state_dict(state_dict, strict=False)
            else:
                dit_model.load_state_dict(state_dict, strict=False)
            
            # Load training state
            training_state = torch.load(training_save_path, map_location=device)
            start_epoch = training_state.get('epoch', 0)
            global_batch_idx = training_state.get('global_batch_idx', 0)
            
            if 'optimizer_state_dict_muon' in training_state:
                optimizer_muon.load_state_dict(training_state['optimizer_state_dict_muon'])

            if 'optimizer_state_dict_adamw' in training_state:
                optimizer_adamw.load_state_dict(training_state['optimizer_state_dict_adamw'])
            
            # Load sample weights if available and length matches
            if 'sample_grad_ema' in training_state:
                saved_weights = training_state['sample_grad_ema']
                if len(saved_weights) == len(dataset):
                    sampler.weights = saved_weights
                    print(f"Loaded sample weights (EMA)")
                else:
                    print(f"Warning: Saved weights length ({len(saved_weights)}) doesn't match dataset length ({len(dataset)}). Starting with fresh weights.")
            
            # Override learning rate with the provided learning_rate parameter
            for param_group in optimizer_muon.param_groups:
                param_group['lr'] = learning_rate
            for param_group in optimizer_adamw.param_groups:
                param_group['lr'] = learning_rate
            print(f"Learning rate set to {learning_rate}")
            
            print(f"Checkpoint loaded: resuming from epoch {start_epoch}, batch {global_batch_idx}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
            global_batch_idx = 0
    
    print(f"\n{'='*60}")
    print(f"Starting DiT training from epoch {start_epoch}")
    print(f"WandB run: {wandb.run.name} (ID: {wandb.run.id})")
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate - Muon: {optimizer_muon.param_groups[0]['lr']:.2e}, AdamW: {optimizer_adamw.param_groups[0]['lr']:.2e}")
    print(f"Mixed precision training: enabled")
    print(f"{'='*60}\n")
    
    # Training loop
    dit_model.train()
    running_loss = 0.0
    best_loss = float('inf')
    
    # Time-based checkpoint and video generation (every 15 minutes)
    checkpoint_interval_seconds = 15 * 60  # 15 minutes
    last_checkpoint_time = time.time() - checkpoint_interval_seconds  # Force initial checkpoint
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        
        self_forcing_alpha = .2
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (indices, batch_data) in enumerate(pbar):
                x_1 = batch_data.to(device)
                indices = indices.cpu().numpy()
                x_0 = torch.randn_like(x_1)
                batch_size, seq_len = x_1.shape[0], x_1.shape[1]
                t = torch.rand(batch_size, seq_len, device=x_1.device)
                t_mid = torch.rand(batch_size, seq_len, device=t.device)
                x_mid = interpolate(x_0, x_1, t_mid)
                
                # Forward pass with mixed precision
                with torch.amp.autocast("cuda"):
                    loss = 0.0
                    v_t_mid_pred, acts_logits_no_acts, vq_loss_no_acts = dit_model(x_mid, t_mid)
                    loss = loss + vq_loss_no_acts
                    v_t_mid = x_1 - x_0
                    fm_loss_no_acts = torch.nn.functional.mse_loss(v_t_mid_pred, v_t_mid)
                    loss = loss + fm_loss_no_acts
                    
                    acts = torch.multinomial(torch.nn.functional.softmax(acts_logits_no_acts, dim=-1).view(-1, acts_logits_no_acts.shape[-1]),num_samples=1).reshape(batch_size, seq_len)
                    self_forcing_x_0 = x_mid - v_t_mid_pred*t_mid.view(batch_size, seq_len, 1, 1)
                    x_0 = (1-self_forcing_alpha)*x_0 + self_forcing_alpha*self_forcing_x_0
                    x_t = interpolate(x_0, x_1, t)
                    v_t = x_1 - x_0

                    v_t_pred, acts_logits, vq_loss = dit_model(x_t, t)
                    fm_loss = torch.nn.functional.mse_loss(v_t_pred, v_t)
                    loss = loss + fm_loss
                    loss = loss + vq_loss

                    # consistency loss
                    acts_logits_for_loss = rearrange(acts_logits,'b s p -> b p s')
                    consistency_loss = 0.01*torch.nn.functional.cross_entropy(acts_logits_for_loss, acts)
                    loss = loss + consistency_loss
                    # predictivity_loss = -0.01*torch.nn.functional.mse_loss(v_t_pred - v_t, v_t_mid_pred - v_t_mid)
                    p = torch.softmax(acts_logits, dim=-1)
                    entropy_loss = 0.01*(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean()
                    loss = loss + entropy_loss
                    # loss = loss + predictivity_loss

                    
                    losses = {
                        'vq_loss': (vq_loss + vq_loss_no_acts).item(),
                        'fm_loss': (fm_loss + fm_loss_no_acts).item(),
                        'consistency_loss': consistency_loss.item(),
                        # 'predictivity_loss': predictivity_loss.item(),
                        'entropy_loss': entropy_loss.item(),
                    }
                    
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}!")
                    print(f"Skipping batch...")
                    continue
                
                # Backward pass with gradient scaling
                optimizer_muon.zero_grad()
                optimizer_adamw.zero_grad()
                scaler.scale(loss).backward()
                
                # Unscale gradients for clipping
                scaler.unscale_(optimizer_muon)
                scaler.unscale_(optimizer_adamw)
                
                # Gradient clipping and get grad norm
                grad_norm = torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=0.5)
                grad_norm_val = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                
                # Update sampler weights using gradient norm as difficulty estimate
                # Higher grad norm = harder batch = sample more often
                # Use same grad norm for all samples in batch as an estimate
                for idx in indices:
                    sampler.weights[idx] = ema_alpha * sampler.weights[idx] + (1 - ema_alpha) * grad_norm_val
                        
                # Step optimizer with scaler
                scaler.step(optimizer_muon)
                scaler.step(optimizer_adamw)
                scaler.update()
                
                global_batch_idx += 1
                running_loss += loss.item()
                epoch_loss += loss.item()
                valid_batches += 1
                
                # Log to wandb
                log_dict = {
                    'train/learning_rate_muon': optimizer_muon.param_groups[0]['lr'],
                    'train/learning_rate_adamw': optimizer_adamw.param_groups[0]['lr'],
                    'train/grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                    'train/grad_scaler_scale': scaler.get_scale(),
                    'train/epoch': epoch,
                    'train/global_step': global_batch_idx,
                    'train/loss': loss.item(),
                }
                # Log all component losses
                for k, v in losses.items():
                    log_dict[f'train/{k}'] = v
            
                # Average loss tracker
                if valid_batches > 0:
                    log_dict['train/avg_loss'] = running_loss / valid_batches
                
                wandb.log(log_dict)
                
                # Update progress bar
                pbar_dict = {
                    'loss': f"{loss.item():.4f}",
                    'lr_muon': f"{optimizer_muon.param_groups[0]['lr']:.2e}",
                    'lr_adamw': f"{optimizer_adamw.param_groups[0]['lr']:.2e}",
                    'grad_norm': f"{grad_norm:.3f}",
                    **losses
                }
                
                pbar_dict['avg_loss'] = f"{running_loss / valid_batches:.4f}"
                
                pbar.set_postfix(pbar_dict)
                
                # Check if 15 minutes have elapsed since last checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
                    print(f"\n15 minutes elapsed - saving checkpoint and generating video...")
                    
                    # Save checkpoint
                    save_model(dit_model, save_dir, epoch + 1, optimizer_muon, optimizer_adamw, global_batch_idx, hyperparams, sampler.weights)
                    
                    # Generate video sample
                    if vae_model is not None:
                        generate_and_log_videos(
                            dit_model, vae_model, dataset,
                            max_seq_len, n_patches, latent_dim, device,
                            log_prefix="generated_video",
                            step_key="train/global_step",
                            step_value=global_batch_idx
                        )
                    
                    last_checkpoint_time = current_time
        
        # Calculate epoch averages
        avg_epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
        
        print(f"Epoch {epoch+1} completed:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        
        # Log epoch metrics
        wandb.log({
            'epoch/loss': avg_epoch_loss,
            'epoch/number': epoch + 1
        })
        
        # Save model checkpoint after each epoch
        print(f"\nSaving checkpoint after epoch {epoch+1}...")
        save_model(dit_model, save_dir, epoch + 1, optimizer_muon, optimizer_adamw, global_batch_idx, hyperparams, sampler.weights)
        
        # Generate and log video after each epoch
        if vae_model is not None:
            generate_and_log_videos(
                dit_model, vae_model, dataset,
                max_seq_len, n_patches, latent_dim, device,
                log_prefix="epoch_video",
                step_key="epoch/number",
                step_value=epoch + 1
            )
        
        # Log sampling statistics at end of epoch
        wandb.log({
            'epoch/mean_sample_weight': sampler.weights.mean().item(),
            'epoch/max_sample_weight': sampler.weights.max().item(),
            'epoch/min_sample_weight': sampler.weights.min().item(),
            'epoch/number': epoch + 1
        })
        
        # No LR scheduling - stability from architecture (RMSNorm + QKNorm)
        
        # Reset running loss
        running_loss = 0.0
    
    # Final save
    save_model(dit_model, save_dir, num_epochs, optimizer_muon, optimizer_adamw, global_batch_idx, hyperparams, sampler.weights)
    
    print("Training completed!")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train DiT model on encoded video dataset")
    parser.add_argument("dataset_path", type=str, 
                       help="Path to encoded dataset (H5 file or directory)")
    parser.add_argument("--sequence_length", type=int, default=24,
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
    parser.add_argument("--max_seq_len", type=int, default=24,
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
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
