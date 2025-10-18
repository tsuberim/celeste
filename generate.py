#!/usr/bin/env python3
"""
Video generation script using DiT model for autoregressive frame prediction.
Uses ODE solver to evolve from noise to clean frames, then decodes with VAE.
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from einops import rearrange
from scipy.integrate import solve_ivp

from dit import create_dit, DiffusionTransformer
from vae2 import create_vae2
from utils import get_device
from video_dataset import VideoDataset


class ODEFlowSolver:
    """ODE solver for flow matching generation"""
    
    def __init__(self, dit_model: DiffusionTransformer):
        self.dit_model = dit_model
    
    def solve_ode(self, gen_frames: torch.Tensor, future_frames: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Solve ODE from t=0 (noise) to t=1 (clean frame)
        
        Args:
            context: Context frames (seq_len, n_patches, latent_dim)
            num_steps: Number of integration steps
            
        Returns:
            Generated frame (n_patches, latent_dim)
        """
        batch_size, seq_len, n_patches, latent_dim = gen_frames.shape

        # Define ODE function for scipy
        def ode_func(t, x_flat):
            f_frames = x_flat
            batch = torch.cat([gen_frames, f_frames], dim=1)
            t = torch.full((batch_size,), t, device=batch.device)
            velocity = self.dit_model(batch, t)[:, gen_frames.shape[1]:]
            return velocity
        

        t_span = (0, 1)
        t_eval = np.array([1.0])  # Only need final result
        # Solve ODE using scipy
        # solution = solve_ivp(
        #     fun=ode_func,
        #     y0=future_frames.flatten().cpu().numpy(),
        #     t_span=t_span,
        #     method='DOP853',  # Dormand-Prince 8(5,3)
        #     t_eval=t_eval,
        #     rtol=1e-1,
        #     atol=1e-2,
        # )
        def euler_solve_mps(x0, steps=30):
            x = x0
            dt = 1.0 / steps
            for i in range(steps):
                t = i * dt#, torch.tensor([i * dt], device=x.device)
                v = ode_func(t, x)
                x = x + v * dt
            return x

        # def rk4_solve(x0, ode_func, steps=50):
        #     """A more stable RK4 solver."""
        #     x = x0
        #     dt = 1.0 / steps
        #     for i in range(steps):
        #         t = i * dt
        #         k1 = ode_func(t, x)
        #         k2 = ode_func(t + 0.5 * dt, x + 0.5 * dt * k1)
        #         k3 = ode_func(t + 0.5 * dt, x + 0.5 * dt * k2)
        #         k4 = ode_func(t + dt, x + dt * k3)
        #         x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        #     return x

        final_state = euler_solve_mps(future_frames)
        
        # Get final state and reshape
        # final_state = torch.from_numpy(solution.y).float().to(gen_frames.device)
        f_frames = final_state.reshape(batch_size, future_frames.shape[1], n_patches, latent_dim)
        
        gen_frame = f_frames[:, 0]
        new_f_frame = torch.randn_like(gen_frame).unsqueeze(1)
        return gen_frame, torch.cat([f_frames[:, 1:], new_f_frame], dim=1)


def generate_video_autoregressive(dit_model: DiffusionTransformer,
                                 vae_model,
                                 max_frames: int,
                                 n_patches: int = 220,
                                 latent_dim: int = 48,
                                 max_seq_len: int = 32,
                                 device: torch.device = None,
                                 past_context_length: int = 31,
                                 prompt_video_path: str = None,
                                 prompt_max_frames: int = None,
                                 batch_size: int = 1,
                                 prompt_sequences: torch.Tensor = None) -> torch.Tensor:
    """
    Generate video frames autoregressively using DiT model
    
    Args:
        dit_model: Trained DiT model
        max_frames: Number of frames to generate
        n_patches: Number of patches per frame
        latent_dim: Latent dimension per patch
        max_seq_len: Maximum context length
        ode_steps: Number of ODE integration steps
        device: Device to run on
        past_context_length: Past context length
    Returns:
        Generated frames (batch_size, max_frames, n_patches, latent_dim)
    """
    if device is None:
        device = get_device()
    
    dit_model = dit_model.to(device)
    
    # Initialize ODE solver
    ode_solver = ODEFlowSolver(dit_model)
    
    print(f"Generating {batch_size} videos of {max_frames} frames each using DiT model...")
    
    # Use provided prompt sequences if available
    if prompt_sequences is not None:
        print(f"Using provided prompt sequences: {prompt_sequences.shape}")
        gen_frames = prompt_sequences.to(device)
    # Load prompt frames from video or use random
    elif prompt_video_path and os.path.exists(prompt_video_path):
        print(f"Loading prompt from video: {prompt_video_path}")
        from torch.utils.data import DataLoader
        
        video_dataset = VideoDataset(prompt_video_path, sequence_length=past_context_length, max_frames=prompt_max_frames)
        dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)
        
        # Get a random sequence from the dataset
        prompt_frames = next(iter(dataloader)).to(device)  # (batch_size, seq_len, 3, H, W)
        
        print(f"Sampled prompt frames from random position: {prompt_frames.shape}")
        
        # Encode with VAE
        vae_model = vae_model.to(device)
        vae_model.eval()
        with torch.no_grad():
            # Reshape for VAE: (batch_size, seq_len, 3, H, W) -> (batch_size * seq_len, 3, H, W)
            B, S, C, H, W = prompt_frames.shape
            prompt_frames_flat = prompt_frames.reshape(B * S, C, H, W)
            mu, logvar = vae_model.encode(prompt_frames_flat)
            # Reparameterize
            from vae2 import reparameterize
            encoded = reparameterize(mu, logvar)  # (batch_size * seq_len, n_patches, latent_dim)
            gen_frames = encoded.reshape(B, S, n_patches, latent_dim)  # (batch_size, seq_len, n_patches, latent_dim)
        
        print(f"Encoded prompt frames: {gen_frames.shape}")
    else:
        print("Using random prompt frames")
        gen_frames = torch.randn(batch_size, past_context_length, n_patches, latent_dim, device=device)
    
    future_frames_prior = torch.randn(batch_size, max_seq_len-past_context_length, n_patches, latent_dim, device=device)
    last_gen_frame = gen_frames[:, -1]  # (batch_size, n_patches, latent_dim)
    # Repeat along the sequence dimension only
    last_gen_frame_repeated = last_gen_frame.unsqueeze(1).expand(batch_size, max_seq_len-past_context_length, n_patches, latent_dim)
    video_t = torch.arange(max_seq_len-past_context_length, device=gen_frames.device).unsqueeze(0) - past_context_length - 1
    w = 4.8
    t = 0.0
    noise_weight = torch.clamp(torch.exp(-(w / (max_seq_len-past_context_length) - past_context_length)*(video_t - t)), max=1.0).unsqueeze(2).unsqueeze(3)
    future_frames = noise_weight * last_gen_frame_repeated + (1 - noise_weight) * future_frames_prior

    print(f"Initial gen_frames: {gen_frames.shape}")
    print(f"Initial future_frames: {future_frames.shape}")

    out_frames = gen_frames

    def add_frame(frame: torch.Tensor):
        nonlocal gen_frames
        nonlocal out_frames
        gen_frames = torch.cat([gen_frames, frame.unsqueeze(1)], dim=1)
        out_frames = torch.cat([out_frames, frame.unsqueeze(1)], dim=1)
        if gen_frames.shape[1] >= past_context_length:
            gen_frames = gen_frames[:, 1:]
        return gen_frames

    with torch.no_grad():
        for _ in tqdm(range(max_frames), desc="Generating frames"):
            new_frame, future_frames = ode_solver.solve_ode(gen_frames, future_frames)
            add_frame(new_frame)

    return out_frames


def decode_frames_with_vae(encoded_frames: torch.Tensor,
                          vae_model,
                          batch_size: int = 8,
                          device: torch.device = None) -> torch.Tensor:
    """
    Decode encoded frames using VAE model
    
    Args:
        encoded_frames: Encoded frames (num_frames, n_patches, latent_dim)
        vae_model: Trained VAE model
        batch_size: Batch size for decoding
        device: Device to run on
        
    Returns:
        Decoded frames (num_frames, 3, H, W)
    """
    if device is None:
        device = get_device()
    
    vae_model = vae_model.to(device)
    vae_model.eval()
    
    num_frames = encoded_frames.shape[0]
    decoded_frames = []
    
    print(f"Decoding {num_frames} frames using VAE...")
    
    with torch.no_grad():
        for i in tqdm(range(0, num_frames, batch_size), desc="Decoding frames"):
            end_idx = min(i + batch_size, num_frames)
            batch_encoded = encoded_frames[i:end_idx].to(device)
            
            # Decode batch
            batch_decoded = vae_model.decode(batch_encoded)  # (batch_size, 3, H, W)
            decoded_frames.append(batch_decoded.cpu())
    
    # Concatenate all decoded frames
    all_decoded = torch.cat(decoded_frames, dim=0)  # (num_frames, 3, H, W)
    
    return all_decoded


def save_video(frames: torch.Tensor, output_path: str, fps: int = 24):
    """
    Save decoded frames as MP4 video
    
    Args:
        frames: Decoded frames (num_frames, 3, H, W) in [-1, 1] range
        output_path: Path to save video
        fps: Frames per second
    """
    num_frames, channels, width, height = frames.shape
    print(f"Video specs: {num_frames} frames, {height}x{width}, {fps} FPS")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Try codecs in order: mp4v (most compatible), then H264, then XVID
    codecs_to_try = [
        ('mp4v', output_path),  # MPEG-4 - most compatible, works without extra deps
        ('H264', output_path),  # H.264 - requires ffmpeg
        ('avc1', output_path),  # H.264 alternative
        ('XVID', output_path.replace('.mp4', '.avi')),  # Xvid fallback
    ]
    
    video_writer = None
    for fourcc_str, path in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        video_writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height), True)
        if video_writer.isOpened():
            output_path = path
            print(f"  ✅ Using codec: {fourcc_str}")
            break
        video_writer.release()
    
    if video_writer is None or not video_writer.isOpened():
        raise RuntimeError(f"Failed to initialize video writer - no compatible codec found")
    
    frames_written = 0
    for frame_idx in tqdm(range(num_frames), desc="Writing video"):
        # Get frame: [3, width, height] in range [-1, 1]
        frame = frames[frame_idx]
        
        # Convert from (C, W, H) to (H, W, C)
        frame_np = frame.permute(2, 1, 0).numpy()  # [height, width, 3]
        
        # Convert from [-1, 1] to [0, 1] then to [0, 255]
        frame_np = (frame_np + 1.0) / 2.0
        frame_np = np.clip(frame_np, 0, 1)
        frame_np = (frame_np * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Verify frame dimensions
        if frame_bgr.shape[:2] != (height, width):
            print(f"  ⚠️  Frame {frame_idx} dimension mismatch: {frame_bgr.shape} vs expected ({height}, {width})")
            continue
        
        # Write frame to video
        success = video_writer.write(frame_bgr)
        if success:
            frames_written += 1
    
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Verify the file was created and has reasonable size
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # At least 1KB
            print(f"  ✅ Saved: {output_path} ({file_size:,} bytes)")
        else:
            print(f"  ⚠️  Video file seems too small: {output_path} ({file_size} bytes)")
    else:
        print(f"  ❌ Video file not created: {output_path}")
    
    print(f"Frames written: {frames_written}/{num_frames}")


def main():
    parser = argparse.ArgumentParser(description="Generate video using DiT model")
    parser.add_argument("--vae_checkpoint", type=str, 
                       default="./models/vae_size2_latent48.safetensors",
                       help="Path to VAE model checkpoint")
    parser.add_argument("--max_frames", type=int, default=32,
                       help="Number of frames to generate")
    parser.add_argument("--max_seq_len", type=int, default=32,
                       help="Maximum context length for DiT")
    parser.add_argument("--output", type=str, default="generated_video.mp4",
                       help="Output video path")
    parser.add_argument("--fps", type=int, default=12,
                       help="Video framerate")
    parser.add_argument("--num_videos", type=int, default=1,
                       help="Number of videos to generate in parallel")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for VAE decoding")
    parser.add_argument("--latent_dim", type=int, default=48,
                       help="Latent dimension of VAE")
    parser.add_argument("--n_patches", type=int, default=220,
                       help="Number of patches per frame")
    parser.add_argument("--vae_size", type=int, default=2,
                       help="VAE size parameter")
    parser.add_argument("--past_context_length", type=int, default=31,
                       help="Past context length")
    parser.add_argument("--num_heads", type=int, default=16,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=16,
                       help="Number of transformer layers")
    parser.add_argument("--embed_dim", type=int, default=768,
                       help="Transformer embedding dimension")
    parser.add_argument("--prompt_video", type=str, default=None,
                       help="Path to video file to use as prompt (optional)")
    parser.add_argument("--prompt_max_frames", type=int, default=None,
                       help="Maximum frames to load from prompt video (None = all)")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    dit_checkpoint = f"./models/dit_seq{args.max_seq_len}_dim{args.latent_dim}_embed{args.embed_dim}_layers{args.num_layers}_heads{args.num_heads}.safetensors"

    # Load DiT model
    print(f"Loading DiT model from {dit_checkpoint}")
    dit_model = create_dit(
        latent_dim=args.latent_dim,
        n_patches=args.n_patches,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    )
    
    # Load DiT checkpoint (safetensors format)
    from safetensors.torch import load_file
    dit_state_dict = load_file(dit_checkpoint)
    dit_model.load_state_dict(dit_state_dict)
    print("DiT model loaded successfully")
    
    # Load VAE model
    print(f"Loading VAE model from {args.vae_checkpoint}")
    vae_model = create_vae2(input_channels=3, latent_dim=args.latent_dim, size=args.vae_size)
    
    # Load VAE checkpoint (safetensors format)
    vae_state_dict = load_file(args.vae_checkpoint)
    if hasattr(vae_model, 'module'):
        vae_model.module.load_state_dict(vae_state_dict)
    else:
        vae_model.load_state_dict(vae_state_dict)
    print("VAE model loaded successfully")
    
    # Generate encoded frames for multiple videos
    encoded_frames = generate_video_autoregressive(
        dit_model=dit_model,
        vae_model=vae_model,
        max_frames=args.max_frames,
        n_patches=args.n_patches,
        latent_dim=args.latent_dim,
        max_seq_len=args.max_seq_len,
        device=device,
        past_context_length=args.past_context_length,
        prompt_video_path=args.prompt_video,
        prompt_max_frames=args.prompt_max_frames,
        batch_size=args.num_videos
    )
    
    print(f"Generated encoded frames shape: {encoded_frames.shape}")
    
    # Decode all videos at once
    B, N, P, L = encoded_frames.shape
    print(f"Decoding {B} videos with {N} frames each...")
    
    # Flatten batch and frames: (B, N, P, L) -> (B*N, P, L)
    encoded_flat = encoded_frames.reshape(B * N, P, L)
    
    # Decode all frames at once
    decoded_flat = decode_frames_with_vae(
        encoded_frames=encoded_flat,
        vae_model=vae_model,
        batch_size=args.batch_size,
        device=device
    )
    
    # Reshape back to videos: (B*N, C, H, W) -> (B, N, C, H, W)
    decoded_frames = decoded_flat.reshape(B, N, *decoded_flat.shape[1:])
    print(f"Decoded frames shape: {decoded_frames.shape}")
    
    # Save each video
    for video_idx in range(B):
        # Get frames for this video
        video_frames = decoded_frames[video_idx]  # (N, C, H, W)
        
        # Create output path for this video
        if B > 1:
            base_name = args.output.replace('.mp4', f'_{video_idx}.mp4')
        else:
            base_name = args.output
        
        # Save video
        save_video(video_frames, base_name, fps=args.fps)
        print(f"Video {video_idx + 1}/{B} saved to {base_name}")
    
    print(f"Generation complete! {B} video(s) saved.")


def generate_and_save_video(dit_model, vae_model, video_path: str, num_frames: int = 32, 
                         past_context_length: int = 31, max_seq_len: int = 32, 
                         n_patches: int = 220, latent_dim: int = 48, fps: int = 12, device=None,
                         batch_size: int = 4, prompt_sequences: torch.Tensor = None,
                         return_arrays: bool = False):
    """
    Generate video frames and optionally save as MP4 or return as numpy arrays
    
    Args:
        dit_model: Trained DiT model
        vae_model: Trained VAE model
        video_path: Path to save MP4 video (ignored if return_arrays=True)
        num_frames: Number of frames to generate
        past_context_length: Context length for generation
        max_seq_len: Maximum sequence length
        n_patches: Number of patches
        latent_dim: Latent dimension
        fps: Frames per second
        device: Device to use
        batch_size: Number of videos to generate
        prompt_sequences: Optional prompt sequences
        return_arrays: If True, return numpy arrays instead of saving to files
        
    Returns:
        If return_arrays=True: List of numpy arrays (N, H, W, C) in [0, 255] uint8
        If return_arrays=False: List of paths to saved MP4 videos
    """
    if device is None:
        device = get_device()
    
    # Generate frames for multiple videos
    encoded_frames = generate_video_autoregressive(
        dit_model=dit_model,
        vae_model=vae_model,
        max_frames=num_frames,
        n_patches=n_patches,
        latent_dim=latent_dim,
        max_seq_len=max_seq_len,
        device=device,
        past_context_length=past_context_length,
        prompt_video_path=None,
        prompt_max_frames=None,
        batch_size=batch_size,
        prompt_sequences=prompt_sequences
    )
    
    # Decode all videos - encoded_frames is (batch_size, num_frames, n_patches, latent_dim)
    B, N, P, L = encoded_frames.shape
    encoded_flat = encoded_frames.reshape(B * N, P, L)
    decoded_frames = decode_frames_with_vae(encoded_flat, vae_model, device=device, batch_size=8)
    decoded_frames = decoded_frames.reshape(B, N, *decoded_frames.shape[1:])  # (batch_size, num_frames, C, W, H)
    
    if return_arrays:
        # Return as numpy array for direct WandB logging
        # decoded_frames is (batch_size, num_frames, C, W, H)
        video_array = decoded_frames.cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 255]
        video_array = (video_array + 1.0) / 2.0
        video_array = np.clip(video_array, 0, 1)
        video_array = (video_array * 255).astype(np.uint8)
        
        # VAE outputs (batch, frames, C, W, H) but WandB needs (batch, frames, C, H, W)
        # Swap W and H: axis 3 and 4
        video_array = video_array.transpose(0, 1, 2, 4, 3)
        
        return video_array
    else:
        # Save each video as a separate MP4
        video_paths = []
        try:
            for i in range(batch_size):
                # Get frames for this video
                video_frames = decoded_frames[i]  # (N, C, W, H)
                
                # Create path for this video
                base_path = video_path.replace('.mp4', f'_{i}.mp4')
                
                # Save using the save_video function
                save_video(video_frames, base_path, fps=fps)
                video_paths.append(base_path)
                print(f"Saved MP4 {i+1}/{batch_size} to {base_path}")
            
            return video_paths
        except Exception as e:
            print(f"Failed to save MP4s: {e}")
            return None


if __name__ == "__main__":
    main()
