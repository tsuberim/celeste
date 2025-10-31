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
from tqdm import tqdm
import cv2
import time

from dit import create_dit, DiffusionTransformer
from vae2 import create_vae2
from utils import get_device
from video_dataset import VideoDataset


def solve_ode(dit_model: DiffusionTransformer, 
              gen_frames: torch.Tensor, 
              gen_actions: torch.Tensor,
              ode_steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve ODE to generate next frame using Euler integration
    
    Args:
        dit_model: DiT model
        gen_frames: Generated frames so far (batch_size, seq_len, n_patches, latent_dim)
        gen_actions: Actions for generated frames (batch_size, seq_len)
        ode_steps: Number of Euler integration steps
        
    Returns:
        (next_frame, next_acts_indices) tuple
        next_frame: (batch_size, n_patches, latent_dim)
        next_acts_indices: (batch_size, 1)
    """
    batch_size, seq_len, n_patches, latent_dim = gen_frames.shape
    device = gen_frames.device
    
    # Initialize with random noise
    x = torch.randn(batch_size, 1, n_patches, latent_dim, device=device, dtype=gen_frames.dtype)
    
    # Pre-allocate tensors to reduce allocations
    zero_action = torch.zeros((batch_size, 1), dtype=gen_actions.dtype, device=device)
    dt = 1.0 / ode_steps
    
    # Euler integration
    for i in range(ode_steps):
        t_scalar = i * dt
        
        # Concatenate generated frames with current prediction
        batch = torch.cat([gen_frames, x], dim=1)
        act_logits = torch.cat([gen_actions, zero_action], dim=1)
        
        # Create per-frame time values
        total_seq_len = batch.shape[1]
        t = torch.full((batch_size, total_seq_len), 0.95, device=device, dtype=gen_frames.dtype)
        t[:, seq_len:] = t_scalar
        
        # Get velocity and prev action logits
        velocity, act_logits = dit_model(batch, t, act_logits)
        
        # Update x using Euler step (in-place)
        x.add_(velocity[:, seq_len:], alpha=dt)
    
    # Sample with Gumbel-Softmax (hard) from act_logits[:, seq_len:]
    # act_logits: (batch_size, 1, num_action_codes)
    gumbel = F.gumbel_softmax(act_logits[:, seq_len:], tau=1.0, hard=True, dim=-1)
    next_acts_indices = gumbel.argmax(dim=-1)

    print(f"next_acts_indices: {next_acts_indices}")

    return x[:, 0], next_acts_indices

def actions_from_frames(dit_model, gen_frames, device):
    B, S = gen_frames.shape[0], gen_frames.shape[1]
    return torch.randint(0, 8, (B, S), device=device)
    t = torch.full((B, S), 1.0, device=device)
    print(f'predicting previous actions...')
    # Next-action prediction removed; return random actions for compatibility
    return torch.randint(0, 8, (B, S), device=device)

def generate_video_autoregressive(dit_model: DiffusionTransformer,
                                 vae_model,
                                 n_patches: int = 220,
                                 latent_dim: int = 48,
                                 max_seq_len: int = 24,
                                 device: torch.device = None,
                                 prompt_video_path: str = None,
                                 prompt_max_frames: int = None,
                                 batch_size: int = 1,
                                 prompt_sequences: torch.Tensor = None,
                                 ode_steps: int = 10,
                                 action_override_callback=None):
    """
    Generate video frames autoregressively using DiT model as a generator
    
    Args:
        dit_model: Trained DiT model
        vae_model: Trained VAE model
        n_patches: Number of patches per frame
        latent_dim: Latent dimension per patch
        max_seq_len: Maximum sequence length (context window size)
        device: Device to run on
        prompt_video_path: Optional video path for prompt
        prompt_max_frames: Max frames from prompt video
        batch_size: Batch size for generation
        prompt_sequences: Optional pre-encoded prompt sequences
        ode_steps: Number of ODE integration steps
        
        
    Yields:
        (encoded_frame, acts_indices, frame_idx) tuples
        encoded_frame: (batch_size, n_patches, latent_dim)
        acts_indices: (batch_size, 1)
        frame_idx: int
    """
    if device is None:
        device = get_device()
    
    inference_dtype = torch.float32

    dit_model = dit_model.to(device=device, dtype=inference_dtype).eval()
    
    # No torch.compile support (removed)
    
    
    if prompt_sequences is not None:
        gen_frames = prompt_sequences.to(device=device, dtype=inference_dtype)
        B, S = prompt_sequences.shape[0], prompt_sequences.shape[1]
        gen_actions = actions_from_frames(dit_model, gen_frames, device)
    # Load prompt frames from video or use random
    elif prompt_video_path and os.path.exists(prompt_video_path):
        from torch.utils.data import DataLoader
        
        context_length = max_seq_len - 1  # Reserve 1 slot for prediction
        video_dataset = VideoDataset(prompt_video_path, sequence_length=context_length, max_frames=prompt_max_frames)
        dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)
        
        # Get a random sequence from the dataset
        prompt_frames = next(iter(dataloader)).to(device)  # (batch_size, seq_len, 3, H, W)
        
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
            gen_frames = encoded.reshape(B, S, n_patches, latent_dim).to(dtype=inference_dtype)  # (batch_size, seq_len, n_patches, latent_dim)
        gen_actions = actions_from_frames(dit_model, gen_frames, device)
    else:
        context_length = max_seq_len - 1  # Reserve 1 slot for prediction
        gen_frames = torch.randn(batch_size, context_length, n_patches, latent_dim, device=device, dtype=inference_dtype)
        gen_actions = torch.randint(0, 8, (batch_size, context_length), device=device)
    
    context_length = max_seq_len - 1  # Reserve 1 slot for prediction
    frame_idx = 0
    with torch.inference_mode():
        while True:
            # Generate next frame
            new_frame, next_acts_indices = solve_ode(dit_model, gen_frames, gen_actions, ode_steps)

            # Optional override of next action via callback
            if action_override_callback is not None:
                try:
                    override = action_override_callback(frame_idx, next_acts_indices)
                    if override is not None:
                        # Expect same shape (B, 1)
                        next_acts_indices = override.to(gen_actions.device)
                except Exception:
                    pass

            # Yield frame - caller decides when to stop
            yield new_frame, next_acts_indices, frame_idx
            
            # Update context (sliding window)
            gen_frames = torch.cat([gen_frames, new_frame.unsqueeze(1)], dim=1)
            gen_actions = torch.cat([gen_actions, next_acts_indices], dim=1)
            if gen_frames.shape[1] > context_length:
                gen_frames = gen_frames[:, 1:]
                gen_actions = gen_actions[:, 1:]
            
            frame_idx += 1


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


def save_video(video_array: np.ndarray, output_path: str, fps: int = 24):
    """
    Save video array as MP4 video
    
    Args:
        video_array: Video frames (num_frames, H, W, C) in [0, 255] uint8 range
        output_path: Path to save video
        fps: Frames per second
    """
    num_frames, height, width, channels = video_array.shape
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
        # Get frame: (H, W, C) in range [0, 255]
        frame_rgb = video_array[frame_idx]
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
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


def generate_video(dit_model, vae_model, num_frames: int, **kwargs):
    """
    Generate video frames and return as numpy array
    
    Args:
        dit_model: Trained DiT model
        vae_model: Trained VAE model
        num_frames: Number of frames to generate
        **kwargs: Additional parameters for generate_video_autoregressive
        
    Returns:
        numpy array (batch, frames, H, W, C) in [0, 255] uint8
    """
    device = kwargs.get('device', None)
    if device is None:
        device = get_device()
        kwargs['device'] = device
    
    # Accumulate encoded frames from generator
    encoded_frames_list = []
    acts_list = []
    
    # Generate frames
    for encoded_frame, acts_indices, idx in generate_video_autoregressive(
        dit_model=dit_model,
        vae_model=vae_model,
        **kwargs
    ):
        encoded_frames_list.append(encoded_frame.cpu())
        acts_list.append(acts_indices.cpu())
        print(f"Generating frame {idx + 1}/{num_frames}", end='\r')
        
        if idx + 1 >= num_frames:
            break
    
    print()  # New line after progress
    
    # Stack accumulated frames: list of (B, P, L) -> (N, B, P, L) -> (B, N, P, L)
    encoded_frames = torch.stack(encoded_frames_list, dim=0).transpose(0, 1)  # (B, N, P, L)
    
    # Decode all videos - encoded_frames is (batch_size, num_frames, n_patches, latent_dim)
    B, N, P, L = encoded_frames.shape
    encoded_flat = encoded_frames.reshape(B * N, P, L)
    decoded_frames = decode_frames_with_vae(encoded_flat, vae_model, device=device, batch_size=8)
    decoded_frames = decoded_frames.reshape(B, N, *decoded_frames.shape[1:])  # (batch_size, num_frames, C, W, H)
    
    # Return as numpy array
    video_array = decoded_frames.cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 255]
    video_array = (video_array + 1.0) / 2.0
    video_array = np.clip(video_array, 0, 1)
    video_array = (video_array * 255).astype(np.uint8)
    
    # VAE outputs (batch, frames, C, W, H) but we return (batch, frames, H, W, C)
    # Transpose to (batch, frames, H, W, C)
    video_array = video_array.transpose(0, 1, 3, 4, 2)
    
    return video_array


def generate_realtime(dit_model: DiffusionTransformer,
                     vae_model,
                     window_name: str = "Realtime Video Generation",
                     **kwargs):
    """
    Generate video frames in realtime and display in a window
    Press ESC to quit
    """
    device = kwargs.get('device', None)
    if device is None:
        device = get_device()
        kwargs['device'] = device
    
    vae_model = vae_model.to(device)
    vae_model.eval()
    
    print("Starting realtime video generation...")
    print("Press ESC to quit")
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    kwargs['batch_size'] = 1  # Always 1 for realtime
    
    frame_count = 0
    start_time = time.time()
    last_frame_time = start_time
    fps_history = []
    
    try:
        # Provide action override callback using keyboard input (0-7 to force, 'c' to clear)
        forced_action = {'val': None}
        def action_override_cb(frame_idx: int, predicted_actions: torch.Tensor):
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7')):
                forced_action['val'] = int(chr(key))
                print(f"Forced action set to {forced_action['val']}")
            elif key in (ord('c'), ord('C')):
                forced_action['val'] = None
                print("Forced action cleared")
            if forced_action['val'] is not None:
                return torch.full_like(predicted_actions, forced_action['val'])
            return None

        for encoded_frame, acts_indices, idx in generate_video_autoregressive(
            dit_model=dit_model,
            vae_model=vae_model,
            action_override_callback=action_override_cb,
            **kwargs
        ):
            # Calculate FPS
            current_time = time.time()
            frame_time = current_time - last_frame_time
            instant_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(instant_fps)
            
            # Keep rolling average of last 10 frames
            if len(fps_history) > 10:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Decode frame immediately
            decoded_frame = vae_model.decode(encoded_frame)  # (1, 3, W, H)
            
            # Convert to numpy for display
            frame_np = decoded_frame[0].cpu().permute(2, 1, 0).numpy()  # (H, W, 3)
            frame_np = (frame_np + 1.0) / 2.0
            frame_np = np.clip(frame_np, 0, 1)
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Add FPS overlay to frame
            fps_text = f"FPS: {avg_fps:.2f}"
            cv2.putText(frame_bgr, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, frame_bgr)
            
            frame_count += 1
            elapsed = current_time - start_time
            overall_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_count} | FPS: {avg_fps:.1f} (avg: {overall_fps:.1f})", end='\r')
            
            last_frame_time = current_time
            
            # Check for ESC key (wait 1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                total_time = time.time() - start_time
                final_fps = frame_count / total_time if total_time > 0 else 0
                print(f"\nStopped after {frame_count} frames | Average FPS: {final_fps:.2f}")
                break
                
    except KeyboardInterrupt:
        print(f"\nInterrupted after {frame_count} frames")
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Generate video using DiT model")
    parser.add_argument("--realtime", action="store_true",
                       help="Display video in realtime window instead of saving to file")
    parser.add_argument("--vae_checkpoint", type=str, 
                       default="./models/vae_size2_latent48.safetensors",
                       help="Path to VAE model checkpoint")
    parser.add_argument("--max_frames", type=int, default=24,
                       help="Number of frames to generate")
    parser.add_argument("--max_seq_len", type=int, default=4,
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
    parser.add_argument("--num_heads", type=int, default=16,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--embed_dim", type=int, default=768,
                       help="Transformer embedding dimension")
    parser.add_argument("--prompt_video", type=str, default=None,
                       help="Path to video file to use as prompt (optional)")
    parser.add_argument("--prompt_max_frames", type=int, default=None,
                       help="Maximum frames to load from prompt video (None = all)")
    parser.add_argument("--ode_steps", type=int, default=3,
                       help="Number of ODE integration steps")
    # AMP removed
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    dit_checkpoint = f"./models/dit_seq24_dim{args.latent_dim}_embed{args.embed_dim}_layers{args.num_layers}_heads{args.num_heads}.safetensors"

    # Load DiT model
    dit_model = create_dit(
        latent_dim=args.latent_dim,
        n_patches=args.n_patches,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    )
    
    # Load DiT checkpoint if exists
    if os.path.exists(dit_checkpoint):
        print(f"Loading DiT model from {dit_checkpoint}")
        from safetensors.torch import load_file
        dit_state_dict = load_file(dit_checkpoint)
        dit_model.load_state_dict(dit_state_dict, strict=False)
        print("DiT model loaded successfully")
    else:
        print(f"Warning: DiT checkpoint not found at {dit_checkpoint}, using random weights")
    
    # Load VAE model
    vae_model = create_vae2(input_channels=3, latent_dim=args.latent_dim, size=args.vae_size)
    
    # Load VAE checkpoint if exists
    if os.path.exists(args.vae_checkpoint):
        print(f"Loading VAE model from {args.vae_checkpoint}")
        from safetensors.torch import load_file
        vae_state_dict = load_file(args.vae_checkpoint)
        if hasattr(vae_model, 'module'):
            vae_model.module.load_state_dict(vae_state_dict)
        else:
            vae_model.load_state_dict(vae_state_dict)
        print("VAE model loaded successfully")
    else:
        print(f"Warning: VAE checkpoint not found at {args.vae_checkpoint}, using random weights")
    
    # Prepare kwargs for generation
    gen_kwargs = {
        'n_patches': args.n_patches,
        'latent_dim': args.latent_dim,
        'max_seq_len': args.max_seq_len,
        'device': device,
        'prompt_video_path': args.prompt_video,
        'prompt_max_frames': args.prompt_max_frames,
        'ode_steps': args.ode_steps
    }
    
    
    
    # Run in selected mode
    if args.realtime:
        generate_realtime(
            dit_model=dit_model,
            vae_model=vae_model,
            **gen_kwargs
        )
    else:
        # Generate and save videos
        print(f"Generating {args.num_videos} video(s) with {args.max_frames} frames each...")
        
        for video_idx in range(args.num_videos):
            # Create output path for this video
            if args.num_videos > 1:
                output_path = args.output.replace('.mp4', f'_{video_idx}.mp4')
            else:
                output_path = args.output
            
            # Generate video array
            gen_kwargs['batch_size'] = 1
            video_array = generate_video(
                dit_model=dit_model,
                vae_model=vae_model,
                num_frames=args.max_frames,
                **gen_kwargs
            )
            
            # Save video (video_array is (batch, frames, H, W, C))
            save_video(video_array[0], output_path, fps=args.fps)
            print(f"Video {video_idx + 1}/{args.num_videos} saved to {output_path}")
        
        print(f"Generation complete! {args.num_videos} video(s) saved.")


if __name__ == "__main__":
    main()
