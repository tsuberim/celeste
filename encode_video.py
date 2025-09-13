#!/usr/bin/env python3
import torch
import numpy as np
import h5py
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
from einops import rearrange

from vae2 import create_vae2
from video_dataset import VideoDataset
from utils import get_device


def encode_video_to_h5(video_path: str, 
                      model_path: str = "./models/vae.safetensors",
                      latent_dim: int = 32,
                      size: int = 1,
                      output_dir: str = "./encoded",
                      max_frames: int = None,
                      batch_size: int = 32):
    """
    Encode a video using the trained VAE and save latent representations to H5 file
    
    Args:
        video_path: Path to input video file
        model_path: Path to trained VAE model
        latent_dim: Latent dimension of the VAE
        size: Size parameter of the VAE
        output_dir: Directory to save encoded H5 files
        max_frames: Maximum number of frames to encode (None for all)
        batch_size: Batch size for encoding
    """
    
    device = get_device()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAE model
    print(f"Loading VAE model from {model_path}")
    vae = create_vae2(input_channels=3, latent_dim=latent_dim, size=size)
    vae = vae.to(device)
    
    # Load model weights
    try:
        if hasattr(vae, 'module'):
            vae.module.load(model_path)
        else:
            vae.load(model_path)
        print("VAE model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    vae.eval()
    
    # Load video using VideoDataset
    print(f"Loading video: {video_path}")
    dataset = VideoDataset(video_path, sequence_length=1, max_frames=max_frames)
    
    if len(dataset) == 0:
        print("No frames found in video")
        return
    
    # Get video name for output file
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}.h5")
    
    # Get frame dimensions and number of patches
    sample_frame = dataset[0][0]  # First frame of first sequence
    sample_tensor = torch.from_numpy(sample_frame).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        # Use forward pass with encode_only flag (benefits from DataParallel, skips decoding)
        _, mu, logvar = vae(sample_tensor, encode_only=True)
        n_patches, latent_dim_actual = mu.shape[1], mu.shape[2]
    
    print(f"Frame shape: {sample_frame.shape}")
    print(f"Latent shape per frame: ({n_patches}, {latent_dim_actual})")
    print(f"Total frames to encode: {len(dataset)}")
    
    # Create H5 file for storing latent representations
    with h5py.File(output_path, 'w') as h5f:
        # Create datasets for latent representations
        mu_dataset = h5f.create_dataset('mu', 
                                       shape=(len(dataset), n_patches, latent_dim_actual),
                                       dtype=np.float32,
                                       compression='gzip')
        logvar_dataset = h5f.create_dataset('logvar',
                                           shape=(len(dataset), n_patches, latent_dim_actual), 
                                           dtype=np.float32,
                                           compression='gzip')
        
        # Store metadata
        h5f.attrs['video_path'] = video_path
        h5f.attrs['video_name'] = video_name
        h5f.attrs['num_frames'] = len(dataset)
        h5f.attrs['latent_dim'] = latent_dim_actual
        h5f.attrs['n_patches'] = n_patches
        h5f.attrs['frame_shape'] = sample_frame.shape
        h5f.attrs['model_path'] = model_path
        h5f.attrs['vae_size'] = size
        
        # Encode frames in batches
        print(f"Encoding frames to {output_path}")
        
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Encoding batches"):
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_size_actual = end_idx - start_idx
                
                # Collect batch of frames
                batch_frames = []
                for i in range(start_idx, end_idx):
                    frame = dataset[i][0]  # Get first (and only) frame from sequence
                    batch_frames.append(frame)
                
                # Convert to tensor and move to device
                batch_tensor = torch.from_numpy(np.stack(batch_frames)).float().to(device)
                
                # Encode batch using forward pass with encode_only flag (benefits from DataParallel, skips decoding)
                _, mu_batch, logvar_batch = vae(batch_tensor, encode_only=True)
                
                # Store in H5 file
                mu_dataset[start_idx:end_idx] = mu_batch.cpu().numpy()
                logvar_dataset[start_idx:end_idx] = logvar_batch.cpu().numpy()
    
    print(f"Video encoded successfully to {output_path}")
    print(f"Encoded {len(dataset)} frames with shape ({n_patches}, {latent_dim_actual}) each")


def load_encoded_video(h5_path: str):
    """
    Load encoded video from H5 file
    
    Args:
        h5_path: Path to H5 file
        
    Returns:
        dict with mu, logvar, and metadata
    """
    with h5py.File(h5_path, 'r') as h5f:
        data = {
            'mu': h5f['mu'][:],
            'logvar': h5f['logvar'][:],
            'metadata': dict(h5f.attrs)
        }
    return data




def main():
    parser = argparse.ArgumentParser(description="Encode videos using VAE")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="./models/vae.safetensors",
                       help="Path to trained VAE model")
    parser.add_argument("--latent_dim", type=int, default=32,
                       help="Latent dimension of VAE")
    parser.add_argument("--size", type=int, default=1,
                       help="Size parameter of VAE")
    parser.add_argument("--output_dir", type=str, default="./encoded",
                       help="Output directory for encoded files")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to encode")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Video file not found: {args.video_path}")
        return
    
    encode_video_to_h5(
        video_path=args.video_path,
        model_path=args.model_path,
        latent_dim=args.latent_dim,
        size=args.size,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
