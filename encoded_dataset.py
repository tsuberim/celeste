#!/usr/bin/env python3
import h5py
import numpy as np
import torch
import os
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from vae2 import reparameterize


class EncodedDataset:
    def __init__(self, 
                 h5_path: str, 
                 sequence_length: int = 24,
                 max_frames: int = None):
        """
        Dataset for reading encoded video frames from H5 files.
        Always performs reparameterization using mu and logvar.
        
        Args:
            h5_path: Path to H5 file containing encoded frames
            sequence_length: Number of consecutive frames per sequence
            max_frames: Maximum number of frames to load (None for all frames)
        """
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        
        # Load metadata and preload data into memory
        print(f"Loading encoded dataset from: {h5_path}")
        with h5py.File(h5_path, 'r') as h5f:
            self.metadata = dict(h5f.attrs)
            total_frames = h5f['mu'].shape[0]
            self.n_patches = h5f['mu'].shape[1]
            self.latent_dim = h5f['mu'].shape[2]
            
            # Determine number of frames to load
            self.num_frames = min(total_frames, max_frames) if max_frames is not None else total_frames
            
            # Preload mu and logvar data into memory with progress bar
            print(f"Preloading {self.num_frames}/{total_frames} frames into memory...")
            
            # Load data in chunks to show progress
            chunk_size = min(1000, self.num_frames)
            self.mu_data = torch.empty((self.num_frames, self.n_patches, self.latent_dim), dtype=torch.float32)
            self.logvar_data = torch.empty((self.num_frames, self.n_patches, self.latent_dim), dtype=torch.float32)
            
            with tqdm(total=self.num_frames, desc="Loading frames", unit="frames") as pbar:
                for start_idx in range(0, self.num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, self.num_frames)
                    
                    # Load chunk
                    mu_chunk = torch.from_numpy(h5f['mu'][start_idx:end_idx]).float()
                    logvar_chunk = torch.from_numpy(h5f['logvar'][start_idx:end_idx]).float()
                    
                    # Store in preallocated tensors
                    self.mu_data[start_idx:end_idx] = mu_chunk
                    self.logvar_data[start_idx:end_idx] = logvar_chunk
                    
                    pbar.update(end_idx - start_idx)
        
        # Calculate number of sequences
        self.num_sequences = max(0, self.num_frames - sequence_length + 1)
        
        print(f"Loaded encoded dataset: {self.num_frames} frames, "
              f"{self.num_sequences} sequences of length {sequence_length}")
        print(f"Latent shape per frame: ({self.n_patches}, {self.latent_dim})")
        print(f"Data preloaded into memory: mu={self.mu_data.shape}, logvar={self.logvar_data.shape}")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int):
        """
        Get sequence of encoded frames at index with reparameterization
        
        Returns:
            Tuple of (idx, sampled_sequence) where:
            - idx: int - the sample index
            - sampled_sequence: tensor of shape (sequence_length, n_patches, latent_dim)
        """
        if idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range (max: {self.num_sequences})")
        
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        
        # Get sequences from preloaded data
        mu_sequence = self.mu_data[start_idx:end_idx]  # (sequence_length, n_patches, latent_dim)
        logvar_sequence = self.logvar_data[start_idx:end_idx]  # (sequence_length, n_patches, latent_dim)
        
        # Always perform reparameterization using VAE2's method
        sampled_sequence = reparameterize(mu_sequence, logvar_sequence)
        
        return idx, sampled_sequence
    
    def get_batch(self, batch_size: int, random: bool = True) -> torch.Tensor:
        """
        Get a batch of sequences
        
        Args:
            batch_size: Number of sequences in batch
            random: If True, randomly sample sequences. If False, take first batch_size sequences
            
        Returns:
            Batched sequences with shape (batch_size, sequence_length, n_patches, latent_dim)
        """
        if batch_size > self.num_sequences:
            batch_size = self.num_sequences
        
        if random:
            indices = np.random.choice(self.num_sequences, batch_size, replace=False)
        else:
            indices = np.arange(batch_size)
        
        # Collect sampled sequences
        batch = torch.stack([self[i] for i in indices])
        return batch
    
    def get_frame(self, frame_idx: int) -> torch.Tensor:
        """
        Get a single encoded frame with reparameterization
        
        Args:
            frame_idx: Index of frame to retrieve
            
        Returns:
            Single sampled frame with shape (n_patches, latent_dim)
        """
        if frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (max: {self.num_frames})")
        
        # Get single frame from preloaded data
        mu_frame = self.mu_data[frame_idx]  # (n_patches, latent_dim)
        logvar_frame = self.logvar_data[frame_idx]  # (n_patches, latent_dim)
        
        # Always perform reparameterization using VAE2's method
        sampled_frame = reparameterize(mu_frame, logvar_frame)
        
        return sampled_frame
    
    def get_metadata(self) -> dict:
        """Get metadata about the encoded video"""
        return self.metadata.copy()


def main():
    """Test script for EncodedDataset"""
    # Test with encoded files if they exist
    encoded_dir = "./encoded"
    
    if not os.path.exists(encoded_dir):
        print(f"Encoded directory not found: {encoded_dir}")
        print("Please run encode_video.py first to create encoded H5 files")
        return
    
    # Find H5 files
    h5_files = list(Path(encoded_dir).glob("*.h5"))
    if not h5_files:
        print(f"No H5 files found in {encoded_dir}")
        print("Please run encode_video.py first to create encoded H5 files")
        return
    
    print(f"Found {len(h5_files)} H5 files: {[f.name for f in h5_files]}")
    
    # Test single dataset
    h5_path = str(h5_files[0])
    print(f"\nTesting single dataset: {h5_path}")
    
    try:
        dataset = EncodedDataset(h5_path, sequence_length=8, max_frames=1000)
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Metadata: {dataset.get_metadata()}")
        
        if len(dataset) > 0:
            # Test single sequence
            sequence = dataset[0]
            print(f"Sequence shape: {sequence.shape}")
            
            # Test single frame
            frame = dataset.get_frame(0)
            print(f"Frame shape: {frame.shape}")
            
            # Test batch
            if len(dataset) >= 4:
                batch = dataset.get_batch(4)
                print(f"Batch shape: {batch.shape}")
    
    except Exception as e:
        print(f"Error with dataset: {e}")


if __name__ == "__main__":
    main()
