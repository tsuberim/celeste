#!/usr/bin/env python3
import cv2
import numpy as np
import os
from typing import Tuple, Iterator
from tqdm import tqdm
import einops
from einops import rearrange

class VideoDataset:
    def __init__(self, video_path: str, sequence_length: int = 16, max_frames: int = None):
        """
        Initialize video dataset
        
        Args:
            video_path: Path to MP4 video file
            sequence_length: Number of frames per sequence
            max_frames: Maximum number of frames to load (None for all frames)
        """
        self.video_path = video_path
        self.sequence_length = sequence_length
        self.max_frames = max_frames
        
        # Load frames to memory
        print(f"Loading video: {video_path}")
        self.frames = self._load_frames()
        print(f"Loaded {len(self.frames)} frames")
        
        # Calculate number of sequences
        self.num_sequences = max(0, len(self.frames) - sequence_length + 1)
    
    def _load_frames(self) -> np.ndarray:
        """Load frames from video to numpy array"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_frames is not None:
            total_frames = min(total_frames, self.max_frames)
        
        # Get frame dimensions from first frame
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read first frame from video")
        
        height, width = first_frame.shape[:2]
        
        # Preallocate array: (frames, channels, width, height)
        frames_array = np.zeros((total_frames, 3, width, height), dtype=np.float16)
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Progress bar for frame loading
        with tqdm(total=total_frames, desc="Loading frames") as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret or (self.max_frames is not None and frame_count >= self.max_frames):
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_normalized = (frame.astype(np.float16) / 255.0) * 2.0 - 1.0

                # Store directly in preallocated array
                frames_array[frame_count] = rearrange(frame_normalized, 'h w c -> c w h')  # (H, W, C) -> (C, W, H)
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        # Trim to actual number of frames loaded
        frames_array = frames_array[:frame_count]
        
        return frames_array
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get sequence of frames at index"""
        if idx >= self.num_sequences:
            raise IndexError("Index out of range")
        
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        
        # Return sequence as (sequence_length, channels, height, width)
        return self.frames[start_idx:end_idx]
    
    def get_batch(self, batch_size: int) -> np.ndarray:
        """Get a batch of sequences"""
        if batch_size > self.num_sequences:
            batch_size = self.num_sequences
        
        # Randomly sample sequences
        indices = np.random.choice(self.num_sequences, batch_size, replace=False)
        batch = np.array([self[i] for i in indices])
        
        # Return as (batch_size, sequence_length, channels, height, width)
        return batch

def main():
    """Test script"""
    # Test with a sample video
    video_path = "./videos/z8r255LoVJc.mp4"  # Update path as needed
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Please download a video first using download.py")
        return
    
    # Create dataset
    dataset = VideoDataset(video_path, sequence_length=8, max_frames=1000)  # Limit to 1000 frames for testing
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Frame shape: {dataset.frames.shape}")
    
    # Test single sequence
    if len(dataset) > 0:
        sequence = dataset[0]
        print(f"Single sequence shape: {sequence.shape}")
        print(f"Sequence frame range: {sequence.min():.3f} to {sequence.max():.3f}")
    
    # Test batch
    if len(dataset) >= 4:
        batch = dataset.get_batch(4)
        print(f"Batch shape: {batch.shape}")
        print(f"Batch frame range: {batch.min():.3f} to {batch.max():.3f}")
    
    # Test iteration
    print("\nFirst 3 sequences:")
    for i in range(min(3, len(dataset))):
        seq = dataset[i]
        print(f"Sequence {i}: shape={seq.shape}, mean={seq.mean():.3f}")
    
    # Visualize samples using matplotlib
    import matplotlib.pyplot as plt
    
    if len(dataset) > 0:
        # Show first sequence frames
        sequence = dataset[0]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'First Sequence - Shape: {sequence.shape}')
        
        for i in range(min(8, sequence.shape[0])):
            row = i // 4
            col = i % 4
            
            # Convert from (channels, height, width) to (height, width, channels) for display
            frame = einops.rearrange(sequence[i], 'c w h -> h w c')
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = np.clip(frame, 0, 1)  # Ensure values are in [0, 1]
            axes[row, col].imshow(frame)
            axes[row, col].set_title(f'Frame {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show random batch samples
        if len(dataset) >= 4:
            batch = dataset.get_batch(4)
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig.suptitle(f'Random Batch - Shape: {batch.shape}')
            
            for seq_idx in range(4):
                for frame_idx in range(4):
                    if frame_idx < batch.shape[1]:
                        # Convert from (channels, height, width) to (height, width, channels) for display
                        frame = einops.rearrange(batch[seq_idx, frame_idx], 'c w h -> h w c')
                        frame = (frame - frame.min()) / (frame.max() - frame.min())
                        frame = np.clip(frame, 0, 1)  # Ensure values are in [0, 1]
                        axes[seq_idx, frame_idx].imshow(frame)
                        axes[seq_idx, frame_idx].set_title(f'Seq {seq_idx}, Frame {frame_idx}')
                        axes[seq_idx, frame_idx].axis('off')
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
