#!/usr/bin/env python3
import re
import sys
import subprocess
import os
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    parsed = urlparse(url)
    if parsed.hostname in ['www.youtube.com', 'youtube.com', 'youtu.be']:
        if parsed.hostname == 'youtu.be':
            return parsed.path[1:]
        query = parse_qs(parsed.query)
        return query.get('v', [None])[0]
    return None

def download_video(url, output_dir='./videos'):
    """Download YouTube video with specified requirements"""
    video_id = extract_video_id(url)
    if not video_id:
        print("Error: Invalid YouTube URL")
        return False
    
    filename = f"{video_id}.mp4"
    output_path = os.path.join(output_dir, filename)
    
    # yt-dlp command for 480p video only, no audio
    cmd = [
        'yt-dlp',
        '--format', 'best[height<=480][ext=mp4]/worst[height<=480][ext=mp4]',
        '--no-audio',
        '--output', output_path,
        url
    ]
    
    try:
        print(f"Downloading {video_id}...")
        subprocess.run(cmd, check=True)
        print(f"Downloaded to {output_path}")
        
        # Process video with ffmpeg: resize, 12fps, no anti-aliasing
        temp_path = f"{video_id}_temp.mp4"
        os.rename(output_path, temp_path)
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_path,
            '-vf', 'scale=320:180:flags=neighbor,fps=fps=12',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-y',  # Overwrite output
            output_path
        ]
        
        print("Processing video...")
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Clean up temp file
        os.remove(temp_path)
        print(f"Video processed and saved to {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print("Error: yt-dlp or ffmpeg not found. Please install them first.")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python download.py <youtube_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    success = download_video(url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
