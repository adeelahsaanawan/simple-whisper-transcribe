#!/usr/bin/env python3

# Author: Adeel Ahsan
# Website: https://www.aeronautyy.com
# License: MIT
# Copyright (c) 2025 Adeel Ahsan


"""
Simple Audio Transcription Script

This script transcribes audio files using OpenAI's Whisper model without requiring FFmpeg.
It uses a simplified approach that may work for some audio formats.
"""

import sys
import argparse
import time
import ssl
from pathlib import Path

def transcribe_audio(audio_path, model_size="tiny"):
    """
    Transcribe an audio file using Whisper with a simplified approach.
    
    Args:
        audio_path: Path to the audio file
        model_size: Size of the Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        A dictionary containing the transcription and segments
    """
    try:
        import whisper
    except ImportError:
        print("Installing OpenAI Whisper...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
        import whisper
    
    print(f"Loading Whisper model ({model_size})...")
    try:
        # Try to load the model
        model = whisper.load_model(model_size)
    except ssl.SSLCertVerificationError:
        print("\n SSL Certificate Verification Error")
        print("This error occurs when trying to download the Whisper model.")
        print("\nPossible solutions:")
        print("1. Temporarily disable SSL verification (not recommended for production):")
        print("   import ssl")
        print("   ssl._create_default_https_context = ssl._create_unverified_context")
        print("\n2. Install proper SSL certificates for Python:")
        print("   For macOS: Run 'Install Certificates.command' in your Python directory")
        print("\n3. Use a pre-downloaded model or run in an environment with proper certificates")
        raise
    except Exception as e:
        print(f"\n Error loading Whisper model: {e}")
        raise
    
    print(f"Transcribing {audio_path}...")
    start_time = time.time()
    
    try:
        # Try to use a direct approach without FFmpeg
        import numpy as np
        from scipy.io import wavfile
        
        # Load the audio file
        print("Loading audio file...")
        try:
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 and normalize
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate}Hz to 16000Hz...")
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                sample_rate = 16000
            
            # Transcribe the audio
            print("Processing audio with Whisper...")
            result = model.transcribe(audio_data)
            
        except Exception as e:
            print(f"Error processing audio with scipy: {e}")
            print("Trying alternative method...")
            
            # Try using the default method (which might still need FFmpeg)
            result = model.transcribe(str(audio_path))
        
        end_time = time.time()
        print(f"Transcription completed in {end_time - start_time:.2f} seconds.")
        return result
    
    except Exception as e:
        print(f"\n Error during transcription: {e}")
        if "ffmpeg" in str(e).lower():
            print("\nThis appears to be an FFmpeg-related error.")
            print("Please install FFmpeg and try again, or use a different audio format.")
            print("\nFFmpeg Installation Instructions:")
            print("--------------------------------")
            print("macOS: brew install ffmpeg")
            print("Ubuntu/Debian: sudo apt-get update && sudo apt-get install ffmpeg")
            print("Windows: Download from https://ffmpeg.org/download.html")
        raise

def save_as_markdown(result, audio_path, output_dir=None):
    """
    Save transcription result as a Markdown file.
    
    Args:
        result: Transcription result from Whisper
        audio_path: Path to the original audio file
        output_dir: Directory to save the markdown file (default: same as audio)
        
    Returns:
        Path to the saved markdown file
    """
    audio_path = Path(audio_path)
    
    # Determine output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = audio_path.parent
    
    # Create markdown filename
    markdown_filename = f"{audio_path.stem}.md"
    markdown_path = output_dir / markdown_filename
    
    # Format the transcription as markdown
    with open(markdown_path, 'w', encoding='utf-8') as f:
        # Write title
        f.write(f"# Transcription: {audio_path.stem}\n\n")
        
        # Write metadata
        f.write("## Metadata\n\n")
        f.write(f"- **Source File:** {audio_path.name}\n")
        f.write(f"- **Date Transcribed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write full transcription
        f.write("## Full Transcription\n\n")
        f.write(result["text"])
        f.write("\n\n")
        
        # Write segments with timestamps if available
        if "segments" in result:
            f.write("## Segments with Timestamps\n\n")
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                f.write(f"**[{start_time} - {end_time}]** {segment['text']}\n\n")
    
    print(f"Transcription saved to {markdown_path}")
    return markdown_path

def format_timestamp(seconds):
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    parser = argparse.ArgumentParser(description="Simple audio transcription using OpenAI's Whisper model")
    parser.add_argument("input", help="Audio file to transcribe (WAV format recommended)")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                        default="tiny", help="Whisper model size to use (default: tiny)")
    parser.add_argument("--output", help="Output directory for transcriptions")
    parser.add_argument("--no-ssl-verify", action="store_true",
                        help="Disable SSL verification (SECURITY RISK: Use only for testing)")
    
    args = parser.parse_args()
    
    # Handle SSL verification bypass if requested
    if args.no_ssl_verify:
        print("SSL verification disabled. This is a security risk!")
        ssl._create_default_https_context = ssl._create_unverified_context
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process a single file
        try:
            # Install required packages
            try:
                import numpy
                import scipy
            except ImportError:
                print("Installing required packages...")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "scipy"], check=True)
            
            result = transcribe_audio(input_path, args.model)
            save_as_markdown(result, input_path, args.output)
            print(f"Successfully transcribed: {input_path}")
        except Exception as e:
            print(f"Error transcribing {input_path}: {e}")
            if "ssl" in str(e).lower() or "certificate" in str(e).lower():
                print("If this is an SSL certificate error, try using --no-ssl-verify (for testing only)")
            sys.exit(1)
    else:
        print(f"Input path {input_path} does not exist.")
        sys.exit(1)

if __name__ == "__main__":
    main()
