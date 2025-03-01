from pydub import AudioSegment
import numpy as np
import torch
import os

def load_audio(file_path):
    """Load audio file with explicit FFmpeg path"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    try:
        audio = AudioSegment.from_file(
            file_path,
            format=file_path.split('.')[-1]
        )
        return audio
    except Exception as e:
        raise Exception(f"Error loading audio: {str(e)}")

def normalize_audio(audio):
    """Normalize audio levels."""
    normalized_audio = audio.normalize(headroom=0.1)
    return normalized_audio

def convert_to_mono(audio):
    """Convert audio to mono channel."""
    return audio.set_channels(1)

def get_audio_array(audio):
    """Convert audio to numpy array."""
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

def split_audio_segments(audio, segment_length=30000):  # 30 seconds
    """Split audio into segments."""
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        segments.append(segment)
    return segments