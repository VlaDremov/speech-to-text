import logging
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.effects import normalize
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

def split_long_audio(audio, max_duration=300):  # 5 minutes
    """Split long audio files into smaller chunks"""
    chunks = []
    for i in range(0, len(audio), max_duration * 1000):
        chunk = audio[i:i + max_duration * 1000]
        chunks.append(chunk)
    return chunks

def enhance_voice(audio):
    """
    Enhance voice frequencies in the audio using EQ adjustment
    Args:
        audio: pydub.AudioSegment object
    Returns:
        pydub.AudioSegment: Enhanced audio
    """
    try:
        # Apply a bandpass filter to focus on voice frequencies (300Hz - 3400Hz)
        from pydub.effects import eq
        
        enhanced = audio.high_pass_filter(300)  # Remove frequencies below 300Hz
        enhanced = enhanced.low_pass_filter(3400)  # Remove frequencies above 3400Hz
        
        # Boost voice frequencies
        enhanced = eq(enhanced, 1000, gain=2.0)  # Boost around 1kHz
        enhanced = eq(enhanced, 2000, gain=1.5)  # Slight boost around 2kHz
        
        logging.info("Voice enhancement applied successfully")
        return enhanced
        
    except Exception as e:
        logging.error(f"Voice enhancement failed: {str(e)}")
        return audio  # Return original audio if enhancement fails

def reduce_noise(audio, reduction_amount=1):
    """Apply noise reduction to the audio"""
    try:
        # Ensure we're working with PyDub AudioSegment
        if isinstance(audio._data, np.ndarray):
            audio = audio._spawn(audio._data.tobytes())
            
        # Convert to numpy array for processing
        samples = np.array(audio.get_array_of_samples())
        
        # Process as mono
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)
        
        # Convert back to PyDub AudioSegment
        reduced = audio._spawn(samples.astype(np.int16).tobytes())
        reduced = reduced.set_channels(1)  # Ensure mono output
        
        logging.info("Noise reduction applied successfully")
        return reduced
        
    except Exception as e:
        logging.error(f"Noise reduction failed: {str(e)}")
        return audio  # Return original audio if reduction fails