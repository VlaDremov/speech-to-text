import logging
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.effects import normalize
import os
from io import BytesIO
import noisereduce as nr

def load_audio(file_path):
    """Load audio file with explicit FFmpeg path."""
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
    """Convert audio to a normalized numpy array (float32)."""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return samples / 32768.0

def split_audio_segments(audio, segment_length=30000):  # 30 seconds
    """Split audio into segments."""
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        segments.append(segment)
    return segments

def split_long_audio(audio, max_duration=300):  # 5 minutes
    """Split long audio files into smaller chunks."""
    chunks = []
    for i in range(0, len(audio), max_duration * 1000):
        chunk = audio[i:i + max_duration * 1000]
        chunks.append(chunk)
    return chunks

def enhance_voice(audio):
    """
    Enhance voice frequencies using EQ adjustment.
    """
    try:
        from pydub.effects import eq
        enhanced = audio.high_pass_filter(300)   # Remove frequencies below 300Hz
        enhanced = enhanced.low_pass_filter(3400)  # Remove frequencies above 3400Hz
        # Boost voice frequencies
        enhanced = eq(enhanced, 1000, gain=2.0)  # Boost around 1kHz
        enhanced = eq(enhanced, 2000, gain=1.5)  # Slight boost around 2kHz
        logging.info("Voice enhancement applied successfully")
        return enhanced
    except Exception as e:
        logging.error(f"Voice enhancement failed: {str(e)}")
        return audio

def reduce_noise(audio, reduction_amount=1):
    """Apply noise reduction using the noisereduce library."""
    try:
        samples = get_audio_array(audio)  # Normalized float array
        sr = audio.frame_rate
        reduced = nr.reduce_noise(y=samples, sr=sr)
        reduced_int16 = (reduced * 32768).astype(np.int16)
        return audio._spawn(reduced_int16.tobytes())
    except Exception as e:
        logging.error(f"Noise reduction failed: {str(e)}")
        return audio
