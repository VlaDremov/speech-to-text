import logging
import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr
import os
import audioread

def load_audio(file_path):
    """Load audio file using audioread."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    try:
        with audioread.audio_open(file_path) as f:
            sr = f.samplerate
            n_channels = f.channels
            audio = np.hstack([np.frombuffer(buf, np.int16) for buf in f])
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            return audio, sr
    except Exception as e:
        raise Exception(f"Error loading audio: {str(e)}")

def normalize_audio(audio, target_dbfs=-20.0):
    """Normalize audio levels."""
    rms = np.sqrt(np.mean(audio**2))
    scalar = 10**(target_dbfs / 20) / rms
    normalized_audio = audio * scalar
    return normalized_audio

def convert_to_mono(audio):
    """Convert audio to mono channel."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(audio, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the audio."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, audio)
    return y

def enhance_voice(audio, sr):
    """
    Enhance voice frequencies using EQ adjustment.
    """
    try:
        enhanced = bandpass_filter(audio, 300, 3400, sr)  # Bandpass filter for voice frequencies
        logging.info("Voice enhancement applied successfully")
        return enhanced
    except Exception as e:
        logging.error(f"Voice enhancement failed: {str(e)}")
        return audio

def reduce_noise(audio, sr, reduction_amount=10):
    """
    Apply noise reduction to the audio.
    """
    try:
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=reduction_amount/100)
        logging.info("Noise reduction applied successfully")
        return reduced_noise
    except Exception as e:
        logging.error(f"Noise reduction failed: {str(e)}")
        return audio

def split_audio_segments(audio, sr, segment_length=30):  # 30 seconds
    """Split audio into segments."""
    segment_samples = segment_length * sr
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)]
    return segments

def split_long_audio(audio, sr, max_duration=300):  # 5 minutes
    """Split long audio files into smaller chunks."""
    chunk_samples = max_duration * sr
    chunks = [audio[i + chunk_samples] for i in range(0, len(audio), chunk_samples)]
    return chunks
