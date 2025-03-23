import logging
import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr
import os
from pydub import AudioSegment
import librosa
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_audio(file_path):
    """Load an .m4a audio file from Google Drive using pydub in Google Colab."""
    logging.info(f"Loading audio file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    try:
        # Load the .m4a file; pydub uses FFmpeg internally.
        audio_segment = AudioSegment.from_file(file_path, format="m4a")
        sr = audio_segment.frame_rate
        # Convert audio samples to a numpy array
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        # If the audio is stereo or multi-channel, average the channels to get mono.
        if audio_segment.channels > 1:
            samples = samples.reshape((-1, audio_segment.channels)).mean(axis=1)
        return samples, sr
    except Exception as e:
        raise Exception(f"Error loading audio: {str(e)}")


def normalize_audio(audio, target_dbfs=-20.0):
    """Normalize audio levels."""
    rms = np.sqrt(np.mean(audio**2))
    scalar = 10 ** (target_dbfs / 20) / rms
    normalized_audio = audio * scalar
    return normalized_audio


def reduce_noise(audio, sr, reduction_amount=10):
    """
    Apply noise reduction to the audio.
    """
    try:
        reduced_noise = nr.reduce_noise(
            y=audio, sr=sr, prop_decrease=reduction_amount / 100
        )
        logging.info("Noise reduction applied successfully")
        return reduced_noise
    except Exception as e:
        logging.error(f"Noise reduction failed: {str(e)}")
        return audio


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
    b, a = butter(order, [low, high], btype="band")
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
        enhanced = bandpass_filter(
            audio, 300, 3400, sr
        )  # Bandpass filter for voice frequencies
        logging.info("Voice enhancement applied successfully")
        return enhanced
    except Exception as e:
        logging.error(f"Voice enhancement failed: {str(e)}")
        return audio


def split_audio_segments(audio, sr, segment_length=30):  # 30 seconds
    """Split audio into segments."""
    segment_samples = segment_length * sr
    segments = [
        audio[i : i + segment_samples] for i in range(0, len(audio), segment_samples)
    ]
    return segments


def split_long_audio(audio, sr, max_duration=300):  # 5 minutes
    """Split long audio files into smaller chunks."""
    chunk_samples = max_duration * sr
    chunks = [audio[i + chunk_samples] for i in range(0, len(audio), chunk_samples)]
    return chunks


def preprocess_audio(self, audio_path: str) -> str:
    """Process audio before diarization to improve quality."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Apply noise reduction
    audio = reduce_noise(audio, sr, reduction_amount=15)

    # Enhance voice frequencies
    audio = enhance_voice(audio, sr)

    # Normalize audio levels
    audio = normalize_audio(audio, target_dbfs=-20.0)

    # Save processed file
    processed_path = audio_path.replace(".", "_processed.")
    sf.write(processed_path, audio, sr)

    return processed_path


def detect_environment(audio, sr):
    """Detect recording environment and adjust processing accordingly."""
    # Analyze noise profile
    noise_level = np.percentile(np.abs(audio), 10)
    # Analyze frequency distribution
    freqs = np.abs(np.fft.rfft(audio))

    if noise_level > 0.01:  # High background noise
        return "noisy"
    elif np.mean(freqs[: int(len(freqs) * 0.1)]) > np.mean(
        freqs
    ):  # Low frequency rumble
        return "room"
    else:
        return "clean"


def apply_preemphasis(audio, coef=0.97):
    """Apply pre-emphasis filter to boost higher frequencies."""
    return np.append(audio[0], audio[1:] - coef * audio[:-1])
