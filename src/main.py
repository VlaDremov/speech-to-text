import os
import logging
import subprocess
from datetime import datetime
from audio_processor import load_audio, normalize_audio, convert_to_mono, enhance_voice, reduce_noise
from diarization import SpeakerDiarization
from transcriber import WhisperTranscriber
import torch
from dotenv import load_dotenv
import librosa
import numpy as np
from scipy.io import wavfile

os.environ['PYTORCH_JIT'] = '0'
os.environ['SPEECHBRAIN_SYMLINK_STRATEGY'] = 'copy'

load_dotenv()

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"speech_to_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def verify_ffmpeg():
    """Verify FFmpeg installation and provide helpful error messages."""
    try:
        ffmpeg_paths = [
            "ffmpeg",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.getenv('PROGRAMFILES', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        ]
        for path in ffmpeg_paths:
            try:
                subprocess.run([path, '-version'], capture_output=True, check=True, text=True)
                logging.info(f"FFmpeg found at: {path}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        logging.error("""
        FFmpeg not found. Please:
        1. Download from https://github.com/BtbN/FFmpeg-Builds/releases
        2. Extract to C:\\ffmpeg
        3. Add C:\\ffmpeg\\bin to system PATH
        4. Restart your terminal/IDE
        """)
        return False
    except Exception as e:
        logging.error(f"Error checking FFmpeg: {str(e)}")
        raise

def merge_short_segments(segments, min_duration=1.0):
    """Merge speaker segments shorter than min_duration with adjacent segments."""
    if not segments:
        return []
    merged = []
    current = segments[0].copy()
    for next_seg in segments[1:]:
        if (current['end'] - current['start']) < min_duration:
            current['end'] = next_seg['end']
            current['speaker'] = next_seg['speaker']
        else:
            merged.append(current)
            current = next_seg.copy()
    merged.append(current)
    return merged

def format_timestamp(seconds):
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

def process_audio(input_path, output_path, auth_token):
    logging.info(f"Starting audio processing for file: {input_path}")
    
    # Load and preprocess audio
    audio, sr = load_audio(input_path)
    logging.info(f"Audio loaded: {len(audio)} samples")
    duration_seconds = len(audio) / sr
    if duration_seconds < 0.5:
        raise ValueError("Audio file too short for processing")
    logging.info(f"Audio duration: {duration_seconds:.2f} seconds")
    logging.info(f"Sample rate: {sr}Hz")
    
    # Convert sample rate if necessary
    if sr != 16000:
        logging.info(f"Converting sample rate from {sr}Hz to 16000Hz")
        audio = audio.astype(np.float32) / np.max(np.abs(audio))  # Convert to floating-point
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Convert to mono if necessary
    audio = convert_to_mono(audio)
    
    # logging.info("Applying voice enhancement...")
    # audio = enhance_voice(audio, sr)
    
    # logging.info("Applying noise reduction...")
    # audio = reduce_noise(audio, sr, reduction_amount=10)
    
    temp_wav = "temp.wav"
    logging.info(f"Exporting to temporary WAV file: {temp_wav}")
    wavfile.write(temp_wav, sr, (audio * 32767).astype(np.int16))  # Convert back to int16 for saving
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    diarizer = SpeakerDiarization(
        auth_token=auth_token, 
        device=device,
        num_speakers=2,
        min_speakers=2,
        max_speakers=2
    )
    
    transcriber = WhisperTranscriber(
        model_size="medium",
        device=device
    )
    
    logging.info("Starting speaker diarization...")
    start_time = datetime.now()
    # Process with chunking (e.g., 45-second chunks with 15-second overlap)
    speakers = diarizer.process(
        temp_wav,
        preprocess=False,
        clustering_post=True,
        chunk_duration=60,
        overlap_duration=5
    )
    diarization_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Diarization completed in {diarization_time:.2f} seconds with {len(speakers)} segments")
    
    speakers = merge_short_segments(speakers)
    
    logging.info("Starting transcription...")
    transcription = transcriber.transcribe(temp_wav)
    logging.info("Transcription completed")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        current_speaker = None
        for segment in transcription:
            max_overlap = 0
            current_speaker_seg = None
            for speaker_seg in speakers:
                overlap_start = max(segment['start'], speaker_seg['start'])
                overlap_end = min(segment['end'], speaker_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    current_speaker_seg = speaker_seg
            timestamp = f"[{format_timestamp(segment['start'])} -> {format_timestamp(segment['end'])}] "
            if current_speaker_seg and max_overlap > 0.5:
                if current_speaker != current_speaker_seg['speaker']:
                    current_speaker = current_speaker_seg['speaker']
                    f.write(f"\n{current_speaker}:\n")
                f.write(f"{timestamp}{segment['text'].strip()}\n")
            else:
                if current_speaker:
                    f.write(f"{timestamp}{segment['text'].strip()}\n")
                else:
                    current_speaker = "НЕИЗВЕСТНЫЙ"
                    f.write(f"\n{current_speaker}:\n{timestamp}{segment['text'].strip()}\n")
    
    logging.info("Cleaning up temporary files...")
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    logging.info("Processing completed successfully")

if __name__ == "__main__":
    if not verify_ffmpeg():
        raise RuntimeError("FFmpeg installation required")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_dir = os.path.join(project_root, "input")
    output_dir = os.path.join(project_root, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    INPUT_FILE = os.path.join(project_root, "input", "audio.m4a")
    OUTPUT_FILE = os.path.join(project_root, "output", "transc_granddad.txt")
    AUTH_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
    if not AUTH_TOKEN:
        raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")
    logging.info(f"Input file: {INPUT_FILE}")
    logging.info(f"Output file: {OUTPUT_FILE}")
    if torch.cuda.is_available():
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        logging.warning("No GPU detected, using CPU only")
    try:
        process_audio(INPUT_FILE, OUTPUT_FILE, AUTH_TOKEN)
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise
