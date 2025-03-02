import os
import logging
import subprocess
from datetime import datetime
from audio_processor import load_audio, normalize_audio, convert_to_mono, enhance_voice, reduce_noise
from diarization import SpeakerDiarization
from transcriber import WhisperTranscriber
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv
os.environ['PYTORCH_JIT'] = '0'
os.environ['SPEECHBRAIN_SYMLINK_STRATEGY'] = 'copy'

# Load environment variables
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
    """Verify FFmpeg installation and provide helpful error messages"""
    try:
        # Check if ffmpeg exists in common locations
        ffmpeg_paths = [
            "ffmpeg",  # System PATH
            r"C:\ffmpeg\bin\ffmpeg.exe",  # Common install location
            os.path.join(os.getenv('PROGRAMFILES'), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        ]
        
        for path in ffmpeg_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                     capture_output=True, 
                                     check=True,
                                     text=True)
                logging.info(f"FFmpeg found at: {path}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
                
        # If we get here, FFmpeg wasn't found
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

def process_audio(input_path, output_path, auth_token):
    logging.info(f"Starting audio processing for file: {input_path}")
    
    # Load and preprocess audio
    logging.info("Loading audio file...")
    audio = load_audio(input_path)
    
    # Validate audio duration and quality
    duration_seconds = len(audio) / 1000
    if duration_seconds < 0.5:  # If less than 0.5 seconds
        raise ValueError("Audio file too short for processing")
    
    logging.info(f"Audio duration: {duration_seconds:.2f} seconds")
    logging.info(f"Sample rate: {audio.frame_rate}Hz")
    logging.info(f"Channels: {audio.channels}")
    
    # Convert sample rate before other processing
    if isinstance(audio._data, np.ndarray):
        # Convert back to PyDub AudioSegment
        audio = audio._spawn(audio._data.tobytes())
    
    if audio.frame_rate != 16000:
        logging.info(f"Converting sample rate from {audio.frame_rate}Hz to 16000Hz")
        audio = audio.set_frame_rate(16000)
    
    # Ensure mono audio
    logging.info("Converting to mono...")
    audio = convert_to_mono(audio)
    
    # Apply audio enhancements after sample rate conversion
    logging.info("Applying voice enhancement...")
    audio = enhance_voice(audio)
    
    logging.info("Applying noise reduction...")
    audio = reduce_noise(audio, reduction_amount=1)
    
    # Save preprocessed audio as WAV for compatibility
    temp_wav = "temp.wav"
    logging.info(f"Exporting to temporary WAV file: {temp_wav}")
    audio.export(temp_wav, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    
    # Initialize models with GPU support and optimization parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize diarizer
    diarizer = SpeakerDiarization(
        auth_token=auth_token, 
        device=device,
        num_speakers=2
    )
    
    # Initialize transcriber with large model for better Russian support
    transcriber = WhisperTranscriber(
        model_size="medium",  # Use large-v3 for better Russian language support
        device=device
    )
    
    # Get diarization results with progress logging
    logging.info("Starting speaker diarization...")
    start_time = datetime.now()
    speakers = diarizer.process(
        temp_wav,
        chunk_duration=45,     # Increased for better speaker detection
        step_duration=15       # Increased overlap for Russian speech patterns
    )
    diarization_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Diarization completed in {diarization_time:.2f} seconds")
    logging.info(f"Found {len(speakers)} speaker segments")
    
    # Merge short segments
    speakers = merge_short_segments(speakers)
    
    # Get transcription
    logging.info("Starting transcription...")
    transcription = transcriber.transcribe(temp_wav)
    logging.info("Transcription completed")
    
    # Combine results with improved speaker matching
    with open(output_path, 'w', encoding='utf-8') as f:
        current_speaker = None
        for segment in transcription:
            # Find speaker with maximum overlap
            max_overlap = 0
            current_speaker_seg = None
            
            for speaker_seg in speakers:
                # Calculate overlap duration
                overlap_start = max(segment['start'], speaker_seg['start'])
                overlap_end = min(segment['end'], speaker_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    current_speaker_seg = speaker_seg
            
            # Only switch speakers if we have significant overlap
            timestamp = f"[{format_timestamp(segment['start'])} -> {format_timestamp(segment['end'])}] "
            if current_speaker_seg and max_overlap > 0.5:  # At least 0.5 seconds overlap
                if current_speaker != current_speaker_seg['speaker']:
                    current_speaker = current_speaker_seg['speaker']
                    f.write(f"\n{current_speaker}:\n")
                f.write(f"{timestamp}{segment['text'].strip()}\n")
            else:
                # Keep previous speaker if overlap is minimal
                if current_speaker:
                    f.write(f"{timestamp}{segment['text'].strip()}\n")
                else:
                    current_speaker = "НЕИЗВЕСТНЫЙ"  # "UNKNOWN" in Russian
                    f.write(f"\n{current_speaker}:\n{timestamp}{segment['text'].strip()}\n")

    # Cleanup
    logging.info("Cleaning up temporary files...")
    os.remove(temp_wav)
    logging.info("Processing completed successfully")

def merge_short_segments(segments, min_duration=2.0):
    """Merge speaker segments shorter than min_duration with adjacent segments"""
    if not segments:
        return []
        
    merged = []
    current = segments[0].copy()
    
    for next_seg in segments[1:]:
        # If current segment is too short, merge with next
        if (current['end'] - current['start']) < min_duration:
            current['end'] = next_seg['end']
            current['speaker'] = next_seg['speaker']
        else:
            merged.append(current)
            current = next_seg.copy()
    
    # Don't forget to add the last segment
    merged.append(current)
    
    return merged

def format_timestamp(seconds):
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

if __name__ == "__main__":
    if not verify_ffmpeg():
        raise RuntimeError("FFmpeg installation required")
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to reach project root
    project_root = os.path.dirname(script_dir)
    
    # Create input and output directories if they don't exist
    input_dir = os.path.join(project_root, "input")
    output_dir = os.path.join(project_root, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct paths relative to project root
    INPUT_FILE = os.path.join(project_root, "input", "audio_test1.m4a")
    OUTPUT_FILE = os.path.join(project_root, "output",  "transcription_test1.txt")
    
    # Get token from environment variable
    AUTH_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
    if not AUTH_TOKEN:
        raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")
    
    # Verify file existence
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found at: {INPUT_FILE}")
        raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}")
    
    logging.info(f"Input file path: {INPUT_FILE}")
    logging.info(f"Output file path: {OUTPUT_FILE}")
    
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