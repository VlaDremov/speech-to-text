import os
import logging
import subprocess
from datetime import datetime
from audio_processor import load_audio, normalize_audio, convert_to_mono
from diarization import SpeakerDiarization
from transcriber import WhisperTranscriber
import torch

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
    
    logging.info("Normalizing audio...")
    audio = normalize_audio(audio)
    
    logging.info("Converting to mono...")
    audio = convert_to_mono(audio)
    
    # Save preprocessed audio as WAV for compatibility
    temp_wav = "temp.wav"
    logging.info(f"Exporting to temporary WAV file: {temp_wav}")
    audio.export(temp_wav, format="wav")
    
    # Initialize models with GPU support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diarizer = SpeakerDiarization(
        auth_token=auth_token, 
        device=device,
        num_speakers=2  # Explicitly set to 2 speakers
    )
    transcriber = WhisperTranscriber(model_size="medium", device=device)
    
    # Get diarization results
    logging.info("Starting speaker diarization...")
    speakers = diarizer.process(temp_wav)
    logging.info(f"Found {len(speakers)} speaker segments")
    
    # Get transcription
    logging.info("Starting transcription...")
    transcription = transcriber.transcribe(temp_wav)
    logging.info("Transcription completed")
    
    # Combine results
    logging.info(f"Writing output to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        current_speaker = None
        for segment in transcription:
            # Find corresponding speaker for this segment
            current_speaker_seg = None
            for speaker_seg in speakers:
                if (segment['start'] >= speaker_seg['start'] and 
                    segment['start'] <= speaker_seg['end']):
                    current_speaker_seg = speaker_seg
                    break
            
            # Write segment with speaker identification
            if current_speaker_seg:
                if current_speaker != current_speaker_seg['speaker']:
                    current_speaker = current_speaker_seg['speaker']
                    f.write(f"\n{current_speaker}:\n")
                f.write(f"{segment['text'].strip()}\n")
            else:
                # Handle segments without clear speaker (optional)
                if current_speaker != "UNKNOWN":
                    current_speaker = "UNKNOWN"
                    f.write("\nUNKNOWN:\n")
                f.write(f"{segment['text'].strip()}\n")
    
    # Cleanup
    logging.info("Cleaning up temporary files...")
    os.remove(temp_wav)
    logging.info("Processing completed successfully")

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
    INPUT_FILE = os.path.join(project_root, "input", "audio_test.m4a")
    OUTPUT_FILE = os.path.join(project_root, "output", "transcription_test.txt")# "transcription_granddad.txt")
    AUTH_TOKEN = "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"
    
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