import os
import logging
import subprocess
from datetime import datetime
from audio_processor import (
    load_audio,
    normalize_audio,
    convert_to_mono,
    reduce_noise,
    enhance_voice,
    detect_environment,
    apply_preemphasis,
    bandpass_filter,
)
from diarization import SimpleDiarization
from transcriber import WhisperTranscriber
import torch
from dotenv import load_dotenv
import librosa
import numpy as np
import soundfile as sf
import os.path

# Fix for Windows symbolic link issues with SpeechBrain
os.environ["PYTORCH_JIT"] = "0"
os.environ["SPEECHBRAIN_SYMLINK_STRATEGY"] = "copy"

load_dotenv()

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(
    log_dir, f"speech_to_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)


def verify_ffmpeg():
    """Verify FFmpeg installation and provide helpful error messages."""
    try:
        ffmpeg_paths = [
            "ffmpeg",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.getenv("PROGRAMFILES", ""), "ffmpeg", "bin", "ffmpeg.exe"),
        ]
        for path in ffmpeg_paths:
            try:
                subprocess.run(
                    [path, "-version"], capture_output=True, check=True, text=True
                )
                logging.info(f"FFmpeg found at: {path}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        logging.error(
            """
        FFmpeg not found. Please:
        1. Download from https://github.com/BtbN/FFmpeg-Builds/releases
        2. Extract to C:\\ffmpeg
        3. Add C:\\ffmpeg\\bin to system PATH
        4. Restart your terminal/IDE
        """
        )
        return False
    except Exception as e:
        logging.error(f"Error checking FFmpeg: {str(e)}")
        raise


def merge_short_segments(segments, min_duration=0.1, max_gap=0.3):
    """
    Merge speaker segments that are too short with adjacent segments
    or merge segments from the same speaker that are close together.
    """
    if not segments:
        return []

    # Sort segments by start time
    segments = sorted(segments, key=lambda x: x["start"])
    merged = []
    current = segments[0].copy()

    for next_seg in segments[1:]:
        # If current segment is too short
        if (current["end"] - current["start"]) < min_duration:
            # If same speaker or very close, merge them
            if current["speaker"] == next_seg["speaker"]:
                current["end"] = next_seg["end"]
            else:
                # If different speakers, choose the longer one
                if (
                    next_seg["end"] - next_seg["start"]
                    > current["end"] - current["start"]
                ):
                    # Keep start time but use next segment's speaker and end time
                    current["speaker"] = next_seg["speaker"]
                    current["end"] = next_seg["end"]
                # If current is longer, just extend it
                else:
                    current["end"] = max(current["end"], next_seg["end"])
        else:
            # Check if gap between segments is small
            gap = next_seg["start"] - current["end"]
            if gap < max_gap and current["speaker"] == next_seg["speaker"]:
                # Merge if same speaker and small gap
                current["end"] = next_seg["end"]
            else:
                # Add current to results and move to next
                merged.append(current)
                current = next_seg.copy()

    merged.append(current)
    return merged


def calculate_speaker_confidence(segment, speaker_segments):
    """
    Calculate a confidence score for speaker assignment based on overlap ratio
    and surrounding segments.
    """
    max_overlap_ratio = 0
    best_speaker = None
    seg_duration = segment["end"] - segment["start"]

    # First pass: find direct overlaps
    for speaker_seg in speaker_segments:
        overlap_start = max(segment["start"], speaker_seg["start"])
        overlap_end = min(segment["end"], speaker_seg["end"])
        overlap = max(0, overlap_end - overlap_start)
        overlap_ratio = overlap / seg_duration if seg_duration > 0 else 0

        if overlap_ratio > max_overlap_ratio:
            max_overlap_ratio = overlap_ratio
            best_speaker = speaker_seg["speaker"]

    # Second pass: check nearby segments if confidence is low
    if max_overlap_ratio < 0.3:
        # Look for nearby segments from the same speaker
        speaker_counts = {}
        window = 2.0  # Look at speakers +/- 2 seconds

        for speaker_seg in speaker_segments:
            # Check if segment is nearby
            if (
                speaker_seg["start"] - window <= segment["end"]
                and speaker_seg["end"] + window >= segment["start"]
            ):
                speaker = speaker_seg["speaker"]
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        # If we have nearby segments, boost confidence
        if speaker_counts:
            most_common_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
            # If we already found an overlapping speaker, give it priority
            if best_speaker and best_speaker in speaker_counts:
                # Adjust overlap ratio based on speaker context
                max_overlap_ratio = max(max_overlap_ratio, 0.3)
            # Otherwise use the most common nearby speaker
            elif max_overlap_ratio < 0.2:
                best_speaker = most_common_speaker
                max_overlap_ratio = 0.2

    return best_speaker, max_overlap_ratio


def format_timestamp(seconds):
    return f"{int(seconds//60):02d}:{int(seconds % 60):02d}"


def main():
    """Main function with centralized configuration."""
    if not verify_ffmpeg():
        raise RuntimeError("FFmpeg installation required")

    # Check if running on Windows and print admin warning
    if os.name == "nt":
        logging.warning(
            "On Windows, this application works best when run as administrator"
        )
        logging.warning("If you encounter permissions errors, please restart as admin")

    # =====================================
    # CONFIGURATION PARAMETERS - Edit these to control the behavior
    # =====================================

    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_dir = os.path.join(project_root, "input")
    output_dir = os.path.join(project_root, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Input/output file configuration
    mode = "local"
    if mode == "local":
        input_file = os.path.join(input_dir, "audio_mom_1_sample.m4a")
        output_filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_file = os.path.join(output_dir, output_filename)
        model_size = "medium"
    else:
        input_file = "/content/drive/MyDrive/audio_wav.wav"
        output_filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_file = os.path.join(output_dir, output_filename)
        model_size = "large-v3"

    # Authentication
    auth_token = os.getenv("HUGGING_FACE_TOKEN")
    if not auth_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")

    # Audio preprocessing parameters
    reduction_amount = 10
    reduction_amount_noisy = 20
    preemphasis_coefficient = 0.97
    target_dbfs = -20.0
    target_sample_rate = 16000

    # Diarization parameters
    default_num_speakers = 2
    default_min_speaker_duration = 0.5
    phone_min_speaker_duration = 0.3

    # Segment merging parameters
    default_merge_duration = 0.8
    default_merge_gap = 0.5
    phone_merge_duration = 0.4
    phone_merge_gap = 0.3

    # Speaker confidence thresholds
    confidence_threshold_default = 0.15
    confidence_threshold_noisy = 0.12
    confidence_threshold_clean = 0.18

    # =====================================
    # MAIN PROCESSING
    # =====================================
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        logging.warning("No GPU detected, using CPU only")

    try:
        # STEP 1: Load and preprocess audio
        logging.info(f"Starting audio processing for file: {input_file}")
        audio, sr = load_audio(input_file)
        logging.info(f"Audio loaded: {len(audio)} samples")
        duration_seconds = len(audio) / sr
        if duration_seconds < 0.5:
            raise ValueError("Audio file too short for processing")
        logging.info(f"Audio duration: {duration_seconds:.2f} seconds")
        logging.info(f"Sample rate: {sr}Hz")

        # STEP 2: Detect environment for adaptive processing
        env_type = detect_environment(audio, sr)
        logging.info(f"Detected environment: {env_type}")

        # STEP 3: Apply noise reduction based on environment
        if env_type == "noisy":
            logging.info("Applying enhanced noise reduction for noisy environment")
            audio = reduce_noise(
                audio, sr, reduction_amount=reduction_amount
            )
        else:
            audio = reduce_noise(
                audio, sr, reduction_amount=reduction_amount_noisy
            )

        # STEP 4: Apply pre-emphasis for speech enhancement
        audio = apply_preemphasis(audio, coef=preemphasis_coefficient)

        # STEP 5: Apply environment-specific voice enhancement
        if env_type == "room":
            audio = bandpass_filter(audio, 150, 4000, sr)
        else:
            audio = enhance_voice(audio, sr)

        # STEP 6: Resample if needed
        if sr != target_sample_rate:
            logging.info(
                f"Converting sample rate from {sr}Hz to {target_sample_rate}Hz"
            )
            audio = audio.astype(np.float32) / np.max(np.abs(audio))
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sample_rate)
            sr = target_sample_rate

        # STEP 7: Convert to mono and normalize
        audio = convert_to_mono(audio)
        audio = normalize_audio(audio, target_dbfs=target_dbfs)

        # STEP 8: Save processed audio
        filename = os.path.basename(input_file)
        base, ext = os.path.splitext(filename)
        processed_filename = f"{base}_processed.wav"
        temp_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else "."
        temp_wav = os.path.join(temp_dir, processed_filename)
        logging.info(f"Exporting to temporary WAV file: {temp_wav}")
        sf.write(temp_wav, audio, sr)
        logging.info(f"Saved processed audio to {temp_wav}")

        # STEP 9: Determine audio-specific parameters
        is_phone_audio = any(
            term in input_file.lower() for term in ["phone", "call", "interview"]
        )

        # Set diarization parameters based on audio type
        num_speakers = default_num_speakers
        min_speaker_duration = default_min_speaker_duration

        if is_phone_audio:
            logging.info("Phone/call audio detected, adjusting diarization parameters")
            min_speaker_duration = phone_min_speaker_duration
        elif duration_seconds > 600:  # Long recordings > 10 minutes
            logging.info("Long recording detected, adjusting diarization parameters")
            num_speakers = min(3, default_num_speakers)

        # STEP 10: Initialize models
        logging.info("Initializing simple speaker diarization...")
        diarizer = SimpleDiarization(
            auth_token=auth_token,
            num_speakers=num_speakers,
            device=device,
            min_speaker_duration=min_speaker_duration,
        )

        transcriber = WhisperTranscriber(model_size=model_size, device=device)

        # STEP 11: Perform diarization
        logging.info("Starting simple speaker diarization...")
        start_time = datetime.now()
        speakers = diarizer.process(temp_wav)
        diarization_time = (datetime.now() - start_time).total_seconds()
        logging.info(
            f"Simple diarization completed in {diarization_time:.2f} seconds with {len(speakers)} segments"
        )

        # STEP 12: Set segment merging parameters based on audio type
        merge_duration = default_merge_duration
        merge_gap = default_merge_gap

        if is_phone_audio:
            merge_duration = phone_merge_duration
            merge_gap = phone_merge_gap

        # STEP 13: Merge segments
        speakers = merge_short_segments(
            speakers, min_duration=merge_duration, max_gap=merge_gap
        )
        logging.info(f"Found {len(speakers)} speaker segments after merging")

        # STEP 14: Perform transcription
        logging.info("Starting transcription...")
        start_time = datetime.now()
        transcription = transcriber.transcribe(temp_wav)
        transcription_time = (datetime.now() - start_time).total_seconds()
        logging.info(
            f"Transcription completed in {transcription_time:.2f} seconds with {len(transcription)} segments"
        )

        # STEP 15: Set confidence threshold based on audio quality
        confidence_threshold = confidence_threshold_default
        if env_type == "noisy":
            confidence_threshold = confidence_threshold_noisy
        elif env_type == "clean":
            confidence_threshold = confidence_threshold_clean

        # STEP 16: Generate transcript
        with open(output_file, "w", encoding="utf-8") as f:
            current_speaker = None
            pending_text = []

            for segment in transcription:
                best_speaker, confidence = calculate_speaker_confidence(
                    segment, speakers
                )
                timestamp = f"[{format_timestamp(segment['start'])} -> {format_timestamp(segment['end'])}] "

                # Apply confidence threshold
                if best_speaker and confidence > confidence_threshold:
                    # If speaker changed, flush pending text and start new speaker section
                    if current_speaker != best_speaker:
                        # Flush any pending text from previous speaker
                        if pending_text and current_speaker:
                            f.write("\n".join(pending_text) + "\n")
                            pending_text = []

                        current_speaker = best_speaker
                        # Write speaker header
                        f.write(f"\n{current_speaker}:\n")

                    # Add this segment's text
                    pending_text.append(f"{timestamp}{segment['text'].strip()}")
                else:
                    # For low confidence segments, handle carefully
                    if current_speaker and pending_text:
                        # If we're currently tracking a speaker, keep going with low confidence
                        pending_text.append(f"{timestamp}{segment['text'].strip()}")
                    else:
                        # Start a new unknown speaker section
                        if pending_text:
                            f.write("\n".join(pending_text) + "\n")
                            pending_text = []

                        current_speaker = "UNKNOWN"
                        f.write(f"\n{current_speaker}:\n")
                        pending_text.append(f"{timestamp}{segment['text'].strip()}")

            # Flush any remaining text
            if pending_text:
                f.write("\n".join(pending_text) + "\n")

        # STEP 17: Clean up
        logging.info("Cleaning up temporary files...")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        logging.info("Processing completed successfully")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
