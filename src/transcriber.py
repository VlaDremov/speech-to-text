import whisper
import torch
import logging

class WhisperTranscriber:
    def __init__(self, model_size="medium", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading Whisper {model_size} model on {self.device}...")
        
        # Use large-v3 model for better Russian language support
        self.model = whisper.load_model(
            model_size,
            device=self.device
        )
        
    def transcribe(self, audio_path):
        try:
            # Add more specific options for short recordings
            result = self.model.transcribe(
                audio_path,
                language="ru",
                task="transcribe",
                initial_prompt="Это разговор на русском языке.",
                word_timestamps=True,
                condition_on_previous_text=False,  # Don't rely on context for short clips
                temperature=0,  # Reduce randomness
                no_speech_threshold=0.3,  # Lower threshold for speech detection
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0  # Be more lenient with word confidence
            )
            
            if not result["segments"]:
                logging.warning("No speech detected in the audio")
                # Add a dummy segment if none were detected
                result["segments"] = [{
                    'start': 0,
                    'end': 0.1,
                    'text': "[Речь не распознана]"  # Speech not recognized
                }]
            
            return result["segments"]
            
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            raise