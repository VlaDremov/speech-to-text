import whisper
import torch
import logging

class WhisperTranscriber:
    def __init__(self, model_size="base", device=None):
        """
        Initialize Whisper model with GPU support.
        Args:
            model_size (str): Model size ("tiny" to "large")
            device (str): Device to use ("cuda" or "cpu")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading Whisper {model_size} model on {self.device}")
        self.model = whisper.load_model(model_size).to(self.device)
        
    def transcribe(self, audio_path):
        """Transcribe audio file using Whisper with GPU acceleration"""
        try:
            result = self.model.transcribe(
                audio_path,
                verbose=False,
                language="ru",
                fp16=(self.device == "cuda")  # Use FP16 only on GPU
            )
            return result["segments"]
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            raise