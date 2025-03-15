import whisper
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class WhisperTranscriber:
    def __init__(self, model_size="medium", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path):
        try:
            result = self.model.transcribe(
                audio_path,
                language="ru",
                task="transcribe",
                initial_prompt="Это разговор на русском языке.",
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.3,
                no_speech_threshold=0.1,
                # compression_ratio_threshold=2.4,
                # logprob_threshold=-1.0
            )
            if not result["segments"]:
                logging.warning("No speech detected in the audio")
                result["segments"] = [
                    {"start": 0, "end": 0.1, "text": "[Речь не распознана]"}
                ]
            return result["segments"]
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            raise
