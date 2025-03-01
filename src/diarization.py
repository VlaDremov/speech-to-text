from pyannote.audio import Pipeline
import torch
import logging

class SpeakerDiarization:
    def __init__(self, auth_token, device=None, num_speakers=2):
        """
        Initialize the diarization pipeline with GPU support
        Args:
            auth_token (str): HuggingFace authentication token
            device (str): Device to use ("cuda" or "cpu")
            num_speakers (int): Number of speakers to detect (optional)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_speakers = num_speakers
        
        try:
            logging.info(f"Initializing diarization pipeline on {self.device}")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=auth_token
            )
            self.pipeline.to(torch.device(self.device))
            
        except Exception as e:
            logging.error(f"Failed to initialize diarization pipeline: {str(e)}")
            raise

    def process(self, audio_path):
        """Process audio file and return diarization results."""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
            
        try:
            # Apply diarization with specific number of speakers
            diarization = self.pipeline(
                audio_path,
                num_speakers=self.num_speakers
            )
            
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            logging.info(f"Diarization completed with {self.num_speakers} speakers")
            return results
            
        except Exception as e:
            logging.error(f"Diarization failed: {str(e)}")
            raise RuntimeError(f"Diarization failed: {str(e)}")