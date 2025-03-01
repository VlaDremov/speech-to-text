from pyannote.audio import Pipeline
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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
            
            # Speed optimization parameters
            self.pipeline.instantiate({
                "segmentation": {
                    "min_duration_off": 0.1,      # Minimum duration of non-speech region
                    "threshold": 0.50,            # Segmentation threshold (lower = faster)
                },
                "clustering": {
                    "coef_dia": 1.0,             # Clustering coefficient
                    "min_duration": 1.0,         # Minimum duration of speech turns
                },
            })
            
        except Exception as e:
            logging.error(f"Failed to initialize diarization pipeline: {str(e)}")
            raise

    def process_chunk(self, chunk_path):
        """Process a single audio chunk"""
        try:
            return self.pipeline(chunk_path, num_speakers=self.num_speakers)
        except Exception as e:
            logging.error(f"Chunk processing failed: {str(e)}")
            return None

    def process(self, audio_path, chunk_duration=30):
        """Process audio file with parallel chunk processing"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
            
        try:
            logging.info("Starting parallel diarization processing...")
            diarization = self.pipeline(
                audio_path,
                num_speakers=self.num_speakers,
                min_speakers=self.num_speakers,
                max_speakers=self.num_speakers
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