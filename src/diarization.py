from pyannote.audio import Pipeline
import torch
import logging
import time
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SimpleDiarization:
    def __init__(
        self, auth_token: str, device: str = "cpu", num_speakers: int = 2
    ) -> None:
        logging.info(f"Loading diarization pipeline on {device}")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=auth_token
        )
        self.pipeline.to(torch.device(device))
        self.num_speakers = num_speakers

    def process(self, audio_path: str) -> List[Dict]:
        logging.info(f"Processing audio file: {audio_path}")
        start_time = time.time()
        try:
            with torch.no_grad():
                diarization = self.pipeline(audio_path, num_speakers=self.num_speakers)
        except Exception as e:
            logging.error(f"Error processing {audio_path}: {e}")
            raise

        segments = [
            {"start": turn.start, "end": turn.end, "speaker": f"SPEAKER_{speaker}"}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        logging.info(f"Found {len(segments)} segments before merging.")
        merged_segments = segments
        logging.info(f"Processing completed in {time.time() - start_time:.2f} seconds.")
        return merged_segments
