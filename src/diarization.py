from pyannote.audio import Pipeline
import torch
import logging
import time
from typing import List, Dict
from pyannote.audio.pipelines.utils.hook import ProgressHook

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

        logging.info(f"Loading VAD pipeline on {device}")
        self.vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=auth_token
        )
        self.vad_pipeline.to(torch.device(device))

    def merge_segments_with_overlap(
        self, segments: List[Dict], tolerance: float = 0.3, min_duration: float = 0.3
    ) -> List[Dict]:
        """
        Merge segments while detecting overlaps.
        When an overlap is detected, create an extra segment for the overlapping region.
        """
        if not segments:
            return segments

        # Ensure segments are sorted by start time.
        segments.sort(key=lambda x: x["start"])
        merged = []

        for seg in segments:
            if not merged:
                merged.append(seg)
                continue

            last = merged[-1]
            # Check if there is an overlap.
            if seg["start"] < last["end"]:
                # Compute overlap boundaries.
                overlap_start = seg["start"]
                overlap_end = min(last["end"], seg["end"])
                overlap_duration = overlap_end - overlap_start

                # Only create an overlap segment if it meets a minimum duration.
                if overlap_duration >= min_duration:
                    # Create an overlap segment that combines both speaker labels.
                    overlap_seg = {
                        "start": overlap_start,
                        "end": overlap_end,
                        "speaker": f"{last['speaker']}_OVERLAP_{seg['speaker']}",
                    }
                    # Optionally, trim the original segments to remove the overlapping part.
                    last["end"] = overlap_start
                    seg["start"] = overlap_end

                    merged.append(overlap_seg)
                    # Check if the trimmed current segment still has enough duration.
                    if seg["end"] - seg["start"] >= min_duration:
                        merged.append(seg)
                else:
                    # If the overlap is too short, decide how to handle it:
                    # If the speakers are the same, merge them.
                    if last["speaker"] == seg["speaker"]:
                        last["end"] = max(last["end"], seg["end"])
                    else:
                        merged.append(seg)
            else:
                # No overlap detected; simply add the segment.
                merged.append(seg)

        # Optionally, merge segments with small gaps (within tolerance) if same speaker.
        smoothed = [merged[0]]
        for seg in merged[1:]:
            last = smoothed[-1]
            gap = seg["start"] - last["end"]
            if gap <= tolerance and last["speaker"] == seg["speaker"]:
                last["end"] = seg["end"]
            else:
                smoothed.append(seg)
        return smoothed

    def process(self, audio_path: str) -> List[Dict]:
        logging.info(f"Processing audio file: {audio_path}")
        start_time = time.time()

        # Run diarization on the audio.
        with torch.no_grad():
            diarization = self.pipeline(audio_path, num_speakers=self.num_speakers)

        segments = [
            {"start": turn.start, "end": turn.end, "speaker": f"SPEAKER_{speaker}"}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        logging.info(f"Found {len(segments)} raw diarization segments.")

        # Run VAD to extract speech regions.
        logging.info("Running voice activity detection (VAD)...")
        vad_result = self.vad_pipeline(audio_path)
        speech_regions = vad_result.get_timeline().support()

        # Filter segments that have little or no overlap with speech.
        filtered_segments = []
        for seg in segments:
            for speech in speech_regions:
                if seg["start"] < speech.end and seg["end"] > speech.start:
                    filtered_segments.append(seg)
                    break
        logging.info(f"{len(filtered_segments)} segments remain after VAD filtering.")
        merged_segments = self.merge_segments_with_overlap(segments)
        logging.info(
            f"{len(merged_segments)} segments after merging with overlap detection."
        )
        logging.info(f"Processing completed in {time.time() - start_time:.2f} seconds.")
        return merged_segments
