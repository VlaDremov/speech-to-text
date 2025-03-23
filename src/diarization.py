from pyannote.audio import Pipeline
import torch
import logging
import time
from typing import List, Dict
import numpy as np
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster, linkage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SimpleDiarization:
    def __init__(
        self,
        auth_token: str,
        device: str = "cpu",
        num_speakers: int = 2,
        min_speaker_duration: float = 0.1,
        clustering_threshold: float = 0.6,
    ) -> None:
        logging.info(f"Loading diarization pipeline on {device}")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=auth_token
        )
        self.pipeline.to(torch.device(device))
        self.num_speakers = num_speakers
        self.min_speaker_duration = min_speaker_duration
        self.clustering_threshold = clustering_threshold

        logging.info(f"Loading VAD pipeline on {device}")
        self.vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=auth_token
        )
        self.vad_pipeline.to(torch.device(device))

        # Try to load speaker embedding model, but gracefully handle failure
        self.embedding_model = None
        try:
            logging.info(f"Loading speaker embedding model on {device}")
            self.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb", device=torch.device(device)
            )
            logging.info("Speaker embedding model loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load speaker embedding model: {e}")
            logging.warning("Continuing without speaker embedding refinement")

    def preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio to improve diarization quality.
        Returns the path to the preprocessed audio.
        """
        # TODO: Implement audio preprocessing steps like:
        # - Noise reduction
        # - Normalization
        # - Silence removal
        # - Resampling if needed
        return audio_path

    def refine_speaker_clusters(
        self, segments: List[Dict], audio_path: str
    ) -> List[Dict]:
        """
        Refine speaker clusters using speaker embeddings.
        Falls back to original segments if embedding model is unavailable.
        """
        if not segments:
            return segments

        # Skip refinement if embedding model isn't available
        if self.embedding_model is None:
            logging.warning("Skipping speaker refinement (no embedding model)")
            return segments

        # Extract embeddings for each segment
        embeddings = []
        for seg in segments:
            try:
                embedding = self.embedding_model.audio_path_to_embedding(
                    audio_path, start=seg["start"], end=seg["end"]
                )
                embeddings.append(embedding)
            except Exception as e:
                logging.warning(f"Failed to extract embedding: {e}")
                embeddings.append(None)

        # Filter out segments where embedding extraction failed
        valid_segments = [
            seg for seg, emb in zip(segments, embeddings) if emb is not None
        ]
        valid_embeddings = [emb for emb in embeddings if emb is not None]

        if not valid_embeddings:
            return segments

        # Perform clustering with our custom threshold
        embeddings_array = np.vstack(valid_embeddings)
        
        # Calculate distance matrix
        distances = cdist(embeddings_array, embeddings_array)
        
        # Use distance-based clustering with our threshold if num_speakers is None
        if self.num_speakers is None:
            # Apply linkage with average method
            Z = linkage(distances, method='average')
            
            # Apply threshold to get clusters
            # Convert our threshold parameter: lower clustering_threshold means higher distance cutoff
            distance_cutoff = (1.0 - self.clustering_threshold) * distances.max()
            clusters = fcluster(Z, t=distance_cutoff, criterion='distance')
            
            # Adjust cluster IDs to be 0-indexed for consistency
            clusters = clusters - 1
        else:
            # Use scikit-learn's clustering with fixed number of clusters
            clustering = AgglomerativeClustering(
                n_clusters=self.num_speakers, 
                metric='precomputed', 
                linkage='average'  # Use average linkage for consistency
            )
            clusters = clustering.fit_predict(distances)

        # Update speaker labels
        for seg, cluster_id in zip(valid_segments, clusters):
            seg["speaker"] = f"SPEAKER_{cluster_id}"

        return valid_segments

    def merge_segments_with_overlap(
        self, segments: List[Dict], tolerance: float = 0.3, min_duration: float = 0.3
    ) -> List[Dict]:
        """
        Merge segments while detecting overlaps.
        When an overlap is detected, create an extra segment for the overlapping region.
        """
        if not segments:
            return segments

        # Ensure segments are sorted by start time
        segments.sort(key=lambda x: x["start"])
        merged = []

        for seg in segments:
            if not merged:
                merged.append(seg)
                continue

            last = merged[-1]
            # Check if there is an overlap
            if seg["start"] < last["end"]:
                # Compute overlap boundaries
                overlap_start = seg["start"]
                overlap_end = min(last["end"], seg["end"])
                overlap_duration = overlap_end - overlap_start

                # Only create an overlap segment if it meets minimum duration
                if overlap_duration >= min_duration:
                    # Create an overlap segment that combines both speaker labels
                    overlap_seg = {
                        "start": overlap_start,
                        "end": overlap_end,
                        "speaker": f"{last['speaker']}_OVERLAP_{seg['speaker']}",
                    }
                    # Trim the original segments to remove the overlapping part
                    last["end"] = overlap_start
                    seg["start"] = overlap_end

                    merged.append(overlap_seg)
                    # Check if the trimmed current segment still has enough
                    # duration
                    if seg["end"] - seg["start"] >= min_duration:
                        merged.append(seg)
                else:
                    # If the overlap is too short, merge if same speaker
                    if last["speaker"] == seg["speaker"]:
                        last["end"] = max(last["end"], seg["end"])
                    else:
                        merged.append(seg)
            else:
                merged.append(seg)

        # Initialize smoothed segments list with first segment
        smoothed = [merged[0]]

        # Iterate through remaining segments
        for seg in merged[1:]:
            last = smoothed[-1]
            # Calculate gap between current and last segment
            gap = seg["start"] - last["end"]

            # If gap is small enough and speakers match,
            # extend the last segment to include this one
            if gap <= tolerance and last["speaker"] == seg["speaker"]:
                last["end"] = seg["end"]
            # Otherwise add as new segment
            else:
                smoothed.append(seg)

        return smoothed

    def process(self, audio_path: str) -> List[Dict]:
        logging.info(f"Processing audio file: {audio_path}")
        start_time = time.time()

        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_path)

        try:
            # Run diarization on the audio with clustering threshold
            with torch.no_grad():
                # Try different approaches to set clustering parameters
                try:
                    # Configure the pipeline with the clustering threshold
                    # This is specific to pyannote/speaker-diarization-3.1
                    if hasattr(self.pipeline, "instantiate"):
                        # For newer versions of pyannote
                        self.pipeline.instantiate({
                            "clustering": {
                                "method": "average",  # Valid methods: single, complete, average, weighted, centroid, median, ward
                                "threshold": self.clustering_threshold,
                            }
                        })
                        diarization = self.pipeline(processed_audio, num_speakers=self.num_speakers)
                    else:
                        # For older versions, modify the pipeline parameters directly
                        if hasattr(self.pipeline, "clustering") and hasattr(self.pipeline.clustering, "threshold"):
                            original_threshold = self.pipeline.clustering.threshold
                            self.pipeline.clustering.threshold = self.clustering_threshold
                            diarization = self.pipeline(processed_audio, num_speakers=self.num_speakers)
                            # Restore original threshold
                            self.pipeline.clustering.threshold = original_threshold
                        else:
                            # If we can't directly set threshold, just run with default
                            logging.warning("Could not set clustering threshold, using default")
                            diarization = self.pipeline(processed_audio, num_speakers=self.num_speakers)
                except Exception as e:
                    # Fall back to default parameters if customization fails
                    logging.warning(f"Error setting custom clustering parameters: {e}")
                    logging.warning("Falling back to default pipeline settings")
                    diarization = self.pipeline(processed_audio, num_speakers=self.num_speakers)

            segments = [
                {"start": turn.start, "end": turn.end, "speaker": f"SPEAKER_{speaker}"}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            logging.info(f"Found {len(segments)} raw diarization segments.")

            # Run VAD to extract speech regions
            logging.info("Running voice activity detection (VAD)...")
            vad_result = self.vad_pipeline(processed_audio)
            speech_regions = vad_result.get_timeline().support()

            # Filter segments that have little or no overlap with speech
            filtered_segments = []
            for seg in segments:
                for speech in speech_regions:
                    if seg["start"] < speech.end and seg["end"] > speech.start:
                        filtered_segments.append(seg)
                        break
            logging.info(f"{len(filtered_segments)} segments remain after VAD filtering.")

            # Refine speaker clusters using embeddings
            refined_segments = self.refine_speaker_clusters(
                filtered_segments, processed_audio
            )
            logging.info(f"{len(refined_segments)} segments after speaker refinement.")

            # Merge segments with overlap detection
            merged_segments = self.merge_segments_with_overlap(refined_segments)
            logging.info(
                f"{len(merged_segments)} segments after merging with overlap detection."
            )

            # Filter out very short segments
            final_segments = [
                seg
                for seg in merged_segments
                if seg["end"] - seg["start"] >= self.min_speaker_duration
            ]
            logging.info(f"{len(final_segments)} segments after duration filtering.")

            logging.info(f"Processing completed in {time.time() - start_time:.2f} seconds.")
            return final_segments
            
        except Exception as e:
            logging.error(f"Error in diarization processing: {str(e)}")
            # Return minimal segments to avoid breaking the pipeline
            logging.warning("Returning minimal speaker segmentation as fallback")
            # Create at least one speaker segment for the whole audio
            import soundfile as sf
            info = sf.info(audio_path)
            duration = info.duration
            return [{"start": 0.0, "end": duration, "speaker": "SPEAKER_FALLBACK"}]
