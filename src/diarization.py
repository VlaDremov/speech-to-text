from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio import Inference, Pipeline
import librosa
import noisereduce as nr
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os
import tempfile

class SpeakerDiarization:
    def __init__(self, auth_token: str, device: str = None, num_speakers: int = 2, 
                 min_speakers: int = None, max_speakers: int = None):
        """
        Initialize the diarization pipeline.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers if min_speakers is not None else num_speakers
        self.max_speakers = max_speakers if max_speakers is not None else num_speakers
        self.auth_token = auth_token
        try:
            logging.info(f"Initializing diarization pipeline on {self.device}")
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
            self.pipeline.to(torch.device(self.device))
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {e}")
            raise

    def preprocess_audio(self, audio_path: str, output_path: str = "processed_audio.wav") -> str:
        """
        Preprocess audio by reducing background noise.
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            logging.info("Audio loaded for preprocessing.")
            reduced_noise = nr.reduce_noise(y=y, sr=sr)
            sf.write(output_path, reduced_noise, sr)
            logging.info(f"Preprocessed audio saved to {output_path}.")
            return output_path
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            raise

    def get_speaker_segments(self, audio_path: str) -> list:
        """
        Run the diarization pipeline on a given file and return segments.
        """
        try:
            diarization = self.pipeline(audio_path, num_speakers=self.num_speakers,
                                          min_speakers=self.min_speakers, max_speakers=self.max_speakers)
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': f"SPEAKER_{speaker}"
                })
            return segments
        except Exception as e:
            logging.error(f"Error during diarization on {audio_path}: {e}")
            return []

    def extract_embeddings(self, audio_path: str, segments: list) -> np.ndarray:
        """
        Extract speaker embeddings for each segment.
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            embeddings = []
            speaker_embedding = Inference("pyannote/embedding", use_auth_token=self.auth_token)
            for segment in segments:
                start = int(segment['start'] * sr)
                end = int(segment['end'] * sr)
                segment_y = y[start:end]
                if len(segment_y) == 0:
                    continue
                segment_y_tensor = torch.tensor(segment_y).unsqueeze(0)
                embedding = speaker_embedding({"waveform": segment_y_tensor, "sample_rate": sr})
                embedding_np = embedding.data
                embeddings.append(embedding_np)
            embeddings = np.vstack(embeddings)
            return embeddings
        except Exception as e:
            logging.error(f"Error extracting embeddings: {e}")
            raise

    def cluster_segments(self, segments: list, embeddings: np.ndarray) -> list:
        """
        Cluster segments based on speaker embeddings to refine labels.
        """
        try:
            clustering = AgglomerativeClustering(n_clusters=self.num_speakers)
            labels = clustering.fit_predict(embeddings)
            for i, segment in enumerate(segments):
                segment['speaker'] = f"SPEAKER_{labels[i]}"
            return segments
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            raise

    def merge_segments(self, segments: list, gap_threshold: float = 0.5) -> list:
        """
        Merge consecutive segments of the same speaker if they are close enough.
        """
        if not segments:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if seg['speaker'] == last['speaker'] and (seg['start'] - last['end'] <= gap_threshold):
                last['end'] = seg['end']
            else:
                merged.append(seg)
        return merged

    def process_in_chunks(self, audio_path: str, chunk_duration: float, overlap_duration: float) -> list:
        """
        Split the audio into chunks and process each concurrently.
        """
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        logging.info(f"Total audio duration: {total_duration:.2f} seconds.")
        segments = []
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        step = chunk_samples - overlap_samples

        with tempfile.TemporaryDirectory() as temp_dir:
            futures = {}
            with ThreadPoolExecutor() as executor:
                for idx in range(0, len(y), step):
                    start_sample = idx
                    end_sample = idx + chunk_samples
                    chunk_y = y[start_sample:end_sample]
                    if len(chunk_y) == 0:
                        continue
                    chunk_file = os.path.join(temp_dir, f"chunk_{idx}.wav")
                    sf.write(chunk_file, chunk_y, sr)
                    future = executor.submit(self.get_speaker_segments, chunk_file)
                    futures[future] = idx / sr

                for future in as_completed(futures):
                    chunk_start_time = futures[future]
                    chunk_segments = future.result()
                    for seg in chunk_segments:
                        seg['start'] += chunk_start_time
                        seg['end'] += chunk_start_time
                    segments.extend(chunk_segments)

        segments.sort(key=lambda x: x['start'])
        merged_segments = self.merge_segments(segments)
        return merged_segments

    def process(self, audio_path: str, preprocess: bool = False, clustering_post: bool = False,
                chunk_duration: float = None, overlap_duration: float = 0) -> list:
        """
        Process an audio file with optional noise reduction, chunk processing, and clustering post-process.
        """
        try:
            if preprocess:
                logging.info("Preprocessing audio...")
                audio_path = self.preprocess_audio(audio_path)
            if chunk_duration:
                logging.info("Processing audio in chunks...")
                segments = self.process_in_chunks(audio_path, chunk_duration, overlap_duration)
            else:
                logging.info("Processing full audio...")
                segments = self.get_speaker_segments(audio_path)
            if not segments:
                logging.warning("No speaker segments detected. Check audio quality and parameters.")
                return segments
            if clustering_post:
                logging.info("Clustering speaker segments...")
                embeddings = self.extract_embeddings(audio_path, segments)
                segments = self.cluster_segments(segments, embeddings)
            merged_segments = self.merge_segments(segments)
            logging.info("Speaker diarization completed successfully.")
            return merged_segments
        except Exception as e:
            logging.error(f"Diarization processing failed: {e}")
            raise
