from pyannote.audio import Pipeline
import librosa
import noisereduce as nr
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import tempfile  # added for temporary file handling

class SpeakerDiarization:
    def __init__(self, auth_token, device=None, num_speakers=2, min_speakers=None, max_speakers=None):
        """
        Initialize the diarization pipeline.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers if min_speakers is not None else num_speakers
        self.max_speakers = max_speakers if max_speakers is not None else num_speakers
        try:
            logging.info(f"Initializing diarization pipeline on {self.device}")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization", use_auth_token=auth_token
            )
            self.pipeline.to(torch.device(self.device))
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {e}")
            raise

    def preprocess_audio(self, audio_path, output_path="processed_audio.wav"):
        """
        Preprocess audio by reducing background noise.
        """
        y, sr = librosa.load(audio_path, sr=None)
        reduced_noise = nr.reduce_noise(y=y, sr=sr)
        sf.write(output_path, reduced_noise, sr)
        return output_path

    def get_speaker_segments(self, audio_path):
        """
        Run the diarization pipeline on a given file and return segments.
        """
        diarization = self.pipeline(
            audio_path,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': f"SPEAKER_{speaker}"
            })
        return segments

    def extract_embeddings(self, audio_path, segments):
        """
        Dummy implementation: extract speaker embeddings for each segment.
        Replace with actual embedding extraction if available.
        """
        return np.random.rand(len(segments), 128)

    def cluster_segments(self, segments, embeddings):
        """
        Cluster segments based on speaker embeddings to refine labels.
        """
        clustering = AgglomerativeClustering(n_clusters=self.num_speakers)
        labels = clustering.fit_predict(embeddings)
        for i, segment in enumerate(segments):
            segment['speaker'] = f"SPEAKER_{labels[i]}"
        return segments

    def process_in_chunks(self, audio_path, chunk_duration, overlap_duration):
        """
        Split the audio into chunks and process each concurrently.
        """
        # Load full audio with librosa for splitting
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        segments = []
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        step = chunk_samples - overlap_samples
        chunk_indices = range(0, len(y), step)
        
        temp_files = []
        futures = []
        with ThreadPoolExecutor() as executor:
            for idx in chunk_indices:
                start_sample = idx
                end_sample = idx + chunk_samples
                chunk_y = y[start_sample:end_sample]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    sf.write(tmp.name, chunk_y, sr)
                    temp_files.append((tmp.name, idx/sr))  # store file and chunk start time
                    futures.append(executor.submit(self.get_speaker_segments, tmp.name))
            
            for (temp_file, start_time), future in zip(temp_files, futures):
                chunk_segments = future.result()
                # Adjust timestamps based on chunk start time
                for seg in chunk_segments:
                    seg['start'] += start_time
                    seg['end'] += start_time
                segments.extend(chunk_segments)
        
        # Cleanup temporary chunk files
        for temp_file, _ in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        segments.sort(key=lambda x: x['start'])
        return segments

    def process(self, audio_path, preprocess=False, clustering_post=False, chunk_duration=None, overlap_duration=0):
        """
        Process an audio file with optional noise reduction, chunk processing, and clustering post-process.
        """
        try:
            logging.info("Starting speaker diarization...")
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
                logging.info("Refining speaker labels...")
                segments = self.cluster_segments(segments, embeddings)
            return segments
        except Exception as e:
            logging.error(f"Diarization processing failed: {e}")
            raise
