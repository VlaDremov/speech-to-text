# Configuration settings for the speech-to-text application

# Path to the pre-trained model for speech recognition
MODEL_PATH = "path/to/your/model"

# Parameters for audio processing
AUDIO_SAMPLE_RATE = 16000  # Sample rate for audio processing
AUDIO_CHANNELS = 1          # Number of audio channels (1 for mono, 2 for stereo)

# Diarization settings
DIARIZATION_MIN_DURATION = 1.0  # Minimum duration (in seconds) for speaker segments
DIARIZATION_MAX_DURATION = 10.0  # Maximum duration (in seconds) for speaker segments

# Output settings
OUTPUT_FORMAT = "txt"  # Format for the output transcription file
OUTPUT_DIRECTORY = "output/"  # Directory to save output files