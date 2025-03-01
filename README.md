# speech-to-text/README.md

# Speech-to-Text with Speaker Diarization

This project converts speech to text with speaker diarization using Whisper and Pyannote.audio.

## Features
- Audio file processing (M4A format support)
- Speaker diarization using Pyannote.audio
- Speech recognition using OpenAI's Whisper
- GPU acceleration support
- Structured output with speaker identification

## Project Structure

```
speech-to-text
├── src
│   ├── main.py
│   ├── audio_processor.py
│   ├── diarization.py
│   ├── transcriber.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   └── file_utils.py
│   └── config
│       └── settings.py
├── tests
│   ├── __init__.py
│   ├── test_audio_processor.py
│   ├── test_diarization.py
│   └── test_transcriber.py
├── input
│   └── .gitkeep
├── output
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites
- Python 3.9+
- FFmpeg
- CUDA-capable GPU (optional)
- HuggingFace account and access token

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd speech-to-text
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Setup

1. Create a `.env` file in the project root:
   ```plaintext
   HUGGING_FACE_TOKEN=your_token_here
   ```

2. Replace `your_token_here` with your HuggingFace access token

## Usage

1. Place your `.m4a` audio files in the `input` directory.
2. Run the application:
   ```
   python src/main.py
   ```
3. The output will be saved in the `output` directory as a structured text file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.