def test_diarization():
    from src.diarization import Diarization

    # Create an instance of the Diarization class
    diarization = Diarization()

    # Test with a sample audio file
    audio_file = "path/to/sample.m4a"
    expected_output = {
        "speaker_1": ["Hello, how are you?"],
        "speaker_2": ["I'm fine, thank you!"],
    }

    # Perform diarization
    result = diarization.perform_diarization(audio_file)

    # Assert that the result matches the expected output
    assert result == expected_output, f"Expected {expected_output}, but got {result}"