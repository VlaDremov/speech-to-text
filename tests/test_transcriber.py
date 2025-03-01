def test_transcriber():
    from src.transcriber import Transcriber

    # Initialize the Transcriber
    transcriber = Transcriber()

    # Test case for transcribing a sample audio segment
    audio_segment = "path/to/sample_audio.m4a"
    expected_output = "Expected transcription text for the audio segment."

    # Perform transcription
    output = transcriber.transcribe(audio_segment)

    # Assert that the output matches the expected output
    assert output == expected_output, f"Expected: {expected_output}, but got: {output}"

    # Additional test cases can be added here
    # For example, testing with different audio segments or edge cases.