import os
import time
from google.cloud import speech_v1p1beta1 as speech

# Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "alpi-457807-086132945a9d.json"

def generate_audio_chunks(file_data, chunk_size=4096):
    while True:
        chunk = file_data.read(chunk_size)
        if not chunk:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)
        time.sleep(0.1)

def transcribe_stream(file_data, file_type):
    client = speech.SpeechClient()

    encoder_map = {
        "flac": (speech.RecognitionConfig.AudioEncoding.FLAC, 48000),
        "wav": (speech.RecognitionConfig.AudioEncoding.LINEAR16, 16000),
        "mp3": (speech.RecognitionConfig.AudioEncoding.MP3, 16000),
    }

    if file_type not in encoder_map:
        raise ValueError(f"Unsupported audio format: {file_type}")

    encoder, sample_rate = encoder_map[file_type]

    config = speech.RecognitionConfig(
        encoding=encoder,
        sample_rate_hertz=sample_rate,
        language_code="de-CH",
        use_enhanced=True,
        model="default",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )

    responses = client.streaming_recognize(
        config=streaming_config,
        requests=generate_audio_chunks(file_data)
    )

    return responses

def save_transcript(transcript_text, output_filename):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        f.write(transcript_text)
