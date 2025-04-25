import os
import time
import streamlit as st
from google.cloud import speech_v1p1beta1 as speech

# Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "alpi-457807-086132945a9d.json"

# Page configuration
st.set_page_config(
    page_title="🎙️ Real-Time Audio Transcription",
    page_icon="🎧",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
        html, body {
            background-color: #f2f4f8;
        }

        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .stFileUploader > label {
            font-size: 1.1rem;
            font-weight: bold;
            color: #444;
        }

        .transcript-box {
            padding: 1.5rem;
            background-color: #f7fafc;
            border-left: 5px solid #3b82f6;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.95rem;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #111827;
            margin-top: 1rem;
        }

        .highlight {
            font-weight: 600;
            color: #3b82f6;
        }

        .loader {
            border: 6px solid #f3f3f3;
            border-top: 6px solid #3b82f6;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎙️ Real-Time Audio Transcription")
st.markdown("Transcribe your audio **in real-time** using Google Speech-to-Text.")

# File uploader
uploaded_file = st.file_uploader("🎵 Upload your audio file (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

# Audio chunk generator
def generate_audio_chunks(file_data, chunk_size=4096):
    while True:
        chunk = file_data.read(chunk_size)
        if not chunk:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)
        time.sleep(0.1)

# Main transcription logic
def stream_transcribe(file_data, file_type):
    client = speech.SpeechClient()

    encoder_map = {
        "flac": (speech.RecognitionConfig.AudioEncoding.FLAC, 48000),
        "wav": (speech.RecognitionConfig.AudioEncoding.LINEAR16, 16000),
        "mp3": (speech.RecognitionConfig.AudioEncoding.MP3, 16000),
    }

    if file_type not in encoder_map:
        st.error("❌ Unsupported audio format.")
        return

    encoder, sample_rate = encoder_map[file_type]

    config = speech.RecognitionConfig(
        encoding=encoder,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
        use_enhanced=True,
        model="phone_call",
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

    transcript_output = ""
    transcript_placeholder = st.empty()

    for response in responses:
        for result in response.results:
            if result.is_final:
                transcript_output += result.alternatives[0].transcript + " "
            else:
                interim = result.alternatives[0].transcript
                words = interim.split()
                transcript_placeholder.markdown(
                    f"<div class='transcript-box'>📝 <span class='highlight'>Live:</span> {transcript_output + ' '.join(words[-1:])}</div>",
                    unsafe_allow_html=True
                )

    # Final transcript
    if transcript_output:
        transcript_placeholder.markdown(
            f"<div class='transcript-box'>✅ <span class='highlight'>Final Transcript:</span>\n{transcript_output}</div>",
            unsafe_allow_html=True
        )

# App logic
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    st.audio(uploaded_file, format=f"audio/{file_type}")
    with st.spinner("🔄 Transcribing in real time..."):
        stream_transcribe(uploaded_file, file_type)
    st.success("🎉 Transcription complete!")
else:
    st.markdown("📝 Please upload an audio file to begin transcription.")
