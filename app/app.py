import os
import sys
import streamlit as st
from processing.speech_text_whisper import transcribe_audio, save_transcript

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
st.markdown("Transcribe your audio **in real-time** using Whisper.")

# File uploader
uploaded_file = st.file_uploader("🎵 Upload your audio file (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    st.audio(uploaded_file, format=f"audio/{file_type}")

    transcript_placeholder = st.empty()

    with st.spinner("🔄 Transcribing in real time..."):
        try:
            transcript_output = transcribe_audio(uploaded_file)
            file_name_without_ext = os.path.splitext(uploaded_file.name)[0]
            output_path = f"data/text/{file_name_without_ext}.txt"
            save_transcript(transcript_output, output_path)

            transcript_placeholder.markdown(
                f"<div class='transcript-box'>✅ <span class='highlight'>Final Transcript:</span>\n{transcript_output}</div>",
                unsafe_allow_html=True
            )
            st.success(f"🎉 Transcription complete! Saved as '{output_path}'.")

        except Exception as e:
            st.error(f"❌ {str(e)}")
else:
    st.markdown("📝 Please upload an audio file to begin transcription.")
