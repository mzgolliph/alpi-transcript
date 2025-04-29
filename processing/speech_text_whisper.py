import os
import whisper
from openai import OpenAI
# Load Whisper model
model = whisper.load_model("base")  # You can use "small", "medium", or "large" depending on your needs

from dotenv import load_dotenv
load_dotenv()

# Add this in processing/speech_text_whisper.py


# You can configure OpenAI client here if needed:

def improve_transcript(raw_transcript, language_hint="Swiss German"):
    """
    Improve transcript using an LLM by cleaning it up, fixing grammar, and formatting.
    """
    prompt = f"""The following text is a raw transcription of an audio file spoken in {language_hint}. 
    Please:
    - Correct transcription errors
    - Improve grammar and punctuation
    - Maintain the original meaning
    - Make it readable and natural

    Raw Transcript:
    {raw_transcript}

    Improved Transcript:"""
    client = OpenAI()
    response =client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo" if you prefer
        messages=[
            {"role": "system", "content": "You are a transcription refinement assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()




def transcribe_audio(file_path):
    """
    Transcribe audio using Whisper model.
    """
    
    

    client = OpenAI()
    audio_file = open(file_path, "rb")

    transcription = client.audio.transcriptions.create(
    file=audio_file,
    model="whisper-1",
    response_format="text",
    prompt="Transcribe the following Swiss German audio file. Please provide a clean and accurate transcription.",
    )

    return transcription

def save_transcript(transcript_text, output_filename):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        f.write(transcript_text)
