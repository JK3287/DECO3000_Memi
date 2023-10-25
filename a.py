# streamlit run '/Users/yungvenuz/Documents/Uni/Year 3 DC/DECO3000/DECO3000_Memi/a.py'

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Record the audio
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Save the recorded audio to a file
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_bytes)

    # Transcribe the audio using OpenAI's Whisper ASR API
    with open("recorded_audio.wav", "rb") as f:
        transcript = openai.Audio.translate(model="whisper-1", file=f, response_format="text")

    # Display the transcript
    st.write(transcript)