import os
from google import genai
from google.genai import types
from datasets import load_dataset, Audio
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()

# 1. Configure Gemini API (New SDK)
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("WARNING: GOOGLE_API_KEY not found.")

client = genai.Client(api_key=api_key)
MODEL_NAME = 'gemini-2.5-flash'

def get_gemini_transcription(audio_bytes, mime_type="audio/webm"):
    if not audio_bytes:
        return "N/A"
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                "Hãy chuyển đoạn âm thanh này thành văn bản tiếng Việt."
            ]
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# 2. Load dataset with streaming (Using HF_TOKEN for gated access)
print("Loading dataset...")
dataset = load_dataset('capleaf/viVoice', split='train', streaming=True, token=os.environ.get("HF_TOKEN"))
dataset = dataset.cast_column("audio", Audio(decode=False))

samples = []
max_samples = 2

for i, example in enumerate(dataset):
    if i >= max_samples:
        break
    audio_info = example.get('audio', {})
    audio_bytes = audio_info.get('bytes')
    path = audio_info.get('path', 'sample.webm')
    
    # Determine mime type
    mime_type = "audio/webm"
    if path.endswith(".wav"): mime_type = "audio/wav"
    elif path.endswith(".mp3"): mime_type = "audio/mp3"
    elif path.endswith(".ogg"): mime_type = "audio/ogg"
    
    print(f"[{i+1}/{max_samples}] Transcribing: {path}")
    transcription = get_gemini_transcription(audio_bytes, mime_type)
    samples.append({'index': i, 'gt': example.get('text', ''), 'gemini': transcription})

print("\nQuick Test Results:")
print(pd.DataFrame(samples))
