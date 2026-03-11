import os
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def test_azure_tts_hd():
    # Credentials from the environment
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment_name = "tts-hd"
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint
    )

    test_text = "Và mọi chuyện thì chưa dừng lại ở đó."
    out_file = "debug_azure_tts_hd.mp3"

    print(f"Testing Azure TTS-HD...")
    print(f"Text: {test_text}")
    print(f"Voice: alloy")
    
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice="alloy",
            input=test_text
        )
        response.stream_to_file(out_file)
        print(f"Saved to {out_file}")
        print(f"File size: {os.path.getsize(out_file)} bytes")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_azure_tts_hd()
