import os
import time
import argparse
from openai import AzureOpenAI
from dotenv import load_dotenv
import soundfile as sf

# Load environment variables from .env file
load_dotenv()

def call_tts_hd(text, output_file="output_ttshd.mp3", voice="alloy"):
    """
    Calls Azure OpenAI TTS-HD model to convert text to speech.
    """
    # Credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = "tts-hd" # Ensure this matches your Azure deployment name
    
    if not api_key or not endpoint:
        print("Error: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not found in environment.")
        return

    print(f"Initializing Azure OpenAI client...")
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint
    )

    print(f"Calling TTS-HD...")
    print(f"  Text: {text}")
    print(f"  Voice: {voice}")
    print(f"  Model: {deployment_name}")

    start_time = time.perf_counter()
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice=voice,
            input=text,
            response_format="mp3"
        )
        
        # Save to file
        response.stream_to_file(output_file)
        
        duration = time.perf_counter() - start_time
        print(f"Success! Saved to: {output_file}")
        print(f"Processing time: {duration:.2f}s")
        
        # Check audio properties if possible
        try:
            data, samplerate = sf.read(output_file)
            audio_duration = len(data) / samplerate
            print(f"Audio Duration: {audio_duration:.2f}s")
            print(f"Sample Rate: {samplerate}Hz")
        except Exception as e:
            print(f"Note: Could not read audio properties directly (likely due to MP3 format): {e}")
            print(f"File Size: {os.path.getsize(output_file)} bytes")

    except Exception as e:
        print(f"Error during TTS call: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call Azure OpenAI TTS-HD")
    parser.add_argument("--text", type=str, default="Và mọi chuyện thì chưa dừng lại ở đó.", help="Text to convert")
    parser.add_argument("--output", type=str, default="test_ttshd.mp3", help="Output file path")
    parser.add_argument("--voice", type=str, default="nova", help="Voice to use (alloy, echo, fable, onyx, nova, shimmer)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    call_tts_hd(args.text, args.output, args.voice)
