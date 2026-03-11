import os
import argparse
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def azure_openai_tts(text, out_file, deployment_name="gpt-4o-mini-tts", voice="alloy"):
    """Converts text to speech using Azure OpenAI TTS."""
    
    # Using credentials from environment
    api_key = os.environ.get("AZURE_OPENAI_API_KEY_SECONDARY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_SECONDARY")
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint
    )

    print(f"Synthesizing text: '{text[:50]}...'")
    print(f"Deployment: {deployment_name}, Voice: {voice}")
    
    start_time = time.perf_counter()
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice=voice, # alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        
        response.stream_to_file(out_file)
        proc_time = time.perf_counter() - start_time
        print(f"Audio content written to file: {out_file}")
        print(f"Processing time: {proc_time:.2f} seconds")
        return out_file
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Standalone Azure OpenAI TTS Demo")
    parser.add_argument("--text", type=str, default="Sáng nay, lúc 08:45, tại sự kiện AI Tech 2026, chuyên gia đã demo một bộ dataset gồm 1.024.500 dòng dữ liệu, chứa các từ khó như 'lắt léo', 'nghằn ngặt' và các thuật ngữ chuyên sâu như 'Recurrent Neural Networks' hay 'Hyperparameter Optimization', khiến hệ thống latency tăng thêm 15.5%.", help="Text to convert to speech")
    parser.add_argument("--output", type=str, default="output/openai_tts_demo1.mp3", help="Output audio file path")
    parser.add_argument("--deployment", type=str, default="gpt-4o-mini-tts", help="Azure deployment name")
    parser.add_argument("--voice", type=str, default="marin", help="Voice: alloy, echo, fable, onyx, nova, shimmer")
    
    args = parser.parse_args()

    azure_openai_tts(
        text=args.text, 
        out_file=args.output, 
        deployment_name=args.deployment, 
        voice=args.voice
    )

if __name__ == "__main__":
    main()
