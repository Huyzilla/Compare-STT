import os
import time
import argparse
import pandas as pd
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from datasets import load_dataset, Audio
from tqdm import tqdm
from dotenv import load_dotenv
import io
import soundfile as sf

# Load environment variables
load_dotenv()

def get_audio_duration(audio_bytes):
    try:
        with io.BytesIO(audio_bytes) as bio:
            data, samplerate = sf.read(bio)
            return len(data) / samplerate
    except Exception:
        return 0.0

def run_google_stt(client, project_id, location, audio_bytes):
    if not audio_bytes:
        return "N/A", 0.0

    # Chirp v2 configuration
    # Note: Chirp v2 is usually accessed via recognizers in v2 API
    # For a simple walkthrough, we'll use a direct request if possible or assume a recognizer
    parent = f"projects/{project_id}/locations/{location}"
    
    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        model="chirp_2", 
        language_codes=["vi-VN"],
        features=speech_v2.RecognitionFeatures(
            enable_word_time_offsets=False,
        ),
    )
    
    request = speech_v2.RecognizeRequest(
        recognizer=f"{parent}/recognizers/_", # Using default recognizer
        config=config,
        content=audio_bytes,
    )

    start_time = time.perf_counter()
    try:
        response = client.recognize(request=request)
        proc_time = time.perf_counter() - start_time
        
        # Extract transcript
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        return transcript.strip(), proc_time
    except Exception as e:
        proc_time = time.perf_counter() - start_time
        return f"Error: {str(e)}", proc_time

def main():
    parser = argparse.ArgumentParser(description="Benchmarking Google Cloud STT (Chirp)")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--audio_col", type=str, default="audio", help="Audio column name")
    parser.add_argument("--text_col", type=str, default="text", help="Ground truth text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="output/google_results.csv", help="Output CSV file")
    parser.add_argument("--location", type=str, default=None, help="Google Cloud location")
    
    args = parser.parse_args()

    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    location = args.location or os.environ.get("GOOGLE_LOCATION", "us-central1")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not project_id:
        print("ERROR: GOOGLE_PROJECT_ID not found in .env")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize client
    client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
    )

    print(f"Loading dataset '{args.dataset}'...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
        dataset = dataset.cast_column(args.audio_col, Audio(decode=False))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    results = []
    print(f"Testing {args.limit} samples with Google Chirp 3...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break
            
        audio_info = example.get(args.audio_col, {})
        audio_bytes = audio_info.get('bytes')
        path = audio_info.get('path', f"sample_{i}")
        
        audio_duration = get_audio_duration(audio_bytes)
        
        transcription, proc_time = run_google_stt(client, project_id, location, audio_bytes)
            
        results.append({
            'index': i,
            'dataset': args.dataset,
            'path': path,
            'audio_duration': audio_duration,
            'processing_time': proc_time,
            'rtf': proc_time / audio_duration if audio_duration > 0 else 0,
            'ground_truth': example.get(args.text_col, ''),
            'google_transcription': transcription
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {args.output}")
    print(f"Average RTF: {df['rtf'].mean():.4f}")

if __name__ == "__main__":
    main()
