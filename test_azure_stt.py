import os
import time
import argparse
import pandas as pd
import base64
from openai import AzureOpenAI
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
        # Fallback if soundfile fails (e.g. mp3 without backend)
        return 0.0

def run_azure_whisper(client, deployment_name, audio_bytes, filename="audio.wav"):
    """Transcribes audio using Azure OpenAI Whisper-1."""
    if not audio_bytes:
        return "N/A", 0.0

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    start_time = time.perf_counter()
    try:
        response = client.audio.transcriptions.create(
            model=deployment_name,
            file=audio_file,
            language="vi"
        )
        proc_time = time.perf_counter() - start_time
        return response.text.strip(), proc_time
    except Exception as e:
        proc_time = time.perf_counter() - start_time
        return f"Error: {str(e)}", proc_time

def run_azure_gpt4o(client, deployment_name, audio_bytes, filename="audio.wav"):
    """Transcribes audio using Azure OpenAI GPT-4o specialized transcription deployment."""
    if not audio_bytes:
        return "N/A", 0.0

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    start_time = time.perf_counter()
    try:
        # User specified this uses the /audio/transcriptions endpoint
        response = client.audio.transcriptions.create(
            model=deployment_name,
            file=audio_file,
            language="vi"
        )
        proc_time = time.perf_counter() - start_time
        return response.text.strip(), proc_time
    except Exception as e:
        proc_time = time.perf_counter() - start_time
        return f"Error: {str(e)}", proc_time

def main():
    parser = argparse.ArgumentParser(description="Benchmarking Azure OpenAI STT")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--audio_col", type=str, default="audio", help="Audio column name")
    parser.add_argument("--text_col", type=str, default="text", help="Ground truth text column name")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to test")
    
    # Model Deployments
    parser.add_argument("--whisper_deployment", type=str, default="whisper", help="Deployment name for Whisper")
    parser.add_argument("--gpt4o_deployment", type=str, default="gpt-4o-transcribe", help="Deployment name for GPT-4o")
    
    parser.add_argument("--whisper_api_version", type=str, default="2024-06-01", help="Azure API Version for Whisper")
    parser.add_argument("--gpt4o_api_version", type=str, default="2025-03-01-preview", help="Azure API Version for GPT-4o")
    parser.add_argument("--whisper_method", type=str, choices=["transcriptions", "translations"], default="transcriptions", help="Whisper methodology")
    
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    
    args = parser.parse_args()

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not api_key or not endpoint:
        print("ERROR: Azure credentials not found in .env")
        return

    # Clients
    client_whisper = AzureOpenAI(api_key=api_key, api_version=args.whisper_api_version, azure_endpoint=endpoint)
    client_gpt4o = AzureOpenAI(api_key=api_key, api_version=args.gpt4o_api_version, azure_endpoint=endpoint)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset '{args.dataset}'...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
        dataset = dataset.cast_column(args.audio_col, Audio(decode=False))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    whisper_results = []
    gpt4o_results = []
    
    print(f"Testing {args.limit} samples...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break
            
        audio_info = example.get(args.audio_col, {})
        audio_bytes = audio_info.get('bytes')
        path = audio_info.get('path', f"sample_{i}.wav")
        audio_duration = get_audio_duration(audio_bytes)
        ground_truth = example.get(args.text_col, '')
        
        mime_type = "audio/wav"
        if path.lower().endswith(".mp3"): mime_type = "audio/mpeg"
        elif path.lower().endswith(".m4a"): mime_type = "audio/m4a"
        
        common_data = {
            'index': i,
            'dataset': args.dataset,
            'path': path,
            'audio_duration': audio_duration,
            'ground_truth': ground_truth
        }

        # Run Whisper
        if args.whisper_deployment:
            if args.whisper_method == "translations":
                start_time = time.perf_counter()
                try:
                    audio_file = io.BytesIO(audio_bytes)
                    audio_file.name = path
                    response = client_whisper.audio.translations.create(model=args.whisper_deployment, file=audio_file)
                    trans = response.text.strip()
                    pt = time.perf_counter() - start_time
                except Exception as e:
                    trans, pt = f"Error: {e}", time.perf_counter() - start_time
            else:
                trans, pt = run_azure_whisper(client_whisper, args.whisper_deployment, audio_bytes, filename=path)
            
            w_row = common_data.copy()
            w_row.update({'whisper_transcription': trans, 'processing_time': pt, 'rtf': pt / audio_duration if audio_duration > 0 else 0})
            whisper_results.append(w_row)

        # Run GPT-4o
        if args.gpt4o_deployment:
            trans, pt = run_azure_gpt4o(client_gpt4o, args.gpt4o_deployment, audio_bytes, filename=path)
            g_row = common_data.copy()
            g_row.update({'gpt4o_transcription': trans, 'processing_time': pt, 'rtf': pt / audio_duration if audio_duration > 0 else 0})
            gpt4o_results.append(g_row)

    # Save Whisper results
    if whisper_results:
        w_output = os.path.join(args.output_dir, f"azure_whisper_results.csv")
        pd.DataFrame(whisper_results).to_csv(w_output, index=False, encoding='utf-8-sig')
        print(f"Whisper results saved to {w_output}")

    # Save GPT-4o results
    if gpt4o_results:
        g_output = os.path.join(args.output_dir, f"azure_gpt4o_results.csv")
        pd.DataFrame(gpt4o_results).to_csv(g_output, index=False, encoding='utf-8-sig')
        print(f"GPT-4o results saved to {g_output}")

if __name__ == "__main__":
    main()
