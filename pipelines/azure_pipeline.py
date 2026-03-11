import os
import time
import argparse
import pandas as pd
import string
import io
import jiwer
from openai import AzureOpenAI
from datasets import load_dataset, Audio
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def azure_tts(client, deployment_name, text, out_file="temp_audio/temp_azure.mp3"):
    """Converts text to speech using Azure OpenAI TTS (gpt-4o-mini-tts)."""
    start_time = time.perf_counter()
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice="alloy", # alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        response.stream_to_file(out_file)
        proc_time = time.perf_counter() - start_time
        return out_file, proc_time
    except Exception as e:
        print(f"TTS Error: {e}")
        return None, 0.0

def run_azure_stt(client, deployment_name, audio_bytes, filename="audio.mp3"):
    """Transcribes audio using Azure OpenAI STT-HD (gpt-4o-transcribe)."""
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

def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI TTS -> STT Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="output/azure_pipeline_results.csv", help="Output CSV file")
    
    parser.add_argument("--stt_deployment", type=str, default="gpt-4o-transcribe", help="Deployment name for STT-HD")
    parser.add_argument("--tts_deployment", type=str, default="tts-hd", help="Deployment name for TTS")
    
    parser.add_argument("--api_version", type=str, default="2025-03-01-preview", help="Azure API Version")
    
    args = parser.parse_args()

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not api_key or not endpoint:
        print("ERROR: Azure credentials not found in .env")
        return

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)

    # Initialize Client
    client = AzureOpenAI(
        api_key=api_key,
        api_version=args.api_version,
        azure_endpoint=endpoint
    )

    print(f"Loading dataset '{args.dataset}'...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
        if "audio" in dataset.features:
            dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    results = []
    print(f"Running Azure pipeline for {args.limit} samples...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break
            
        original_text = example.get(args.text_col, '')
        if not original_text or len(original_text.strip()) == 0:
            continue

        audio_path = f"temp_audio/azure_tts_{i}.mp3"
        
        try:
            # 1. Azure TTS
            _, tts_time = azure_tts(client, args.tts_deployment, original_text, out_file=audio_path)
            
            if not os.path.exists(audio_path):
                raise Exception("TTS failed to generate audio file")

            # 2. Read audio
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            # 3. Azure STT-HD
            transcription, stt_time = run_azure_stt(client, args.stt_deployment, audio_bytes, filename=os.path.basename(audio_path))
            
            # 4. Metrics
            punct_remover = str.maketrans("", "", string.punctuation)
            norm_original = original_text.lower().translate(punct_remover)
            norm_transcription = transcription.lower().translate(punct_remover)
            
            wer = jiwer.wer(norm_original, norm_transcription)
            cer = jiwer.cer(norm_original, norm_transcription)
                
            results.append({
                'index': i,
                'original_text': original_text,
                'azure_transcription': transcription,
                'wer': wer,
                'cer': cer,
                'tts_time': tts_time,
                'stt_time': stt_time
            })
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            results.append({
                'index': i,
                'original_text': original_text,
                'azure_transcription': f"Error: {e}",
                'wer': 1.0,
                'cer': 1.0,
                'tts_time': 0,
                'stt_time': 0
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to {args.output}")
        print(f"Average WER: {df['wer'].mean():.4f}")
        print(f"Average CER: {df['cer'].mean():.4f}")
        print(f"Average TTS Time: {df['tts_time'].mean():.2f}s")
        print(f"Average STT Time: {df['stt_time'].mean():.2f}s")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
