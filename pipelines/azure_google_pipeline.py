import os
import time
import argparse
import pandas as pd
import string
import io
import jiwer
from openai import AzureOpenAI
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from datasets import load_dataset, Audio
from tqdm import tqdm
from dotenv import load_dotenv
import soundfile as sf

# Load environment variables
load_dotenv()

def get_audio_duration(file_path):
    try:
        data, samplerate = sf.read(file_path)
        return len(data) / samplerate
    except Exception:
        return 0.0

def azure_tts(client, deployment_name, text, out_file="temp_audio/temp_azure.mp3"):
    """Converts text to speech using Azure OpenAI TTS (gpt-4o-mini-tts)."""
    start_time = time.perf_counter()
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice="nova", # alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        response.stream_to_file(out_file)
        proc_time = time.perf_counter() - start_time
        return out_file, proc_time
    except Exception as e:
        print(f"TTS Error: {e}")
        return None, 0.0

def run_google_stt(stt_client, project_id, location, audio_bytes):
    """Transcribes audio using Google Cloud STT V2 (Chirp)."""
    if not audio_bytes:
        return "N/A", 0.0

    parent = f"projects/{project_id}/locations/{location}"
    
    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        model="chirp_2", 
        language_codes=["vi-VN"],
        features=speech_v2.RecognitionFeatures(
            enable_word_time_offsets=False,
            enable_automatic_punctuation=True,
        ),
    )
    
    request = speech_v2.RecognizeRequest(
        recognizer=f"{parent}/recognizers/_", 
        config=config,
        content=audio_bytes,
    )

    start_time = time.perf_counter()
    try:
        response = stt_client.recognize(request=request)
        proc_time = time.perf_counter() - start_time
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        return transcript.strip(), proc_time
    except Exception as e:
        proc_time = time.perf_counter() - start_time
        return f"Error: {str(e)}", proc_time

def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI TTS -> Google STT Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="output/azure_ttshd_google_pipeline_results.csv", help="Output CSV file")
    
    parser.add_argument("--tts_deployment", type=str, default="tts-hd", help="Azure TTS deployment name")
    parser.add_argument("--google_location", type=str, default="us-central1", help="Google Cloud location")
    
    args = parser.parse_args()

    # Azure Credentials
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    # Google Credentials
    google_project_id = os.environ.get("GOOGLE_PROJECT_ID")
    google_location = args.google_location
    
    hf_token = os.environ.get("HF_TOKEN")
    
    if not azure_api_key or not azure_endpoint:
        print("ERROR: Azure credentials not found in .env")
        return
    if not google_project_id:
        print("ERROR: GOOGLE_PROJECT_ID not found in .env")
        return

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)

    # Initialize Clients
    azure_client = AzureOpenAI(
        api_key=azure_api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=azure_endpoint
    )
    google_stt_client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{google_location}-speech.googleapis.com")
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
    print(f"Running Azure TTS + Google STT pipeline for {args.limit} samples...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break
            
        original_text = example.get(args.text_col, '')
        if not original_text or len(original_text.strip()) == 0:
            continue

        audio_path = f"temp_audio/azure_tts_{i}.mp3"
        
        try:
            # 1. Azure TTS
            _, tts_time = azure_tts(azure_client, args.tts_deployment, original_text, out_file=audio_path)
            
            if not os.path.exists(audio_path):
                raise Exception("Azure TTS failed to generate audio file")

            # 2. Get Duration for RTF
            audio_duration = get_audio_duration(audio_path)

            # 3. Read audio bytes
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            # 4. Google STT (Chirp)
            transcription, stt_time = run_google_stt(google_stt_client, google_project_id, google_location, audio_bytes)
            
            # 5. Metrics
            punct_remover = str.maketrans("", "", string.punctuation)
            norm_original = original_text.lower().translate(punct_remover)
            norm_transcription = transcription.lower().translate(punct_remover)
            
            wer = jiwer.wer(norm_original, norm_transcription)
            cer = jiwer.cer(norm_original, norm_transcription)
            rtf = stt_time / audio_duration if audio_duration > 0 else 0
                
            results.append({
                'index': i,
                'original_text': original_text,
                'google_transcription': transcription,
                'wer': wer,
                'cer': cer,
                'tts_time': tts_time,
                'stt_time': stt_time,
                'audio_duration': audio_duration,
                'rtf': rtf
            })
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            results.append({
                'index': i,
                'original_text': original_text,
                'google_transcription': f"Error: {e}",
                'wer': 1.0,
                'cer': 1.0,
                'tts_time': 0,
                'stt_time': 0,
                'audio_duration': 0,
                'rtf': 0
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to {args.output}")
        print(f"Average WER: {df['wer'].mean():.4f}")
        print(f"Average CER: {df['cer'].mean():.4f}")
        print(f"Average RTF (STT): {df['rtf'].mean():.4f}")
        print(f"Average TTS time: {df['tts_time'].mean():.2f}s")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
