import os
import time
import argparse
import pandas as pd
import string
from google.cloud import texttospeech
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from datasets import load_dataset, Audio
from tqdm import tqdm
from dotenv import load_dotenv
import io
import jiwer

# Load environment variables
load_dotenv()

def google_tts(tts_client, text, out_file="temp_audio/temp.mp3"):
    """Converts text to speech using Google Cloud TTS."""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Configure the voice request (Vietnamese)
    # Voices: vi-VN-Standard-A, vi-VN-Wavenet-A, etc.
    voice = texttospeech.VoiceSelectionParams(
        language_code="vi-VN", 
        name="vi-VN-Standard-A"
    )
    
    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    # The response's audio_content is binary.
    with open(out_file, "wb") as f:
        f.write(response.audio_content)
    
    return out_file

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
    parser = argparse.ArgumentParser(description="Google Cloud TTS -> STT Pipeline")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="output/google_pipeline_results.csv", help="Output CSV file")
    parser.add_argument("--location", type=str, default="us-central1", help="Google Cloud location")
    
    args = parser.parse_args()

    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    location = args.location
    hf_token = os.environ.get("HF_TOKEN")
    
    if not project_id:
        print("ERROR: GOOGLE_PROJECT_ID not found in .env")
        return

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)

    # Initialize Clients
    tts_client = texttospeech.TextToSpeechClient()
    stt_client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
    )

    print(f"Loading dataset '{args.dataset}'...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
        # Disable audio decoding to avoid torchcodec ImportError
        # We only need the text column for this pipeline
        if "audio" in dataset.features:
            dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    results = []
    print(f"Running pipeline for {args.limit} samples...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break
            
        original_text = example.get(args.text_col, '')
        if not original_text or len(original_text.strip()) == 0:
            continue

        audio_path = f"temp_audio/google_tts_{i}.mp3"
        
        try:
            # 1. Google TTS
            google_tts(tts_client, original_text, out_file=audio_path)
            
            # 2. Read audio
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            # 3. Google STT (Chirp)
            transcription, stt_time = run_google_stt(stt_client, project_id, location, audio_bytes)
            
            # 4. Metrics
            # Normalize for a fair comparison
            punct_remover = str.maketrans("", "", string.punctuation)
            norm_original = original_text.lower().translate(punct_remover)
            norm_transcription = transcription.lower().translate(punct_remover)
            
            wer = jiwer.wer(norm_original, norm_transcription)
            cer = jiwer.cer(norm_original, norm_transcription)
                
            results.append({
                'index': i,
                'original_text': original_text,
                'google_transcription': transcription,
                'wer': wer,
                'cer': cer,
                'stt_time': stt_time
            })
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            results.append({
                'index': i,
                'original_text': original_text,
                'google_transcription': f"Error: {e}",
                'wer': 1.0,
                'cer': 1.0,
                'stt_time': 0
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {args.output}")
    print(f"Average WER: {df['wer'].mean():.4f}")
    print(f"Average CER: {df['cer'].mean():.4f}")

if __name__ == "__main__":
    main()
