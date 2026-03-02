import os
import time
import argparse
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
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

def run_google_stt(client, project_id, location, audio_bytes, language_codes, hints=None):
    if not audio_bytes:
        return "N/A", 0.0

    # parent identifier for the location
    parent = f"projects/{project_id}/locations/{location}"
    
    adaptation = None
    if hints:
        adaptation = speech_v2.SpeechAdaptation(
            phrase_sets=[
                speech_v2.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=speech_v2.PhraseSet(
                        phrases=[{"value": hint} for hint in hints]
                    )
                )
            ]
        )

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        model="chirp_3", 
        language_codes=language_codes,
        adaptation=adaptation,
        features=speech_v2.RecognitionFeatures(
            enable_word_time_offsets=False,
            enable_automatic_punctuation=True,
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
    parser = argparse.ArgumentParser(description="Demo Google Cloud STT (Chirp) with local audio")
    parser.add_argument("--audio", type=str, required=True, help="Path to local audio file")
    parser.add_argument("--location", type=str, default=None, help="Google Cloud location")
    parser.add_argument("--languages", type=str, nargs="+", default=["en-US"], help="Language codes (provide multiple for auto-detection, e.g. --languages vi-VN en-US)")
    parser.add_argument("--hints", type=str, nargs="+", default=["Viettel"], help="Phrases to boost (e.g., --hints Viettel)")
    
    args = parser.parse_args()

    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    location = args.location or os.environ.get("GOOGLE_LOCATION", "us-central1")
    
    if not project_id:
        print("ERROR: GOOGLE_PROJECT_ID not found in .env")
        return

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        return

    # Initialize client
    client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
    )

    print(f"Reading audio file: {args.audio}")
    with open(args.audio, "rb") as f:
        audio_bytes = f.read()

    audio_duration = get_audio_duration(audio_bytes)
    print(f"Audio duration: {audio_duration:.2f} seconds")

    print(f"Languages: {args.languages}")
    if args.hints:
        print(f"Hints: {args.hints}")
    
    print(f"Transcribing using Google Chirp 2...")
    transcription, proc_time = run_google_stt(client, project_id, location, audio_bytes, args.languages, args.hints)
    
    print("-" * 30)
    print(f"Transcription:\n{transcription}")
    print("-" * 30)
    print(f"Processing time: {proc_time:.2f} seconds")
    if audio_duration > 0:
        print(f"RTF: {proc_time / audio_duration:.4f}")

if __name__ == "__main__":
    main()
