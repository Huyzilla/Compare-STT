import os
import time
from openai import AzureOpenAI
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from dotenv import load_dotenv
import soundfile as sf

load_dotenv()

def debug_tts_and_stt():
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment_name = "tts-hd"
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint
    )

    test_text = "Và mọi chuyện thì chưa dừng lại ở đó."
    out_file = "debug_diagnostic.mp3"

    print(f"--- Azure TTS Debug ---")
    print(f"Text: {test_text}")
    
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice="alloy",
            input=test_text,
            response_format="mp3"
        )
        # Using write_to_file if stream_to_file doesn't exist or is deprecated
        if hasattr(response, 'stream_to_file'):
            response.stream_to_file(out_file)
        else:
            response.write_to_file(out_file)
            
        print(f"Saved to {out_file}")
        
        # Check audio info
        data, samplerate = sf.read(out_file)
        duration = len(data) / samplerate
        print(f"Samplerate: {samplerate}Hz")
        print(f"Duration: {duration:.3f}s")
        print(f"Data peak: {max(abs(data)):.4f}")
        
        # --- Google STT Debug ---
        print(f"\n--- Google STT Debug ---")
        google_project_id = os.environ.get("GOOGLE_PROJECT_ID")
        google_location = "us-central1"
        
        stt_client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{google_location}-speech.googleapis.com")
        )
        
        with open(out_file, "rb") as f:
            audio_bytes = f.read()
            
        parent = f"projects/{google_project_id}/locations/{google_location}"
        config = speech_v2.RecognitionConfig(
            auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
            model="chirp_2", 
            language_codes=["vi-VN"],
            features=speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        
        request = speech_v2.RecognizeRequest(
            recognizer=f"{parent}/recognizers/_", 
            config=config,
            content=audio_bytes,
        )

        response = stt_client.recognize(request=request)
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        print(f"Transcription: {transcript}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    debug_tts_and_stt()
