import os
import argparse
from google.cloud import texttospeech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def google_tts(text, out_file, language_code="vi-VN", voice_name="vi-VN-Standard-A"):
    """Converts text to speech using Google Cloud TTS."""
    # Initialize the client
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Configure the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, 
        name=voice_name
    )
    
    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    print(f"Synthesizing text: '{text[:50]}...'")
    print(f"Language: {language_code}, Voice: {voice_name}")
    
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    
    # The response's audio_content is binary.
    with open(out_file, "wb") as f:
        f.write(response.audio_content)
    
    print(f"Audio content written to file: {out_file}")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Standalone Google Cloud TTS Demo")
    parser.add_argument("--text", type=str, default="Sáng nay, lúc 08:45, tại sự kiện AI Tech 2026, chuyên gia đã demo một bộ dataset gồm 1.024.500 dòng dữ liệu, chứa các từ khó như 'lắt léo', 'nghằn ngặt' và các thuật ngữ chuyên sâu như 'Recurrent Neural Networks' hay 'Hyperparameter Optimization', khiến hệ thống latency tăng thêm 15.5%.",  help="Text to convert to speech")
    parser.add_argument("--output", type=str, default="output/demo_tts.mp3", help="Output audio file path")
    parser.add_argument("--language", type=str, default="vi-VN", help="Language code (e.g., vi-VN, en-US)")
    parser.add_argument("--voice", type=str, default="vi-VN-Chirp3-HD-Zephyr", help="Voice name (e.g., vi-VN-Standard-D, en-US-Journey-F)")
    
    args = parser.parse_args()

    # Check for GOOGLE_APPLICATION_CREDENTIALS if needed, 
    # but usually the library finds it from env or default location.
    
    try:
        google_tts(
            text=args.text, 
            out_file=args.output, 
            language_code=args.language, 
            voice_name=args.voice
        )
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
