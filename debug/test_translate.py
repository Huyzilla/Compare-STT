import os
import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env
load_dotenv()

# Cấu hình mặc định từ biến môi trường
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")

def speech_to_vietnamese_auto(audio_file):
    if not PROJECT_ID:
        print("ERROR: GOOGLE_PROJECT_ID không tìm thấy trong file .env")
        return

    if not os.path.exists(audio_file):
        print(f"ERROR: File âm thanh không tồn tại: {audio_file}")
        return

    # Khởi tạo client với endpoint vùng
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{LOCATION}-speech.googleapis.com",
        )
    )

    with open(audio_file, "rb") as f:
        audio_content = f.read()

    # Cấu hình nhận dạng
    config = cloud_speech.RecognitionConfig(
        # 1. AUTO-DETECTION: Liệt kê các ngôn ngữ dự kiến người nói sẽ dùng
        language_codes=["en-US", "ja-JP", "fr-FR", "vi-VN"], 
        
        model="chirp_2",
        
        # 2. TRANSLATION (Lưu ý): Hiện tại Google STT v2 chỉ hỗ trợ dịch sang tiếng Anh (target_language="en-US").
        # Việc dịch sang tiếng Việt (vi-VN) trực tiếp trong API này chưa được hỗ trợ.
        # Nếu bạn muốn dịch sang tiếng Việt, hãy dùng Google Cloud Translation API sau khi có kết quả STT.
        # translation_config=cloud_speech.TranslationConfig(target_language="en-US"),
        
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
        ),
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
        config=config,
        content=audio_content,
    )

    print(f"Đang nhận diện ngôn ngữ và dịch sang tiếng Việt cho file: {audio_file}...")
    try:
        response = client.recognize(request=request)

        for result in response.results:
            # Ngôn ngữ mà Google đã tự động nhận diện được
            detected_lang = result.language_code
            print(f"Detected Language: {detected_lang}")

            # Transcript gốc
            print(f"Original Text: {result.alternatives[0].transcript}")

            # Bản dịch tiếng Việt
            if result.alternatives[0].translations:
                print(f"Vietnamese Translation: {result.alternatives[0].translations[0].text}")
            print("-" * 30)
    except Exception as e:
        print(f"Lỗi khi gọi API Google: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Speech-to-Text with Auto Translation")
    parser.add_argument("--audio", type=str, help="Đường dẫn đến file âm thanh cần dịch")
    
    args = parser.parse_args()
    
    if args.audio:
        speech_to_vietnamese_auto(args.audio)
    else:
        print("Vui lòng cung cấp đường dẫn file âm thanh qua --audio")
        print("Ví dụ: python test_translate.py --audio sample.wav")