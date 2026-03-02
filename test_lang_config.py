import os
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def test_language_configs():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = "tts-hd"
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint
    )

    text = "Và mọi chuyện thì chưa dừng lại ở đó."
    
    # Attempt 1: Standard
    print("Test 1: Standard call")
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice="alloy",
            input=text
        )
        print("  Test 1: Success (Standard)")
    except Exception as e:
        print(f"  Test 1: Failed - {e}")

    # Attempt 2: extra_body with language
    print("\nTest 2: extra_body={'language': 'vi-VN'}")
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice="alloy",
            input=text,
            extra_body={"language": "vi-VN"}
        )
        print("  Test 2: Success (extra_body language)")
    except Exception as e:
        print(f"  Test 2: Failed - {e}")

    # Attempt 3: extra_headers with x-ms-language
    print("\nTest 3: extra_headers={'x-ms-language': 'vi-VN'}")
    try:
        response = client.audio.speech.create(
            model=deployment_name,
            voice="alloy",
            input=text,
            extra_headers={"x-ms-language": "vi-VN"}
        )
        print("  Test 3: Success (x-ms-language header)")
    except Exception as e:
        print(f"  Test 3: Failed - {e}")

if __name__ == "__main__":
    test_language_configs()
