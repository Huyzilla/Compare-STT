import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")

if api_key:
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents="Say hello in Vietnamese"
        )
        print("Gemini response:", response.text)
    except Exception as e:
        print("Error calling Gemini:", e)
else:
    print("Please set GOOGLE_API_KEY in .env file")
