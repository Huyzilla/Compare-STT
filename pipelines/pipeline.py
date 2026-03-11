import os
import time
import json
import base64
import hashlib
import hmac
import ssl
import threading
import argparse
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlencode
import websocket
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from datasets import load_dataset, Audio
from tqdm import tqdm
from dotenv import load_dotenv
import io
import soundfile as sf
import jiwer

# Load environment variables
load_dotenv()

IFLYTEK_APPID = os.environ.get("IFLYTEK_APPID")
IFLYTEK_APIKEY = os.environ.get("IFLYTEK_APIKEY")
IFLYTEK_SECRET = os.environ.get("IFLYTEK_SECRET")

# iFlytek TTS (MP3)
def iflytek_tts(text, out_file="out.mp3", use_wss=True):
    HOST = "tts-api-sg.xf-yun.com"
    PATH = "/v2/tts"

    def rfc1123_utc_now():
        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    def build_ws_url(api_key, api_secret, use_wss=True):
        date = rfc1123_utc_now()
        signature_origin = f"host: {HOST}\ndate: {date}\nGET {PATH} HTTP/1.1"
        signature_sha = hmac.new(api_secret.encode(), signature_origin.encode(), digestmod=hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature_sha).decode()
        authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_b64}"'
        authorization = base64.b64encode(authorization_origin.encode()).decode()
        params = {"authorization": authorization, "date": date, "host": HOST}
        scheme = "wss" if use_wss else "ws"
        return f"{scheme}://{HOST}{PATH}?" + urlencode(params)

    if os.path.exists(out_file):
        os.remove(out_file)

    done = threading.Event()
    err = {"msg": None}
    ws_url = build_ws_url(IFLYTEK_APIKEY, IFLYTEK_SECRET, use_wss=use_wss)

    def on_open(ws):
        payload = {
            "common": {"app_id": IFLYTEK_APPID},
            "business": {
                "aue": "lame",
                "auf": "audio/L16;rate=16000",
                "vcn": "x2_ViVn_ThuHien",
                "speed": 50,
                "volume": 50,
                "pitch": 50,
                "tte": "UTF8",
            },
            "data": {
                "status": 2,
                "text": base64.b64encode(text.encode()).decode(),
            },
        }
        ws.send(json.dumps(payload))

    def on_message(ws, message):
        try:
            resp = json.loads(message)
            code = resp.get("code", -1)
            if code != 0:
                err["msg"] = f"iFLYTEK error code={code}, message={resp.get('message')}"
                ws.close()
                return
            data = resp.get("data", {})
            audio_b64 = data.get("audio", "")
            status = data.get("status", None)
            if audio_b64:
                audio = base64.b64decode(audio_b64)
                with open(out_file, "ab") as f:
                    f.write(audio)
            if status == 2:
                done.set()
                ws.close()
        except Exception as e:
            err["msg"] = f"parse error: {e}"
            ws.close()

    def on_error(ws, error):
        err["msg"] = str(error)
        done.set()

    def on_close(ws, *args):
        done.set()

    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    sslopt = {"cert_reqs": ssl.CERT_NONE} if use_wss else None
    ws.run_forever(sslopt=sslopt)

    if err["msg"]:
        raise RuntimeError(err["msg"])
    if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
        raise RuntimeError("No audio saved. Check voice (vcn) permission or text/content.")
    return out_file

def run_google_stt(client, project_id, location, audio_bytes):
    if not audio_bytes:
        return "N/A", 0.0

    parent = f"projects/{project_id}/locations/{location}"
    
    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        model="chirp_3", 
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
        response = client.recognize(request=request)
        proc_time = time.perf_counter() - start_time
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        return transcript.strip(), proc_time
    except Exception as e:
        proc_time = time.perf_counter() - start_time
        return f"Error: {str(e)}", proc_time

def main():
    parser = argparse.ArgumentParser(description="Benchmarking iFlytek TTS vs Google Cloud STT")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--audio_col", type=str, default="audio", help="Audio column name")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="output/iflytek_google_results.csv", help="Output CSV file")
    parser.add_argument("--location", type=str, default=None, help="Google Cloud location")
    
    args = parser.parse_args()

    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    location = args.location or os.environ.get("GOOGLE_LOCATION", "us-central1")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not all([IFLYTEK_APPID, IFLYTEK_APIKEY, IFLYTEK_SECRET, project_id]):
        print("ERROR: Missing API credentials in .env")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)

    # Initialize client
    client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
    )

    print(f"Loading dataset '{args.dataset}'...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
        # Disable audio decoding to avoid ImportError: torchcodec
        dataset = dataset.cast_column(args.audio_col, Audio(decode=False))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    results = []
    print(f"Testing {args.limit} samples...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break
            
        original_text = example.get(args.text_col, '')
        if not original_text:
            continue

        audio_file = os.path.join("temp_audio", f"sample_{i}.mp3")
        
        try:
            # 1. TTS with iFlytek
            iflytek_tts(original_text, out_file=audio_file)
            
            # 2. Read audio bytes
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            
            # 3. STT with Google
            transcription, proc_time = run_google_stt(client, project_id, location, audio_bytes)
            
            # 4. Calculate WER
            wer = jiwer.wer(original_text.lower(), transcription.lower())
                
            results.append({
                'index': i,
                'original_text': original_text,
                'google_transcription': transcription,
                'wer': wer,
                'processing_time': proc_time
            })
        except Exception as e:
            print(f"Error at index {i}: {e}")
            results.append({
                'index': i,
                'original_text': original_text,
                'google_transcription': f"Error: {e}",
                'wer': 1.0,
                'processing_time': 0
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {args.output}")
    print(f"Average WER: {df['wer'].mean():.4f}")

if __name__ == "__main__":
    main()
