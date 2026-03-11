import os
import json
import time
import base64
import hashlib
import hmac
import argparse
import sys
from datetime import datetime, timezone
from urllib.parse import urlencode
import websocket
import _thread as thread
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import itertools
from datasets import load_dataset, Audio
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

IFLYTEK_APPID = os.environ.get("IFLYTEK_APPID")
IFLYTEK_APIKEY = os.environ.get("IFLYTEK_APIKEY")
IFLYTEK_SECRET = os.environ.get("IFLYTEK_SECRET")
HF_TOKEN = os.environ.get("HF_TOKEN")

class IFlytekASR:
    def __init__(self, appid, apikey, apisecret, audio_content, language="vi_VN"):
        self.APPID = appid
        self.APIKey = apikey
        self.APISecret = apisecret
        self.audio_content = audio_content # Bytes of PCM data
        self.language = language
        self.HOST = "iat-api-sg.xf-yun.com"
        self.PATH = "/v2/iat"
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {
            "domain": "iat",
            "language": self.language,
            "vinfo": 1,
            "vad_eos": 5000
        }
        self.result_text = []
        self.processing_time = 0

    def rfc1123_utc_now(self):
        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    def create_url(self):
        url = f"wss://{self.HOST}{self.PATH}"
        date = self.rfc1123_utc_now()
        signature_origin = f"host: {self.HOST}\ndate: {date}\nGET {self.PATH} HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode(), signature_origin.encode(), digestmod=hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature_sha).decode()
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_b64}"'
        authorization = base64.b64encode(authorization_origin.encode()).decode()
        v = {"authorization": authorization, "date": date, "host": self.HOST}
        return url + "?" + urlencode(v)

    def run(self):
        wsUrl = self.create_url()
        self.result_text = []
        start_time = time.perf_counter()
        
        def on_message(ws, message):
            try:
                resp = json.loads(message)
                code = resp.get("code", -1)
                if code != 0:
                    print(f"ERROR: {resp.get('message')} (code: {code})")
                    ws.close()
                    return
                
                data = resp.get("data", {})
                result = data.get("result", {})
                ws_list = result.get("ws", [])
                for seg in ws_list:
                    for w in seg.get("cw", []):
                        self.result_text.append(w.get("w", ""))
                
                if data.get("status") == 2:
                    self.processing_time = time.perf_counter() - start_time
                    ws.close()
            except Exception:
                ws.close()

        def on_error(ws, error):
            ws.close()

        def on_close(ws, *args):
            pass

        def on_open(ws):
            def run_thread(*args):
                frameSize = 8000
                interval = 0.04
                for i in range(0, len(self.audio_content), frameSize):
                    buf = self.audio_content[i:i + frameSize]
                    if not buf:
                        break
                    
                    curr_status = 1
                    if i == 0:
                        curr_status = 0
                        d = {
                            "common": self.CommonArgs,
                            "business": self.BusinessArgs,
                            "data": {
                                "status": 0,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(buf).decode(),
                                "encoding": "raw"
                            }
                        }
                    else:
                        if i + frameSize >= len(self.audio_content):
                            curr_status = 2
                        d = {
                            "data": {
                                "status": curr_status,
                                "audio": base64.b64encode(buf).decode(),
                            }
                        }
                    ws.send(json.dumps(d))
                    time.sleep(interval)
                
                # Ensure status 2 is sent
                if len(self.audio_content) % frameSize != 0 or i + frameSize < len(self.audio_content):
                     pass # handled in loop logic mostly, but iFlytek is sensitive
            
            thread.start_new_thread(run_thread, ())

        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.on_open = on_open
        ws.run_forever()
        return ''.join(self.result_text), self.processing_time

def convert_to_pcm16le(audio_bytes, target_sr=16000):
    """Converts raw audio bytes to 16kHz PCM16LE mono and returns duration."""
    with io.BytesIO(audio_bytes) as bio:
        x, sr = sf.read(bio, dtype="float32")
    
    duration = len(x) / sr
    if x.ndim == 2:
        x = x.mean(axis=1)
    
    if sr != target_sr:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up, down)
        sr = target_sr

    x = np.clip(x, -1.0, 1.0)
    x16 = (x * 32767.0).astype(np.int16)
    return x16.tobytes(), duration

def main():
    parser = argparse.ArgumentParser(description="Benchmarking iFlytek ASR")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--audio_col", type=str, default="audio", help="Audio column name")
    parser.add_argument("--text_col", type=str, default="text", help="Ground truth text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="output/iflytek_results.csv", help="Output CSV file")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not all([IFLYTEK_APPID, IFLYTEK_APIKEY, IFLYTEK_SECRET]):
        print("ERROR: Missing iFlytek credentials in .env")
        return

    print(f"Loading dataset '{args.dataset}'...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=HF_TOKEN)
        dataset = dataset.cast_column(args.audio_col, Audio(decode=False))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    iterator = iter(dataset)
    try:
        first_example = next(iterator)
    except StopIteration:
        print("ERROR: Dataset is empty.")
        return

    if args.text_col not in first_example:
        candidates = [
            "transcription",
            "text",
            "sentence",
            "transcript",
            "label",
        ]
        detected = next((c for c in candidates if c in first_example), None)
        if detected:
            print(
                f"WARNING: text_col='{args.text_col}' not found. "
                f"Using detected ground-truth column: '{detected}'. "
                f"Available keys: {list(first_example.keys())}"
            )
            args.text_col = detected
        else:
            print(
                f"WARNING: text_col='{args.text_col}' not found and no known ground-truth column detected. "
                f"Available keys: {list(first_example.keys())}"
            )

    results = []
    print(f"Testing {args.limit} samples with iFlytek...")

    stream = itertools.chain([first_example], iterator)
    for i, example in enumerate(tqdm(stream, total=args.limit)):
        if i >= args.limit:
            break
            
        audio_info = example.get(args.audio_col, {})
        audio_bytes = audio_info.get('bytes')
        path = audio_info.get('path', f"sample_{i}")
        
        try:
            pcm_bytes, audio_duration = convert_to_pcm16le(audio_bytes)
            asr = IFlytekASR(IFLYTEK_APPID, IFLYTEK_APIKEY, IFLYTEK_SECRET, pcm_bytes)
            transcription, proc_time = asr.run()
        except Exception as e:
            transcription, proc_time, audio_duration = f"Error: {e}", 0, 0
            
        results.append({
            'index': i,
            'dataset': args.dataset,
            'path': path,
            'audio_duration': audio_duration,
            'processing_time': proc_time,
            'rtf': proc_time / audio_duration if audio_duration > 0 else 0,
            'ground_truth': example.get(args.text_col, ''),
            'iflytek_transcription': transcription
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {args.output}")
    print(f"Average RTF: {df['rtf'].mean():.4f}")

if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
