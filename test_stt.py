import os
import time
import argparse
import pandas as pd
from google import genai
from google.genai import types
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from tqdm import tqdm
import io
import soundfile as sf

# Load environment variables
load_dotenv()

def get_audio_duration(audio_bytes):
    """Calculates audio duration in seconds from bytes."""
    try:
        with io.BytesIO(audio_bytes) as bio:
            data, samplerate = sf.read(bio)
            return len(data) / samplerate
    except Exception:
        return 0.0

def get_gemini_transcription(client, model_name, audio_bytes, mime_type="audio/webm"):
    """Sends audio bytes to Gemini and measures processing time."""
    if not audio_bytes:
        return "N/A", 0.0
    
    start_time = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                "Trích xuất văn bản từ âm thanh. YÊU CẦU: CHỈ trả về nội dung nghe được, không thêm bất kỳ từ nào như 'Đoạn âm thanh là', không dùng dấu ngoặc kép, không giải thích, không định dạng Markdown. Nếu không nghe rõ, trả về một khoảng trắng."
            ]
        )
        duration = time.perf_counter() - start_time
        return response.text.strip(), duration
    except Exception as e:
        duration = time.perf_counter() - start_time
        return f"Error: {str(e)}", duration

def main():
    parser = argparse.ArgumentParser(description="Benchmarking Gemini STT")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--audio_col", type=str, default="audio", help="Audio column name")
    parser.add_argument("--text_col", type=str, default="text", help="Ground truth text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Gemini model name")
    parser.add_argument("--output", type=str, default="output/gemini_results.csv", help="Output CSV file")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    api_key = os.environ.get("GOOGLE_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)

    print(f"Loading dataset '{args.dataset}' (split: {args.split})...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
        dataset = dataset.cast_column(args.audio_col, Audio(decode=False))
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    results = []
    print(f"Testing {args.limit} samples with model {args.model}...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break
            
        audio_info = example.get(args.audio_col, {})
        audio_bytes = audio_info.get('bytes')
        path = audio_info.get('path', f"sample_{i}")
        
        # Audio duration for RTF calculation
        audio_duration = get_audio_duration(audio_bytes)
        
        # Determine mime type
        mime_type = "audio/webm"
        if path.endswith(".wav"): mime_type = "audio/wav"
        elif path.endswith(".mp3"): mime_type = "audio/mp3"
        
        transcription, proc_time = get_gemini_transcription(client, args.model, audio_bytes, mime_type)
        
        results.append({
            'index': i,
            'dataset': args.dataset,
            'path': path,
            'audio_duration': audio_duration,
            'processing_time': proc_time,
            'rtf': proc_time / audio_duration if audio_duration > 0 else 0,
            'ground_truth': example.get(args.text_col, ''),
            'gemini_transcription': transcription
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {args.output}")
    
    # Quick summary
    avg_time = df['processing_time'].mean()
    avg_rtf = df['rtf'].mean()
    print(f"Average Processing Time: {avg_time:.2f}s")
    print(f"Average RTF: {avg_rtf:.4f}")

if __name__ == "__main__":
    main()
