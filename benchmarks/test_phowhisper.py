import argparse
import io
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

load_dotenv()


def get_audio_duration(audio_bytes: Optional[bytes]) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with io.BytesIO(audio_bytes) as bio:
            data, sample_rate = sf.read(bio)
        if sample_rate <= 0:
            return 0.0
        return float(len(data) / sample_rate)
    except Exception:
        return 0.0


def decode_audio(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    with io.BytesIO(audio_bytes) as bio:
        audio, sample_rate = sf.read(bio, dtype="float32")

    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != target_sr:
        audio = resample_poly(audio, target_sr, sample_rate).astype(np.float32)

    return np.asarray(audio, dtype=np.float32)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def transcribe_phowhisper(
    model,
    processor,
    audio_array: np.ndarray,
    device: str,
    torch_dtype: torch.dtype,
    language: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    start = time.perf_counter()
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device=device, dtype=torch_dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=max_new_tokens,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    elapsed = time.perf_counter() - start
    return normalize_text(text), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PhoWhisper on a Hugging Face dataset")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--audio_col", type=str, default="audio", help="Audio column name")
    parser.add_argument("--text_col", type=str, default="text", help="Ground truth text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument(
        "--model_id",
        type=str,
        default="vinai/PhoWhisper-small",
        help="Hugging Face model ID",
    )
    parser.add_argument("--language", type=str, default="vi", help="Language code for Whisper decoding")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated tokens")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/phowhisper_small_results.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model '{args.model_id}' on {device}...")
    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch_dtype,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()

    print(f"Loading dataset '{args.dataset}' (split: {args.split})...")
    dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
    dataset = dataset.cast_column(args.audio_col, Audio(decode=False))

    results = []
    print(f"Testing {args.limit} samples with model {args.model_id}...")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break

        audio_info = example.get(args.audio_col, {})
        audio_bytes = audio_info.get("bytes")
        path = audio_info.get("path") or f"sample_{i}"
        ground_truth = example.get(args.text_col, "")
        audio_duration = get_audio_duration(audio_bytes)

        if not audio_bytes:
            transcription = "N/A"
            processing_time = 0.0
        else:
            try:
                audio_array = decode_audio(audio_bytes)
                transcription, processing_time = transcribe_phowhisper(
                    model=model,
                    processor=processor,
                    audio_array=audio_array,
                    device=device,
                    torch_dtype=torch_dtype,
                    language=args.language,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                processing_time = 0.0
                transcription = f"Error: {type(exc).__name__}: {exc}"

        results.append(
            {
                "index": i,
                "dataset": args.dataset,
                "path": path,
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "rtf": (processing_time / audio_duration) if audio_duration > 0 else 0,
                "ground_truth": ground_truth,
                "model_id": args.model_id,
                "phowhisper_transcription": transcription,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    avg_time = float(df["processing_time"].mean()) if len(df) else 0.0
    avg_rtf = float(df["rtf"].mean()) if len(df) else 0.0
    print(f"Saved: {args.output}")
    print(f"Average Processing Time: {avg_time:.2f}s")
    print(f"Average RTF: {avg_rtf:.4f}")

    # Hugging Face background threads can crash the process during Python finalization
    # after the CSV has already been written. Exit immediately to keep notebook subprocesses stable.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()