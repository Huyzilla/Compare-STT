import argparse
import io
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()

TRANSCRIBE_PROMPT = (
    "Trích xuất văn bản từ âm thanh. YÊU CẦU: CHỈ trả về nội dung nghe được, không thêm bất kỳ từ nào như 'Đoạn âm thanh là', không dùng dấu ngoặc kép, không giải thích, không định dạng Markdown. Nếu không nghe rõ, trả về một khoảng trắng."
)

MAX_ERROR_CHARS = 280


def _compact_error_text(text: str, max_chars: int = MAX_ERROR_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_gemini_exception(exc: Exception) -> str:
    """Turns Gemini SDK exception string into a compact, CSV-friendly one-liner."""
    raw = _compact_error_text(str(exc), max_chars=4000)

    # Examples we see:
    # "403 PERMISSION_DENIED. {'error': {'code': 403, 'message': '...', 'details': [{'reason': 'API_KEY_SERVICE_BLOCKED', ...}]}}"
    # "404 NOT_FOUND. {'error': {'code': 404, 'message': 'models/... not found ...', 'status': 'NOT_FOUND'}}"
    code_match = re.search(r"\b(\d{3})\b", raw)
    status_match = re.search(r"\b([A-Z_]{3,})\b", raw)
    reason_match = re.search(r"reason'\s*:\s*'([^']+)'", raw)
    message_match = re.search(r"message'\s*:\s*'([^']+)'", raw)

    parts: List[str] = []
    if code_match:
        parts.append(code_match.group(1))
    if status_match:
        parts.append(status_match.group(1))
    if reason_match:
        parts.append(reason_match.group(1))
    if message_match:
        parts.append(message_match.group(1))

    if parts:
        return _compact_error_text(" ".join(parts), max_chars=MAX_ERROR_CHARS)

    return _compact_error_text(raw, max_chars=MAX_ERROR_CHARS)


def extract_transcript(raw_text: str) -> str:
    """Best-effort extraction of plain transcript from model outputs.

    Important: do NOT drop content just because it's multi-line.
    Some models wrap the transcript across lines; we join "content lines" back.
    """
    if raw_text is None:
        return ""

    raw = str(raw_text).replace("\r", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    # Normalize common prefixes without losing text.
    raw = re.sub(r"(?im)^\s*(?:transcription|transcript)\s*:\s*", "", raw)
    raw = re.sub(r"(?im)^\s*result\s*:\s*", "", raw)

    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return ""

    noise_patterns = [
        r"^i need to\b",
        r"^i must\b",
        r"^audio\b\s*:?",
        r"^the instruction\b",
        r"^instruction\b\s*:?",
        r"^think\b\s*:?",
        r"^analysis\b\s*:?",
    ]

    def is_noise(line: str) -> bool:
        return any(re.search(pat, line, flags=re.IGNORECASE) for pat in noise_patterns)

    content_lines: List[str] = []
    for ln in lines:
        if is_noise(ln):
            continue
        # Remove trailing "THINK:" tails if the model leaked them.
        ln = re.split(r"\bTHINK\b\s*:?.*$", ln, maxsplit=1)[0].strip()
        if not ln:
            continue
        content_lines.append(ln)

    text = " ".join(content_lines) if content_lines else " ".join(lines)
    text = text.strip().strip('"“”\'‘’')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_audio_duration(audio_bytes: Optional[bytes]) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with io.BytesIO(audio_bytes) as bio:
            data, samplerate = sf.read(bio)
        if samplerate <= 0:
            return 0.0
        return float(len(data) / samplerate)
    except Exception:
        return 0.0


def guess_mime_type(audio_bytes: Optional[bytes], path: Optional[str]) -> str:
    if audio_bytes:
        header = audio_bytes[:16]
        if header.startswith(b"RIFF") and b"WAVE" in header:
            return "audio/wav"
        if header.startswith(b"fLaC"):
            return "audio/flac"
        if header.startswith(b"OggS"):
            return "audio/ogg"
        if header[:4] == bytes([0x1A, 0x45, 0xDF, 0xA3]):
            return "audio/webm"
        if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
            return "audio/mpeg"

    if path:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            return "audio/wav"
        if ext == ".mp3":
            return "audio/mpeg"
        if ext == ".flac":
            return "audio/flac"
        if ext == ".ogg":
            return "audio/ogg"
        if ext == ".webm":
            return "audio/webm"

    # Fallback: HF audio often comes as wav/flac; wav is the safest default for raw bytes.
    return "audio/wav"


def _is_retryable_error(message: str) -> bool:
    msg = message.lower()
    return any(
        token in msg
        for token in [
            "429",
            "rate limit",
            "resource exhausted",
            "deadline exceeded",
            "timed out",
            "timeout",
            "temporarily",
            "internal",
            "unavailable",
            "503",
            "500",
        ]
    )


def gemini_transcribe(
    client: genai.Client,
    model_name: str,
    audio_bytes: Optional[bytes],
    mime_type: str,
    max_output_tokens: int = 1024,
    max_retries: int = 3,
    retry_backoff_s: float = 1.5,
) -> Tuple[str, float]:
    if not audio_bytes:
        return "N/A", 0.0

    attempt = 0
    start = time.perf_counter()
    last_error: Optional[str] = None

    while True:
        attempt += 1
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=TRANSCRIBE_PROMPT,
                    temperature=0,
                    max_output_tokens=max_output_tokens,
                ),
            )
            elapsed = time.perf_counter() - start
            text = getattr(response, "text", None)
            if text is None:
                return "", elapsed
            return extract_transcript(str(text)), elapsed
        except Exception as e:
            elapsed = time.perf_counter() - start
            last_error = format_gemini_exception(e)

            if attempt >= max_retries or not _is_retryable_error(last_error):
                return f"Error: {last_error}", elapsed

            time.sleep(retry_backoff_s ** attempt)


def sanitize_model_for_filename(model_name: str) -> str:
    name = model_name.lower().strip()
    name = name.replace("/", "-")
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = name.replace(".", "_")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gemini STT for multiple models")
    parser.add_argument("--dataset", type=str, default="capleaf/viVoice", help="Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--audio_col", type=str, default="audio", help="Audio column name")
    parser.add_argument("--text_col", type=str, default="text", help="Ground truth text column name")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument(
        "--models",
        type=str,
        default="gemini-2.5-flash,gemini-3.0-flash",
        help="Comma-separated Gemini model names",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save per-model CSV files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix (e.g. viet_youtube). If omitted, derived from dataset name.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=1024,
        help="Max tokens for Gemini output. Increase if transcripts look truncated.",
    )

    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found (set it in .env).")
        return

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        print("ERROR: No models provided.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    client = genai.Client(api_key=api_key)

    print(f"Loading dataset '{args.dataset}' (split: {args.split})...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=hf_token)
        dataset = dataset.cast_column(args.audio_col, Audio(decode=False))
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    prefix = args.prefix or args.dataset.split("/")[-1]

    results_by_model: Dict[str, List[dict]] = {m: [] for m in model_list}
    model_enabled: Dict[str, bool] = {m: True for m in model_list}
    model_disabled_reason: Dict[str, str] = {}

    print(f"Testing {args.limit} samples with Gemini models: {', '.join(model_list)}")

    for i, example in enumerate(tqdm(dataset, total=args.limit)):
        if i >= args.limit:
            break

        audio_info = example.get(args.audio_col, {})
        audio_bytes = audio_info.get("bytes")
        path = audio_info.get("path") or f"sample_{i}"

        ground_truth = example.get(args.text_col, "")
        audio_duration = get_audio_duration(audio_bytes)
        mime_type = guess_mime_type(audio_bytes, path)

        for model_name in model_list:
            if not model_enabled.get(model_name, True):
                continue
            transcription, proc_time = gemini_transcribe(
                client,
                model_name=model_name,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                max_output_tokens=args.max_output_tokens,
            )

            # If the model is clearly unavailable/blocked, stop calling it to avoid spam.
            if isinstance(transcription, str) and transcription.startswith("Error:"):
                err = transcription
                if any(
                    token in err
                    for token in [
                        "API_KEY_SERVICE_BLOCKED",
                        "SERVICE_DISABLED",
                        "NOT_FOUND",
                        "is not found for API version",
                    ]
                ):
                    model_enabled[model_name] = False
                    model_disabled_reason[model_name] = err
                    print(f"WARNING: Disabling model '{model_name}' due to fatal error: {err}")

            results_by_model[model_name].append(
                {
                    "index": i,
                    "dataset": args.dataset,
                    "path": path,
                    "audio_duration": audio_duration,
                    "processing_time": proc_time,
                    "rtf": (proc_time / audio_duration) if audio_duration > 0 else 0,
                    "ground_truth": ground_truth,
                    "gemini_model": model_name,
                    "gemini_transcription": transcription,
                }
            )

    for model_name, rows in results_by_model.items():
        df = pd.DataFrame(rows)
        out_name = f"{prefix}_{sanitize_model_for_filename(model_name)}_results.csv"
        out_path = os.path.join(args.output_dir, out_name)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        avg_time = float(df["processing_time"].mean()) if len(df) else 0.0
        avg_rtf = float(df["rtf"].mean()) if len(df) else 0.0
        print(f"Saved: {out_path} | Avg time: {avg_time:.2f}s | Avg RTF: {avg_rtf:.4f}")

    # Print a short hint if any model got disabled.
    for model_name, reason in model_disabled_reason.items():
        if "NOT_FOUND" in reason or "is not found for API version" in reason:
            print(
                f"HINT: Model '{model_name}' was NOT_FOUND. "
                "It may not exist or may not support generateContent on v1beta. "
                "Use the Gemini console / SDK ListModels to find the exact available 'flash' model name."
            )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
