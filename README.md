## compare-stt

Repo này là tập các script Python để benchmark/so sánh:

- **STT (Speech-to-Text / ASR)**: Google Cloud Speech v2 (Chirp), Azure OpenAI (Whisper + GPT-4o Transcribe), Gemini audio-to-text, iFlytek ASR.
- **TTS (Text-to-Speech)**: Google Cloud TTS, Azure OpenAI TTS, iFlytek TTS.
- **Pipeline TTS → STT**: tạo audio từ text rồi nhận dạng lại để tính WER/CER.

Dataset mặc định được dùng trong hầu hết các script là **HuggingFace**: `capleaf/viVoice`, `dolly-vn/dolly-audio-1000h-vietnamese`, `linhtran92/viet_youtube_asr_corpus_v2`(streaming). Kết quả thường được lưu dạng CSV trong thư mục `output/`.

## Cấu trúc thư mục

- `benchmarks/`: các script benchmark STT/ASR.
- `pipelines/`: các pipeline TTS -> STT.
- `demos/`: các script demo / smoke test nhanh.
- `debug/`: script debug, thử nghiệm, kiểm tra cấu hình.
- `utils/`: tiện ích tổng hợp kết quả.
- `test.ipynb`: notebook điều phối chính, vẫn để ở root để gọi các script theo path tương đối.

---

## 1) Chuẩn bị cấu hình (.env)

Xem file mẫu: `.env.example`. Tạo `.env` và điền giá trị phù hợp.

### HuggingFace
- `HF_TOKEN`: cần nếu dataset bị gated/require auth.

### Google Cloud
- `GOOGLE_PROJECT_ID`
- `GOOGLE_LOCATION` (vd: `us-central1`)
- `GOOGLE_APPLICATION_CREDENTIALS`: đường dẫn tới service account JSON (để SDK Google auth).

### Gemini
- `GOOGLE_API_KEY`: API key cho Gemini (SDK `google-genai`).

### Azure OpenAI
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT` (vd: `https://<resource>.openai.azure.com/`)

Một số demo dùng tài khoản phụ:
- `AZURE_OPENAI_API_KEY_SECONDARY`
- `AZURE_OPENAI_ENDPOINT_SECONDARY`

### iFlytek
- `IFLYTEK_APPID`
- `IFLYTEK_APIKEY`
- `IFLYTEK_SECRET`

---

## 2) Luồng chung của các script benchmark

Hầu hết các file benchmark đều theo pattern:

1. `load_dotenv()` để nạp biến môi trường.
2. `load_dataset(..., streaming=True, token=HF_TOKEN)` để stream dataset.
3. `dataset.cast_column("audio", Audio(decode=False))` để lấy `audio.bytes` trực tiếp (tránh decode).
4. Với mỗi sample:
	- lấy `audio_bytes` (hoặc `text`) làm input
	- gọi model/service
	- đo `processing_time` (thời gian xử lý)
	- tính `audio_duration` và `rtf = processing_time / audio_duration` (nếu có)
5. Lưu `pandas.DataFrame` ra CSV trong `output/...`.

---

## 3) Chi tiết luồng từng script

### A. Pipeline TTS → STT (có WER/CER)

#### 1) `pipelines/google_pipeline.py` — Google TTS → Google STT

Mục tiêu: Dùng **Google Cloud TTS** tạo MP3 từ `text`, rồi dùng **Google Cloud STT v2 (Chirp)** để nhận dạng lại, sau đó tính **WER/CER**.

Luồng:
1. Stream dataset → lấy `original_text`.
2. `google_tts()` gọi `texttospeech.TextToSpeechClient().synthesize_speech(...)` → ghi MP3 ra `temp_audio/google_tts_{i}.mp3`.
3. Đọc file MP3 thành `audio_bytes`.
4. `run_google_stt()` gọi `speech_v2.SpeechClient().recognize(...)` với:
	- `AutoDetectDecodingConfig()`
	- `model="chirp_2"`
	- `language_codes=["vi-VN"]`
5. Normalize (lower + bỏ punctuation) và tính `jiwer.wer`, `jiwer.cer`.
6. Ghi CSV mặc định: `output/google_pipeline_results.csv`.

Lệnh chạy mẫu:
```bash
python pipelines/google_pipeline.py --limit 50 --output output/google_pipeline_results.csv --location us-central1
```

#### 2) `pipelines/azure_pipeline.py` — Azure TTS → Azure STT-HD

Mục tiêu: Dùng **Azure OpenAI TTS** tạo MP3 từ `text`, rồi dùng **Azure OpenAI Transcription (STT-HD)** để nhận dạng lại, sau đó tính **WER/CER**.

Luồng:
1. Stream dataset → lấy `original_text`.
2. `azure_tts()` gọi `client.audio.speech.create(...)` → `stream_to_file()` ra `temp_audio/azure_tts_{i}.mp3`.
3. Đọc `audio_bytes`.
4. `run_azure_stt()` gọi `client.audio.transcriptions.create(...)` với `language="vi"`.
5. Normalize và tính WER/CER.
6. Ghi CSV mặc định: `output/azure_pipeline_results.csv`.

Lưu ý cấu hình:
- `--tts_deployment` và `--stt_deployment` là **tên deployment** trên Azure (không nhất thiết trùng tên model public).
- `--api_version` mặc định là `2025-03-01-preview`.

Lệnh chạy mẫu:
```bash
python pipelines/azure_pipeline.py --limit 50 --output output/azure_pipeline_results.csv \
  --tts_deployment tts-hd --stt_deployment gpt-4o-transcribe
```

#### 3) `pipelines/azure_google_pipeline.py` — Azure TTS → Google STT

Mục tiêu: Azure TTS tạo audio, Google STT (Chirp) nhận dạng, tính WER/CER và **RTF**.

Luồng:
1. Stream dataset → lấy `original_text`.
2. Azure TTS → file MP3.
3. Tính `audio_duration` bằng `soundfile` (đọc file).
4. Google STT v2 recognize (Chirp) → transcript + `stt_time`.
5. WER/CER + `rtf = stt_time / audio_duration`.
6. Ghi CSV mặc định: `output/azure_ttshd_google_pipeline_results.csv`.

Lệnh chạy mẫu:
```bash
python pipelines/azure_google_pipeline.py --limit 50 --output output/azure_ttshd_google_pipeline_results.csv \
  --tts_deployment tts-hd --google_location us-central1
```

#### 4) `pipelines/pipeline.py` — iFlytek TTS → Google STT

Mục tiêu: iFlytek TTS (qua WebSocket) tạo MP3 từ text → Google STT nhận dạng → tính WER.

Luồng:
1. Stream dataset → lấy `original_text`.
2. `iflytek_tts()`:
	- build WS URL bằng HMAC (`IFLYTEK_APIKEY` + `IFLYTEK_SECRET`)
	- gửi payload TTS và nhận audio chunks (base64)
	- append ra file MP3.
3. Đọc file MP3 thành `audio_bytes`.
4. Google STT v2 recognize (mặc định `model="chirp_3"` trong file này).
5. Tính WER bằng `jiwer.wer(original.lower(), hyp.lower())`.
6. Ghi CSV mặc định: `output/iflytek_google_results.csv`.

Lệnh chạy mẫu:
```bash
python pipelines/pipeline.py --limit 50 --output output/iflytek_google_results.csv --location us-central1
```

---

### B. Benchmark STT (ASR-only)

#### 1) `benchmarks/test_google_stt.py` — Google STT (Chirp)

Luồng:
1. Stream dataset, lấy `audio.bytes`.
2. Tính `audio_duration` từ bytes bằng `soundfile`.
3. Google `speech_v2.SpeechClient().recognize(...)` (Chirp).
4. Lưu CSV: `index, path, audio_duration, processing_time, rtf, ground_truth, google_transcription`.

Lệnh chạy mẫu:
```bash
python benchmarks/test_google_stt.py --limit 100 --output output/google_results.csv
```

#### 2) `benchmarks/test_azure_stt.py` — Azure Whisper + GPT-4o Transcribe

Luồng:
1. Stream dataset, lấy `audio.bytes`.
2. Tính duration từ bytes.
3. Chạy 2 nhánh (nếu bật):
	- Whisper: `client_whisper.audio.transcriptions.create(...)` hoặc `translations.create(...)`.
	- GPT-4o: `client_gpt4o.audio.transcriptions.create(...)`.
4. Lưu 2 file CSV trong `--output_dir`:
	- `azure_whisper_results.csv`
	- `azure_gpt4o_results.csv`

Lệnh chạy mẫu:
```bash
python benchmarks/test_azure_stt.py --limit 100 --output_dir output/capleaf \
  --whisper_deployment whisper --gpt4o_deployment gpt-4o-transcribe
```

#### 3) `benchmarks/test_stt.py` — Gemini audio-to-text benchmark

Luồng:
1. Stream dataset, lấy `audio.bytes` + `path` để suy ra `mime_type`.
2. Tính `audio_duration` từ bytes.
3. `client.models.generate_content(...)` với:
	- `types.Part.from_bytes(data=audio_bytes, mime_type=...)`
	- prompt yêu cầu chỉ trả transcript.
4. Lưu CSV: `output/gemini_results.csv` (mặc định) gồm `processing_time`, `rtf`, `ground_truth`, `gemini_transcription`.

Lệnh chạy mẫu:
```bash
python benchmarks/test_stt.py --limit 100 --model gemini-3-flash-preview --output output/capleaf/google_gemini_results.csv
```

#### 4) `benchmarks/test_iflytek.py` — iFlytek ASR benchmark

Điểm quan trọng: iFlytek ASR endpoint trong file này expect **PCM16 mono 16kHz**.

Luồng:
1. Stream dataset, lấy `audio.bytes`.
2. `convert_to_pcm16le()`:
	- đọc audio bytes bằng `soundfile`
	- mixdown stereo → mono
	- resample về 16kHz (`resample_poly`)
	- convert float → int16 bytes.
3. `IFlytekASR.run()`:
	- build WS URL bằng HMAC
	- stream audio theo frame (8000 bytes) với `status` 0/1/2.
4. Lưu CSV: `output/iflytek_results.csv` gồm `processing_time`, `rtf`.

Lệnh chạy mẫu:
```bash
python benchmarks/test_iflytek.py --limit 100 --output output/capleaf/iflytek_results.csv
```

#### 5) `benchmarks/test_gemini_models.py` — Gemini multi-model benchmark

Mục tiêu: chạy nhiều model Gemini trong một lần benchmark và ghi ra mỗi model một file CSV riêng.

Lệnh chạy mẫu:
```bash
python benchmarks/test_gemini_models.py --dataset capleaf/viVoice --limit 100 \
	--models gemini-2.5-flash,gemini-3-flash-preview --output_dir output/capleaf_huy
```

#### 6) `benchmarks/test_phowhisper.py` — PhoWhisper local benchmark

Mục tiêu: benchmark model local `vinai/PhoWhisper-small` trên dataset Hugging Face và lưu kết quả ra CSV.

Lệnh chạy mẫu:
```bash
python benchmarks/test_phowhisper.py --dataset capleaf/viVoice --limit 100 \
	--model_id vinai/PhoWhisper-small --output output/capleaf_huy/phowhisper_small_results.csv
```

---

### C. Tổng hợp kết quả

#### `utils/compare_results.py` — gom nhiều CSV thành bảng summary

Mục tiêu: đọc tất cả `*.csv` trong một thư mục (mặc định `output/capleaf`), tự tìm cột transcript và tính WER/CER tổng.

Luồng:
1. List `*.csv` trong `--dir`.
2. Mỗi file:
	- tìm cột ground truth (`ground_truth` hoặc `gt`)
	- tự dò cột transcript (`*_transcription`)
	- normalize và tính WER/CER bằng `jiwer`.
3. In bảng summary (không ghi file mới).

Lệnh chạy mẫu:
```bash
python utils/compare_results.py --dir output/capleaf
```

---

## 4) Demo/Debug nhanh (không phải benchmark chính)

- `demos/demo_google_stt_local.py`: STT cho file audio local, có `--languages` (auto-detect) và `--hints` (phrase boosting).
- `demos/demo_google_tts_local.py`: TTS ra MP3 từ text với voice tùy chọn.
- `demos/demo_openai_tts_local.py`: Azure OpenAI TTS demo (dùng endpoint/key SECONDARY).
- `demos/call_tts_hd.py`: gọi nhanh TTS-HD và lưu MP3.
- `demos/test_azure_tts_hd.py`: smoke test Azure TTS-HD.
- `debug/debug_tts.py`: debug Azure TTS → Google STT.
- `debug/debug_dataset.py`: in vài dòng text từ dataset để kiểm tra HF_TOKEN.
- `debug/test_translate.py`: ví dụ cấu hình nhiều `language_codes`; phần translation được comment vì API v2 không “dịch sang vi” trực tiếp trong file.

---

## 5) Gợi ý thứ tự chạy để xác minh setup

1) Kiểm tra Gemini key:
```bash
python debug/simple_gemini_test.py
```

2) Kiểm tra đọc dataset:
```bash
python debug/debug_dataset.py
```

3) Benchmark STT (nhanh, không TTS):
```bash
python benchmarks/test_google_stt.py --limit 10
python benchmarks/test_stt.py --limit 10
python benchmarks/test_azure_stt.py --limit 10 --output_dir output/capleaf
```

4) Chạy pipeline TTS→STT (tốn phí hơn):
```bash
python pipelines/google_pipeline.py --limit 10
python pipelines/azure_pipeline.py --limit 10
python pipelines/azure_google_pipeline.py --limit 10
```

5) Tổng hợp:
```bash
python utils/compare_results.py --dir output/capleaf
```

