# Universal ASR

A standalone speech recognition server that supports any Whisper-family model. Point it at a HuggingFace model ID and it handles downloading, converting, and serving — no manual setup required.

## Quick Start

```bash
git clone https://github.com/Khoality-dev/universal-asr.git
cd universal-asr
docker compose up -d
```

Open http://localhost:9000 for the web UI.

## Using Models

### Standard Whisper models

Set `UASR_DEFAULT_MODEL` in `docker-compose.yml` to any Whisper size:

`tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`

These download automatically on first startup.

### HuggingFace models (e.g. PhoWhisper)

Any HuggingFace Whisper model works — it gets auto-converted to CTranslate2 format on first use.

**Via the web UI:** Go to the Models tab, type a model ID (e.g. `vinai/PhoWhisper-base`), and click Pull.

**Via API:**
```bash
curl -X POST http://localhost:9000/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "vinai/PhoWhisper-base"}'
```

## Web UI

Available at http://localhost:9000 with three tabs:

- **Transcribe** — Upload an audio file or record from your microphone, pick a model, and transcribe.
- **Language Detection** — Detect the spoken language with confidence scores.
- **Models** — Pull new models, view installed models, delete models you no longer need.

## API

### Transcribe audio (OpenAI-compatible)

```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=base
```

Response:
```json
{"text": "Hello world", "language": "en"}
```

### Transcribe raw PCM

```bash
curl -X POST "http://localhost:9000/asr?language=en" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.pcm
```

Expects Int16 PCM at 16kHz mono. Response:
```json
{"text": "Hello world", "language": "en"}
```

### Detect language

```bash
curl -X POST http://localhost:9000/v1/audio/detect-language \
  -F file=@audio.wav
```

Response:
```json
{"language": "vi", "confidence": 0.95, "probabilities": {"vi": 0.95, "en": 0.03}}
```

### List models

```bash
curl http://localhost:9000/v1/models
```

### Delete a model

```bash
curl -X DELETE http://localhost:9000/v1/models/vinai_PhoWhisper-base
```

## Configuration

All settings are environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `UASR_HOST` | `0.0.0.0` | Bind address |
| `UASR_PORT` | `9000` | Server port |
| `UASR_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `UASR_COMPUTE_TYPE` | `auto` | `auto`, `float16`, `int8`, `float32` |
| `UASR_DEFAULT_MODEL` | `base` | Model loaded on startup |
| `UASR_DATA_DIR` | `/app/data` | Where models are cached |

## GPU Support

Requires NVIDIA GPU with Docker GPU support. The container uses CUDA automatically when available. To run on CPU only, set `UASR_DEVICE=cpu`.
