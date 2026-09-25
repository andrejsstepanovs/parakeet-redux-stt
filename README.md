# Parakeet Redux STT

An OpenAI-compatible speech-to-text service built around Moondream's ternary
[`parakeet-redux`](https://huggingface.co/moondream/parakeet-redux), a 1.58-bit
distillation of NVIDIA's Parakeet TDT 0.6B v3.

It is a leaner sibling of the `~/stt` service: same HTTP contract, same optional
LLM post-processing, but the model is **178 MB** and runs entirely on the **CPU**
via [Photon](https://moondream.ai/photon) (the `moondream`/`kestrel` runtime),
with no ONNX Runtime, no Silero VAD and no audio chunking.

## Features

- OpenAI-compatible `POST /v1/audio/transcriptions` (`json`, `text`, `srt`,
  `vtt`, `verbose_json`) plus a multi-file `/v1/audio/transcriptions/batch`.
- 25 languages with automatic language detection; native word timestamps.
- Long audio handled by the model's own VAD head and pause segmentation.
- Optional LLM cleanup of transcripts through a LiteLLM proxy.
- Built-in web UI and `/health`, `/healthz`, `/metrics`.
- Runs on CPU (default) or CUDA/MPS where Photon supports it.

## Measured on this host (GB10, aarch64, CPU)

| Metric | Value |
| --- | --- |
| Model weights | 171 MB on disk |
| Warm startup | ~4 s (after weights are cached) |
| Inference, 11.5 s clip | ~0.5 s (~22x realtime) |
| Peak RSS | ~1.6 GB |
| GPU usage | none (`STT2_DEVICE=cpu`) |

## Layout

```
server.py                     uvicorn entry point (default port 5093)
stt2_service/
  config.py                   STT2_* environment configuration
  engine.py                   Photon/Kestrel client singleton
  routes.py                   OpenAI-compatible API + web UI + health
  audio.py                    upload spooling + FFmpeg fallback
  postprocess.py              optional LiteLLM cleanup
  main.py                     FastAPI app + lifespan
templates/index.html          web UI
Dockerfile.cpu                standalone CPU image (bundles model weights)
docker-compose.yml            compose service "stt2"
deploy/install.sh             sudo-less installer
deploy/stt2.service           per-user systemd unit
tests/                        unit + API tests
```

## Quick start (native)

Requires Python 3.10-3.14. FFmpeg is optional (Photon decodes most containers;
FFmpeg is only a fallback for exotic inputs).

```bash
cd ~/stt2
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Optional: authenticate to Hugging Face for a faster first download.
export HF_TOKEN=hf_xxx

# Loads ~/stt2/models, serves http://0.0.0.0:5093
python server.py
```

Configuration is read from the environment; the checked-in `.env` is loaded by
Docker Compose. For a native run, export the values you need, e.g.:

```bash
set -a; . ./.env; set +a
python server.py
```

## Install (Docker, no sudo)

```bash
cd ~/stt2
./deploy/install.sh
```

The script builds the CPU image, creates `.env` from `.env.example` if missing,
starts the container, waits for it to become healthy, and installs a **per-user**
systemd unit (`~/.config/systemd/user/stt2.service`). Nothing runs as root and
no polkit rule is needed.

Manage the service without sudo:

```bash
systemctl --user status stt2.service
systemctl --user restart stt2.service
systemctl --user stop stt2.service
docker compose logs -f stt2
```

Boot persistence comes from the container's `restart: unless-stopped` policy, so
the service comes back after a reboot without a login session or lingering. To
uninstall, run `./deploy/uninstall.sh` (it also offers to drop the model cache).

The image is **standalone**: the model weights are downloaded at build time and
bundled at `/app/models`, and the container runs with `HF_HUB_OFFLINE=1`. There
is no runtime volume and no network access is needed to serve requests.

## Run the prebuilt image (Docker Hub)

The published image already bundles the model weights and starts offline:

**`wormhit/parakeet-redux-stt:latest`**

```bash
# If the repository is private, authenticate first.
docker login

docker pull wormhit/parakeet-redux-stt:latest

docker run -d --name stt2 \
  -p 5093:5093 \
  --restart unless-stopped \
  wormhit/parakeet-redux-stt:latest
```

To enable LLM post-processing, pass a config file (create one from
`.env.example`) so the container sees `STT2_POST_PROCESS` and the `LITELLM_*`
variables:

```bash
docker run -d --name stt2 \
  -p 5093:5093 \
  --env-file .env \
  --restart unless-stopped \
  wormhit/parakeet-redux-stt:latest
```

Or use the included image-only Compose file, which pulls the published image
instead of building:

```bash
docker compose -f docker-compose.hub.yml up -d
```

The service is then at <http://localhost:5093> — the web UI at `/`, interactive
docs at `/docs`, health at `/health`, and the OpenAI endpoint at
`/v1/audio/transcriptions`.

> The image is `linux/arm64`. On a different CPU architecture, build the image
> locally instead (see below), or add `--platform` with a compatible tag.

## Build and publish (Docker Hub)

Because the weights are baked in, the image can be pushed and run offline on any
machine with the same CPU architecture:

```bash
# Build. An authenticated Hugging Face download is faster; put HF_TOKEN=hf_xxx
# in .env or pass it inline (it is a build secret, never baked into the image).
HF_TOKEN=hf_xxx docker compose build stt2

# Tag (this project publishes as wormhit/parakeet-redux-stt) and push.
docker tag parakeet-redux-stt:cpu wormhit/parakeet-redux-stt:latest
docker login
docker push wormhit/parakeet-redux-stt:latest

# On another machine (see "Run the prebuilt image" above):
docker run -d --name stt2 -p 5093:5093 wormhit/parakeet-redux-stt:latest
```

Notes:

- The image is **architecture-specific** (this host produces `linux/arm64`).
  Build on each target architecture, or use
  `docker buildx build --platform linux/amd64,linux/arm64`; the Dockerfile is
  generic and pip fetches the matching torch wheel.
- `HF_TOKEN` is passed as a BuildKit secret, so it never appears in image
  history. It only affects the build-time download.
- To refresh the weights, rebuild the image; there is no runtime volume.

## API usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5093/v1", api_key="sk-no-key-required")

with open("audio.mp3", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="moondream/parakeet-redux",
        file=audio,
        response_format="text",
    )
print(transcript)
```

The service also accepts the model names existing `~/stt` clients use
(`parakeet-tdt-0.6b-v3`, `parakeet-redux`) as aliases for
`moondream/parakeet-redux`.

```bash
curl -F file=@audio.flac -F response_format=srt \
  http://127.0.0.1:5093/v1/audio/transcriptions
```

### Open WebUI

Settings → Audio → STT Engine `OpenAI`, base URL
`http://<host>:5093/v1`, API key `sk-no-key-required`, model
`moondream/parakeet-redux`.

## Post-processing

Set `STT2_POST_PROCESS=true` and point `LITELLM_BASE_URL` / `LITELLM_API_KEY` /
`LITELLM_MODEL` at a LiteLLM (or any OpenAI-compatible) chat endpoint. Each
transcript segment is cleaned before the response is built, so SRT/VTT
timestamps stay aligned. Failures fall back to the raw transcript.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `STT2_DEVICE` | `cpu` | `cpu`, `cuda` or `mps` |
| `STT2_MODEL` | `moondream/parakeet-redux` | Default model id |
| `STT2_CPU_THREADS` | unset | Native CPU worker pool size |
| `STT2_TIMESTAMPS` | `word` | `none`, `segment`, `word`, `character` |
| `STT2_PORT` | `5093` | HTTP port |
| `STT2_MAX_UPLOAD_BYTES` | 512 MB | Per-file upload limit |
| `STT2_MAX_AUDIO_SECONDS` | 7200 | Audio length limit |
| `STT2_MAX_BATCH_FILES` / `STT2_MAX_BATCH_BYTES` | 16 / 512 MB | Batch limits |
| `STT2_POST_PROCESS` | `false` | Enable LLM cleanup |
| `STT2_POST_PROCESS_PROMPT` | built-in | Cleanup system prompt |
| `STT2_POST_PROCESS_TIMEOUT` | `90` | LLM request timeout (s) |
| `STT2_MODELS_DIR` | `./models` | Weight cache for native runs (`HF_HOME`) |
| `HF_TOKEN` | unset | Build-time only: HF token for faster/gated model downloads |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Notes

- On aarch64 the default PyPI `torch` wheel bundles CUDA/cuDNN runtime
  libraries, so the image/venv is large even though only the CPU path is used.
  Install torch from the CPU index (`--index-url
  https://download.pytorch.org/whl/cpu`) to slim it down.
- Photon reports basic usage telemetry (model, GPU type/memory, hostname,
  aggregate counters). Prompts and audio are never sent.
- Model license is CC-BY-4.0, same as the original Parakeet.
