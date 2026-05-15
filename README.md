# Mandarin Speech Coach

A Gradio web app for **Mandarin pronunciation practice**. Record or upload your voice, compare it to a target phrase, and see **pitch (F0)**, **character-level alignment**, and **tone-colored** segments.

Speech recognition uses **OpenAI Whisper**; alignment defaults to a **Wav2Vec2 CTC** model (no extra tools). **Montreal Forced Aligner (MFA)** is optional for classic forced alignment.

Default target phrase: **我喜欢机器学习** (`wǒ xǐhuān jīqì xuéxí`).

---

## How it works (short)

1. **Record / upload** audio in the browser.
2. **Trim silence**, then **Whisper** transcribes what you said (for comparison).
3. **Forced alignment** maps each character of the target text to a time range:
   - **CTC (default)** — Wav2Vec2 Chinese model, pure Python/PyTorch.
   - **MFA (optional)** — Montreal Forced Aligner via the `mfa` CLI.
4. **Parselmouth (Praat)** extracts pitch; **Plotly** draws F0 with tone-colored bands.

---

## Requirements

| Component                  | Role                                    |
| -------------------------- | --------------------------------------- |
| **Python** ≥ 3.10          | Runtime                                 |
| **ffmpeg**                 | Audio I/O (used by Gradio / librosa)    |
| **libsndfile**             | Reading/writing WAV (`soundfile`)       |
| **PyTorch + torchaudio**   | Models and audio tensors                |
| **transformers**           | Whisper + Wav2Vec2                      |
| **gradio**                 | Web UI                                  |
| **librosa**, **soundfile** | Load, trim, resample audio              |
| **praat-parselmouth**      | Pitch analysis (bundles Praat bindings) |
| **pypinyin**               | Pinyin with tone marks                  |
| **praatio**                | Read MFA TextGrid output                |
| **plotly**                 | Interactive pitch plot                  |
| **numpy**                  | Numerics                                |

**Optional — MFA only**

| Component                              | Role                                                                  |
| -------------------------------------- | --------------------------------------------------------------------- |
| **Montreal Forced Aligner** (`mfa`)    | CLI aligner ([docs](https://montreal-forced-aligner.readthedocs.io/)) |
| **mandarin_mfa** acoustic + dictionary | Pretrained models (`mfa model download …`)                            |

Hugging Face models (downloaded on first run if not cached):

- `openai/whisper-small`
- `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`

---

## Install with pip (standard Python)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install .
```

**Editable** install while developing:

```bash
python -m pip install -e .
```

**System packages** (examples):

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# macOS (Homebrew)
brew install ffmpeg libsndfile
```

**GPU (optional):** default wheels are CPU-friendly. For CUDA, install matching `torch` / `torchaudio` from [pytorch.org](https://pytorch.org/get-started/locally/) *before* or *after* installing this project, following their matrix for your OS and driver.

**Run:**

```bash
python mandarin_speech_demo.py
```

Open the URL printed in the terminal (default `http://127.0.0.1:7860`).

---

## Install with uv (recommended for speed)

[uv](https://docs.astral.sh/uv/) resolves and installs dependencies quickly from `pyproject.toml`.

```bash
# Create venv and install the project
uv venv
source .venv/bin/activate
uv pip install -e .

# Or run without activating the venv
uv run python mandarin_speech_demo.py
```

**Reproducible installs** (uses `uv.lock`):

```bash
uv sync
uv run python mandarin_speech_demo.py
```

---

## Montreal Forced Aligner (optional)

CTC alignment works without MFA. Use MFA only if you select **MFA** in the UI.

Official install path is **Conda / Miniconda** ([installation guide](https://montreal-forced-aligner.readthedocs.io/en/stable/installation.html)):

```bash
conda config --add channels conda-forge
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
mfa --help
```

Download Mandarin models (required once per environment):

```bash
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_mfa
```

Verify:

```bash
mfa model inspect acoustic mandarin_mfa
mfa model inspect dictionary mandarin_mfa
```

Run the app with `mfa` on your `PATH` (e.g. keep `conda activate aligner` in the same shell, or install MFA in the same env as the app).

**Environment overrides** (defaults match the commands above):

| Variable             | Default        | Meaning                             |
| -------------------- | -------------- | ----------------------------------- |
| `MFA_ACOUSTIC_MODEL` | `mandarin_mfa` | Acoustic model name for `mfa align` |
| `MFA_DICTIONARY`     | `mandarin_mfa` | Dictionary name for `mfa align`     |

**Note:** MFA pulls in Kaldi and other native tooling via Conda. It is intentionally **not** listed in `pyproject.toml`; use Conda for MFA, or use the **Docker** image below.

---

## Docker (all-in-one, including MFA)

The `Dockerfile` installs Python dependencies, pre-downloads Hugging Face models, installs MFA via Conda, and fetches `mandarin_mfa` models. The first build can take a long time and needs several GB of disk.

```bash
docker build -t mandarin-speech-coach .
docker run --rm -p 7860:7860 mandarin-speech-coach
```

Or with Compose:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:7860`.

**Useful environment variables in containers:**

| Variable | Default in image | Purpose |
|----------|------------------|---------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Listen on all interfaces |
| `GRADIO_SERVER_PORT` | `7860` | HTTP port |
| `GRADIO_DEBUG` | `0` | Set to `1` for Gradio debug mode |
| `HF_HOME` | `/app/.cache/huggingface` | Model cache directory |

---

## Other ways to simplify setup

| Approach | When to use |
|----------|-------------|
| **Docker / Compose** (above) | Avoid local Conda, ffmpeg, and MFA setup |
| **`uv sync` + committed `uv.lock`** | Fast, reproducible local/CI installs |
| **CTC only** | Skip MFA entirely; smallest local install |
| **Pre-download models** | Run once: `python -c "from transformers import ..."` (see `Dockerfile`) to avoid first-run delay |
| **Make / script wrapper** | Optional `make run` → `uv run python mandarin_speech_demo.py` for one command |
| **Dev Container** | Optional `.devcontainer/` for VS Code / Cursor with extensions and `postCreateCommand: uv sync` |

---

## Project layout

```text
.
├── mandarin_speech_demo.py   # Main Gradio app
├── pyproject.toml            # Dependencies and package metadata
├── uv.lock                   # Pinned versions (commit this)
├── Dockerfile                # Full image with MFA + models
├── docker-compose.yml
└── README.md
```

---

## License

MIT (see `pyproject.toml`). Hugging Face and MFA models have their own licenses; check their model cards before redistribution.
