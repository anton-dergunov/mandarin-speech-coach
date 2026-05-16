# Mandarin Speech Coach — full stack: Python deps, Hugging Face models, and MFA.
# First build downloads several GB; allow time and disk space.
FROM mambaorg/micromamba:2.0.0

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY apps/ ./apps/

RUN micromamba install -y -n base -c conda-forge \
      python=3.12 \
      montreal-forced-aligner \
      pip \
  && micromamba run -n base pip install --no-cache-dir . \
  && micromamba run -n base mfa model download acoustic mandarin_mfa \
  && micromamba run -n base mfa model download dictionary mandarin_mfa \
  && micromamba run -n base python -c "\
from transformers import ( \
    WhisperProcessor, WhisperForConditionalGeneration, \
    Wav2Vec2ForCTC, Wav2Vec2Processor, \
); \
WhisperProcessor.from_pretrained('openai/whisper-small'); \
WhisperForConditionalGeneration.from_pretrained('openai/whisper-small'); \
Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn'); \
Wav2Vec2ForCTC.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')"

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PYTHONPATH="/app/src:/app"

EXPOSE 7860

USER ${MAMBA_USER}
CMD ["python", "apps/gradio_demo/app.py"]
