import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

WHISPER_MODEL_NAME = "openai/whisper-small"
WAV2VEC2_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

_whisper_processor = None
_whisper_model = None

_ctc_processor = None
_ctc_model = None


def get_device():
    return DEVICE


def get_whisper():
    global _whisper_processor, _whisper_model

    if _whisper_processor is None:
        _whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)

    if _whisper_model is None:
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_NAME
        ).to(DEVICE)

    return _whisper_processor, _whisper_model


def get_ctc():
    global _ctc_processor, _ctc_model

    if _ctc_processor is None:
        _ctc_processor = Wav2Vec2Processor.from_pretrained(
            WAV2VEC2_MODEL_NAME
        )

    if _ctc_model is None:
        _ctc_model = Wav2Vec2ForCTC.from_pretrained(
            WAV2VEC2_MODEL_NAME
        ).to(DEVICE)

    return _ctc_processor, _ctc_model
