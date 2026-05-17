from unittest.mock import MagicMock, patch

import numpy as np
import torch

from mandarin_speech_coach.asr.whisper_asr import (
    WhisperASR,
)


@patch("mandarin_speech_coach.asr.whisper_asr.get_device")
@patch("mandarin_speech_coach.asr.whisper_asr.get_whisper")
def test_whisper_asr_transcribe(
    mock_get_whisper,
    mock_get_device,
):
    mock_get_device.return_value = "cpu"

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.input_features = torch.zeros((1, 80, 300))

    mock_inputs.to.return_value = mock_inputs

    mock_processor.return_value = mock_inputs

    mock_processor.get_decoder_prompt_ids.return_value = [
        [1, 2]
    ]

    mock_model.generate.return_value = torch.tensor([
        [1, 2, 3]
    ])

    mock_processor.batch_decode.return_value = [
        "你好"
    ]

    mock_get_whisper.return_value = (
        mock_processor,
        mock_model,
    )

    with patch(
        "librosa.load",
        return_value=(np.zeros(16000), 16000),
    ):
        asr = WhisperASR()

        result = asr.transcribe("dummy.wav")

    assert result.text == "你好"

    mock_model.generate.assert_called_once()
