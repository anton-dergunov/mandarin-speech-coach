from unittest.mock import MagicMock, patch

import torch

from mandarin_speech_coach.alignment.ctc.aligner import (
    CTCAligner,
)


def build_mock_processor():
    processor = MagicMock()

    processor.return_value.input_values.to.return_value = (
        torch.zeros((1, 100))
    )

    processor.tokenizer.pad_token_id = 0

    token_map = {
        "你": 10,
        "好": 11,
    }

    def processor_side_effect(*args, **kwargs):
        if "text" in kwargs:
            text = kwargs["text"]

            ids = [
                token_map[ch]
                for ch in text
            ]

            result = MagicMock()
            result.input_ids = torch.tensor([ids])

            return result

        return MagicMock(
            input_values=torch.zeros((1, 100))
        )

    processor.side_effect = processor_side_effect

    reverse_map = {
        10: "你",
        11: "好",
    }

    processor.decode.side_effect = (
        lambda token: reverse_map[int(token)]
    )

    return processor


def test_ctc_alignment_basic():
    processor = build_mock_processor()

    model = MagicMock()

    logits = torch.full(
        (1, 6, 20),
        -10.0,
    )

    # blank
    logits[0, 0, 0] = 10
    logits[0, 5, 0] = 10

    # 你
    logits[0, 1, 10] = 10
    logits[0, 2, 10] = 10

    # 好
    logits[0, 3, 11] = 10
    logits[0, 4, 11] = 10

    model.return_value.logits = logits

    with patch(
        "mandarin_speech_coach.alignment.ctc.aligner.get_ctc",
        return_value=(processor, model),
    ), patch(
        "mandarin_speech_coach.alignment.ctc.aligner.get_device",
        return_value="cpu",
    ), patch(
        "torchaudio.load",
        return_value=(torch.zeros((1, 16000)), 16000),
    ):
        aligner = CTCAligner()

        result = aligner.align(
            "dummy.wav",
            "你好",
        )

    assert result.method == "ctc"

    assert len(result.segments) == 2

    assert result.segments[0].label == "你"
    assert result.segments[1].label == "好"

    assert result.segments[0].start < result.segments[0].end

    assert (
        result.segments[0].end
        <= result.segments[1].start
    )


def test_ctc_alignment_repeated_character():
    processor = build_mock_processor()

    model = MagicMock()

    logits = torch.full(
        (1, 8, 20),
        -10.0,
    )

    logits[0, :, 0] = 5

    logits[0, 1, 10] = 10
    logits[0, 2, 10] = 10

    logits[0, 5, 10] = 10
    logits[0, 6, 10] = 10

    model.return_value.logits = logits

    with patch(
        "mandarin_speech_coach.alignment.ctc.aligner.get_ctc",
        return_value=(processor, model),
    ), patch(
        "mandarin_speech_coach.alignment.ctc.aligner.get_device",
        return_value="cpu",
    ), patch(
        "torchaudio.load",
        return_value=(torch.zeros((1, 16000)), 16000),
    ):
        aligner = CTCAligner()

        result = aligner.align(
            "dummy.wav",
            "你你",
        )

    assert len(result.segments) == 2

    assert result.segments[0].label == "你"
    assert result.segments[1].label == "你"

    assert (
        result.segments[0].end
        < result.segments[1].start
    )
