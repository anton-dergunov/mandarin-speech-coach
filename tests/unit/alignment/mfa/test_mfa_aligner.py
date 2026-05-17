from unittest.mock import MagicMock, patch

from mandarin_speech_coach.alignment.mfa.aligner import (
    MFAAligner,
)


@patch("shutil.which")
def test_mfa_missing(mock_which):
    mock_which.return_value = None

    aligner = MFAAligner()

    try:
        aligner.align("dummy.wav", "你")
    except Exception as exc:
        assert "Montreal Forced Aligner" in str(exc)
    else:
        raise AssertionError("Expected exception")


@patch("praatio.textgrid.openTextgrid")
@patch("subprocess.run")
@patch("shutil.which")
def test_mfa_alignment(
    mock_which,
    mock_run,
    mock_open_textgrid,
):
    mock_which.return_value = "/usr/bin/mfa"

    mock_run.return_value = MagicMock(
        returncode=0,
    )

    mock_textgrid = MagicMock()

    mock_textgrid.tierNames = ["words"]

    mock_tier = MagicMock()

    mock_tier.entries = [
        MagicMock(
            label="你",
            start=0.0,
            end=0.5,
        ),
        MagicMock(
            label="好",
            start=0.5,
            end=1.0,
        ),
    ]

    mock_textgrid.getTier.return_value = mock_tier

    mock_open_textgrid.return_value = mock_textgrid

    with patch(
        "librosa.load",
        return_value=([0.0] * 16000, 16000),
    ):
        aligner = MFAAligner()

        result = aligner.align(
            "dummy.wav",
            "你好",
        )

    assert result.method == "mfa"

    assert len(result.segments) == 2

    assert result.segments[0].label == "你"

    mock_run.assert_called_once()
