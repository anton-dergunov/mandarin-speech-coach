from unittest.mock import MagicMock, patch

import numpy as np

from mandarin_speech_coach.pitch.praat_pitch import (
    PraatPitchExtractor,
)


@patch("parselmouth.Sound")
def test_pitch_extraction(mock_sound):
    mock_snd = MagicMock()

    mock_pitch = MagicMock()

    mock_pitch.xs.return_value = np.array([
        0.0,
        0.5,
        1.0,
    ])

    mock_pitch.selected_array = {
        "frequency": np.array([
            440.0,
            0.0,
            220.0,
        ])
    }

    mock_snd.to_pitch.return_value = mock_pitch
    mock_snd.duration = 1.0

    mock_sound.return_value = mock_snd

    extractor = PraatPitchExtractor()

    track = extractor.extract("dummy.wav")

    assert track.duration == 1.0

    assert np.isnan(track.frequencies[1])

    assert track.frequencies[0] == 440.0
    assert track.frequencies[2] == 220.0
