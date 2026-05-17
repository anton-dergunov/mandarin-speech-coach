import os
import tempfile

import numpy as np
import soundfile as sf

from mandarin_speech_coach.audio.preprocessing import (
    trim_silence,
)


def test_trim_silence_removes_padding():
    sr = 16000

    silence = np.zeros(sr // 2)

    tone = 0.2 * np.sin(
        2 * np.pi * 440 * np.linspace(0, 1, sr)
    )

    waveform = np.concatenate([
        silence,
        tone,
        silence,
    ])

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:
        input_path = tmp.name

    sf.write(input_path, waveform, sr)

    try:
        trimmed_path = trim_silence(input_path)

        trimmed_audio, trimmed_sr = sf.read(trimmed_path)

        assert trimmed_sr == sr

        assert len(trimmed_audio) < len(waveform)

        # should still contain most of the tone
        assert len(trimmed_audio) > sr * 0.8

    finally:
        os.remove(input_path)

        if os.path.exists(trimmed_path):
            os.remove(trimmed_path)
