import numpy as np
import parselmouth

from mandarin_speech_coach.core.types import PitchTrack


class PraatPitchExtractor:
    def extract(self, audio_path: str) -> PitchTrack:
        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values[pitch_values == 0] = np.nan
        return PitchTrack(
            times=pitch.xs(),
            frequencies=pitch_values,
            duration=snd.duration,
        )

