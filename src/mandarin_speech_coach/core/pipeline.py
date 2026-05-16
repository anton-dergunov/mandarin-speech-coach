from mandarin_speech_coach.audio.preprocessing import (
    trim_silence,
)
from mandarin_speech_coach.asr.whisper_asr import (
    WhisperASR,
)
from mandarin_speech_coach.alignment.ctc.aligner import (
    CTCAligner,
)
from mandarin_speech_coach.alignment.mfa.aligner import (
    MFAAligner,
)
from mandarin_speech_coach.pitch.praat_pitch import (
    PraatPitchExtractor,
)
from mandarin_speech_coach.core.types import (
    PronunciationAnalysis,
)


class PronunciationPipeline:
    def __init__(self):
        self.asr = WhisperASR()

        self.aligners = {
            "ctc": CTCAligner(),
            "mfa": MFAAligner(),
        }

        self.pitch_extractor = PraatPitchExtractor()

    def run(
        self,
        audio_path: str,
        target_text: str,
        aligner_name: str = "ctc",
    ):
        trimmed_audio = trim_silence(audio_path)

        recognition = self.asr.transcribe(trimmed_audio)

        alignment = self.aligners[aligner_name].align(trimmed_audio, target_text)

        pitch_track = self.pitch_extractor.extract(trimmed_audio)

        return PronunciationAnalysis(
            recognition=recognition,
            alignment=alignment,
            pitch_track=pitch_track,
        )
