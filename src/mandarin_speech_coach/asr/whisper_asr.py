import librosa

from mandarin_speech_coach.core.types import RecognitionResult
from mandarin_speech_coach.models.registry import (
    get_whisper,
    get_device,
)


class WhisperASR:
    def __init__(self):
        self.processor, self.model = get_whisper()
        self.device = get_device()

    def transcribe(self, audio_path: str) -> RecognitionResult:
        waveform, _ = librosa.load(audio_path, sr=16000)

        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="chinese",
            task="transcribe"
        )

        predicted_ids = self.model.generate(
            inputs.input_features,
            forced_decoder_ids=forced_decoder_ids
        )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return RecognitionResult(text=text)
