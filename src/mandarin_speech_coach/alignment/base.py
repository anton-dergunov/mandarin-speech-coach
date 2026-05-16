from abc import ABC, abstractmethod

from mandarin_speech_coach.core.types import AlignmentResult


class BaseAligner(ABC):
    @abstractmethod
    def align(
        self,
        audio_path: str,
        text: str
    ) -> AlignmentResult:
        pass
