from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Segment:
    label: str
    start: float
    end: float
    confidence: Optional[float] = None


@dataclass
class AlignmentResult:
    method: str
    segments: List[Segment]
    metadata: dict = field(default_factory=dict)


@dataclass
class RecognitionResult:
    text: str
    confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PitchTrack:
    times: np.ndarray
    frequencies: np.ndarray
    duration: float
    metadata: dict = field(default_factory=dict)


@dataclass
class PronunciationAnalysis:
    recognition: RecognitionResult
    alignment: AlignmentResult
    pitch_track: PitchTrack
