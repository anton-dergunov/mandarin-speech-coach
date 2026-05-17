"""Lenient sanity checks for integration test outputs."""

from __future__ import annotations

import numpy as np
from pypinyin import Style, pinyin

from mandarin_speech_coach.core.types import AlignmentResult, PitchTrack
from mandarin_speech_coach.visualization.pronunciation_plot import (
    PronunciationPlotBuilder,
)


def ordered_label_coverage(labels: list[str], target: str) -> float:
    """Share of target characters matched in order (allows skipped chars)."""
    if not target:
        return 1.0

    ti = 0
    matched = 0
    for label in labels:
        while ti < len(target) and target[ti] != label:
            ti += 1
        if ti < len(target) and target[ti] == label:
            matched += 1
            ti += 1

    return matched / len(target)


def assert_segments_look_reasonable(
    alignment: AlignmentResult,
    target: str,
    audio_duration: float,
    *,
    min_coverage: float = 0.75,
    min_span_fraction: float = 0.35,
) -> None:
    segments = alignment.segments
    assert segments, "expected at least one alignment segment"

    assert ordered_label_coverage(
        [s.label for s in segments],
        target,
    ) >= min_coverage

    for seg in segments:
        assert seg.start < seg.end
        assert seg.start >= -0.05
        assert seg.end <= audio_duration + 0.35

    for i in range(1, len(segments)):
        assert segments[i].start >= segments[i - 1].start - 0.05

    span = segments[-1].end - segments[0].start
    assert span >= min_span_fraction * audio_duration
    assert segments[-1].end <= audio_duration + 0.35


def assert_pitch_track_looks_reasonable(track: PitchTrack) -> None:
    assert track.duration > 0.2
    assert len(track.times) == len(track.frequencies)
    assert len(track.times) > 10
    assert np.all(np.diff(track.times) >= 0)

    voiced = track.frequencies[~np.isnan(track.frequencies)]
    assert len(voiced) > 0

    voiced_fraction = len(voiced) / len(track.frequencies)
    assert voiced_fraction >= 0.15

    assert 50 <= float(np.nanmin(track.frequencies)) <= 600
    assert 50 <= float(np.nanmax(track.frequencies)) <= 600

    assert track.times[0] >= 0
    assert track.times[-1] <= track.duration + 0.1


def assert_pronunciation_plot_looks_reasonable(
    alignment: AlignmentResult,
    pitch_track: PitchTrack,
    target: str,
) -> None:
    target_tones = pinyin(target, style=Style.TONE3)
    fig = PronunciationPlotBuilder().build(
        alignment,
        pitch_track,
        target_tones,
    )

    assert fig.data, "pitch trace should be present"
    assert len(fig.layout.shapes) >= max(1, len(alignment.segments) - 1)
    assert len(fig.layout.annotations) >= max(1, len(alignment.segments) - 1)

    x_range = fig.layout.xaxis.range
    assert x_range is not None
    assert x_range[1] - x_range[0] > 0.2
