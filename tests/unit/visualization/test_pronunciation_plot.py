import numpy as np

from mandarin_speech_coach.core.types import (
    AlignmentResult,
    PitchTrack,
    Segment,
)

from mandarin_speech_coach.visualization.pronunciation_plot import (
    PronunciationPlotBuilder,
)


def test_pronunciation_plot_builder():
    alignment = AlignmentResult(
        method="ctc",
        segments=[
            Segment(
                label="你",
                start=0.0,
                end=0.5,
            ),
            Segment(
                label="好",
                start=0.5,
                end=1.0,
            ),
        ],
    )

    pitch_track = PitchTrack(
        times=np.array([0.0, 0.5, 1.0]),
        frequencies=np.array([100.0, 110.0, 120.0]),
        duration=1.0,
    )

    builder = PronunciationPlotBuilder()

    fig = builder.build(
        alignment,
        pitch_track,
        [["ni3"], ["hao3"]],
    )

    assert len(fig.data) == 1

    assert len(fig.layout.shapes) == 2

    assert len(fig.layout.annotations) == 2

    assert fig.layout.annotations[0].text == "<b>你</b>"

    assert fig.layout.xaxis.range == (0, 1.0)
