import pytest

from mandarin_speech_coach.core.pipeline import PronunciationPipeline

from checks import (
    assert_pitch_track_looks_reasonable,
    assert_pronunciation_plot_looks_reasonable,
    assert_segments_look_reasonable,
)


@pytest.fixture(scope="module")
def pipeline():
    return PronunciationPipeline()


@pytest.mark.integration
def test_ctc_pipeline_on_example_audio(
    pipeline,
    example_wav_path,
    target_text,
):
    result = pipeline.run(
        example_wav_path,
        target_text,
        aligner_name="ctc",
    )

    assert result.alignment.method == "ctc"
    assert result.recognition.text.strip()

    duration = result.pitch_track.duration
    assert_segments_look_reasonable(
        result.alignment,
        target_text,
        duration,
    )
    assert_pitch_track_looks_reasonable(result.pitch_track)
    assert_pronunciation_plot_looks_reasonable(
        result.alignment,
        result.pitch_track,
        target_text,
    )


@pytest.mark.integration
def test_mfa_pipeline_on_example_audio(
    pipeline,
    example_wav_path,
    target_text,
    require_mfa,
):
    result = pipeline.run(
        example_wav_path,
        target_text,
        aligner_name="mfa",
    )

    assert result.alignment.method == "mfa"
    assert result.recognition.text.strip()

    duration = result.pitch_track.duration
    assert_segments_look_reasonable(
        result.alignment,
        target_text,
        duration,
        min_span_fraction=0.5,
    )
    assert_pitch_track_looks_reasonable(result.pitch_track)
    assert_pronunciation_plot_looks_reasonable(
        result.alignment,
        result.pitch_track,
        target_text,
    )
