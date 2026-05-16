import os

import gradio as gr
from pypinyin import pinyin, Style

from mandarin_speech_coach.core.pipeline import (
    PronunciationPipeline,
)
from mandarin_speech_coach.visualization.pronunciation_plot import (
    PronunciationPlotBuilder,
)
from mandarin_speech_coach.tones.pinyin_utils import (
    format_pinyin,
)

from apps.gradio_demo.css import PHRASE_CARD_CSS
from apps.gradio_demo.ui_helpers import (
    phrase_card_html,
)

TARGET_TEXT = "我喜欢机器学习"

TARGET_PINYIN = format_pinyin(TARGET_TEXT)

TARGET_PINYIN_TONES = pinyin(
    TARGET_TEXT,
    style=Style.TONE3,
)

pipeline = PronunciationPipeline()

plot_builder = PronunciationPlotBuilder()

TARGET_PHRASE_HTML = phrase_card_html(
    "Say this phrase",
    TARGET_TEXT,
    TARGET_PINYIN,
    variant="target",
)

HEARD_PLACEHOLDER_HTML = phrase_card_html(
    "What we heard",
    "—",
    "Record audio, then click Analyze",
    variant="placeholder",
)


def analyze(audio_filepath, alignment_method):
    aligner = (
        "ctc"
        if "CTC" in alignment_method
        else "mfa"
    )

    analysis = pipeline.run(
        audio_filepath,
        TARGET_TEXT,
        aligner_name=aligner,
    )

    fig = plot_builder.build(
        analysis.alignment,
        analysis.pitch_track,
        TARGET_PINYIN_TONES,
    )

    heard_html = phrase_card_html(
        "What we heard",
        analysis.recognition.text,
        format_pinyin(analysis.recognition.text),
        variant="heard",
    )

    return heard_html, fig


with gr.Blocks(
    css=PHRASE_CARD_CSS,
    title="Mandarin Speech Coach",
) as demo:
    gr.Markdown("# Mandarin Pronunciation Practice Tool")

    with gr.Row(elem_classes=["phrase-row"]):
        target_phrase = gr.HTML(
            value=TARGET_PHRASE_HTML
        )
        heard_phrase = gr.HTML(
            value=HEARD_PLACEHOLDER_HTML
        )

    with gr.Row():
        with gr.Column(scale=1):
            mic_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record Your Voice",
            )

            alignment_method_dd = gr.Radio(
                ["CTC (Default)", "MFA"],
                label="Alignment Method",
                info="CTC is faster and runs anywhere. MFA is classic but requires local setup.",
                value="CTC (Default)",
            )

            analyze_btn = gr.Button(
                "Analyze Pronunciation",
                variant="primary",
            )

        with gr.Column(scale=2):
            output_plot = gr.Plot()

    analyze_btn.click(
        fn=analyze,
        inputs=[
            mic_input,
            alignment_method_dd,
        ],
        outputs=[
            heard_phrase,
            output_plot,
        ],
    )

demo.launch(
    debug=os.environ.get(
        "GRADIO_DEBUG",
        "0",
    ) == "1",

    server_name=os.environ.get(
        "GRADIO_SERVER_NAME",
        "127.0.0.1",
    ),

    server_port=int(
        os.environ.get(
            "GRADIO_SERVER_PORT",
            "7860",
        )
    ),
)
