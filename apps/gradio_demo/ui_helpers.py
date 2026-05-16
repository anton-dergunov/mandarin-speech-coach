from mandarin_speech_coach.tones.pinyin_utils import (
    format_pinyin,
)


def phrase_card_html(
    label,
    hanzi,
    pinyin_text,
    variant="target",
):
    hanzi_display = (
        hanzi
        if hanzi and hanzi != "—"
        else "—"
    )

    pinyin_display = (
        pinyin_text
        if pinyin_text and pinyin_text != "—"
        else "—"
    )

    extra = (
        " placeholder"
        if variant == "placeholder"
        else ""
    )

    card_class = (
        "phrase-card-target"
        if variant == "target"
        else f"phrase-card-heard{extra}"
    )

    return (
        f'<div class="phrase-card {card_class}">'
        f'<p class="phrase-label">{label}</p>'
        f'<p class="phrase-hanzi">{hanzi_display}</p>'
        f'<p class="phrase-pinyin">{pinyin_display}</p>'
        "</div>"
    )
