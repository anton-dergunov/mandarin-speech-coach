from mandarin_speech_coach.tones.pinyin_utils import (
    format_pinyin,
    get_tone_sequence,
)


def test_format_pinyin_basic():
    assert format_pinyin("你好") == "nǐ hǎo"


def test_format_pinyin_filters_non_hanzi():
    assert format_pinyin("你好123") == "nǐ hǎo"
    assert format_pinyin("Hello你好!") == "nǐ hǎo"


def test_format_pinyin_empty_inputs():
    assert format_pinyin("") == "—"
    assert format_pinyin(None) == "—"
    assert format_pinyin("123") == "—"


def test_get_tone_sequence():
    result = get_tone_sequence("你好")
    assert result == [["ni3"], ["hao3"]]
