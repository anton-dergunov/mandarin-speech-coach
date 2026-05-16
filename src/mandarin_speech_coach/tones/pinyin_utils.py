from pypinyin import pinyin, Style


def format_pinyin(text):
    hanzi = "".join(
        ch
        for ch in (text or "")
        if "\u4e00" <= ch <= "\u9fff"
    )

    if not hanzi:
        return "—"

    return " ".join(
        syl[0]
        for syl in pinyin(hanzi, style=Style.TONE)
    )


def get_tone_sequence(text):
    return pinyin(text, style=Style.TONE3)
