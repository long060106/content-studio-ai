"""
caption_timing.py

Turns a short clip into word-accurate burned-in captions — the big bouncing
text that carries a motivational short.

Whisper runs locally with `word_timestamps=True` on the *extracted clip only*
(15-35 seconds), never the full talk. On CPU that's a few seconds per clip and
costs nothing. YouTube's own captions are too coarse for this: they group 3-5
seconds of speech into one block, so you can't highlight the word being spoken.

Output is an ASS subtitle file. ASS (rather than drawtext) because libass
handles the outline, shadow, wrapping and per-word colouring for us, and one
`subtitles=` filter burns the whole thing in.

The karaoke effect works by emitting one Dialogue line per word: each line
shows the same group of words, with only the current word in the accent
colour. Cheap, robust, and it looks like the captions every motivational
short on TikTok uses.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

# Whisper prints an FP16 warning on every CPU run; it's noise in the pipeline log.
warnings.filterwarnings("ignore", message=".*FP16.*")

# 1080x1920 vertical canvas.
VIDEO_W = 1080
VIDEO_H = 1920

# Words shown together on screen. Three reads well on a phone: big enough to
# fill the width, short enough that the eye lands on the highlighted word.
WORDS_PER_CARD = 3

# ASS colours are &HAABBGGRR — alpha first, then *blue, green, red*.
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
ACCENT = "&H0000E5FF"  # amber/gold


@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe_words(media_path: str, model_size: str = "base") -> list[Word]:
    """Word-level timings for a short media file, using local Whisper."""
    import whisper  # imported lazily: torch is a heavy import

    model = whisper.load_model(model_size)
    result = model.transcribe(media_path, word_timestamps=True, fp16=False)

    words: list[Word] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = w.get("word", "").strip()
            if not text:
                continue
            words.append(Word(text=text, start=float(w["start"]), end=float(w["end"])))
    return words


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):d}:{int(minutes):02d}:{secs:05.2f}"


def _escape(text: str) -> str:
    """ASS treats braces as override blocks and backslashes as escapes."""
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _header(font: str, font_size: int) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_W}
PlayResY: {VIDEO_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{font_size},{WHITE},{WHITE},{BLACK},{BLACK},1,0,0,0,100,100,0,0,1,7,4,2,90,90,430,1
Style: Hook,{font},{int(font_size * 0.78)},{WHITE},{WHITE},{BLACK},{BLACK},1,0,0,0,100,100,0,0,1,6,3,8,80,80,240,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _card_text(card: list[Word], active_index: int) -> str:
    """One caption card with a single word highlighted."""
    parts = []
    for i, w in enumerate(card):
        body = _escape(w.text)
        if i == active_index:
            # Accent colour plus a slight scale-up: the "pop" on the spoken word.
            parts.append(f"{{\\c{ACCENT}\\fscx108\\fscy108}}{body}{{\\c{WHITE}\\fscx100\\fscy100}}")
        else:
            parts.append(body)
    return " ".join(parts)


def build_ass(
    words: list[Word],
    out_path: str,
    hook: str = "",
    hook_seconds: float = 2.6,
    font: str = "Arial Black",
    font_size: int = 96,
    offset: float = 0.0,
) -> str:
    """Write an ASS file with karaoke captions and an optional opening hook."""
    lines = [_header(font, font_size)]

    if hook:
        lines.append(
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(hook_seconds)},Hook,,0,0,0,"
            f",{{\\fad(200,300)}}{_escape(hook.upper())}"
        )

    for i in range(0, len(words), WORDS_PER_CARD):
        card = words[i : i + WORDS_PER_CARD]
        for j, word in enumerate(card):
            start = max(0.0, word.start - offset)
            # Hold the last word of a card until the next card starts so there's
            # no flicker of blank screen between groups.
            if j + 1 < len(card):
                end = max(start, card[j + 1].start - offset)
            else:
                end = max(start, word.end - offset)
            if end <= start:
                end = start + 0.08
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,"
                f",{_card_text(card, j)}"
            )

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def captions_for_clip(
    media_path: str,
    out_path: str,
    hook: str = "",
    model_size: str = "base",
    font_size: int = 96,
) -> tuple[str, list[Word]]:
    """Convenience: transcribe a clip and write its caption file in one call."""
    words = transcribe_words(media_path, model_size=model_size)
    build_ass(words, out_path, hook=hook, font_size=font_size)
    return out_path, words


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python caption_timing.py <clip.mp4> <out.ass> [hook text]")
        sys.exit(1)

    hook_text = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
    path, found = captions_for_clip(sys.argv[1], sys.argv[2], hook=hook_text)
    print(f"{len(found)} words -> {path}")
