"""
end_card.py

The two seconds of black that end every short: one short statement, then the
handle.

The picture stops and a single line stays. It gives the viewer a beat to hold
the idea they just heard — the moment a short is most likely to be saved or
replayed — and it ends the video on a statement rather than on the speaker's
face mid-blink.

The line is *not* the hook repeated. A hook is written to stop a scroll and is
usually a sentence; this is the idea distilled to two or three words —
"being yourself", "spreading the love" — the kind of phrase that reads as a
value rather than a summary. It has to come from the whole clip, so it is
generated from the same resolved meaning the footage search uses rather than
from the literal words spoken.

Text is wrapped by measuring the real font. Anton is condensed, so counting
characters is a poor guess at width: "WILL" and "iiii" are both four
characters and nothing like the same size.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from kinetic_captions import _ff_path, _ff_text, _measurer, find_font

SECONDS = 2.4

STATEMENT_SIZE = 96
FOLLOW_SIZE = 40
HANDLE_SIZE = 46
LINE_SPACING = 1.18
SIDE_PAD = 100
MAX_LINES = 3

COLOUR = "0xF2EFE9"          # off-white; pure white on black is harsh on a phone
MUTED = "0x9A958C"           # the follow line sits back from the statement

FOLLOW_TEXT = "follow for more"

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You write the closing line for a short motivational video.

It appears alone on a black screen for two seconds, after the speaker has \
finished. It is the last thing the viewer reads.

RULES

- Two to four words. Three is usually right.
- Name a value or a practice, not a summary of the clip. "being yourself", \
"spreading the love", "starting again", "doing the work".
- Lowercase. It is a quiet closing note, not a headline.
- No punctuation, no hashtags, no emoji.
- It must follow from THIS clip specifically. If it would fit any \
motivational video, it is wrong.
- Never repeat the hook or the speaker's own phrasing. The hook stops a \
scroll; this settles it.

Respond with ONLY a JSON object: {"line": "..."}"""


def closing_line(
    hook: str,
    quote: str = "",
    theme: str = "",
    means: str = "",
    api_key: Optional[str] = None,
) -> str:
    """The short statement for the end card.

    `means` is the clip's point with any metaphor already resolved — the same
    field `topic_tags` produces. A line drawn from the literal words of "you
    have the lock, a teacher has the key" would be about keys.

    Returns an empty string on failure; the caller falls back to the handle
    alone rather than losing the card.
    """
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv

        load_dotenv()
        client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

        parts = [f"The clip's point: {hook}"]
        if means:
            parts.append(f"Put plainly: {means}")
        if quote:
            parts.append(f'The speaker says: "{quote}"')
        if theme:
            parts.append(f"Theme: {theme}")

        response = client.messages.create(
            model=MODEL, max_tokens=100, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n".join(parts)}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[4:] if raw.startswith("json") else raw
        line = json.loads(raw.strip()).get("line", "")
        return " ".join(str(line).split()).lower()
    except Exception:
        return ""


def wrap(text: str, font_path: str, size: int, max_width: int) -> list[str]:
    """Break a line into rows that fit, measured with the actual font."""
    measure = _measurer(font_path)
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and measure(trial, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def build_filter(
    statement: str,
    handle: str,
    width: int,
    height: int,
    fps: int,
    label_out: str = "outro",
    seconds: float = SECONDS,
) -> str | None:
    """The filter chain drawing the end card, or None if it cannot be built."""
    font = find_font()
    statement = " ".join((statement or "").split())
    handle = (handle or "").strip()
    if not font or (not statement and not handle):
        return None

    max_width = width - 2 * SIDE_PAD
    size = STATEMENT_SIZE
    lines = wrap(statement, font, size, max_width) if statement else []

    # Shrink rather than overflow: a long line is better small than cropped.
    while len(lines) > MAX_LINES and size > 46:
        size = int(size * 0.88)
        lines = wrap(statement, font, size, max_width)

    line_h = int(size * LINE_SPACING)
    block_h = line_h * len(lines)

    # The statement sits a little above centre, with the follow line beneath —
    # centring the pair as a group leaves the statement looking low.
    top = (height - block_h) // 2 - 90

    draws = []
    for i, line in enumerate(lines):
        draws.append(
            f"drawtext=fontfile='{_ff_path(font)}':text={_ff_text(line)}:"
            f"fontcolor={COLOUR}:fontsize={size}:"
            f"x=(w-text_w)/2:y={top + i * line_h}"
        )

    if handle:
        follow_y = top + block_h + 120
        draws.append(
            f"drawtext=fontfile='{_ff_path(font)}':text={_ff_text(FOLLOW_TEXT)}:"
            f"fontcolor={MUTED}:fontsize={FOLLOW_SIZE}:"
            f"x=(w-text_w)/2:y={follow_y}"
        )
        draws.append(
            f"drawtext=fontfile='{_ff_path(font)}':text={_ff_text(handle)}:"
            f"fontcolor={COLOUR}:fontsize={HANDLE_SIZE}:"
            f"x=(w-text_w)/2:y={follow_y + 62}"
        )

    return (
        f"color=c=black:s={width}x{height}:r={fps}:d={seconds:.2f}[oc_bg];"
        f"[oc_bg]{','.join(draws)},setsar=1,format=yuv420p[{label_out}]"
    )


if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Preview an end card.")
    parser.add_argument("statement")
    parser.add_argument("--handle", default="@wentbackforthis1")
    parser.add_argument("--out", default="end_card.png")
    args = parser.parse_args()

    chain = build_filter(args.statement, args.handle, 1080, 1920, 30, "o")
    if not chain:
        raise SystemExit("could not build the card")
    subprocess.run(["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-filter_complex", chain, "-map", "[o]", "-frames:v", "1", args.out],
                   check=True)
    print(f"wrote {args.out}")
