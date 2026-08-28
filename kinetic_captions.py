"""
kinetic_captions.py

Word-by-word captions that appear as they are spoken, drawn onto the picture.

The reference style does this by hand in CapCut: split the auto-caption into
individual words, place each one, size it, and drag it back to the frame where
that word is actually said. For a forty-second clip that is sixty words placed
by hand, and the word timings needed to do it automatically are already
computed here for the SRT.

**The text is opaque, not blended.** An earlier version composited the words on
a separate canvas and blended them into the frame, so each word picked up the
image behind it. That is what the reference does and it was rejected here: on
dark footage the words came out faint, and raising the opacity to fix that
removed the exact quality that made it a blend. A caption exists to be read at
a glance. Solid white does that; a clever one does not.

Words are laid out by measuring the real font rather than counting characters.
Anton is condensed, so "WILL" and "iiii" are both four characters and nothing
like the same width.
"""

from __future__ import annotations

import math
import os
import subprocess

# Modern, clean, slightly condensed — the closest thing on a stock Windows
# install to the display faces these edits use. Ordered by preference; the
# first that exists wins.
# Heavy weights first. The earlier list led with semibold faces, which read as
# thin and administrative at caption size — the words have to hold their own
# against a moving picture behind them, and a medium weight does not.
#
# A downloaded display face (Poppins, Anton, Bebas Neue) would be better still;
# these are the heaviest faces a stock Windows install actually has.
BUNDLED_FONT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "fonts"
)

FONT_CANDIDATES = [
    # Anton, shipped with the project. It is the heavy condensed face these
    # edits actually use, and it is a real step up from anything Windows
    # includes — the stock faces are all either too light or too wide.
    #
    # Bundled rather than assumed: a caption that silently falls back to a
    # different face changes the look of every clip, and nothing about the
    # output would say why. Licensed under the SIL Open Font License, which
    # permits redistribution; OFL.txt sits beside it as that licence requires.
    os.path.join(BUNDLED_FONT_DIR, "Anton-Regular.ttf"),
    os.path.join(BUNDLED_FONT_DIR, "Poppins-Bold.ttf"),
    os.path.join(BUNDLED_FONT_DIR, "BebasNeue-Regular.ttf"),
    r"C:\Windows\Fonts\Poppins-Bold.ttf",
    r"C:\Windows\Fonts\Anton-Regular.ttf",
    r"C:\Windows\Fonts\Montserrat-Bold.ttf",
    r"C:\Windows\Fonts\seguibl.ttf",              # Segoe UI Black
    r"C:\Windows\Fonts\ariblk.ttf",               # Arial Black
    r"C:\Windows\Fonts\impact.ttf",               # Impact
    r"C:\Windows\Fonts\seguisb.ttf",              # Segoe UI Semibold
    r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Words held on screen together. The reference keeps these short — three words
# is a glance, seven is reading, and a viewer who is reading has stopped
# listening.
WORDS_PER_PHRASE = 3

# A phrase breaks early when the speaker pauses this long, so the text follows
# the sense of the sentence rather than an arbitrary count.
PHRASE_GAP = 0.55

# Large. The reference puts one short phrase across most of the frame's width,
# which is what makes it readable at a glance on a phone.
FONT_SIZE = 104

# How far the caption sits below the picture's bottom edge.
#
# The square window leaves 460px of black under the picture, and that is where
# the caption goes. Placing it there is what makes "never over the speaker" a
# property of the layout rather than something measured and hoped for — there
# is no character in the black.
#
# An earlier version divided the picture into bands, measured each for
# busyness, and put the text in the quietest one. That reasoning is sound for a
# wide letterbox and wrong for a square: a square window is chosen precisely
# because a person fills it, so every band has the subject in it and "quietest"
# picks his chest instead of his face. The measurement was working; the place
# it was measuring had no right answer.
CAPTION_GAP = 40

# Gap between words on a line, as a fraction of the base size.
WORD_GAP = 0.26

# Which band to use when the margin is too shallow to hold the text and the
# caption has to go back inside the picture. The upper area is the safer
# default: subjects sit centre and low far more often than high.
DEFAULT_ROW = 1

# The last word of a phrase is the one that lands, so it gets to be bigger.
#
# 1.28x was too timid to read as emphasis — it looked like inconsistent sizing
# rather than a deliberate accent. The gap has to be obvious enough that the
# eye goes to the payoff word without being told.
EMPHASIS_SCALE = 1.75

# Keep the block clear of the picture's edges, so a long word never runs into
# the rounded corner or off the frame.
EDGE_PAD = 48


# Plain white. The off-white was chosen to help the words take colour from the
# image while blending; drawn opaque there is nothing to blend with, and white
# is what reads.
CAPTION_COLOUR = "white"


def find_font() -> str | None:
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _ff_path(path: str) -> str:
    """A Windows font path ffmpeg's filter parser will accept.

    Backslashes and the drive colon both mean something inside a filtergraph,
    so the path is given with forward slashes and an escaped colon.
    """
    return path.replace("\\", "/").replace(":", r"\:")


def _ff_text(text: str) -> str:
    """One word as a single-quoted drawtext value.

    Two escaping rules fight each other here, and getting them the wrong way
    round cost two failed renders:

    - Option values **must** be quoted. ffmpeg parses a filtergraph twice —
      once to split on commas and semicolons, then again per filter to split
      options on colons — and a backslash-escaped comma inside an unquoted
      value still derails the first pass. It fails with "Error parsing
      filterchain ... around: [label]", which points at the output label rather
      than at the comma actually responsible.
    - A quote **cannot** be backslash-escaped inside a single-quoted string.
      The only way to include one is to close the quote, emit an escaped quote,
      and reopen: `'he'\\''s'`. Backslashing it instead ends the string early
      and corrupts everything after it — which is why the first attempt died
      on a timestamp, several words downstream of the real cause.

    Apostrophes are not an edge case in speech: "he's", "don't" and "you've"
    all appear in the first ten seconds of the test clip.
    """
    return "'" + text.replace("'", "'\\''") + "'"


def _measurer(font_path: str):
    """A function giving the pixel width of a word at a given size.

    Measured with the real font rather than estimated from character count.
    Falls back to a rough estimate if Pillow cannot open the face, since a
    slightly wrong margin is much better than no captions at all.
    """
    try:
        from PIL import ImageFont
    except ImportError:
        return lambda text, size: int(len(text) * size * 0.55)

    cache: dict[int, object] = {}

    def width(text: str, size: int) -> int:
        face = cache.get(size)
        if face is None:
            try:
                face = ImageFont.truetype(font_path, size)
            except OSError:
                return int(len(text) * size * 0.55)
            cache[size] = face
        try:
            box = face.getbbox(text)
            return int(box[2] - box[0])
        except Exception:
            return int(len(text) * size * 0.55)

    return width


def shot_at(shots: list, when: float) -> tuple[str, float] | None:
    """Which clip is on screen at `when`, and where to seek inside it.

    `shots` is the render's own plan — (path, source_start, seconds) in playing
    order — so this answers the question the caption placer actually needs:
    what picture will this word be drawn on top of?
    """
    running = 0.0
    for path, source_start, seconds in shots:
        span = max(0.2, float(seconds))
        if when < running + span:
            return path, float(source_start) + (when - running)
        running += span
    if shots:
        path, source_start, seconds = shots[-1]
        return path, float(source_start) + max(0.0, float(seconds) - 0.1)
    return None


def read_bands(path: str, seek: float, rows: int = 6) -> list[tuple[float, float]] | None:
    """Brightness and busyness of each horizontal band of one frame.

    Returns `(mean, detail)` per band, top to bottom, both 0..1. `detail` is
    the standard deviation of luminance — low means an empty region like sky or
    a plain wall, high means a face or foliage.

    This is what lets a caption be placed where nothing is happening and
    coloured against what is behind it, rather than dropped in a fixed spot and
    hoped for.
    """
    w, h = 32, 6 * rows
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-ss", f"{max(0.0, seek):.2f}", "-i", path, "-vframes", "1",
         "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True)
    d = r.stdout
    if len(d) < w * h:
        return None

    out = []
    per = h // rows
    for b in range(rows):
        vals = [d[y * w + x] / 255 for y in range(b * per, (b + 1) * per) for x in range(w)]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        out.append((mean, math.sqrt(var)))
    return out


def group_phrases(words: list, per_phrase: int = WORDS_PER_PHRASE,
                  gap: float = PHRASE_GAP) -> list[list]:
    """Split the word stream into short phrases.

    Breaks on a real pause as well as on length, so the grouping follows how
    the line was actually spoken instead of chopping every three words
    regardless.
    """
    phrases: list[list] = []
    current: list = []
    for i, word in enumerate(words):
        current.append(word)
        last = i + 1 >= len(words)
        text = word.text.strip()

        # Punctuation is the strongest signal available and the cheapest to
        # read. Without it the grouping chops across sense — "the cup is" then
        # "empty, now it" — because a fixed count knows nothing about grammar.
        ends_clause = text.endswith((",", ".", "!", "?", ";", ":"))
        long_enough = len(current) >= per_phrase
        pause_next = (
            not last
            and float(words[i + 1].start) - float(word.end) >= gap
        )
        if last or ends_clause or long_enough or pause_next:
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases


def build_filter(
    words: list,
    band_top: int,
    band_height: int,
    picture_left: int,
    picture_width: int,
    label_in: str = "vbase",
    label_out: str = "vtxt",
    shots: list | None = None,
    frame_height: int = 1920,
) -> str | None:
    """Draw each phrase under the picture, in a colour that contrasts.

    Three rules, all taken from what the reference edits actually do:

    - **Never over the subject.** The caption sits in the black margin below the
      picture. Not "usually clear of him" — structurally clear, because that
      part of the frame holds no picture at all.
    - **Never past the frame.** Text is sized down until the whole line fits
      inside the picture's width, so a long word cannot run off the edge or
      collide with the rounded corner.
    - **Never the same colour as what is behind it.** White on black is the
      normal case. When the margin is too shallow to hold the line and the text
      has to sit on the picture, the pixels it will cover are measured: a bright
      shot gets black text, a dark one white.

    `shots` is the render's shot plan, and is only consulted in that last case —
    on black there is nothing to measure and nothing to decide.
    """
    font = find_font()
    if not font or not words:
        return None

    phrases = group_phrases(words)
    measure = _measurer(font)
    draws: list[str] = []

    rows = 6
    band_h = band_height // rows
    band_bottom = band_top + band_height
    margin_below = max(0, frame_height - band_bottom)
    usable = picture_width - 2 * EDGE_PAD

    for p, phrase in enumerate(phrases):
        phrase_end = float(phrase[-1].end) + 0.28
        if p + 1 < len(phrases):
            phrase_end = min(phrase_end, float(phrases[p + 1][0].start) - 0.02)
        phrase_end = max(phrase_end, float(phrase[-1].end) + 0.05)

        texts = [w.text.strip() for w in phrase if w.text.strip()]
        if not texts:
            continue

        # Lay the words out at a size that fits, with the last one larger.
        #
        # The last word of a phrase is where the sense lands, and sizing it up
        # is what makes the caption look edited rather than transcribed. It has
        # to be part of the fitting loop rather than applied afterwards, or the
        # emphasis is exactly what pushes the line off the frame.
        size = FONT_SIZE
        while size > 30:
            sizes = [size] * len(texts)
            sizes[-1] = int(size * EMPHASIS_SCALE)
            widths = [measure(t, s) for t, s in zip(texts, sizes)]
            total = sum(widths) + int(size * WORD_GAP) * (len(texts) - 1)
            if total <= usable:
                break
            size = int(size * 0.92)

        line_h = max(sizes)

        # Below the picture if the margin can hold it, which with a square
        # window it always can; inside the quietest band only as a fallback.
        if margin_below >= line_h + CAPTION_GAP:
            y = min(band_bottom + CAPTION_GAP, frame_height - line_h - 12)
        else:
            y = band_top + DEFAULT_ROW * band_h + (band_h - line_h) // 2
            y = max(band_top + EDGE_PAD,
                    min(y, band_bottom - line_h - EDGE_PAD))

        colour, shadow = _colours_at(y, line_h, band_top, band_bottom, rows,
                                     shots, float(phrase[0].start))

        x = picture_left + (picture_width - total) // 2
        for text, word_size, width in zip(texts, sizes, widths):
            draws.append(
                f"drawtext=fontfile='{_ff_path(font)}':text={_ff_text(text)}:"
                f"fontcolor={colour}:fontsize={word_size}:"
                f"shadowcolor={shadow}:shadowx=2:shadowy=2:"
                # Bottom-aligned, not top-aligned: drawtext positions the top of
                # the box, so equal y values would leave the emphasised word
                # hanging below the others instead of sharing their baseline.
                f"x={x}:y={y + line_h - word_size}:"
                f"enable='between(t,{float(phrase[0].start):.3f},{phrase_end:.3f})'"
            )
            x += width + int(size * WORD_GAP)

    if not draws:
        return None

    # Drawn straight onto the picture, opaque. The blend this replaced went
    # faint on dark footage, and raising its opacity removed the very quality
    # that made it a blend.
    return f"[{label_in}]{','.join(draws)},format=yuv420p[{label_out}]"


def _colours_at(y: int, line_h: int, band_top: int, band_bottom: int,
                rows: int, shots: list | None, when: float) -> tuple[str, str]:
    """Text and shadow colours for whatever the line will be drawn over.

    The measurement is of the strip the text actually covers, not of the shot
    in general. That distinction is the whole point: a shot can be bright at the
    top and black at the bottom, and a caption low in the frame cares only about
    the bottom.

    Text sitting clear of the picture needs no measurement — the margin is
    black, so the answer is white and a frame does not have to be decoded to
    learn it.
    """
    overlaps = y + line_h > band_top and y < band_bottom
    if not overlaps or not shots:
        return _contrast_for(0.0)

    at = shot_at(shots, when)
    info = read_bands(at[0], at[1], rows=rows) if at else None
    if not info:
        return _contrast_for(0.0)

    # Which bands of the picture the text crosses.
    band_h = max(1, (band_bottom - band_top) // rows)
    first = max(0, (y - band_top) // band_h)
    last = min(rows - 1, (y + line_h - band_top) // band_h)
    covered = info[int(first):int(last) + 1] or info
    return _contrast_for(sum(m for m, _ in covered) / len(covered))


def _contrast_for(mean: float) -> tuple[str, str]:
    """Text and shadow colours for a background of this brightness.

    The shadow matters most in the middle of the range, where neither black nor
    white is clearly right and a word can otherwise disappear into the picture
    for the second it is on screen.
    """
    if mean > 0.58:
        return "black", "white@0.35"
    if mean < 0.34:
        return "white", "black@0.45"
    return "white", "black@0.75"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview the caption grouping.")
    parser.add_argument("media")
    args = parser.parse_args()

    from caption_timing import transcribe_words

    words = transcribe_words(args.media)
    print(f"font: {find_font()}")
    for phrase in group_phrases(words):
        span = f"{float(phrase[0].start):6.2f}-{float(phrase[-1].end):6.2f}"
        print(f"  {span}  {' '.join(w.text.strip() for w in phrase)}")
