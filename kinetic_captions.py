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

import os

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

FONT_SIZE = 62

# Line spacing as a multiple of the largest word on that line, not a fixed
# number of pixels.
#
# It was fixed at 78px, which quietly guaranteed overlap: the emphasised word
# of each phrase is drawn at 1.28x, so a 62px base becomes 79px — taller than
# the gap meant to hold it. Any constant here is wrong the moment a size varies,
# and the sizes vary by design.
LINE_SPACING = 1.34

INDENT = 54          # how far each line steps right — the "pyramid" stagger

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
) -> str | None:
    """The filter chain that draws the words and blends them in.

    Returns None when there is nothing to draw or no usable font, so the caller
    can fall through to the ungraded video rather than special-casing.
    """
    font = find_font()
    if not font or not words:
        return None

    phrases = group_phrases(words)
    draws: list[str] = []
    measure = _measurer(font)

    for p, phrase in enumerate(phrases):
        # The phrase stays up until its last word has been said, then clears —
        # but never past the moment the next phrase starts drawing.
        #
        # Without that cap the phrases overlap. Speech rarely leaves 0.28s
        # between one clause and the next, so the outgoing phrase was still on
        # screen when the incoming one appeared, and both were drawn at once:
        # two stacks of words on top of each other, unreadable, and easy to
        # mistake for a layout or spacing problem rather than a timing one.
        phrase_end = float(phrase[-1].end) + 0.28
        if p + 1 < len(phrases):
            next_start = float(phrases[p + 1][0].start)
            phrase_end = min(phrase_end, next_start - 0.02)
        # A phrase whose words all land inside a hair of each other would
        # otherwise get a negative window and never draw at all.
        phrase_end = max(phrase_end, float(phrase[-1].end) + 0.05)

        # Lay the block out first, measuring every word, then draw it. Sizes
        # differ within a phrase, so spacing has to follow the actual glyphs
        # rather than a constant that is only correct for one of them.
        sizes = [
            int(FONT_SIZE * (EMPHASIS_SCALE if i == len(phrase) - 1 else 1.0))
            for i in range(len(phrase))
        ]
        heights = [int(s * LINE_SPACING) for s in sizes]
        block_h = sum(heights)
        top = band_top + (band_height - block_h) // 2

        offsets, running = [], 0
        for h in heights:
            offsets.append(running)
            running += h

        for i, word in enumerate(phrase):
            text = _ff_text(word.text.strip())
            if not text:
                continue
            size = sizes[i]
            x = picture_left + EDGE_PAD + i * INDENT
            # Pull a long word back so it cannot run past the picture's edge.
            width = measure(word.text.strip(), size)
            right_limit = picture_left + picture_width - EDGE_PAD
            if x + width > right_limit:
                x = max(picture_left + EDGE_PAD, right_limit - width)
            y = top + offsets[i]
            # Each word appears when it is spoken and stays for the rest of the
            # phrase — that is what builds the stack rather than flashing one
            # word at a time.
            # Quoted, so the commas inside between() survive the graph-level
            # split. See _ff_text for why quoting rather than escaping.
            enable = f"'between(t,{float(word.start):.3f},{phrase_end:.3f})'"
            draws.append(
                f"drawtext=fontfile='{_ff_path(font)}':text={text}:"
                f"fontcolor={CAPTION_COLOUR}:fontsize={size}:x={x}:y={y}:"
                f"enable={enable}"
            )

    if not draws:
        return None

    # Drawn straight onto the picture, opaque.
    #
    # This used to composite the words on a separate canvas and blend them in,
    # so the text picked up the image underneath. That was the reference
    # style's look and it was rejected: on dark footage the words were faint,
    # and lifting the opacity to fix that removed the very quality that made it
    # a blend. Solid white reads at a glance, which is what a caption is for.
    return f"[{label_in}]{','.join(draws)},format=yuv420p[{label_out}]"



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
