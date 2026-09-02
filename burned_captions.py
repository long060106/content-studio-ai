"""
burned_captions.py

Speech-locked captions for any 16:9 video: white Inter Black with a heavy black
outline, low in the frame, cutting hard from phrase to phrase.

    python burned_captions.py talk.mp4                 # writes talk.captions.ass
    python burned_captions.py talk.mp4 --burn          # ...and talk.captioned.mp4
    python burned_captions.py talk.mp4 --burn --words words.json

**Why a subtitle file rather than a drawtext filter.** The pipeline's other
caption module, `kinetic_captions.py`, builds an ffmpeg `drawtext` chain — one
filter per word, positions measured in Python. That is the right shape when the
captions are part of a render that is happening anyway. It is the wrong shape
here, because what was asked for is a track that can be laid over *any* 16:9
video: an `.ass` file is a few kilobytes of text, it carries the font, the
outline, the tracking and every timing with it, any editor can import it, and
ffmpeg burns it in with a single filter. The styling lives in one `Style:` line
instead of being spread across sixty filter invocations.

**Why not reuse `caption_timing.build_ass`.** That one is karaoke: it holds a
card of several words and recolours them one at a time, and it fades the hook
in and out. This style is the opposite of karaoke — the whole phrase appears at
once on a hard cut, in one colour, and vanishes the instant the next phrase
starts. Bending the karaoke builder into this would have meant disabling the
highlight, the accent colour, the scale pop and the fade, which is most of what
it does.

**The outline replaces the brightness rule.** An earlier caption style here
inverted the text colour against the shot behind it — black on a light cut,
white on a dark one. A 5px black outline around white solves the same problem
in one fixed treatment, holds over any footage without measuring it, and cannot
flip colour mid-phrase when the shot changes underneath a caption that is still
on screen.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from caption_timing import Word, _ass_time, _escape

FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "fonts", "Inter-Black.ttf"
)
# The *full* name, not the family. `Inter-Black.ttf` reports its family as
# "Inter" and its style as "Black", and asking libass for "Inter" does not
# resolve to it even when Black is the only Inter face in `fontsdir` — it
# silently falls back to a default grotesque and renders the whole track in a
# regular weight. Setting the style's Bold flag does not rescue it either; that
# produced the same fallback face with no extra weight.
#
# "Inter Black" resolves correctly. The failure is silent and looks like
# working captions, so check the weight on a rendered frame after any change
# here rather than trusting that ffmpeg exited zero.
FONT_FAMILY = "Inter Black"

# Left at 0 deliberately. The face is already the heaviest Inter ships, and
# asking libass to embolden it on top synthesises a smeared outline.
FONT_BOLD = 0

# ASS colours are &HAABBGGRR — alpha first, then *blue* first. 00 alpha is
# fully opaque, which is what "text is fully opaque" means here.
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"

# Reference geometry. Everything below is a fraction of these, so the same
# numbers hold at 720p and 4K — libass scales the whole script from PlayRes.
REF_W, REF_H = 1920, 1080

# Caption size as a fraction of frame height. 0.083 of 1080 is 90px, which is
# the weight the reference sits at: large enough to read on a phone at arm's
# length, small enough that seven words still fit on two lines.
FONT_SIZE_FRAC = 0.083

# Outline thickness. The brief asks for 4-6px at 1080p; 5 is the middle of it.
# ScaledBorderAndShadow keeps this proportional when the video is not 1080p.
OUTLINE_PX = 5

# Shadow off, and it is not the same thing as outline. A shadow offset would
# read as a second, blurred copy of the word — the brief rules it out along
# with the background box and the glow.
SHADOW_PX = 0

# Tracking, in the thousandths-of-an-em that design tools use. ASS `Spacing` is
# in pixels, so this is converted against the font size rather than pasted in.
TRACKING_EM = -0.030          # -30, the middle of the -20..-40 the brief gives

# Where the caption block sits, as a fraction of the way down the frame. This
# is the *centre* of the block, and the block is centred on it with \an5 rather
# than anchored to the bottom — a two-line phrase then grows evenly in both
# directions instead of pushing its first line up into the picture.
CAPTION_CENTRE_FRAC = 0.80

# How wide a line may run before it is broken, as a fraction of frame width.
# Not the full width: text that reaches the frame edge reads as overflowing
# even when it technically fits.
MAX_LINE_FRAC = 0.82

# Never more than this on screen at once. The brief's ceiling is six to eight;
# seven is the middle, and the pause and width rules below usually break a
# phrase well before it gets here.
MAX_WORDS = 7

# Silence long enough to be heard as a pause, in seconds. A gap this size ends
# the phrase, which is what makes the breaks land where the speaker breathes
# rather than where a word counter happened to run out.
PHRASE_GAP = 0.32

# Held past the last word so the phrase does not vanish on the closing
# consonant. The brief allows 0.1-0.3; this is trimmed automatically whenever
# the next phrase starts sooner, because a hard cut to the next phrase always
# wins over the hold.
HOLD = 0.22

# A phrase shorter than this is unreadable no matter how briefly it was said.
MIN_ON = 0.30

# Sentence-final punctuation ends a phrase regardless of the gap: the sense has
# landed, and carrying the next sentence's first words in the same card makes
# the viewer read across a full stop.
_SENTENCE_END = ".!?"

# The account handle. **Off by default here, and that is deliberate.**
#
# The watermark is rendered by `shorts_builder.py` now, so every file the
# pipeline produces already carries it — including `short.mp4`, which is what
# this module burns captions onto. Adding one here as well put two handles on
# the same clip. It moved into the pipeline because stamping afterwards did not
# survive a re-run: the next batch rewrote short.mp4 and short_plain.mp4 from
# scratch and the mark was silently gone.
#
# It stays available for a video from anywhere else, via `--watermark`.
WATERMARK_TEXT = "@gobackforthis"

# Top-right, and the corner is chosen by elimination rather than by taste.
#
# These renders have no letterbox bars to hide in — cropdetect reports the
# picture filling all 1920x1080 on almost every shot — so the mark has to sit
# on the image, and the only question is which part of the image it damages
# least. The lower-centre band is where the captions live. The middle is where
# a face is. The top-centre is where a speaker's head reaches on a
# medium-close shot. The top corners are what is left, and the right one is
# where a viewer already expects a channel mark.
WATERMARK_X_FRAC = 0.955      # right edge of the text
WATERMARK_Y_FRAC = 0.058      # centre line of the text

# Smaller than the captions and considerably quieter. A handle is a signature,
# not a message: it should be legible when looked for and ignorable when not,
# which is the opposite of what the caption styling is trying to do.
WATERMARK_SIZE_FRAC = 0.030   # ~32px at 1080p, against the captions' 90

# ASS alpha runs 00 opaque to FF invisible, so this is about 60% opaque.
WATERMARK_ALPHA = "66"

# A thin outline, not the captions' heavy one. Without any outline the mark
# disappears entirely against a bright sky or a white wall — which is exactly
# the footage a top corner tends to contain — and with the full 5px it stops
# reading as a signature and starts competing with the captions.
WATERMARK_OUTLINE_PX = 2


def _measurer(font_path: str, size: int):
    """Measure real text width, because character counts are not widths.

    "WILL" and "iiii" are four characters each and nothing like the same width,
    so a line-break rule that counts characters breaks in the wrong place on
    roughly every other phrase.

    Falls back to an estimate if Pillow is missing rather than refusing to run —
    a slightly wrong break is a much smaller problem than no captions.
    """
    try:
        from PIL import ImageFont

        face = ImageFont.truetype(font_path, size)

        def width(text: str) -> float:
            return face.getbbox(text)[2] + TRACKING_EM * size * len(text)

        return width
    except Exception:
        def width(text: str) -> float:
            return size * 0.58 * len(text)

        return width


def group_phrases(words: list[Word]) -> list[list[Word]]:
    """Split the word stream into the cards that will appear one at a time.

    Three things end a phrase, and they are checked in this order because they
    describe the speech, the sense and the frame in decreasing authority:

    1. **A pause.** The speaker stopped; the caption should stop with them.
    2. **A full stop.** The thought finished, whether or not they paused.
    3. **The word ceiling.** A backstop for speech that never pauses.

    Width is handled by wrapping rather than by splitting, so a long phrase
    becomes two lines instead of two cards — the brief prefers a single card of
    up to seven words over two cards that cut a clause in half.

    **The ceiling does not cut at exactly seven words.** It used to, and on
    speech with no audible pause it produced cards ending mid-clause: "Now, I'm
    not saying you need a" — a card whose last word is an article, with the noun
    on the next card. When the ceiling fires, the break is placed at the best
    pause *inside* the card instead: the longest gap, preferring one that also
    falls after a comma. The words past that point are carried forward into the
    next card rather than thrown away.
    """
    # How far back from the ceiling the forced break may be pulled to land on a
    # pause. Two words, and the narrowness is the point: searching the whole
    # card for the best pause sounds better and is not — with no audible gaps
    # the tiebreak lands near the middle, the card is cut in half, the leftover
    # refills to the ceiling and is cut in half again, and the whole track
    # settles into three-word cards. The ceiling is a backstop; it should nudge
    # the break onto a pause, not re-plan the card.
    BREAK_WINDOW = 2

    def best_break(card: list[Word], gap_after: float) -> int:
        """Where to cut an over-long card. Returns a count of words to keep."""
        best, best_score = len(card), None
        for k in range(max(2, len(card) - BREAK_WINDOW), len(card) + 1):
            gap = gap_after if k == len(card) else card[k].start - card[k - 1].end
            after_comma = card[k - 1].text.rstrip().endswith((",", ";", ":", "—"))
            # Longest pause wins, a comma is worth a small pause on its own, and
            # ties go to the longer card so the default stays at the ceiling.
            score = (gap + (0.15 if after_comma else 0.0), k)
            if best_score is None or score > best_score:
                best, best_score = k, score
        return best

    phrases: list[list[Word]] = []
    current: list[Word] = []

    for i, word in enumerate(words):
        current.append(word)

        if i + 1 >= len(words):
            break

        gap = words[i + 1].start - word.end
        ends_sentence = word.text.rstrip().endswith(tuple(_SENTENCE_END))
        if gap >= PHRASE_GAP or ends_sentence:
            phrases.append(current)
            current = []
        elif len(current) >= MAX_WORDS:
            keep = best_break(current, gap)
            phrases.append(current[:keep])
            current = current[keep:]

    if current:
        phrases.append(current)
    return phrases


def wrap_lines(texts: list[str], width_of, max_width: float) -> list[str]:
    """One line if it fits, otherwise two, broken as evenly as possible.

    The break point is chosen by balance rather than by filling the first line:
    a first line running the full width above a second holding one short word
    reads as an accident. Splitting near the middle looks deliberate, and on a
    phrase of five or six words the middle is almost always a natural boundary
    anyway.
    """
    joined = " ".join(texts)
    if width_of(joined) <= max_width or len(texts) < 2:
        return [joined]

    def cost_of(split: int):
        top = " ".join(texts[:split])
        bottom = " ".join(texts[split:])
        widest = max(width_of(top), width_of(bottom))
        # Narrowest widest line first; break ties toward the more balanced pair.
        return (widest, abs(width_of(top) - width_of(bottom)))

    # A comma or a full stop is a pause the speaker actually made, so breaking
    # there is the brief's "break only at natural speech pauses" — but as a
    # *preference*, not an override.
    #
    # Tried as an override first, and it broke "I've studied scripture for
    # many, | many years." on the comma, splitting a repeated pair to obey a
    # rule. A discount rather than a veto keeps the good case ("I buried a
    # mother, | father, sister, brother,") and lets plain balance win when the
    # punctuated split is badly lopsided.
    def punctuated(split: int) -> bool:
        return texts[split - 1].rstrip().endswith((",", ";", ":", ".", "!", "?", "—"))

    def ranked(split: int):
        widest, imbalance = cost_of(split)
        # 15% off the deciding measurement. Enough to win a close call, not
        # enough to buy a line half again as wide as the alternative.
        return (widest * (0.85 if punctuated(split) else 1.0), imbalance)

    best = min(range(1, len(texts)), key=ranked)
    return [" ".join(texts[:best]), " ".join(texts[best:])]


def _header(font_size: int, video_w: int, video_h: int) -> str:
    spacing = round(TRACKING_EM * font_size, 2)
    mark_size = max(10, int(round(video_h * WATERMARK_SIZE_FRAC)))
    # The handle is set at normal tracking. The captions are tightened because
    # they are set very large, where the default spacing reads as loose; at a
    # third of that size the same tightening just makes the letters touch.
    mark_spacing = 0
    # MarginV is unused for placement — every line carries an explicit \pos —
    # but libass still wants the style complete, so it is set to something sane.
    margin_v = int(video_h * 0.08)
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Burned,{FONT_FAMILY},{font_size},{WHITE},{WHITE},{BLACK},{BLACK},{FONT_BOLD},0,0,0,100,100,{spacing},0,1,{OUTLINE_PX},{SHADOW_PX},5,60,60,{margin_v},1
Style: Mark,{FONT_FAMILY},{mark_size},&H{WATERMARK_ALPHA}FFFFFF,&H{WATERMARK_ALPHA}FFFFFF,&H{WATERMARK_ALPHA}000000,{BLACK},{FONT_BOLD},0,0,0,100,100,{mark_spacing},0,1,{WATERMARK_OUTLINE_PX},0,9,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(
    words: list[Word],
    out_path: str,
    video_w: int = REF_W,
    video_h: int = REF_H,
    offset: float = 0.0,
    duration: float | None = None,
    watermark: str | None = None,
) -> str:
    """Write the caption track. Returns the path written.

    `duration` is only needed for the watermark, which has to be told how long
    the video is — it is the one event here with no speech to derive its timing
    from. Passing None falls back to the last word's end, which is right for a
    clip that ends on the last word and short by whatever tail follows it.
    Callers with the real duration should pass it.

    `watermark=None` writes captions alone, and `words=[]` with a watermark
    writes the handle alone — which is how any other video gets stamped.

    Timing is the whole point of this function, so it is worth being explicit
    about what each phrase's end is:

    - Normally the last word's end plus `HOLD`, so the phrase does not vanish
      on a closing consonant.
    - Never past the next phrase's start. A new phrase cuts the previous one
      dead; two cards overlapping for even two frames reads as a stutter, and
      the brief asks for the previous text to disappear instantly.
    - Never shorter than `MIN_ON`, because a word said very fast still has to
      be readable — unless the next phrase starts sooner, in which case the cut
      still wins.
    """
    font_size = max(12, int(round(video_h * FONT_SIZE_FRAC)))
    width_of = _measurer(FONT_PATH, font_size)
    max_width = video_w * MAX_LINE_FRAC

    x = video_w // 2
    y = int(round(video_h * CAPTION_CENTRE_FRAC))

    phrases = group_phrases(words)
    lines = [_header(font_size, video_w, video_h)]

    for i, phrase in enumerate(phrases):
        start = max(0.0, phrase[0].start - offset)
        end = max(start, phrase[-1].end - offset) + HOLD
        if end - start < MIN_ON:
            end = start + MIN_ON
        if i + 1 < len(phrases):
            next_start = max(0.0, phrases[i + 1][0].start - offset)
            end = min(end, next_start)
        if end <= start:
            end = start + 0.08

        rows = wrap_lines([_escape(w.text) for w in phrase], width_of, max_width)
        # \N is a hard line break in ASS. \an5 anchors the block by its centre
        # so one line and two lines sit at the same point in the frame.
        text = "\\N".join(rows)
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Burned,,0,0,0,"
            f",{{\\an5\\pos({x},{y})}}{text}"
        )

    if watermark:
        # One event for the entire running time. Layer 1 so it draws above a
        # caption rather than under one — they do not overlap by design, but a
        # mark that flickers behind text on the one clip where they do would be
        # a strange thing to have to debug later.
        if duration is None:
            duration = max((w.end for w in words), default=0.0) - offset
        end = max(0.1, duration)
        mx = int(round(video_w * WATERMARK_X_FRAC))
        my = int(round(video_h * WATERMARK_Y_FRAC))
        # \an6 is middle-right, so the text grows leftward from `mx` and the
        # right edge stays put however long the handle is.
        lines.append(
            f"Dialogue: 1,{_ass_time(0.0)},{_ass_time(end)},Mark,,0,0,0,"
            f",{{\\an6\\pos({mx},{my})}}{_escape(watermark)}"
        )

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def probe_duration(video: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _ff_filter_path(path: str) -> str:
    """Escape a Windows path for the inside of an ffmpeg filter argument.

    `C:\\x\\y.ass` has to reach libass as `'C\\:/x/y.ass'` — three separate
    things, and leaving out any one of them fails differently:

    - **Backslashes become forward slashes**, or the filter parser eats them as
      escapes.
    - **The drive colon is escaped**, or everything after `C` is read as the
      next filter option. The give-away is ffmpeg complaining that it cannot
      parse the rest of your path as an `original_size`.
    - **The whole value is quoted**, because this project lives under
      `Documents/Projects/Content Studio AI` and the space in the folder name
      ends the option without it.

    Quoting alone is not enough: the colon still has to be escaped *inside* the
    quotes. That combination is the single most common reason a subtitles
    filter silently "does nothing" on Windows.
    """
    return "'" + path.replace("\\", "/").replace(":", "\\:") + "'"


def _encoder_args(crf: int) -> list:
    """Hardware encoder when available, software otherwise."""
    try:
        from shorts_builder import video_encoder_args
        return list(video_encoder_args())
    except Exception:
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                "-pix_fmt", "yuv420p"]


def burn(video: str, ass_path: str, out_path: str, crf: int = 18) -> str:
    """Render the captions into the picture. Returns the path written."""
    fonts_dir = os.path.dirname(FONT_PATH)
    vf = (
        f"subtitles={_ff_filter_path(ass_path)}"
        f":fontsdir={_ff_filter_path(fonts_dir)}"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", video,
        "-vf", vf,
        # The project's own encoder pick, which is the GPU when this machine
        # has it. Measured elsewhere in the codebase: 9.1s with libx264 at
        # -preset medium against 4.1s with QSV, and a smaller file. Captions
        # are burned once per short, so hardcoding the slow encoder here cost
        # roughly half the caption time on every batch.
        *_encoder_args(crf),
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def probe_size(video: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def load_words(path: str) -> list[Word]:
    """Word timings from JSON: [{"text": ..., "start": ..., "end": ...}, ...]"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("words", [])
    return [Word(w["text"], float(w["start"]), float(w["end"])) for w in raw]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speech-locked burned-in captions for any 16:9 video.")
    parser.add_argument("video")
    parser.add_argument("--words", help="word-timing JSON; transcribes if omitted")
    parser.add_argument("--out", help="path for the .ass (default: alongside the video)")
    parser.add_argument("--burn", action="store_true", help="also render a captioned mp4")
    parser.add_argument("--model", default="base", help="whisper size: tiny/base/small")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="seconds to subtract from every timing")
    parser.add_argument("--handle", default=WATERMARK_TEXT,
                        help=f"account handle to stamp (default: {WATERMARK_TEXT})")
    parser.add_argument("--watermark", action="store_true",
                        help="also stamp the handle. Off by default because "
                             "the pipeline already renders it into its own "
                             "output; use this for a video from elsewhere")
    parser.add_argument("--watermark-only", action="store_true",
                        help="stamp the handle and write no captions — for a "
                             "video that is captioned elsewhere, or not at all")
    args = parser.parse_args()

    if args.watermark_only:
        words = []
    elif args.words:
        words = load_words(args.words)
    else:
        from caption_timing import transcribe_words
        print(f"  transcribing {os.path.basename(args.video)}...")
        words = transcribe_words(args.video, model_size=args.model)

    if not words and not args.watermark_only:
        raise SystemExit("no word timings — nothing to caption")

    width, height = probe_size(args.video)
    stem = os.path.splitext(args.video)[0]
    ass_path = args.out or f"{stem}.captions.ass"
    mark = args.handle if (args.watermark or args.watermark_only) else None

    build_ass(words, ass_path, width, height, offset=args.offset,
              duration=probe_duration(args.video), watermark=mark)
    phrases = group_phrases(words)
    stamped = f", watermark {mark}" if mark else ""
    print(f"  {len(words)} words -> {len(phrases)} phrase(s){stamped} -> {ass_path}")

    if args.burn:
        burned = f"{stem}.captioned.mp4"
        burn(args.video, ass_path, burned)
        print(f"  burned in -> {burned}")


if __name__ == "__main__":
    main()
