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

import atexit
import json
import os
import threading
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


# Whisper transcription runs one clip at a time, even when the pipeline is
# building several shorts in parallel.
#
# This is not caution: running three transcriptions concurrently silently
# produced a clip with zero captions, while the same clip transcribed fine on
# its own. Torch models are not safe to drive from several threads at once, and
# the failure mode is the worst kind — no error, just a short that quietly ships
# without captions.
#
# It costs little. Transcription is roughly four seconds of a thirty-second
# clip, and everything around it (ffmpeg cutting, the GPU encode, the tag API
# call) still overlaps freely.
_WHISPER_LOCK = threading.Lock()
_WHISPER_MODELS: dict[str, object] = {}


def transcribe_words(
    media_path: str,
    model_size: str = "base",
    language: str = "en",
) -> list[Word]:
    """Word-level timings for a short media file.

    Uses faster-whisper (CTranslate2) rather than openai-whisper. Two reasons,
    and the first is what forced the change:

    **It doesn't need numba.** openai-whisper imports numba for its alignment
    code, and Windows Application Control blocked numba's `_box` extension on
    this machine — which killed transcription outright, took the SRT with it,
    and silently collapsed every clip to a single b-roll shot because the shot
    planner had no word timings to cut on. faster-whisper has no such
    dependency.

    **It punctuates.** YouTube's auto-captions frequently arrive with no full
    stops at all — one talk here had zero across 415 segments — which leaves
    both this code and the model guessing where a sentence ends, and is why
    clips were stopping mid-thought. These timings come with punctuation, so
    sentence boundaries are visible again.

    It is also several times faster than the original for the same model size.

    **Both engines are kept, because this machine blocks them alternately.**
    Windows Smart App Control judges native extensions by reputation, and that
    verdict is not stable: openai-whisper was blocked first (numba's `_box`
    extension), which is why faster-whisper was adopted — and later PyAV, which
    faster-whisper imports, was blocked in turn, mid-session, with no change to
    the code. Transcription is load-bearing here: without it there are no
    captions, no shot plan and no ending refinement, so the run dies whole.

    So the fast path is tried and the older engine catches it. Two engines that
    fail for unrelated reasons are worth their weight when either one going
    down stops everything.
    """
    cached = _cached_words(media_path, model_size, language)
    if cached is not None:
        return cached

    words = _transcribe_either(media_path, model_size, language)
    _remember_words(media_path, model_size, language, words)
    return words


def _transcribe_either(media_path: str, model_size: str,
                       language: str) -> list[Word]:
    try:
        return _transcribe_faster(media_path, model_size, language)
    except Exception as fast_error:
        try:
            return _transcribe_openai(media_path, model_size, language)
        except Exception:
            # Report the first failure: it is the one describing the engine
            # this project prefers, and the more useful of the two to read.
            raise fast_error


# Word timings, cached on disk against the clip they came from.
#
# Transcription is the one thing in this pipeline that a blocked native package
# can stop outright, and it is the one thing a re-render does not actually need
# to redo: the same clip yields the same timings. When Windows Application
# Control took numba *and* PyAV on the same afternoon, a rerun of three talks
# produced nothing — and every one of those clips had been transcribed an hour
# earlier.
#
# The lookup deliberately sits in front of the engines rather than inside them.
# Both `_transcribe_faster` and `_transcribe_openai` import their dependencies
# lazily, so a cache hit never reaches the import at all, and a re-render
# survives a block that would otherwise take the whole run down.
#
# Keyed by size and modification time as well as path: a clip re-cut at
# different timings is a different clip and must be transcribed again. When the
# moment is unchanged the pipeline reuses the existing file untouched, which is
# exactly the case this is for.
_WORDS_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", ".word_timings.json")
_WORDS_CACHE: dict | None = None
_WORDS_DIRTY = False
_WORDS_LOCK = threading.Lock()


def _words_key(media_path: str, model_size: str, language: str) -> str | None:
    try:
        st = os.stat(media_path)
    except OSError:
        return None
    return (f"{os.path.abspath(media_path)}::{st.st_size}::{int(st.st_mtime)}"
            f"::{model_size}::{language}")


def _words_cache() -> dict:
    global _WORDS_CACHE
    if _WORDS_CACHE is None:
        try:
            with open(_WORDS_CACHE_PATH, encoding="utf-8") as f:
                _WORDS_CACHE = json.load(f)
        except (OSError, ValueError):
            _WORDS_CACHE = {}
    return _WORDS_CACHE


def _cached_words(media_path: str, model_size: str,
                  language: str) -> list[Word] | None:
    key = _words_key(media_path, model_size, language)
    if not key:
        return None
    rows = _words_cache().get(key)
    if not rows:
        return None
    try:
        return [Word(text=r[0], start=float(r[1]), end=float(r[2]))
                for r in rows]
    except (TypeError, ValueError, IndexError):
        return None


def _remember_words(media_path: str, model_size: str, language: str,
                    words: list[Word]) -> None:
    global _WORDS_DIRTY
    key = _words_key(media_path, model_size, language)
    if not key or not words:
        return
    with _WORDS_LOCK:
        _words_cache()[key] = [[w.text, round(float(w.start), 3),
                                round(float(w.end), 3)] for w in words]
        _WORDS_DIRTY = True


def save_word_cache() -> None:
    """Write the cache out. Registered at exit; safe to call any time."""
    global _WORDS_DIRTY
    with _WORDS_LOCK:
        if not _WORDS_DIRTY or _WORDS_CACHE is None:
            return
        try:
            os.makedirs(os.path.dirname(_WORDS_CACHE_PATH), exist_ok=True)
            with open(_WORDS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_WORDS_CACHE, f)
            _WORDS_DIRTY = False
        except OSError:
            pass


atexit.register(save_word_cache)


def _transcribe_faster(media_path: str, model_size: str, language: str) -> list[Word]:
    from faster_whisper import WhisperModel

    with _WHISPER_LOCK:
        # Cached because the pipeline calls this once per short, and reloading
        # the model each time is pure waste. int8 on CPU is the right trade
        # here: the clips are short and the accuracy difference is not audible
        # in a caption.
        model = _WHISPER_MODELS.get(model_size)
        if model is None:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            _WHISPER_MODELS[model_size] = model
        # Language is stated, not detected. Auto-detection runs on a short
        # window and gets accented English wrong: on a clip of a German-based
        # teacher speaking English it returned fluent Malay, correctly timed
        # and completely useless. Every talk this pipeline handles is English,
        # so guessing buys nothing and occasionally destroys a clip.
        segments, _info = model.transcribe(
            media_path, word_timestamps=True, language=language
        )

        words: list[Word] = []
        # `segments` is a generator — the work happens as it is consumed, so it
        # has to be drained inside the lock.
        for segment in segments:
            for w in (segment.words or []):
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append(Word(text=text, start=float(w.start), end=float(w.end)))

    return words


def _transcribe_openai(media_path: str, model_size: str, language: str) -> list[Word]:
    """The fallback engine: openai-whisper, via PyTorch instead of PyAV.

    Slower and heavier, and it punctuates less consistently — but it depends on
    an entirely different set of native extensions, which is the whole point of
    keeping it. Measured on this project: five seconds for a thirty-second clip
    once the model is loaded.
    """
    import whisper

    key = f"openai:{model_size}"
    with _WHISPER_LOCK:
        model = _WHISPER_MODELS.get(key)
        if model is None:
            model = whisper.load_model(model_size)
            _WHISPER_MODELS[key] = model
        result = model.transcribe(
            media_path, word_timestamps=True, language=language, fp16=False
        )

    words: list[Word] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []) or []:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            words.append(
                Word(text=text, start=float(w["start"]), end=float(w["end"]))
            )
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


def _srt_time(seconds: float) -> str:
    """SRT wants HH:MM:SS,mmm — a comma before the milliseconds, not a point."""
    seconds = max(0.0, seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding can tip a whole second
        millis, secs = 0, secs + 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _srt_words_per_card() -> int:
    """Words per subtitle cue — the same number the render uses.

    Read from `kinetic_captions` rather than kept as its own constant here.
    The subtitle file exists so an editor can reproduce the burned-in captions,
    and two constants that are meant to agree are two constants that will
    eventually disagree — at which point the import silently regroups the words
    and the two versions of the same clip read differently.
    """
    try:
        from kinetic_captions import WORDS_PER_PHRASE

        return max(1, int(WORDS_PER_PHRASE))
    except Exception:
        return 2


def words_to_srt(words: list[Word], out_path: str,
                 per_card: int | None = None) -> str:
    """Write the same word timings as a subtitle file CapCut can import.

    The alternative to burning captions into the picture. Editors re-cut these
    clips — b-roll over the speech — and text baked into the frame fights that
    edit. But throwing the timings away entirely means retyping every caption by
    hand, which is the slowest part of assembling one of these videos.

    An SRT keeps the timing and hands the styling to the editor: CapCut imports
    it, matches the words to the audio, and the look is chosen there.

    Grouped exactly as the burned-in captions are, so importing this file
    reproduces the render rather than a variation on it. Pass `per_card` only
    to override that deliberately.
    """
    if per_card is None:
        per_card = _srt_words_per_card()

    # Grouped by sense, not by counting to three.
    #
    # This chunked every `per_card` words regardless of grammar, which split
    # sentences mid-phrase — "I'm saying? I" followed by "want to say," — and
    # that is the version an editor has to repair cue by cue, by hand, which is
    # the slowest part of finishing one of these videos and the exact work the
    # file exists to remove.
    #
    # `group_phrases` already breaks on punctuation and on a real pause, which
    # is how a person groups them anyway. Importing it here rather than
    # duplicating the rule keeps the subtitle file and the burned-in captions
    # phrased identically, so switching between them changes the look and not
    # the words.
    try:
        from kinetic_captions import group_phrases

        cards = group_phrases(words, per_phrase=per_card)
    except Exception:
        # Never lose the file over the grouping: a badly split subtitle is
        # repairable, a missing one costs the whole transcription again.
        cards = [words[i : i + per_card] for i in range(0, len(words), per_card)]

    lines: list[str] = []
    index = 1

    for n, card in enumerate(cards):
        if not card:
            continue
        start = card[0].start
        end = max(card[-1].end, start + 0.08)

        # Hold each card until the next one starts, so there is no flicker of
        # empty screen between groups.
        if n + 1 < len(cards) and cards[n + 1]:
            end = max(end, cards[n + 1][0].start)

        lines.append(str(index))
        lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        # Cleaned the same way the burned-in captions are — lowercase, no
        # commas, a full stop only where the statement ends. The subtitle file
        # and the render have to read identically, or importing it into an
        # editor silently changes the words.
        try:
            from kinetic_captions import caption_text

            shown = " ".join(t for t in (caption_text(w.text) for w in card) if t)
        except Exception:
            shown = " ".join(w.text for w in card)
        lines.append(shown)
        lines.append("")
        index += 1

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
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
