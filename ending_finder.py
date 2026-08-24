"""
ending_finder.py

Moves the end of a cut onto the moment the speaker actually finishes.

The moment finder chooses *what* to clip. It works from YouTube's transcript,
and that transcript is the problem: auto-generated captions carry no
punctuation — one talk here had zero full stops across 415 segments — so the
only remaining clue about where a sentence ends is the gap between segments.
That clue is unreliable. The `duration` values are approximate, and a gap of
over two seconds has appeared in the middle of a sentence. Endings chosen that
way came out as:

    "...but it was him or her that gave you that"   | spark that gave you...
    "...embody something inside of you and"         | if you have a goal...
    "...because also my time"                       | is going to run out...

Each one reads as the speaker running out of breath, which is exactly what a
viewer hears.

This module goes back to the audio. It transcribes a short window around the
proposed ending with faster-whisper, which *does* punctuate and which returns
real word timings, then picks the ending that satisfies both conditions a good
ending needs:

    the sentence is grammatically finished  (punctuation, and no dangling word)
    and there is real silence after it      (a measured gap, not an assumed one)

Only one of the two is enough to fail on. A complete sentence that ends in the
same instant the next one starts still sounds cut off, and a long silence after
a dangling "and" is still a broken clip.

It runs before anything is cut, because it changes where the cut is.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

# How much audio to look at around the proposed ending.
#
# The lookahead is deliberately much larger than the distance the ending is
# allowed to travel, and that margin is the whole reason this works. Whisper
# punctuates badly at the trailing edge of its window, where it has no
# following context to tell a finished sentence from an interrupted one. At a
# twelve-second lookahead it transcribed a clear "...able to unlock it." as
# "...able to unlock" with nothing after it, the ending looked unfinished, and
# the search went backwards and threw away nine seconds of the clip. The same
# audio in a twenty-two-second window punctuates correctly. So: keep every
# candidate ending well inside the window, never near its edge.
LOOKBACK = 16.0
LOOKAHEAD = 22.0

# How far the ending may travel. Forward is preferred and gets more room: the
# usual repair is letting the speaker finish the sentence they were part-way
# through, which *adds* the missing words. Pulling back removes content instead
# — often the payoff line the clip was chosen for — so it is capped tighter and
# priced higher (see PULLBACK_COST).
MAX_EXTEND = 10.0
MAX_PULLBACK = 5.0

# The breath. Below MIN_GAP there is no real pause, and the ending will sound
# slightly rushed however complete the sentence is. MAX_LULL stops a long
# silence being swallowed whole — three seconds of nothing at the end is its
# own kind of bad.
MIN_GAP = 0.30
MAX_LULL = 1.10
EDGE_MARGIN = 0.08

# The fallback, for passages Whisper declines to punctuate at all.
#
# It does that more often than the docstring above suggests — one thirty-second
# stretch of this talk came back as a single unbroken run of words with no full
# stop anywhere in it, and a search that insists on punctuation finds nothing
# and leaves the ending broken. When there is no punctuation to read, silence
# is the only evidence left that the speaker finished.
#
# That silence is measured from the audio, not from the word timings, and the
# difference is not academic. Whisper derives word boundaries from attention
# alignment, and inside a single transcribed segment it frequently hands back
# spans that touch: every word starting exactly where the last one ended. One
# passage here ran 109 consecutive words without a single gap, which read as
# unbroken speech and was nothing of the sort. ffmpeg hears the pauses that
# those timings erase.
#
# The bar is deliberately higher than MIN_GAP. A third of a second is enough of
# a breath to *keep* once a sentence is known to have ended; it is not enough on
# its own to prove that one did. Half a second and more is someone stopping.
PAUSE_FALLBACK = 0.55

# What counts as silence in a room with a microphone in it. Not digital
# zero — a lecture hall has air conditioning, an audience, and mic self-noise.
SILENCE_DB = -30
SILENCE_ALIGN = 0.30  # how close the silence must start to the word's end

# Scoring weights, in "seconds of movement" so they can be compared directly.
#
# Distance dominates, and that ordering is load-bearing. An earlier draft
# weighted breathing room heavily enough that a sentence ending eight seconds
# back with a slightly longer pause beat the correct ending — the clip ended on
# "Not huge jumping." instead of the line it was built around. Silence is worth
# having, but it is never worth seconds of the actual statement.
GAP_BONUS = 1.5        # per second of pause, up to GAP_BONUS_CAP
GAP_BONUS_CAP = 1.0    # more pause than this stops helping
NO_BREATH_PENALTY = 2.0
PULLBACK_COST = 2.0    # pulling back costs this many times extending

# How far over the length target a short may run in order to finish its
# sentence. Matches the existing rule elsewhere in this project: a couple of
# seconds long reads as fine, ending mid-word does not.
OVERRUN_ALLOWANCE = 5.0

SENTENCE_END = (".", "!", "?")

# An ellipsis is the opposite of a full stop. Whisper writes one when the
# speaker trails off or is cut short mid-thought, so "...how you're going to
# get..." matches every test for a finished sentence and is precisely the
# ending this module exists to avoid. It has to be excluded before the full
# stop at the end of it is believed.
TRAILING_OFF = ("...", "…")

# Words that cannot end a statement, whatever punctuation Whisper put after
# them. Transcription places a full stop in the wrong spot often enough that
# punctuation alone is not proof — "...inside of you and." is plausible Whisper
# output and a terrible place to stop.
#
# Only words that grammatically *require* something after them belong here:
# conjunctions, articles, prepositions, auxiliaries. Pronouns do not. An
# earlier version listed them, on the reasoning that a clip ending on "you" or
# "it" sounds unfinished — and it rejected "...starts to embody something
# inside of you." and "...it's you yourself who also is able to unlock it.",
# which are both complete sentences and both the right place to end. Three of
# eight clips were left unfixed by that one mistake. A pronoun is a perfectly
# good last word; a preposition never is.
DANGLING = {
    "and", "but", "or", "so", "because", "if", "when", "while", "that",
    "which", "who", "the", "a", "an", "of", "to", "in", "on", "at", "for",
    "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "my", "your", "its", "our", "their", "these", "those",
}


def _bare(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalpha())


def _extract_audio(source: str, start: float, duration: float, out_path: str) -> None:
    """Pull one window of audio out of the talk.

    Audio only, 16 kHz mono — which is what Whisper resamples to anyway, so
    handing it that directly costs nothing and keeps the window tiny. `-ss`
    before `-i` seeks instead of decoding, so a window from twenty minutes into
    a talk is as cheap as one from the start.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}",
            "-i", source,
            "-vn", "-ac", "1", "-ar", "16000",
            out_path,
        ],
        check=True,
        capture_output=True,
    )


def _silences(wav: str, origin: float) -> list[tuple[float, float]]:
    """Where the audio actually goes quiet, as absolute times in the talk.

    ffmpeg's `silencedetect` reports to stderr rather than producing output, so
    the run is pointed at the null muxer and the log is parsed. Failures return
    nothing, which simply means the fallback finds no candidates and the
    ending is left alone.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-i", wav,
                "-af", f"silencedetect=noise={SILENCE_DB}dB:d={PAUSE_FALLBACK}",
                "-f", "null", "-",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return []

    spans: list[tuple[float, float]] = []
    opened: float | None = None
    for line in (result.stderr or b"").decode("utf-8", "replace").splitlines():
        if "silence_start:" in line:
            try:
                opened = float(line.split("silence_start:")[1].split()[0])
            except (ValueError, IndexError):
                opened = None
        elif "silence_end:" in line and opened is not None:
            try:
                closed = float(line.split("silence_end:")[1].split()[0])
            except (ValueError, IndexError):
                opened = None
                continue
            spans.append((origin + opened, origin + closed))
            opened = None
    return spans


def _candidates(
    words: list,
    origin: float,
    proposed: float,
    max_end: float | None = None,
    punctuated: bool = True,
    silences: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float, float]]:
    """Every place the speaker finishes a statement near the proposed ending.

    Returns `(absolute_end_time, gap_after, score)`, best first.

    `punctuated` chooses the evidence. With it set, an ending must carry a full
    stop — the strong signal, and the one to try first. Without it, a clear
    pause stands in, for the passages Whisper leaves unpunctuated. Either way
    the word itself must be able to end a statement; a pause after "and" proves
    only that the speaker took a breath mid-sentence.

    Finishing a statement is a requirement, not a preference — anything that
    fails it never becomes a candidate. Among the endings that qualify, the
    nearest one wins unless something close by breathes noticeably better.
    """
    out: list[tuple[float, float, float]] = []
    for i, w in enumerate(words):
        text = (w.text or "").strip()
        if text.endswith(TRAILING_OFF):
            continue
        if punctuated and not text.endswith(SENTENCE_END):
            continue
        if _bare(text) in DANGLING:
            continue
        end = origin + float(w.end)
        if not (proposed - MAX_PULLBACK <= end <= proposed + MAX_EXTEND):
            continue
        if max_end is not None and end > max_end:
            continue
        gap = (
            origin + float(words[i + 1].start) - end
            if i + 1 < len(words)
            else MAX_LULL
        )

        if not punctuated:
            # With no full stop to go on, only measured silence counts, and it
            # has to begin about where the word does — silence that starts a
            # second later belongs to a different boundary.
            quiet = [
                (a, b) for a, b in (silences or [])
                if abs(a - end) <= SILENCE_ALIGN and (b - a) >= PAUSE_FALLBACK
            ]
            if not quiet:
                continue
            a, b = max(quiet, key=lambda s: s[1] - s[0])
            gap = max(gap, b - max(a, end))

        moved = end - proposed
        cost = abs(moved) * (1.0 if moved >= 0 else PULLBACK_COST)
        score = min(gap, GAP_BONUS_CAP) * GAP_BONUS - cost
        if gap < MIN_GAP:
            # Not disqualified — sometimes a speaker genuinely runs one
            # sentence straight into the next and there is nothing better on
            # offer — but never chosen over a nearby ending that can breathe.
            score -= NO_BREATH_PENALTY
        out.append((end, gap, score))
    out.sort(key=lambda c: c[2], reverse=True)
    return out


def refine_end(
    source: str,
    start: float,
    end: float,
    model_size: str = "base",
    final: bool = True,
    max_end: float | None = None,
) -> tuple[float, str]:
    """Where this cut should really end. Returns `(end_seconds, what_happened)`.

    `final` separates the last cut of a short from one stitched into another. A
    stitched cut runs straight on into the next piece, so a full breath there
    is dead air in the middle of the video; it still has to finish its
    sentence, but it gets a shorter tail.

    `max_end` is the latest the cut may run to before the short is simply too
    long to post. Finishing the sentence is worth going a little over the
    length target for; it is not worth doubling the clip.

    Any failure returns the original ending unchanged. A slightly worse ending
    is not worth losing a clip over, and this runs across every moment.
    """
    from caption_timing import transcribe_words

    origin = max(0.0, end - LOOKBACK)
    duration = (end - origin) + LOOKAHEAD

    handle, wav = tempfile.mkstemp(suffix=".wav", prefix="ending_")
    os.close(handle)
    try:
        try:
            _extract_audio(source, origin, duration, wav)
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            return end, f"audio window failed ({detail[-1][:60] if detail else 'ffmpeg'})"

        words = transcribe_words(wav, model_size=model_size)
        if not words:
            return end, "no words heard"

        # Punctuation first, because it is the stronger signal. Silence is the
        # fallback rather than an equal partner: a full stop says the sentence
        # ended, whereas a pause only says the speaker stopped making noise.
        found = _candidates(words, origin, end, max_end=max_end)
        evidence = ""
        if not found:
            found = _candidates(
                words, origin, end, max_end=max_end, punctuated=False,
                silences=_silences(wav, origin),
            )
            evidence = ", on a measured pause"
        if not found:
            return end, "no place to end nearby — left alone"

        best, gap, _score = found[0]

        # Never run into the next sentence, and never let the tail go slack.
        lull = max(0.0, min(gap - EDGE_MARGIN, MAX_LULL if final else 0.35))
        refined = round(best + lull, 2)
        if max_end is not None:
            refined = round(min(refined, max_end), 2)

        if refined <= start + 0.5:
            return end, "refined ending landed before the start"

        moved = refined - end
        if abs(moved) < 0.05:
            return refined, f"already right (+{lull:.2f}s breath{evidence})"
        direction = "extended" if moved > 0 else "pulled back"
        return refined, (
            f"{direction} {abs(moved):.2f}s (+{lull:.2f}s breath{evidence})"
        )
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass


def refine_moment(
    moment,
    source: str,
    model_size: str = "base",
    max_total: float | None = None,
    log=None,
) -> None:
    """Fix every cut ending in one moment, in place.

    Earlier cuts matter too. A stitched short joins one cut to the next with a
    hard edit, and a first cut that stops mid-word makes that join sound like a
    mistake rather than a choice.

    `max_total` is the length target for the whole short. Each cut is allowed
    to grow into whatever slack the others leave, plus a small overrun — the
    existing rule in this project is that running a couple of seconds long
    reads as fine while ending mid-word does not, and the same trade applies
    here.
    """
    say = log if log is not None else (lambda _msg: None)
    cuts = list(getattr(moment, "cuts", []) or [])
    for n, cut in enumerate(cuts, start=1):
        final = n == len(cuts)

        max_end = None
        if max_total is not None:
            others = sum(c.duration for c in cuts if c is not cut)
            max_end = cut.start_seconds + max(2.0, (max_total - others) + OVERRUN_ALLOWANCE)

        new_end, note = refine_end(
            source, cut.start_seconds, cut.end_seconds,
            model_size=model_size, final=final, max_end=max_end,
        )
        if new_end != cut.end_seconds:
            cut.end_seconds = new_end
        label = "ending" if final else f"cut {n} ending"
        say(f"{label}: {note}")
