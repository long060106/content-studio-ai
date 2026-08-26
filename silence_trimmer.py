"""
silence_trimmer.py

Tightens the dead air inside a cut so the clip keeps moving.

Editors doing this by hand extract the audio, look for the flat stretches in
the waveform, cut them out and drag everything left — for every gap, in every
clip. It is the definition of work worth automating, and this pipeline is
already holding the one thing needed to do it: word-level timings from Whisper.

**The one rule that must not be broken.** This project has a standing decision
that a short ends on a finished statement with an audible breath after it, and
that a clipped ending is a defect rather than a rough edge. `ending_finder`
goes to real trouble to *add* that silence. A naive silence remover would strip
it straight back out — the two features would fight, and the newer one would
quietly win. So the tail after the last word is never touched.

The same caution applies in miniature inside the clip. A pause is not
automatically dead air: a speaker who stops for half a second before the
important word is doing something deliberate, and flattening every gap to zero
turns considered delivery into a rush. So gaps are *shortened toward* a floor,
never removed outright, and short ones are left alone entirely.
"""

from __future__ import annotations

import os
import subprocess

# Gaps at or under this are part of normal speech rhythm. Left alone.
#
# Raised from 0.32. At the old values the result was technically correct and
# sounded wrong: every pause collapsed to about a quarter of a second, so one
# statement began before the previous had landed and the whole thing read as
# rushed and amateur. Removing dead air and removing the beat between thoughts
# are not the same job, and the first was quietly doing the second.
KEEP_UNDER = 0.55

# What a longer gap is shortened to. Not zero — see the module docstring.
#
# This is the beat a listener needs to register that a sentence finished. It is
# the same reasoning `ending_finder` uses for the breath at the end of a clip,
# applied to the joins in the middle.
TARGET_GAP = 0.46

# Below this saving, the re-encode is not worth it. A clip that loses a tenth
# of a second is the same clip, and every re-encode costs quality.
MIN_SAVING = 0.45


def plan_keep_ranges(
    words: list,
    duration: float,
    keep_under: float = KEEP_UNDER,
    target_gap: float = TARGET_GAP,
) -> list[tuple[float, float]]:
    """The spans to keep, in order, with the long silences squeezed out.

    Takes `caption_timing.Word`-shaped objects — anything with `.start` and
    `.end` in seconds.

    The head is trimmed to just before the first word, so a cut that opens on a
    beat of silence does not. The tail is kept in full, deliberately: that is
    the breath `ending_finder` put there.
    """
    timed = [w for w in words if getattr(w, "end", 0) > getattr(w, "start", 0)]
    if not timed:
        return [(0.0, duration)]

    ranges: list[tuple[float, float]] = []
    # A little air before the first word so it does not start on the attack.
    cursor = max(0.0, float(timed[0].start) - target_gap)

    for i, word in enumerate(timed[:-1]):
        gap = float(timed[i + 1].start) - float(word.end)
        if gap <= keep_under:
            continue
        # Keep everything up to this word plus the allowance, then jump to just
        # before the next one.
        ranges.append((cursor, float(word.end) + target_gap / 2))
        cursor = max(0.0, float(timed[i + 1].start) - target_gap / 2)

    # Everything from the cursor to the end of the file, tail included.
    ranges.append((cursor, duration))
    return [(a, b) for a, b in ranges if b - a > 0.05]


def remap_words(words: list, ranges: list[tuple[float, float]]) -> list:
    """Move word timings onto the tightened timeline.

    Cutting silence out shifts every word after each cut earlier, so the
    timings that produced the plan no longer describe the file it produced.
    Captions built from stale timings drift further out of sync with every gap
    removed — and the shot plan cuts the picture on those same timings, so the
    b-roll would drift with them.

    Recomputed arithmetically rather than by transcribing the new file again.
    The mapping is exactly known — each kept range starts where the previous
    ones ended — so a second Whisper pass would cost ten seconds a clip to
    rediscover something already certain.
    """
    if not ranges:
        return list(words)

    out = []
    for word in words:
        offset = 0.0
        for start, end in ranges:
            if word.start < start:
                break
            if word.start <= end:
                new_start = offset + (word.start - start)
                new_end = new_start + max(0.0, float(word.end) - float(word.start))
                moved = type(word)(text=word.text, start=new_start, end=new_end)
                out.append(moved)
                break
            offset += end - start
    return out


def tighten(
    media_path: str,
    words: list,
    duration: float,
    out_path: str,
    encoder_args: list[str] | None = None,
) -> tuple[str, float]:
    """Write a tightened copy. Returns `(path_used, seconds_removed)`.

    Returns the original path untouched when there is too little to gain, so
    the caller can treat the result uniformly without checking first.
    """
    ranges = plan_keep_ranges(words, duration)
    kept = sum(b - a for a, b in ranges)
    saving = duration - kept

    if saving < MIN_SAVING or len(ranges) < 2:
        return media_path, 0.0

    # One filtergraph rather than N temporary files: trim each span, reset its
    # timestamps, and concat. Audio has to be cut on exactly the same spans or
    # the voice drifts out of sync with the picture.
    parts, vlabels, alabels = [], [], []
    for i, (start, end) in enumerate(ranges):
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]"
        )
        parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
        vlabels.append(f"[v{i}]")
        alabels.append(f"[a{i}]")

    streams = "".join(v + a for v, a in zip(vlabels, alabels))
    parts.append(f"{streams}concat=n={len(ranges)}:v=1:a=1[v][a]")

    cmd = (
        ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", os.path.abspath(media_path),
         "-filter_complex", ";".join(parts),
         "-map", "[v]", "-map", "[a]"]
        + (encoder_args or ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
        + ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", os.path.abspath(out_path)]
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return media_path, 0.0
    if result.returncode != 0 or not os.path.isfile(out_path):
        return media_path, 0.0

    return out_path, saving


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Squeeze dead air out of a clip.")
    parser.add_argument("media")
    parser.add_argument("out")
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    from caption_timing import transcribe_words

    words = transcribe_words(args.media, model_size=args.model)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", args.media],
        capture_output=True, text=True,
    )
    dur = float(probe.stdout.strip() or 0)
    path, saved = tighten(args.media, words, dur, args.out)
    if saved:
        print(f"removed {saved:.2f}s of {dur:.2f}s -> {path}")
    else:
        print("nothing worth removing; original kept")
