"""
sound_design.py

Puts a click on every cut.

This is the cheapest trick in cinematic editing and the one that does the most
work. A hard cut with nothing under it reads as a video changing shots; the
same cut with a short mechanical click under it reads as a decision. Editors
doing this by hand drag a click onto the timeline and nudge it onto each cut,
one at a time, for every cut in the edit — which is exactly the kind of
mechanical, repetitive work worth handing to a program.

The pipeline already knows where every cut is, because it placed them. So the
click track is not something to detect; it is something to render straight from
the shot plan.

**The clicks are synthesised, not downloaded.** A sound library would mean an
API key, a licence to honour per file, and a network call in the middle of a
render. A click is a burst of noise that decays in about thirty milliseconds —
cheap to generate, free of licensing, identical every run, and it works with no
internet at all. Drop real files into `assets/sfx/click/` and those are used
instead, on exactly the same principle as the curated b-roll library: hand-
picked beats generated, always.

Everything here is standard library plus one ffmpeg call for decoding. That is
deliberate — the machine this runs on blocks numba's native extension, which
already cost this project a working transcriber once.
"""

from __future__ import annotations

import array
import math
import os
import random
import subprocess
import wave

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLICK_DIR = os.path.join(BASE_DIR, "assets", "sfx", "click")

SAMPLE_RATE = 48000          # matches the render's audio rate; no resampling
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}

# How loud a click sits under the speech. Clicks are transients — they read as
# loud even at a low level, and pushing them up buries the voice, which is the
# one thing the viewer is actually there for.
CLICK_GAIN = 0.22

# A click that lands exactly on the cut frame reads as slightly late, because
# the transient takes a moment to register. Nudging it a hair earlier makes it
# feel locked to the picture.
CLICK_LEAD = 0.012


def _decode_to_pcm(path: str) -> array.array:
    """One audio file as mono 16-bit samples at SAMPLE_RATE.

    Uses ffmpeg rather than the `wave` module so the folder can hold mp3s and
    whatever else, and so sample rates that don't match are resampled properly
    rather than played back at the wrong pitch.
    """
    try:
        out = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", path,
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
            ],
            capture_output=True, timeout=60, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return array.array("h")

    samples = array.array("h")
    # An odd trailing byte would raise; s16le is always even, but a truncated
    # file is not worth crashing a render over.
    data = out.stdout
    samples.frombytes(data[: len(data) - (len(data) % 2)])
    return samples


def synthesise_click(seed: int) -> array.array:
    """One short mechanical click.

    Built from two parts, because either alone sounds wrong. A noise burst on
    its own is a hiss; a tone on its own is a beep. Together, with a decay fast
    enough to be over before it is consciously heard, they read as a mechanical
    click.

    `seed` makes each click slightly different in pitch and length while
    keeping the whole track reproducible — a render run twice produces the same
    audio. Identical clicks in a row sound like a machine, which is worse than
    it sounds like a mistake.
    """
    rng = random.Random(seed)
    decay = rng.uniform(0.018, 0.032)          # seconds to near-silence
    tone_hz = rng.uniform(1800.0, 3200.0)
    length = int(SAMPLE_RATE * decay * 2.2)

    out = array.array("h", bytes(length * 2))
    prev = 0.0
    for n in range(length):
        t = n / SAMPLE_RATE
        env = math.exp(-t / (decay * 0.32))
        if env < 0.0005:
            break

        # High-passed noise: the difference of consecutive noise samples tilts
        # the spectrum up, which is what makes it read as a "tick" rather than
        # a "thud".
        white = rng.uniform(-1.0, 1.0)
        noise = white - prev
        prev = white

        tone = math.sin(2.0 * math.pi * tone_hz * t)
        sample = env * (0.62 * noise + 0.38 * tone)
        out[n] = max(-32768, min(32767, int(sample * 32767 * 0.8)))
    return out


def _click_bank(count: int) -> list[array.array]:
    """The clicks to draw from — curated if any exist, synthesised otherwise."""
    curated: list[array.array] = []
    if os.path.isdir(CLICK_DIR):
        for name in sorted(os.listdir(CLICK_DIR)):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                pcm = _decode_to_pcm(os.path.join(CLICK_DIR, name))
                if len(pcm):
                    curated.append(pcm)
    if curated:
        return curated
    # A handful of variants is enough; they are cycled with an offset so the
    # same click rarely lands twice in a row.
    return [synthesise_click(seed) for seed in range(6)]


def build_click_track(
    cut_times: list[float],
    total_duration: float,
    out_path: str,
    gain: float = CLICK_GAIN,
) -> str | None:
    """Write a mono WAV holding one click at each cut. Returns the path.

    Returns None when there is nothing to place, so the caller can skip mixing
    entirely rather than paying for a silent track.
    """
    usable = [t for t in cut_times if 0.0 < t < total_duration]
    if not usable or total_duration <= 0:
        return None

    bank = _click_bank(len(usable))
    if not bank:
        return None

    total_samples = int(total_duration * SAMPLE_RATE) + SAMPLE_RATE
    mix = array.array("i", bytes(total_samples * 4))   # 32-bit while summing

    for i, when in enumerate(sorted(usable)):
        click = bank[i % len(bank)]
        start = int(max(0.0, when - CLICK_LEAD) * SAMPLE_RATE)
        for n, value in enumerate(click):
            pos = start + n
            if pos >= total_samples:
                break
            mix[pos] += int(value * gain)

    # Down to 16-bit, clamping rather than wrapping. Two clicks landing on top
    # of each other is rare but a wrapped sample is an audible crack.
    out = array.array("h", bytes(total_samples * 2))
    for n in range(total_samples):
        out[n] = max(-32768, min(32767, mix[n]))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(out.tobytes())
    return out_path


def cut_times_from_shots(shots: list) -> list[float]:
    """Where the picture changes, as offsets into the finished short.

    The first shot's start is not a cut — nothing precedes it to cut from — so
    the boundaries are the running totals up to but excluding the last shot.
    """
    times: list[float] = []
    running = 0.0
    for _path, _start, seconds in shots[:-1]:
        running += max(0.2, float(seconds))
        times.append(running)
    return times


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a click track.")
    parser.add_argument("out", help="output .wav")
    parser.add_argument("--every", type=float, default=1.5,
                        help="seconds between clicks (default 1.5)")
    parser.add_argument("--length", type=float, default=12.0)
    args = parser.parse_args()

    times = []
    t = args.every
    while t < args.length:
        times.append(t)
        t += args.every
    path = build_click_track(times, args.length, args.out)
    print(f"{len(times)} clicks -> {path}")
