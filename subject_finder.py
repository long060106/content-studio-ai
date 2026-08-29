"""
subject_finder.py

Finds where the subject sits across a shot, so a portrait crop can follow them
instead of assuming they stand in the middle.

    python subject_finder.py clip.mp4 --start 3 --seconds 2

**Why this exists.** Filling a 1000x1776 window from 1920x1080 footage keeps
about a third of the width, and until now the crop took that third from the
horizontal centre. Whether the speaker survived was luck: measured on a real
clip, two frames out of two landed on the cage wall and a sponsor board while
the fighter — whose voice is playing — stood outside the frame. Cropping to the
centre is not a neutral default, it is a guess that is wrong whenever the
subject is not centred, which for a filmed interview is most of the time.

**How it decides.** Two cheap signals, combined:

- **Motion.** A person talking moves and a room does not. Frame-to-frame
  difference, summed per column, is the strongest available signal for a
  talking-head shot and costs nothing to compute.
- **Detail.** Local variance per column. A face and a body carry structure;
  a wall, sky or seamless backdrop does not. This carries the shot when the
  subject happens to be still.

Neither is face detection, and that is deliberate. OpenCV would be more precise
and it is another native dependency, and Windows Smart App Control has already
blocked three of those in this project without warning. ffmpeg has never failed
here, and for the question actually being asked — which third of the frame is
worth keeping — motion plus detail is enough.

The result is one number per shot rather than per frame, so the crop is fixed
for the length of the shot. A crop that tracked continuously would drift and
wobble, which reads as a camera move nobody made.
"""

from __future__ import annotations

import argparse
import subprocess

# Analysis resolution. Small on purpose: the question is which *third* of the
# frame holds the subject, and 64 columns answers that to within a few pixels
# of the source while keeping the whole thing to one ffmpeg call.
COLS, ROWS = 64, 36

# How many frames to look at. Motion needs at least two; more is steadier on a
# shot where the subject pauses.
FRAMES = 6

# Motion counts for more than detail. A bright, busy background can out-score a
# person on detail alone, but it does not move the way a person does.
MOTION_WEIGHT = 2.4
DETAIL_WEIGHT = 1.0

# Blend the measurement toward centre. At 1.0 the crop chases whatever moved,
# which on a cutaway with a passing car looks like a mistake; at 0 it is the old
# fixed centre. This keeps most of the correction while staying anchored.
CONFIDENCE = 0.85

# How close a window has to be to the best one to count as a tie.
#
# Everything inside this band is treated as equally good, and the tie is broken
# toward the centre of frame. Too small and the bistable flip comes back; too
# large and every shot resolves to centre and the whole module does nothing.
#
# 0.05 is measured, not guessed. Against clips built with the subject at a known
# position — far left, centre, far right — the detector reports 0.248 / 0.500 /
# 0.739 against an ideal of 0.21 / 0.50 / 0.79, while a real interview clip stays
# put as the window width is varied. Below 0.05 the real clip starts swinging a
# third of a frame on a one per cent change; above it, separation begins to
# collapse back toward centre.
TIE_TOLERANCE = 0.05


def _sample(path: str, start: float, seconds: float) -> list[list[int]] | None:
    """`FRAMES` greyscale frames from the shot, as flat rows of pixel values.

    One ffmpeg call rather than one per frame: the cost here is process
    spawning, not decoding, and a render analyses every shot in every clip.
    """
    span = max(0.4, float(seconds))
    rate = max(1.0, FRAMES / span)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-ss", f"{max(0.0, start):.2f}", "-t", f"{span:.2f}", "-i", path,
         "-vf", f"fps={rate:.2f},scale={COLS}:{ROWS},format=gray",
         "-frames:v", str(FRAMES), "-f", "rawvideo", "-"],
        capture_output=True)
    data = r.stdout
    size = COLS * ROWS
    if len(data) < size:
        return None
    return [list(data[i * size:(i + 1) * size])
            for i in range(len(data) // size)]


def _column_scores(frames: list[list[int]]) -> list[float]:
    """How interesting each column is, left to right."""
    motion = [0.0] * COLS
    detail = [0.0] * COLS

    for f in range(len(frames)):
        frame = frames[f]
        prev = frames[f - 1] if f else None
        for c in range(COLS):
            column = [frame[r * COLS + c] for r in range(ROWS)]
            mean = sum(column) / ROWS
            detail[c] += sum((v - mean) ** 2 for v in column) / ROWS
            if prev is not None:
                motion[c] += sum(
                    abs(frame[r * COLS + c] - prev[r * COLS + c])
                    for r in range(ROWS)
                )

    def normalise(xs):
        top = max(xs) or 1.0
        return [x / top for x in xs]

    m, d = normalise(motion), normalise(detail)
    return [MOTION_WEIGHT * m[i] + DETAIL_WEIGHT * d[i] for i in range(COLS)]


def horizontal_focus(path: str, start: float = 0.0, seconds: float = 2.0,
                     keep: float = 0.33) -> float:
    """Where to centre a crop that keeps `keep` of the width. 0.0 left, 1.0 right.

    `keep` is the fraction of the frame's width the crop will retain, so the
    window being scored is the same width as the window that will actually be
    taken. Scoring a different width would find the wrong place.

    Returns 0.5 — the old centred behaviour — when the shot cannot be read.
    """
    frames = _sample(path, start, seconds)
    if not frames or len(frames) < 2:
        return 0.5

    scores = _smooth(_column_scores(frames))
    width = max(1, min(COLS, round(COLS * max(0.05, min(1.0, keep)))))
    if width >= COLS:
        return 0.5

    # Total interest in every candidate window, by sliding sum.
    totals = []
    running = sum(scores[:width])
    totals.append(running)
    for i in range(1, COLS - width + 1):
        running += scores[i + width - 1] - scores[i - 1]
        totals.append(running)

    # Take the best window — but *not* by bare argmax.
    #
    # A shot often has two places worth looking at, the speaker and something
    # bright behind them, and their totals can land within a hair of each other.
    # Bare argmax then flips between them on the strength of nothing: measured
    # here, changing the window width by one per cent moved the answer from
    # 0.487 to 0.693, a third of the frame away. Deterministic, and unstable —
    # which is worse than a fixed centre, because it frames the same speaker
    # differently from shot to shot.
    #
    # So: among all windows that are nearly as good as the best, prefer the one
    # closest to the middle. Strong evidence still moves the crop; weak or
    # divided evidence settles quietly back to centre instead of picking a side.
    best = max(totals)
    if best <= 0:
        return 0.5
    floor = best * (1.0 - TIE_TOLERANCE)
    contenders = [i for i, t in enumerate(totals) if t >= floor]
    best_at = min(contenders, key=lambda i: abs((i + width / 2) / COLS - 0.5))

    centre = (best_at + width / 2) / COLS
    return 0.5 + (centre - 0.5) * CONFIDENCE


def _smooth(scores: list[float], radius: int = 3) -> list[float]:
    """Moving average, so one bright column cannot decide the whole crop."""
    out = []
    for i in range(len(scores)):
        lo, hi = max(0, i - radius), min(len(scores), i + radius + 1)
        out.append(sum(scores[lo:hi]) / (hi - lo))
    return out


def crop_offset(source_w: int, source_h: int, out_w: int, out_h: int,
                focus: float) -> int:
    """Left edge, in scaled pixels, of a crop centred on `focus`.

    Mirrors what `scale=...:force_original_aspect_ratio=increase` does — both
    dimensions are scaled by whichever factor is larger — so the number handed
    to ffmpeg is computed rather than expressed, and there are no filtergraph
    escaping rules to get wrong.
    """
    if source_w <= 0 or source_h <= 0:
        return max(0, (source_w - out_w) // 2)
    scale = max(out_w / source_w, out_h / source_h)
    scaled_w = source_w * scale
    x = focus * scaled_w - out_w / 2
    return int(max(0, min(scaled_w - out_w, x)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the subject's column.")
    parser.add_argument("clip")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--keep", type=float, default=0.33)
    args = parser.parse_args()

    f = horizontal_focus(args.clip, args.start, args.seconds, args.keep)
    where = "left" if f < 0.42 else "right" if f > 0.58 else "centre"
    print(f"focus {f:.3f}  ({where})")
