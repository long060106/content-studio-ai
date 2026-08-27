"""
scene_extractor.py

Cuts a whole film into individual shots, keeps the ones worth keeping, and
files them as b-roll.

    python scene_extractor.py "D:/films/kingdom-of-heaven.mkv" --category epic
    python scene_extractor.py movie.mkv --category resolve --count 30
    python scene_extractor.py movie.mkv --dry-run      # just report what it finds

The problem this solves is the one that makes a film library expensive: a
two-hour film holds perhaps two thousand shots, a few dozen of which are worth
cutting to, and finding them by scrubbing a timeline is a day's work per film.
Shot boundaries are not a matter of taste, though — they are a measurable
discontinuity between consecutive frames, so a machine can find every one of
them and leave only the judgement to a person.

**Why ffmpeg rather than a scene-detection library.** PySceneDetect is the
obvious tool and it is good. It is also another native dependency, and on this
machine Windows Smart App Control has already blocked two of those without
warning — numba, which killed transcription once, and PyAV, which killed it
again months later. ffmpeg is the one component that has never failed here.
Its `select='gt(scene,N)'` filter reports the same boundaries.

**Why not a website.** They exist — Kapwing and Veed both do scene splitting —
but a feature film is several gigabytes, and uploading that to cut it into
pieces you then download again is slower than decoding it locally, costs a
subscription, and puts the film on someone else's server.

The scoring is deliberately crude and is only a sort order. It cannot tell a
beautiful shot from a dull one; it can tell a black frame from a picture, and
a flat evenly-lit shot from one with depth. The last step is still a person
looking at a contact sheet.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess

# A shot shorter than this is a flash — usually a cut inside an action beat,
# and useless as a cutaway. Longer than the upper bound and it is a scene
# rather than a shot, which the pipeline would only trim again anyway.
MIN_SHOT = 1.6
MAX_SHOT = 12.0

# How different two consecutive frames must be to count as a cut. Lower finds
# more boundaries including false ones inside camera moves; higher misses soft
# cuts and dissolves. 0.3 is a reasonable middle for film.
SCENE_THRESHOLD = 0.30

# Detection decodes the whole film, so it runs on a downscaled copy. Shot
# boundaries are a whole-frame property and survive the downscale intact.
DETECT_WIDTH = 320


def probe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def detect_cuts(path: str, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Every shot boundary in the file, in seconds.

    Decodes once at low resolution and reads the timestamps ffmpeg prints for
    frames that differ enough from their predecessor.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info",
        "-i", path,
        "-filter:v", f"scale={DETECT_WIDTH}:-2,select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except (OSError, subprocess.SubprocessError):
        return []
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr or "")]


def shots_from_cuts(cuts: list[float], duration: float,
                    min_shot: float = MIN_SHOT,
                    max_shot: float = MAX_SHOT) -> list[tuple[float, float]]:
    """Turn boundaries into usable shots, dropping the too-short and too-long.

    A long gap between boundaries is not necessarily one shot — it is often a
    static dialogue take the detector correctly saw no cut in — so those are
    trimmed to `max_shot` from their start rather than discarded.
    """
    marks = [0.0] + sorted(cuts) + [duration]
    shots: list[tuple[float, float]] = []
    for a, b in zip(marks, marks[1:]):
        span = b - a
        if span < min_shot:
            continue
        # Start a beat after the cut: the first frames of a shot often carry
        # the tail of a dissolve.
        start = a + 0.25
        shots.append((start, min(start + max_shot, b)))
    return shots


def score_shot(path: str, start: float) -> dict | None:
    """Cheap visual stats for one shot, used only to sort candidates."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.2f}", "-i", path, "-vframes", "1",
         "-vf", "scale=48:27,format=rgb24", "-f", "rawvideo", "-"],
        capture_output=True)
    d = r.stdout
    if len(d) < 48 * 27 * 3:
        return None
    px = [(d[i], d[i + 1], d[i + 2]) for i in range(0, 48 * 27 * 3, 3)]
    lum = [(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) / 255 for p in px]
    mean = sum(lum) / len(lum)
    contrast = math.sqrt(sum((x - mean) ** 2 for x in lum) / len(lum))
    sat = sum((max(p) - min(p)) / 255 for p in px) / len(px)
    return {"bright": mean, "contrast": contrast, "sat": sat}


def keep_score(stats: dict) -> float:
    """Higher is more worth keeping.

    Depth is what is being rewarded — contrast carries most of the weight.
    Near-black frames (credits, fades) and blown-out ones are pushed down
    hard, because both are common in a film and useless as b-roll.
    """
    if stats["bright"] < 0.045 or stats["bright"] > 0.93:
        return -10.0
    return stats["contrast"] * 3.0 - abs(stats["bright"] - 0.38) * 1.4 - stats["sat"] * 0.5


def extract(path: str, out_dir: str, count: int, prefix: str,
            threshold: float = SCENE_THRESHOLD, dry_run: bool = False) -> int:
    duration = probe_duration(path)
    if duration <= 0:
        print(f"Could not read {path}")
        return 0

    print(f"  {os.path.basename(path)}: {duration/60:.0f} min — detecting shots...")
    cuts = detect_cuts(path, threshold)
    shots = shots_from_cuts(cuts, duration)
    print(f"  {len(cuts)} boundaries, {len(shots)} usable shots")
    if not shots:
        return 0

    scored = []
    for start, end in shots:
        s = score_shot(path, start + (end - start) / 2)
        if s:
            scored.append((keep_score(s), start, end))
    scored.sort(reverse=True)
    chosen = scored[:count]
    print(f"  keeping the best {len(chosen)}")

    if dry_run:
        for sc, start, end in chosen[:15]:
            print(f"    {sc:5.2f}  {start/60:5.1f} min  {end-start:4.1f}s")
        return len(chosen)

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for i, (sc, start, end) in enumerate(chosen, 1):
        dest = os.path.join(out_dir, f"{prefix}-{i:02d}.mp4")
        cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start:.2f}", "-t", f"{end-start:.2f}", "-i", path,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
               "-pix_fmt", "yuv420p", "-an", dest]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            written += 1
    print(f"  wrote {written} clips to {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut a film into shots and keep the best as b-roll.")
    parser.add_argument("movie", help="path to the film file")
    parser.add_argument("--category", default="epic",
                        help="library folder to file under (default: epic)")
    parser.add_argument("--count", type=int, default=25,
                        help="how many shots to keep (default 25)")
    parser.add_argument("--prefix", default=None,
                        help="filename stem; defaults to the film's filename")
    parser.add_argument("--threshold", type=float, default=SCENE_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what it would keep without writing")
    args = parser.parse_args()

    if not os.path.isfile(args.movie):
        raise SystemExit(f"No such file: {args.movie}")

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "broll", "film", args.category)
    prefix = args.prefix or re.sub(
        r"[^a-z0-9]+", "-", os.path.splitext(os.path.basename(args.movie))[0].lower()
    ).strip("-")[:40]

    n = extract(args.movie, base, args.count, prefix,
                threshold=args.threshold, dry_run=args.dry_run)
    if n and not args.dry_run:
        print("\nLook at them before trusting them — the score sorts, it does not judge.")


if __name__ == "__main__":
    main()
