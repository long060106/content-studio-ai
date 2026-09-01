"""
broll_trimmer.py

Finds b-roll clips that contain more than one scene and trims them to one.

    python broll_trimmer.py --dry-run
    python broll_trimmer.py

**The bug this fixes.** A clip is supposed to be a single unbroken shot. Scene
detection at extraction used a threshold of 0.30, which is right for hard cuts
in a feature film and misses softer ones — a dissolve, a whip pan, a cut between
two similarly lit frames. When a boundary is missed the clip runs straight
through it and carries a fragment of the next shot.

The fragments are short, which is exactly why they read as a fault rather than
as an edit: `aton-02-person-swimming-water-closeup.mp4` is 1.71s long with a cut
at 1.63s, so the last two or three frames are a different scene entirely. On
screen it looks like the render flashed, and no amount of work on the shot plan
would have fixed it, because the flash was inside the material.

**Why trim rather than re-cut from the source.** The films are no longer on
disk for most of these, and the fragment is nearly always at one end. Keeping
the longest clean run between detected boundaries preserves the shot that was
wanted and drops only the part that never belonged.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess

# Sensitive on purpose. Extraction used 0.30 and that is what let these through;
# the job here is to catch what that missed, so it errs toward finding cuts.
SCENE_THRESHOLD = 0.22

# A boundary this close to an end is the detector's own first-frame artefact
# rather than a real cut. Small on purpose: at 0.05 it also swallowed genuine
# fragments, and a two-frame flash at the end of a clip is exactly the fault
# being hunted — the first pass "fixed" a clip while leaving 0.042s of the next
# scene on it, then reported the library clean.
EDGE = 0.015

# Cut back this far from a detected boundary rather than landing on it.
#
# Two reasons, and either alone is enough. The boundary is reported at the
# first frame of the *new* scene, so landing exactly on it keeps that frame.
# And the encoder does not stop precisely where asked: -t 1.6266 produced a
# 1.6683s file. A tenth of a second of the wanted shot is a cheap price for
# never showing a frame of the wrong one.
SAFETY = 0.10

# Below this a clip is not worth keeping once trimmed.
MIN_KEEP = 1.2


def duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def internal_cuts(path: str) -> list[float]:
    """Scene boundaries strictly inside the clip."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i", path,
         "-filter:v", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True)
    found = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr or "")]
    total = duration(path)
    return [t for t in found if EDGE < t < total - EDGE]


def longest_run(cuts: list[float], total: float) -> tuple[float, float]:
    """The longest stretch between boundaries — the shot actually wanted."""
    marks = [0.0] + sorted(cuts) + [total]
    # Seeded with the first gap, not with the whole clip. Seeding it with
    # (0, total) means the candidate to beat is the entire clip including every
    # cut in it, which nothing can, so the trim silently did nothing at all.
    best = (marks[0], marks[1])
    for a, b in zip(marks, marks[1:]):
        if b - a > best[1] - best[0]:
            best = (a, b)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Trim multi-scene b-roll to one scene.")
    ap.add_argument("folder", nargs="?",
                    default=os.path.join("assets", "broll", "film"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    clips = [os.path.join(r, f)
             for r, _d, fs in os.walk(args.folder)
             for f in sorted(fs) if f.lower().endswith(".mp4")]
    if args.limit:
        clips = clips[: args.limit]

    trimmed = dropped = clean = 0
    for path in clips:
        total = duration(path)
        if total <= 0:
            continue
        cuts = internal_cuts(path)
        if not cuts:
            clean += 1
            continue

        a, b = longest_run(cuts, total)
        # Step inside the run at both ends, but only where there is a boundary
        # to step away from — the clip's own start and end need no margin.
        if a > 0:
            a += SAFETY
        if b < total:
            b -= SAFETY
        keep = b - a
        name = os.path.basename(path)

        if keep < MIN_KEEP:
            print(f"  drop  {name[:56]}  (best run only {keep:.2f}s)")
            dropped += 1
            if not args.dry_run:
                os.remove(path)
            continue

        print(f"  trim  {name[:56]}  {total:.2f}s -> {keep:.2f}s "
              f"({len(cuts)} internal cut(s))")
        trimmed += 1
        if args.dry_run:
            continue

        tmp = path + ".trim.mp4"
        r = subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-ss", f"{a:.3f}", "-t", f"{keep:.3f}", "-i", path,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
             "-pix_fmt", "yuv420p", "-an", tmp],
            capture_output=True)
        if r.returncode == 0 and os.path.getsize(tmp) > 1000:
            os.replace(tmp, path)
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"    ! could not trim {name[:48]}, left as is")

    print(f"\n{clean} single-scene, {trimmed} trimmed, {dropped} dropped")


if __name__ == "__main__":
    main()
