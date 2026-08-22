"""
heatmap.py

Reads YouTube's "most replayed" graph so clips can be cut where the audience
actually rewatched, instead of where a model guessed from reading a transcript.

YouTube exposes a 100-point retention curve for videos with enough views, and
yt-dlp already surfaces it as `info["heatmap"]` — each entry a `start_time`,
`end_time` and a normalised `value`. No scraping, and it's a dependency the
project already has.

**The opening spike is not a signal.** Every video's curve is highest in its
first few seconds, because viewers start there and drift away — not because
that content is good. On the Lara Boyd talk the 0s point scores 0.87, and the
transcript at that timestamp reads "Transcriber: Jessica Lee / Reviewer: Denise
RQ". Cutting the hottest raw window would produce the credits, every time.

So peaks are found by **local prominence** — how far a point rises above its own
neighbourhood — rather than by absolute height, and the intro is excluded
outright. On the same talk that correctly surfaces the 285s peak, where she
explains that people who read Braille have larger hand sensory areas: a genuine
"wait, what?" moment that 45 million viewers went back for.

Not every video has a heatmap; below roughly a few hundred thousand views
YouTube doesn't publish one. `fetch()` returns None in that case and callers
fall back to reading the transcript alone.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Optional

# Retention is dominated by drop-off for the opening stretch, so nothing before
# this is trusted as a content signal. Whichever is longer: a flat 30 seconds,
# or the first 5% of the video.
INTRO_SECONDS = 30.0
INTRO_FRACTION = 0.05

# How far either side of a point to measure its local baseline from.
NEIGHBOURHOOD = 3


@dataclass
class HotWindow:
    start: float
    end: float
    value: float           # raw normalised intensity, 0-1
    prominence: float      # how far it rises above its neighbourhood
    rank: int = 0

    @property
    def centre(self) -> float:
        return (self.start + self.end) / 2

    def to_dict(self) -> dict:
        return asdict(self)


def fetch(url: str, info: Optional[dict] = None) -> Optional[list[dict]]:
    """The raw heatmap for a video, or None when YouTube doesn't publish one.

    Pass an existing yt-dlp `info` dict to avoid a second metadata call.
    """
    if info is None:
        import yt_dlp

        from youtube_extractor import yt_dlp_base_opts

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            **yt_dlp_base_opts(),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception:
            return None

    heatmap = info.get("heatmap")
    if not heatmap:
        return None

    cleaned = []
    for point in heatmap:
        try:
            cleaned.append({
                "start": float(point["start_time"]),
                "end": float(point["end_time"]),
                "value": float(point["value"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return cleaned or None


def find_hot_windows(
    heatmap: list[dict],
    duration: float = 0.0,
    count: int = 8,
    pad: float = 6.0,
) -> list[HotWindow]:
    """The moments the audience genuinely went back for, best first.

    Ranked by local prominence rather than raw height, so a slow decline from
    the intro doesn't outrank a real spike in the middle of the talk.
    """
    if not heatmap:
        return []

    points = sorted(heatmap, key=lambda p: p["start"])
    duration = duration or points[-1]["end"]
    intro_cutoff = max(INTRO_SECONDS, duration * INTRO_FRACTION)

    values = [p["value"] for p in points]
    baseline = statistics.median(values) if values else 0.0

    candidates: list[HotWindow] = []
    for i, point in enumerate(points):
        if point["start"] < intro_cutoff:
            continue

        lo = max(0, i - NEIGHBOURHOOD)
        hi = min(len(points), i + NEIGHBOURHOOD + 1)
        neighbours = [p["value"] for j, p in enumerate(points[lo:hi], start=lo) if j != i]
        if not neighbours:
            continue

        local = statistics.median(neighbours)
        prominence = point["value"] - local

        # A peak has to beat both its immediate surroundings and the video's
        # overall level — otherwise a flat, mildly-noisy stretch produces
        # dozens of meaningless "peaks".
        if prominence <= 0 or point["value"] <= baseline:
            continue

        candidates.append(HotWindow(
            start=max(0.0, point["start"] - pad),
            end=point["end"] + pad,
            value=point["value"],
            prominence=prominence,
        ))

    candidates.sort(key=lambda w: w.prominence, reverse=True)

    # Collapse peaks that sit on top of each other — a broad hot region often
    # spans several points and shouldn't occupy the whole shortlist.
    merged: list[HotWindow] = []
    for window in candidates:
        if any(abs(window.centre - kept.centre) < 20.0 for kept in merged):
            continue
        merged.append(window)
        if len(merged) >= count:
            break

    for rank, window in enumerate(merged, start=1):
        window.rank = rank
    return merged


def describe(windows: list[HotWindow], segments: list[dict]) -> list[dict]:
    """Pair each hot window with what's actually being said there."""
    described = []
    for window in windows:
        said = " ".join(
            s["text"] for s in segments
            if window.start <= float(s["start"]) <= window.end
        ).strip()
        described.append({
            "rank": window.rank,
            "start": round(window.start, 1),
            "end": round(window.end, 1),
            "value": round(window.value, 3),
            "prominence": round(window.prominence, 3),
            "text": said,
        })
    return described


def for_video(
    url: str,
    segments: Optional[list[dict]] = None,
    count: int = 8,
    info: Optional[dict] = None,
) -> Optional[list[dict]]:
    """Hot windows with their transcript text, or None if there's no heatmap."""
    raw = fetch(url, info=info)
    if not raw:
        return None
    windows = find_hot_windows(raw, count=count)
    if not windows:
        return None
    return describe(windows, segments or [])


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Show a video's most-replayed moments.")
    parser.add_argument("url")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--transcript", help="transcript.json, to show what's said there")
    args = parser.parse_args()

    segments = []
    if args.transcript and os.path.isfile(args.transcript):
        with open(args.transcript, "r", encoding="utf-8") as f:
            segments = json.load(f).get("transcript_segments", [])

    hot = for_video(args.url, segments=segments, count=args.count)
    if not hot:
        print("No heatmap for this video (YouTube only publishes one above a view threshold).")
        raise SystemExit(0)

    for window in hot:
        print(f"\n#{window['rank']}  {window['start']:.0f}s-{window['end']:.0f}s"
              f"  intensity {window['value']:.2f}  prominence {window['prominence']:+.2f}")
        if window["text"]:
            print(f"    {window['text'][:300]}")
