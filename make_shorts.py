"""
make_shorts.py

One talk in, a batch of motivational shorts out.

    python make_shorts.py "https://www.youtube.com/watch?v=VIDEO_ID"

What happens:

  1. Transcript      Reuses output/<video_id>/transcript.json if it's already
                     there, otherwise pulls YouTube's captions. The full talk
                     is never sent through Whisper — that's the slow, expensive
                     way to do this, and the captions are good enough to *find*
                     moments with.
  2. Moments         One Claude call works through the replay peaks and returns
                     one moment per peak that stands alone — however many that
                     is, up to 8 — each expanded outwards from the peak to the
                     complete statement, with a hook and visual search terms.
  3. Clips           Only those windows are downloaded, via the same ranged
                     yt-dlp path the main pipeline uses.
  4. Brief           brief.md next to each clip: where it came from, what is
                     said, why it was chosen, and what must not be trimmed.
                     The clip alone doesn't carry that, and an editor without
                     it cuts the setup off the front.
  5. Assets          B-roll comes from the local library, topped up from
                     stock APIs when a key is configured. No music — see
                     pick_assets for why.
  6. Render          ffmpeg assembles 1080x1920. Captions are off by default:
                     these clips get re-edited, and burned-in text fights that.
                     Pass --captions to burn them in anyway.
  7. Carousel        Optional: the card copy as text, for designing by hand.

Everything lands in output/<video_id>/shorts/, one folder per short, and the
downloaded clip is kept so you can re-render in a different style without
touching the network again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading

import asset_library
from caption_timing import captions_for_clip
from moment_finder import MAX_TOTAL_SECONDS, Moment, find_moments
from shorts_builder import (
    ShortSpec,
    build_rough_cut,
    build_short,
    join_clips,
    probe_duration,
)


# How many clips are built at once. Three is a deliberate floor rather than
# the core count: each worker keeps its own Whisper model in memory, and the
# ffmpeg encodes already use the GPU, so more lanes buy contention rather than
# speed. Override with --workers.
DEFAULT_WORKERS = 3

# Squeeze the long silences out of each cut before rendering.
#
# On the test talk this took a 37.7s clip to 27.8s, and almost all of it came
# from three holes — 4.9s, 2.4s and 1.5s — rather than from the speaker's
# natural rhythm. Short pauses are left alone deliberately, and the breath at
# the end that `ending_finder` works to create is never touched.
TIGHTEN_SILENCE = True


def _slug(text: str, limit: int = 28) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (out[:limit].rstrip("_") or "moment")


def load_or_fetch_transcript(url: str, output_dir: str) -> dict:
    """Prefer the transcript we already have; only hit the network if we must."""
    from youtube_extractor import extract, extract_video_id

    video_id = extract_video_id(url)
    path = os.path.join(output_dir, video_id, "transcript.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("transcript_segments"):
            print(f"  ✓ Reusing existing transcript ({len(data['transcript_segments'])} segments)")
            return data

    print("  · No local transcript, pulling captions...")
    video = extract(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(video.to_json())
    return json.loads(video.to_json())


def pick_assets(moment: Moment, fetch: bool) -> tuple[str | None, dict]:
    """Choose b-roll for one moment, fetching stock if allowed.

    No music. These shorts are published without a bed: they're deliberately
    short, so a publisher typically stitches two or three together into one
    post and lays music over that finished sequence. Music baked into each
    short would only have to be cut apart and re-matched at that point.
    """
    keywords = list(moment.visual_keywords) + [moment.theme]
    credits: dict = {}

    broll = asset_library.pick("video", keywords, min_duration=moment.duration)
    if broll is None and fetch and moment.visual_keywords:
        query = moment.visual_keywords[0]
        print(f"    · library has no matching b-roll, fetching \"{query}\"...")
        got = asset_library.fetch_stock(query, kind="video", count=2)
        if got:
            broll = asset_library.pick("video", keywords, min_duration=moment.duration)

    if broll:
        credits["broll"] = {"path": broll.path, "source": broll.source,
                            "credit": broll.credit, "licence": broll.licence}
    return (broll.abs_path if broll else None, credits)


def _remove_tree(path: str) -> bool:
    """Delete a folder, clearing read-only flags that would otherwise stop it.

    OneDrive marks synced folders ReadOnly with a reparse point, and plain
    rmtree fails on those with "Access is denied" even when nothing holds the
    files open. Clearing the attribute and retrying is what Explorer does.
    """
    import shutil
    import stat

    def on_error(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=on_error)
    return not os.path.exists(path)


def _clear_stale_clips(shorts_dir: str, keep: set[str]) -> list[str]:
    """Remove clip folders left behind by an earlier run of the same talk.

    Folder names are built from the hook, and the hook is written fresh by the
    model each run, so re-cutting a talk leaves the previous folders sitting
    beside the new ones — same clip, different name, its own copy of the video.
    The library then shows one clip twice.

    Deliberately called *after* the clips are built rather than before. Clearing
    up front would mean a run that fails halfway destroys the clips you already
    had, which is the wrong direction to fail in. Nothing here runs unless the
    run actually produced something.
    """
    removed: list[str] = []
    if not os.path.isdir(shorts_dir):
        return removed

    for name in sorted(os.listdir(shorts_dir)):
        folder = os.path.join(shorts_dir, name)
        if not os.path.isdir(folder) or os.path.abspath(folder) in keep:
            continue
        # Only ever touch numbered clip folders: never `carousel/`, and never
        # anything a future version of this script might put here.
        if not re.fullmatch(r"\d+_.*", name):
            continue
        if _remove_tree(folder):
            removed.append(name)
    return removed


def _spoken_text(moment: Moment, segments: list[dict]) -> str:
    """The words actually said inside this moment's cuts.

    Taken from the transcript rather than from Whisper, so the brief is
    available whether or not captions were timed.
    """
    parts: list[str] = []
    for cut in moment.cuts:
        said = " ".join(
            s["text"] for s in segments
            if cut.start_seconds <= float(s["start"]) < cut.end_seconds
        ).strip()
        if said:
            parts.append(said)
    return "\n\n[hard cut]\n\n".join(parts)


# The cutting rhythm. Short holds on the speaker, one or two quick b-roll
# shots, then back to the speaker — the pattern the reference videos use.
#
# It always opens on the speaker. Establishing who is talking before cutting
# away is what stops the clip reading as a stock-footage montage with a voice
# over it; once the face is established, the b-roll illustrates rather than
# replaces.
SPEAKER_HOLD_SECONDS = 2.0
BROLL_SHOT_SECONDS = 1.5
BROLL_RUN = 2          # b-roll shots before cutting back to the speaker
# One clip per cutaway, never reused inside a short. A repeat is noticeable:
# the same footage returning twenty seconds later reads as running out of
# material rather than as a deliberate callback. The ceiling exists only to
# stop a pathological case downloading a hundred files.
MAX_DISTINCT_BROLL = 24


def _text_between(words: list, start: float, end: float) -> str:
    """What is being said across a span, for the shot list."""
    said = [w.text for w in words if w.start < end and w.end > start]
    return " ".join(said).strip()


def _shot_plan(
    total: float,
    speech_path: str,
    broll_paths: list[str],
) -> list[tuple[str, float, float, str]]:
    """Alternate speaker and b-roll across the clip.

    Returns (path, source_start, seconds, kind) in playing order.

    Speaker shots carry their position in the clip as `source_start`, because
    cutting back to the speaker only works if their picture matches the audio
    being heard at that moment.

    With no b-roll available this degrades to a single continuous speaker shot,
    which is exactly the old behaviour.
    """
    plan: list[tuple[str, float, float, str]] = []
    t = 0.0
    index = 0
    # Anything shorter than this isn't a shot, it's a flicker.
    epsilon = 0.2

    while t < total - epsilon:
        span = min(SPEAKER_HOLD_SECONDS, total - t)
        plan.append((speech_path, t, span, "original"))
        t += span

        if not broll_paths:
            continue
        for _ in range(BROLL_RUN):
            if t >= total - epsilon:
                break
            span = min(BROLL_SHOT_SECONDS, total - t)
            plan.append((broll_paths[index % len(broll_paths)], 0.0, span, "b-roll"))
            index += 1
            t += span

    if not plan:
        plan = [(speech_path, 0.0, total, "original")]
    return plan


def _broll_slots(total: float) -> int:
    """How many distinct b-roll clips a short of this length needs.

    Counted from the plan itself rather than estimated, so the fetch asks for
    exactly as many as there are cutaways and nothing has to repeat.
    """
    plan = _shot_plan(total, "SPEECH", ["placeholder"])
    cutaways = sum(1 for _p, _s, _d, kind in plan if kind == "b-roll")
    return min(max(1, cutaways), MAX_DISTINCT_BROLL)


def _shotlist(
    plan: list[tuple[str, float, float, str]],
    words: list,
    hook: str,
) -> str:
    """The edit, written out: what is on screen when, and what is being said.

    This is what makes the folder assemblable rather than a pile of files, and
    what lets one shot be swapped without rebuilding the edit.
    """
    lines: list[str] = [f"# Shot list - {hook}", ""]
    lines.append("The speaker's audio runs underneath the whole way; only the")
    lines.append("picture changes. Swap any b-roll shot you don't like and keep")
    lines.append("the timings.")
    lines.append("")

    t = 0.0
    for path, _source_start, seconds, kind in plan:
        end = t + seconds
        said = _text_between(words, t, end)
        label = "ORIGINAL (the speaker)" if kind == "original" else f"broll/{os.path.basename(path)}"
        lines.append(f"### {t:.1f}s - {end:.1f}s   ({seconds:.1f}s)")
        lines.append("")
        lines.append(f"**On screen:** {label}")
        if said:
            lines.append("")
            lines.append(f"> {said}")
        lines.append("")
        t = end

    return "\n".join(lines)


def _clip_brief(
    moment: Moment,
    index: int,
    url: str,
    segments: list[dict],
    talk_title: str,
    channel: str,
    tags=None,
) -> str:
    """An editing brief for whoever cuts the final video.

    The clip on its own doesn't say why it was chosen, where it sits in the
    talk, or what would break if it were trimmed further. An editor working
    without that will cut the setup off the front — which is exactly the
    mistake the peak-expansion logic exists to avoid — so the reasoning ships
    alongside the file.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# Clip {index:02d} — {moment.hook}")
    add("")
    if talk_title:
        add(f"**Source:** {talk_title}" + (f" ({channel})" if channel else ""))
    add(f"**Watch the original:** {url}&t={int(moment.cuts[0].start_seconds)}s")
    add(f"**Length:** {moment.duration:.1f}s across {len(moment.cuts)} cut(s)")
    add(f"**Theme:** {moment.theme}   **Tone:** {moment.tone}")
    if moment.peak_rank:
        add(f"**Replay peak #{moment.peak_rank}** — this is a stretch the audience "
            f"rewatched, not a passage picked from reading the transcript.")
    add("")

    add("## Where it comes from")
    add("")
    for n, cut in enumerate(moment.cuts, 1):
        add(f"- Cut {n}: `{cut.start_seconds:.1f}s` to `{cut.end_seconds:.1f}s` "
            f"({cut.duration:.1f}s) — {url}&t={int(cut.start_seconds)}s")
    if moment.stitch_reason:
        add("")
        add(f"**Why these join:** {moment.stitch_reason}")
    add("")

    add("## What is said")
    add("")
    spoken = _spoken_text(moment, segments)
    for para in spoken.split("\n\n"):
        add(f"> {para}" if not para.startswith("[") else para)
    add("")

    add("## Why this one")
    add("")
    add(moment.reason or "—")
    add("")

    add("## Constraints — read before editing")
    add("")
    add("- **Do not trim the opening.** The cut already starts where the thought "
        "begins, not where the strong line lands. Cutting into it leaves the "
        "payoff sounding like a non-sequitur.")
    add("- **Do not trim the ending.** It stops on a finished sentence. Ending "
        "early is the most common way a clip stops feeling complete.")
    if len(moment.cuts) > 1:
        joined = moment.cuts[0].duration
        add(f"- **There is a hard cut at {joined:.1f}s.** The audio jumps to a "
            "different part of the talk. Change the picture on that beat so the "
            "jump reads as an edit rather than a mistake.")
    add("- **Keep the speaker's audio continuous.** It carries the whole clip; "
        "any music sits under it, never over it.")
    add("- Aim to stay under 25s. Past that, completion rate falls and the "
        "platform stops pushing it.")
    add("")

    add("## Footage to cut over the top")
    add("")
    add("The speaker's voice runs underneath; the picture is yours to choose.")
    add("")
    queries = list(moment.visual_keywords)
    if tags is not None:
        for q in getattr(tags, "visual_queries", []):
            if q not in queries:
                queries.append(q)
    for q in queries:
        add(f"- {q}")
    add("")

    if tags is not None and getattr(tags, "hashtags", None):
        add("## Hashtags")
        add("")
        add(" ".join(f"#{t}" for t in tags.hashtags))
        add("")

    return "\n".join(lines)


def _publishing_set(data: dict, video_dir: str) -> list[str]:
    """Write the post-everywhere formats for the whole talk.

    The clips are only half the job. A short goes out with a caption on
    TikTok, a description on YouTube, a thread on X and a carousel on
    Instagram, and writing those by hand for every talk is most of the work
    that is left after the video is cut.

    All of it hangs off one `ContentBrief`, which is what every generator in
    this project already expects — so the brief is produced first and the rest
    read from it. That is also why they agree with each other: the blog post
    and the thread are not independent takes on the same video, they are the
    same understanding of it in different shapes.

    The VideoData is rebuilt from transcript.json rather than re-fetched. It
    was written from exactly that object, so the round trip is lossless and
    costs nothing.

    Every step is caught separately: a thread that fails should not take the
    blog post down with it, and whatever did work stays on disk.
    """
    from youtube_extractor import VideoData

    written: list[str] = []

    def save(name: str, text: str) -> None:
        path = os.path.join(video_dir, name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(name)

    import dataclasses

    fields = {f.name for f in dataclasses.fields(VideoData)}
    video = VideoData(**{k: data[k] for k in fields if k in data})
    title = data.get("title", "")

    from content_brief import generate_brief

    print("  · Content brief...")
    brief = generate_brief(video)
    save("brief.json", brief.to_json())

    print("  · Blog post...")
    try:
        from blog_generator import generate_blog_post

        save("blog_post.md", generate_blog_post(brief, video_title=title))
    except Exception as e:
        print(f"    ⚠ blog post failed: {e}")

    print("  · X / Twitter thread...")
    try:
        from twitter_thread_generator import generate_twitter_thread

        thread = generate_twitter_thread(brief)
        save("twitter_thread.json", thread.to_json())
        save("twitter_thread.txt", thread.to_text())
    except Exception as e:
        print(f"    ⚠ thread failed: {e}")

    print("  · Captions and descriptions...")
    try:
        from captions_generator import generate_captions

        captions = generate_captions(brief)
        save("captions.json", captions.to_json())
        save("captions.txt", captions.to_text())
    except Exception as e:
        print(f"    ⚠ captions failed: {e}")

    return written


def build_quote_carousel(
    moments: list[Moment],
    out_dir: str,
    talk_title: str = "",
    channel: str = "",
    fetch: bool = True,
) -> list[str]:
    """Write the carousel as words, not pictures.

    This used to search the asset library for a background photo per card and
    render finished PNGs. It no longer does either: the designing happens by
    hand afterwards, so rendering images here produced work that was thrown
    away, and the image search was one of the slowest steps in a run — several
    seconds per card, sometimes a stock API call.

    What comes out is the copy for each card, ready to paste into whatever
    design tool is being used.
    """
    if not moments:
        return []

    os.makedirs(out_dir, exist_ok=True)

    cards: list[dict] = [{
        "card": 1,
        "role": "hook",
        "headline": moments[0].hook,
        "subtext": talk_title,
        "source": channel,
    }]

    for moment in moments:
        cards.append({
            "card": len(cards) + 1,
            "role": "quote",
            "headline": moment.quote or moment.hook,
            "subtext": moment.hook if moment.quote else "",
            "source": channel,
        })

    cards.append({
        "card": len(cards) + 1,
        "role": "close",
        "headline": "Save this.",
        "subtext": "Then go do the work.",
        "source": channel,
    })

    json_path = os.path.join(out_dir, "carousel.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"cards": cards}, f, indent=2, ensure_ascii=False)

    lines: list[str] = []
    for card in cards:
        lines.append(f"--- Card {card['card']} ({card['role']}) ---")
        lines.append(card["headline"])
        if card["subtext"]:
            lines.append(f"    {card['subtext']}")
        lines.append("")
    text_path = os.path.join(out_dir, "carousel.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return [text_path, json_path]


def make_shorts(
    url: str,
    count: int | None = None,
    style: str = "broll",
    output_dir: str = "output",
    model_size: str = "base",
    fetch: bool = True,
    carousel: bool = True,
    formats: bool = True,
    min_seconds: int | None = None,
    max_seconds: int | None = None,
    source_file: str | None = None,
    workers: int = DEFAULT_WORKERS,
    captions: bool = False,
    keep_old: bool = False,
) -> list[dict]:
    from video_clipper import download_clip, download_full, cut_from_file, YouTubeBlockedDownload

    # Footage already claimed by another clip in this run. Two shorts from the
    # same talk opening on the same stock runner looks like a template, so the
    # set is shared and the lock keeps three workers from claiming one clip.
    used_broll: set = set()
    BROLL_LOCK = threading.Lock()
    from youtube_extractor import extract_video_id

    video_id = extract_video_id(url)
    print(f"→ Building shorts for {video_id}")

    data = load_or_fetch_transcript(url, output_dir)
    segments = data.get("transcript_segments") or []
    if not segments:
        print("✗ This video has no timestamped transcript, so moments can't be located.")
        sys.exit(1)

    # Ask YouTube where the audience actually rewatched. Not every video has a
    # heatmap (it needs a view threshold), so this is best-effort — without it
    # the moments are chosen from the transcript alone, as before.
    hot_windows = None
    try:
        import heatmap as heatmap_module

        hot_windows = heatmap_module.for_video(url, segments=segments, count=8)
        if hot_windows:
            print(f"  ✓ Most-replayed data: {len(hot_windows)} peak(s) to cut around")
            for w in hot_windows[:3]:
                print(f"     {w['start']:.0f}s-{w['end']:.0f}s (intensity {w['value']:.2f})")
        else:
            print("  · No most-replayed data for this video — using the transcript alone")
    except Exception as e:
        print(f"  ⚠ Couldn't read most-replayed data ({e}); using the transcript alone")

    if count is None:
        peaks = len(hot_windows) if hot_windows else 0
        print(
            f"→ Finding moments with Claude ({peaks} replay peak(s) to work through)..."
            if peaks else
            "→ Finding moments with Claude (no replay data — reading the transcript)..."
        )
    else:
        print(f"→ Finding the {count} strongest moments with Claude...")
    from moment_finder import MAX_TOTAL_SECONDS, MIN_TOTAL_SECONDS

    moments = find_moments(
        segments,
        count=count,
        title=data.get("title", ""),
        hot_windows=hot_windows,
        min_total=min_seconds or MIN_TOTAL_SECONDS,
        max_total=max_seconds or MAX_TOTAL_SECONDS,
    )
    if not moments:
        print("✗ No usable moments came back.")
        sys.exit(1)
    print(f"  ✓ {len(moments)} moments found")
    for i, m in enumerate(moments, 1):
        heat = f" · replay {m.heat:.2f}" if m.heat else ""
        shape = f"{len(m.cuts)} cuts" if len(m.cuts) > 1 else "1 cut"
        print(f"     {i}. {m.duration:.0f}s ({shape}) {m.theme}{heat}: {m.hook}")

    asset_library.ensure_dirs()
    asset_library.scan()

    shorts_dir = os.path.join(output_dir, video_id, "shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    results: list[dict] = []

    # Fetch the talk once, then take every cut from the local file.
    #
    # A ranged download costs 20-30s here, of which only ~7s is transfer: the
    # rest is yt-dlp re-extracting the video and re-solving YouTube's JS
    # challenge, paid again per cut. One full download pays that once, and it's
    # cached, so re-running a talk later costs no network at all.
    total_cuts = sum(len(m.cuts) for m in moments)
    # The source is now fetched even for a single cut, because the ending
    # refinement below listens to the audio around each cut and needs the file
    # to do it. It is cached, so a re-run costs nothing either way.
    if source_file is None:
        candidate = os.path.join(output_dir, video_id, "source.mp4")
        cached = os.path.isfile(candidate)
        print(
            f"  ✓ Reusing the downloaded talk"
            if cached else
            f"→ Downloading the talk once, for all {total_cuts} cuts..."
        )
        try:
            source_file = download_full(url, candidate)
            if not cached:
                size = os.path.getsize(source_file) / 1e6
                print(f"  ✓ Got the source ({size:.0f} MB) — cuts are local from here")
        except Exception as e:
            # Not fatal: fall back to the per-cut path, just slower.
            print(f"  ⚠ Couldn't fetch the whole talk ({e}); cutting per clip instead")
            source_file = None

    # Put every ending on the moment the speaker actually finishes.
    #
    # This has to happen here — after the moments are chosen, before anything
    # is cut — because it changes where the cuts are. It is also the first
    # point where both things it needs exist at once: the chosen moments, and
    # the audio to check them against.
    #
    # It runs in sequence rather than inside the parallel builders. Whisper is
    # already serialised by a lock, so there is nothing to gain from spreading
    # it out, and doing it up front means the printed cut list matches what is
    # actually built.
    if source_file:
        from ending_finder import refine_moment

        print("→ Checking every ending against the audio...")
        for i, moment in enumerate(moments, 1):
            before = moment.duration
            refine_moment(
                moment,
                source_file,
                model_size=model_size,
                max_total=max_seconds or MAX_TOTAL_SECONDS,
                log=lambda msg, n=i: print(f"     {n}. {msg}"),
            )
            after = moment.duration
            if abs(after - before) >= 0.05:
                print(f"        now {after:.1f}s (was {before:.1f}s)")

    def build_one(i: int, moment: Moment) -> tuple[dict | None, list[str]]:
        """Produce one short. Returns (record, its log lines).

        The log is collected instead of printed because several of these run at
        the same time, and interleaved output from three clips at once is
        unreadable. Each clip's story is buffered and printed whole when it
        finishes, so the transcript still reads top to bottom.
        """
        log: list[str] = []
        say = log.append

        folder = os.path.join(shorts_dir, f"{i:02d}_{_slug(moment.theme + '_' + moment.hook)}")
        os.makedirs(folder, exist_ok=True)
        say(f"→ Short {i}/{len(moments)}: {moment.hook}")

        raw_clip = os.path.join(folder, "clip_raw.mp4")

        # A cached clip is only reusable if it was cut at the timings this run
        # is asking for. That used to be a safe assumption and no longer is:
        # the endings are refined against the audio before anything is cut, so
        # the same moment can come back with a different end and the file on
        # disk would silently keep the old one — the exact clipped ending this
        # is meant to fix, preserved by the cache.
        wanted = [[c.start_seconds, c.end_seconds] for c in moment.cuts]
        record_path = os.path.join(folder, "moment.json")
        cached_cuts = None
        if os.path.isfile(record_path):
            try:
                with open(record_path, encoding="utf-8") as f:
                    previous = json.load(f)
                cached_cuts = [
                    [c["start_seconds"], c["end_seconds"]]
                    for c in previous.get("cuts", [])
                ]
            except (OSError, ValueError, KeyError, TypeError):
                cached_cuts = None

        if os.path.isfile(raw_clip) and cached_cuts == wanted:
            say("  ✓ Clip already cut at these timings")
        else:
            if os.path.isfile(raw_clip):
                say("  · Timings changed — recutting")
                try:
                    os.remove(raw_clip)
                except OSError:
                    pass
            pieces: list[str] = []
            failed = None
            for n, cut in enumerate(moment.cuts, start=1):
                label = f"cut {n}/{len(moment.cuts)}" if len(moment.cuts) > 1 else "clip"
                piece = (
                    raw_clip if len(moment.cuts) == 1
                    else os.path.join(folder, f"cut_{n:02d}.mp4")
                )
                if source_file:
                    # Everything comes out of the one local file — no network,
                    # so no retry loop and nothing to be blocked.
                    say(f"  · Cutting {label}: {cut.duration:.0f}s from {cut.start_seconds:.0f}s...")
                    try:
                        cut_from_file(source_file, cut.start_seconds, cut.end_seconds, piece)
                    except Exception as e:
                        failed = e
                        break
                    pieces.append(piece)
                    continue

                say(f"  · Downloading {label}: {cut.duration:.0f}s from {cut.start_seconds:.0f}s...")

                # The ranged download shells out to ffmpeg through yt-dlp and
                # occasionally crashes on a window that downloads fine on a
                # second attempt, so one retry rather than losing the moment.
                last_error = None
                for attempt in (1, 2):
                    try:
                        download_clip(url, cut.start_seconds, cut.end_seconds, piece)
                        last_error = None
                        break
                    except YouTubeBlockedDownload as e:
                        # An access block is deterministic — a second attempt
                        # fails the same way, and every remaining cut will too.
                        last_error = e
                        break
                    except Exception as e:
                        last_error = e
                        if attempt == 1:
                            say(f"    · failed ({e}); retrying once...")
                if last_error is not None:
                    failed = last_error
                    break
                pieces.append(piece)

            if failed is not None:
                say(f"  ⚠ Couldn't download this moment: {failed}")
                return None, log

            if len(pieces) > 1:
                try:
                    join_clips(pieces, raw_clip)
                    say(f"  ✓ {len(pieces)} cuts stitched")
                    for piece in pieces:
                        try:
                            os.remove(piece)
                        except OSError:
                            pass
                except Exception as e:
                    say(f"  ⚠ Couldn't stitch the cuts together: {e}")
                    return None, log
            else:
                say("  ✓ Clip ready")

        # Render against what actually landed on disk, not the length planned
        # from the transcript. Ranged downloads snap to keyframes, so a cut
        # asked for as 18.0s often arrives as 17.6s, and rendering the planned
        # length against a shorter file freezes the last frame.
        actual = probe_duration(raw_clip)
        render_duration = actual if actual > 0.5 else moment.duration
        if actual > 0.5 and abs(actual - moment.duration) > 0.4:
            say(f"  · Clip is {actual:.1f}s (planned {moment.duration:.0f}s) — rendering to the real length")

        # Whisper always runs now. The SRT is a deliverable in its own right —
        # it is what saves retyping every caption by hand in CapCut — and the
        # word timings are also what the shot list cuts the picture on. The
        # `captions` flag only decides whether they are ALSO burned into the
        # video, which is off by default because baked-in text fights the edit.
        words: list = []
        captions_path = None
        try:
            from caption_timing import build_ass, transcribe_words, words_to_srt

            words = transcribe_words(raw_clip, model_size=model_size)

            # Squeeze out the dead air before anything else uses these timings.
            #
            # It has to happen here, between transcribing and everything
            # downstream: the SRT, the shot plan and the render all read these
            # word times, and they must all describe the same file. The word
            # timings are remapped arithmetically rather than by transcribing
            # the tightened clip again, which would cost a second Whisper pass
            # to rediscover something already known exactly.
            if TIGHTEN_SILENCE:
                try:
                    from silence_trimmer import plan_keep_ranges, remap_words, tighten
                    from shorts_builder import video_encoder_args

                    ranges = plan_keep_ranges(words, render_duration)
                    tight_path = os.path.join(folder, "clip_tight.mp4")
                    used, saved = tighten(
                        raw_clip, words, render_duration, tight_path,
                        encoder_args=video_encoder_args(),
                    )
                    if saved > 0:
                        raw_clip = used
                        words = remap_words(words, ranges)
                        render_duration = max(0.5, render_duration - saved)
                        say(f"  ✓ Tightened {saved:.1f}s of dead air → {render_duration:.1f}s")
                except Exception as e:
                    say(f"  ⚠ Couldn't tighten silences ({str(e)[:60]}) — using the full cut")

            words_to_srt(words, os.path.join(folder, "captions.srt"))
            say(f"  ✓ {len(words)} words timed → captions.srt")
            if captions:
                captions_path = os.path.join(folder, "captions.ass")
                build_ass(words, captions_path, hook=moment.hook)
        except Exception as e:
            # Whisper is the most fragile piece here — it pulls in numba, which
            # Windows Application Control has blocked outright. Falling back to
            # transcript timings keeps the shot list and a usable SRT; only the
            # word-level precision is lost.
            say(f"  ⚠ Whisper unavailable ({str(e)[:60]}) — using transcript timings")
            try:
                from caption_timing import words_to_srt

                words = _words_from_segments(moment, segments)
                if words:
                    words_to_srt(words, os.path.join(folder, "captions.srt"))
                    say(f"  ✓ {len(words)} caption lines → captions.srt (line-level, not word-level)")
            except Exception as inner:
                say(f"  ⚠ Transcript timings failed too: {inner}")

        # Hashtags for the caption, visual queries for the footage search — two
        # different jobs, one call. Failing here shouldn't cost you the short.
        tags = None
        try:
            import topic_tags
            # The talk's own title and channel tell the query writer what world
            # this lives in. Without it, a line about consistency from a
            # basketball player produced "athlete meal prep" and returned stock
            # footage of vegetables being chopped.
            talk_context = " — ".join(
                part for part in (data.get("title", ""), data.get("channel", "")) if part
            )
            tags = topic_tags.for_moment(moment, context=talk_context)
            with open(os.path.join(folder, "hashtags.json"), "w", encoding="utf-8") as f:
                json.dump(tags.to_dict(), f, indent=2, ensure_ascii=False)
            say(f"  ✓ {len(tags.hashtags)} hashtags, {len(tags.visual_queries)} visual queries")
        except Exception as e:
            say(f"  ⚠ Tag generation failed: {e}")

        try:
            with open(os.path.join(folder, "brief.md"), "w", encoding="utf-8") as f:
                f.write(_clip_brief(
                    moment, i, url, segments,
                    data.get("title", ""), data.get("channel", ""), tags,
                ))
            say("  ✓ brief.md")
        except Exception as e:
            say(f"  ⚠ Couldn't write the brief: {e}")

        # The post caption, ready to paste.
        try:
            caption_lines = [moment.hook, ""]
            if moment.quote:
                caption_lines += [f"“{moment.quote}”", ""]
            if tags is not None and tags.hashtags:
                caption_lines.append(" ".join(f"#{t}" for t in tags.hashtags))
            with open(os.path.join(folder, "caption.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(caption_lines).strip() + "\n")
        except Exception as e:
            say(f"  ⚠ Couldn't write the caption: {e}")

        # The comment to post under your own short, and pin.
        #
        # A short lands on a feed with no thread under it, and the first comment
        # decides whether one forms. Left alone it is usually praise for the
        # speaker, which reads well and generates nothing, because there is no
        # reply to praise. An open question gives every later viewer something
        # obvious to do.
        #
        # Written per clip rather than once per talk: the question has to be
        # about *this* statement. One that would fit any motivational video is
        # a wasted comment.
        try:
            import comment_prompt

            prompt = comment_prompt.for_moment(
                moment,
                means=(tags.means if tags is not None else ""),
                context=talk_context if tags is not None else "",
            )
            with open(os.path.join(folder, "comment.txt"), "w", encoding="utf-8") as f:
                f.write(prompt.to_text())
            with open(os.path.join(folder, "comment.json"), "w", encoding="utf-8") as f:
                json.dump(prompt.to_dict(), f, indent=2, ensure_ascii=False)
            say(f"  ✓ comment to pin: {prompt.question}")
        except Exception as e:
            say(f"  ⚠ Couldn't write the comment: {e}")

        # --- b-roll, one distinct shot per beat --------------------------------
        broll_local: list[str] = []
        credits: dict = {}

        if fetch:
            # Context-aware queries lead. `moment.visual_keywords` is written
            # while reading the transcript alone, before anything knows who is
            # speaking — that is where "athlete meal prep" came from, and why
            # it returned footage of vegetables being chopped for a basketball
            # interview. Kept only as a fallback when the tag call failed.
            queries: list[str] = []
            if tags is not None:
                queries = list(getattr(tags, "visual_queries", []))
            if not queries:
                queries = list(moment.visual_keywords)
            picked = []
            wanted = _broll_slots(render_duration)
            try:
                # Serialised: three workers hitting the same stock API at once
                # gets rate-limited, and the shared "already used" set has to be
                # read and written without two clips claiming the same footage.
                with BROLL_LOCK:
                    # Hand-picked footage always wins. Stock only fills the gap
                    # it leaves, so the library gets better every time you add
                    # to assets/broll/ without anything else changing.
                    picked = asset_library.curated_broll(
                        queries, count=wanted, exclude=used_broll
                    )
                    # The library is finite. Once earlier shorts have claimed
                    # most of it, insisting on unused clips pushes the rest of
                    # the run onto stock — which is worse than a clip appearing
                    # in two different videos that get posted days apart.
                    # Repeats within a single short are what actually look
                    # cheap, and those are prevented by asking for `wanted`
                    # distinct clips here.
                    if len(picked) < wanted:
                        already = {a.id for a in picked}
                        picked += asset_library.curated_broll(
                            queries, count=wanted - len(picked), exclude=already
                        )
                    if len(picked) < wanted:
                        for asset in picked:
                            used_broll.add(asset.id)
                        picked += asset_library.fetch_broll_set(
                            queries, count=wanted - len(picked), exclude=used_broll
                        )
                    for asset in picked:
                        used_broll.add(asset.id)
            except Exception as e:
                say(f"  ⚠ B-roll fetch failed: {e}")

            curated_count = sum(1 for a in picked if a.source == "curated")
            if curated_count:
                say(f"  ✓ {curated_count} curated + {len(picked) - curated_count} stock")

            if picked:
                import shutil as _shutil

                broll_dir = os.path.join(folder, "broll")
                os.makedirs(broll_dir, exist_ok=True)

                # Clear what a previous run left behind.
                #
                # Selection changes between runs — the library gets pruned, the
                # tags change, the model picks differently — so the old files
                # are not merely redundant, they are footage this edit does not
                # use. Left in place they accumulate: eighty-one stale clips
                # across one batch, so a folder opened in CapCut showed
                # thirty-four options when eleven were used. That folder is
                # meant to be the shot list made physical.
                for old in os.listdir(broll_dir):
                    if os.path.splitext(old)[1].lower() in asset_library.VIDEO_EXTS:
                        try:
                            os.remove(os.path.join(broll_dir, old))
                        except OSError:
                            pass

                for n, asset in enumerate(picked, 1):
                    target = os.path.join(
                        broll_dir, f"{n:02d}_{os.path.basename(asset.path)}"
                    )
                    try:
                        if not os.path.isfile(target):
                            _shutil.copyfile(asset.abs_path, target)
                        broll_local.append(target)
                    except OSError:
                        continue
                credits["broll"] = [
                    {"path": a.path, "source": a.source, "credit": a.credit,
                     "licence": a.licence}
                    for a in picked
                ]

        # Opens on the speaker, cuts away for a second or two, comes back.
        plan = _shot_plan(render_duration, raw_clip, broll_local)
        cutaways = sum(1 for _p, _s, _d, kind in plan if kind == "b-roll")
        if broll_local:
            say(f"  ✓ {len(broll_local)} b-roll clip(s), {len(plan)} shots ({cutaways} cutaways)")

        try:
            with open(os.path.join(folder, "shotlist.md"), "w", encoding="utf-8") as f:
                f.write(_shotlist(plan, words, moment.hook))
        except Exception as e:
            say(f"  ⚠ Couldn't write the shot list: {e}")

        out_path = os.path.join(folder, "short.mp4")
        try:
            if broll_local:
                shots = [(path, src, dur) for path, src, dur, _kind in plan]
                build_rough_cut(raw_clip, shots, out_path, render_duration,
                                words=words)
                say(f"  ✓ {out_path} (speaker, cutting to b-roll and back)")
            else:
                # No footage available — fall back to the speaker's own picture
                # so the run still produces something usable.
                build_short(ShortSpec(
                    speech_source=raw_clip,
                    duration=render_duration,
                    out_path=out_path,
                    captions_path=captions_path,
                    style="speaker",
                ))
                say(f"  ✓ {out_path} (speaker footage — no b-roll available)")
        except Exception as e:
            say(f"  ⚠ Render failed: {e}")
            return None, log

        record = moment.to_dict()
        record.update({
            "index": i,
            "video_id": video_id,
            "source_url": url,
            "style": style,
            "folder": folder,
            "short": out_path,
            "assets": credits,
            "caption_words": len(words) if captions_path else 0,
        })
        with open(os.path.join(folder, "moment.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return record, log

    # Clips are independent of each other, so they overlap. The work is a mix
    # of local ffmpeg cutting, Whisper on the CPU, one API call and a GPU
    # encode, which is exactly the shape that benefits: while one clip waits on
    # the network, another is encoding.
    #
    # Kept deliberately low. Each worker holds its own Whisper model in memory,
    # and pushing past the core count turns a speed-up into contention.
    lanes = max(1, min(workers, len(moments)))

    if lanes == 1:
        for i, moment in enumerate(moments, 1):
            print()
            record, log = build_one(i, moment)
            print("\n".join(log))
            if record is not None:
                results.append(record)
    else:
        from concurrent.futures import ThreadPoolExecutor

        print(f"\n→ Building {len(moments)} shorts, {lanes} at a time...")
        with ThreadPoolExecutor(max_workers=lanes) as pool:
            # Submitted all at once so they genuinely overlap; read back in
            # order so the output still reads as short 1, 2, 3.
            pending = [pool.submit(build_one, i, m) for i, m in enumerate(moments, 1)]
            for future in pending:
                record, log = future.result()
                print()
                print("\n".join(log))
                if record is not None:
                    results.append(record)

    # Re-cutting a talk renames the folders (they are built from the hook), so
    # the previous run's clips would otherwise sit beside the new ones — same
    # moment, different name, its own copy of the video.
    if results and not keep_old:
        keep = {os.path.abspath(r["folder"]) for r in results}
        stale = _clear_stale_clips(shorts_dir, keep)
        if stale:
            print()
            print(f"  · Removed {len(stale)} folder(s) from an earlier run of this talk:")
            for name in stale:
                print(f"      {name}")

    # Everything needed to actually post: brief, blog, thread, captions and
    # descriptions. Written once per talk rather than once per clip, because
    # they describe the talk, not any single cut of it.
    format_files: list[str] = []
    if formats:
        print()
        print("→ Writing the publishing set...")
        try:
            format_files = _publishing_set(data, os.path.join(output_dir, video_id))
            print(f"  ✓ {len(format_files)} file(s): {', '.join(format_files)}")
        except Exception as e:
            print(f"  ⚠ Publishing set failed: {e}")

    carousel_paths: list[str] = []
    if carousel:
        print("\n→ Writing carousel copy from the same moments...")
        try:
            carousel_paths = build_quote_carousel(
                moments,
                os.path.join(shorts_dir, "carousel"),
                talk_title=data.get("title", ""),
                channel=data.get("channel", ""),
                fetch=fetch,
            )
            print(f"  ✓ carousel copy written ({len(carousel_paths)} file(s))")
        except Exception as e:
            print(f"  ⚠ Carousel copy failed: {e}")

    with open(os.path.join(shorts_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "source_url": url,
            "shorts": results,
            "carousel": carousel_paths,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done. {len(results)} short(s) in {shorts_dir}")
    for r in results:
        print(f"    {r['short']}")
    for p in carousel_paths:
        print(f"    {p}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut a long talk into vertical motivational shorts."
    )
    parser.add_argument("url", help="YouTube URL or bare video ID")
    parser.add_argument("--count", type=int, default=None,
                        help="how many moments to cut. Omit to let the talk decide — "
                             "one per replay peak worth cutting, up to 8")
    parser.add_argument("--style", default="broll", choices=["broll", "speaker", "split"],
                        help="broll: stock footage; speaker: the talk's own video; split: both")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--whisper-model", default="base",
                        help="tiny/base/small — larger is more accurate, slower")
    parser.add_argument("--no-fetch", action="store_true",
                        help="use only what's already in assets/, never call a stock API")
    parser.add_argument("--no-carousel", action="store_true",
                        help="skip the carousel copy (text only, no images)")
    parser.add_argument("--no-formats", action="store_true",
                        help="skip the publishing set — brief, blog post, X thread, "
                             "captions and descriptions. Saves about four Claude "
                             "calls per talk when you only want the clips")
    parser.add_argument("--min-seconds", type=int, default=None,
                        help="shortest finished short (default 7)")
    parser.add_argument("--max-seconds", type=int, default=None,
                        help=f"longest finished short (default {MAX_TOTAL_SECONDS})")
    parser.add_argument("--keep-old", action="store_true",
                        help="don't remove clip folders left by an earlier run of "
                             "the same talk (they show up as duplicates)")
    parser.add_argument("--captions", action="store_true",
                        help="burn karaoke captions into the clip. Off by default, "
                             "because these clips are usually re-edited afterwards")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"how many clips to build at once (default {DEFAULT_WORKERS}; "
                             "1 makes the log easier to read when debugging)")
    parser.add_argument("--source-file", default=None,
                        help="cut from this local video instead of downloading; "
                             "the URL is still used for the transcript and heatmap")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY not set. Add it to .env first.")
        sys.exit(1)

    if args.source_file and not os.path.isfile(args.source_file):
        print(f"✗ Source file not found: {args.source_file}")
        sys.exit(1)

    make_shorts(
        args.url,
        count=args.count,
        style=args.style,
        output_dir=args.output_dir,
        model_size=args.whisper_model,
        fetch=not args.no_fetch,
        carousel=not args.no_carousel,
        formats=not args.no_formats,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        source_file=args.source_file,
        workers=args.workers,
        captions=args.captions,
        keep_old=args.keep_old,
    )


if __name__ == "__main__":
    main()
