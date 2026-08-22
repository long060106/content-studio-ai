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
  4. Captions        Whisper runs on each short clip alone for word-accurate
                     karaoke timing. Seconds per clip, and free.
  5. Assets          B-roll comes from the local library, topped up from
                     stock APIs when a key is configured. No music — see
                     pick_assets for why.
  6. Render          ffmpeg assembles 1080x1920 with captions burned in.
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

import asset_library
from caption_timing import captions_for_clip
from moment_finder import Moment, find_moments
from shorts_builder import ShortSpec, build_short, join_clips, probe_duration


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
    carousel: bool = False,
    min_seconds: int | None = None,
    max_seconds: int | None = None,
    source_file: str | None = None,
) -> list[dict]:
    from video_clipper import download_clip, download_full, cut_from_file, YouTubeBlockedDownload
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
    if source_file is None and total_cuts > 1:
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

    for i, moment in enumerate(moments, 1):
        folder = os.path.join(shorts_dir, f"{i:02d}_{_slug(moment.theme + '_' + moment.hook)}")
        os.makedirs(folder, exist_ok=True)
        print(f"\n→ Short {i}/{len(moments)}: {moment.hook}")

        raw_clip = os.path.join(folder, "clip_raw.mp4")
        if os.path.isfile(raw_clip):
            print("  ✓ Clip already downloaded")
        else:
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
                    print(f"  · Cutting {label}: {cut.duration:.0f}s from {cut.start_seconds:.0f}s...")
                    try:
                        cut_from_file(source_file, cut.start_seconds, cut.end_seconds, piece)
                    except Exception as e:
                        failed = e
                        break
                    pieces.append(piece)
                    continue

                print(f"  · Downloading {label}: {cut.duration:.0f}s from {cut.start_seconds:.0f}s...")

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
                            print(f"    · failed ({e}); retrying once...")
                if last_error is not None:
                    failed = last_error
                    break
                pieces.append(piece)

            if failed is not None:
                print(f"  ⚠ Couldn't download this moment: {failed}")
                continue

            if len(pieces) > 1:
                try:
                    join_clips(pieces, raw_clip)
                    print(f"  ✓ {len(pieces)} cuts downloaded and stitched")
                    for piece in pieces:
                        try:
                            os.remove(piece)
                        except OSError:
                            pass
                except Exception as e:
                    print(f"  ⚠ Couldn't stitch the cuts together: {e}")
                    continue
            else:
                print("  ✓ Clip downloaded")

        # Render against what actually landed on disk, not the length planned
        # from the transcript. Ranged downloads snap to keyframes, so a cut
        # asked for as 18.0s often arrives as 17.6s, and rendering the planned
        # length against a shorter file freezes the last frame.
        actual = probe_duration(raw_clip)
        render_duration = actual if actual > 0.5 else moment.duration
        if actual > 0.5 and abs(actual - moment.duration) > 0.4:
            print(f"  · Clip is {actual:.1f}s (planned {moment.duration:.0f}s) — rendering to the real length")

        try:
            print("  · Timing captions with Whisper...")
            captions_path = os.path.join(folder, "captions.ass")
            _, words = captions_for_clip(
                raw_clip, captions_path, hook=moment.hook, model_size=model_size
            )
            print(f"  ✓ {len(words)} words timed")
        except Exception as e:
            print(f"  ⚠ Caption timing failed, rendering without captions: {e}")
            captions_path = None

        # Hashtags for the caption, visual queries for the image search — two
        # different jobs, one call. Failing here shouldn't cost you the short.
        tags = None
        try:
            import topic_tags
            tags = topic_tags.for_moment(moment)
            with open(os.path.join(folder, "hashtags.json"), "w", encoding="utf-8") as f:
                json.dump(tags.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"  ✓ {len(tags.hashtags)} hashtags, {len(tags.visual_queries)} visual queries")
        except Exception as e:
            print(f"  ⚠ Tag generation failed: {e}")

        broll_path, credits = pick_assets(moment, fetch=fetch)
        if broll_path:
            print(f"  ✓ B-roll: {os.path.basename(broll_path)}")
        elif style == "broll":
            print("  · No b-roll available, falling back to the speaker's own footage")
        out_path = os.path.join(folder, "short.mp4")
        try:
            print("  · Rendering...")
            build_short(ShortSpec(
                speech_source=raw_clip,
                duration=render_duration,
                out_path=out_path,
                captions_path=captions_path,
                broll_path=broll_path,
                style=style,
            ))
            print(f"  ✓ {out_path}")
        except Exception as e:
            print(f"  ⚠ Render failed: {e}")
            continue

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
        results.append(record)

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
    parser.add_argument("--carousel", action="store_true",
                        help="also write the carousel copy (text only) from the same moments")
    parser.add_argument("--min-seconds", type=int, default=None,
                        help="shortest finished short (default 7)")
    parser.add_argument("--max-seconds", type=int, default=None,
                        help="longest finished short (default 20)")
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
        carousel=args.carousel,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        source_file=args.source_file,
    )


if __name__ == "__main__":
    main()
