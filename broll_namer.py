"""
broll_namer.py

Names b-roll clips after what is visible in them, so the library can be
searched.

    python broll_namer.py assets/broll/film/solitude --dry-run
    python broll_namer.py assets/broll/film/solitude
    python broll_namer.py assets/broll/film --pattern "br49-*"

**Why this exists.** `asset_library.curated_broll` matches footage to a moment
by splitting the filename into words. A clip cut automatically out of a film is
called `br49-07.mp4`, which produces the tags `['br49']` — a token no query
will ever contain. Several hundred such clips do not form a library; they form
a pile that gets sampled in folder order.

Naming them by hand works and does not scale: eighteen clips took a contact
sheet and a careful pass, and there are now over a thousand.

**What it asks for, and what it deliberately does not.** The model is asked to
describe the *picture* — what is physically in frame — and explicitly not to
identify the film, the actors or the characters. That is not only a rights
question: a clip called `blade-runner-2049-k-walking.mp4` matches the words
"blade", "runner" and "walking", of which only one describes anything a person
would search for. `neon-street-rain-night.mp4` matches four.

Haiku is used rather than a larger model because the task is naming what is in
a photograph, which it does as well as anything and for a fraction of the cost
across a thousand clips.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

MODEL = "claude-haiku-4-5"

# Small on purpose. The frame only has to be legible enough to say what is in
# it, and image tokens scale with area — 320px wide costs a fraction of 1080p
# and produces the same three words.
FRAME_WIDTH = 320

# Where the undo record goes, so a bad batch can be put back.
LEDGER_NAME = ".names.json"

PROMPT = """Describe what is physically visible in this frame, as 3-5 words for a filename.

Rules:
- Describe the picture only: subject, setting, light, weather, time of day.
- Do NOT name the film, the actors, or any character.
- Concrete words a person would search for: "man", "desert", "rain", "neon",
  "corridor", "fire", "crowd", "snow", "silhouette", "close-up".
- Lowercase, separated by single hyphens, nothing else in your reply.

Examples of good answers:
neon-street-rain-night
lone-figure-desert-dusk
close-up-face-firelight
crowd-marching-snow-wide"""


def mid_frame(path: str) -> bytes | None:
    """One JPEG frame from the middle of the clip."""
    dur = 0.0
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    try:
        dur = float(r.stdout.strip())
    except ValueError:
        pass

    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-ss", f"{max(0.1, dur / 2):.2f}", "-i", path, "-vframes", "1",
         "-vf", f"scale={FRAME_WIDTH}:-2", "-f", "mjpeg", "-q:v", "6", "-"],
        capture_output=True)
    return out.stdout or None


def _slug(text: str) -> str:
    """The model's answer as a filename fragment, whatever it actually sent."""
    text = (text or "").strip().lower().splitlines()[0] if text else ""
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    # Five words is the cap: past that the name stops being scannable and the
    # extra tags are noise rather than signal.
    return "-".join(text.split("-")[:5])[:60]


def describe(client, path: str) -> str | None:
    frame = mid_frame(path)
    if not frame:
        return None
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(frame).decode(),
                    }},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
    except Exception as e:
        print(f"    ! {os.path.basename(path)}: {e}")
        return None

    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _slug(text) or None


def already_named(name: str) -> bool:
    """True if the file already carries words rather than just a prefix and index.

    `br49-07.mp4` is unnamed; `br49-07-neon-street-rain.mp4` is named. Checked so
    the script can be re-run over a folder without paying for the same clips
    twice.
    """
    stem = os.path.splitext(name)[0]
    return bool(re.search(r"[a-z]{3,}", re.sub(r"^[a-z0-9]+-\d+", "", stem)))


def collect(root: str, pattern: str) -> list[str]:
    found = []
    for r, _dirs, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(".mp4") and fnmatch.fnmatch(f, pattern):
                found.append(os.path.join(r, f))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Name b-roll after what is in it.")
    parser.add_argument("folder")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="show the names without renaming anything")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    import anthropic

    client = anthropic.Anthropic()

    clips = [c for c in collect(args.folder, args.pattern)
             if not already_named(os.path.basename(c))]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print("Nothing to name — every clip already carries words.")
        return

    print(f"Naming {len(clips)} clip(s) with {MODEL}...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        names = list(pool.map(lambda c: describe(client, c), clips))

    ledger_path = os.path.join(args.folder, LEDGER_NAME)
    try:
        with open(ledger_path, encoding="utf-8") as f:
            ledger = json.load(f)
    except (OSError, ValueError):
        ledger = {}

    renamed = 0
    for path, slug in zip(clips, names):
        base = os.path.basename(path)
        if not slug:
            print(f"  ?  {base}  (no answer)")
            continue
        stem, ext = os.path.splitext(base)
        dest = os.path.join(os.path.dirname(path), f"{stem}-{slug}{ext}")
        print(f"  {base}  ->  {os.path.basename(dest)}")
        if args.dry_run or os.path.exists(dest):
            continue
        try:
            os.rename(path, dest)
            ledger[os.path.basename(dest)] = base
            renamed += 1
        except OSError as e:
            print(f"    ! could not rename: {e}")

    if renamed and not args.dry_run:
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=1)
        print(f"\nrenamed {renamed}; undo map in {LEDGER_NAME}")
    elif args.dry_run:
        print("\ndry run — nothing renamed")


if __name__ == "__main__":
    main()
