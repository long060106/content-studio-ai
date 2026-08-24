"""
gather_broll.py

Builds and fills `assets/broll/` — the hand-picked footage library that the
shorts pipeline prefers over anything it fetches on the fly.

    python gather_broll.py                      # fill everything that's thin
    python gather_broll.py --only emotion/lost  # just one folder
    python gather_broll.py --per 6              # more clips per query
    python gather_broll.py --list               # what's there now

Why a curated library exists at all: stock search is fine at matching *words*
and poor at matching *feeling*. Asked for footage about acceptance it returns a
handshake in an office. The folders below are organised by emotion instead, so
the pipeline can ask for "lost" and get fog, empty roads and long silences —
whatever you decided those look like.

Folder names are tags. `emotion/lost/road-fog.mp4` matches a query mentioning
"lost", "road" or "fog", so filing a clip in the right folder is enough; naming
it well is a bonus rather than a chore.

On the `cinematic/` folder: it stays empty here. The look those reference
accounts have comes from film scenes, and this script will not download
copyrighted film. Fill it yourself from something you have the right to use —
public-domain film on archive.org, or a subscription library like Artgrid —
and the pipeline will prefer it automatically.
"""

from __future__ import annotations

import argparse
import os
import shutil

import asset_library

# What each folder should contain, expressed as searches that return footage
# carrying that feeling. These are deliberately concrete: "grateful" returns
# nothing usable, "hands open warm light" returns the thing that *feels*
# grateful.
LIBRARY: dict[str, list[str]] = {
    # The general mood shots — dark, slow, cinematic-adjacent.
    "general/night": ["silhouette night", "city night alone", "street lamp rain night"],
    "general/rain": ["slow motion rain", "rain on window", "storm clouds dark"],
    "general/training": ["dark gym single light", "hands chalk closeup", "boxer shadow training"],
    "general/nature": ["fog forest run", "cold breath winter", "mountain fog sunrise"],
    "general/solitude": ["empty road night", "lone figure walking", "empty hallway light"],

    # Emotional register. The pipeline names a feeling before it picks
    # footage, so these map straight onto what it asks for.
    "emotion/angry": ["boxer punching bag dark", "storm sea waves crashing", "clenched fist closeup"],
    "emotion/depressed": ["man alone dark room", "rain window night sad", "empty bed morning"],
    "emotion/lost": ["fog forest walking alone", "empty crossroads", "person staring distance"],
    "emotion/sad": ["rain on window slow", "empty park bench", "looking away window"],
    "emotion/happy": ["friends laughing golden hour", "sunrise silhouette jumping", "child running field"],
    "emotion/grateful": ["hands open warm light", "sunrise mountains silhouette", "candle flame dark"],
    "emotion/accepted": ["still lake sunrise", "calm breathing meditation", "sitting quietly window"],
    "emotion/numb": ["blank stare window", "empty room grey light", "grey ocean horizon"],
    "emotion/resolve": ["running stairs dawn", "lifting heavy weight strain", "walking into storm"],
}

# Left for the user to fill; see the module docstring.
MANUAL_FOLDERS = ["cinematic"]


def existing(folder: str) -> int:
    path = os.path.join(asset_library.CURATED_DIR, folder)
    if not os.path.isdir(path):
        return 0
    return sum(
        1 for n in os.listdir(path)
        if os.path.splitext(n)[1].lower() in asset_library.VIDEO_EXTS
    )


def gather(folder: str, queries: list[str], per_query: int, target: int) -> int:
    """Fill one folder up to `target` clips. Returns how many were added."""
    path = os.path.join(asset_library.CURATED_DIR, folder)
    os.makedirs(path, exist_ok=True)

    have = existing(folder)
    if have >= target:
        print(f"  {folder:24} {have:3} clips — already full")
        return 0

    added = 0
    for query in queries:
        if have + added >= target:
            break
        try:
            found = asset_library.fetch_stock(query, kind="video", count=per_query)
        except Exception as e:
            print(f"  {folder:24} search failed for {query!r}: {str(e)[:60]}")
            continue

        for asset in found:
            if have + added >= target:
                break
            src = asset.abs_path
            if not os.path.isfile(src):
                continue
            # Named after the query so the filename carries the meaning too,
            # not just the folder.
            stem = query.replace(" ", "-")
            dest = os.path.join(path, f"{stem}-{os.path.basename(asset.path)}")
            if os.path.isfile(dest):
                continue
            try:
                shutil.copyfile(src, dest)
                added += 1
            except OSError:
                continue

    print(f"  {folder:24} {have:3} -> {have + added:3} clips")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the curated b-roll library.")
    parser.add_argument("--only", help="fill just this folder, e.g. emotion/lost")
    parser.add_argument("--per", type=int, default=4, help="results per search (default 4)")
    parser.add_argument("--target", type=int, default=6,
                        help="clips to keep per folder (default 6)")
    parser.add_argument("--list", action="store_true", help="show what's already there")
    args = parser.parse_args()

    os.makedirs(asset_library.CURATED_DIR, exist_ok=True)
    for folder in MANUAL_FOLDERS:
        os.makedirs(os.path.join(asset_library.CURATED_DIR, folder), exist_ok=True)

    if args.list:
        total = 0
        for folder in sorted(list(LIBRARY) + MANUAL_FOLDERS):
            count = existing(folder)
            total += count
            note = "  (fill this yourself)" if folder in MANUAL_FOLDERS else ""
            print(f"  {folder:24} {count:3} clips{note}")
        print(f"\n  {total} clips total")
        return

    if not os.environ.get("PEXELS_API_KEY", "").strip():
        print("PEXELS_API_KEY isn't set — nothing can be fetched.")
        raise SystemExit(1)

    folders = {args.only: LIBRARY[args.only]} if args.only else LIBRARY
    if args.only and args.only not in LIBRARY:
        print(f"Unknown folder {args.only!r}. Known: {', '.join(sorted(LIBRARY))}")
        raise SystemExit(1)

    print(f"Filling {len(folders)} folder(s) to {args.target} clips each...\n")
    added = sum(
        gather(folder, queries, args.per, args.target)
        for folder, queries in folders.items()
    )
    print(f"\n{added} new clip(s) added to assets/broll/")
    print("The pipeline prefers these over anything fetched at run time.")


if __name__ == "__main__":
    main()
