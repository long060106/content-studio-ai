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
    # --- the emotional register ------------------------------------------
    #
    # These folder names are the vocabulary the pipeline actually speaks:
    # `topic_tags` names a feeling for each clip before any footage is chosen,
    # and the match is made against these names and the filenames inside them.
    # Adding a folder here adds a feeling the selector can ask for.
    "emotion/sad": ["rain running down glass", "empty park bench wide", "face profile window light"],
    "emotion/happy": ["friends laughing golden hour", "sunrise field wide", "running through field"],
    "emotion/inspire": ["mountain summit sunrise wide", "vast landscape figure small", "light through cathedral window"],
    "emotion/motivation": ["running stairs profile", "walking into storm wide", "rowing sunrise wide"],
    "emotion/kindness": ["hands helping up", "two people embrace warm light", "sharing food table warm"],
    "emotion/growth": ["seedling breaking soil", "time lapse plant growing", "path through forest opening"],
    "emotion/depressed": ["man sitting dark room wide", "grey rain street", "empty apartment window light"],
    "emotion/passion": ["hands playing piano close", "painter working canvas", "dancer rehearsing alone"],
    "emotion/angry": ["storm sea waves wide", "boxer profile hitting bag", "fire burning close"],
    "emotion/lost": ["fog road wide", "empty crossroads aerial", "figure small in landscape"],
    "emotion/grateful": ["hands open warm light", "sunrise mountains wide", "candle flame dark"],
    "emotion/accepted": ["still lake horizon", "calm sea sunrise wide", "sitting by window profile"],
    "emotion/numb": ["grey ocean horizon", "empty room grey light", "blank window stare profile"],

    # --- the cinematic register -------------------------------------------
    #
    # Drawn from the reference board rather than invented. What those clips
    # have in common is not action: it is intimacy and warm light — faces
    # thinking, people alone in rooms, rain on glass, painterly landscapes.
    # That is the register to match, and it is largely reachable with stock.
    "cinematic/intimate": ["face lit by window thinking", "hands close warm lamplight", "person alone room warm light"],
    "cinematic/rain": ["rain on window slow warm", "wet street reflections night", "rain falling lit backlight"],
    "cinematic/light": ["god rays through dust", "sunlight through curtain slow", "silhouette against bright window"],

    # The Kingdom of Heaven register, sourced legitimately.
    #
    # Its look is well documented: sweeping desert landscapes under brutally
    # harsh sun, burnt-ochre sand against deep blacks, dust in the air, and the
    # cold steely blue of the northern scenes against all that gold. None of
    # that needs the film itself — it needs the same light and the same
    # subjects, which stock libraries have.
    "cinematic/epic": [
        "desert dunes wind wide", "sandstorm dust light", "horizon heat haze desert",
        "stone fortress walls", "torch flame stone wall", "horse riding desert wide",
    ],
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
            # Landscape, not portrait — which is the reverse of what this used
            # to ask for. B-roll now renders into the same wide band as the
            # speaker instead of filling the screen, so a 1920x1080 clip scales
            # to exactly that band with nothing cropped, while a portrait clip
            # has about two thirds of its height discarded. The queries above
            # were rewritten for the same reason: compositions that read across
            # the frame rather than down it.
            found = asset_library.fetch_stock(
                query, kind="video", count=per_query, vertical=False
            )
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
