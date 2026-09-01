"""
broll_sorter.py

Files each b-roll clip by what the clip shows, into a smaller set of folders.

    python broll_sorter.py --dry-run
    python broll_sorter.py

**The problem this fixes.** Clips were filed one category per *film*, so all
thirty shots cut from a bleak film landed in `depressed` whatever each one
actually showed. Measured on the library: twenty-one clips with plainly warm
descriptions were sitting in dark folders — `couple-embracing-wheat-field-golden`
under `epic`, `massive-crowd-confetti-celebration-wide` under `motivation`,
three dancing shots under `depressed` — while `happy`, `inspire`, `kindness`
and `accepted` held nothing at all. Asking the library for a joyful cutaway
returned nothing, because the joy was all filed under sorrow.

The category describes the film. It should describe the picture.

**Why keywords rather than another model pass.** Every clip already carries a
description in its filename, written by looking at the frame. Sorting on words
that are already there is deterministic, instant, free, and re-runnable — and
if a rule is wrong it can be read and corrected, which is not true of a second
opinion from a model.

**Why eight folders.** Seventeen was too many to hold in your head, and the
distinctions between `motivation`, `passion` and `inspire` were ones nobody
could apply consistently — least of all the code. The film each clip came from
survives in its filename prefix, so nothing is lost by dropping film-shaped
categories.
"""

from __future__ import annotations

import argparse
import os
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "assets", "broll", "film")

# First match wins, so the order is the priority. The specific and emotional
# come before the scenic: a couple embracing in a field is about the couple,
# not the field.
RULES: list[tuple[str, tuple[str, ...]]] = [
    # A trailing * matches the start of a word ("smil*" catches smiling and
    # smile); a hyphenated entry matches that sequence of words; anything else
    # must be a whole word. Substring matching was tried first and filed a
    # railway "train-platform" under effort and "two-soldiers-talking" under
    # tension, because it cannot tell a word from a fragment of one.
    ("happy", (
        "smil*", "laugh*", "celebrat*", "joy*", "cheer*", "danc*", "confetti",
        "applau*", "party", "grin*", "delight*", "festive", "playing",
        "toast", "crowd-cheer", "fireworks", "sunlit", "golden-light",
    )),
    ("love", (
        "embrac*", "couple", "couples", "kiss*", "lovers", "holding-hands",
        "wedding", "tender", "cradl*", "mother", "family", "reunion",
    )),
    ("sorrow", (
        "weep*", "crying", "cries", "grief", "funeral", "tears", "mourn*",
        "coffin", "sorrow", "despair", "head-in-hands", "hands-on-face",
        "grave", "graves", "cemetery",
    )),
    ("effort", (
        "running", "runner", "climb*", "training", "sweat*", "boxing",
        "drummer", "drumming", "workshop", "factory", "machinery",
        "industrial", "construction", "lifting", "rowing", "racing",
        "sprint*", "workout", "practice", "rehears*",
    )),
    ("tension", (
        "blood*", "horror", "gun", "guns", "weapon*", "explosion", "fight*",
        "burning", "flames", "smoke", "storm*", "chase", "wreck*", "war",
        "battle", "riot", "confrontation", "screaming",
    )),
    ("solitude", (
        "lone", "alone", "empty", "silhouette", "silhouettes", "solitary",
        "isolated", "abandoned", "deserted", "single-figure",
    )),
    ("epic", (
        "army", "armies", "fortress", "castle", "ship", "ships", "mountain",
        "mountains", "desert", "marching", "aerial", "cityscape", "dragon",
        "cathedral", "temple", "palace", "crowd", "spacecraft", "throne",
        "canyon", "valley", "skyline",
    )),
    ("stillness", (
        "ocean", "sea", "sky", "dawn", "dusk", "sunset", "sunrise", "stars",
        "space", "forest", "mist*", "fog*", "snow*", "water", "rain", "field",
        "window", "corridor", "room", "interior", "street", "night", "light",
        "shore", "lake", "river", "clouds", "trees", "path", "road",
    )),
]

FALLBACK = "other"


def categorise(filename: str) -> str:
    """Which folder this clip belongs in, from the words in its own name."""
    name = os.path.splitext(filename.lower())[0]
    # Skip the film prefix and index (`subs-02-`) so a film's short code cannot
    # accidentally match a rule.
    parts = name.split("-")
    tokens = parts[2:] if len(parts) > 2 else parts
    body = set(tokens)

    for category, words in RULES:
        for w in words:
            if "-" in w:                      # a phrase: match it in sequence
                if w in "-".join(tokens):
                    return category
            elif w.endswith("*"):             # a stem: match the start of a word
                stem = w[:-1]
                if any(tok.startswith(stem) for tok in tokens):
                    return category
            elif w in body:                   # otherwise the whole word
                return category
    return FALLBACK


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-file b-roll by content.")
    parser.add_argument("--dry-run", action="store_true")
    # The film library is the default because it is the one that grows every
    # time a film is cut up. Stock lives in its own tree beside it and needs
    # the same treatment for a reason worth stating: the folder a clip sits in
    # is one of its tags, so a flat folder of nature footage carries only the
    # word "stock" and loses every match against a mood word like "solitude"
    # to a film clip that happens to sit in a folder named after one.
    parser.add_argument("--root", default=ROOT,
                        help="folder to sort (default: the film library)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    clips: list[str] = []
    for r, _dirs, files in os.walk(root):
        clips += [os.path.join(r, f) for f in files if f.lower().endswith(".mp4")]

    if not clips:
        print(f"No clips under {root}")
        return

    plan: dict[str, list[str]] = {}
    for path in clips:
        plan.setdefault(categorise(os.path.basename(path)), []).append(path)

    print(f"{len(clips)} clips -> {len(plan)} folders\n")
    for category in sorted(plan, key=lambda c: -len(plan[c])):
        print(f"  {category:11} {len(plan[category]):5}")
        if args.dry_run:
            for p in plan[category][:2]:
                print(f"                 e.g. {os.path.basename(p)[:62]}")

    if args.dry_run:
        print("\ndry run — nothing moved")
        return

    moved = 0
    for category, paths in plan.items():
        dest_dir = os.path.join(root, category)
        os.makedirs(dest_dir, exist_ok=True)
        for path in paths:
            dest = os.path.join(dest_dir, os.path.basename(path))
            if os.path.abspath(dest) == os.path.abspath(path):
                continue
            if os.path.exists(dest):
                # The same clip is already filed correctly, so this one is a
                # leftover from an earlier arrangement. Skipping the move left
                # it stranded in a folder that then could not be cleaned up as
                # empty — one clip kept a whole obsolete category alive.
                try:
                    if os.path.getsize(dest) == os.path.getsize(path):
                        os.remove(path)
                except OSError:
                    pass
                continue
            try:
                shutil.move(path, dest)
                moved += 1
            except OSError as e:
                print(f"  ! {os.path.basename(path)}: {e}")

    # Drop the folders nothing landed in. An empty category is a category
    # nobody can use, and it was half the reason the list felt unmanageable.
    removed = 0
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        if not any(f.lower().endswith(".mp4")
                   for _r, _d, fs in os.walk(folder) for f in fs):
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1

    print(f"\nmoved {moved}, removed {removed} empty folder(s)")


if __name__ == "__main__":
    main()
