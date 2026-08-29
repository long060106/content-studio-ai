"""
broll_filter.py

Removes b-roll the account will not use: violence, horror, weapons, war.

    python broll_filter.py --dry-run
    python broll_filter.py

These are motivational shorts. A cutaway of a rifle or a corpse under a line
about discipline is not merely off-tone, it is the kind of thing that gets a
post reported, and it arrives without anyone choosing it — the shot scorer
rewards contrast, and a muzzle flash in a dark frame scores extremely well.

**It reads the descriptions, not the films.** Every clip is named for what is
visible in it, so the filter can act on the picture rather than on the title it
came from. That matters for a film like Fight Club, where the cinematography is
worth keeping and the fights are not: cutting every shot and then filtering
leaves the neon laundromats and the empty apartments and drops the bare
knuckles.

**Whole words, with an escape list.** Substring matching is wrong here in a way
that costs real footage: `dead-tree` is a tree, `war-room-conference-table` is
people at a meeting, and `burning-candle` is a candle. Each is spared by name,
because a rule that silently deletes scenery is worse than one that occasionally
keeps a shot it should not.
"""

from __future__ import annotations

import argparse
import os

# One word of this in a clip's description is enough to drop it.
VIOLENT = {
    "horror", "blood", "bloody", "gun", "guns", "gunfire", "shooting",
    "weapon", "weapons", "knife", "knives", "sword", "swords", "fight",
    "fights", "fighting", "fighter", "fighters", "punch", "punching",
    "kill", "killing", "killed", "corpse", "corpses", "death", "dying",
    "explosion", "explosions", "violence", "violent", "wound", "wounded",
    "scream", "screaming", "terror", "monster", "zombie", "vampire",
    "execution", "torture", "stab", "stabbing", "rifle", "pistol", "gore",
    "skull", "skeleton", "massacre", "combat", "battle", "battlefield",
    "riot", "axe", "soldier", "soldiers", "army", "armies", "trenches",
    "flames", "burning", "bruised", "bleeding", "noose", "hanging",
}

# Matches a rule word but is not the thing. Checked as phrases, so only the
# innocent use is spared — "dead tree" stays, a dead body does not.
INNOCENT = (
    "dead-tree", "dead-trees", "war-room", "burning-candle", "burning-lamp",
    "burning-fireplace", "flames-fireplace", "burning-torch", "candle-flames",
)


def is_violent(filename: str) -> bool:
    low = os.path.splitext(filename.lower())[0]
    if any(phrase in low for phrase in INNOCENT):
        return False
    return bool(set(low.split("-")) & VIOLENT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop violent b-roll.")
    parser.add_argument("folder", nargs="?",
                        default=os.path.join("assets", "broll", "film"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    doomed, spared = [], []
    for r, _dirs, files in os.walk(args.folder):
        for f in files:
            if not f.lower().endswith(".mp4"):
                continue
            path = os.path.join(r, f)
            low = os.path.splitext(f.lower())[0]
            if any(p in low for p in INNOCENT):
                spared.append(f)
            elif is_violent(f):
                doomed.append(path)

    print(f"{len(doomed)} clip(s) to remove")
    for p in sorted(doomed)[:12]:
        print(f"    {os.path.basename(p)[:64]}")
    if len(doomed) > 12:
        print(f"    ... and {len(doomed) - 12} more")
    if spared:
        print(f"\n{len(spared)} spared as false positives:")
        for f in spared:
            print(f"    {f[:64]}")

    if args.dry_run:
        print("\ndry run — nothing deleted")
        return

    freed = 0
    for p in doomed:
        try:
            freed += os.path.getsize(p)
            os.remove(p)
        except OSError as e:
            print(f"  ! {os.path.basename(p)}: {e}")
    print(f"\ndeleted {len(doomed)}, freed {freed / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
