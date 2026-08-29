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

# Whole films whose footage is unusable here, by clip prefix.
#
# Keyword filtering works on what a description says, and a description can be
# accurate and still miss the point entirely. A clip from The Substance was
# named `figure-sitting-bathroom-fluorescent-light` — a fair account of the
# composition, and nothing in it hints at the nudity and body horror actually
# on screen. It went into a finished short.
#
# There was a warning and it was not acted on: the namer *refused* to describe
# one frame from that film, which was evidence about the source rather than a
# one-off glitch. When a model declines to look at a film, that film does not
# belong in a library for motivational shorts.
#
# So horror sources are excluded whole. A per-clip rule cannot see what a
# per-clip description leaves out.
BLOCKED_FILMS = (
    # Horror sources. The Substance put nudity and body horror into a finished
    # short behind the description `figure-sitting-bathroom-fluorescent-light`.
    "subs-", "nosf-", "28yl-", "alsg-",
    # Films where graphic violence is a signature rather than an occasional
    # scene. RoboCop put an extreme close-up of a bloodied eye into a short,
    # described as `close-up-hand-dark-object` — the second time a composition
    # was recorded accurately while what made it unusable went unmentioned.
    #
    # Two escapes of the same shape is the argument for judging the source
    # instead of the shot. A description is written by looking at a frame for a
    # moment; whether a film is full of gore is known before a single frame is
    # cut.
    "rbcp-", "sinn-", "sqg2-", "pngn-", "dda1-", "dda2-",
)

# Matches a rule word but is not the thing. Checked as phrases, so only the
# innocent use is spared — "dead tree" stays, a dead body does not.
INNOCENT = (
    "dead-tree", "dead-trees", "war-room", "burning-candle", "burning-lamp",
    "burning-fireplace", "flames-fireplace", "burning-torch", "candle-flames",
)


# Clips that are a card rather than a shot: opening titles, credits, a logo.
#
# One of these put the name of another series across a finished short in
# letters a foot high. They are footage in the sense that they move, and they
# are useless here — the account's own captions are the only text that belongs
# on screen.
#
# Matched as whole words, which is the entire reason this is a separate set:
# "texture" and "textured" both contain "text", and two perfectly good
# close-ups were nearly deleted for it.
CARDS = {
    "title", "titles", "credits", "logo", "subtitle", "subtitles",
    "caption", "captions", "typography", "lettering", "watermark", "text",
}


def is_violent(filename: str) -> bool:
    low = os.path.splitext(filename.lower())[0]
    if set(low.split("-")) & CARDS:
        return True
    # The film is checked before the words, because the words are exactly what
    # failed: a blocked film's clip can carry a perfectly innocent description.
    if low.startswith(BLOCKED_FILMS):
        return True
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
