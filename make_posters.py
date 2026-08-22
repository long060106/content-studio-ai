"""
make_posters.py

Writes a batch of statements and renders each one as a poster.

    python make_posters.py --count 8
    python make_posters.py --count 6 --theme discipline --layout stack
    python make_posters.py --from-talk LNHBMFCzznE      # reuse a talk's moments

Images come from the asset library, matched to each statement's own visual
keywords, so a statement about training gets a gym rather than a lake. With a
stock key set it will fetch what's missing; without one it uses what's already
in `assets/image/`.

Output lands in `output/posters/<run>/`, one PNG per statement plus a
`statements.json` recording the text, the image used and its licence.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import asset_library
from poster_renderer import RED, WHITE, Poster, render_poster
from statement_writer import Statement, write_statements


def pick_image(statement: Statement, used: set[str], fetch: bool) -> tuple[str | None, dict]:
    keywords = list(statement.visual_keywords) + [statement.theme]
    asset = asset_library.pick("image", keywords, exclude=used)

    if asset is None and fetch and statement.visual_keywords:
        asset_library.fetch_stock(statement.visual_keywords[0], kind="image", count=2)
        asset = asset_library.pick("image", keywords, exclude=used)

    if asset is None:
        return None, {}
    used.add(asset.id)
    return asset.abs_path, {
        "path": asset.path, "source": asset.source,
        "credit": asset.credit, "licence": asset.licence,
    }


def make_posters(
    count: int = 8,
    theme: str | None = None,
    layout: str = "bleed",
    size: str = "portrait",
    colour: object = "auto",
    signature: str = "",
    out_dir: str | None = None,
    fetch: bool = True,
    source_quotes: list[str] | None = None,
) -> list[dict]:
    out_dir = out_dir or os.path.join("output", "posters", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"→ Writing {count} statements...")
    statements = write_statements(count=count, theme=theme, source_quotes=source_quotes)
    print(f"  ✓ {len(statements)} written")

    asset_library.ensure_dirs()
    asset_library.scan()

    used: set[str] = set()
    records: list[dict] = []

    for i, statement in enumerate(statements, start=1):
        image, credit = pick_image(statement, used, fetch)
        if image is None:
            print(f"  ⚠ {i}. no image available for {statement.theme} — skipped")
            continue

        out_path = os.path.join(out_dir, f"poster_{i:02d}_{statement.theme or 'statement'}.png")
        try:
            render_poster(Poster(
                setup=statement.setup, payoff=statement.payoff, image=image,
                out_path=out_path, layout=layout, size=size,
                colour=colour, signature=signature,
            ))
        except Exception as e:
            print(f"  ⚠ {i}. render failed: {e}")
            continue

        print(f"  ✓ {i}. {statement.setup} {statement.payoff}")
        record = statement.to_dict()
        record.update({"poster": out_path, "image": credit, "layout": layout})
        records.append(record)

    with open(os.path.join(out_dir, "statements.json"), "w", encoding="utf-8") as f:
        json.dump({"statements": records}, f, indent=2, ensure_ascii=False)
    asset_library.write_credits()

    print(f"\n✓ Done. {len(records)} poster(s) in {out_dir}")
    return records


def quotes_from_talk(video_id: str, output_dir: str = "output") -> list[str]:
    """Pull the hooks and quotes a shorts run already found for this talk."""
    index_path = os.path.join(output_dir, video_id, "shorts", "index.json")
    if not os.path.isfile(index_path):
        return []
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    quotes = []
    for short in data.get("shorts", []):
        quotes += [q for q in (short.get("quote"), short.get("hook")) if q]
    return quotes


def main() -> None:
    parser = argparse.ArgumentParser(description="Write and render statement posters.")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--theme")
    parser.add_argument("--layout", default="bleed", choices=["bleed", "stack"])
    parser.add_argument("--size", default="portrait", choices=["portrait", "story", "square"])
    parser.add_argument("--colour", default="auto", choices=["auto", "red", "white"])
    parser.add_argument("--signature", default="")
    parser.add_argument("--out-dir")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--from-talk", help="video id whose moments should seed the statements")
    args = parser.parse_args()

    colour = {"auto": "auto", "red": RED, "white": WHITE}[args.colour]
    quotes = quotes_from_talk(args.from_talk) if args.from_talk else None
    if args.from_talk and not quotes:
        print(f"  · no shorts found for {args.from_talk}; writing fresh statements instead")

    make_posters(
        count=args.count, theme=args.theme, layout=args.layout, size=args.size,
        colour=colour, signature=args.signature, out_dir=args.out_dir,
        fetch=not args.no_fetch, source_quotes=quotes,
    )


if __name__ == "__main__":
    main()
