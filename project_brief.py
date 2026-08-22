"""
project_brief.py

Builds a content brief from photographs instead of from a talk transcript.

This is the image-only customer path: someone arrives with a folder of photos
and no video at all, and still needs a brief, a carousel and captions. The work
can be anything photographable — furniture, ceramics, baking, nails, tailoring,
detailing, art. The domain is read from the pictures rather than assumed.

Why this exists: every generator in this project reads from a `ContentBrief`.
`content_brief.py` produces one from a transcript, which is useless when the
source material is a pile of workshop photos. This module produces the same
object by *looking at the images*, so `captions_generator`, `carousel_generator`,
`twitter_thread_generator`, `blog_generator` and the poster renderer all work on
a maker's projects with no further changes.

What it actually does with the photos:

  - Identifies each piece: what it is, what it's made of, how it was made,
    and the one detail worth pointing a camera at. Specifics are the whole
    game whatever the craft — "quartersawn white oak, through-tenons, hand-
    rubbed oil" earns enquiries, and so does "48-hour cold ferment, 82%
    hydration, semolina dusting". "Beautiful handmade table" earns nothing.
  - Sorts shots into build progress vs finished, then orders them into a video.
    A build-up → reveal runs far better than a gallery of finished pieces,
    because the process is the reason anyone stops scrolling.
  - Writes to a stated goal. For commission work that means the craft detail
    carries the credibility and every post ends with an actual way to enquire.

Images are downsampled before sending. Telling oak from pine, or a proper
crumb from a dense one, doesn't need full resolution, and full-size photos cost
thousands of tokens each.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from content_brief import ContentBrief

load_dotenv()

MODEL = "claude-opus-5"

# Big enough to read texture, technique and finish; small enough to send a dozen.
IMAGE_EDGE = 720

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}
MAX_IMAGES = 14

GOALS = {
    "commissions": (
        "They want paid work. Lead with craft credibility — materials, "
        "technique, the decisions a buyer wouldn't know to ask about. Every "
        "piece of copy ends with a concrete way to enquire. Never beg for "
        "engagement; confidence in the work is what sells it."
    ),
    "portfolio": (
        "This is a record of their work, not a sales pitch. Quietly proud. "
        "Describe the piece and the making; ask for nothing."
    ),
    "community": (
        "They're talking to other people who do this craft. Go technical — "
        "tools, materials, process, timings, what went wrong and what they'd "
        "do differently."
    ),
    "neutral": (
        "Keep it usable for any purpose. Describe the work well and leave the "
        "call to action generic enough to swap per post."
    ),
}

SYSTEM_PROMPT = """You are a content strategist who works with people who make things. You are looking at photographs of one maker's work and writing the brief that every caption, carousel and post for it will be built from.

FIRST, work out what craft you are actually looking at. It could be furniture, ceramics, baking, nail art, tailoring, car detailing, leatherwork, painting, floristry, metalwork — anything. Name it in `domain`, and let that decide which details matter. Do not force one craft's vocabulary onto another.

Then look at the photographs properly and say what you actually see:
- The piece: what it is, its form, its proportions.
- Materials: what it appears to be made of, in the terms that craft uses — timber species and cut, clay body, flour and hydration, fabric and weave, pigment and ground. Say "looks like" rather than inventing certainty.
- Technique: how it was made, as far as the photo shows — the joinery, the throwing and trimming, the lamination, the stitching, the layering.
- Finish: the surface and how it was treated — oil, wax, glaze, varnish, polish, plating, styling, raw.
- Craft signals a customer wouldn't notice but another maker would. These are the details that prove competence to the audience worth having.

Rules that matter:
- SPECIFICS ONLY. "Beautiful handmade table" is worthless — it could describe anything and sells nothing. "Through-tenons wedged in contrasting walnut", or "48-hour cold ferment, blistered open crumb", is what makes someone stop.
- If you cannot tell something from the photo, say so in `uncertain` rather than guessing. A confident wrong claim about material or technique destroys credibility with exactly the audience worth having, and they will be asked about it in the comments.
- Never invent a backstory, a client, a timescale, or a price.
- Process shots are the most valuable thing here. Order the sequence so process comes before reveal — that is what holds attention, in every craft.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "domain": string,              // the craft itself, e.g. "furniture making", "baking"
  "pieces": [
    {
      "name": string,            // e.g. "oak dining table", "sourdough boule"
      "material": string,        // what it appears to be made of
      "technique": string,       // how it was made, as far as the photo shows
      "finish": string,          // surface / treatment / presentation, or unclear
      "standout": string,        // the one detail worth pointing a camera at
      "image_indexes": [number]  // 1-based indexes of the photos showing this piece
    }
  ],
  "summary": string,             // 2-3 sentences on the body of work as a whole
  "target_audience": string,     // who would commission this
  "tone": string,                // the voice the copy should use
  "hooks": [string],             // 3 opening lines for a short-form video, specific to these pieces
  "title_suggestions": [string], // 3 titles for the video
  "key_points": [                // 4-6 things worth saying, best first
    { "point": string, "supporting_quote": string }   // supporting_quote = a concrete
                                                      // observable detail, not a made-up quotation
  ],
  "shot_order": [                // the video, in order
    { "image_index": number, "role": string, "on_screen_text": string }
      // role: "hook" | "process" | "detail" | "reveal" | "close"
  ],
  "topics": [string],            // 4-6 short tags for categorisation
  "call_to_action": string,      // what the viewer should do
  "uncertain": [string]          // anything you could not determine from the photos
}
"""


@dataclass
class Piece:
    name: str
    material: str = ""
    technique: str = ""
    finish: str = ""
    standout: str = ""
    image_indexes: list[int] = field(default_factory=list)


@dataclass
class ProjectBrief:
    project_id: str
    domain: str
    pieces: list[Piece]
    summary: str
    target_audience: str
    tone: str
    hooks: list[str]
    title_suggestions: list[str]
    key_points: list[dict]
    shot_order: list[dict]
    topics: list[str]
    call_to_action: str
    uncertain: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_content_brief(self) -> ContentBrief:
        """The same object the transcript pipeline produces.

        This is the point of the module: once the photos become a ContentBrief,
        every existing generator works on them unchanged.
        """
        return ContentBrief(
            video_id=self.project_id,
            title_suggestions=self.title_suggestions,
            summary=self.summary,
            target_audience=self.target_audience,
            tone=self.tone,
            key_points=self.key_points,
            hooks=self.hooks,
            # No spoken quotes exist for a photo set; the standout craft
            # details play the same role for downstream copy.
            notable_quotes=[p.standout for p in self.pieces if p.standout],
            call_to_action=self.call_to_action,
            topics=self.topics,
        )


def find_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No such folder: {folder}")
    found = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    ]
    if not found:
        raise FileNotFoundError(f"No images found in {folder}")
    return found


def _encode(path: str, edge: int = IMAGE_EDGE) -> Optional[tuple[str, str]]:
    from PIL import Image

    try:
        with Image.open(path) as raw:
            img = raw.convert("RGB")
            img.thumbnail((edge, edge), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=80)
    except (OSError, ValueError):
        return None
    return "image/jpeg", base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def analyse(
    folder: str,
    goal: str = "commissions",
    notes: str = "",
    api_key: Optional[str] = None,
    model: str = MODEL,
) -> ProjectBrief:
    paths = find_images(folder)[:MAX_IMAGES]

    content: list = [{
        "type": "text",
        "text": (
            f"These are {len(paths)} photographs of one maker's projects.\n\n"
            f"Goal: {GOALS.get(goal, GOALS['neutral'])}\n"
            + (f"\nWhat he's told us about them: {notes}\n" if notes else "")
            + "\nThe photos are numbered in the order shown. Refer to them by "
            "those numbers in `image_indexes` and `shot_order`.\n\n"
            + SCHEMA_DESCRIPTION
        ),
    }]

    used: list[str] = []
    for i, path in enumerate(paths, start=1):
        encoded = _encode(path)
        if not encoded:
            continue
        media_type, data = encoded
        used.append(path)
        content.append({"type": "text", "text": f"Photo {len(used)}: {os.path.basename(path)}"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })

    if not used:
        raise ValueError(f"None of the images in {folder} could be read.")

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The request was declined by safety classifiers.")

    raw = "".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    ).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start : end + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{raw[:2000]}") from e

    project_id = re.sub(r"[^a-z0-9]+", "_", os.path.basename(os.path.abspath(folder)).lower()).strip("_")

    return ProjectBrief(
        project_id=project_id or "project",
        domain=str(data.get("domain", "")).strip(),
        pieces=[
            Piece(
                name=str(p.get("name", "")).strip(),
                material=str(p.get("material", "")).strip(),
                technique=str(p.get("technique", "")).strip(),
                finish=str(p.get("finish", "")).strip(),
                standout=str(p.get("standout", "")).strip(),
                image_indexes=[int(i) for i in p.get("image_indexes", []) if str(i).isdigit()],
            )
            for p in data.get("pieces", [])
        ],
        summary=str(data.get("summary", "")).strip(),
        target_audience=str(data.get("target_audience", "")).strip(),
        tone=str(data.get("tone", "")).strip(),
        hooks=[str(h).strip() for h in data.get("hooks", []) if str(h).strip()],
        title_suggestions=[str(t).strip() for t in data.get("title_suggestions", []) if str(t).strip()],
        key_points=data.get("key_points", []),
        shot_order=data.get("shot_order", []),
        topics=[str(t).strip() for t in data.get("topics", []) if str(t).strip()],
        call_to_action=str(data.get("call_to_action", "")).strip(),
        uncertain=[str(u).strip() for u in data.get("uncertain", []) if str(u).strip()],
        images=used,
    )


def to_markdown(brief: ProjectBrief) -> str:
    lines = [f"# Content brief — {brief.project_id.replace('_', ' ')}", ""]
    lines += [brief.summary, ""]
    lines += [f"**Who it's for:** {brief.target_audience}", "",
              f"**Voice:** {brief.tone}", ""]

    lines += ["## The pieces", ""]
    for piece in brief.pieces:
        lines.append(f"### {piece.name}")
        for label, value in (("Material", piece.material), ("Technique", piece.technique),
                             ("Finish", piece.finish), ("Worth filming", piece.standout)):
            if value:
                lines.append(f"- **{label}:** {value}")
        if piece.image_indexes:
            lines.append(f"- **Photos:** {', '.join(str(i) for i in piece.image_indexes)}")
        lines.append("")

    lines += ["## Hooks (pick one to open with)", ""]
    lines += [f"{i}. {h}" for i, h in enumerate(brief.hooks, 1)] + [""]

    lines += ["## Video, in order", "", "| # | Photo | Role | On screen |", "|---|---|---|---|"]
    for i, shot in enumerate(brief.shot_order, 1):
        lines.append(
            f"| {i} | {shot.get('image_index', '')} | {shot.get('role', '')} | "
            f"{str(shot.get('on_screen_text', '')).replace('|', '/')} |"
        )
    lines.append("")

    lines += ["## Worth saying", ""]
    for point in brief.key_points:
        lines.append(f"- **{point.get('point', '')}**")
        if point.get("supporting_quote"):
            lines.append(f"  - {point['supporting_quote']}")
    lines.append("")

    lines += [f"## Call to action", "", brief.call_to_action, ""]

    if brief.uncertain:
        lines += ["## Check these before posting", "",
                  "Read off the photos and not certain — confirm with him rather "
                  "than publishing a guess:", ""]
        lines += [f"- {u}" for u in brief.uncertain] + [""]

    lines += [f"*Topics: {', '.join(brief.topics)}*", ""]
    return "\n".join(lines)


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Build a content brief from photos of physical projects."
    )
    parser.add_argument("folder", help="folder of photos")
    parser.add_argument("--goal", default="commissions", choices=sorted(GOALS))
    parser.add_argument("--notes", default="", help="anything the customer has told you about the work")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--no-carousel", action="store_true",
                        help="skip the carousel and captions; write the brief only")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY not set. Add it to .env first.")
        sys.exit(1)

    print(f"→ Reading photos from {args.folder}")
    brief = analyse(args.folder, goal=args.goal, notes=args.notes, model=args.model)
    print(f"  ✓ {len(brief.images)} photo(s) analysed, {len(brief.pieces)} piece(s) identified")
    if brief.domain:
        print(f"  ✓ Craft: {brief.domain}")

    out_dir = os.path.join(args.output_dir, "projects", brief.project_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "project_brief.json"), "w", encoding="utf-8") as f:
        json.dump(brief.to_dict(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "brief.json"), "w", encoding="utf-8") as f:
        f.write(brief.to_content_brief().to_json())
    with open(os.path.join(out_dir, "brief.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(brief))

    content_brief = brief.to_content_brief()
    written = ["project_brief.json", "brief.json", "brief.md"]

    if not args.no_carousel:
        # The customer's own photographs are the slide backgrounds — nothing is
        # fetched from a stock library here. They photographed the work, and
        # that IS the asset; a stock photo would be a downgrade.
        from carousel_generator import generate_carousel
        from carousel_renderer import render_carousel
        from captions_generator import generate_captions

        title = brief.title_suggestions[0] if brief.title_suggestions else ""

        print("→ Writing the carousel...")
        try:
            carousel = generate_carousel(content_brief, video_title=title)
            photos = brief.images
            slides = []
            for i, slide in enumerate(carousel.slides):
                slides.append({
                    "headline": slide.headline,
                    "subtext": slide.subtext,
                    "flow": slide.flow,
                    "source": slide.source,
                    # Cycle through the photos so each card shows different work.
                    "background": photos[i % len(photos)] if photos else None,
                })
            carousel_dir = os.path.join(out_dir, "carousel")
            paths = render_carousel(slides, carousel_dir)
            with open(os.path.join(carousel_dir, "carousel.json"), "w", encoding="utf-8") as f:
                f.write(carousel.to_json())
            print(f"  ✓ {len(paths)} slide(s) rendered")
            written.append("carousel/")
        except Exception as e:
            print(f"  ⚠ Carousel failed: {e}")

        print("→ Writing captions...")
        try:
            caption_set = generate_captions(content_brief)
            with open(os.path.join(out_dir, "captions.json"), "w", encoding="utf-8") as f:
                f.write(caption_set.to_json())
            with open(os.path.join(out_dir, "captions.txt"), "w", encoding="utf-8") as f:
                f.write(caption_set.to_text())
            print(f"  ✓ {len(caption_set.captions)} caption variant(s)")
            written += ["captions.json", "captions.txt"]
        except Exception as e:
            print(f"  ⚠ Captions failed: {e}")

    print(f"\n✓ Saved to {out_dir}")
    for name in written:
        print(f"    {os.path.join(out_dir, name)}")

    for piece in brief.pieces:
        print(f"\n  {piece.name}")
        print(f"    {piece.material} · {piece.technique}")
    if brief.uncertain:
        print("\n  ⚠ Confirm before posting:")
        for u in brief.uncertain:
            print(f"    - {u}")


if __name__ == "__main__":
    main()
