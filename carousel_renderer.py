"""
carousel_renderer.py

Renders Instagram carousel slides as PNG images using Pillow. Kept
intentionally simple — solid background, centered wrapped text — so it
works without a browser/headless-rendering dependency.

Typography hierarchy (consistent across every slide):
  headline (large, bold, white)
  flow     (medium, bold, muted accent — optional minimal text diagram)
  subtext  (medium, light gray — optional supporting line)
  source   (small, dim — optional attribution/citation)
  slide number (small, dim, bottom center)

Tries to use a real system font (Arial on Windows/Mac, DejaVu Sans on Linux)
for readable output, falling back to Pillow's built-in bitmap font (which
looks rough but always works) if none is found.

Slides can also be rendered over a photo (`background=`). The image is
cover-cropped to the slide, then darkened under a gradient scrim — heaviest
where the text sits — because white text over an unmodified photo is
unreadable the moment the photo has a bright patch. Text also picks up a
soft shadow for the same reason.
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Standard Instagram portrait carousel size (4:5 ratio)
SLIDE_WIDTH = 1080
SLIDE_HEIGHT = 1350

BACKGROUND_COLOR = (17, 17, 17)       # near-black
HEADLINE_COLOR = (255, 255, 255)      # white
FLOW_COLOR = (150, 170, 255)          # muted accent, distinguishes it from prose
SUBTEXT_COLOR = (180, 180, 180)       # light gray
SOURCE_COLOR = (120, 120, 120)        # dim gray, smaller — citation-style
SLIDE_NUMBER_COLOR = (90, 90, 90)     # dimmest, bottom center

_FONT_CANDIDATES_BOLD = [
    r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONT_CANDIDATES_REGULAR = [
    r"C:\Windows\Fonts\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(candidates: list[str], size: int):
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older Pillow versions don't accept a size argument here.
        return ImageFont.load_default()


def _draw_wrapped_text(draw, text, font, max_width, y, fill, line_spacing=1.3, shadow=False) -> float:
    """Wraps text to fit max_width, draws it centered, returns the y position after the text."""
    avg_char_width = font.getlength("x") or 10
    chars_per_line = max(1, int(max_width / avg_char_width))
    wrapped = textwrap.wrap(text, width=chars_per_line) or [text]

    line_height = int(getattr(font, "size", 32) * line_spacing)
    for line in wrapped:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = (SLIDE_WIDTH - line_width) / 2
        if shadow:
            draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _cover_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Fill the frame without distorting — same idea as CSS `object-fit: cover`."""
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _photo_background(path: str) -> Optional[Image.Image]:
    """Load a photo, cover-crop it, and lay a readability scrim over the top."""
    try:
        with Image.open(path) as raw:
            img = _cover_crop(raw.convert("RGB"), SLIDE_WIDTH, SLIDE_HEIGHT)
    except (OSError, ValueError):
        return None

    # A touch of blur keeps busy photos from competing with the headline.
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    # Vertical gradient: dark at the edges, darkest at the bottom where the
    # subtext, source and slide number sit.
    scrim = Image.new("L", (1, SLIDE_HEIGHT))
    for y in range(SLIDE_HEIGHT):
        pos = y / SLIDE_HEIGHT
        if pos < 0.25:
            alpha = 150 - int(40 * (pos / 0.25))     # 150 -> 110
        elif pos < 0.7:
            alpha = 110 + int(40 * ((pos - 0.25) / 0.45))  # 110 -> 150
        else:
            alpha = 150 + int(70 * ((pos - 0.7) / 0.3))    # 150 -> 220
        scrim.putpixel((0, y), alpha)

    overlay = Image.new("RGB", (SLIDE_WIDTH, SLIDE_HEIGHT), BACKGROUND_COLOR)
    return Image.composite(overlay, img, scrim.resize((SLIDE_WIDTH, SLIDE_HEIGHT)))


def render_slide(
    headline: str,
    slide_number: int,
    total_slides: int,
    output_path: str,
    subtext: Optional[str] = None,
    flow: Optional[str] = None,
    source: Optional[str] = None,
    background: Optional[str] = None,
) -> str:
    img = None
    if background and os.path.exists(background):
        img = _photo_background(background)
    on_photo = img is not None
    if img is None:
        img = Image.new("RGB", (SLIDE_WIDTH, SLIDE_HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    headline_font = _load_font(_FONT_CANDIDATES_BOLD, 64)
    flow_font = _load_font(_FONT_CANDIDATES_BOLD, 34)
    subtext_font = _load_font(_FONT_CANDIDATES_REGULAR, 36)
    source_font = _load_font(_FONT_CANDIDATES_REGULAR, 24)
    number_font = _load_font(_FONT_CANDIDATES_REGULAR, 28)

    max_text_width = SLIDE_WIDTH - 160  # side margins

    y = SLIDE_HEIGHT * 0.36  # roughly vertically centered start point
    y = _draw_wrapped_text(
        draw, headline, headline_font, max_text_width, y, HEADLINE_COLOR, shadow=on_photo
    )

    if flow:
        y += 28
        y = _draw_wrapped_text(
            draw, flow.upper(), flow_font, max_text_width, y, FLOW_COLOR, shadow=on_photo
        )

    if subtext:
        y += 20
        # Plain white reads better than grey once there's a photo behind it.
        subtext_colour = (235, 235, 235) if on_photo else SUBTEXT_COLOR
        y = _draw_wrapped_text(
            draw, subtext, subtext_font, max_text_width, y, subtext_colour, shadow=on_photo
        )

    if source:
        # Small citation line, placed a fixed distance above the slide number
        # rather than immediately after the content block, so it reads as a
        # footnote rather than part of the main text flow.
        source_y = SLIDE_HEIGHT - 140
        bbox = draw.textbbox((0, 0), source, font=source_font)
        source_width = bbox[2] - bbox[0]
        draw.text(
            ((SLIDE_WIDTH - source_width) / 2, source_y),
            source,
            font=source_font,
            fill=SOURCE_COLOR,
        )

    number_text = f"{slide_number} / {total_slides}"
    bbox = draw.textbbox((0, 0), number_text, font=number_font)
    number_width = bbox[2] - bbox[0]
    draw.text(
        ((SLIDE_WIDTH - number_width) / 2, SLIDE_HEIGHT - 80),
        number_text,
        font=number_font,
        fill=SLIDE_NUMBER_COLOR,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def render_carousel(slides: list[dict], output_dir: str) -> list[str]:
    """Renders a full carousel.

    slides = [{"headline": str, "subtext": str|None, "flow": str|None,
               "source": str|None, "background": str|None}, ...]

    `background` is an optional path to a photo for that slide; without it the
    slide renders on the plain near-black background as before.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(slides)
    paths = []
    for i, slide in enumerate(slides, start=1):
        path = os.path.join(output_dir, f"slide_{i:02d}.png")
        render_slide(
            headline=slide.get("headline", ""),
            subtext=slide.get("subtext"),
            flow=slide.get("flow"),
            source=slide.get("source"),
            background=slide.get("background"),
            slide_number=i,
            total_slides=total,
            output_path=path,
        )
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 3:
        print("Usage: python carousel_renderer.py <path_to_carousel.json> <output_dir>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    paths = render_carousel(data.get("slides", []), sys.argv[2])
    print(f"Rendered {len(paths)} slides:")
    for p in paths:
        print(f"  {p}")
