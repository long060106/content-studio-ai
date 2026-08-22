"""
poster_renderer.py

Renders the statement posters: a photo, a two-part line of text, nothing else.

Two layouts, taken from how the genre actually looks:

  bleed   The setup runs edge to edge in enormous lowercase type across the
          upper third, and the payoff sits under its right end in serif italic.
          High contrast, usually red or white. The text is deliberately allowed
          to run off both margins — it reads as a headline, not a caption.

  stack   Setup in a modest weight, payoff directly beneath it in something much
          heavier and larger, left-aligned in a corner. Quieter, and it survives
          busier photographs.

The part that decides whether these look designed or thrown together is
placement. Text dropped in the middle of a photo covers the face; text in a
fixed corner lands on whatever happens to be there. So `_best_band()` scores
horizontal bands of the image on how *busy* they are (local variance) and how
far they sit from the brightest, most detailed region, then puts the text in
the calmest band. On a portrait with sky above the subject the text goes up; on
a landscape with a clear foreground it drops down.

Text is never drawn straight onto a photo without help — a soft shadow and,
where the underlying band is bright, a localised scrim, otherwise white or red
type disappears the moment the picture has a highlight in it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

# Instagram portrait and story/reel.
SIZES = {
    "portrait": (1080, 1350),
    "story": (1080, 1920),
    "square": (1080, 1080),
}

RED = (228, 30, 24)
WHITE = (255, 255, 255)
NEAR_BLACK = (14, 14, 16)

_HEAVY = [
    r"C:\Windows\Fonts\ariblk.ttf",              # Arial Black
    r"C:\Windows\Fonts\seguibl.ttf",             # Segoe UI Black
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_BOLD = [
    r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_SERIF_ITALIC = [
    r"C:\Windows\Fonts\georgiaz.ttf",            # Georgia Bold Italic
    r"C:\Windows\Fonts\timesbi.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
]


@dataclass
class Poster:
    setup: str
    payoff: str
    image: str
    out_path: str
    layout: str = "bleed"          # bleed | stack
    size: str = "portrait"
    colour: object = "auto"   # RED, WHITE, or "auto" to decide per photo
    signature: str = ""


def _font(candidates: Sequence[str], size: int):
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _fit_width(text: str, candidates: Sequence[str], target_width: int,
               max_size: int, min_size: int = 24):
    """Largest font size at which `text` fits `target_width`."""
    low, high, best = min_size, max_size, _font(candidates, min_size)
    while low <= high:
        mid = (low + high) // 2
        font = _font(candidates, mid)
        if font.getlength(text) <= target_width:
            best, low = font, mid + 1
        else:
            high = mid - 1
    return best


def _band_scores(img: Image.Image, bands: int = 6) -> list[float]:
    """Per-band 'busyness': edge energy plus brightness spread. Lower is calmer."""
    grey = img.convert("L").resize((180, int(180 * img.height / img.width)))
    edges = grey.filter(ImageFilter.FIND_EDGES)
    height = grey.height
    step = max(1, height // bands)

    scores = []
    for i in range(bands):
        top, bottom = i * step, min(height, (i + 1) * step)
        if bottom <= top:
            scores.append(float("inf"))
            continue
        edge_stat = ImageStat.Stat(edges.crop((0, top, grey.width, bottom)))
        tone_stat = ImageStat.Stat(grey.crop((0, top, grey.width, bottom)))
        scores.append(edge_stat.mean[0] * 2.0 + tone_stat.stddev[0])
    return scores


def _skin_fractions(img: Image.Image, bands: int) -> list[float]:
    """Rough share of skin-toned pixels per band.

    There's no face detection here — that would mean a new dependency and a
    model download — but text across someone's eyes is the one placement that
    always looks wrong, and a plain RGB skin-tone rule is enough to push the
    headline off a face and into the hair, background or below the chin. It is
    a heuristic: warm-toned backgrounds can read as skin, which costs a band
    that would have been fine rather than producing a bad placement.
    """
    small = img.convert("RGB").resize((120, max(bands, int(120 * img.height / img.width))))
    pixels = small.load()
    width, height = small.size
    step = max(1, height // bands)

    fractions = []
    for i in range(bands):
        top, bottom = i * step, min(height, (i + 1) * step)
        if bottom <= top:
            fractions.append(0.0)
            continue
        skin = total = 0
        for y in range(top, bottom):
            for x in range(width):
                r, g, b = pixels[x, y][:3]
                total += 1
                if (r > 95 and g > 40 and b > 20 and max(r, g, b) - min(r, g, b) > 15
                        and abs(r - g) > 15 and r > g and r > b):
                    skin += 1
        fractions.append(skin / max(total, 1))
    return fractions


def _best_band(img: Image.Image, bands: int = 6, prefer: str = "any",
               allowed: Optional[range] = None) -> int:
    """Index of the calmest, darkest band — where light text will actually read.

    Busyness alone isn't enough: a blown-out sky is perfectly calm and the worst
    possible place for white type, so brightness is scored too. `allowed`
    restricts the choice to a slice of the frame, which is how the bleed layout
    keeps its headline in the top third where the genre puts it.
    """
    busy = _band_scores(img, bands)
    skin = _skin_fractions(img, bands)
    grey = img.convert("L")
    step = max(1, img.height // bands)

    candidates = list(allowed) if allowed else list(range(bands))
    candidates = [i for i in candidates if 0 <= i < bands] or list(range(bands))

    best, best_score = candidates[0], float("inf")
    for i in candidates:
        luma = _band_luma(grey, i * step, (i + 1) * step)
        # Skin dominates the score: a band that's half face is disqualifying
        # however calm and dark it happens to be.
        score = busy[i] * 1.6 + luma * 0.5 + skin[i] * 260.0
        if prefer == "top":
            score += i * 4.0
        elif prefer == "bottom":
            score += (bands - 1 - i) * 4.0
        if score < best_score:
            best, best_score = i, score
    return best


def _band_luma(img: Image.Image, top: int, bottom: int) -> float:
    top, bottom = max(0, top), min(img.height, bottom)
    if bottom <= top:
        return 128.0
    return ImageStat.Stat(img.convert("L").crop((0, top, img.width, bottom))).mean[0]


def _scrim(img: Image.Image, top: int, bottom: int, strength: int = 130) -> Image.Image:
    """Darken a horizontal band, fading out at both edges so it isn't a stripe.

    The feather is capped rather than proportional: at a third of the band the
    ramp ate the whole scrim on short bands, so the text was left sitting on an
    almost untouched photo.
    """
    height = img.height
    top, bottom = max(0, top), min(height, bottom)
    if bottom <= top or strength <= 0:
        return img

    mask = Image.new("L", (1, height), 0)
    feather = max(1, min(110, (bottom - top) // 4))
    for y in range(top, bottom):
        edge = min(y - top, bottom - y)
        alpha = strength if edge >= feather else int(strength * (edge / feather))
        mask.putpixel((0, y), alpha)
    overlay = Image.new("RGB", img.size, NEAR_BLACK)
    return Image.composite(overlay, img, mask.resize(img.size))


def _luma(colour: tuple) -> float:
    r, g, b = colour[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _target_luma(colour: tuple) -> float:
    """How dark the background has to get for this text colour to read.

    Brightness contrast is what the eye resolves, not hue. Red is a mid-luma
    colour (~88), so red type over a background of luma 80 is nearly invisible
    however different the two look described in words — it needs a genuinely
    dark backdrop, which is why the posters in this genre put red over night
    skies and arena roofs. White has luma 255 and is far more forgiving.
    """
    return max(25.0, min(150.0, _luma(colour) * 0.4))


def _auto_scrim(img: Image.Image, top: int, bottom: int,
                colour: tuple = WHITE) -> Image.Image:
    """Darken only as much as this photo and this text colour need.

    A night shot gets left alone; a snowfield gets pushed down hard.
    """
    target = _target_luma(colour)
    luma = _band_luma(img, top, bottom)
    if luma <= target:
        return img
    strength = int(min(205, 255 * (luma - target) / max(luma, 1.0)))
    return _scrim(img, top, bottom, strength=strength)


def _pick_colour(img: Image.Image, top: int, bottom: int, requested) -> tuple:
    """Resolve colour="auto": red only where it can actually be read."""
    if requested != "auto":
        return requested
    # Red survives on a band that's already dark, or close enough that a
    # moderate scrim gets it there without flattening the whole photograph.
    return RED if _band_luma(img, top, bottom) < 115 else WHITE


def _shadowed(draw: ImageDraw.ImageDraw, xy, text, font, fill, blur_layer=None, offset=4):
    x, y = xy
    draw.text((x + offset, y + offset), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def render_poster(poster: Poster) -> str:
    width, height = SIZES.get(poster.size, SIZES["portrait"])

    with Image.open(poster.image) as raw:
        img = _cover(raw.convert("RGB"), width, height)

    margin = int(width * 0.055)
    setup = (poster.setup or "").strip()
    payoff = (poster.payoff or "").strip()

    if poster.layout == "bleed":
        # The headline is the picture's loudest element: as wide as the frame,
        # bleeding slightly past both margins the way the reference does.
        head_font = _fit_width(setup or payoff, _HEAVY, int(width * 1.02), max_size=460)
        head_text = setup or payoff
        head_w = head_font.getlength(head_text)
        ascent, descent = head_font.getmetrics()
        head_h = ascent + descent

        # The headline belongs in the upper third — that's the convention, and
        # on a normal portrait it clears the subject. On a tight face crop the
        # whole upper third *is* face, so rather than stamping type across
        # someone's eyes, fall back to the emptiest band anywhere in the frame
        # (usually below the chin, which the genre also does).
        skin = _skin_fractions(img, 8)
        band = _best_band(img, bands=8, prefer="top", allowed=range(0, 3))
        if skin[band] > 0.35:
            band = _best_band(img, bands=8)
        # The draw call lifts the headline by 0.18 of its ascent to sit the
        # glyphs optically where you'd expect, so the top margin has to allow
        # for that or the type clips against the edge of the frame.
        top_limit = int(height * 0.06 + ascent * 0.18)
        y = max(top_limit, min(int(band * height / 8), int(height * 0.72)))
        x = (width - head_w) / 2

        sub_font = _fit_width(payoff, _SERIF_ITALIC, int(width * 0.5), max_size=118) if (setup and payoff) else None
        block_bottom = y + head_h * (1.35 if sub_font else 1.1)
        band_top, band_bottom = int(y - head_h * 0.2), int(block_bottom)
        colour = _pick_colour(img, band_top, band_bottom, poster.colour)
        img = _auto_scrim(img, band_top, band_bottom, colour)

        draw = ImageDraw.Draw(img)
        _shadowed(draw, (x, y - ascent * 0.18), head_text, head_font,
                  colour, offset=max(3, head_font.size // 40))

        if sub_font:
            sub_w = sub_font.getlength(payoff)
            sub_x = min(x + head_w - sub_w, width - margin - sub_w)
            sub_y = y + head_h * 0.82
            _shadowed(draw, (sub_x, sub_y), payoff, sub_font, colour, offset=3)
            text_bottom = sub_y + sub_font.size
        else:
            text_bottom = y + head_h
    else:
        # stack: quiet setup, heavy payoff underneath, left-aligned.
        setup_font = _fit_width(setup, _BOLD, int(width * 0.62), max_size=74) if setup else None
        payoff_font = _fit_width(payoff, _HEAVY, int(width * 0.80), max_size=132)

        block_h = (setup_font.size * 1.35 if setup_font else 0) + payoff_font.size * 1.3
        band = _best_band(img, bands=6)
        y = int(band * height / 6) + int(height * 0.02)
        y = max(margin, min(y, height - int(block_h) - margin))

        colour = _pick_colour(img, int(y), int(y + block_h), poster.colour)
        if colour == RED:
            colour = WHITE  # the stack layout is a white-type layout
        img = _auto_scrim(img, int(y - height * 0.025), int(y + block_h + height * 0.025), colour)
        draw = ImageDraw.Draw(img)

        if setup_font:
            _shadowed(draw, (margin, y), setup, setup_font, WHITE, offset=3)
            y += setup_font.size * 1.25
        _shadowed(draw, (margin, y), payoff, payoff_font, WHITE, offset=4)
        text_bottom = y + payoff_font.size

    if poster.signature:
        sig_font = _font(_BOLD, 26)
        sig_w = sig_font.getlength(poster.signature)
        sig_x = width - margin - sig_w if poster.layout == "bleed" else margin
        draw = ImageDraw.Draw(img)
        _shadowed(draw, (sig_x, min(text_bottom + 14, height - 60)),
                  poster.signature, sig_font, colour if poster.layout == "bleed" else WHITE,
                  offset=2)

    parent = os.path.dirname(os.path.abspath(poster.out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    img.save(poster.out_path, "PNG")
    return poster.out_path


def render_set(
    statements: list,
    images: list[str],
    out_dir: str,
    layout: str = "bleed",
    size: str = "portrait",
    colour: object = "auto",
    signature: str = "",
) -> list[str]:
    """Render one poster per statement, cycling through the images given."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, statement in enumerate(statements, start=1):
        if not images:
            break
        image = images[(i - 1) % len(images)]
        setup = getattr(statement, "setup", "") or statement.get("setup", "")
        payoff = getattr(statement, "payoff", "") or statement.get("payoff", "")
        out_path = os.path.join(out_dir, f"poster_{i:02d}.png")
        render_poster(Poster(
            setup=setup, payoff=payoff, image=image, out_path=out_path,
            layout=layout, size=size, colour=colour, signature=signature,
        ))
        paths.append(out_path)
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render one statement poster.")
    parser.add_argument("image")
    parser.add_argument("out_path")
    parser.add_argument("--setup", default="")
    parser.add_argument("--payoff", required=True)
    parser.add_argument("--layout", default="bleed", choices=["bleed", "stack"])
    parser.add_argument("--size", default="portrait", choices=list(SIZES))
    parser.add_argument("--white", action="store_true", help="white text instead of red")
    parser.add_argument("--signature", default="")
    args = parser.parse_args()

    print(render_poster(Poster(
        setup=args.setup, payoff=args.payoff, image=args.image, out_path=args.out_path,
        layout=args.layout, size=args.size,
        colour=WHITE if args.white else RED, signature=args.signature,
    )))
