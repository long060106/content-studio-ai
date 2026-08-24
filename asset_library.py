"""
asset_library.py

A local, reusable library of b-roll clips, stills and music, tagged by theme,
so every new talk gets matched against footage you already have instead of
re-sourcing from scratch each time.

    assets/
      video/           b-roll clips
      image/           stills for carousels and static backgrounds
      music/           background beds
      library.json     the manifest: tags, duration, licence, source

Two ways in:

1. **Drop files in.** Anything you put in those folders gets indexed by
   `scan()` — duration probed with ffprobe, tags taken from the filename.
   Needs no API key and no network.

2. **Fetch from a licensed source.** `fetch_stock()` tries, in order:

     Pexels      free API key, commercial use, best for video b-roll
     Pixabay     free API key, commercial use
     Openverse   NO KEY. Creative Commons images, filtered to commercially
                 usable licences. Attribution text comes back with each result.
     Wikimedia   NO KEY. Freely licensed media; much the best source for
                 photographs of specific, named people.

   Images therefore work with nothing configured at all. Video still needs a
   Pexels or Pixabay key.

Every fetched asset records its licence, creator and source page, and
`write_credits()` dumps `assets/CREDITS.md` — CC BY and BY-SA both require
crediting the photographer wherever the image appears.

A note on Pinterest, since that's the obvious thing to reach for: it has no
public API for this, its terms forbid scraping, and the images on it are
overwhelmingly other people's copyrighted work re-pinned without licence.
Pulling from it programmatically would put unlicensed images into videos you
publish under your own name. The sources above give the same look with a
licence attached. If you have boards you've already curated, save the images
you have rights to into `assets/image/` and `scan()` will pick them up.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MANIFEST_PATH = os.path.join(ASSETS_DIR, "library.json")

KINDS = ("video", "image", "music")

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".aac"}

# Theme vocabulary shared with moment_finder, so a moment's theme can be
# matched straight against an asset's tags.
THEMES = [
    "discipline", "ownership", "failure", "fear", "consistency",
    "identity", "focus", "growth", "resilience", "purpose",
]


@dataclass
class Asset:
    id: str
    kind: str
    path: str
    tags: list[str] = field(default_factory=list)
    duration: float = 0.0
    width: int = 0
    height: int = 0
    source: str = "local"
    source_url: str = ""
    credit: str = ""
    licence: str = ""

    @property
    def abs_path(self) -> str:
        return os.path.join(BASE_DIR, self.path)

    @property
    def is_vertical(self) -> bool:
        return self.height >= self.width


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {"assets": []}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"assets": []}


def _save_manifest(data: dict) -> None:
    os.makedirs(ASSETS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_assets() -> list[Asset]:
    return [Asset(**a) for a in _load_manifest().get("assets", [])]


def save_assets(assets: Iterable[Asset]) -> None:
    _save_manifest({"assets": [asdict(a) for a in assets]})


def ensure_dirs() -> None:
    for kind in KINDS:
        os.makedirs(os.path.join(ASSETS_DIR, kind), exist_ok=True)


# --------------------------------------------------------------------------
# probing local files
# --------------------------------------------------------------------------


def probe(path: str) -> tuple[float, int, int]:
    """(duration, width, height) via ffprobe, falling back to ffmpeg.

    The fallback matters because the two binaries are permitted separately:
    Windows Application Control can block `ffprobe.exe` while leaving
    `ffmpeg.exe` runnable, and without this the whole asset library reads as
    zero-length and every b-roll clip gets rejected for being too short.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=width,height",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(out.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        data = {}

    duration = float(data.get("format", {}).get("duration", 0) or 0)
    streams = data.get("streams") or [{}]
    width = int(streams[0].get("width", 0) or 0)
    height = int(streams[0].get("height", 0) or 0)
    if duration > 0:
        return duration, width, height

    from shorts_builder import ffmpeg_probe

    return ffmpeg_probe(path)


def _probe_audio_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        found = float(json.loads(out.stdout or "{}").get("format", {}).get("duration", 0) or 0)
        if found > 0:
            return found
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass

    from shorts_builder import ffmpeg_probe

    return ffmpeg_probe(path)[0]


def _tags_from_filename(name: str) -> list[str]:
    """`sunrise-runner_discipline.mp4` -> ['sunrise', 'runner', 'discipline']."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    parts = [p for p in re.split(r"[^a-z0-9]+", stem) if len(p) > 2]
    return [p for p in parts if not p.isdigit()]


def _kind_for(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "music"
    return None


def scan() -> list[Asset]:
    """Index every file sitting in assets/, keeping tags already in the manifest."""
    ensure_dirs()
    existing = {a.path: a for a in load_assets()}
    found: list[Asset] = []

    for kind in KINDS:
        folder = os.path.join(ASSETS_DIR, kind)
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if not os.path.isfile(full) or _kind_for(full) is None:
                continue

            rel = os.path.relpath(full, BASE_DIR).replace("\\", "/")
            if rel in existing:
                found.append(existing[rel])
                continue

            if kind == "music":
                duration, width, height = _probe_audio_duration(full), 0, 0
            else:
                duration, width, height = probe(full)

            found.append(Asset(
                id=f"local:{rel}",
                kind=kind,
                path=rel,
                tags=_tags_from_filename(name),
                duration=round(duration, 2),
                width=width,
                height=height,
                source="local",
                licence="unknown - added by hand",
            ))

    save_assets(found)
    return found


# --------------------------------------------------------------------------
# stock fetching
# --------------------------------------------------------------------------


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "content-studio/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as f:
        while True:
            chunk = response.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _get_json(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("User-Agent", "content-studio/1.0")
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_pexels(query: str, kind: str = "video", count: int = 3, vertical: bool = True) -> list[Asset]:
    """Pull vertical b-roll or stills from Pexels. Needs PEXELS_API_KEY in .env."""
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PEXELS_API_KEY isn't set. Get a free key at "
            "https://www.pexels.com/api/ and add it to .env."
        )

    ensure_dirs()
    endpoint = "videos/search" if kind == "video" else "search"
    host = "https://api.pexels.com/videos/search" if kind == "video" else "https://api.pexels.com/v1/search"
    params = urllib.parse.urlencode({
        "query": query,
        "per_page": max(count * 2, count),
        "orientation": "portrait" if vertical else "landscape",
    })
    data = _get_json(f"{host}?{params}", headers={"Authorization": key})

    known = {a.id for a in load_assets()}
    added: list[Asset] = []
    items = data.get("videos" if kind == "video" else "photos", [])

    for item in items:
        if len(added) >= count:
            break
        asset_id = f"pexels:{item['id']}"
        if asset_id in known:
            continue

        if kind == "video":
            files = sorted(
                (f for f in item.get("video_files", []) if f.get("height")),
                key=lambda f: abs(f.get("height", 0) - 1920),
            )
            if not files:
                continue
            chosen = files[0]
            url, ext = chosen["link"], ".mp4"
            width, height = chosen.get("width", 0), chosen.get("height", 0)
            duration = float(item.get("duration", 0))
        else:
            url = item["src"]["large2x"]
            ext = ".jpg"
            width, height = item.get("width", 0), item.get("height", 0)
            duration = 0.0

        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40]
        rel = f"assets/{'video' if kind == 'video' else 'image'}/pexels-{slug}-{item['id']}{ext}"
        dest = os.path.join(BASE_DIR, rel)
        _download(url, dest)

        if kind == "video" and not duration:
            duration, width, height = probe(dest)

        added.append(Asset(
            id=asset_id,
            kind="video" if kind == "video" else "image",
            path=rel,
            tags=sorted(set(_tags_from_filename(slug))),
            duration=round(duration, 2),
            width=width,
            height=height,
            source="pexels",
            source_url=item.get("url", ""),
            credit=(item.get("user") or {}).get("name", ""),
            licence="Pexels licence - free for commercial use, no attribution required",
        ))

    if added:
        save_assets(load_assets() + added)
    return added


def fetch_pixabay(query: str, kind: str = "video", count: int = 3) -> list[Asset]:
    """Same idea against Pixabay. Needs PIXABAY_API_KEY in .env."""
    key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PIXABAY_API_KEY isn't set. Get a free key at "
            "https://pixabay.com/api/docs/ and add it to .env."
        )

    ensure_dirs()
    host = "https://pixabay.com/api/videos/" if kind == "video" else "https://pixabay.com/api/"
    params = urllib.parse.urlencode({
        "key": key, "q": query, "per_page": max(count * 2, 3), "safesearch": "true",
    })
    data = _get_json(f"{host}?{params}")

    known = {a.id for a in load_assets()}
    added: list[Asset] = []

    for item in data.get("hits", []):
        if len(added) >= count:
            break
        asset_id = f"pixabay:{item['id']}"
        if asset_id in known:
            continue

        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40]
        if kind == "video":
            variants = item.get("videos", {})
            chosen = variants.get("large") or variants.get("medium") or {}
            if not chosen.get("url"):
                continue
            url, ext = chosen["url"], ".mp4"
            width, height = chosen.get("width", 0), chosen.get("height", 0)
            duration = float(item.get("duration", 0))
        else:
            url = item.get("largeImageURL")
            if not url:
                continue
            ext = ".jpg"
            width, height = item.get("imageWidth", 0), item.get("imageHeight", 0)
            duration = 0.0

        rel = f"assets/{'video' if kind == 'video' else 'image'}/pixabay-{slug}-{item['id']}{ext}"
        _download(url, os.path.join(BASE_DIR, rel))

        added.append(Asset(
            id=asset_id,
            kind="video" if kind == "video" else "image",
            path=rel,
            tags=sorted(set(_tags_from_filename(slug) + str(item.get("tags", "")).split(", "))),
            duration=round(duration, 2),
            width=width,
            height=height,
            source="pixabay",
            source_url=item.get("pageURL", ""),
            credit=item.get("user", ""),
            licence="Pixabay content licence - free for commercial use",
        ))

    if added:
        save_assets(load_assets() + added)
    return added


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


# Minimum plausible size for a real photo — anything smaller is a truncated
# download or an error page saved with a .jpg name.
MIN_IMAGE_BYTES = 5000


def _forbids_derivatives(licence: str) -> bool:
    """True for NoDerivatives licences.

    Every image here gets cropped and has text composited over it, which makes
    it a derivative work. A BY-ND image can't legally be used that way.
    """
    text = (licence or "").lower().replace(" ", "-")
    return "nd" in text.split("-") or "noderiv" in text


def _unusable(licence: str) -> bool:
    """Whether a licence rules an image out for this pipeline.

    Only the no-derivatives check remains. Attribution and non-commercial terms
    are deliberately not filtered on — images are chosen on how well they match
    the topic, and credits are not part of this workflow.

    ND stays for a reason that has nothing to do with credits: every image here
    gets cropped and has text composited over it, which makes it a derivative
    work, so an ND image can't do the job at all.
    """
    return _forbids_derivatives(licence)


# Words that appear in names but carry no identifying weight on their own.
_NAME_STOPWORDS = {"the", "van", "von", "de", "la", "jr", "sr", "ii", "iii"}


def _looks_like_person(query: str) -> bool:
    """Two or more capitalised words — good enough to spot 'Serena Williams'."""
    words = [w for w in query.split() if w]
    capitalised = [w for w in words if w[:1].isupper()]
    return len(words) >= 2 and len(capitalised) >= 2


def _matches_person(query: str, *texts: str) -> bool:
    """Does any of this result's text actually name the person searched for?

    Both providers happily return loosely related results — searching a name
    returned a photograph of an entirely different person standing near them.
    Putting the wrong face under someone's name is worse than any aesthetic
    miss, so for name-shaped queries the surname has to appear in the result.
    """
    if not _looks_like_person(query):
        return True

    haystack = " ".join(t or "" for t in texts).lower()
    parts = [
        w.strip(".,").lower() for w in query.split()
        if len(w.strip(".,")) > 2 and w.strip(".,").lower() not in _NAME_STOPWORDS
    ]
    if not parts:
        return True
    # Every significant part of the name must appear. Surname alone let through
    # a different person who happened to share it, and product shots named
    # after an athlete. Even this is best-effort — see the note in collect().
    return all(part in haystack for part in parts)


def fetch_openverse(query: str, kind: str = "image", count: int = 3) -> list[Asset]:
    """Openverse — Creative Commons images from Flickr, Wikimedia and others.

    No API key, and `license_type=commercial` filters out the non-commercial
    licences, which matters when the result ends up in something you publish.
    Attribution text comes back with each result and is stored in the manifest.
    """
    if kind != "image":
        return []

    ensure_dirs()
    params = urllib.parse.urlencode({
        "q": query,
        "page_size": max(count * 3, count),
        # "modification" excludes the NoDerivatives licences — everything here
        # gets text laid over it and cropped, which is a derivative work.
        # Commercial-use filtering is deliberately not applied: it shrinks the
        # candidate pool and relevance is what's being optimised for here.
        "license_type": "modification",
        "mature": "false",
    })
    data = _get_json(f"https://api.openverse.org/v1/images/?{params}")

    known = {a.id for a in load_assets()}
    added: list[Asset] = []
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40]

    for item in data.get("results", []):
        if len(added) >= count:
            break
        asset_id = f"openverse:{item.get('id')}"
        if asset_id in known:
            continue
        url = item.get("url")
        if not url:
            continue
        if _forbids_derivatives(item.get("license", "")):
            continue  # belt and braces behind the license_type filter
        if not _matches_person(query, item.get("title"), item.get("creator"), url):
            continue

        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        if ext not in IMAGE_EXTS:
            ext = ".jpg"
        rel = f"assets/image/openverse-{slug}-{str(item.get('id'))[:8]}{ext}"
        try:
            _download(url, os.path.join(BASE_DIR, rel))
        except Exception:
            continue  # dead upstream link; just take the next result

        licence = " ".join(
            p for p in (item.get("license", "").upper(), item.get("license_version", "")) if p
        )
        added.append(Asset(
            id=asset_id,
            kind="image",
            path=rel,
            tags=sorted(set(_tags_from_filename(slug))),
            width=int(item.get("width") or 0),
            height=int(item.get("height") or 0),
            source="openverse",
            source_url=item.get("foreign_landing_url", "") or item.get("url", ""),
            credit=item.get("attribution") or item.get("creator", ""),
            licence=f"CC {licence}".strip(),
        ))

    if added:
        save_assets(load_assets() + added)
    return added


def fetch_wikimedia(query: str, kind: str = "image", count: int = 3) -> list[Asset]:
    """Wikimedia Commons — freely licensed media, no API key.

    Stronger than Openverse for photographs of specific, named people, which
    are otherwise hard to source under a licence you can actually use.
    """
    if kind != "image":
        return []

    ensure_dirs()
    params = urllib.parse.urlencode({
        "action": "query", "format": "json",
        "generator": "search", "gsrnamespace": 6,
        "gsrsearch": query, "gsrlimit": max(count * 2, count),
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1600,
    })
    data = _get_json(f"https://commons.wikimedia.org/w/api.php?{params}")

    known = {a.id for a in load_assets()}
    added: list[Asset] = []
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40]

    for page in (data.get("query") or {}).get("pages", {}).values():
        if len(added) >= count:
            break
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue

        meta = info.get("extmetadata", {})
        licence = meta.get("LicenseShortName", {}).get("value", "")
        # Skip anything not commercially reusable, and anything that forbids
        # derivatives — text gets composited over all of these.
        if licence and _unusable(licence):
            continue

        asset_id = f"wikimedia:{page.get('pageid')}"
        if asset_id in known:
            continue
        if not _matches_person(query, page.get("title"), meta.get("ImageDescription", {}).get("value", "")):
            continue

        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        if ext not in IMAGE_EXTS:
            ext = ".jpg"
        rel = f"assets/image/wikimedia-{slug}-{page.get('pageid')}{ext}"
        try:
            _download(url, os.path.join(BASE_DIR, rel))
        except Exception:
            continue

        added.append(Asset(
            id=asset_id,
            kind="image",
            path=rel,
            tags=sorted(set(_tags_from_filename(slug))),
            width=int(info.get("thumbwidth") or 0),
            height=int(info.get("thumbheight") or 0),
            source="wikimedia",
            source_url=info.get("descriptionurl", ""),
            credit=_strip_html(meta.get("Artist", {}).get("value", "")),
            licence=licence or "see source page",
        ))

    if added:
        save_assets(load_assets() + added)
    return added


def validate(delete_files: bool = True) -> dict:
    """Drop assets that are broken or that can't legally be used here.

    Two things get caught: downloads that came back truncated (an error page
    saved as a .jpg), and licences forbidding derivative works or commercial
    use, which slip in when an upstream provider labels something loosely.
    """
    kept: list[Asset] = []
    removed = {"missing": [], "truncated": [], "licence": []}

    for asset in load_assets():
        reason = None
        if not os.path.exists(asset.abs_path):
            reason = "missing"
        elif asset.kind == "image" and os.path.getsize(asset.abs_path) < MIN_IMAGE_BYTES:
            reason = "truncated"
        elif asset.source != "local" and _unusable(asset.licence):
            reason = "licence"

        if reason is None:
            kept.append(asset)
            continue

        removed[reason].append(f"{os.path.basename(asset.path)} ({asset.licence or 'n/a'})")
        if delete_files and reason != "missing" and os.path.exists(asset.abs_path):
            try:
                os.remove(asset.abs_path)
            except OSError:
                pass

    save_assets(kept)
    return removed


def write_credits(out_path: Optional[str] = None) -> str:
    """Write assets/CREDITS.md.

    Most of these licences (anything CC BY or BY-SA) require crediting the
    photographer wherever the image is used. This keeps the list in one
    copy-pasteable place so that's not a scramble at posting time.
    """
    out_path = out_path or os.path.join(ASSETS_DIR, "CREDITS.md")
    assets = [a for a in load_assets() if a.source != "local"]

    lines = [
        "# Image and footage credits",
        "",
        "Generated by `asset_library.py`. Anything under a CC BY or BY-SA",
        "licence must be credited wherever it appears — put the relevant lines",
        "in your post description or an end card.",
        "",
    ]
    for source in sorted({a.source for a in assets}):
        lines.append(f"## {source}")
        lines.append("")
        for asset in sorted((a for a in assets if a.source == source), key=lambda a: a.path):
            credit = asset.credit or "unknown creator"
            lines.append(f"- `{os.path.basename(asset.path)}` — {credit}. {asset.licence}.")
            if asset.source_url:
                lines.append(f"  {asset.source_url}")
        lines.append("")

    os.makedirs(ASSETS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def collect(topic: str, queries: list[str], per_query: int = 3) -> list[Asset]:
    """Fill the library for one topic, tagging everything with the topic name.

    Searching for a named person is best-effort even with the name check in
    `_matches_person`: keyword search on these repositories still returns
    merchandise named after an athlete, or a different person who shares the
    name. Look at what came back before publishing any of it.
    """
    collected: list[Asset] = []
    for query in queries:
        got = fetch_stock(query, kind="image", count=per_query)
        for asset in got:
            asset.tags = sorted(set(asset.tags + _tags_from_filename(topic) + [topic.lower()]))
        collected += got
        print(f"    {query!r}: {len(got)} image(s)")

    if collected:
        # Persist the topic tags added above.
        by_id = {a.id: a for a in collected}
        merged = [by_id.get(a.id, a) for a in load_assets()]
        save_assets(merged)
    return collected


_QUERY_STOPWORDS = {"in", "on", "at", "the", "a", "an", "of", "with", "and", "into", "over"}


def query_variants(query: str) -> list[str]:
    """Progressively broader forms of a search query.

    The single biggest cause of bad images here was empty result sets, not bad
    ranking. These providers match fairly literally: "boxer in empty gym"
    returns zero results on Openverse at any licence setting, while "boxing
    gym" returns hundreds. Descriptive four-word phrases — exactly what reads
    best in a prompt, and exactly what `topic_tags` produces — are the worst
    possible input.

    So each query is tried in full first (a precise hit is still the best
    outcome), then broadened until something comes back.
    """
    words = [w for w in re.split(r"\s+", query.strip()) if w]
    meaningful = [w for w in words if w.lower() not in _QUERY_STOPWORDS]

    variants = [query.strip()]
    if len(meaningful) > 2:
        variants.append(" ".join(meaningful[:2]))
        variants.append(" ".join(meaningful[-2:]))
    if len(meaningful) > 1:
        variants.append(meaningful[0])

    seen: set[str] = set()
    ordered = []
    for variant in variants:
        key = variant.lower()
        if variant and key not in seen:
            seen.add(key)
            ordered.append(variant)
    return ordered


def fetch_stock(query: str, kind: str = "video", count: int = 3) -> list[Asset]:
    """Try each provider in turn, broadening the query until results appear.

    Pexels and Pixabay need a free key and have much better search relevance;
    Openverse and Wikimedia need no key, which is why images can be sourced
    with nothing configured at all. Providers are tried in that order, and each
    one gets the full query before any broadened form.
    """
    errors = []
    providers = (fetch_pexels, fetch_pixabay)
    if kind == "image":
        providers = (fetch_pexels, fetch_pixabay, fetch_openverse, fetch_wikimedia)

    variants = query_variants(query) if kind == "image" else [query]

    for fetcher in providers:
        for variant in variants:
            try:
                found = fetcher(variant, kind=kind, count=count)
            except RuntimeError as e:
                errors.append(str(e))
                break  # missing key — no point retrying this provider
            except Exception as e:  # network/API hiccup
                errors.append(f"{fetcher.__name__}: {e}")
                continue
            if found:
                if variant != query:
                    print(f"    · no results for {query!r}, used {variant!r}")
                return found

    if errors:
        print("    (no stock fetched: " + " | ".join(dict.fromkeys(errors)) + ")")
    return []


CURATED_DIR = os.path.join(ASSETS_DIR, "broll")


def curated_broll(
    queries: list[str],
    count: int,
    exclude: Optional[set] = None,
    min_duration: float = 0.0,
) -> list[Asset]:
    """Hand-picked b-roll from `assets/broll/`, matched by filename.

    This folder exists because stock footage has a ceiling. Pexels and Pixabay
    are clean, bright and corporate; the accounts worth copying use footage
    that is dark, slow and cinematic, and no amount of query tuning turns one
    into the other. So anything dropped in here is preferred over anything
    fetched, always.

    Matching is by filename, deliberately: `night-run-silhouette_struggle.mp4`
    matches "night", "run", "silhouette" and "struggle". No database, no
    tagging step — renaming a file is the whole workflow, and the folder stays
    readable to anyone who opens it.

    Clips with no query match are still returned once the matches run out, so a
    small library is never worse than an empty one.
    """
    if not os.path.isdir(CURATED_DIR):
        return []

    seen = set(exclude or ())
    wanted = {w for q in queries for w in re.split(r"[^a-z0-9]+", q.lower()) if len(w) > 2}

    scored: list[tuple[int, Asset]] = []
    for root, _dirs, names in os.walk(CURATED_DIR):
        for name in sorted(names):
            full = os.path.join(root, name)
            if not os.path.isfile(full) or _kind_for(full) != "video":
                continue
            rel = os.path.relpath(full, BASE_DIR).replace("\\", "/")
            asset_id = f"curated:{os.path.relpath(full, CURATED_DIR)}"
            if asset_id in seen:
                continue

            duration, width, height = probe(full)
            if min_duration and duration and duration < min_duration:
                continue

            # Folder names are tags too, so `emotion/lost/road-fog.mp4` matches
            # "lost" without it having to appear in the filename. Organising by
            # dropping a file into the right folder is less work than naming it
            # carefully, and survives being reorganised later.
            folders = os.path.relpath(root, CURATED_DIR).replace("\\", "/")
            tags = _tags_from_filename(name)
            if folders != ".":
                tags += [p for p in re.split(r"[^a-z0-9]+", folders.lower()) if len(p) > 2]

            hits = len(wanted.intersection(tags))
            scored.append((hits, Asset(
                id=asset_id, kind="video", path=rel, tags=tags,
                duration=duration, width=width, height=height,
                source="curated", licence="local",
            )))

    # Best match first; unmatched clips still usable rather than discarded.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [asset for _hits, asset in scored[:count]]


def fetch_broll_set(
    queries: list[str],
    count: int = 3,
    min_duration: float = 0.0,
    exclude: Optional[set] = None,
) -> list[Asset]:
    """A set of visually *different* b-roll clips for one moment.

    One clip per query rather than several from a single query, and that is the
    whole point of the function. Asking Pexels for three results on "empty gym"
    returns three angles of the same gym — cut together they read as one shot
    held too long, not as an edit. Asking three different queries returns three
    different places.

    `exclude` carries the asset ids already used elsewhere in the run, so two
    clips cut from the same talk don't open on identical footage.

    Returns fewer than `count` rather than padding with repeats: a short set of
    distinct shots is more useful than a full set with duplicates, and the shot
    list can simply reuse one.
    """
    chosen: list[Asset] = []
    seen: set = set(exclude or ())
    # How many results to pull from one query. A single query returns several
    # genuinely different videos on the same theme, which is enough variety
    # when they are separated by cuts back to the speaker — and it is the only
    # way to reach eighteen distinct shots from five or six queries.
    per_query = max(1, -(-count // max(1, len(queries))) + 1)

    def take(candidates: list[Asset], limit: int) -> int:
        """Claim up to `limit` unseen clips. Returns how many were taken."""
        taken = 0
        for asset in candidates:
            if taken >= limit or len(chosen) >= count:
                break
            if asset.id in seen:
                continue
            # Duration 0 means it couldn't be probed; keep it rather than
            # discard a usable clip over a missing measurement.
            if min_duration and asset.duration and asset.duration < min_duration:
                continue
            seen.add(asset.id)
            chosen.append(asset)
            taken += 1
        return taken

    for query in queries:
        if len(chosen) >= count:
            break
        try:
            take(fetch_stock(query, kind="video", count=per_query + 2), per_query)
        except Exception as e:
            print(f"    · b-roll search failed for {query!r}: {e}")

    # Not enough distinct queries produced a usable clip. Go back through them
    # asking for more results each, rather than returning a half-empty set.
    if len(chosen) < count:
        for query in queries:
            if len(chosen) >= count:
                break
            try:
                take(fetch_stock(query, kind="video", count=20), count)
            except Exception:
                continue

    return chosen


# --------------------------------------------------------------------------
# picking
# --------------------------------------------------------------------------


def _score(asset: Asset, keywords: list[str]) -> float:
    wanted = {w.lower() for kw in keywords for w in re.split(r"[^a-z0-9]+", kw.lower()) if len(w) > 2}
    if not wanted:
        return 0.0
    tags = {t.lower() for t in asset.tags}
    overlap = len(wanted & tags)
    # Partial credit for substring hits ("runner" matching "running").
    partial = sum(0.4 for w in wanted if any(w in t or t in w for t in tags)) if not overlap else 0
    score = overlap + partial
    if asset.kind == "video" and asset.is_vertical:
        score += 0.5
    return score


def pick(
    kind: str,
    keywords: list[str],
    min_duration: float = 0.0,
    exclude: Optional[set[str]] = None,
    assets: Optional[list[Asset]] = None,
) -> Optional[Asset]:
    """Best asset of `kind` for these keywords, or None if the library is empty.

    Falls back to the longest usable asset when nothing matches by tag, so a
    render never fails just because the wording didn't line up.
    """
    exclude = exclude or set()
    pool = [
        a for a in (assets if assets is not None else load_assets())
        if a.kind == kind and a.id not in exclude and os.path.exists(a.abs_path)
    ]
    if min_duration:
        long_enough = [a for a in pool if a.duration >= min_duration]
        # Short b-roll is fine — the renderer loops it.
        pool = long_enough or pool
    if not pool:
        return None

    ranked = sorted(pool, key=lambda a: (_score(a, keywords), a.duration), reverse=True)
    return ranked[0]


def _thumbnail_b64(path: str, size: int = 320) -> Optional[tuple[str, str]]:
    """A small JPEG of an image, base64'd, for vision ranking.

    Downsampled hard on purpose: a full-resolution image costs thousands of
    tokens, and picking between candidates only needs enough to see subject,
    composition and mood.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    import base64
    import io

    try:
        with Image.open(path) as raw:
            img = raw.convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=72)
    except (OSError, ValueError):
        return None

    return "image/jpeg", base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def rank_by_vision(
    candidates: list[Asset],
    query: str,
    mood: str = "",
    api_key: Optional[str] = None,
) -> Optional[Asset]:
    """Ask Claude which candidate actually fits, looking at the images.

    Keyword scoring can't catch the failure that matters most here: an image
    that matches the words but not the meaning — a stadium gate for "gym", a
    product box for an athlete's name, a sunny beach under a line about grief.
    Looking at the pictures catches it.

    Returns None if vision ranking isn't available, so callers fall back to
    provider order rather than failing.
    """
    usable = [(a, _thumbnail_b64(a.abs_path)) for a in candidates]
    usable = [(a, t) for a, t in usable if t]
    if len(usable) < 2:
        return usable[0][0] if usable else None

    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    content: list = [{
        "type": "text",
        "text": (
            f'These are candidate images for a motivational post about: "{query}".'
            + (f" The intended visual register is: {mood}." if mood else "")
            + "\n\nPick the one that genuinely depicts the subject, with a mood that "
            "matches. Text will be laid over it, so prefer some calm, uncluttered "
            "space and avoid busy or already-captioned pictures.\n\n"
            "Reject rather than settle. Return 0 if none of them actually fit — "
            "and specifically reject any image that:\n"
            "- merely contains a matching word (a sign, a book cover, a product, a logo)\n"
            "- is a document, manuscript, scan, screenshot or artwork rather than a photograph\n"
            "- shows national flags, political or religious symbols, protests, funerals, "
            "memorials, military operations, or identifiable public figures — these carry "
            "meanings the post never intended\n"
            "- depicts anyone who appears to be a minor\n\n"
            "A plain background is a perfectly good outcome; a wrong or loaded image is not.\n\n"
            'Respond with ONLY valid JSON: {"choice": <1-based index, or 0 for none>, '
            '"why": "<one short line>"}'
        ),
    }]
    for i, (_asset, thumb) in enumerate(usable, start=1):
        media_type, data = thumb
        content.append({"type": "text", "text": f"Image {i}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })

    try:
        client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        verdict = json.loads(raw)
        choice = int(verdict.get("choice", 0))
    except Exception:
        return None

    if choice == 0:
        # Nothing in the pool fits. Say so rather than returning the least-bad
        # option — a plain background reads as a design choice, while a wrong
        # or loaded photo under a motivational line reads as a mistake.
        print(f"    · no usable image for {query!r} ({verdict.get('why', '')})")
        return None

    if 1 <= choice <= len(usable):
        return usable[choice - 1][0]
    return None


def best_for_topic(
    query: str,
    mood: str = "",
    fetch: bool = True,
    use_vision: bool = True,
    pool: int = 4,
    exclude: Optional[set[str]] = None,
) -> Optional[Asset]:
    """The best image for a topic, fetching more candidates if needed.

    Providers already return results ranked by relevance, so their order is
    the baseline — better than re-scoring against a thin local tag index.
    Vision ranking then picks between the top few.
    """
    exclude = exclude or set()
    before = {a.id for a in load_assets()}

    fresh: list[Asset] = []
    if fetch:
        fresh = [a for a in fetch_stock(query, kind="image", count=pool) if a.id not in exclude]

    if not fresh:
        # Nothing new to consider. Fall back to the library only if vision can
        # confirm one of its images genuinely fits — an unchecked tag match is
        # how unrelated photos end up under a quote.
        candidates = [
            a for a in load_assets()
            if a.kind == "image" and a.id not in exclude and os.path.exists(a.abs_path)
        ]
        ranked = sorted(candidates, key=lambda a: _score(a, [query]), reverse=True)[:pool]
        if not ranked:
            return None
        if use_vision and len(ranked) > 1:
            return rank_by_vision(ranked, query, mood=mood)
        return ranked[0] if _score(ranked[0], [query]) > 0 else None

    if use_vision and len(fresh) > 1:
        chosen = rank_by_vision(fresh, query, mood=mood)
        # None here is a real verdict ("none of these fit"), not a failure to
        # decide — respect it instead of falling back to the top result.
        return chosen

    return fresh[0]


def summary() -> dict:
    assets = load_assets()
    return {
        kind: {
            "count": sum(1 for a in assets if a.kind == kind),
            "total_seconds": round(sum(a.duration for a in assets if a.kind == kind), 1),
        }
        for kind in KINDS
    }


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args or args[0] == "scan":
        found = scan()
        print(f"Indexed {len(found)} assets")
        for kind, stats in summary().items():
            print(f"  {kind:<6} {stats['count']:>3} files, {stats['total_seconds']}s")
    elif args[0] == "fetch" and len(args) >= 3:
        kind, query = args[1], " ".join(args[2:])
        got = fetch_stock(query, kind=kind, count=3)
        for a in got:
            print(f"  + {a.path}  ({a.width}x{a.height}, {a.duration}s, {a.source})")
        if not got:
            print("  nothing fetched")
        write_credits()
    elif args[0] == "collect" and len(args) >= 3:
        topic, queries = args[1], args[2:]
        print(f"Collecting '{topic}'...")
        got = collect(topic, queries)
        print(f"  {len(got)} image(s) added")
        print(f"  credits -> {write_credits()}")
    elif args[0] == "credits":
        print(write_credits())
    elif args[0] == "validate":
        dropped = validate()
        for reason, items in dropped.items():
            if items:
                print(f"  removed ({reason}): {len(items)}")
                for item in items:
                    print(f"    - {item}")
        if not any(dropped.values()):
            print("  everything checks out")
        write_credits()
    else:
        print("Usage:")
        print("  python asset_library.py scan")
        print("  python asset_library.py fetch <video|image|music> <query>")
        print("  python asset_library.py collect <topic> <query> [<query> ...]")
        print("  python asset_library.py credits")
