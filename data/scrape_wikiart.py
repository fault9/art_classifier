"""
Scrape painting images from WikiArt by art movement/style.
Uses WikiArt's public JSON API (no authentication required).

Filtering (three layers):
  1. Artist exclusion — architects and non-painter sculptors whose WikiArt
     entries are buildings, drawings, or sculpture photos, not paintings.
  2. Title keywords — excludes works titled with architecture/sculpture terms.
  3. Aspect ratio — extreme ratios flag tapestries, banners, panoramas.

Output:
  data/images/{class_name}/{slug}.jpg
  data/wikiart_metadata.csv
  data/exclusion_log.csv     — every excluded work with reason

Usage:
  python data/scrape_wikiart.py [--max-per-class 200] [--dry-run] [--classes Baroque]
"""

import re
import csv
import time
import argparse
import logging
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DELAY = 1.5
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ArtClassifier-Research/1.0; academic use)",
    "Accept": "application/json, text/html",
    "Referer": "https://www.wikiart.org/",
}

CLASS_STYLES = {
    "Renaissance":   ["high-renaissance", "early-renaissance"],
    "Baroque":       ["baroque"],
    "Impressionism": ["impressionism"],
    "Expressionism": ["expressionism"],
    "Cubism":        ["cubism"],
    "Abstract":      ["abstract-expressionism", "abstract-art"],
    "Surrealism":    ["surrealism"],
    "Pop Art":       ["pop-art"],
}

IMAGES_DIR   = Path("data/images")
META_PATH    = Path("data/wikiart_metadata.csv")
EXCLUDE_LOG  = Path("data/exclusion_log.csv")

# ---------------------------------------------------------------------------
# Filter rules
# ---------------------------------------------------------------------------

# Artists primarily known as architects — their WikiArt entries under
# painting styles are usually architectural drawings, building elevations, or
# photographs of facades, not paintings.
EXCLUDED_ARCHITECTS: set[str] = {
    "donato bramante",
    "filippo brunelleschi",
    "andrea palladio",
    "le corbusier",
    "frank lloyd wright",
    "antoni gaudí",
    "antonio gaudi",
    "leon battista alberti",
    "michelozzo di bartolomeo",
    "giuliano da sangallo",
    "antonio da sangallo the younger",
    "bernardo rossellino",
    "filarete",
    "giacomo della porta",
    "carlo maderno",
    "baldassare peruzzi",
}

# Artists primarily known as sculptors whose WikiArt entries under
# painting styles tend to be photographs of their sculptures.
# NOTE: Michelangelo and Picasso are intentionally EXCLUDED from this list —
# both produced major paintings alongside sculpture.
EXCLUDED_SCULPTORS: set[str] = {
    "gian lorenzo bernini",
    "lorenzo bernini",
    "auguste rodin",
    "constantin brancusi",
    "donatello",
    "antonio canova",
    "bertel thorvaldsen",
    "jean-baptiste carpeaux",
    "camille claudel",
    "aristide maillol",
    "gutzon borglum",
}

EXCLUDED_ARTISTS: set[str] = EXCLUDED_ARCHITECTS | EXCLUDED_SCULPTORS

# Title keywords that almost certainly indicate non-painting works.
# Word-boundary matched, case-insensitive.
_TITLE_EXCLUDE = re.compile(
    r"\b("
    r"basilica|cathedral|church|chapel|palazzo|palais|palace"
    r"|facade|fa[çc]ade|fa[çc]ades"
    r"|monastery|convent|abbey|temple|mosque|synagogue"
    r"|baptistery|baptistry|campanile|loggia"
    r"|architectural|architecture|floor plan|ground plan|elevation|cross.section"
    r"|sculpture|sculpted|bust|relief|statue|statuette|figurine"
    r"|monument|cenotaph|tomb|mausoleum|sarcophagus|urn"
    r"|engraving|etching|lithograph|woodcut|woodblock|drypoint|aquatint"
    r"|tapestry|embroidery|textile|weaving|mosaic|stained.glass"
    r"|photograph|photography"
    r")\b",
    re.IGNORECASE,
)

# Aspect ratio bounds (height / width).
# Outside these bounds → almost certainly not a conventional painting.
ASPECT_MIN = 0.25   # very wide: panoramas, architectural frieze details
ASPECT_MAX = 4.5    # very tall: tapestries, banners, scroll paintings


# ---------------------------------------------------------------------------
# Filtering function
# ---------------------------------------------------------------------------

def filter_painting(p: dict) -> str | None:
    """
    Return a rejection reason string, or None if the work should be kept.
    `p` is a raw dict from the WikiArt listing API.
    """
    artist = (p.get("artistName") or "").strip()
    title  = (p.get("title")      or "").strip()
    width  = p.get("width",  0) or 0
    height = p.get("height", 0) or 0

    if artist.lower() in EXCLUDED_ARTISTS:
        return f"excluded_artist:{artist}"

    if _TITLE_EXCLUDE.search(title):
        match = _TITLE_EXCLUDE.search(title).group(0)
        return f"title_keyword:{match}"

    if width > 0 and height > 0:
        ratio = height / width
        if ratio < ASPECT_MIN:
            return f"aspect_too_wide:{ratio:.2f}"
        if ratio > ASPECT_MAX:
            return f"aspect_too_tall:{ratio:.2f}"

    return None  # passes all filters


# ---------------------------------------------------------------------------
# Exclusion log writer
# ---------------------------------------------------------------------------

class ExclusionLogger:
    def __init__(self, path: Path):
        self._path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["class", "style", "artist", "title", "source_url", "reason"])

    def log(self, class_name: str, style: str, artist: str,
            title: str, src_url: str, reason: str) -> None:
        self._writer.writerow([class_name, style, artist, title, src_url, reason])
        self._file.flush()
        log.info(f"  [excluded/{reason}] {artist} — {title[:50]}")

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# WikiArt API helpers
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", (s or "").lower().strip())[:80]


def fetch_paintings_page(style: str, page: int, session: requests.Session) -> list[dict]:
    url = (
        f"https://www.wikiart.org/en/paintings-by-style/{style}"
        f"?json=2&layout=new&page={page}&resultType=masonry"
    )
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json().get("Paintings", [])
    except Exception as e:
        log.warning(f"  [API error] {style} p{page}: {e}")
        return []


def image_url_large(raw: str) -> str:
    """Return a !Large.jpg CDN variant, or empty string if the raw URL has no .jpg/.jpeg extension."""
    if not raw:
        return ""
    # Strip any existing size suffix
    base = re.sub(r"!(?:Large|Small|PinterestSmall|630|Original)\.jpe?g$", "", raw)
    # Only attempt the !Large trick if the base ends in .jpg or .jpeg
    base = re.sub(r"\.jpe?g$", "", base)
    if not base:
        return ""
    return base + ".jpg!Large.jpg"


def download_image(url_large: str, url_original: str, dest: Path,
                   session: requests.Session) -> bool:
    if dest.exists():
        return True
    # Try !Large first, fall back to original URL
    candidates = [c for c in [url_large, url_original] if c]
    for candidate in candidates:
        try:
            resp = session.get(candidate, timeout=30, stream=True)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            if "image" not in resp.headers.get("Content-Type", ""):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception as e:
            log.debug(f"  [img error] {candidate}: {e}")
    return False


# ---------------------------------------------------------------------------
# Per-class scraper
# ---------------------------------------------------------------------------

def scrape_class(
    class_name: str,
    styles: list[str],
    max_count: int,
    session: requests.Session,
    dry_run: bool,
    excl_log: ExclusionLogger,
    existing_records: list[dict] | None = None,
) -> list[dict]:
    class_dir = IMAGES_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    # Pre-populate seen from existing records to avoid re-scraping on resume
    seen: set[str] = set()
    if existing_records:
        for r in existing_records:
            if r.get("label") == class_name:
                title  = str(r.get("title",  "") or "")
                artist = str(r.get("artist", "") or "")
                seen.add(slugify(f"{artist}_{title}"))

    for style in styles:
        if len(records) >= max_count:
            break
        log.info(f"    style={style}  have={len(records)}/{max_count}")
        page = 1

        while len(records) < max_count:
            paintings = fetch_paintings_page(style, page, session)
            if not paintings:
                break

            for p in paintings:
                if len(records) >= max_count:
                    break

                title  = (p.get("title")      or "untitled").strip()
                artist = (p.get("artistName") or "unknown").strip()
                year   = p.get("year", "")
                slug   = slugify(f"{artist}_{title}")

                if slug in seen:
                    continue
                seen.add(slug)

                page_url = p.get("paintingUrl", "")
                src_url  = f"https://www.wikiart.org{page_url}" if page_url else ""

                # --- Apply filters ---
                reason = filter_painting(p)
                if reason:
                    excl_log.log(class_name, style, artist, title, src_url, reason)
                    continue

                raw_url  = p.get("image", "")
                img_url  = image_url_large(raw_url)
                dest     = class_dir / f"{slug}.jpg"

                if not dry_run:
                    if not download_image(img_url, raw_url, dest, session):
                        log.info(f"  [skip/download] {artist} — {title[:40]}")
                        continue
                    time.sleep(DELAY)

                records.append({
                    "file_path":     str(dest),
                    "label":         class_name,
                    "source":        "wikiart",
                    "artist":        artist,
                    "title":         title,
                    "year":          year,
                    "source_url":    src_url,
                    "wikiart_style": style,
                })
                if len(records) % 5 == 0:
                    log.info(f"  [{len(records)}/{max_count}] {artist} — {title[:40]}")

            page += 1
            time.sleep(DELAY)

    log.info(f"  -> {len(records)} paintings kept for {class_name}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true",
                        help="Metadata only — do not download images")
    parser.add_argument("--classes", nargs="+", default=list(CLASS_STYLES.keys()))
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    all_records: list[dict] = []
    if META_PATH.exists():
        existing = pd.read_csv(META_PATH)
        all_records = existing.to_dict("records")
        log.info(f"Resuming from {len(all_records)} existing records")

    excl_log = ExclusionLogger(EXCLUDE_LOG)

    try:
        for class_name in args.classes:
            if class_name not in CLASS_STYLES:
                log.warning(f"Unknown class '{class_name}' — skipping")
                continue
            already = sum(1 for r in all_records if r["label"] == class_name)
            need    = args.max_per_class - already
            if need <= 0:
                log.info(f"{class_name}: already complete ({already})")
                continue

            log.info(f"\n=== {class_name} (need {need} more) ===")
            new = scrape_class(
                class_name, CLASS_STYLES[class_name], need, session,
                args.dry_run, excl_log, all_records,
            )
            all_records.extend(new)
            pd.DataFrame(all_records).to_csv(META_PATH, index=False)
    finally:
        excl_log.close()

    df = pd.DataFrame(all_records)
    df.to_csv(META_PATH, index=False)
    log.info(f"\nTotal: {len(df)} records saved to {META_PATH}")
    log.info(f"Exclusion log: {EXCLUDE_LOG}")
    if not df.empty:
        log.info("\nClass distribution:\n" + df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
