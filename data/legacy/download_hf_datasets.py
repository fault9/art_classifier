"""
Download paintings from the huggan/wikiart HuggingFace dataset.
Maps WikiArt style labels to our 8 classes and saves images + metadata.

Output:
  data/images/{class_name}/{hash}.jpg
  data/hf_metadata.csv

Usage:
  python data/legacy/download_hf_datasets.py [--max-per-class 200]
"""

import io
import re
import hashlib
import logging
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

IMAGES_DIR = Path("data/images")
META_PATH  = Path("data/hf_metadata.csv")

# huggan/wikiart style names (ClassLabel) → our 8 classes
# Only styles with a clear mapping are included; ambiguous ones are skipped.
STYLE_MAP = {
    "Abstract_Expressionism": "Abstract",
    "Action_painting":        "Abstract",
    "Color_Field_Painting":   "Abstract",
    "Minimalism":             "Abstract",
    "Analytical_Cubism":      "Cubism",
    "Cubism":                 "Cubism",
    "Synthetic_Cubism":       "Cubism",
    "Baroque":                "Baroque",
    "Early_Renaissance":      "Renaissance",
    "High_Renaissance":       "Renaissance",
    "Northern_Renaissance":   "Renaissance",
    "Expressionism":          "Expressionism",
    "Impressionism":          "Impressionism",
    "Pop_Art":                "Pop Art",
    # Surrealism is not in the huggan/wikiart style list —
    # the WikiArt scraper handles it.
}

# Genres to exclude — these are not finished paintings
EXCLUDED_GENRES = {"illustration", "sketch_and_study"}

# Same artist exclusion list as the scraper
EXCLUDED_ARTISTS = {
    "donato bramante", "filippo brunelleschi", "andrea palladio",
    "le corbusier", "frank lloyd wright", "antoni gaudí", "antonio gaudi",
    "leon battista alberti", "gian lorenzo bernini", "lorenzo bernini",
    "auguste rodin", "constantin brancusi", "donatello",
    "antonio canova", "bertel thorvaldsen",
}

# Title keyword filter (same as scraper)
_TITLE_EXCLUDE = re.compile(
    r"\b(basilica|cathedral|church|chapel|palazzo|facade|fa[çc]ade"
    r"|sculpture|bust|relief|statue|monument|tomb"
    r"|engraving|etching|lithograph|woodcut|tapestry|mosaic|photograph)\b",
    re.IGNORECASE,
)

OUR_CLASSES = list(dict.fromkeys(STYLE_MAP.values()))


def save_image(img: Image.Image, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(dest, "JPEG", quality=90)
        return True
    except Exception as e:
        log.debug(f"  [save error] {dest}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=200)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Run: pip install datasets")
        return

    log.info("Loading huggan/wikiart (streaming)...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)

    # Get label name lists from features so we can decode integer IDs
    style_names  = ds.features["style"].names   # list indexed by int id
    genre_names  = ds.features["genre"].names
    artist_names = ds.features["artist"].names

    log.info(f"Style labels available: {style_names}")

    counts: dict[str, int] = {c: 0 for c in OUR_CLASSES}
    records: list[dict] = []

    for item in ds:
        # Decode integer ClassLabel IDs to strings
        style_name  = style_names[item["style"]]
        genre_name  = genre_names[item["genre"]]
        artist_slug = artist_names[item["artist"]]
        artist_display = artist_slug.replace("-", " ").title()

        # Map to our class
        our_class = STYLE_MAP.get(style_name)
        if our_class is None:
            continue
        if counts[our_class] >= args.max_per_class:
            continue

        # Genre filter
        if genre_name in EXCLUDED_GENRES:
            continue

        # Artist filter
        if artist_display.lower() in EXCLUDED_ARTISTS:
            continue

        # Get image
        img = item.get("image")
        if img is None:
            continue

        # Generate stable filename from image content hash
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        slug = hashlib.md5(buf.getvalue()).hexdigest()[:14]
        dest = IMAGES_DIR / our_class / f"hf_{slug}.jpg"

        if not dest.exists():
            if not save_image(img, dest):
                continue

        counts[our_class] += 1
        records.append({
            "file_path":     str(dest),
            "label":         our_class,
            "source":        "huggingface",
            "artist":        artist_display,
            "title":         "",
            "year":          "",
            "source_url":    "",
            "wikiart_style": style_name,
        })

        if counts[our_class] % 10 == 0:
            log.info(f"  {our_class:<22} {counts[our_class]}/{args.max_per_class}")

        if all(counts[c] >= args.max_per_class for c in OUR_CLASSES):
            log.info("All classes complete.")
            break

    df = pd.DataFrame(records)
    df.to_csv(META_PATH, index=False)
    log.info(f"\nSaved {len(df)} records to {META_PATH}")
    log.info("\nClass distribution:\n" + df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
