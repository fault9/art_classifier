"""
Consolidate all image sources, filter non-painting works, validate images,
perform stratified train/test split, and report class distribution.

Output:
  data/metadata.csv       — all valid paintings with labels and splits
  data/train.csv          — 80% training split
  data/test.csv           — 20% test split
  data/exclusion_log.csv  — appended with any works removed here (not caught by scraper)

Usage:
  python data/build_dataset.py [--test-size 0.2] [--min-images 50] [--skip-validation]
"""

import re
import argparse
import logging
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CLASSES = [
    "Renaissance", "Baroque", "Impressionism", "Expressionism",
    "Cubism", "Abstract", "Surrealism", "Pop Art",
]

SOURCE_METAS = [
    Path("data/wikiart_metadata.csv"),
    Path("data/hf_metadata.csv"),
]

OUT_META    = Path("data/metadata.csv")
OUT_TRAIN   = Path("data/train.csv")
OUT_TEST    = Path("data/test.csv")
EXCLUDE_LOG = Path("data/exclusion_log.csv")

# Same title-keyword filter as the scraper — catches HuggingFace data
_TITLE_EXCLUDE = re.compile(
    r"\b("
    r"basilica|cathedral|church|chapel|palazzo|palais|palace"
    r"|facade|fa[çc]ade|monastery|convent|abbey|temple|mosque|synagogue"
    r"|baptistery|campanile|loggia|architectural|architecture"
    r"|floor plan|ground plan|elevation|cross.section"
    r"|sculpture|sculpted|bust|relief|statue|statuette|figurine"
    r"|monument|cenotaph|tomb|mausoleum|sarcophagus"
    r"|engraving|etching|lithograph|woodcut|woodblock|drypoint|aquatint"
    r"|tapestry|embroidery|textile|weaving|mosaic|stained.glass"
    r"|photograph|photography"
    r")\b",
    re.IGNORECASE,
)

# Artists excluded regardless of source
_EXCLUDED_ARTISTS = {
    "donato bramante", "filippo brunelleschi", "andrea palladio",
    "le corbusier", "frank lloyd wright", "antoni gaudí", "antonio gaudi",
    "leon battista alberti", "michelozzo di bartolomeo",
    "gian lorenzo bernini", "lorenzo bernini",
    "auguste rodin", "constantin brancusi", "donatello",
    "antonio canova", "bertel thorvaldsen", "jean-baptiste carpeaux",
    "camille claudel", "aristide maillol",
}

# Aspect ratio limits (height / width) for actual image pixels
ASPECT_MIN = 0.20
ASPECT_MAX = 5.0


def get_image_dimensions(path: str) -> tuple[int, int] | None:
    """Return (width, height) or None if unreadable."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def validate_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def filter_row(row: pd.Series) -> str | None:
    """Return exclusion reason or None if the row should be kept."""
    title  = str(row.get("title",  "") or "")
    artist = str(row.get("artist", "") or "").lower().strip()
    path   = str(row.get("file_path", "") or "")

    if artist in _EXCLUDED_ARTISTS:
        return f"excluded_artist:{artist}"

    if _TITLE_EXCLUDE.search(title):
        kw = _TITLE_EXCLUDE.search(title).group(0)
        return f"title_keyword:{kw}"

    # Aspect ratio from actual image pixels (catches anything the scraper missed)
    if path and Path(path).exists():
        dims = get_image_dimensions(path)
        if dims:
            w, h = dims
            if w > 0:
                ratio = h / w
                if ratio < ASPECT_MIN:
                    return f"aspect_too_wide:{ratio:.2f}"
                if ratio > ASPECT_MAX:
                    return f"aspect_too_tall:{ratio:.2f}"

    return None


def load_sources() -> pd.DataFrame:
    frames = []
    for p in SOURCE_METAS:
        if p.exists():
            df = pd.read_csv(p)
            log.info(f"  {p}: {len(df)} rows")
            frames.append(df)
        else:
            log.warning(f"  {p} not found — skipping")
    if not frames:
        raise FileNotFoundError("No metadata files found. Run scrapers first.")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-images", type=int, default=50)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip per-image file validation (faster re-runs)")
    args = parser.parse_args()

    log.info("Loading source metadata...")
    df = load_sources()
    log.info(f"Total rows: {len(df)}")

    # Keep only known classes and remove exact duplicates
    df = df[df["label"].isin(CLASSES)].copy()
    df = df.drop_duplicates(subset=["file_path"])
    log.info(f"After class filter + dedup: {len(df)}")

    # --- Content filter (second pass after scraper) ---
    log.info("Applying content filters (artist / title / aspect ratio)...")
    exclusions = []
    keep_mask  = []
    for _, row in df.iterrows():
        reason = filter_row(row)
        if reason:
            exclusions.append({
                "class":      row.get("label", ""),
                "style":      row.get("wikiart_style", ""),
                "artist":     row.get("artist", ""),
                "title":      row.get("title", ""),
                "source_url": row.get("source_url", ""),
                "reason":     reason,
                "stage":      "build_dataset",
            })
            log.info(f"  [excluded/{reason}] {row.get('artist','')} — {str(row.get('title',''))[:50]}")
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    if exclusions:
        excl_df = pd.DataFrame(exclusions)
        # Append to existing exclusion log if present
        if EXCLUDE_LOG.exists():
            existing = pd.read_csv(EXCLUDE_LOG)
            excl_df = pd.concat([existing, excl_df], ignore_index=True)
        excl_df.to_csv(EXCLUDE_LOG, index=False)
        log.info(f"Excluded {len(exclusions)} works — see {EXCLUDE_LOG}")

    df = df[keep_mask].copy()
    log.info(f"After content filter: {len(df)}")

    # --- Validate image files ---
    if not args.skip_validation:
        log.info("Validating image files...")
        valid_mask = df["file_path"].apply(validate_image)
        n_bad = (~valid_mask).sum()
        if n_bad:
            log.warning(f"Removing {n_bad} corrupted / missing images")
        df = df[valid_mask].copy()
        log.info(f"Valid images: {len(df)}")

    # --- Class distribution report ---
    log.info("\nClass distribution:")
    counts = df["label"].value_counts()
    for cls in CLASSES:
        n    = counts.get(cls, 0)
        flag = "  ⚠ BELOW MINIMUM" if n < args.min_images else ""
        log.info(f"  {cls:<22} {n:>4}{flag}")

    if df["label"].nunique() < len(CLASSES):
        missing = set(CLASSES) - set(df["label"].unique())
        log.warning(f"Missing classes: {missing}")

    # --- Stratified 80/20 split ---
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df["label"]
    )
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["split"] = "train"
    test_df["split"]  = "test"
    df["split"] = "unknown"
    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index,  "split"] = "test"

    df.to_csv(OUT_META, index=False)
    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST, index=False)
    log.info(f"\nSplit: {len(train_df)} train / {len(test_df)} test")
    log.info(f"Saved: {OUT_META}, {OUT_TRAIN}, {OUT_TEST}")


if __name__ == "__main__":
    main()
