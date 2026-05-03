"""
Cap overrepresented artists at max N paintings each.
Regenerates train/test splits from the trimmed metadata.

Usage:
  python data/legacy/cap_artists.py [--cap 20] [--dry-run]
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

BASE_DIR   = Path(__file__).parent.parent
META_PATH  = BASE_DIR / "data/metadata.csv"
TRAIN_PATH = BASE_DIR / "data/train.csv"
TEST_PATH  = BASE_DIR / "data/test.csv"

CLASSES = [
    "Renaissance", "Baroque", "Impressionism", "Expressionism",
    "Cubism", "Abstract", "Surrealism", "Pop Art",
]

# Artists to cap and their per-class cap (artist name substring, cap)
CAPS = [
    ("Pietro Perugino",   20),
    ("Claude Monet",      20),
    ("Edgar Degas",       20),
    ("Alex Katz",         20),
    ("Pablo Picasso",     20),
    ("Edvard Munch",      20),
    ("Annibale Carracci", 20),
    ("Agostino Carracci", 20),
    ("Caravaggio",        20),
    ("Paul Cezanne",      20),
    ("Max Ernst",         20),
    ("Leonardo da Vinci", 20),
    ("Joan Miro",         20),
    ("Wassily Kandinsky", 20),
]


def cap_artist(df: pd.DataFrame, artist_kw: str, cap: int) -> pd.DataFrame:
    """Return df with at most `cap` rows whose artist contains artist_kw."""
    mask   = df["artist"].str.contains(artist_kw, case=False, na=False)
    over   = df[mask]
    under  = df[~mask]

    if len(over) <= cap:
        return df

    # Prefer keeping test-split rows so held-out evaluation is stable
    test_rows  = over[over["split"] == "test"]
    train_rows = over[over["split"] == "train"]

    if len(test_rows) >= cap:
        kept = test_rows.sample(n=cap, random_state=42)
    else:
        need_train = cap - len(test_rows)
        kept = pd.concat([
            test_rows,
            train_rows.sample(n=min(need_train, len(train_rows)), random_state=42),
        ])

    removed = len(over) - len(kept)
    log.info("  %-24s  %3d → %3d  (removed %d)", artist_kw, len(over), len(kept), removed)
    return pd.concat([under, kept]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap",     type=int,  default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(META_PATH)
    log.info("Before: %d paintings", len(df))
    log.info("Class distribution:\n%s", df["label"].value_counts().reindex(CLASSES).to_string())

    log.info("\nCapping artists at %d:", args.cap)
    for artist_kw, cap in CAPS:
        df = cap_artist(df, artist_kw, min(cap, args.cap))

    log.info("\nAfter: %d paintings", len(df))
    log.info("Class distribution:\n%s", df["label"].value_counts().reindex(CLASSES).to_string())
    log.info("\nTop artists per class after capping:")
    for cls in CLASSES:
        top = df[df["label"] == cls]["artist"].value_counts().head(5)
        log.info("  %s: %s", cls, dict(top))

    if args.dry_run:
        log.info("Dry run — no files written.")
        return

    # Fresh stratified split
    df = df.reset_index(drop=True)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["split"] = "train"
    test_df["split"]  = "test"
    df["split"] = "unknown"
    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index,  "split"] = "test"

    df.to_csv(META_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,  index=False)
    log.info("Saved: %s  |  train=%d  test=%d", META_PATH, len(train_df), len(test_df))


if __name__ == "__main__":
    main()
