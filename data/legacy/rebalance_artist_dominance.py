"""
Rebalance artist dominance in the local WikiArt dataset.

Steps:
  1. Cap configured overrepresented artists at 20 paintings per target class.
  2. Download targeted replacement artists from huggan/wikiart.
  3. Validate and resize all new images to 224x224.
  4. Write updated metadata.csv, train.csv, test.csv, and audit CSVs.

Usage:
  python data/legacy/rebalance_artist_dominance.py [--dry-run] [--skip-download]
  python data/legacy/rebalance_artist_dominance.py --download-source wikiart
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "metadata.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
REMOVED_PATH = DATA_DIR / "artist_cap_removed.csv"
ADDED_PATH = DATA_DIR / "artist_rebalance_added.csv"
BEFORE_PATH = DATA_DIR / "artist_distribution_before_rebalance.csv"
AFTER_PATH = DATA_DIR / "artist_distribution_after_rebalance.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ARTIST_PER_CLASS = 20

CLASSES = [
    "Renaissance", "Baroque", "Impressionism", "Expressionism",
    "Cubism", "Abstract", "Surrealism", "Pop Art",
]


@dataclass(frozen=True)
class CapRule:
    label: str
    artist: str
    cap: int = MAX_ARTIST_PER_CLASS


@dataclass(frozen=True)
class TargetRule:
    label: str
    artist: str
    target: int
    styles: tuple[str, ...]


@dataclass(frozen=True)
class WikiArtTargetRule:
    label: str
    artist: str
    target: int
    style_slugs: tuple[str, ...]


CAP_RULES = [
    CapRule("Renaissance", "Pietro Perugino"),
    CapRule("Impressionism", "Claude Monet"),
    CapRule("Impressionism", "Edgar Degas"),
    CapRule("Pop Art", "Alex Katz"),
    CapRule("Cubism", "Pablo Picasso"),
    CapRule("Expressionism", "Edvard Munch"),
    CapRule("Baroque", "Annibale Carracci"),
    CapRule("Baroque", "Agostino Carracci"),
    CapRule("Baroque", "Caravaggio"),
    CapRule("Cubism", "Paul Cezanne"),
    CapRule("Surrealism", "Max Ernst"),
    CapRule("Renaissance", "Leonardo da Vinci"),
    CapRule("Surrealism", "Joan Miro"),
    CapRule("Abstract", "Wassily Kandinsky"),
]

TARGET_RULES = [
    TargetRule("Renaissance", "Sandro Botticelli", 15, ("Early_Renaissance", "High_Renaissance")),
    TargetRule("Renaissance", "Raphael", 15, ("High_Renaissance",)),
    TargetRule("Renaissance", "Titian", 10, ("High_Renaissance", "Mannerism_Late_Renaissance")),
    TargetRule("Renaissance", "Albrecht Durer", 8, ("Northern_Renaissance",)),
    TargetRule("Renaissance", "Fra Angelico", 5, ("Early_Renaissance",)),
    TargetRule("Baroque", "Rembrandt", 20, ("Baroque",)),
    TargetRule("Baroque", "Vermeer", 10, ("Baroque",)),
    TargetRule("Baroque", "Diego Velazquez", 15, ("Baroque",)),
    TargetRule("Baroque", "Artemisia Gentileschi", 8, ("Baroque",)),
    TargetRule("Baroque", "Georges de La Tour", 8, ("Baroque",)),
    TargetRule("Impressionism", "Camille Pissarro", 15, ("Impressionism",)),
    TargetRule("Impressionism", "Alfred Sisley", 15, ("Impressionism",)),
    TargetRule("Impressionism", "Gustave Caillebotte", 15, ("Impressionism",)),
    TargetRule("Impressionism", "Mary Cassatt", 10, ("Impressionism",)),
    TargetRule("Impressionism", "Pierre-Auguste Renoir", 10, ("Impressionism",)),
    TargetRule("Impressionism", "Berthe Morisot", 10, ("Impressionism",)),
    TargetRule("Expressionism", "Ernst Ludwig Kirchner", 10, ("Expressionism",)),
    TargetRule("Expressionism", "Egon Schiele", 10, ("Expressionism",)),
    TargetRule("Expressionism", "Emil Nolde", 5, ("Expressionism",)),
    TargetRule("Expressionism", "Max Beckmann", 4, ("Expressionism",)),
    TargetRule("Cubism", "Juan Gris", 15, ("Cubism", "Synthetic_Cubism", "Analytical_Cubism")),
    TargetRule("Cubism", "Robert Delaunay", 10, ("Cubism", "Synthetic_Cubism", "Analytical_Cubism")),
    TargetRule("Cubism", "Fernand Leger", 10, ("Cubism", "Synthetic_Cubism", "Analytical_Cubism")),
    TargetRule("Cubism", "Georges Braque", 10, ("Cubism", "Synthetic_Cubism", "Analytical_Cubism")),
    TargetRule("Cubism", "Marcel Duchamp", 5, ("Cubism", "Synthetic_Cubism", "Analytical_Cubism")),
    TargetRule("Abstract", "Mark Rothko", 15, ("Color_Field_Painting", "Abstract_Expressionism")),
    TargetRule("Abstract", "Kazimir Malevich", 10, ("Suprematism", "Abstract_Art")),
    TargetRule("Abstract", "Franz Kline", 5, ("Abstract_Expressionism", "Action_painting")),
    TargetRule("Abstract", "Agnes Martin", 3, ("Minimalism",)),
    TargetRule("Surrealism", "Salvador Dali", 15, ("Surrealism",)),
    TargetRule("Surrealism", "Rene Magritte", 10, ("Surrealism",)),
    TargetRule("Surrealism", "Giorgio de Chirico", 5, ("Surrealism", "Metaphysical_art")),
    TargetRule("Surrealism", "Remedios Varo", 3, ("Surrealism",)),
    TargetRule("Pop Art", "David Hockney", 10, ("Pop_Art",)),
    TargetRule("Pop Art", "Roy Lichtenstein", 10, ("Pop_Art",)),
    TargetRule("Pop Art", "Andy Warhol", 10, ("Pop_Art",)),
    TargetRule("Pop Art", "Robert Indiana", 4, ("Pop_Art",)),
]

WIKIART_TARGET_RULES = [
    WikiArtTargetRule("Renaissance", "Sandro Botticelli", 15, ("early-renaissance", "high-renaissance")),
    WikiArtTargetRule("Renaissance", "Raphael", 15, ("high-renaissance",)),
    WikiArtTargetRule("Renaissance", "Titian", 10, ("high-renaissance", "mannerism-late-renaissance")),
    WikiArtTargetRule("Renaissance", "Albrecht Durer", 8, ("northern-renaissance",)),
    WikiArtTargetRule("Renaissance", "Fra Angelico", 5, ("early-renaissance",)),
    WikiArtTargetRule("Baroque", "Rembrandt", 20, ("baroque",)),
    WikiArtTargetRule("Baroque", "Vermeer", 10, ("baroque",)),
    WikiArtTargetRule("Baroque", "Diego Velazquez", 15, ("baroque",)),
    WikiArtTargetRule("Baroque", "Artemisia Gentileschi", 8, ("baroque",)),
    WikiArtTargetRule("Baroque", "Georges de La Tour", 8, ("baroque",)),
    WikiArtTargetRule("Impressionism", "Camille Pissarro", 15, ("impressionism",)),
    WikiArtTargetRule("Impressionism", "Alfred Sisley", 15, ("impressionism",)),
    WikiArtTargetRule("Impressionism", "Gustave Caillebotte", 15, ("impressionism",)),
    WikiArtTargetRule("Impressionism", "Mary Cassatt", 10, ("impressionism",)),
    WikiArtTargetRule("Impressionism", "Pierre-Auguste Renoir", 10, ("impressionism",)),
    WikiArtTargetRule("Impressionism", "Berthe Morisot", 10, ("impressionism",)),
    WikiArtTargetRule("Expressionism", "Ernst Ludwig Kirchner", 10, ("expressionism",)),
    WikiArtTargetRule("Expressionism", "Egon Schiele", 10, ("expressionism",)),
    WikiArtTargetRule("Expressionism", "Emil Nolde", 5, ("expressionism",)),
    WikiArtTargetRule("Expressionism", "Max Beckmann", 4, ("expressionism",)),
    WikiArtTargetRule("Cubism", "Juan Gris", 15, ("cubism",)),
    WikiArtTargetRule("Cubism", "Robert Delaunay", 10, ("cubism", "orphism")),
    WikiArtTargetRule("Cubism", "Fernand Leger", 10, ("cubism",)),
    WikiArtTargetRule("Cubism", "Georges Braque", 10, ("cubism",)),
    WikiArtTargetRule("Cubism", "Marcel Duchamp", 5, ("cubism",)),
    WikiArtTargetRule("Abstract", "Mark Rothko", 15, ("color-field-painting", "abstract-expressionism")),
    WikiArtTargetRule("Abstract", "Kazimir Malevich", 10, ("suprematism", "abstract-art")),
    WikiArtTargetRule("Abstract", "Franz Kline", 5, ("abstract-expressionism", "action-painting")),
    WikiArtTargetRule("Abstract", "Agnes Martin", 3, ("minimalism", "abstract-art")),
    WikiArtTargetRule("Surrealism", "Salvador Dali", 15, ("surrealism",)),
    WikiArtTargetRule("Surrealism", "Rene Magritte", 10, ("surrealism",)),
    WikiArtTargetRule("Surrealism", "Giorgio de Chirico", 5, ("surrealism", "metaphysical-art")),
    WikiArtTargetRule("Surrealism", "Remedios Varo", 3, ("surrealism",)),
    WikiArtTargetRule("Pop Art", "David Hockney", 10, ("pop-art",)),
    WikiArtTargetRule("Pop Art", "Roy Lichtenstein", 10, ("pop-art",)),
    WikiArtTargetRule("Pop Art", "Andy Warhol", 10, ("pop-art",)),
    WikiArtTargetRule("Pop Art", "Robert Indiana", 4, ("pop-art",)),
]

WIKIART_DELAY = 1.2
WIKIART_MAX_PAGES_PER_STYLE = 12
WIKIART_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ArtClassifier-Research/1.0; academic use)",
    "Accept": "application/json, text/html",
    "Referer": "https://www.wikiart.org/",
}

EXCLUDED_GENRES = {"illustration", "sketch_and_study"}
TITLE_EXCLUDE = re.compile(
    r"\b(basilica|cathedral|church|chapel|palazzo|palais|palace|facade|fa[çc]ade"
    r"|monastery|convent|abbey|temple|mosque|synagogue|baptistery|campanile"
    r"|loggia|architectural|architecture|floor plan|ground plan|elevation"
    r"|sculpture|sculpted|bust|relief|statue|statuette|figurine|monument"
    r"|cenotaph|tomb|mausoleum|sarcophagus|engraving|etching|lithograph"
    r"|woodcut|woodblock|drypoint|aquatint|tapestry|embroidery|textile"
    r"|weaving|mosaic|stained glass|photograph|photography|poster|design)\b",
    re.IGNORECASE,
)


def norm_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "untitled"


def artist_matches(series: pd.Series, artist: str) -> pd.Series:
    wanted = norm_name(artist)
    return series.fillna("").map(norm_name).eq(wanted)


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["label", "artist"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["label", "count", "artist"], ascending=[True, False, True])
    )


def cap_dominant_artists(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    removed_parts: list[pd.DataFrame] = []
    working = df.copy()

    for rule in CAP_RULES:
        mask = (working["label"] == rule.label) & artist_matches(working["artist"], rule.artist)
        rows = working[mask].copy()
        if len(rows) <= rule.cap:
            log.info("  %-18s %-24s %3d <= %3d", rule.label, rule.artist, len(rows), rule.cap)
            continue

        test_rows = rows[rows["split"].eq("test")]
        train_rows = rows[~rows["split"].eq("test")]
        if len(test_rows) >= rule.cap:
            keep_idx = test_rows.sample(n=rule.cap, random_state=RANDOM_STATE).index
        else:
            need = rule.cap - len(test_rows)
            train_idx = train_rows.sample(n=need, random_state=RANDOM_STATE).index
            keep_idx = test_rows.index.union(train_idx)

        remove_idx = rows.index.difference(keep_idx)
        removed = working.loc[remove_idx].copy()
        removed["cap_label"] = rule.label
        removed["cap_artist"] = rule.artist
        removed["cap_limit"] = rule.cap
        removed_parts.append(removed)
        working = working.drop(index=remove_idx)
        log.info("  %-18s %-24s %3d -> %3d  removed=%d",
                 rule.label, rule.artist, len(rows), rule.cap, len(remove_idx))

    removed_df = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()
    return working.reset_index(drop=True), removed_df


def cap_all_artists(df: pd.DataFrame, cap: int = MAX_ARTIST_PER_CLASS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same artist cap to every class after additions."""
    removed_parts: list[pd.DataFrame] = []
    working = df.copy()
    artist_keys = working["artist"].fillna("").map(norm_name)

    groups = (
        working.assign(_artist_key=artist_keys)
        .groupby(["label", "_artist_key"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    groups = groups[(groups["_artist_key"].ne("")) & (groups["count"] > cap)]

    for _, group in groups.iterrows():
        label = group["label"]
        artist_key = group["_artist_key"]
        rows = working[(working["label"].eq(label)) & (working["artist"].fillna("").map(norm_name).eq(artist_key))]
        artist_name = str(rows["artist"].dropna().iloc[0]) if rows["artist"].notna().any() else artist_key

        test_rows = rows[rows["split"].eq("test")]
        train_rows = rows[~rows["split"].eq("test")]
        if len(test_rows) >= cap:
            keep_idx = test_rows.sample(n=cap, random_state=RANDOM_STATE).index
        else:
            need = cap - len(test_rows)
            keep_idx = test_rows.index.union(train_rows.sample(n=need, random_state=RANDOM_STATE).index)

        remove_idx = rows.index.difference(keep_idx)
        removed = working.loc[remove_idx].copy()
        removed["cap_label"] = label
        removed["cap_artist"] = artist_name
        removed["cap_limit"] = cap
        removed_parts.append(removed)
        working = working.drop(index=remove_idx)
        log.info("  final cap %-18s %-24s %3d -> %3d  removed=%d",
                 label, artist_name, len(rows), cap, len(remove_idx))

    removed_df = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()
    return working.reset_index(drop=True), removed_df


def split_metadata(df: pd.DataFrame, preserve_existing: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.reset_index(drop=True).copy()
    if preserve_existing and "split" in df.columns:
        new_mask = df["split"].isna() | df["split"].eq("") | df["split"].eq("new")
        old_mask = ~new_mask
    else:
        new_mask = pd.Series(True, index=df.index)
        old_mask = ~new_mask

    if new_mask.any():
        new_df = df[new_mask].copy()
        if len(new_df) >= 2 and new_df["label"].nunique() > 1:
            train_new, test_new = train_test_split(
                new_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=new_df["label"]
            )
        else:
            train_new, test_new = train_test_split(new_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        df.loc[train_new.index, "split"] = "train"
        df.loc[test_new.index, "split"] = "test"

    train_df = df[df["split"].eq("train")].copy()
    test_df = df[df["split"].eq("test")].copy()
    return df, train_df, test_df


def get_label_names(ds: Any, column: str) -> list[str] | None:
    feature = ds.features.get(column)
    return getattr(feature, "names", None)


def decode(value: Any, names: list[str] | None) -> str:
    if names is not None and isinstance(value, int):
        return names[value]
    return str(value or "")


def item_title(item: dict[str, Any]) -> str:
    for key in ("title", "painting_name", "name"):
        if item.get(key):
            return str(item[key])
    return ""


def item_year(item: dict[str, Any]) -> str:
    for key in ("year", "date"):
        if item.get(key) is not None:
            return str(item[key])
    return ""


def is_painting_candidate(title: str, genre_name: str) -> bool:
    if genre_name in EXCLUDED_GENRES:
        return False
    if title and TITLE_EXCLUDE.search(title):
        return False
    return True


def resize_and_save(img: Image.Image, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = ImageOps.fit(img, (224, 224), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        img.save(dest, "JPEG", quality=92, optimize=True)
        with Image.open(dest) as check:
            check.verify()
        return True
    except Exception as exc:
        log.debug("save failed for %s: %s", dest, exc)
        return False


def build_filename(label: str, artist: str, title: str, img: Image.Image) -> Path:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    digest = hashlib.md5(buf.getvalue()).hexdigest()[:10]
    stem = f"hf_{slugify(artist)}_{slugify(title)[:72]}_{digest}"
    return IMAGES_DIR / label / f"{stem}.jpg"


def build_wikiart_filename(label: str, artist: str, title: str, image_bytes: bytes) -> Path:
    digest = hashlib.md5(image_bytes).hexdigest()[:10]
    stem = f"wikiart_{slugify(artist)}_{slugify(title)[:72]}_{digest}"
    return IMAGES_DIR / label / f"{stem}.jpg"


def known_image_hashes(prefixes: tuple[str, ...] = ("hf_", "wikiart_")) -> set[str]:
    hashes: set[str] = set()
    for path in IMAGES_DIR.glob("*/*"):
        if not path.is_file() or not path.name.startswith(prefixes):
            continue
        try:
            hashes.add(hashlib.md5(path.read_bytes()).hexdigest())
        except OSError:
            pass
    return hashes


def download_targets(existing_df: pd.DataFrame) -> pd.DataFrame:
    from datasets import load_dataset

    log.info("Loading huggan/wikiart from HuggingFace...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    style_names = get_label_names(ds, "style")
    genre_names = get_label_names(ds, "genre")
    artist_names = get_label_names(ds, "artist")

    targets_by_artist = {norm_name(t.artist): t for t in TARGET_RULES}
    existing_targeted = existing_df[
        existing_df.get("source", pd.Series("", index=existing_df.index)).isin({"huggan_wikiart", "wikiart_targeted"})
    ]
    existing_counts = count_added_targets(existing_targeted)
    counts = {
        norm_name(t.artist): existing_counts.get((t.label, norm_name(t.artist)), 0)
        for t in TARGET_RULES
    }
    existing_paths = set(existing_df["file_path"].astype(str))
    existing_hashes = known_image_hashes()
    records: list[dict[str, Any]] = []

    for item in ds:
        artist = decode(item.get("artist"), artist_names).replace("-", " ").title()
        key = norm_name(artist)
        target = targets_by_artist.get(key)
        if target is None or counts[key] >= target.target:
            continue

        style_name = decode(item.get("style"), style_names)
        if target.styles and style_name not in target.styles:
            continue

        genre_name = decode(item.get("genre"), genre_names)
        title = item_title(item)
        if not is_painting_candidate(title, genre_name):
            continue

        img = item.get("image")
        if img is None:
            continue

        img_buf = io.BytesIO()
        img.convert("RGB").save(img_buf, "JPEG", quality=85)
        content_hash = hashlib.md5(img_buf.getvalue()).hexdigest()
        if content_hash in existing_hashes:
            continue

        dest = build_filename(target.label, target.artist, title or f"work_{counts[key] + 1}", img)
        rel_dest = dest.relative_to(BASE_DIR).as_posix()
        if rel_dest in existing_paths or dest.exists():
            continue
        if not resize_and_save(img, dest):
            continue

        existing_hashes.add(content_hash)
        counts[key] += 1
        source_url = str(item.get("image_url") or item.get("url") or "")
        records.append({
            "file_path": rel_dest,
            "label": target.label,
            "source": "huggan_wikiart",
            "artist": target.artist,
            "title": title,
            "year": item_year(item),
            "source_url": source_url,
            "wikiart_style": style_name,
            "split": "new",
        })

        if counts[key] == target.target or counts[key] % 5 == 0:
            log.info("  %-18s %-24s %2d/%2d",
                     target.label, target.artist, counts[key], target.target)

        if all(counts[norm_name(t.artist)] >= t.target for t in TARGET_RULES):
            break

    missing = [
        f"{t.label}: {t.artist} {counts[norm_name(t.artist)]}/{t.target}"
        for t in TARGET_RULES
        if counts[norm_name(t.artist)] < t.target
    ]
    if missing:
        log.warning("Targets not fully satisfied:\n  %s", "\n  ".join(missing))

    return pd.DataFrame(records)


def count_added_targets(added_df: pd.DataFrame, source: str | None = None) -> dict[tuple[str, str], int]:
    if added_df.empty:
        return {}
    df = added_df.copy()
    if source is not None and "source" in df.columns:
        df = df[df["source"].eq(source)]
    counts: dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        counts[(str(row["label"]), norm_name(str(row["artist"])))] = (
            counts.get((str(row["label"]), norm_name(str(row["artist"]))), 0) + 1
        )
    return counts


def wikiart_large_url(url: str) -> str:
    if not url:
        return ""
    if "!" in url:
        return re.sub(r"![^!/]+$", "!Large.jpg", url)
    if url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        stem, suffix = url.rsplit(".", 1)
        return f"{stem}!Large.{suffix}"
    return url


def fetch_wikiart_style_page(session: requests.Session, style_slug: str, page: int) -> list[dict[str, Any]]:
    url = f"https://www.wikiart.org/en/paintings-by-style/{style_slug}"
    params = {
        "json": 2,
        "layout": "new",
        "page": page,
        "resultType": "masonry",
    }
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    paintings = data.get("Paintings") or data.get("paintings") or []
    return paintings if isinstance(paintings, list) else []


def wikiart_artist_name(item: dict[str, Any]) -> str:
    for key in ("artistName", "artistNameOriginal", "artistUrl", "ArtistName"):
        value = item.get(key)
        if value:
            text = str(value)
            if "/" in text:
                text = text.rstrip("/").split("/")[-1].replace("-", " ")
            return text
    return ""


def wikiart_item_title(item: dict[str, Any]) -> str:
    for key in ("title", "Title", "paintingName"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def wikiart_source_url(item: dict[str, Any]) -> str:
    for key in ("paintingUrl", "url", "paintingUrlJson"):
        value = item.get(key)
        if value:
            text = str(value)
            if text.startswith("/"):
                return f"https://www.wikiart.org{text}"
            return text
    return ""


def wikiart_image_url(item: dict[str, Any]) -> str:
    for key in ("image", "imageUrl", "imageUrlLarge", "thumbnailUrl"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def download_wikiart_image(session: requests.Session, item: dict[str, Any], dest: Path) -> bytes | None:
    raw_url = wikiart_image_url(item)
    for url in dict.fromkeys([wikiart_large_url(raw_url), raw_url]):
        if not url:
            continue
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            img_bytes = response.content
            with Image.open(io.BytesIO(img_bytes)) as img:
                if resize_and_save(img, dest):
                    return img_bytes
        except Exception as exc:
            log.debug("WikiArt image download failed for %s: %s", url, exc)
    return None


def download_wikiart_targets(existing_df: pd.DataFrame, previous_added_df: pd.DataFrame | None = None) -> pd.DataFrame:
    existing_targeted = existing_df[
        existing_df.get("source", pd.Series("", index=existing_df.index)).isin({"huggan_wikiart", "wikiart_targeted"})
    ]
    previous_counts = count_added_targets(existing_targeted)
    for key, value in count_added_targets(previous_added_df if previous_added_df is not None else pd.DataFrame()).items():
        previous_counts[key] = previous_counts.get(key, 0) + value
    counts = {
        (rule.label, norm_name(rule.artist)): previous_counts.get((rule.label, norm_name(rule.artist)), 0)
        for rule in WIKIART_TARGET_RULES
    }
    existing_paths = set(existing_df["file_path"].astype(str))
    existing_hashes = known_image_hashes()
    records: list[dict[str, Any]] = []

    session = requests.Session()
    session.headers.update(WIKIART_HEADERS)

    log.info("Filling remaining targets from WikiArt...")
    for target in WIKIART_TARGET_RULES:
        key = (target.label, norm_name(target.artist))
        if counts[key] >= target.target:
            continue

        for style_slug in target.style_slugs:
            if counts[key] >= target.target:
                break
            for page in range(1, WIKIART_MAX_PAGES_PER_STYLE + 1):
                if counts[key] >= target.target:
                    break
                try:
                    paintings = fetch_wikiart_style_page(session, style_slug, page)
                except Exception as exc:
                    log.warning("WikiArt fetch failed for %s page %d: %s", style_slug, page, exc)
                    break
                if not paintings:
                    break

                for item in paintings:
                    if counts[key] >= target.target:
                        break
                    artist = wikiart_artist_name(item)
                    if norm_name(artist) != norm_name(target.artist):
                        continue

                    title = wikiart_item_title(item)
                    if not is_painting_candidate(title, ""):
                        continue

                    raw_url = wikiart_image_url(item)
                    if not raw_url:
                        continue
                    url_for_hash = wikiart_large_url(raw_url) or raw_url
                    dest = build_wikiart_filename(
                        target.label,
                        target.artist,
                        title or f"work_{counts[key] + 1}",
                        url_for_hash.encode("utf-8"),
                    )
                    rel_dest = dest.relative_to(BASE_DIR).as_posix()
                    if rel_dest in existing_paths or dest.exists():
                        continue

                    img_bytes = download_wikiart_image(session, item, dest)
                    if img_bytes is None:
                        continue
                    content_hash = hashlib.md5(dest.read_bytes()).hexdigest()
                    if content_hash in existing_hashes:
                        dest.unlink(missing_ok=True)
                        continue

                    existing_paths.add(rel_dest)
                    existing_hashes.add(content_hash)
                    counts[key] += 1
                    records.append({
                        "file_path": rel_dest,
                        "label": target.label,
                        "source": "wikiart_targeted",
                        "artist": target.artist,
                        "title": title,
                        "year": str(item.get("year") or item.get("date") or ""),
                        "source_url": wikiart_source_url(item),
                        "wikiart_style": style_slug,
                        "split": "new",
                    })

                    if counts[key] == target.target or counts[key] % 5 == 0:
                        log.info("  %-18s %-24s %2d/%2d",
                                 target.label, target.artist, counts[key], target.target)
                time.sleep(WIKIART_DELAY)

    missing = [
        f"{t.label}: {t.artist} {counts[(t.label, norm_name(t.artist))]}/{t.target}"
        for t in WIKIART_TARGET_RULES
        if counts[(t.label, norm_name(t.artist))] < t.target
    ]
    if missing:
        log.warning("WikiArt targets not fully satisfied:\n  %s", "\n  ".join(missing))

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--download-source",
        choices=("both", "huggan", "wikiart"),
        default="both",
        help="Download from HuggingFace, WikiArt, or HuggingFace followed by WikiArt fallback.",
    )
    args = parser.parse_args()

    df = pd.read_csv(META_PATH)
    distribution(df).to_csv(BEFORE_PATH, index=False)
    log.info("Loaded %d paintings", len(df))

    capped_df, removed_df = cap_dominant_artists(df)
    log.info("After capping: %d paintings (removed %d)", len(capped_df), len(removed_df))

    added_df = pd.DataFrame()
    if not args.skip_download:
        added_parts: list[pd.DataFrame] = []
        if args.download_source in {"both", "huggan"}:
            added_parts.append(download_targets(capped_df))
        if args.download_source in {"both", "wikiart"}:
            previous = pd.concat([p for p in added_parts if not p.empty], ignore_index=True) if added_parts else pd.DataFrame()
            added_parts.append(download_wikiart_targets(capped_df, previous))
        added_df = pd.concat([p for p in added_parts if not p.empty], ignore_index=True) if added_parts else pd.DataFrame()
        log.info("Downloaded %d targeted replacement paintings", len(added_df))

    final_df = pd.concat([capped_df, added_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["file_path"]).reset_index(drop=True)
    final_df, extra_removed_df = cap_all_artists(final_df)
    if not extra_removed_df.empty:
        removed_df = pd.concat([removed_df, extra_removed_df], ignore_index=True)
    final_df, train_df, test_df = split_metadata(final_df, preserve_existing=True)
    distribution(final_df).to_csv(AFTER_PATH, index=False)

    log.info("Final class distribution:\n%s", final_df["label"].value_counts().reindex(CLASSES).to_string())
    log.info("Final split: train=%d test=%d", len(train_df), len(test_df))

    if args.dry_run:
        log.info("Dry run; no metadata files written.")
        return

    final_df.to_csv(META_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    removed_df.to_csv(REMOVED_PATH, index=False)
    added_df.to_csv(ADDED_PATH, index=False)
    log.info("Saved updated metadata and audit CSVs.")


if __name__ == "__main__":
    main()
