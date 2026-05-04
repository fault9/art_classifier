"""
Arnheim perceptual dimension analysis for painting art movement classification.

Scores every painting on 6 Arnheim dimensions (Balance, Shape, Depth, Tension,
Light, Color) using anchor-based projection in CLIP embedding space.

Usage:
  python arnheim_analysis.py [--skip-encode] [--batch-size 32]
"""

import argparse
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# All paths relative to the script's own directory so it runs from any CWD.
BASE_DIR   = Path(__file__).parent
DATA_CSV   = BASE_DIR / "data/metadata.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs/arnheim"
CACHE_PATH = MODELS_DIR / "embeddings_clip.npz"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "Renaissance", "Baroque", "Impressionism", "Expressionism",
    "Cubism", "Abstract", "Surrealism", "Pop Art",
]

# ---------------------------------------------------------------------------
# Wölfflin theoretical positions — copied verbatim from train.py
# ---------------------------------------------------------------------------
WOLFFLIN_THEORY = {
    "Renaissance":   [-1.0, -0.8, -0.9, -0.7, -1.0],
    "Baroque":       [ 0.9,  0.8,  0.7,  0.8,  0.6],
    "Impressionism": [ 0.9,  0.3,  0.6,  0.5,  0.5],
    "Expressionism": [ 0.6,  0.2,  0.7,  0.7,  0.4],
    "Cubism":        [-0.5, -0.7, -0.3, -0.4, -0.3],
    "Abstract":      [ 0.7, -0.1,  0.8,  0.8,  0.7],
    "Surrealism":    [-0.3,  0.2,  0.1,  0.2,  0.3],
    "Pop Art":       [-0.8, -0.5, -0.6, -0.5, -0.7],
}
WOLFFLIN_AXES = ["Linear↔Painterly", "Plane↔Recession", "Closed↔Open",
                 "Multiplicity↔Unity", "Clearness↔Unclearness"]

# ---------------------------------------------------------------------------
# Arnheim dimension definitions — painting-level anchors
#
# Each anchor entry is either:
#   str                  → match all paintings by that artist
#   (artist_kw, title_kw) → match paintings where BOTH artist and title
#                           contain the given substrings (case-insensitive).
#                           title_kw=None means artist-level (same as str form).
#
# Only artists confirmed in the 1436-painting dataset are used.
# ---------------------------------------------------------------------------
DIMENSIONS = {
    "Balance": {
        "description": "High = symmetrical/centered composition, Low = asymmetrical/dynamic",
        # HIGH: Renaissance altarpiece masters whose compositions are rigidly centred and
        # hierarchical (Perugino, Cima, Bellini, Carpaccio). Pop Art iconic imagery with
        # symmetrical frontal presentation (Warhol soup cans, Johns flags/targets).
        "high": [
            "Pietro Perugino",          # centred Madonna/saint altarpieces throughout
            "Cima da Conegliano",       # frontal devotional altarpieces
            "Giovanni Bellini",         # centred devotional Madonnas
            ("Vittore Carpaccio", "Dream of St"),  # sleeping figure, perfectly centred
            ("Leonardo da Vinci", "Lady with an Ermine"),
            ("Leonardo da Vinci", "Madonna Litta"),
            ("Andy Warhol", "Campbell"),
            ("Andy Warhol", "100 Cans"),
            ("Jasper Johns", "Three Flags"),
            ("Jasper Johns", "Green Target"),
        ],
        # LOW: Impressionist snapshot framing (Degas off-centre racing/dance scenes, Monet
        # atmospheric landscapes with no focal centre); gestural Abstract Expressionism
        # with centrifugal all-over energy (Pollock); Munch's off-kilter psychological
        # compositions; Miro's scattered biomorphic forms.
        "low": [
            ("Edgar Degas", "Races"),
            ("Edgar Degas", "Jockey"),
            ("Edgar Degas", "Steeplechase"),
            "Claude Monet",             # landscapes — no dominant centre
            "Jackson Pollock",          # all-over, centrifugal
            ("Edvard Munch", "Scream"),
            ("Edvard Munch", "Anxiety"),
            ("Edvard Munch", "Despair"),
            "Joan Miro",                # scattered biomorphic
            "Max Ernst",                # collage chaos
        ],
    },
    "Shape": {
        "description": "High = geometric/regular forms, Low = organic/fluid forms",
        # HIGH: Cubist plane-fragmentation (Braque, Picasso Cubist works, Leger, Mondrian);
        # mathematical graphic construction (Escher); flat hard-edged Pop (Jasper Johns flags/targets,
        # Stuart Davis geometric pop, Lichtenstein comic panels).
        "high": [
            "Georges Braque",           # Cubist geometric fragmentation throughout
            "Fernand Leger",            # mechanical tubular geometry
            "Piet Mondrian",            # De Stijl grid
            ("Pablo Picasso", "Reservoir"),
            ("Pablo Picasso", "Head of a Woman"),
            ("Pablo Picasso", "Guitar"),
            ("Pablo Picasso", "Plaster head"),
            "M.C. Escher",              # mathematical precision
            "Stuart Davis",             # flat hard-edged graphic Pop
            ("Jasper Johns", "Flag"),
            ("Jasper Johns", "Target"),
        ],
        # LOW: Impressionist dissolved-edge painting (Monet, Renoir, Morisot, Boudin);
        # gestural AbstractExpressionist mark-making that is maximally anti-geometric
        # (Pollock drip, Gorky biomorphic); Rubens' flowing fleshy baroque forms.
        "low": [
            "Claude Monet",
            "Pierre-Auguste Renoir",
            "Berthe Morisot",
            "Eugene Boudin",
            "Jackson Pollock",
            "Arshile Gorky",
            "Peter Paul Rubens",
            "Johan Jongkind",
            "Frederic Bazille",
        ],
    },
    "Depth": {
        "description": "High = deep spatial recession, Low = flat/planar",
        # HIGH: Renaissance linear-perspective specialists who construct deep architectural
        # and landscape recessions (Perugino, Carpaccio, Mantegna, Leonardo);
        # Baroque atmospheric landscape painters (Bril, Elsheimer, Annibale Carracci).
        "high": [
            "Pietro Perugino",          # deep perspective arches throughout
            "Vittore Carpaccio",        # deep harbour and narrative scenes
            "Leonardo da Vinci",        # atmospheric sfumato recession
            "Andrea Mantegna",          # steep foreshortening, deep architectural space
            "Paul Bril",                # deep rolling landscape vistas
            "Adam Elsheimer",           # deep nocturnal atmospheric landscapes
            ("Annibale Carracci", "River"),
            ("Annibale Carracci", "Flight into Egypt"),
            "Luca Signorelli",
            "Filippino Lippi",
        ],
        # LOW: All-over Abstract canvases that deny any pictorial depth (Pollock, Kandinsky,
        # Miro, Morris Graves); Pop Art flat screen-print surface (Warhol, Lichtenstein, Johns,
        # Stuart Davis); De Stijl flat grid (Mondrian).
        "low": [
            "Jackson Pollock",
            "Andy Warhol",
            "Roy Lichtenstein",
            "Wassily Kandinsky",
            "Joan Miro",
            "Jasper Johns",
            "Stuart Davis",
            "Piet Mondrian",
            "Morris Graves",
            "Hassan Massoudy",
        ],
    },
    "Tension": {
        "description": "High = energetic/dynamic movement, Low = static/calm",
        # HIGH: Cross-movement but painting-specific anchors for violent narrative,
        # psychological pressure, gestural energy, and graphic action. Broad artist
        # anchors are avoided here because many artists also have calm portraits,
        # landscapes, or still lifes that weaken the axis.
        "high": [
            # Baroque dramatic action
            ("Peter Paul Rubens", "Laocoon"),
            ("Peter Paul Rubens", "Battle of the Amazons"),
            ("Peter Paul Rubens", "Battle of Anghiari"),
            ("Peter Paul Rubens", "Equestrian Portrait"),
            ("Caravaggio", "Judith"),
            ("Caravaggio", "Conversion of Saint Paul"),
            ("Caravaggio", "Taking of Christ"),
            ("Caravaggio", "Medusa"),
            ("Caravaggio", "Crowning with Thorns"),
            ("Adam Elsheimer", "Judith"),
            ("Adam Elsheimer", "Martyrdom"),
            ("Adam Elsheimer", "Great Flood"),
            ("Adam Elsheimer", "fire of Troy"),
            ("Annibale Carracci", "Mocking of Christ"),
            ("Annibale Carracci", "Martyrdom"),
            ("Annibale Carracci", "Samson"),
            ("Agostino Carracci", "Bachus"),
            # Expressionist anxiety and social conflict
            ("Edvard Munch", "Scream"),
            ("Edvard Munch", "Despair"),
            ("Edvard Munch", "Vampire"),
            ("Edvard Munch", "Storm"),
            ("James Ensor", "Death and the Masks"),
            ("James Ensor", "Skeletons Fighting"),
            ("James Ensor", "Christ's Entry"),
            ("James Ensor", "Calvary"),
            ("Kathe Kollwitz", "Death"),
            ("Kathe Kollwitz", "Revolt"),
            ("Kathe Kollwitz", "March of the Weavers"),
            # Abstract and Surrealist energy/distortion
            ("Jackson Pollock", "Moon-Woman"),
            ("Jackson Pollock", "Stenographic"),
            ("Jackson Pollock", "Mural"),
            ("Jackson Pollock", "She-Wolf"),
            ("Arshile Gorky", "Liver is the Cock"),
            ("Wassily Kandinsky", "Composition VII"),
            ("Pablo Picasso", "dance"),
            ("Pablo Picasso", "Kiss"),
            ("Salvador Dali", "Honey Is Sweeter Than Blood"),
            # Pop/graphic action
            ("Roy Lichtenstein", "Brattata"),
            ("Eduardo Paolozzi", "Sack-o-sauce"),
        ],
        # LOW: Calm, stable anchors across movements: Renaissance devotional balance,
        # quiet Impressionist landscapes/domestic scenes, restrained interiors, and
        # static Pop still lifes.
        "low": [
            ("Fra Bartolomeo", "Holy Family"),
            ("Cima da Conegliano", "St. Helena"),
            ("Cima da Conegliano", "Annunciation"),
            ("Cima da Conegliano", "Madonna and Child"),
            ("Pietro Perugino", "Pieta"),
            ("Pietro Perugino", "Nativity"),
            ("Pietro Perugino", "Assumption"),
            ("Giovanni Bellini", "Madonna"),
            ("Leonardo da Vinci", "Lady with an Ermine"),
            ("Claude Monet", "Regatta"),
            ("Claude Monet", "Fishing Boats"),
            ("Claude Monet", "Beach at Honfleur"),
            ("Camille Pissarro", "Village Corner"),
            ("Berthe Morisot", "Young Lady Seated"),
            ("Berthe Morisot", "Farm in Normandy"),
            ("Pierre-Auguste Renoir", "Sleeping Cat"),
            ("Pierre-Auguste Renoir", "Spring Flowers"),
            ("Carl Holsoe", "Woman in an Interior"),
            ("Amedeo Modigliani", "Tuscan Road"),
            ("Paul Bril", "River Landscape"),
            ("Annibale Carracci", "River Landscape"),
            ("Annibale Carracci", "Flight into Egypt"),
            ("Wayne Thiebaud", "Bakery Counter"),
            ("Wayne Thiebaud", "Pies"),
            ("Mark Rothko", "Untitled"),
            ("Agnes Martin", "Desert Rain"),
        ],
    },
    "Light": {
        "description": "High = dramatic contrast/chiaroscuro, Low = even/diffused light",
        # HIGH: Caravaggio and the Baroque tenebrism tradition (Annibale/Agostino Carracci,
        # Adam Elsheimer's candlelit nocturnes, Guido Reni); Expressionist high-contrast
        # darkness (Kollwitz, Munch, Ensor).
        "high": [
            ("Caravaggio", "Calling of Saint Matthew"),
            ("Caravaggio", "Judith"),
            ("Caravaggio", "Supper at Emmaus"),
            ("Caravaggio", "Inspiration"),
            ("Caravaggio", "Taking of Christ"),
            ("Adam Elsheimer", "Jacob"),
            ("Adam Elsheimer", "Saint Christopher"),
            ("Adam Elsheimer", "Glorification"),
            "Annibale Carracci",
            "Agostino Carracci",
            "Guido Reni",
            "Kathe Kollwitz",
            "James Ensor",
        ],
        # LOW: Open-air Impressionists painting in diffuse outdoor light with no directional
        # shadows (Monet, Boudin, Renoir, Morisot, Bazille); Pop Art flat shadowless colour
        # surfaces (Warhol, Lichtenstein, Alex Katz, Stuart Davis).
        "low": [
            "Claude Monet",
            "Eugene Boudin",
            "Frederic Bazille",
            "Pierre-Auguste Renoir",
            "Berthe Morisot",
            "Andy Warhol",
            "Roy Lichtenstein",
            "Alex Katz",
            "Stuart Davis",
            "Johan Jongkind",
        ],
    },
    "Color": {
        "description": "High = vivid/saturated, Low = muted/tonal",
        # HIGH: Painting-specific vivid colour anchors across Pop, Abstract,
        # Surrealism, Expressionism, and Orphic/Fauvist Cubism. This avoids using
        # broad artist pools where the same artist may also have muted work.
        "high": [
            ("Stuart Davis", "Owh"),
            ("Stuart Davis", "Mellow Pad"),
            ("Stuart Davis", "Visa"),
            ("Stuart Davis", "Colonial Cubism"),
            ("Hiro Yamagata", "Bubbles"),
            ("Hiro Yamagata", "American in Paris"),
            ("Hiro Yamagata", "Holographic"),
            ("Roy Lichtenstein", "Girl with ball"),
            ("Roy Lichtenstein", "Mickey"),
            ("Roy Lichtenstein", "Brattata"),
            ("Robert Indiana", "Four Winds"),
            ("Henri Matisse", "Destiny"),
            ("Henri Matisse", "Codomas"),
            ("Henri Matisse", "Circus"),
            ("Henri Matisse", "Lagoon"),
            ("Henri Matisse", "Icarus"),
            ("Joan Miro", "The Air"),
            ("Joan Miro", "Landscape"),
            ("Joan Miro", "Stars"),
            ("Joan Miro", "Dancer"),
            ("Wassily Kandinsky", "Composition VII"),
            ("Wassily Kandinsky", "Red Sun"),
            ("Wassily Kandinsky", "Glass Painting"),
            ("Robert Delaunay", "Rhythm"),
            ("Robert Delaunay", "Window"),
            ("Marc Chagall", "I and the Village"),
            ("Marc Chagall", "Big Wheel"),
            ("James Ensor", "Death and the Masks"),
            ("Edvard Munch", "Scream"),
        ],
        # LOW: Painting-specific muted/tonal anchors across old masters, Cubism,
        # Impressionism, and Expressionism. The pool is intentionally mixed so the
        # colour axis is less identical to Depth or Light.
        "low": [
            ("Titian", None),
            ("Rembrandt", None),
            ("Peter Paul Rubens", "Portrait of a man"),
            ("Paul Bril", "Landscape with Roman Ruins"),
            ("Agostino Carracci", "Landscape with Bathers"),
            ("Annibale Carracci", "River Landscape"),
            ("Cima da Conegliano", "The Healing of Anianus"),
            ("Georges Braque", "Piano and Mandolin"),
            ("Georges Braque", "Still Life with a Metronome"),
            ("Georges Braque", "Fruit Dish"),
            ("Georges Braque", "Violin"),
            ("Fernand Leger", "Portrait of Henry Viel"),
            ("Paul Cezanne", "Portrait of Peasant"),
            ("Kathe Kollwitz", "Death"),
            ("Kathe Kollwitz", "Revolt"),
            ("Kathe Kollwitz", "Pregnant woman"),
            ("Albin Egger-Lienz", "portrait painter"),
            ("Albin Egger-Lienz", "Johanneskirche"),
            ("Morris Graves", "Surf and Bird"),
            ("Morris Graves", "Chalice"),
            ("James McNeill Whistler", "Grey and Silver"),
            ("James McNeill Whistler", "Battersea"),
            ("Eugene Boudin", "Trouville, Black Rocks"),
            ("Johan Jongkind", "Notre-Dame"),
            ("Camille Pissarro", "Village Corner"),
            ("Giorgio de Chirico", "Portrait"),
            ("Edward Wadsworth", "Dunkerque"),
        ],
    },
}

DIM_NAMES = list(DIMENSIONS.keys())


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str, size: int = 224) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLIP encoding (same pattern as train.py)
# ---------------------------------------------------------------------------

def encode_clip(image_paths: list[str], batch_size: int) -> np.ndarray:
    log.info("Loading CLIP model from %s ...", CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP encode"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [load_image(p) for p in batch_paths]
        valid = [(img, j) for j, img in enumerate(imgs) if img is not None]
        if not valid:
            embeddings.append(np.zeros((len(batch_paths), 512)))
            continue
        batch_imgs = [v[0] for v in valid]
        idx        = [v[1] for v in valid]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = np.zeros((len(batch_paths), feats.shape[-1]))
        for k, orig_idx in enumerate(idx):
            arr[orig_idx] = feats[k].cpu().numpy()
        embeddings.append(arr)

    del model
    return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# Embedding cache helpers
# ---------------------------------------------------------------------------

def load_or_encode(df: pd.DataFrame, batch_size: int, skip_encode: bool) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, paths). Load from cache if available; encode otherwise."""
    all_paths = df["file_path"].tolist()
    abs_paths = [str(BASE_DIR / p) for p in all_paths]

    if CACHE_PATH.exists():
        log.info("Loading embeddings from cache: %s", CACHE_PATH)
        data = np.load(CACHE_PATH, allow_pickle=True)
        cached_paths = list(data["paths"])
        # Build indexes by absolute path and by relative data/... path. The latter
        # keeps the cache usable after renaming or moving the project folder.
        path_to_idx = {p: i for i, p in enumerate(cached_paths)}
        rel_to_idx: dict[str, int] = {}
        for i, cached_path in enumerate(cached_paths):
            parts = Path(str(cached_path)).parts
            if "data" in parts:
                data_pos = parts.index("data")
                rel_to_idx[str(Path(*parts[data_pos:]))] = i

        D = data["embeddings"].shape[1]
        X = np.zeros((len(abs_paths), D))
        missing = 0
        for i, rel_path in enumerate(all_paths):
            abs_path = str(BASE_DIR / rel_path)
            if abs_path in path_to_idx:
                X[i] = data["embeddings"][path_to_idx[abs_path]]
            elif rel_path in rel_to_idx:
                X[i] = data["embeddings"][rel_to_idx[rel_path]]
            else:
                missing += 1
        if missing:
            log.warning("%d paths not found in cache; they will be zero-filled.", missing)
        return X, all_paths

    if skip_encode:
        raise FileNotFoundError(
            f"Cache not found at {CACHE_PATH} and --skip-encode was set."
        )

    log.info("No cache found — encoding %d images with CLIP.", len(abs_paths))
    X = encode_clip(abs_paths, batch_size)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(CACHE_PATH), embeddings=X, paths=np.array(abs_paths))
    log.info("Saved embedding cache to %s", CACHE_PATH)
    return X, all_paths


# ---------------------------------------------------------------------------
# Anchor matching
# ---------------------------------------------------------------------------

def find_anchor_indices(df: pd.DataFrame,
                        anchors: list,
                        side_label: str) -> list[int]:
    """
    Return row indices matching the anchor list.

    Each entry in anchors is either:
      str                   — matches all paintings by that artist
      (artist_kw, title_kw) — matches paintings where artist contains artist_kw
                              AND title contains title_kw (both case-insensitive).
                              title_kw=None falls back to artist-only matching.
    """
    artist_lower = df["artist"].str.lower().fillna("")
    title_lower  = df["title"].str.lower().fillna("")

    found:     list[str] = []
    not_found: list[str] = []
    indices:   list[int] = []

    for entry in anchors:
        if isinstance(entry, str):
            artist_kw, title_kw = entry, None
        else:
            artist_kw, title_kw = entry[0], entry[1]

        a_mask = artist_lower.str.contains(artist_kw.lower(), regex=False)
        if title_kw:
            t_mask = title_lower.str.contains(title_kw.lower(), regex=False)
            mask   = a_mask & t_mask
            label  = f"{artist_kw} / {title_kw}"
        else:
            mask  = a_mask
            label = artist_kw

        hits = df.index[mask].tolist()
        if hits:
            found.append(f"{label} ({len(hits)})")
            indices.extend(hits)
        else:
            not_found.append(label)

    indices = sorted(set(indices))

    if not_found:
        log.warning("  [%s] NOT FOUND: %s", side_label, " | ".join(not_found))
    log.info("  [%s] %d paintings from: %s",
             side_label, len(indices), ", ".join(found))

    return indices


# ---------------------------------------------------------------------------
# Axis construction & projection
# ---------------------------------------------------------------------------

def build_axis(emb: np.ndarray, high_idx: list[int], low_idx: list[int],
               dim_name: str) -> dict:
    """
    Compute Arnheim axis from anchor embeddings.
    Returns dict with axis_unit, c_high, c_low, low_proj, high_proj.
    """
    if len(high_idx) < 3:
        log.warning("  [%s] Only %d HIGH anchors (< 3) — axis may be noisy.", dim_name, len(high_idx))
    if len(low_idx) < 3:
        log.warning("  [%s] Only %d LOW anchors (< 3)  — axis may be noisy.", dim_name, len(low_idx))

    c_high = emb[high_idx].mean(axis=0)
    c_low  = emb[low_idx].mean(axis=0)

    axis      = c_high - c_low  # raw, not unit
    norm      = np.linalg.norm(axis)
    axis_unit = axis / (norm + 1e-8)

    low_proj  = float(np.dot(c_low,  axis_unit))
    high_proj = float(np.dot(c_high, axis_unit))

    return dict(axis_unit=axis_unit, c_high=c_high, c_low=c_low,
                low_proj=low_proj, high_proj=high_proj)


def project_scores(emb: np.ndarray, axis_info: dict) -> np.ndarray:
    """Project embeddings onto an axis with robust percentile calibration.

    The anchor centroids define the direction. The score scale is calibrated from
    the empirical 5th/95th percentiles of the dataset, when available, so a few
    extreme paintings do not stretch the visual interpretation.
    """
    raw = emb @ axis_info["axis_unit"]
    lo, hi = axis_info["low_proj"], axis_info["high_proj"]
    span = hi - lo
    if abs(span) < 1e-8:
        return np.zeros(len(emb))

    anchor_score = 2.0 * (raw - lo) / span - 1.0
    score_low = axis_info.get("score_low")
    score_high = axis_info.get("score_high")
    if score_low is None or score_high is None or abs(score_high - score_low) < 1e-8:
        return anchor_score

    calibrated = 2.0 * (anchor_score - score_low) / (score_high - score_low) - 1.0
    return np.clip(calibrated, -1.5, 1.5)


# ---------------------------------------------------------------------------
# Axis independence validation
# ---------------------------------------------------------------------------

def validate_axis_independence(axes_info: dict, built_dims: list[str],
                                output_dir: Path) -> None:
    """
    Compute pairwise cosine similarity between all axis direction vectors.
    High |similarity| means two axes point in nearly the same (or opposite)
    direction in embedding space — they are measuring the same underlying
    dimension, just possibly inverted.

    Prints a formatted matrix and warnings, saves a heatmap PNG.
    """
    n = len(built_dims)
    vecs = np.stack([axes_info[d]["axis_unit"] for d in built_dims])  # (n, D)

    # Cosine similarity matrix: vecs are already unit vectors, so it's just vecs @ vecs.T
    sim = vecs @ vecs.T  # (n, n)
    np.fill_diagonal(sim, 1.0)  # fix any float rounding on diagonal

    # --- Print table ---
    col_w = 9
    header = " " * 12 + "".join(f"{d:>{col_w}}" for d in built_dims)
    sep    = "─" * len(header)
    log.info("")
    log.info("═" * len(header))
    log.info("Axis Independence Check  (cosine similarity between axis vectors)")
    log.info("═" * len(header))
    log.info("Interpretation: |r| < 0.3 = independent · 0.3–0.6 = moderate · > 0.6 = collinear")
    log.info(sep)
    log.info(header)
    log.info(sep)
    for i, d in enumerate(built_dims):
        row = f"{d:<12}" + "".join(f"{sim[i, j]:>{col_w}.3f}" for j in range(n))
        log.info(row)
    log.info(sep)

    # --- Warnings for collinear pairs ---
    warned = False
    for i in range(n):
        for j in range(i + 1, n):
            s = sim[i, j]
            abs_s = abs(s)
            if abs_s > 0.6:
                direction = "positively correlated" if s > 0 else "anti-correlated (inverted)"
                severity  = "HIGH ⚠" if abs_s > 0.75 else "MODERATE"
                log.warning(
                    "  [%s] %s ↔ %s  similarity=%.3f  → %s — "
                    "these axes may not be measuring independent dimensions.",
                    severity, built_dims[i], built_dims[j], s, direction,
                )
                warned = True
    if not warned:
        log.info("  ✓ All axis pairs have |similarity| ≤ 0.6 — dimensions appear reasonably independent.")
    log.info("")

    # --- Effective rank: how many truly independent axes does the set span? ---
    # Singular values of the axis matrix reveal this.
    _, sv, _ = np.linalg.svd(vecs, full_matrices=False)
    sv_norm  = sv / sv.sum()
    cum_var  = np.cumsum(sv_norm)
    eff_rank = int(np.searchsorted(cum_var, 0.95)) + 1  # dims needed for 95% of variance
    log.info("  Effective rank of axis set: %d / %d (axes needed to capture 95%% of inter-axis variance)",
             eff_rank, n)
    log.info("  Singular value spectrum: %s", "  ".join(f"{v:.3f}" for v in sv))
    log.info("")

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.eye(n, dtype=bool)  # mask diagonal for readability
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        sim, annot=True, fmt=".2f", cmap=cmap,
        vmin=-1, vmax=1, center=0,
        xticklabels=built_dims, yticklabels=built_dims,
        linewidths=0.5, ax=ax, mask=mask,
        annot_kws={"size": 11},
    )
    # Annotate diagonal manually (always 1.00)
    for i in range(n):
        ax.text(i + 0.5, i + 0.5, "1.00", ha="center", va="center",
                fontsize=11, color="black", fontweight="bold")

    ax.set_title("Arnheim Axis Independence\n(cosine similarity; |r| > 0.6 = collinear ⚠)", fontsize=11)
    plt.tight_layout()
    out = output_dir / "arnheim_axis_independence.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved axis independence heatmap → %s", out)


# ---------------------------------------------------------------------------
# Radar / spider chart helpers
# ---------------------------------------------------------------------------

def _radar_axes(n: int):
    """Return evenly spaced angles for a radar chart with n dimensions."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    return angles


def draw_radar(ax, values: list[float], labels: list[str],
               color: str, label: str, alpha: float = 0.15) -> None:
    angles = _radar_axes(len(labels))
    vals = list(values) + [values[0]]
    ax.plot(angles, vals, color=color, linewidth=1.5, label=label)
    ax.fill(angles, vals, color=color, alpha=alpha)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "", "0", "", "1"], fontsize=6)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_radar_overlay(scores_df: pd.DataFrame) -> None:
    """All 8 genres overlaid on one radar chart."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors = plt.cm.Set1(np.linspace(0, 1, len(CLASSES)))

    for cls, color in zip(CLASSES, colors):
        subset = scores_df[scores_df["label"] == cls]
        if subset.empty:
            continue
        means = subset[DIM_NAMES].mean().tolist()
        draw_radar(ax, means, DIM_NAMES, color=color, label=cls)

    ax.set_title("Arnheim Dimensions — All Genres", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "arnheim_radar_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_radar_per_genre(scores_df: pd.DataFrame) -> None:
    """2×4 grid, one radar per genre."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9),
                             subplot_kw={"polar": True})
    colors = plt.cm.Set1(np.linspace(0, 1, len(CLASSES)))

    for ax, cls, color in zip(axes.flat, CLASSES, colors):
        subset = scores_df[scores_df["label"] == cls]
        if subset.empty:
            ax.set_title(f"{cls}\n(no data)", fontsize=9)
            continue
        means = subset[DIM_NAMES].mean().tolist()
        draw_radar(ax, means, DIM_NAMES, color=color, label=cls)
        ax.set_title(cls, fontsize=10, pad=12)

    fig.suptitle("Arnheim Dimensions per Genre", fontsize=14, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "arnheim_radar_per_genre.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_violins(scores_df: pd.DataFrame) -> None:
    """2×3 grid of violin plots, one per Arnheim dimension."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, dim in zip(axes.flat, DIM_NAMES):
        # Build long-form subset for this dimension
        plot_data = scores_df[["label", dim]].copy()
        plot_data.columns = ["Genre", "Score"]
        sns.violinplot(data=plot_data, x="Genre", y="Score",
                       order=CLASSES, ax=ax, palette="Set1",
                       inner="quartile", cut=0)
        ax.set_title(dim, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(-2.5, 2.5)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Arnheim Score Distributions by Genre", fontsize=14)
    plt.tight_layout()
    out = OUTPUT_DIR / "arnheim_violins.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_wolfflin_heatmap(scores_df: pd.DataFrame) -> None:
    """
    Correlate per-genre mean Arnheim scores (8×6) with Wölfflin theoretical
    scores (8×5). Show Spearman ρ as a heatmap: rows = Wölfflin axes,
    columns = Arnheim dimensions.
    """
    # Per-genre means for Arnheim dimensions (8 rows × 6 cols)
    arnheim_means = (
        scores_df.groupby("label")[DIM_NAMES]
        .mean()
        .reindex(CLASSES)  # consistent ordering
        .fillna(0)
    )

    # Wölfflin theoretical matrix (8 rows × 5 cols)
    wolfflin_matrix = np.array([WOLFFLIN_THEORY[c] for c in CLASSES])  # (8,5)
    arnheim_matrix  = arnheim_means.values                               # (8,6)

    # Spearman ρ for each (Wölfflin axis, Arnheim dim) pair
    n_w = len(WOLFFLIN_AXES)
    n_a = len(DIM_NAMES)
    corr_matrix = np.zeros((n_w, n_a))
    for i in range(n_w):
        for j in range(n_a):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(wolfflin_matrix[:, i], arnheim_matrix[:, j])
            corr_matrix[i, j] = rho if not np.isnan(rho) else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=DIM_NAMES, yticklabels=WOLFFLIN_AXES,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Spearman ρ: Wölfflin Theoretical vs Arnheim Empirical Scores\n"
                 "(per-genre means, 8 observations)", fontsize=11)
    ax.set_xlabel("Arnheim Dimension")
    ax.set_ylabel("Wölfflin Axis")
    plt.tight_layout()
    out = OUTPUT_DIR / "arnheim_wolfflin_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_tsne(embeddings: np.ndarray, scores_df: pd.DataFrame) -> None:
    """
    2D dimensionality reduction of all embeddings coloured by genre.
    Overlays Arnheim dimension arrows as a pseudo-biplot.
    Tries UMAP first; falls back to t-SNE.
    """
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1, metric="cosine")
        method_label = "UMAP"
        log.info("Reducing with UMAP...")
    except ImportError:
        from sklearn.manifold import TSNE
        import sklearn
        tsne_kwargs = dict(n_components=2, random_state=42, perplexity=30, metric="cosine")
        # n_iter renamed to max_iter in sklearn 1.5
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
            tsne_kwargs["max_iter"] = 1000
        else:
            tsne_kwargs["n_iter"] = 1000
        reducer = TSNE(**tsne_kwargs)
        method_label = "t-SNE"
        log.info("UMAP not available — using t-SNE (this takes 2–5 min, no progress bar)...")

    log.info("Running %s on %d embeddings — please wait...", method_label, len(embeddings))
    X_2d = reducer.fit_transform(embeddings)  # (N, 2)
    log.info("%s done.", method_label)

    arnheim_matrix = scores_df[DIM_NAMES].values  # (N, 6)

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.Set1(np.linspace(0, 1, len(CLASSES)))
    labels = scores_df["label"].values

    for cls, color in zip(CLASSES, colors):
        mask = labels == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[color], s=8, alpha=0.5, label=cls, rasterized=True)

    # Pseudo-biplot arrows: regress each Arnheim score onto 2D coordinates
    centroid = X_2d.mean(axis=0)
    # Scale arrows so they are visible relative to the data spread
    scale = np.std(X_2d) * 1.5

    arrow_colors = plt.cm.tab10(np.linspace(0, 1, len(DIM_NAMES)))
    for j, (dim, ac) in enumerate(zip(DIM_NAMES, arrow_colors)):
        y = arnheim_matrix[:, j]
        reg = LinearRegression().fit(X_2d, y)
        direction = reg.coef_  # shape (2,)
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-8:
            continue
        direction = direction / d_norm * scale
        ax.annotate(
            "", xy=centroid + direction, xytext=centroid,
            arrowprops=dict(arrowstyle="->", color=ac, lw=2),
        )
        offset = direction * 1.12
        ax.text(centroid[0] + offset[0], centroid[1] + offset[1],
                dim, fontsize=9, color=ac, fontweight="bold",
                ha="center", va="center")

    ax.set_title(f"{method_label} of CLIP Embeddings with Arnheim Dimension Arrows", fontsize=12)
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.tight_layout()
    out = OUTPUT_DIR / "arnheim_tsne.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def write_summary(scores_df: pd.DataFrame) -> None:
    """Write arnheim_summary.txt with per-class mean scores and observations."""
    per_class = (
        scores_df.groupby("label")[DIM_NAMES]
        .mean()
        .reindex(CLASSES)
        .round(3)
    )

    lines = []
    lines.append("=" * 72)
    lines.append("Arnheim Perceptual Dimension Analysis — Summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Per-class mean scores (range approx. -1 to +1)")
    lines.append("")

    # Header
    col_w = 14
    header = f"{'Class':<22}" + "".join(f"{d:>{col_w}}" for d in DIM_NAMES)
    lines.append(header)
    lines.append("-" * len(header))
    for cls in CLASSES:
        if cls not in per_class.index:
            continue
        row = f"{cls:<22}" + "".join(f"{per_class.loc[cls, d]:>{col_w}.3f}" for d in DIM_NAMES)
        lines.append(row)

    lines.append("")
    lines.append("=" * 72)
    lines.append("Observations")
    lines.append("=" * 72)

    for dim in DIM_NAMES:
        col = per_class[dim].dropna()
        if col.empty:
            continue
        highest = col.idxmax()
        lowest  = col.idxmin()
        lines.append(f"\n{dim} ({DIMENSIONS[dim]['description']})")
        lines.append(f"  Highest: {highest:<22} ({col[highest]:+.3f})")
        lines.append(f"  Lowest:  {lowest:<22} ({col[lowest]:+.3f})")

    lines.append("")
    out = OUTPUT_DIR / "arnheim_summary.txt"
    out.write_text("\n".join(lines))
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_top_bottom(scores_df: pd.DataFrame, n: int = 10) -> None:
    """Print top-n and bottom-n paintings per Arnheim dimension as a sanity check."""
    def _short_title(value) -> str:
        if pd.isna(value):
            return ""
        return str(value)[:45]

    log.info("")
    log.info("═" * 72)
    log.info("Top / Bottom %d paintings per dimension", n)
    log.info("═" * 72)
    for dim in DIM_NAMES:
        col = scores_df[dim]
        log.info("")
        log.info("── %s (%s) ──", dim, DIMENSIONS[dim]["description"])
        top = scores_df.nlargest(n, dim)[["artist", "title", "label", dim]]
        log.info("  TOP %d (highest score):", n)
        for _, row in top.iterrows():
            log.info("    [%s] %s — %s  (%.3f)", row["label"], row["artist"], _short_title(row["title"]), row[dim])
        bot = scores_df.nsmallest(n, dim)[["artist", "title", "label", dim]]
        log.info("  BOTTOM %d (lowest = most %s):", n, "LOW")
        for _, row in bot.iterrows():
            log.info("    [%s] %s — %s  (%.3f)", row["label"], row["artist"], _short_title(row["title"]), row[dim])


def main():
    parser = argparse.ArgumentParser(
        description="Score paintings on Arnheim perceptual dimensions via CLIP anchors."
    )
    parser.add_argument("--skip-encode", action="store_true",
                        help="Load embeddings from cache only; fail if not found.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for CLIP encoding (default: 32).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: outputs/arnheim).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load metadata --
    log.info("Loading metadata from %s", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    df = df.reset_index(drop=True)
    log.info("  %d paintings across %d classes", len(df), df["label"].nunique())

    # -- Embeddings --
    embeddings, rel_paths = load_or_encode(df, args.batch_size, args.skip_encode)
    log.info("Embeddings shape: %s", embeddings.shape)

    # -- Build Arnheim axes --
    log.info("")
    log.info("Building Arnheim axes from artist anchors...")
    axes_info: dict[str, dict] = {}

    for dim_name, dim_cfg in DIMENSIONS.items():
        log.info("  Dimension: %s", dim_name)
        high_idx = find_anchor_indices(df, dim_cfg["high"], f"{dim_name}/HIGH")
        low_idx  = find_anchor_indices(df, dim_cfg["low"],  f"{dim_name}/LOW")

        if not high_idx or not low_idx:
            log.error("  Cannot build axis for %s — no anchors on one side. Skipping.", dim_name)
            continue

        axes_info[dim_name] = build_axis(embeddings, high_idx, low_idx, dim_name)
        log.info("  → axis norm = %.4f  low_proj = %.4f  high_proj = %.4f",
                 np.linalg.norm(axes_info[dim_name]["axis_unit"]),
                 axes_info[dim_name]["low_proj"],
                 axes_info[dim_name]["high_proj"])

    # Use only successfully built dimensions
    built_dims = [d for d in DIM_NAMES if d in axes_info]
    if not built_dims:
        raise RuntimeError("No Arnheim axes could be built. Check that metadata.csv has artist info.")

    # -- Robust score calibration --
    log.info("")
    log.info("Calibrating Arnheim score scales from dataset percentiles...")
    for dim_name in built_dims:
        raw_scores = project_scores(embeddings, axes_info[dim_name])
        score_low, score_high = np.percentile(raw_scores, [5, 95])
        axes_info[dim_name]["score_low"] = float(score_low)
        axes_info[dim_name]["score_high"] = float(score_high)
        log.info("  %s: 5th=%.3f 95th=%.3f", dim_name, score_low, score_high)

    # -- Axis independence validation --
    validate_axis_independence(axes_info, built_dims, out_dir)

    # -- Score all paintings --
    log.info("")
    log.info("Projecting %d paintings onto %d calibrated axes...", len(embeddings), len(built_dims))
    score_cols: dict[str, np.ndarray] = {}
    for dim_name in built_dims:
        score_cols[dim_name] = project_scores(embeddings, axes_info[dim_name])

    # -- Assemble scores dataframe --
    scores_df = df[["file_path", "label", "split", "artist", "title"]].copy()
    for dim_name in built_dims:
        scores_df[dim_name] = score_cols[dim_name]
    for dim_name in DIM_NAMES:
        if dim_name not in scores_df.columns:
            scores_df[dim_name] = np.nan

    # -- Top/bottom sanity check --
    print_top_bottom(scores_df)

    # -- Per-class means --
    log.info("")
    log.info("Per-class mean Arnheim scores:")
    means = scores_df.groupby("label")[built_dims].mean().reindex(CLASSES).round(3)
    log.info("\n%s", means.to_string())

    # -- Save scores CSV --
    out_csv = out_dir / "arnheim_scores.csv"
    scores_df.to_csv(out_csv, index=False)
    log.info("Saved scores to %s", out_csv)

    # -- Save axes to models/ for app.py --
    axes_array    = np.stack([axes_info[d]["axis_unit"] for d in built_dims])
    c_low_array   = np.stack([axes_info[d]["c_low"]     for d in built_dims])
    c_high_array  = np.stack([axes_info[d]["c_high"]    for d in built_dims])
    low_proj_arr  = np.array([axes_info[d]["low_proj"]  for d in built_dims])
    high_proj_arr = np.array([axes_info[d]["high_proj"] for d in built_dims])
    score_low_arr = np.array([axes_info[d].get("score_low", -1.0) for d in built_dims])
    score_high_arr = np.array([axes_info[d].get("score_high", 1.0) for d in built_dims])

    axes_out = MODELS_DIR / "arnheim_axes.npz"
    np.savez_compressed(
        str(axes_out),
        dim_names=np.array(built_dims),
        axes=axes_array,
        c_low=c_low_array,
        c_high=c_high_array,
        low_proj=low_proj_arr,
        high_proj=high_proj_arr,
        score_low=score_low_arr,
        score_high=score_high_arr,
    )
    log.info("Saved axes to %s", axes_out)

    profiles = scores_df.groupby("label")[built_dims].mean().reindex(CLASSES).round(6)
    profiles_out = MODELS_DIR / "arnheim_profiles.json"
    profiles_payload = {
        "dimensions": built_dims,
        "normalization": "anchor_direction_with_dataset_5th_95th_percentile_calibration",
        "profiles": {
            label: {dim: float(profiles.loc[label, dim]) for dim in built_dims}
            for label in profiles.index
        },
    }
    profiles_out.write_text(json.dumps(profiles_payload, indent=2))
    log.info("Saved empirical profiles to %s", profiles_out)

    # -- Visualisations (patched to use out_dir) --
    _orig_output_dir = OUTPUT_DIR

    def _save(fig_fn, *a, **kw):
        pass  # plots patch their own out path via out_dir below

    log.info("")
    log.info("Generating visualisations...")

    # Monkey-patch OUTPUT_DIR in the plot functions via a helper
    def _plot(fn, *a, **kw):
        import types
        g = fn.__globals__
        old = g.get("OUTPUT_DIR")
        g["OUTPUT_DIR"] = out_dir
        try:
            fn(*a, **kw)
        finally:
            if old is not None:
                g["OUTPUT_DIR"] = old

    _plot(plot_radar_overlay, scores_df)
    _plot(plot_radar_per_genre, scores_df)
    _plot(plot_violins, scores_df)
    _plot(plot_wolfflin_heatmap, scores_df)
    _plot(plot_tsne, embeddings, scores_df)
    _plot(write_summary, scores_df)

    log.info("")
    log.info("Done. All outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
