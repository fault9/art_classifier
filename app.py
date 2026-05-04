"""
Gradio demo — Art Movement Classifier with interpretability views.

Tab 1: Painting Classifier
  Upload a painting → predicted movement + confidence bar chart + genre description

Tab 2: Collection Analyzer
  Upload multiple paintings → radar chart of movement distribution across collection

Tab 3: Wölfflin Analysis
  Upload a painting → position on each of Wölfflin's 5 theoretical axes,
  compared to the theoretical position of the predicted movement

Tab 4: Arnheim Analysis
  Upload a painting → perceptual scores from CLIP anchor projections
"""

import io
import os
import json
import requests
import numpy as np
import joblib
import gradio as gr
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    ViTModel, AutoFeatureExtractor,
)

MODELS_DIR = Path("models")

CLASSES = [
    "Renaissance", "Baroque", "Impressionism", "Expressionism",
    "Cubism", "Abstract", "Surrealism", "Pop Art",
]

GENRE_DESCRIPTIONS = {
    "Renaissance": (
        "Renaissance (14th–17th c.) — mathematical perspective, balanced proportion, "
        "classical mythology and religious subjects. Linear clarity, closed compositions, "
        "every element precisely articulated. Wölfflin's 'classical' pole."
    ),
    "Baroque": (
        "Baroque (17th–18th c.) — dramatic chiaroscuro lighting, deep spatial recession, "
        "emotional intensity, sweeping diagonal movement. Rembrandt's shadows, "
        "Caravaggio's spotlight. Wölfflin's 'non-classical' pole."
    ),
    "Impressionism": (
        "Impressionism (1860s–1880s) — visible, energetic brushstrokes capturing "
        "light and atmosphere. Everyday scenes, plein-air landscapes, broken color. "
        "The painted surface matters as much as the subject."
    ),
    "Expressionism": (
        "Expressionism (1900s–1930s) — deliberately distorted forms, non-naturalistic "
        "color used to convey psychological states. Emotional truth over visual accuracy. "
        "Munch's anxiety, Schiele's rawness."
    ),
    "Cubism": (
        "Cubism (1908–1920s) — geometric fragmentation of objects into planes, "
        "simultaneous multiple viewpoints flattened onto a single surface. "
        "Analytical and Synthetic phases. Picasso and Braque."
    ),
    "Abstract": (
        "Abstract/Abstract Expressionism — non-representational art using pure "
        "color, form, and gesture. Kandinsky's spiritual geometry, Pollock's drip "
        "paintings, Rothko's color fields. The subject is the paint itself."
    ),
    "Surrealism": (
        "Surrealism (1920s–1940s) — dreamlike impossible imagery drawn from the "
        "subconscious. Dalí's hyper-precise rendering of impossible scenes, "
        "Magritte's philosophical uncanny, Ernst's frottage."
    ),
    "Pop Art": (
        "Pop Art (1950s–1970s) — bold flat colors, commercial imagery, mass culture "
        "references. Warhol's silkscreens, Lichtenstein's Ben-Day dots. "
        "Ironic appropriation of consumer society."
    ),
}

GENRE_COLORS = {
    "Renaissance":  "#8B4513",
    "Baroque":      "#2F4F4F",
    "Impressionism":"#87CEEB",
    "Expressionism":"#FF4500",
    "Cubism":       "#708090",
    "Abstract":     "#9400D3",
    "Surrealism":   "#228B22",
    "Pop Art":      "#FF1493",
}

EXAMPLE_IMAGE_PATHS = [
    ["Raphael - Self Portrait (expected: Renaissance, held-out test)", "examples/raphael_self_portrait.jpg"],
    ["Caravaggio - Supper at Emmaus (expected: Baroque, not in training split)", "examples/caravaggio_supper_at_emmaus.jpg"],
    ["Monet - The Coast at Sainte-Adresse (expected: Impressionism, held-out test)", "examples/monet_coast_at_sainte_adresse.jpg"],
    ["Georges Braque - Fruit Dish (expected: Cubism, held-out test)", "examples/georges_braque_fruit_dish.jpg"],
]

EXAMPLE_IMAGE_BY_LABEL = {label: path for label, path in EXAMPLE_IMAGE_PATHS}


# Wölfflin theoretical positions (−1 = classical, +1 = non-classical)
WOLFFLIN_THEORY = {
    "Renaissance":  [-1.0, -0.8, -0.9, -0.7, -1.0],
    "Baroque":      [ 0.9,  0.8,  0.7,  0.8,  0.6],
    "Impressionism":[ 0.9,  0.3,  0.6,  0.5,  0.5],
    "Expressionism":[ 0.6,  0.2,  0.7,  0.7,  0.4],
    "Cubism":       [-0.5, -0.7, -0.3, -0.4, -0.3],
    "Abstract":     [ 0.7, -0.1,  0.8,  0.8,  0.7],
    "Surrealism":   [-0.3,  0.2,  0.1,  0.2,  0.3],
    "Pop Art":      [-0.8, -0.5, -0.6, -0.5, -0.7],
}
WOLFFLIN_AXES = [
    "Linear ↔ Painterly",
    "Plane ↔ Recession",
    "Closed ↔ Open",
    "Multiplicity ↔ Unity",
    "Clearness ↔ Unclearness",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_URL_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ArtClassifier/1.0)"}


def load_from_url(url: str) -> Image.Image | None:
    url = (url or "").strip()
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=15, headers=_URL_HEADERS)
        resp.raise_for_status()
        if "image" not in resp.headers.get("Content-Type", ""):
            return None
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def resolve_image(img: Image.Image | None, url: str) -> Image.Image | None:
    """Return uploaded image if present, otherwise fetch from URL."""
    if img is not None:
        return img
    return load_from_url(url)




def load_example_image(label: str):
    """Load a bundled example into the shared image input."""
    return EXAMPLE_IMAGE_BY_LABEL.get(label), ""


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_embed_model  = None
_processor    = None
_classifier   = None
_label_encoder = None
_cfg: dict    = {}


def load_models():
    global _embed_model, _processor, _classifier, _label_encoder, _cfg

    cfg_path = MODELS_DIR / "config.json"
    if cfg_path.exists() and not _cfg:
        with open(cfg_path) as f:
            _cfg = json.load(f)

    emb_name  = _cfg.get("embedding_name", "CLIP")
    emb_model = _cfg.get("embedding_model", "openai/clip-vit-base-patch32")

    if _embed_model is None:
        if emb_name == "CLIP":
            _processor   = CLIPProcessor.from_pretrained(emb_model)
            _embed_model = CLIPModel.from_pretrained(emb_model).to(DEVICE)
        else:
            _processor   = AutoFeatureExtractor.from_pretrained(emb_model)
            _embed_model = ViTModel.from_pretrained(emb_model).to(DEVICE)
        _embed_model.eval()

    clf_path = MODELS_DIR / "classifier.joblib"
    le_path  = MODELS_DIR / "label_encoder.joblib"
    if _classifier is None and clf_path.exists():
        _classifier    = joblib.load(clf_path)
        _label_encoder = joblib.load(le_path)


def embed_image(img: Image.Image) -> np.ndarray:
    """Embed a single PIL image, return 1D numpy vector."""
    emb_name = _cfg.get("embedding_name", "CLIP")
    img = img.convert("RGB")
    if emb_name == "CLIP":
        inputs = _processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feat = _embed_model.get_image_features(pixel_values=inputs["pixel_values"])
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()
    else:
        inputs = _processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out  = _embed_model(**inputs)
            feat = out.last_hidden_state[:, 0, :]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Arnheim axis loading (lazy, cached)
# ---------------------------------------------------------------------------

_arnheim_axes: dict | None = None


def load_arnheim_axes() -> dict | None:
    global _arnheim_axes
    if _arnheim_axes is not None:
        return _arnheim_axes
    axes_path = MODELS_DIR / "arnheim_axes.npz"
    if not axes_path.exists():
        return None
    data = np.load(str(axes_path), allow_pickle=True)
    _arnheim_axes = {
        "dim_names":  list(data["dim_names"]),
        "axes":       data["axes"],
        "c_low":      data["c_low"],
        "c_high":     data["c_high"],
        "low_proj":   data["low_proj"],
        "high_proj":  data["high_proj"],
    }
    return _arnheim_axes


def score_arnheim(embedding: np.ndarray, axes: dict) -> dict[str, float]:
    scores = {}
    for i, dim in enumerate(axes["dim_names"]):
        axis_unit = axes["axes"][i]
        raw       = float(np.dot(embedding, axis_unit))
        lp        = float(axes["low_proj"][i])
        hp        = float(axes["high_proj"][i])
        if abs(hp - lp) < 1e-8:
            scores[dim] = 0.0
        else:
            s = 2.0 * (raw - lp) / (hp - lp) - 1.0
            scores[dim] = float(np.clip(s, -1.5, 1.5))
    return scores


ARNHEIM_DESCRIPTIONS = {
    "Balance": "High = symmetrical/centered composition  ·  Low = asymmetrical/dynamic",
    "Shape":   "High = geometric/regular forms  ·  Low = organic/fluid forms",
    "Depth":   "High = deep spatial recession  ·  Low = flat/planar",
    "Tension": "High = energetic/dynamic movement  ·  Low = static/calm",
    "Light":   "High = dramatic contrast/chiaroscuro  ·  Low = even/diffused light",
    "Color":   "High = vivid/saturated palette  ·  Low = muted/tonal palette",
}

# Empirical Arnheim profiles per genre.
# These are generated from outputs/arnheim/arnheim_scores.csv and should match
# the saved anchor axes in models/arnheim_axes.npz. They are used as a fallback
# if models/arnheim_profiles.json is not available on the Space.
_DIM_ORDER = ["Balance", "Shape", "Depth", "Tension", "Light", "Color"]
ARNHEIM_PROFILES = {
    "Abstract":      [-1.071999,  0.845304, -0.947955,  1.238481, -0.847324,  0.973768],
    "Baroque":       [ 0.507236, -0.619518,  0.849286,  0.073132,  1.169032, -1.625671],
    "Cubism":        [-0.985060,  0.567781, -0.616797,  0.630737, -0.797741,  0.424626],
    "Expressionism": [-0.680028, -0.172826, -0.238499,  0.353154, -0.442601, -0.317481],
    "Impressionism": [-0.652877, -1.327408,  0.086022, -1.282897, -0.960565, -1.055453],
    "Pop Art":       [-0.832993,  0.751233, -0.850416,  0.942616, -0.885262,  0.991585],
    "Renaissance":   [ 0.882364, -0.429339,  1.004560, -0.123539,  1.169254, -1.484537],
    "Surrealism":    [-0.947914,  0.782817, -0.663534,  1.144011, -0.486911,  0.600223],
}

_arnheim_profiles_cache: dict[str, list[float]] | None = None

def load_arnheim_profiles() -> dict[str, list[float]]:
    global _arnheim_profiles_cache
    if _arnheim_profiles_cache is not None:
        return _arnheim_profiles_cache
    profiles_path = MODELS_DIR / "arnheim_profiles.json"
    if profiles_path.exists():
        with open(profiles_path) as f:
            payload = json.load(f)
        profiles = {
            genre: [float(values[d]) for d in _DIM_ORDER]
            for genre, values in payload.get("profiles", {}).items()
        }
        if profiles:
            _arnheim_profiles_cache = profiles
            return profiles
    _arnheim_profiles_cache = ARNHEIM_PROFILES
    return _arnheim_profiles_cache


def arnheim_profile_matches(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Return empirical perceptual-profile similarities, not class predictions."""
    vec = np.array([scores.get(d, 0.0) for d in _DIM_ORDER], dtype=float)
    matches: list[tuple[str, float]] = []

    for genre, profile in load_arnheim_profiles().items():
        p = np.array(profile, dtype=float)
        distance = float(np.linalg.norm(vec - p))
        # Convert distance to a readable bounded similarity. 1.0 is identical.
        similarity = 1.0 / (1.0 + distance)
        matches.append((genre, similarity))

    return sorted(matches, key=lambda item: item[1], reverse=True)


def arnheim_match(scores: dict[str, float]) -> tuple[str, float]:
    """Backward-compatible helper for the nearest perceptual profile."""
    return arnheim_profile_matches(scores)[0]


# ---------------------------------------------------------------------------
# Tab 1: Painting Classifier
# ---------------------------------------------------------------------------

def classify_painting(img: Image.Image, url: str = ""):
    img = resolve_image(img, url)
    if img is None:
        return "Upload a painting or paste an image URL.", None, ""

    load_models()
    if _classifier is None:
        return "Model not found — run train.py first.", None, ""

    vec   = embed_image(img).reshape(1, -1)
    probs = _classifier.predict_proba(vec)[0]
    classes = _label_encoder.classes_

    order           = np.argsort(probs)[::-1]
    sorted_classes  = [classes[i] for i in order]
    sorted_probs    = [probs[i]   for i in order]

    predicted   = sorted_classes[0]
    confidence  = sorted_probs[0]
    label_md    = f"## {predicted}\n**{confidence*100:.1f}% confidence**"

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in sorted_probs],
        y=sorted_classes,
        orientation="h",
        marker_color=[GENRE_COLORS.get(c, "#888") for c in sorted_classes],
        text=[f"{p*100:.1f}%" for p in sorted_probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Confidence per Art Movement",
        xaxis_title="Probability (%)",
        xaxis_range=[0, 110],
        yaxis={"autorange": "reversed"},
        height=400,
        margin=dict(l=160, r=60, t=60, b=40),
    )

    description = GENRE_DESCRIPTIONS.get(predicted, "")
    return label_md, fig, description


# ---------------------------------------------------------------------------
# Tab 2: Collection Analyzer
# ---------------------------------------------------------------------------

def analyze_collection(files, urls_text: str = ""):
    # Gradio 6.x gr.File returns file objects; older versions may pass PIL images
    loaded: list[Image.Image] = []
    for f in (files or []):
        if f is None:
            continue
        try:
            if isinstance(f, Image.Image):
                loaded.append(f.convert("RGB"))
            else:
                path = f.name if hasattr(f, "name") else str(f)
                loaded.append(Image.open(path).convert("RGB"))
        except Exception:
            pass
    for line in (urls_text or "").splitlines():
        fetched = load_from_url(line)
        if fetched is not None:
            loaded.append(fetched)

    if not loaded:
        return "Upload paintings or paste image URLs (one per line).", None

    load_models()
    if _classifier is None:
        return "Model not found — run train.py first.", None

    movement_counts = {c: 0 for c in CLASSES}
    details = []

    for img in loaded:
        if img is None:
            continue
        vec   = embed_image(img).reshape(1, -1)
        probs = _classifier.predict_proba(vec)[0]
        pred  = _label_encoder.classes_[np.argmax(probs)]
        conf  = probs.max()
        movement_counts[pred] += 1
        details.append(f"- **{pred}** ({conf*100:.0f}%)")

    total = sum(movement_counts.values()) or 1
    values = [movement_counts[c] / total * 100 for c in CLASSES]
    values_closed = values + [values[0]]
    cats_closed   = CLASSES + [CLASSES[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill="toself", opacity=0.7,
        line=dict(color="#5B9BD5"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) + 5])),
        title="Collection — Art Movement Distribution",
        height=500,
    )

    summary = f"**{len(details)} paintings analysed:**\n\n" + "\n".join(details)
    return summary, fig


# ---------------------------------------------------------------------------
# Tab 3: Wölfflin Analysis
# ---------------------------------------------------------------------------

WOLFFLIN_DISCLAIMER = (
    "> **Note:** Wölfflin's framework was developed mainly to compare Renaissance "
    "and Baroque art. Here it is used as a cautious theoretical reference for the "
    "classifier's predicted movement, not as a direct measurement of the uploaded image."
)


def wolfflin_analysis(img: Image.Image, url: str = ""):
    img = resolve_image(img, url)
    if img is None:
        return WOLFFLIN_DISCLAIMER + "\n\nUpload a painting or paste an image URL.", None

    load_models()
    if _classifier is None:
        return "Model not found — run train.py first.", None

    vec   = embed_image(img).reshape(1, -1)
    probs = _classifier.predict_proba(vec)[0]
    pred  = _label_encoder.classes_[np.argmax(probs)]

    theory = WOLFFLIN_THEORY[pred]  # 5 values, −1 to +1

    # Normalise to 0–100 for radar display
    norm = [(v + 1) / 2 * 100 for v in theory]
    norm_closed = norm + [norm[0]]
    axes_closed = WOLFFLIN_AXES + [WOLFFLIN_AXES[0]]

    # Also show Renaissance (classical pole) and Baroque (non-classical) as reference
    ren_norm  = [(v + 1) / 2 * 100 for v in WOLFFLIN_THEORY["Renaissance"]]
    bar_norm  = [(v + 1) / 2 * 100 for v in WOLFFLIN_THEORY["Baroque"]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ren_norm + [ren_norm[0]], theta=axes_closed,
        fill="toself", opacity=0.25, name="Renaissance (classical ref.)",
        line=dict(color="#8B4513", dash="dot"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=bar_norm + [bar_norm[0]], theta=axes_closed,
        fill="toself", opacity=0.25, name="Baroque (non-classical ref.)",
        line=dict(color="#2F4F4F", dash="dot"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_closed, theta=axes_closed,
        fill="toself", opacity=0.6,
        name=f"Predicted: {pred}",
        line=dict(color=GENRE_COLORS.get(pred, "#555"), width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                   tickvals=[0, 25, 50, 75, 100],
                                   ticktext=["Classical", "", "Neutral", "", "Non-classical"])),
        title=f"Wölfflin Profile — Predicted: {pred}",
        showlegend=True, height=550,
    )

    # Build axis summary
    lines = [
        WOLFFLIN_DISCLAIMER,
        f"\n\n**Predicted movement: {pred}**\n",
        "| Axis | Theoretical Position |",
        "|------|---------------------|",
    ]
    axis_labels = [
        "Linear ↔ Painterly",
        "Plane ↔ Recession",
        "Closed ↔ Open",
        "Multiplicity ↔ Unity",
        "Clearness ↔ Unclearness",
    ]
    for ax, val in zip(axis_labels, theory):
        if val < -0.4:
            pos = f"Classical ({ax.split('↔')[0].strip()})"
        elif val > 0.4:
            pos = f"Non-classical ({ax.split('↔')[1].strip()})"
        else:
            pos = "Ambiguous / Mixed"
        lines.append(f"| {ax} | {pos} |")

    return "\n".join(lines), fig


# ---------------------------------------------------------------------------
# Tab 4: Perceptual Analysis (Arnheim)
# ---------------------------------------------------------------------------

ARNHEIM_NOTE = (
    "> **Note:** Scores are derived from the painting's position in CLIP embedding space "
    "relative to perceptual *anchor* paintings, inspired by Rudolf Arnheim's "
    "*Art and Visual Perception* (1954). These scores describe visual tendencies, "
    "not definitive art-historical labels."
)


def arnheim_perceptual(img: Image.Image, url: str = ""):
    img = resolve_image(img, url)
    if img is None:
        return ARNHEIM_NOTE + "\n\nUpload a painting or paste an image URL.", None

    axes = load_arnheim_axes()
    if axes is None:
        return (
            ARNHEIM_NOTE + "\n\n**Axes not found** — run `python arnheim_analysis.py` first.",
            None,
        )

    load_models()
    if _classifier is None:
        return "Model not found — run train.py first.", None

    embedding = embed_image(img)
    scores    = score_arnheim(embedding, axes)

    # Classifier prediction (embedding-based)
    probs        = _classifier.predict_proba(embedding.reshape(1, -1))[0]
    clf_pred     = _label_encoder.classes_[np.argmax(probs)]
    clf_conf     = probs.max()

    # Empirical Arnheim profile similarities. These are interpretive perceptual
    # matches, not movement predictions; the classifier remains the label source.
    profile_matches = arnheim_profile_matches(scores)
    profiles = load_arnheim_profiles()
    reference_genre = clf_pred if clf_pred in profiles else profile_matches[0][0]

    dim_names = list(_DIM_ORDER)  # canonical order for display

    # --- Radar chart ---
    # Painting scores (0-100 scale)
    painting_vals = [(scores.get(d, 0.0) + 1) / 2 * 100 for d in dim_names]
    # Empirical profile for the classifier's movement, used as interpretation reference.
    match_profile = [(v + 1) / 2 * 100 for v in profiles[reference_genre]]

    closed_dims = dim_names + [dim_names[0]]

    fig = go.Figure()
    # Empirical class profile of the classifier's movement (reference)
    fig.add_trace(go.Scatterpolar(
        r=match_profile + [match_profile[0]],
        theta=closed_dims,
        fill="toself", opacity=0.20,
        line=dict(color=GENRE_COLORS.get(reference_genre, "#888"), dash="dot", width=1.5),
        name=f"Empirical class profile: {reference_genre}",
    ))
    # Painting's actual scores
    fig.add_trace(go.Scatterpolar(
        r=painting_vals + [painting_vals[0]],
        theta=closed_dims,
        fill="toself", opacity=0.60,
        line=dict(color=GENRE_COLORS.get(clf_pred, "#555"), width=2.5),
        name="This painting",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True, range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["Low", "", "Neutral", "", "High"],
        )),
        title="Arnheim Perceptual Profile",
        showlegend=True,
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )

    # --- Markdown summary ---
    nearest = ", ".join(f"{genre} ({sim:.2f})" for genre, sim in profile_matches[:3])

    lines = [
        ARNHEIM_NOTE,
        "",
        f"| | Result |",
        f"|---|---|",
        f"| **Classifier movement** | **{clf_pred}** - {clf_conf*100:.1f}% confidence |",
        f"| **Radar reference** | Empirical **{reference_genre}** class profile |",
        f"| **Nearest perceptual profiles** | {nearest} |",
        "",
        "The nearest perceptual profile is not a second movement prediction. It only shows which class averages this image resembles on the six Arnheim-inspired visual axes.",
        "",
        "---",
        "",
        "**Perceptual scores** (−1 = Low pole · 0 = Neutral · +1 = High pole)",
        "",
        "| Dimension | Score | Interpretation |",
        "|-----------|------:|----------------|",
    ]
    for d in dim_names:
        s = scores.get(d, 0.0)
        bar = "█" * int(abs(s) * 5)
        if s > 0.3:
            interp = ARNHEIM_DESCRIPTIONS[d].split("·")[0].replace("High = ", "").strip()
        elif s < -0.3:
            interp = ARNHEIM_DESCRIPTIONS[d].split("·")[1].replace("Low = ", "").strip()
        else:
            interp = "Neutral"
        lines.append(f"| **{d}** | {s:+.2f} {'▲' if s > 0 else '▼'}{bar} | {interp} |")

    lines += [
        "",
        "---",
        "",
        "*Dimension key:*",
    ]
    for d in dim_names:
        lines.append(f"- **{d}** — {ARNHEIM_DESCRIPTIONS[d]}")

    return "\n".join(lines), fig




def analyze_single_painting(img: Image.Image, url: str, mode: str):
    """Run the selected single-painting analysis with one shared input/output path."""
    mode = mode or "Classifier"
    if mode == "Classifier":
        label_md, fig, description = classify_painting(img, url)
        if description:
            label_md = f"{label_md}\n\n{description}"
        return label_md, fig
    if mode == "Wölfflin":
        return wolfflin_analysis(img, url)
    if mode == "Arnheim":
        return arnheim_perceptual(img, url)
    return "Choose an analysis mode.", None

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Art Movement Classifier") as demo:
    gr.Markdown(
        "# Art Movement Classifier\n"
        "Upload a painting to classify its likely art movement using CLIP image embeddings "
        "and a trained MLP classifier. Wölfflin gives a mainly Renaissance-Baroque "
        "theoretical reference, while Arnheim shows perceptual similarity on anchor-based visual axes."
    )

    with gr.Tab("Single Painting"):
        gr.Markdown(
            "Use one image input for all analyses. Pick a mode, run it, then switch modes "
            "without re-uploading the painting."
        )
        with gr.Row():
            with gr.Column(scale=1):
                single_img = gr.Image(type="pil", sources=["upload", "clipboard"], label="Upload or drag painting")
                single_url = gr.Textbox(placeholder="https://...  (direct image URL)", label="Or paste image URL")
                example_choice = gr.Dropdown(
                    choices=[label for label, _ in EXAMPLE_IMAGE_PATHS],
                    label="Try an example painting",
                    value=None,
                )
                load_example_btn = gr.Button("Load Example")
                mode = gr.Radio(
                    choices=["Classifier", "Wölfflin", "Arnheim"],
                    value="Classifier",
                    label="Analysis mode",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                single_md = gr.Markdown()
                single_plot = gr.Plot()

        load_example_btn.click(
            load_example_image,
            inputs=example_choice,
            outputs=[single_img, single_url],
        )
        analyze_btn.click(
            analyze_single_painting,
            inputs=[single_img, single_url, mode],
            outputs=[single_md, single_plot],
        )

        gr.Markdown(
            "**Mode notes:** Classifier shows movement probabilities. Wölfflin shows the "
            "theoretical Renaissance-Baroque profile of the predicted movement. Arnheim scores "
            "the uploaded image itself and lists nearest perceptual profiles; those profiles are "
            "interpretive similarities, not extra class predictions."
        )

    with gr.Tab("Collection Analyzer"):
        gr.Markdown(
            "Classify multiple paintings and summarize the predicted movement distribution. "
            "This is useful for quickly checking whether a small collection is visually concentrated "
            "in one movement or spread across several."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gallery_in = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Upload paintings (drag & drop multiple files)",
                )
                urls_in    = gr.Textbox(
                    lines=4,
                    placeholder="https://example.com/painting1.jpg\nhttps://example.com/painting2.jpg",
                    label="Or paste image URLs (one per line)",
                )
                analyze_collection_btn = gr.Button("Analyse Collection", variant="primary")
            with gr.Column(scale=2):
                collection_md    = gr.Markdown()
                collection_radar = gr.Plot()
        analyze_collection_btn.click(
            analyze_collection,
            inputs=[gallery_in, urls_in],
            outputs=[collection_md, collection_radar],
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
