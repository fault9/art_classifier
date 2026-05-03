"""
Print before/after dataset, classifier, and Arnheim summaries after rebalancing.

Usage:
  python compare_rebalance_results.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARNHEIM_DIR = BASE_DIR / "outputs" / "arnheim"


def top_artists(path: Path, n: int = 8) -> None:
    df = pd.read_csv(path)
    print(f"\nArtist distribution: {path.name}")
    for label, group in df.groupby("label"):
        top = group["artist"].value_counts().head(n)
        total = len(group)
        rendered = ", ".join(f"{artist}={count} ({count / total:.1%})" for artist, count in top.items())
        print(f"  {label:<15} n={total:<4} {rendered}")


def classification_accuracy() -> None:
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print(
            "\nClassification accuracy: "
            f"{metrics['heldout_accuracy']:.4f} "
            f"({metrics['best_embedding']} + {metrics['best_classifier']}, held-out test)"
        )
        return

    config_path = MODELS_DIR / "config.json"
    cache_path = MODELS_DIR / "embeddings_clip.npz"
    test_path = DATA_DIR / "test.csv"
    if not (config_path.exists() and cache_path.exists() and test_path.exists()):
        print("\nClassification accuracy: unavailable; train.py has not produced all artifacts.")
        return

    config = json.loads(config_path.read_text())
    if config.get("embedding_name") != "CLIP":
        print(f"\nClassification accuracy: trained embedding is {config.get('embedding_name')}; report script currently reads CLIP cache.")
        return

    test_df = pd.read_csv(test_path)
    clf = joblib.load(MODELS_DIR / "classifier.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    cache = np.load(cache_path, allow_pickle=True)
    path_to_idx = {str(p): i for i, p in enumerate(cache["paths"])}
    abs_paths = [str(Path(p).resolve()) for p in test_df["file_path"]]
    missing = [p for p in abs_paths if p not in path_to_idx]
    if missing:
        print(f"\nClassification accuracy: unavailable; {len(missing)} test embeddings missing.")
        return

    X_test = np.stack([cache["embeddings"][path_to_idx[p]] for p in abs_paths])
    y_test = le.transform(test_df["label"])
    y_pred = clf.predict(X_test)
    print(f"\nClassification accuracy: {accuracy_score(y_test, y_pred):.4f}")


def arnheim_axis_independence() -> None:
    axes_path = MODELS_DIR / "arnheim_axes.npz"
    if not axes_path.exists():
        print("\nArnheim axis independence: unavailable; arnheim_analysis.py has not run.")
        return
    axes_npz = np.load(axes_path, allow_pickle=True)
    names = [str(x) for x in axes_npz["dim_names"]]
    axes = axes_npz["axes"]
    sim = axes @ axes.T
    print("\nArnheim axis independence (cosine similarity):")
    print("  " + " ".join(f"{name[:7]:>8}" for name in names))
    for i, name in enumerate(names):
        vals = " ".join(f"{sim[i, j]:8.3f}" for j in range(len(names)))
        print(f"  {name[:7]:<7} {vals}")


def arnheim_profiles() -> None:
    scores_path = ARNHEIM_DIR / "arnheim_scores.csv"
    if not scores_path.exists():
        print("\nPer-class Arnheim profiles: unavailable; arnheim_analysis.py has not run.")
        return
    df = pd.read_csv(scores_path)
    dims = [c for c in ["Balance", "Shape", "Depth", "Tension", "Light", "Color"] if c in df.columns]
    means = df.groupby("label")[dims].mean().round(3)
    print("\nPer-class Arnheim profiles:")
    print(means.to_string())


def previous_accuracy_from_logs() -> None:
    candidates = sorted(BASE_DIR.glob("*.log")) + sorted((BASE_DIR / "outputs").glob("**/*.txt"))
    pattern = re.compile(r"Held-out test accuracy:\s*([0-9.]+)")
    hits = []
    for path in candidates:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for match in pattern.finditer(text):
            hits.append((path, match.group(1)))
    if hits:
        path, value = hits[-1]
        print(f"\nPrevious logged accuracy: {value} from {path.relative_to(BASE_DIR)}")


def main() -> None:
    top_artists(DATA_DIR / "metadata.csv")
    before = DATA_DIR / "artist_distribution_before_rebalance.csv"
    after = DATA_DIR / "artist_distribution_after_rebalance.csv"
    if before.exists() and after.exists():
        print(f"\nBefore/after artist distribution CSVs: {before.relative_to(BASE_DIR)}, {after.relative_to(BASE_DIR)}")
    previous_accuracy_from_logs()
    classification_accuracy()
    arnheim_axis_independence()
    arnheim_profiles()


if __name__ == "__main__":
    main()
