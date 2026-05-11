"""
Train and compare classifiers for painting art movement classification.

Steps:
  1. Load train/test metadata CSVs
  2. Encode all images with CLIP and ViT
  3. 5-fold stratified CV: compare 3 classifiers × 2 embedding models
  4. Re-train best combination on full train set; evaluate on held-out test
  5. Confusion matrix heatmap, per-class F1, confident correct/wrong predictions
  6. Wölfflin embedding space analysis — do class centroids align with theory?
  7. Save best model + config

Usage:
  python train.py [--skip-vit] [--batch-size 64]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    ViTModel, AutoFeatureExtractor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

TRAIN_CSV  = Path("data/train.csv")
TEST_CSV   = Path("data/test.csv")
MODELS_DIR = Path("models")
RANDOM_STATE = 42

CLASSES = [
    "Renaissance", "Baroque", "Impressionism", "Expressionism",
    "Cubism", "Abstract", "Surrealism", "Pop Art",
]

# ---------------------------------------------------------------------------
# Wölfflin theoretical positions for each class.
# Scale: -1.0 = strongly classical pole, +1.0 = strongly non-classical pole.
# Axes: linear↔painterly, plane↔recession, closed↔open, multiplicity↔unity,
#        clearness↔unclearness
# ---------------------------------------------------------------------------
WOLFFLIN_THEORY = {
    # axis order: linear_painterly, plane_recession, closed_open, multiplicity_unity, clearness_unclearness
    "Renaissance":  [-1.0, -0.8, -0.9, -0.7, -1.0],
    "Baroque":      [ 0.9,  0.8,  0.7,  0.8,  0.6],
    "Impressionism":[ 0.9,  0.3,  0.6,  0.5,  0.5],
    "Expressionism":[ 0.6,  0.2,  0.7,  0.7,  0.4],
    "Cubism":       [-0.5, -0.7, -0.3, -0.4, -0.3],  # geometric but fragmented
    "Abstract":     [ 0.7, -0.1,  0.8,  0.8,  0.7],
    "Surrealism":   [-0.3,  0.2,  0.1,  0.2,  0.3],  # linear rendering, unclear content
    "Pop Art":      [-0.8, -0.5, -0.6, -0.5, -0.7],  # flat, clear, but anti-classical intent
}
WOLFFLIN_AXES = ["Linear↔Painterly", "Plane↔Recession", "Closed↔Open",
                 "Multiplicity↔Unity", "Clearness↔Unclearness"]

EMBEDDING_MODELS = {
    "CLIP":  "openai/clip-vit-base-patch32",
    "ViT":   "google/vit-base-patch16-224",
}

CLASSIFIERS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    "SVM-RBF":            lambda: SVC(kernel="rbf", probability=True, C=1.0, random_state=RANDOM_STATE),
    "MLP":                lambda: MLPClassifier(
        hidden_layer_sizes=(512, 256), max_iter=500,
        early_stopping=True, n_iter_no_change=15, random_state=RANDOM_STATE
    ),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Image loading & augmentation
# ---------------------------------------------------------------------------

def load_image(path: str, size: int = 224) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def encode_clip(image_paths: list[str], batch_size: int) -> np.ndarray:
    log.info("  Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(EMBEDDING_MODELS["CLIP"])
    model = CLIPModel.from_pretrained(EMBEDDING_MODELS["CLIP"]).to(DEVICE)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="  CLIP encode"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [load_image(p) for p in batch_paths]
        valid = [(img, j) for j, img in enumerate(imgs) if img is not None]
        if not valid:
            # fill with zeros for failed images
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


def encode_vit(image_paths: list[str], batch_size: int) -> np.ndarray:
    log.info("  Loading ViT model...")
    extractor = AutoFeatureExtractor.from_pretrained(EMBEDDING_MODELS["ViT"])
    model     = ViTModel.from_pretrained(EMBEDDING_MODELS["ViT"]).to(DEVICE)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="  ViT encode"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [load_image(p) for p in batch_paths]
        valid = [(img, j) for j, img in enumerate(imgs) if img is not None]
        if not valid:
            embeddings.append(np.zeros((len(batch_paths), 768)))
            continue
        batch_imgs = [v[0] for v in valid]
        idx        = [v[1] for v in valid]
        inputs = extractor(images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out   = model(**inputs)
            feats = out.last_hidden_state[:, 0, :]  # CLS token
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = np.zeros((len(batch_paths), feats.shape[-1]))
        for k, orig_idx in enumerate(idx):
            arr[orig_idx] = feats[k].cpu().numpy()
        embeddings.append(arr)
    del model
    return np.vstack(embeddings)


ENCODERS = {"CLIP": encode_clip, "ViT": encode_vit}


def encode_with_cache(
    emb_name: str,
    encoder,
    image_paths: list[str],
    batch_size: int,
) -> np.ndarray:
    """Load cached embeddings and encode only missing images.

    Caches may contain absolute paths from an older project folder name, so we
    match both absolute paths and relative data/... paths before deciding an
    image needs to be re-encoded.
    """
    cache_file = MODELS_DIR / f"embeddings_{emb_name.lower()}.npz"
    abs_paths = [str(Path(p).resolve()) for p in image_paths]
    rel_paths = [str(Path(p)) for p in image_paths]
    cached_by_abs: dict[str, np.ndarray] = {}
    cached_by_rel: dict[str, np.ndarray] = {}
    dim: int | None = None

    if cache_file.exists():
        data = np.load(str(cache_file), allow_pickle=True)
        cached_paths = [str(p) for p in data["paths"]]
        cached_embeddings = data["embeddings"]
        dim = cached_embeddings.shape[1]
        for i, cached_path in enumerate(cached_paths):
            emb = cached_embeddings[i]
            cached_by_abs[cached_path] = emb
            parts = Path(cached_path).parts
            if "data" in parts:
                data_pos = parts.index("data")
                cached_by_rel[str(Path(*parts[data_pos:]))] = emb
        log.info("  Loaded %d cached %s embeddings from %s", len(cached_paths), emb_name, cache_file)

    missing_abs_paths = []
    encoded_missing = False
    X_rows: list[np.ndarray | None] = []
    for abs_path, rel_path in zip(abs_paths, rel_paths):
        if abs_path in cached_by_abs:
            X_rows.append(cached_by_abs[abs_path])
        elif rel_path in cached_by_rel:
            X_rows.append(cached_by_rel[rel_path])
        else:
            X_rows.append(None)
            missing_abs_paths.append(abs_path)

    if missing_abs_paths:
        log.info("  Encoding %d new/missing %s images", len(missing_abs_paths), emb_name)
        missing_embeddings = encoder(missing_abs_paths, batch_size)
        encoded_missing = True
        dim = missing_embeddings.shape[1]
        missing_iter = iter(missing_embeddings)
        for i, row in enumerate(X_rows):
            if row is None:
                X_rows[i] = next(missing_iter)
    else:
        log.info("  Reusing cached %s embeddings for all %d images", emb_name, len(abs_paths))

    if dim is None:
        raise RuntimeError(f"No {emb_name} embeddings available")

    X = np.stack([row for row in X_rows if row is not None])
    if encoded_missing:
        np.savez_compressed(str(cache_file), embeddings=X, paths=np.array(abs_paths))
        log.info("  Saved current %s embedding cache -> %s", emb_name, cache_file)
    return X


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_cv(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> dict[str, tuple[float, float]]:
    results = {}
    for clf_name, clf_fn in CLASSIFIERS.items():
        print(f"    {clf_name:<22}", end=" ", flush=True)
        scores = cross_validate(clf_fn(), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean, std = scores["test_score"].mean(), scores["test_score"].std()
        results[clf_name] = (mean, std)
        print(f"{mean:.3f} ± {std:.3f}")
    return results


def comparison_results_df(all_results: dict) -> pd.DataFrame:
    rows = []
    for (emb, clf), (mean, std) in all_results.items():
        rows.append({
            "embedding": emb,
            "classifier": clf,
            "cv_accuracy": float(mean),
            "cv_std": float(std),
        })
    return pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)


def save_comparison_results(all_results: dict) -> pd.DataFrame:
    comparison_df = comparison_results_df(all_results)
    out_csv = MODELS_DIR / "classifier_comparison.csv"
    out_json = MODELS_DIR / "classifier_comparison.json"
    comparison_df.to_csv(out_csv, index=False)
    comparison_df.to_json(out_json, orient="records", indent=2)
    print(f"Saved classifier comparison -> {out_csv}")
    print(f"Saved classifier comparison -> {out_json}")
    return comparison_df


def print_comparison_table(all_results: dict) -> tuple[str, str]:
    print("\n" + "═" * 70)
    print(f"{'Model Comparison — 5-fold Stratified CV':^70}")
    print("═" * 70)
    print(f"  {'Embedding':<12} {'Classifier':<24} {'Acc':>7}   {'Std':>7}")
    print("─" * 70)
    sorted_r = sorted(all_results.items(), key=lambda x: -x[1][0])
    best_key = sorted_r[0][0]
    for (emb, clf), (mean, std) in sorted_r:
        marker = "  ←" if (emb, clf) == best_key else ""
        print(f"  {emb:<12} {clf:<24} {mean:.3f}   ± {std:.3f}{marker}")
    print("═" * 70)
    best_acc = all_results[best_key][0]
    print(f"\nBest: {best_key[0]} + {best_key[1]}  (CV acc = {best_acc:.3f})")
    return best_key

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def print_confident_predictions(
    paths_test: list[str], y_test: np.ndarray, y_pred: np.ndarray,
    probs: np.ndarray, le: LabelEncoder, n: int = 5
) -> None:
    max_conf = probs.max(axis=1)
    correct  = y_pred == y_test
    print(f"\n{'═'*65}")
    print(f"Top {n} most confident CORRECT predictions")
    print("─" * 65)
    idxs = np.where(correct)[0]
    for i in idxs[np.argsort(max_conf[idxs])[::-1][:n]]:
        print(f"  [{le.classes_[y_test[i]]:<18}]  conf={max_conf[i]*100:.1f}%")
        print(f"  {paths_test[i]}")
        print()
    print(f"Top {n} most confident WRONG predictions")
    print("─" * 65)
    idxs = np.where(~correct)[0]
    if not len(idxs):
        print("  (none!)")
        return
    for i in idxs[np.argsort(max_conf[idxs])[::-1][:n]]:
        print(f"  true=[{le.classes_[y_test[i]]:<18}]  pred=[{le.classes_[y_pred[i]]:<18}]  conf={max_conf[i]*100:.1f}%")
        print(f"  {paths_test[i]}")
        print()


# ---------------------------------------------------------------------------
# Wölfflin embedding-space analysis
# ---------------------------------------------------------------------------

def wolfflin_analysis(X_all: np.ndarray, y_all: np.ndarray, le: LabelEncoder,
                      emb_name: str) -> None:
    """
    Project class centroids onto Wölfflin's theoretical axes using PCA,
    then measure correlation between PC positions and theoretical scores.
    """
    print(f"\n{'═'*65}")
    print("Wölfflin Embedding-Space Analysis")
    print("─" * 65)

    # Compute per-class centroid
    centroids = {}
    for cls in CLASSES:
        cls_id = le.transform([cls])[0]
        mask = y_all == cls_id
        if mask.sum() == 0:
            continue
        centroids[cls] = X_all[mask].mean(axis=0)

    cls_names = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in cls_names])

    # PCA to 2D for visualisation
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_2d = pca.fit_transform(centroid_matrix)

    # Compare PC1 ranking vs Wölfflin linear↔painterly axis
    theory_lp = [WOLFFLIN_THEORY[c][0] for c in cls_names]
    pc1_vals  = coords_2d[:, 0]

    from scipy.stats import spearmanr
    try:
        rho, pval = spearmanr(pc1_vals, theory_lp)
        print(f"\nPC1 vs Wölfflin Linear↔Painterly axis:")
        print(f"  Spearman ρ = {rho:.3f}  (p = {pval:.3f})")
        if abs(rho) > 0.5:
            direction = "aligns" if rho > 0 else "inverts"
            print(f"  → Embedding PC1 {direction} with Wölfflin's classical↔non-classical axis")
        else:
            print("  → Weak alignment — embedding organises paintings differently from Wölfflin")
    except ImportError:
        print("  (install scipy for correlation analysis)")

    # Save 2D centroid plot
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, len(cls_names)))
    for i, cls in enumerate(cls_names):
        ax.scatter(*coords_2d[i], s=200, color=colors[i], zorder=3, label=cls)
        ax.annotate(cls, coords_2d[i], fontsize=9,
                    xytext=(8, 4), textcoords="offset points")

    # Annotate with Wölfflin theory position (linear↔painterly as color intensity)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"Class Centroids in {emb_name} Embedding Space\n"
                 f"(left = Wölfflin Classical, right = Non-classical — hypothesis)")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    plt.tight_layout()
    out = MODELS_DIR / f"wolfflin_pca_{emb_name.lower()}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wölfflin PCA plot saved to {out}")

    # Print ranking on PC1
    ranking = sorted(zip(cls_names, pc1_vals), key=lambda x: x[1])
    print("\nClass ranking on PC1 (low → high):")
    for cls, val in ranking:
        theory_pos = WOLFFLIN_THEORY.get(cls, [0])[0]
        print(f"  {cls:<22} PC1={val:+.3f}   theory_LP={theory_pos:+.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vit", action="store_true",
                        help="Skip ViT encoder (faster)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    # Load data
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError("Run data/build_dataset.py first")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    all_df   = pd.concat([train_df, test_df], ignore_index=True)

    le = LabelEncoder()
    le.fit(CLASSES)

    train_paths = train_df["file_path"].tolist()
    test_paths  = test_df["file_path"].tolist()
    all_paths   = all_df["file_path"].tolist()

    y_train = le.transform(train_df["label"])
    y_test  = le.transform(test_df["label"])
    y_all   = le.transform(all_df["label"])

    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_results:   dict[tuple[str, str], tuple[float, float]] = {}
    embeddings_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    models_to_run = dict(EMBEDDING_MODELS)
    if args.skip_vit:
        models_to_run.pop("ViT", None)

    for emb_name, emb_model_name in models_to_run.items():
        print(f"\n── Embedding: {emb_name} ({emb_model_name}) ──")
        encoder = ENCODERS[emb_name]

        try:
            X_all = encode_with_cache(emb_name, encoder, all_paths, args.batch_size)
            X_train = X_all[:len(train_paths)]
            X_test = X_all[len(train_paths):]
        except Exception as e:
            print(f"  [skip] Encoding failed: {e}")
            continue

        embeddings_cache[emb_name] = (X_train, X_test, X_all)

        print(f"  5-fold CV on training set ({len(train_df)} images)...")
        clf_results = run_cv(X_train, y_train, cv)
        for clf_name, stats in clf_results.items():
            all_results[(emb_name, clf_name)] = stats

    if not all_results:
        print("No results. Check that embedding models are downloadable.")
        return

    best_emb_name, best_clf_name = print_comparison_table(all_results)
    comparison_df = save_comparison_results(all_results)
    comparison_records = comparison_df.to_dict(orient="records")
    X_train_best, X_test_best, X_all_best = embeddings_cache[best_emb_name]

    # Final evaluation
    print(f"\n── Final evaluation: {best_emb_name} + {best_clf_name} ──")
    best_clf = CLASSIFIERS[best_clf_name]()
    best_clf.fit(X_train_best, y_train)
    y_pred = best_clf.predict(X_test_best)
    probs  = best_clf.predict_proba(X_test_best)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nHeld-out test accuracy: {acc:.4f}  ({acc*100:.1f}%)")
    print("\nPer-class F1:")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    hardest = []
    for cls in le.classes_:
        r = report[cls]
        hardest.append((cls, r["f1-score"]))
        print(f"  {cls:<22}  F1={r['f1-score']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}")
    hardest.sort(key=lambda x: x[1])
    print(f"\nHardest to classify: {hardest[0][0]} (F1={hardest[0][1]:.3f})")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, list(le.classes_),
        f"Confusion Matrix — {best_emb_name} + {best_clf_name}",
        MODELS_DIR / "confusion_matrix.png",
    )

    print_confident_predictions(test_paths, y_test, y_pred, probs, le)

    metrics = {
        "heldout_accuracy": float(acc),
        "best_embedding": best_emb_name,
        "best_classifier": best_clf_name,
        "cv_accuracy": float(all_results[(best_emb_name, best_clf_name)][0]),
        "cv_std": float(all_results[(best_emb_name, best_clf_name)][1]),
        "classifier_comparison": comparison_records,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
            }
            for cls in le.classes_
        },
    }
    with open(MODELS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {MODELS_DIR}/evaluation_metrics.json")

    # Wölfflin analysis on training embeddings
    wolfflin_analysis(X_train_best, y_train, le, best_emb_name)

    # Re-train on full data and save
    print("\nRetraining on full dataset...")
    final_clf = CLASSIFIERS[best_clf_name]()
    final_clf.fit(X_all_best, y_all)

    joblib.dump(final_clf,  MODELS_DIR / "classifier.joblib")
    joblib.dump(le,         MODELS_DIR / "label_encoder.joblib")

    config = {
        "embedding_model": EMBEDDING_MODELS[best_emb_name],
        "embedding_name":  best_emb_name,
        "classifier":      best_clf_name,
        "classes":         CLASSES,
    }
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved → {MODELS_DIR}/classifier.joblib")
    print(f"Saved → {MODELS_DIR}/config.json  {config}")


if __name__ == "__main__":
    main()
