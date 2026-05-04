"""
Diagnostics for Arnheim anchor calibration.

This script uses the live Arnheim anchors from arnheim_analysis.py, checks
coverage/coherence/axis strength, lists top and bottom scoring paintings, and
summarizes empirical profile overlap. It is meant for calibrating the axes, not
for forcing Arnheim results to agree with classifier labels.

Usage:
    python diagnose_anchors.py
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from arnheim_analysis import DIMENSIONS, DIM_NAMES, CLASSES

BASE_DIR = Path(__file__).parent
DATA_CSV = BASE_DIR / "data/metadata.csv"
CACHE_PATH = BASE_DIR / "models/embeddings_clip.npz"
SCORES_PATH = BASE_DIR / "outputs/arnheim/arnheim_scores.csv"
OUT_DIR = BASE_DIR / "outputs/arnheim/anchor_diagnostics"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


def avg_pairwise_cosine(vecs: np.ndarray) -> float:
    if len(vecs) < 2:
        return float("nan")
    normed = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    sim_mat = normed @ normed.T
    idx = np.triu_indices(len(vecs), k=1)
    return float(sim_mat[idx].mean())


def anchor_label(entry) -> str:
    if isinstance(entry, str):
        return entry
    artist_kw, title_kw = entry[0], entry[1]
    return f"{artist_kw} / {title_kw}" if title_kw else str(artist_kw)


def find_anchor_indices(df: pd.DataFrame, anchors: list) -> tuple[dict[str, int], list[str], list[int]]:
    artist_lower = df["artist"].str.lower().fillna("")
    title_lower = df["title"].str.lower().fillna("")
    found: dict[str, int] = {}
    not_found: list[str] = []
    indices: list[int] = []

    for entry in anchors:
        if isinstance(entry, str):
            artist_kw, title_kw = entry, None
        else:
            artist_kw, title_kw = entry[0], entry[1]

        mask = artist_lower.str.contains(str(artist_kw).lower(), regex=False)
        if title_kw:
            mask = mask & title_lower.str.contains(str(title_kw).lower(), regex=False)

        hits = df.index[mask].tolist()
        label = anchor_label(entry)
        if hits:
            found[label] = len(hits)
            indices.extend(hits)
        else:
            not_found.append(label)

    return found, not_found, sorted(set(indices))


def load_embeddings(df: pd.DataFrame) -> np.ndarray:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Embedding cache not found at {CACHE_PATH}. Run python arnheim_analysis.py first."
        )

    cache = np.load(CACHE_PATH, allow_pickle=True)
    cached_paths = list(cache["paths"])
    cached_embs = cache["embeddings"]
    path_to_idx = {p: i for i, p in enumerate(cached_paths)}
    rel_to_idx: dict[str, int] = {}
    for i, cached_path in enumerate(cached_paths):
        parts = Path(str(cached_path)).parts
        if "data" in parts:
            data_pos = parts.index("data")
            rel_to_idx[str(Path(*parts[data_pos:]))] = i

    x = np.zeros((len(df), cached_embs.shape[1]))
    missing = 0
    for i, fp in enumerate(df["file_path"]):
        abs_path = str(BASE_DIR / fp)
        if abs_path in path_to_idx:
            x[i] = cached_embs[path_to_idx[abs_path]]
        elif fp in rel_to_idx:
            x[i] = cached_embs[rel_to_idx[fp]]
        else:
            missing += 1

    if missing:
        print(f"WARNING: {missing} metadata paths missing from embedding cache; zero-filled")
    return x


def score_profile_matches(scores_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = scores_df.groupby("label")[DIM_NAMES].mean().reindex(CLASSES)
    rows = []
    for _, row in scores_df.dropna(subset=DIM_NAMES).iterrows():
        vec = row[DIM_NAMES].to_numpy(dtype=float)
        matches = []
        for label, prof in profiles.iterrows():
            p = prof.to_numpy(dtype=float)
            dist = float(np.linalg.norm(vec - p))
            sim = 1.0 / (1.0 + dist)
            matches.append((label, sim))
        matches.sort(key=lambda item: item[1], reverse=True)
        rows.append({
            "file_path": row["file_path"],
            "label": row["label"],
            "artist": row.get("artist", ""),
            "title": row.get("title", ""),
            "nearest_profile": matches[0][0],
            "nearest_similarity": matches[0][1],
            "label_profile_similarity": dict(matches).get(row["label"], math.nan),
            "top3_profiles": ", ".join(f"{g}:{s:.2f}" for g, s in matches[:3]),
        })
    match_df = pd.DataFrame(rows)
    matrix = pd.crosstab(match_df["label"], match_df["nearest_profile"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )
    return match_df, matrix



def dataframe_to_markdown(df: pd.DataFrame) -> str:
    out = []
    reset = df.reset_index()
    cols = [str(c) for c in reset.columns]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in reset.iterrows():
        vals = []
        for c in reset.columns:
            value = row[c]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)

def write_markdown_report(
    df: pd.DataFrame,
    x_normed: np.ndarray,
    scores_df: pd.DataFrame | None,
    summary_rows: list[dict],
    match_matrix: pd.DataFrame | None,
) -> None:
    lines = [
        "# Arnheim Anchor Calibration Report",
        "",
        "This report checks whether the Arnheim axes are visually plausible. It should be used to tune anchor pools, not to force Arnheim profile matches to agree with classifier labels.",
        "",
        "## Anchor Coverage And Axis Strength",
        "",
        "| Dimension | High Anchors | Low Anchors | High Coherence | Low Coherence | Axis Distance | Flags |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dimension']} | {row['high_count']} | {row['low_count']} | "
            f"{row['high_coherence']:.3f} | {row['low_coherence']:.3f} | "
            f"{row['axis_distance']:.3f} | {row['flags']} |"
        )

    lines += [
        "",
        "## Top And Bottom Paintings By Axis",
        "",
        "Use these lists as the main sanity check. If the top/bottom examples do not look like the intended perceptual pole, adjust the anchor pool for that axis.",
    ]

    if scores_df is None:
        lines.append("")
        lines.append("`outputs/arnheim/arnheim_scores.csv` was not found, so score extremes were skipped.")
    else:
        for dim in DIM_NAMES:
            lines += ["", f"### {dim}", "", f"{DIMENSIONS[dim]['description']}", ""]
            sub = scores_df[["artist", "title", "label", "split", "file_path", dim]].dropna(subset=[dim])
            top = sub.nlargest(10, dim)
            bottom = sub.nsmallest(10, dim)
            lines += ["**Highest scores**", "", "| Score | Label | Artist | Title | Split |", "|---:|---|---|---|---|"]
            for _, r in top.iterrows():
                lines.append(f"| {r[dim]:.3f} | {r['label']} | {r['artist']} | {str(r['title'])[:55]} | {r['split']} |")
            lines += ["", "**Lowest scores**", "", "| Score | Label | Artist | Title | Split |", "|---:|---|---|---|---|"]
            for _, r in bottom.iterrows():
                lines.append(f"| {r[dim]:.3f} | {r['label']} | {r['artist']} | {str(r['title'])[:55]} | {r['split']} |")

        lines += ["", "## Per-Class Mean Scores", ""]
        means = scores_df.groupby("label")[DIM_NAMES].mean().reindex(CLASSES).round(3)
        lines.append(dataframe_to_markdown(means))

    if match_matrix is not None:
        lines += [
            "",
            "## Nearest Empirical Profile Overlap",
            "",
            "Rows are dataset labels; columns are the nearest Arnheim perceptual profile. This is not classifier accuracy. It reveals where the six-axis perceptual space overlaps across movements.",
            "",
            dataframe_to_markdown(match_matrix),
        ]

    lines += [
        "",
        "## Calibration Rules",
        "",
        "- Tune anchors only when the top/bottom paintings are visibly wrong for the intended axis.",
        "- Prefer specific `(artist, title)` anchors over broad artist anchors when an artist has multiple visual modes.",
        "- Do not tune an axis simply because a painting's nearest profile differs from its movement label.",
        "- After anchor edits, rerun `python arnheim_analysis.py --skip-encode`, then rerun this script.",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration_report.md").write_text("\n".join(lines))


def plot_anchor_pca(df: pd.DataFrame, x_normed: np.ndarray, summary_rows: list[dict]) -> None:
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_normed)
    for row in summary_rows:
        dim = row["dimension"]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(x_pca[:, 0], x_pca[:, 1], c="lightgrey", s=5, alpha=0.25, label="All paintings")
        low_idx = row["low_idx"]
        high_idx = row["high_idx"]
        if low_idx:
            ax.scatter(x_pca[low_idx, 0], x_pca[low_idx, 1], c="steelblue", s=40, alpha=0.8, label="LOW anchors")
        if high_idx:
            ax.scatter(x_pca[high_idx, 0], x_pca[high_idx, 1], c="crimson", s=40, alpha=0.8, label="HIGH anchors")

        for idx in high_idx[:8]:
            ax.annotate(str(df.loc[idx, "artist"])[:24], (x_pca[idx, 0], x_pca[idx, 1]), fontsize=6, color="darkred")
        for idx in low_idx[:8]:
            ax.annotate(str(df.loc[idx, "artist"])[:24], (x_pca[idx, 0], x_pca[idx, 1]), fontsize=6, color="navy")

        ax.set_title(f"{dim}: anchor positions in CLIP PCA space")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"pca_{dim.lower()}.png", dpi=130)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV).reset_index(drop=True)
    x = load_embeddings(df)
    x_normed = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    summary_rows: list[dict] = []
    coverage_lines = []
    for dim in DIM_NAMES:
        cfg = DIMENSIONS[dim]
        high_found, high_missing, high_idx = find_anchor_indices(df, cfg["high"])
        low_found, low_missing, low_idx = find_anchor_indices(df, cfg["low"])

        high_coh = avg_pairwise_cosine(x_normed[high_idx]) if len(high_idx) >= 2 else float("nan")
        low_coh = avg_pairwise_cosine(x_normed[low_idx]) if len(low_idx) >= 2 else float("nan")
        if high_idx and low_idx:
            high_centroid = x_normed[high_idx].mean(axis=0)
            low_centroid = x_normed[low_idx].mean(axis=0)
            axis_distance = 1.0 - cosine_similarity(high_centroid, low_centroid)
        else:
            axis_distance = float("nan")

        flags = []
        if len(high_idx) < 8:
            flags.append(f"few HIGH ({len(high_idx)})")
        if len(low_idx) < 8:
            flags.append(f"few LOW ({len(low_idx)})")
        if not math.isnan(high_coh) and high_coh < 0.45:
            flags.append(f"HIGH broad ({high_coh:.2f})")
        if not math.isnan(low_coh) and low_coh < 0.45:
            flags.append(f"LOW broad ({low_coh:.2f})")
        if not math.isnan(axis_distance) and axis_distance < 0.08:
            flags.append(f"weak axis ({axis_distance:.2f})")

        coverage_lines += [
            "",
            f"## {dim}",
            cfg["description"],
            "",
            f"HIGH: {len(high_idx)} paintings from {len(high_found)}/{len(cfg['high'])} anchors",
            *(f"  FOUND {k}: {v}" for k, v in high_found.items()),
            *(f"  MISSING {k}" for k in high_missing),
            f"LOW: {len(low_idx)} paintings from {len(low_found)}/{len(cfg['low'])} anchors",
            *(f"  FOUND {k}: {v}" for k, v in low_found.items()),
            *(f"  MISSING {k}" for k in low_missing),
            f"High coherence: {high_coh:.3f}",
            f"Low coherence: {low_coh:.3f}",
            f"Axis distance: {axis_distance:.3f}",
            f"Flags: {', '.join(flags) if flags else 'OK'}",
        ]

        summary_rows.append({
            "dimension": dim,
            "high_count": len(high_idx),
            "low_count": len(low_idx),
            "high_coherence": high_coh,
            "low_coherence": low_coh,
            "axis_distance": axis_distance,
            "flags": ", ".join(flags) if flags else "OK",
            "high_idx": high_idx,
            "low_idx": low_idx,
        })

    scores_df = pd.read_csv(SCORES_PATH) if SCORES_PATH.exists() else None
    match_df = None
    match_matrix = None
    if scores_df is not None:
        match_df, match_matrix = score_profile_matches(scores_df)
        match_df.to_csv(OUT_DIR / "profile_matches.csv", index=False)
        match_matrix.to_csv(OUT_DIR / "profile_overlap_matrix.csv")

        extreme_rows = []
        for dim in DIM_NAMES:
            sub = scores_df[["file_path", "label", "split", "artist", "title", dim]].dropna(subset=[dim])
            for rank, (_, r) in enumerate(sub.nlargest(20, dim).iterrows(), start=1):
                extreme_rows.append({"dimension": dim, "pole": "high", "rank": rank, **r.to_dict()})
            for rank, (_, r) in enumerate(sub.nsmallest(20, dim).iterrows(), start=1):
                extreme_rows.append({"dimension": dim, "pole": "low", "rank": rank, **r.to_dict()})
        pd.DataFrame(extreme_rows).to_csv(OUT_DIR / "dimension_extremes.csv", index=False)

    pd.DataFrame([{k: v for k, v in row.items() if not k.endswith("_idx")} for row in summary_rows]).to_csv(
        OUT_DIR / "anchor_summary.csv", index=False
    )
    (OUT_DIR / "anchor_coverage.txt").write_text("\n".join(coverage_lines).strip() + "\n")
    write_markdown_report(df, x_normed, scores_df, summary_rows, match_matrix)
    plot_anchor_pca(df, x_normed, summary_rows)

    print(f"Saved calibration report to {OUT_DIR / 'calibration_report.md'}")
    print(f"Saved anchor summary to {OUT_DIR / 'anchor_summary.csv'}")
    if scores_df is not None:
        print(f"Saved extremes to {OUT_DIR / 'dimension_extremes.csv'}")
        print(f"Saved profile overlap matrix to {OUT_DIR / 'profile_overlap_matrix.csv'}")


if __name__ == "__main__":
    main()
