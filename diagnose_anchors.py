"""
Anchor diagnostic script for Arnheim analysis.

Checks anchor coverage, within-group coherence, axis strength,
and top/bottom paintings per dimension.

Run from lab3_wölfflin/ with venv activated:
    python diagnose_anchors.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("/Users/folander/Scripts/Information Retrieval/lab3_wölfflin")
DATA_CSV   = BASE_DIR / "data/metadata.csv"
CACHE_PATH = BASE_DIR / "models/embeddings_clip.npz"
OUT_DIR    = BASE_DIR / "outputs/arnheim/anchor_diagnostics"

# ---------------------------------------------------------------------------
# Dimension definitions — updated anchors (must stay in sync with arnheim_analysis.py)
# ---------------------------------------------------------------------------
DIMENSIONS = {
    "Balance": {
        "description": "High = symmetrical/centered composition, Low = asymmetrical/dynamic",
        "high": ["Pietro Perugino", "Cima da Conegliano", "Giovanni Bellini", "Filippino Lippi",
                 "Luca Signorelli", "Vittore Carpaccio", "Andy Warhol", "Roy Lichtenstein",
                 "Stuart Davis", "Andrea Solario"],
        "low":  ["Edgar Degas", "Claude Monet", "Jackson Pollock", "Willem de Kooning",
                 "Arshile Gorky", "Caravaggio", "Edouard Manet", "Eugene Boudin",
                 "Frederic Bazille", "Peter Paul Rubens"],
    },
    "Shape": {
        "description": "High = geometric/regular forms, Low = organic/fluid forms",
        "high": ["Georges Braque", "Paul Cezanne", "Fernand Leger", "Kazimir Malevich",
                 "Piet Mondrian", "Stuart Davis", "Roy Lichtenstein", "Albert Gleizes",
                 "Roger de La Fresnaye", "Natalia Goncharova"],
        "low":  ["Claude Monet", "Pierre-Auguste Renoir", "Berthe Morisot", "Eugene Boudin",
                 "Jackson Pollock", "Arshile Gorky", "Morris Graves", "Edvard Munch",
                 "Johan Jongkind", "Frederic Bazille"],
    },
    "Depth": {
        "description": "High = deep spatial recession, Low = flat/planar",
        "high": ["Pietro Perugino", "Vittore Carpaccio", "Leonardo da Vinci", "Andrea Mantegna",
                 "Luca Signorelli", "Paul Bril", "Adam Elsheimer", "Giovanni Antonio Boltraffio",
                 "Cima da Conegliano", "Filippino Lippi"],
        "low":  ["Jackson Pollock", "Andy Warhol", "Roy Lichtenstein", "Wassily Kandinsky",
                 "Joan Miro", "Jasper Johns", "Stuart Davis", "Piet Mondrian",
                 "Richard Pousette-Dart", "Hassan Massoudy"],
    },
    "Tension": {
        "description": "High = energetic/dynamic movement, Low = static/calm",
        "high": ["Caravaggio", "Annibale Carracci", "Agostino Carracci", "Peter Paul Rubens",
                 "Edvard Munch", "James Ensor", "Kathe Kollwitz", "Jackson Pollock",
                 "Arshile Gorky", "Max Ernst"],
        "low":  ["Claude Monet", "Eugene Boudin", "Pierre-Auguste Renoir", "Berthe Morisot",
                 "Pietro Perugino", "Cima da Conegliano", "Giovanni Bellini", "Frederic Bazille",
                 "Johan Jongkind", "Camille Pissarro"],
    },
    "Light": {
        "description": "High = dramatic contrast/chiaroscuro, Low = even/diffused light",
        "high": ["Caravaggio", "Annibale Carracci", "Agostino Carracci", "Guido Reni",
                 "Adam Elsheimer", "Kathe Kollwitz", "Edvard Munch", "James Ensor",
                 "Frans Francken the Younger", "Giovanni Bellini"],
        "low":  ["Claude Monet", "Eugene Boudin", "Frederic Bazille", "Pierre-Auguste Renoir",
                 "Berthe Morisot", "Camille Pissarro", "Andy Warhol", "Roy Lichtenstein",
                 "Stuart Davis", "Johan Jongkind"],
    },
    "Color": {
        "description": "High = vivid/saturated, Low = muted/tonal",
        "high": ["Henri Matisse", "Marc Chagall", "Edvard Munch", "James Ensor",
                 "Hiro Yamagata", "Andy Warhol", "Alex Katz", "Joan Miro",
                 "Eduardo Paolozzi", "Wassily Kandinsky"],
        "low":  ["Eugene Boudin", "Johan Jongkind", "Frederic Bazille", "Camille Pissarro",
                 "Caravaggio", "Annibale Carracci", "Kathe Kollwitz", "Albin Egger-Lienz",
                 "Morris Graves", "James McNeill Whistler"],
    },
}

DIM_NAMES = list(DIMENSIONS.keys())


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


def avg_pairwise_cosine(vecs):
    """Average pairwise cosine similarity among a set of vectors."""
    if len(vecs) < 2:
        return float("nan")
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    normed = vecs / norms
    # Sim matrix
    sim_mat = normed @ normed.T
    n = len(vecs)
    # Upper triangle, excluding diagonal
    idx = np.triu_indices(n, k=1)
    return float(sim_mat[idx].mean())


def find_anchor_indices(df, artist_names, side_label):
    """Return (found_artists, not_found_artists, indices, counts_per_artist)."""
    found = {}
    not_found = []
    indices = []
    artist_lower = df["artist"].str.lower().fillna("")

    for name in artist_names:
        mask = artist_lower.str.contains(name.lower(), regex=False)
        hits = df.index[mask].tolist()
        if hits:
            found[name] = len(hits)
            indices.extend(hits)
        else:
            not_found.append(name)

    indices = sorted(set(indices))
    return found, not_found, indices


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    df = pd.read_csv(DATA_CSV)
    df = df.reset_index(drop=True)

    # Load embeddings cache and match to metadata rows
    cache = np.load(CACHE_PATH, allow_pickle=True)
    cached_paths = list(cache["paths"])
    cached_embs  = cache["embeddings"]  # shape (N_cache, 512)

    # Build lookup: absolute path -> row in cache
    path_to_idx = {p: i for i, p in enumerate(cached_paths)}

    # Convert relative paths in metadata to absolute
    abs_paths = [str(BASE_DIR / fp) for fp in df["file_path"]]
    missing = sum(1 for p in abs_paths if p not in path_to_idx)
    if missing:
        print(f"WARNING: {missing} metadata paths not found in cache (will be zero)")

    D = cached_embs.shape[1]
    X = np.zeros((len(df), D))
    for i, p in enumerate(abs_paths):
        if p in path_to_idx:
            X[i] = cached_embs[path_to_idx[p]]

    # Normalize embeddings (unit vectors)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_normed = X / norms

    print(f"Loaded {len(df)} paintings, {D}-dim embeddings\n")

    # -------------------------------------------------------------------------
    # Step 1: Anchor coverage + coherence + axis strength
    # -------------------------------------------------------------------------
    lines = []

    summary_rows = []

    for dim_name in DIM_NAMES:
        d = DIMENSIONS[dim_name]
        high_names = d["high"]
        low_names  = d["low"]

        high_found, high_not_found, high_idx = find_anchor_indices(df, high_names, f"{dim_name}/HIGH")
        low_found,  low_not_found,  low_idx  = find_anchor_indices(df, low_names,  f"{dim_name}/LOW")

        lines.append(f"\n{'='*70}")
        lines.append(f"DIMENSION: {dim_name}")
        lines.append(f"{'='*70}")
        lines.append(f"  HIGH side ({len(high_found)}/{len(high_names)} artists found, {len(high_idx)} paintings):")
        for name, cnt in sorted(high_found.items()):
            lines.append(f"    FOUND:     {name}: {cnt} paintings")
        for name in high_not_found:
            lines.append(f"    NOT FOUND: {name}")

        lines.append(f"  LOW side ({len(low_found)}/{len(low_names)} artists found, {len(low_idx)} paintings):")
        for name, cnt in sorted(low_found.items()):
            lines.append(f"    FOUND:     {name}: {cnt} paintings")
        for name in low_not_found:
            lines.append(f"    NOT FOUND: {name}")

        # Coherence
        high_coh = avg_pairwise_cosine(X_normed[high_idx]) if len(high_idx) >= 2 else float("nan")
        low_coh  = avg_pairwise_cosine(X_normed[low_idx])  if len(low_idx)  >= 2 else float("nan")

        # Axis strength: cosine distance between centroids
        if len(high_idx) >= 1 and len(low_idx) >= 1:
            c_high = X_normed[high_idx].mean(axis=0)
            c_low  = X_normed[low_idx].mean(axis=0)
            axis_cos_sim = cosine_similarity(c_high, c_low)
            axis_dist = 1.0 - axis_cos_sim
        else:
            axis_dist = float("nan")

        lines.append(f"  HIGH coherence (avg pairwise cos sim): {high_coh:.3f}")
        lines.append(f"  LOW  coherence (avg pairwise cos sim): {low_coh:.3f}")
        lines.append(f"  Axis cosine distance (HIGH vs LOW centroid): {axis_dist:.3f}")

        # Flag issues
        issues = []
        if len(high_idx) < 4:
            issues.append(f"FEW HIGH ANCHORS ({len(high_idx)})")
        if len(low_idx) < 4:
            issues.append(f"FEW LOW ANCHORS ({len(low_idx)})")
        if not np.isnan(high_coh) and high_coh < 0.45:
            issues.append(f"HIGH incoherent ({high_coh:.3f})")
        if not np.isnan(low_coh) and low_coh < 0.45:
            issues.append(f"LOW incoherent ({low_coh:.3f})")
        if not np.isnan(axis_dist) and axis_dist < 0.25:
            issues.append(f"WEAK AXIS ({axis_dist:.3f})")

        lines.append(f"  Issues: {', '.join(issues) if issues else 'OK'}")

        summary_rows.append({
            "Dimension": dim_name,
            "HIGH found": f"{len(high_found)}/{len(high_names)}",
            "LOW found":  f"{len(low_found)}/{len(low_names)}",
            "HIGH coh":   f"{high_coh:.3f}" if not np.isnan(high_coh) else "N/A",
            "LOW coh":    f"{low_coh:.3f}"  if not np.isnan(low_coh)  else "N/A",
            "Axis dist":  f"{axis_dist:.3f}" if not np.isnan(axis_dist) else "N/A",
            "Issues":     ", ".join(issues) if issues else "OK",
            # Store for later use
            "_high_idx": high_idx,
            "_low_idx":  low_idx,
        })

    # -------------------------------------------------------------------------
    # Step 2: Top/bottom 10 per dimension
    # -------------------------------------------------------------------------
    # We need arnheim_scores.csv for this
    scores_path = BASE_DIR / "outputs/arnheim/arnheim_scores.csv"
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        lines.append(f"\n{'='*70}")
        lines.append("TOP / BOTTOM 10 PAINTINGS PER DIMENSION (from arnheim_scores.csv)")
        lines.append("="*70)
        for dim_name in DIM_NAMES:
            if dim_name not in scores_df.columns:
                lines.append(f"\n  {dim_name}: column not found in scores CSV")
                continue
            sub = scores_df[["artist", "title", "label", dim_name]].dropna(subset=[dim_name])
            sub_sorted = sub.sort_values(dim_name, ascending=False)
            lines.append(f"\n  {dim_name} — TOP 10 (most {DIMENSIONS[dim_name]['description'].split(',')[0].replace('High = ', '')}):")
            for _, row in sub_sorted.head(10).iterrows():
                lines.append(f"    {row['artist']:35s}  {str(row['title'])[:40]:40s}  {row['label']:15s}  {row[dim_name]:.3f}")
            lines.append(f"\n  {dim_name} — BOTTOM 10:")
            for _, row in sub_sorted.tail(10).iterrows():
                lines.append(f"    {row['artist']:35s}  {str(row['title'])[:40]:40s}  {row['label']:15s}  {row[dim_name]:.3f}")
    else:
        lines.append("\n  arnheim_scores.csv not found — skipping top/bottom listing")

    # -------------------------------------------------------------------------
    # Step 3: Summary flag table
    # -------------------------------------------------------------------------
    lines.append(f"\n{'='*70}")
    lines.append("SUMMARY FLAG TABLE")
    lines.append("="*70)
    header = f"{'Dimension':<12} | {'HIGH found':>10} | {'LOW found':>9} | {'HIGH coh':>8} | {'LOW coh':>7} | {'Axis dist':>9} | Issues"
    sep    = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    for row in summary_rows:
        lines.append(
            f"{row['Dimension']:<12} | {row['HIGH found']:>10} | {row['LOW found']:>9} | "
            f"{row['HIGH coh']:>8} | {row['LOW coh']:>7} | {row['Axis dist']:>9} | {row['Issues']}"
        )
    lines.append(sep)

    # -------------------------------------------------------------------------
    # Step 4: Artist pool grouped by label
    # -------------------------------------------------------------------------
    lines.append(f"\n{'='*70}")
    lines.append("ARTIST POOL GROUPED BY LABEL")
    lines.append("="*70)
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        artist_counts = sub.groupby("artist").size().sort_values(ascending=False)
        lines.append(f"\n  {label} ({len(artist_counts)} artists, {len(sub)} paintings):")
        for artist, cnt in artist_counts.items():
            lines.append(f"    {artist}: {cnt}")

    # Print and save
    output_text = "\n".join(lines)
    print(output_text)

    out_txt = OUT_DIR / "anchor_coverage.txt"
    with open(out_txt, "w") as f:
        f.write(output_text)
    print(f"\nSaved coverage report to {out_txt}")

    # -------------------------------------------------------------------------
    # Step 5: PCA scatter plots per dimension
    # -------------------------------------------------------------------------
    print("\nGenerating PCA scatter plots...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_normed)

    for row in summary_rows:
        dim_name = row["Dimension"]
        high_idx = row["_high_idx"]
        low_idx  = row["_low_idx"]

        fig, ax = plt.subplots(figsize=(10, 7))

        # Background: all paintings (light grey)
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c="lightgrey", s=5, alpha=0.3, label="All paintings")

        # LOW anchors (blue)
        if low_idx:
            ax.scatter(X_pca[low_idx, 0], X_pca[low_idx, 1],
                       c="steelblue", s=40, alpha=0.8, label="LOW anchors", zorder=3)

        # HIGH anchors (red)
        if high_idx:
            ax.scatter(X_pca[high_idx, 0], X_pca[high_idx, 1],
                       c="crimson", s=40, alpha=0.8, label="HIGH anchors", zorder=4)

        # Label a few anchor points
        labeled_high = {}
        for idx in high_idx:
            artist = df.loc[idx, "artist"]
            if artist not in labeled_high:
                labeled_high[artist] = idx
        for artist, idx in list(labeled_high.items())[:8]:
            ax.annotate(artist, (X_pca[idx, 0], X_pca[idx, 1]),
                        fontsize=6, color="darkred", alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

        labeled_low = {}
        for idx in low_idx:
            artist = df.loc[idx, "artist"]
            if artist not in labeled_low:
                labeled_low[artist] = idx
        for artist, idx in list(labeled_low.items())[:8]:
            ax.annotate(artist, (X_pca[idx, 0], X_pca[idx, 1]),
                        fontsize=6, color="navy", alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_title(f"{dim_name}: Anchor positions in PCA space\n{DIMENSIONS[dim_name]['description']}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

        out_fig = OUT_DIR / f"pca_{dim_name.lower()}.png"
        plt.savefig(out_fig, dpi=120)
        plt.close()
        print(f"  Saved {out_fig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
