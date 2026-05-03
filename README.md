# Art Movement Classifier — Wölfflin Edition

An embeddings-based image classifier that identifies the art movement of a painting, with a theoretical analysis layer grounded in Heinrich Wölfflin's *Principles of Art History* (1915).

## Embeddings Lab Submission Summary

The short academic report for the course submission is in [`REPORT.md`](REPORT.md).

**Domain issue:** Art movement classifiers can learn artist-specific shortcuts instead of movement-level visual structure. In the original data, some artists dominated their class, so the dataset was rebalanced by capping artists and adding underrepresented artists.

**Challenge addressed with embeddings:** The project uses pretrained image embeddings to classify paintings with a small custom dataset, avoiding the need to train a full vision model from scratch.

**Embedding type:** CLIP image embeddings from `openai/clip-vit-base-patch32`.

**Trained model:** scikit-learn MLP classifier trained on frozen CLIP embeddings.

**Custom dataset:** 1,478 paintings across 8 art movements, with fixed train/test splits and artist-dominance audit files.

**Held-out accuracy:** 0.8093.

**Demo:** Gradio app in `app.py` with classification, collection analysis, Wölfflin profiles, and Arnheim perceptual profiles.

## The 8 Art Movement Classes

| Movement | Era | Key Artists | Visual Signature |
|----------|-----|-------------|-----------------|
| **Renaissance** | 14th–17th c. | Da Vinci, Michelangelo, Raphael, Botticelli | Mathematical perspective, balanced proportion, linear clarity |
| **Baroque** | 17th–18th c. | Caravaggio, Rembrandt, Vermeer, Rubens | Chiaroscuro, deep recession, diagonal movement, emotional intensity |
| **Impressionism** | 1860s–1880s | Monet, Renoir, Degas, Pissarro | Visible brushwork, broken color, light and atmosphere, everyday scenes |
| **Expressionism** | 1900s–1930s | Munch, Kirchner, Schiele, Nolde | Distorted forms, non-naturalistic color, psychological rawness |
| **Cubism** | 1908–1920s | Picasso, Braque, Gris, Léger | Geometric fragmentation, multiple simultaneous perspectives |
| **Abstract** | 1910s–present | Kandinsky, Pollock, Rothko, Mondrian | Non-representational, pure form and color, gestural or geometric |
| **Surrealism** | 1920s–1940s | Dalí, Magritte, Ernst, Miró | Dreamlike impossible imagery, subconscious, uncanny juxtapositions |
| **Pop Art** | 1950s–1970s | Warhol, Lichtenstein, Hockney, Rauschenberg | Bold flat color, commercial imagery, mass culture, irony |

---

## Wölfflin's Framework

Heinrich Wölfflin identified **five pairs of opposing visual principles** that describe the shift from Renaissance to Baroque — and by extension, from classical to non-classical visual art:

| Axis | Classical Pole | Non-Classical Pole |
|------|---------------|-------------------|
| **1. Linear ↔ Painterly** | Clear outlines, firm contours | Dissolved edges, color masses, visible brushwork |
| **2. Plane ↔ Recession** | Flat, layered parallel planes | Diagonal depth, foreshortening, spatial recession |
| **3. Closed ↔ Open** | Balanced, symmetrical, self-contained | Dynamic, asymmetric, breaking the frame |
| **4. Multiplicity ↔ Unity** | Each part independently articulated | Parts fused, subordinated to one dominant element |
| **5. Clearness ↔ Unclearness** | Everything visible and defined | Selective focus, shadow, suggestion, ambiguity |

### Theoretical Mapping to Our 8 Classes

| Genre | Linear↔Painterly | Plane↔Recession | Closed↔Open | Multiplicity↔Unity | Clearness↔Unclearness |
|-------|:---:|:---:|:---:|:---:|:---:|
| Renaissance | Classical | Classical | Classical | Classical | Classical |
| Baroque | Non-classical | Non-classical | Non-classical | Non-classical | Mixed |
| Impressionism | Non-classical | Mixed | Non-classical | Mixed | Mixed |
| Expressionism | Mixed | Mixed | Non-classical | Non-classical | Mixed |
| Cubism | Mixed classical | Classical | Mixed | Classical | Mixed |
| Abstract | Non-classical | Mixed | Non-classical | Non-classical | Non-classical |
| Surrealism | Classical | Mixed | Mixed | Mixed | Mixed |
| Pop Art | Classical | Classical | Classical | Classical | Classical |

**Key hypothesis:** If CLIP or ViT embeddings encode genuine visual properties, the principal components of the embedding space should correlate with Wölfflin's axes — especially the Linear↔Painterly axis, which is the most visually salient. `train.py` tests this with Spearman correlation between PC1 rankings and theoretical positions.

**Surrealism** is a fascinating edge case: Dalí paints with photorealistic precision (Classical/Linear) but depicts impossible dreamlike content (Unclear). It straddles the axes in a way Wölfflin never anticipated.

---

## Arnheim's Perceptual Framework

Rudolf Arnheim (*Art and Visual Perception*, 1954) analyzed how humans perceive visual form through Gestalt psychology. Unlike Wölfflin (who compares movements historically), Arnheim's dimensions describe **perceptual properties of individual paintings** that can be scored on a spectrum.

### The 6 Arnheim Dimensions

| Dimension | Low Pole | High Pole | Example Anchors |
|-----------|----------|-----------|-----------------|
| **Balance** | Asymmetrical, off-center, dynamic | Symmetrical, centered, stable | Low: Pollock, Degas / High: Raphael, Rothko |
| **Shape** | Organic, fluid, irregular | Geometric, angular, structured | Low: Monet, Munch / High: Mondrian, Picasso (Cubist) |
| **Depth** | Flat, planar, no recession | Deep space, perspective, recession | Low: Warhol, Mondrian / High: Leonardo, Vermeer |
| **Tension** | Static, calm, contemplative | Energetic, violent, turbulent | Low: Vermeer, Monet / High: Pollock, Rubens |
| **Light** | Even, diffused, luminous | Dramatic contrast, chiaroscuro | Low: Monet, Warhol / High: Caravaggio, Rembrandt |
| **Color** | Muted, tonal, restrained | Vivid, saturated, bold | Low: Rembrandt, Braque / High: Warhol, Kirchner |

### How Arnheim Complements Wölfflin

| Wölfflin Axis | Corresponding Arnheim Dimension(s) |
|--------------|-------------------------------------|
| Linear ↔ Painterly | Shape (geometric vs organic) |
| Plane ↔ Recession | Depth |
| Closed ↔ Open | Balance + Tension |
| Multiplicity ↔ Unity | Shape + Balance |
| Clearness ↔ Unclearness | Light + Shape |

**Key difference:** Wölfflin assigns a fixed theoretical position to each *genre* (all Baroque paintings get the same score). Arnheim scores *individual paintings* based on their actual position in CLIP embedding space relative to anchor paintings — so two paintings in the same genre can differ significantly.

### Implementation

Scores are derived by **anchor-based projection**:
1. For each dimension, define HIGH and LOW anchor paintings by artist name
2. Find those paintings in the training set and compute anchor centroids
3. The axis vector points from the LOW centroid to the HIGH centroid in 512-D CLIP space
4. Every painting is projected onto this axis → score in approximately [−1, +1]

If an anchor artist isn't in the training set (it's only ~1,400 paintings), their contribution is simply absent — the remaining anchors still define the axis.

### Limitations

- Anchor selection involves art-historical judgment and could be contested
- Projection is approximate: CLIP embeddings encode many properties simultaneously
- With a small training set, some anchor groups contain only 5–20 paintings — sparse representation of the perceptual extreme
- Scores are relative to the paintings in this specific dataset, not to art in general

---

## Approach

1. **Image embeddings** from pretrained vision models (CLIP, ViT)
2. **Shallow classifier** (logistic regression / SVM / MLP) on top of frozen embeddings
3. **5-fold cross-validation** to compare models; best combination selected automatically
4. **Wölfflin analysis** on class centroids via PCA — tests whether embedding geometry mirrors art historical theory

---

## Dataset

### Source 1: WikiArt (primary)
Scraped via WikiArt's public JSON API. ~200 images per class at web resolution.

### Source 2: HuggingFace — huggan/wikiart
~80k paintings with WikiArt labels. Used as supplementary source where WikiArt scraping is insufficient.

**Target:** 100–200 images per class, 800–1600 total.

---

## File Structure

```
lab3_wölfflin/
├── data/
│   ├── images/                   # Final painting images grouped by movement
│   ├── metadata.csv              # Final dataset metadata
│   ├── train.csv                 # Training split
│   ├── test.csv                  # Held-out test split
│   ├── scrape_wikiart.py         # Active WikiArt scraper
│   ├── build_dataset.py          # Active validator/splitter
│   ├── README.md                 # Data folder guide
│   └── legacy/                   # Archived experimental data scripts
├── models/
│   ├── classifier.joblib
│   ├── label_encoder.joblib
│   ├── config.json
│   ├── embeddings_clip.npz        # Cached CLIP embeddings (written by train.py)
│   ├── arnheim_axes.npz           # Arnheim axis vectors (written by arnheim_analysis.py)
│   ├── confusion_matrix.png
│   └── wolfflin_pca_{model}.png
├── outputs/
│   └── arnheim/
│       ├── arnheim_scores.csv
│       ├── arnheim_radar_overlay.png
│       ├── arnheim_radar_per_genre.png
│       ├── arnheim_violins.png
│       ├── arnheim_wolfflin_heatmap.png
│       ├── arnheim_tsne.png
│       └── arnheim_summary.txt
├── train.py                       # Training pipeline + Wölfflin analysis
├── arnheim_analysis.py            # Arnheim perceptual scoring + visualisations
├── app.py                         # Gradio demo (4 tabs)
├── requirements.txt
└── README.md
```

---

## How to Reproduce

The final dataset is already included in `data/metadata.csv`, `data/train.csv`, `data/test.csv`, and `data/images/`. For normal lab use, start at training:

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train
python train.py              # compare CLIP + ViT × 3 classifiers
python train.py --skip-vit   # CLIP only (faster, also saves embeddings_clip.npz)

# 3. Arnheim analysis (requires trained model + embeddings cache from step 2)
python arnheim_analysis.py           # encodes images + saves axes (~10 min first run)
python arnheim_analysis.py --skip-encode  # fast re-run once cache exists

# 4. Run demo
python app.py
```

To rebuild the dataset from source metadata, use `data/scrape_wikiart.py` and `data/build_dataset.py`. The older Hugging Face downloader and rebalance utilities are archived in `data/legacy/` for provenance only.

---

## Expected Results and Genre Confusion

Based on the Wölfflin mapping, the confusion matrix should show predictable patterns:

- **Renaissance ↔ Baroque**: highest confusion pair — both are Old Masters painting in oil on canvas with similar subjects. The visual difference (linear vs chiaroscuro) is subtle at thumbnail resolution.
- **Impressionism ↔ Expressionism**: both feature visible brushwork and non-classical composition; the distinction is in color naturalism (Impressionism) vs emotional distortion (Expressionism).
- **Abstract ↔ Expressionism**: Abstract Expressionism is literally at their intersection. Pollock could be classified as either.
- **Cubism ↔ Abstract**: both involve geometric non-representational surface — Cubism is earlier and still references objects; Abstract dissolves them entirely.
- **Surrealism**: unique cross-axis position should make it both easy to identify (nothing else looks like melting clocks) and easy to confuse (the hyper-realistic rendering technique looks like Baroque).

The Wölfflin analysis in `train.py` quantifies how well the embedding space captures these theoretical distances.

---

## HuggingFace Hub

Prepared upload cards and helper script live in `hf_artifacts/`.

```bash
hf auth login
bash hf_artifacts/upload_to_hf.sh {username}
```

This creates/uploads:

- Dataset repo: `{username}/art-movement-rebalanced-dataset`
- Model repo: `{username}/art-movement-clip-classifier`
- Space repo: `{username}/art-movement-classifier-demo`
