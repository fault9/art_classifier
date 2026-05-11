# Art Movement Classifier

A CLIP-embedding image classifier for eight art movements, with two interpretability views. The classifier predicts the movement label. Wölfflin is used mainly as a Renaissance-Baroque art-historical reference, while the Arnheim layer compares an individual painting to empirical perceptual profiles built from anchor-based CLIP projections. Arnheim is interpretive: it explains visual similarity on perceptual axes, not a second movement prediction.

## Embeddings Lab Submission Summary

The short academic report for the course submission is in [`REPORT.md`](REPORT.md).

**Project links:**

- GitHub repo: https://github.com/fault9/art_classifier
- Hugging Face dataset: https://huggingface.co/datasets/fault9/art-movement-rebalanced-dataset
- Hugging Face model: https://huggingface.co/fault9/art-movement-clip-classifier
- Hugging Face demo: https://huggingface.co/spaces/fault9/art-movement-classifier-demo

**Domain issue:** Art movement classifiers can learn artist-specific shortcuts instead of movement-level visual structure. In the original data, some artists dominated their class, so the dataset was rebalanced by capping artists and adding underrepresented artists.

**Challenge addressed with embeddings:** The project uses pretrained image embeddings to classify paintings with a small custom dataset, avoiding the need to train a full vision model from scratch.

**Embedding type:** CLIP image embeddings from `openai/clip-vit-base-patch32`.

**Trained model:** scikit-learn MLP classifier trained on frozen CLIP embeddings after comparing Logistic Regression, SVM-RBF, and MLP classifiers.

**Custom dataset:** 1,478 paintings across 8 art movements, with fixed train/test splits and artist-dominance audit files.

**Held-out accuracy:** 0.8093. Best 5-fold CV model: CLIP + MLP at 0.8138 +/- 0.0173.

**Demo:** Gradio app in `app.py` with single-image classification, collection analysis, Wölfflin profiles, and an Arnheim radar that can overlay selected empirical movement profiles for perceptual comparison.

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

Heinrich Wölfflin identified **five pairs of opposing visual principles** that describe the shift from Renaissance to Baroque. This project uses that framework mainly as a Renaissance-Baroque reference point, then cautiously extends the mapping to the other classes for comparison:

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

**Interpretive use:** The Wölfflin view in the demo is theoretical: it shows the Wölfflin profile assigned to the classifier's predicted movement. This is most historically meaningful for Renaissance and Baroque, the two periods Wölfflin's framework was designed to contrast. For later movements, the mapping is a cautious comparison layer rather than a claim that Wölfflin directly described those styles. `train.py` also saves a PCA visualization of CLIP class centroids to inspect whether embedding geometry roughly follows these art-historical contrasts.

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

**Key difference:** Wölfflin assigns a fixed theoretical position to each *genre* (all Baroque paintings get the same score). Arnheim scores *individual paintings* based on their actual position in CLIP embedding space relative to anchor paintings, so two paintings in the same genre can differ significantly. The nearest Arnheim profile should be read as perceptual resemblance, not as an art movement label. For example, a Cubist still life can remain classified as Cubism while sitting close to Surrealism or Pop Art on the Arnheim axes because those classes share flattened space, strong shape structure, and visual tension in this dataset.

### Implementation

Scores are derived by **anchor-based projection**:
1. For each dimension, define HIGH and LOW anchor paintings by artist name or by specific `(artist, title)` pairs
2. Find those paintings in the training set and compute anchor centroids
3. The axis vector points from the LOW centroid to the HIGH centroid in 512-D CLIP space
4. Every painting is projected onto this axis
5. Scores are calibrated with the dataset's 5th/95th percentiles so the UI scale remains roughly interpretable as Low ↔ High without a few extreme images stretching the axis

If an anchor artist isn't in the training set (it's only ~1,400 paintings), their contribution is simply absent — the remaining anchors still define the axis.

### Limitations And Calibration

- Anchor selection involves art-historical judgment and could be contested
- Projection is approximate: CLIP embeddings encode many properties simultaneously
- With a small training set, some anchor groups contain only 5–20 paintings — sparse representation of the perceptual extreme
- Scores are relative to the paintings in this specific dataset, not to art in general

`diagnose_anchors.py` generates `outputs/arnheim/anchor_diagnostics/calibration_report.md`, `dimension_extremes.csv`, and profile-overlap tables. Use these files to tune the anchor pools by checking whether the highest and lowest paintings on each axis actually match the intended perceptual poles. The calibration target is visual plausibility of the axes, not agreement between Arnheim nearest profiles and classifier labels. The current diagnostics flag Tension and Color as weaker axes in CLIP space, so those are the best candidates for future anchor refinement.

---

## Approach

1. **Image embeddings** from `openai/clip-vit-base-patch32`
2. **MLP classifier** trained on frozen CLIP embeddings
3. **Model selection** comparing Logistic Regression, SVM-RBF, and MLP with 5-fold stratified cross-validation
4. **Held-out test evaluation** on the fixed test split
5. **Wölfflin analysis** on CLIP class centroids via PCA
6. **Arnheim analysis** through anchor-based projection in CLIP space, reported as perceptual similarity rather than classification

---

## Dataset

The final dataset contains **1,478 paintings** across eight art movements, with **1,090 training images** and **388 held-out test images**. It was built from WikiArt-derived metadata plus targeted Hugging Face/WikiArt additions.

The dataset was rebalanced to reduce artist-dominance bias: overrepresented artists were capped at 20 paintings per class, and underrepresented artists were added where source data was available. Audit files in `data/` document the cap removals, additions, and copyright-risk review.

Sources:

- `wikiart`: primary WikiArt scrape
- `huggan_wikiart`: additions from the Hugging Face `huggan/wikiart` dataset
- `wikiart_targeted`: targeted WikiArt additions for underrepresented artists

---

## File Structure

```
art_classifier/
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
│   ├── classifier_comparison.csv
│   ├── classifier_comparison.json
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
├── app.py                         # Gradio demo with mode selector and collection analysis
├── requirements.txt
└── README.md
```

---

## How to Reproduce

The GitHub repository includes source code, metadata, model artifacts, and analysis outputs. The full image folder is not committed to GitHub; it is distributed through the Hugging Face dataset because it is large and has mixed copyright status. To retrain locally, download or restore `data/images/` so the paths in `data/metadata.csv` resolve. If you only want to run the demo, use the Hugging Face Space or the committed model files.

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train, after data/images/ is available locally
python train.py              # train/evaluate the CLIP + MLP classifier

# 3. Arnheim analysis
python arnheim_analysis.py           # encodes images if needed and saves axes
python arnheim_analysis.py --skip-encode  # fast re-run once embeddings are cached
python diagnose_anchors.py           # calibration report for anchor quality and profile overlap

# 4. Run demo
python app.py
```

To rebuild the dataset from source metadata, use `data/scrape_wikiart.py` and `data/build_dataset.py`. The older Hugging Face downloader and rebalance utilities are archived in `data/legacy/` for provenance only. The final dataset metadata and audit CSVs are included so the construction process remains inspectable without committing the image binaries to GitHub.

---

## Expected Results and Genre Confusion

The held-out test accuracy is **0.8093**. The training pipeline compares several classifiers on the same CLIP embeddings before selecting the final model:

| Embedding | Classifier | 5-fold CV accuracy | Std. dev. |
|---|---:|---:|---:|
| CLIP | MLP | 0.8138 | 0.0173 |
| CLIP | SVM-RBF | 0.8101 | 0.0125 |
| CLIP | Logistic Regression | 0.7606 | 0.0128 |

Common confusions are expected where movements share subjects or visual structure:

- **Renaissance ↔ Baroque**: highest confusion pair — both are Old Masters painting in oil on canvas with similar subjects. The visual difference (linear vs chiaroscuro) is subtle at thumbnail resolution.
- **Impressionism ↔ Expressionism**: both feature visible brushwork and non-classical composition; the distinction is in color naturalism (Impressionism) vs emotional distortion (Expressionism).
- **Abstract ↔ Expressionism**: Abstract Expressionism is literally at their intersection. Pollock could be classified as either.
- **Cubism ↔ Abstract**: both involve geometric non-representational surface — Cubism is earlier and still references objects; Abstract dissolves them entirely.
- **Surrealism**: unique cross-axis position should make it both easy to identify (nothing else looks like melting clocks) and easy to confuse (the hyper-realistic rendering technique looks like Baroque).

The saved confusion matrix and Wölfflin/Arnheim visualizations are in `models/` and `outputs/arnheim/`. In the demo, the Arnheim radar always includes the classifier movement as the reference and can optionally overlay other selected movement profiles. These matches are useful for interpreting cross-movement visual overlap; they are not counted as classifier accuracy.

---

## Hugging Face Hub

Prepared upload cards and helper script live in `hf_artifacts/`.

```bash
hf auth login
bash hf_artifacts/upload_to_hf.sh {username}
```

This creates/uploads:

- Dataset repo: `{username}/art-movement-rebalanced-dataset`
- Model repo: `{username}/art-movement-clip-classifier`
- Space repo: `{username}/art-movement-classifier-demo`
