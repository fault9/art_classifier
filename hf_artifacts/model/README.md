---
license: cc-by-nc-4.0
library_name: scikit-learn
pipeline_tag: image-classification
tags:
- art
- art-history
- clip
- embeddings
- scikit-learn
- image-classification
---

# Art Movement Classifier with CLIP Embeddings

This model classifies paintings into eight art movements using frozen CLIP image embeddings and a trained scikit-learn classifier.

## Lab Framing

**Issue:** Art movement classification is difficult because visual movements overlap in subject matter, period, materials, and artist-specific style.

**Challenge addressed with embeddings:** Instead of training a vision model from scratch on a small dataset, the project uses pretrained CLIP image embeddings as a compact visual representation, then trains a domain-specific classifier on top of those embeddings.

**Embedding type:** `openai/clip-vit-base-patch32` image embeddings.

**Classifier:** MLP classifier trained on CLIP embeddings.

## Classes

- Renaissance
- Baroque
- Impressionism
- Expressionism
- Cubism
- Abstract
- Surrealism
- Pop Art

## Results

Held-out test accuracy: **0.8093**

Cross-validation accuracy: **0.8138 ± 0.0173**

Per-class F1:

| Class | F1 |
|---|---:|
| Abstract | 0.727 |
| Baroque | 0.860 |
| Cubism | 0.774 |
| Expressionism | 0.758 |
| Impressionism | 0.906 |
| Pop Art | 0.761 |
| Renaissance | 0.904 |
| Surrealism | 0.684 |

## Files

- `classifier.joblib`: trained scikit-learn classifier
- `label_encoder.joblib`: label encoder for class ids
- `config.json`: embedding/model configuration
- `evaluation_metrics.json`: held-out and cross-validation metrics
- `confusion_matrix.png`: normalized confusion matrix
- `wolfflin_pca_clip.png`: Wölfflin-inspired PCA analysis
- `arnheim_axes.npz`: Arnheim perceptual axis vectors used by the demo

## Usage

The model is intended to be used through the accompanying Gradio Space. The app loads CLIP, embeds an uploaded image, and applies the trained classifier.

## Limitations

- The model can confuse visually similar old-master religious paintings, especially Renaissance vs Baroque.
- Predictions reflect visual similarity in the training distribution, not definitive art-historical attribution.
- The Wölfflin tab in the demo is a theoretical profile of the predicted class, not a direct pixel-level measurement.
- Arnheim perceptual scores are anchor-based projections in CLIP space and should be interpreted as exploratory.
