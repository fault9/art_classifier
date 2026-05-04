---
title: Art Movement Classifier
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.14.0
python_version: 3.11
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# Art Movement Classifier

This Gradio Space is the working demo for an embeddings lab project. Upload a painting or paste an image URL to classify it into one of eight art movements using CLIP image embeddings and a trained MLP classifier.

## What The Demo Shows

- Painting classifier with confidence scores
- Collection analyzer for multiple images
- Wölfflin theoretical profile for the predicted movement, mainly meaningful as a Renaissance-Baroque reference
- Arnheim-inspired perceptual profile using CLIP anchor axes, with selectable movement overlays shown as perceptual similarity rather than a second prediction

## Project Summary

The project addresses artist-dominance bias in art movement classification. The custom dataset caps overrepresented artists and adds underrepresented artists so the classifier learns broader movement-level visual features rather than only artist-specific shortcuts.

Embedding model: `openai/clip-vit-base-patch32`

Classifier: scikit-learn MLP on frozen CLIP image embeddings

Held-out accuracy: `0.8093`

Classes:

- Renaissance
- Baroque
- Impressionism
- Expressionism
- Cubism
- Abstract
- Surrealism
- Pop Art

## Notes

Predictions are exploratory and educational. The app is not an art authentication system, and difficult borderline paintings can be misclassified when their visual structure resembles another movement. Arnheim profile matches are interpretive perceptual comparisons, not classifier outputs.
