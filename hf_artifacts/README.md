# Hugging Face Submission Package

This folder contains the cards and helper script for publishing the lab deliverables to Hugging Face.

The project should be published as three Hub repos:

1. Dataset repo: custom rebalanced art movement dataset
2. Model repo: trained CLIP-embedding classifier
3. Space repo: Gradio demo

Recommended repo names:

```text
art-movement-rebalanced-dataset
art-movement-clip-classifier
art-movement-classifier-demo
```

Before uploading, make sure these local files exist:

```text
data/images/
data/metadata.csv
data/train.csv
data/test.csv
models/classifier.joblib
models/label_encoder.joblib
models/config.json
models/evaluation_metrics.json
models/arnheim_axes.npz
app.py
requirements.txt
```

The upload script intentionally uploads the final dataset and trained model artifacts, not the legacy data-construction scripts.

Then run:

```bash
bash hf_artifacts/upload_to_hf.sh YOUR_HF_USERNAME
```

If you are not logged in:

```bash
hf auth login
```
