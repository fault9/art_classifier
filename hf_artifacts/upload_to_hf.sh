#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash hf_artifacts/upload_to_hf.sh YOUR_HF_USERNAME"
  exit 1
fi

USER_NAME="$1"

DATASET_REPO="${USER_NAME}/art-movement-rebalanced-dataset"
MODEL_REPO="${USER_NAME}/art-movement-clip-classifier"
SPACE_REPO="${USER_NAME}/art-movement-classifier-demo"

echo "Creating repos if needed..."
hf repo create "$DATASET_REPO" --repo-type dataset --exist-ok
hf repo create "$MODEL_REPO" --repo-type model --exist-ok
hf repo create "$SPACE_REPO" --repo-type space --space-sdk gradio --exist-ok

echo "Uploading dataset..."
hf upload "$DATASET_REPO" hf_artifacts/dataset/README.md README.md --repo-type dataset
hf upload "$DATASET_REPO" data/images images --repo-type dataset
hf upload "$DATASET_REPO" data/metadata.csv metadata.csv --repo-type dataset
hf upload "$DATASET_REPO" data/train.csv train.csv --repo-type dataset
hf upload "$DATASET_REPO" data/test.csv test.csv --repo-type dataset
hf upload "$DATASET_REPO" data/artist_distribution_before_rebalance.csv artist_distribution_before_rebalance.csv --repo-type dataset
hf upload "$DATASET_REPO" data/artist_distribution_after_rebalance.csv artist_distribution_after_rebalance.csv --repo-type dataset
hf upload "$DATASET_REPO" data/artist_rebalance_added.csv artist_rebalance_added.csv --repo-type dataset
hf upload "$DATASET_REPO" data/artist_cap_removed.csv artist_cap_removed.csv --repo-type dataset
if [ -f data/exclusion_log.csv ]; then
  hf upload "$DATASET_REPO" data/exclusion_log.csv exclusion_log.csv --repo-type dataset
fi
if [ -f data/copyright_risk_audit.csv ]; then
  hf upload "$DATASET_REPO" data/copyright_risk_audit.csv copyright_risk_audit.csv --repo-type dataset
fi

echo "Uploading model..."
hf upload "$MODEL_REPO" hf_artifacts/model/README.md README.md --repo-type model
hf upload "$MODEL_REPO" models/classifier.joblib classifier.joblib --repo-type model
hf upload "$MODEL_REPO" models/label_encoder.joblib label_encoder.joblib --repo-type model
hf upload "$MODEL_REPO" models/config.json config.json --repo-type model
hf upload "$MODEL_REPO" models/evaluation_metrics.json evaluation_metrics.json --repo-type model
hf upload "$MODEL_REPO" models/confusion_matrix.png confusion_matrix.png --repo-type model
hf upload "$MODEL_REPO" models/wolfflin_pca_clip.png wolfflin_pca_clip.png --repo-type model
hf upload "$MODEL_REPO" models/arnheim_axes.npz arnheim_axes.npz --repo-type model

echo "Uploading Space..."
hf upload "$SPACE_REPO" hf_artifacts/space/README.md README.md --repo-type space
hf upload "$SPACE_REPO" app.py app.py --repo-type space
hf upload "$SPACE_REPO" hf_artifacts/space/requirements.txt requirements.txt --repo-type space
hf upload "$SPACE_REPO" examples examples --repo-type space
hf upload "$SPACE_REPO" models/classifier.joblib models/classifier.joblib --repo-type space
hf upload "$SPACE_REPO" models/label_encoder.joblib models/label_encoder.joblib --repo-type space
hf upload "$SPACE_REPO" models/config.json models/config.json --repo-type space
hf upload "$SPACE_REPO" models/arnheim_axes.npz models/arnheim_axes.npz --repo-type space

echo ""
echo "Done."
echo "Dataset: https://huggingface.co/datasets/${DATASET_REPO}"
echo "Model:   https://huggingface.co/${MODEL_REPO}"
echo "Space:   https://huggingface.co/spaces/${SPACE_REPO}"
