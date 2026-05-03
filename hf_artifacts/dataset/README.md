---
license: cc-by-nc-4.0
task_categories:
- image-classification
language:
- en
tags:
- art
- art-history
- image-classification
- clip
- wikiart
- embeddings
pretty_name: Rebalanced Art Movement Dataset
size_categories:
- 1K<n<10K
---

# Rebalanced Art Movement Dataset

This custom dataset supports an embeddings lab project on art movement classification. It contains 1,478 painting images across eight art movements, with metadata and fixed train/test splits.

## Project Motivation

The domain problem is that art movement classifiers can learn shortcuts from dominant artists instead of learning broader movement-level visual structure. In the original collection, some artists represented a large share of their class. The dataset was rebalanced by capping any single artist at 20 paintings per movement and adding underrepresented artists from WikiArt/Hugging Face sources where possible.

## Classes

- Renaissance
- Baroque
- Impressionism
- Expressionism
- Cubism
- Abstract
- Surrealism
- Pop Art

## Dataset Size

- Total images: 1,478
- Train split: 1,090
- Test split: 388

## Sources

- `wikiart`: 1,097 images scraped from WikiArt metadata
- `huggan_wikiart`: 285 images from the Hugging Face `huggan/wikiart` dataset
- `wikiart_targeted`: 96 targeted images downloaded from WikiArt to improve artist diversity

## Files

- `images/`: painting images grouped by class
- `metadata.csv`: full metadata table
- `train.csv`: train split
- `test.csv`: held-out test split
- `artist_distribution_before_rebalance.csv`: artist distribution before the final rebalance pass
- `artist_distribution_after_rebalance.csv`: artist distribution after capping and additions
- `artist_rebalance_added.csv`: audit of newly added images
- `artist_cap_removed.csv`: audit of capped/removed rows
- `exclusion_log.csv`: invalid/non-painting/filtered image audit log, when available
- `copyright_risk_audit.csv`: heuristic copyright-risk bucket for each row

Metadata columns:

```text
file_path, label, source, artist, title, year, source_url, wikiart_style, split
```

## Intended Use

This dataset is intended for educational image-classification experiments using pretrained image embeddings, especially CLIP embeddings. It is not intended as an authoritative art-historical catalog.

## Copyright And Use Notes

- Source metadata can contain errors or inconsistent titles.
- The painting-only filter is heuristic.
- Some desired artists were not available from the accessible endpoints.
- The dataset is relatively small, so class and artist coverage remain imperfect.
- Copyright status varies by artwork and image source. WikiArt includes both public-domain and copyright-protected artworks. This dataset is intended for educational coursework and non-commercial experimentation.
- `copyright_risk_audit.csv` provides a heuristic bucket based on available year/class metadata. It is not a legal determination.
