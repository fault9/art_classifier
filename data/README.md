# Data Folder

This folder contains the final dataset used for the embeddings lab.

## Final Dataset Files

- `metadata.csv`: full final dataset metadata
- `train.csv`: training split
- `test.csv`: held-out test split
- `images/`: painting images grouped by art movement
- `wikiart_metadata.csv`: primary WikiArt scrape used as the base source
- `exclusion_log.csv`: filtered non-painting or invalid works
- `artist_distribution_before_rebalance.csv`: artist distribution audit before final rebalance
- `artist_distribution_after_rebalance.csv`: artist distribution audit after final rebalance
- `artist_rebalance_added.csv`: paintings added during artist rebalance
- `artist_cap_removed.csv`: paintings removed by artist cap
- `copyright_risk_audit.csv`: heuristic copyright-risk bucket for each metadata row

## Active Pipeline

For normal use, do not rebuild the dataset. The final dataset is already in:

```text
metadata.csv
train.csv
test.csv
images/
```

To train and run the lab:

```bash
python train.py
python arnheim_analysis.py --skip-encode
python app.py
```

## Scripts

- `scrape_wikiart.py`: active WikiArt scraper used to collect the primary source data
- `build_dataset.py`: active validator/splitter for source metadata
- `legacy/`: older experimental/rebalancing scripts kept for provenance only

The scripts in `legacy/` are not part of the normal reproduction path. They document how the final artist-dominance fixes were explored.

## Copyright Risk Audit

`copyright_risk_audit.csv` adds a heuristic risk bucket to each row:

- `lower_by_date_before_1930`: dated before 1930
- `high_date_1930_or_later`: dated 1930 or later
- `medium_unknown_year_modern_class`: missing year in a modern class
- `unknown_year_check_source`: missing year in an older class

This is not a legal determination. It is an audit aid for coursework/research transparency.
