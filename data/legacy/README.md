# Legacy Data Scripts

These scripts are kept for provenance, but they are not the normal path for using the final lab dataset.

## Files

- `download_hf_datasets.py`: older broad downloader for `huggan/wikiart`
- `rebalance_artist_dominance.py`: post-processing script used to cap dominant artists and add targeted artists
- `cap_artists.py`: earlier one-off artist-cap utility

## Why They Are Archived

The final dataset has already been built and is stored in:

```text
data/metadata.csv
data/train.csv
data/test.csv
data/images/
```

New users should not run these legacy scripts unless they intentionally want to recreate or modify the dataset construction process. Running them can change the metadata and splits.
