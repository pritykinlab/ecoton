Demo pipeline for ecoton
=======================

This folder contains `run_demo_pipeline.py`, a small smoke-test that runs the
package pipeline end-to-end on synthetic data:

- compute metatranscripts (DBSCAN per gene)
- analytic null over metatranscripts
- convert stats to an igraph and run archetypal analysis (modules)
- compute niche maps for archetypes
- compute simple niche statistics and plot modules

Run locally:

```bash
cd /Genomics/pritykinlab/tamjeed/github_packages/ecoton
python -m pip install -e .  # ensure dependencies installed
python scripts/run_demo_pipeline.py
```

If you want to run on your own transcript table, edit the synthetic data block
in `run_demo_pipeline.py` and replace it with a `pd.read_parquet(...)` or
similar data-loading call producing a DataFrame with columns:

- `feature_name`, `x_location`, `y_location`, `qv`, `codeword_category`
