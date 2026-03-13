# Ecoton
Expression colocalization of transcripts for niche discovery

## CLI workflow

Ecoton now includes an integrated command-line workflow runner.

### Run as module

```bash
python -m ecoton \
	--transcripts-path ../spatial_5k/data/transcripts.parquet \
	--transcripts-format parquet \
	--output-dir processed_data
```

### Run as installed command

After installing/updating the package (for example `pip install -e .`), run:

```bash
ecoton \
	--transcripts-path ../spatial_5k/data/transcripts.parquet \
	--transcripts-format parquet \
	--output-dir processed_data
```

### Useful options

- `--mode` (default: `xenium_5k`)
- `--k` (default: `25`)
- `--seed` (default: `1`)
- `--organism` (default: `Human`)
- `--min-points` (default: `3`)
- `--bin-size` (default: `8.0`)
- `--smoothing-radius` (default: `8.0`)
- `--weight-threshold` (default: `0.3`)
- `--save-module-plot` to save selected module visualization

### Outputs

The workflow writes these files into `--output-dir`:

- `workflow.pkl`
- `runtime_tracking.pkl`
- `selected_archetypal_modules.png` (when `--save-module-plot` is set)
