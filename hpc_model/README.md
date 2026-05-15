# Model Hippocampus

This directory contains the dual modern Hopfield network hippocampus checks
shown in Supplementary Figure S2. The pipeline uses ROC story stimuli, xRAG
embeddings, generated detail strings, and cached evaluation sweeps to test
selection and recall behaviour.

Outputs are written under `hpc_model/outputs/`. The figure is written to
`figures/Figure S2.pdf`, with source-data CSVs in `source_data/`.

## Reproduce Results From Cached Data

Reuse existing xRAG embedding/detail caches and result JSON files:

```bash
python -m hpc_model.run_all
```

This reads from:

- `hpc_model/outputs/cache/`
- `hpc_model/outputs/results/`

and refreshes `figures/Figure S2.pdf` plus `source_data/Figure_S2_*.csv`.

## Reproduce Results From Scratch

Use both force flags to bypass cached embeddings/details/distances and cached
result sweeps:

```bash
python -m hpc_model.run_all --force_cache --force_results
```

`--force_cache` rebuilds xRAG embedding, generated-detail, and distance-matrix
caches. `--force_results` recomputes the baseline, decay sweep, and beta sweep
JSON results.

## Useful Options

```bash
# Regenerate plots/results without re-embedding xRAG
python -m hpc_model.run_all --force_results

# Use a different output root
python -m hpc_model.run_all --out_root hpc_model/outputs

# Adjust the number of Raykov typical stimuli
python -m hpc_model.run_all --n 100
```

Key scripts:

- `run_all.py`: cached end-to-end pipeline and Figure S2 generation.
- `cache_xrag.py`: builds or loads document/query embedding caches.
- `cache_details.py`: builds or loads generated detail strings.
- `model_eval.py`: baseline and sweep evaluations.
- `plotting.py`: three-panel Supplementary Figure S2 plotting and source-data
  export.
