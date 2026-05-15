# Reproducing the Figures

This file explains how to reproduce the figures in the `figures/` directory. It maps each PDF to the code and cached data that produce or support it. 

Current artifacts in `figures/` are `Figure 1.pdf` through `Figure 7.pdf` and
`Figure S1.pdf` through `Figure S3.pdf`. (Figures 1, 2, and S1 are schematics created outside of the simulation scripts.)

## Reproduce Results From Cached Data

Use these commands from the repository root when the cached outputs and trained
model directories already exist. These commands regenerate the plotted PDFs and
source-data tables while reusing cached simulation/model results where the
scripts support it.

```bash
# Figures 3 and 4
cd full_model
python collate_all_figures.py
cd ..

# Figure 5
cd full_model/narratives
python collate_figures.py
cd ../..

# Figures 6 and 7, plus Figure S3 cache/source data
python inference/generate_figures.py --config inference/inference_config.json

# Supplementary Figure S2
python -m hpc_model.run_all
```

Cache locations used by these commands:

- Figures 3/4: `full_model/output/data/`
- Figure 5: `full_model/narratives/output_raykov_xrag_fixed_firstline_50/` and
  `full_model/narratives/output_twostage/`
- Figures 6/7/S3:
  `inference/outputs_graph/`, `inference/outputs_tree/`, and
  `inference/data/rel_inf_cache/`
- Figure S2: `hpc_model/outputs/cache/` and `hpc_model/outputs/results/`

## Reproduce Results From Scratch

To reproduce results from scratch, clear the relevant cached data first. 

```bash
# Figures 3 and 4.
# --force removes full_model/output cached data and saved models.
cd full_model
python memory_simulation.py --force

# Optional for a full refresh of Figure 4 GPT ratings:
# export OPENAI_API_KEY=...
# python rate_simulated_llm_attributes.py --force

python collate_all_figures.py
cd ..

# Figure 5:
# First clear or rename these cached output directories:
#   full_model/narratives/output_raykov_xrag_fixed_firstline_50/
#   full_model/narratives/output_twostage/
#   full_model/narratives/bartlett_encoding_vs_consolidation/
# Then rerun the simulations and collate with cache overwrite.
cd full_model/narratives
python run_all.py
python collate_figures.py --overwrite
cd ../..

# Figures 6, 7, and S3:
# First clear or rename the trained model directories if retraining from scratch:
#   inference/outputs_graph/
#   inference/outputs_tree/
# Then train and regenerate caches/figures.
cd inference
python graph_sequence_model.py
cd ..
python inference/generate_figures.py \
  --config inference/inference_config.json \
  --clear-cache

# Supplementary Figure S2:
# --force_cache rebuilds xRAG embedding/detail/distance caches;
# --force_results recomputes result JSON files.
python -m hpc_model.run_all --force_cache --force_results
```

Figure 4 depends on GPT ratings stored in
`full_model/output/data/story_llm_ratings_simulated.csv` and
`full_model/output/data/forgetting_llm_ratings.csv`. If those CSVs are left in
place, `collate_all_figures.py` reuses them. To regenerate them, set
`OPENAI_API_KEY` and run `rate_simulated_llm_attributes.py --force` after
`memory_simulation.py` has produced the recalled stories.
