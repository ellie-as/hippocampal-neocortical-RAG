# Inference (relational inference + RAG composition)

This folder contains the code for the **relational inference** experiments (spatial grid + family tree) and the **RAG composition** experiment.

## Directory layout

```
inference/
├── graph_sequence_model.py      # Train spatial & family models
├── generate_figures.py          # One-command Figure 6/Figure 7 generation
├── collate_inf_figures.py       # Assemble Figure 6.pdf & Figure 7.pdf
├── export_figure6_panels.py     # Export Figure 6g/h/i/k source-data tables
├── build_figure6_full.py        # Add diagrams/loss panels to full Figure 6
├── inference_config.json        # Runtime paths for trained models/cache/output
├── run_config.py                # Shared config loader
├── plot_reps.py                 # Spatial representation geometry (used by collate)
├── plot_family_reps.py          # Family representation geometry (used by collate)
├── rag_composition.py           # RAG composition simulation
├── outputs_graph/               # Trained spatial model (written by training)
├── outputs_tree/                # Trained family model (written by training)
├── _cache/                      # Cached computation results
├── data/                        # Inference caches and local Figure 6 panel tables
├── ../figures/Figure 6.pdf      # Output: Figure 6, inference behaviour
├── ../figures/Figure 7.pdf      # Output: Figure 7, internal representations
```

## Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `transformers`, `torch`, `matplotlib`, `numpy`, `scipy`, `scikit-learn`, `umap-learn`.

## Runtime config

Edit `inference/inference_config.json` to point the plotting/evaluation scripts
at trained model directories. Relative paths are resolved from the repository
root; absolute paths are also supported.

Example for an external trained-model directory:

```json
{
  "spatial_model_dir": "/path/to/final_rel_inf/outputs_graph",
  "family_model_dir": "/path/to/final_rel_inf/outputs_tree",
  "cache_dir": "inference/data/rel_inf_cache",
  "figures_dir": "figures",
  "data_dir": "inference/data",
  "seed": 321
}
```

All model-generated caches for the inference figures are written to
`inference/data/rel_inf_cache/` by default.

## Reproduce Results From Cached Data

If `inference/outputs_graph/`, `inference/outputs_tree/`, and
`inference/data/rel_inf_cache/` already exist, regenerate Figures 6 and 7 while
reusing cached evaluations:

```bash
python inference/generate_figures.py --config inference/inference_config.json
```

This refreshes the PDFs and source-data exports from the configured trained
models and cached JSON results.

## Reproduce Results From Scratch

For a scratch run, clear or rename the trained model directories first:

- `inference/outputs_graph/`
- `inference/outputs_tree/`

Then train the models and clear generated figure caches during figure
generation:

```bash
cd inference
python graph_sequence_model.py
cd ..
python inference/generate_figures.py \
  --config inference/inference_config.json \
  --clear-cache
```

The `--clear-cache` flag removes and recomputes the JSON caches under
`inference/data/rel_inf_cache/`. It does not retrain models; model retraining is
done by `graph_sequence_model.py`.

## Step 1: Train the models

```bash
cd inference
python graph_sequence_model.py
```

This creates `outputs_graph/` (spatial model) and `outputs_tree/` (family model).  Each contains a fine-tuned GPT-2 causal LM plus a custom tokenizer with two-letter entity tokens.

What it does:
- Generates random-walk training data from 100k graphs/trees (20 walks each, max length 50)
- Trains via Hugging Face Trainer by calling `../scripts/run_clm.py`
- Evaluates loop-closure inference (2/4/6/8-hop) and grid generalisation
- Runs imagination (open-ended generation at various temperatures)

Options:

```bash
python graph_sequence_model.py --reuse-models   # skip training, re-run eval only
python graph_sequence_model.py --smoke           # small datasets for a quick test
python graph_sequence_model.py --seed 42         # set random seed
```

## Step 2: Generate Figure 6 and Figure 7

Once the model folders exist:

```bash
python inference/generate_figures.py \
  --config inference/inference_config.json \
  --clear-cache
```

This computes the model-generated caches and assembles publication-quality PDFs:

- `figures/Figure 6.pdf` — full Figure 6, including raster schematics from `figures/diagrams/`, loss panels, and inference panels
- `figures/Figure 7.pdf` — internal representations (PCA of spatial/family embeddings, boxplots, correlation vs layer/context length)

Options:

```bash
python inference/generate_figures.py --config inference/inference_config.json
python inference/generate_figures.py --config inference/inference_config.json --clear-cache
python inference/generate_figures.py --config inference/inference_config.json --smoke
python inference/generate_figures.py --config inference/inference_config.json --skip-panel-export
```

Results are cached under `inference/data/rel_inf_cache/` by default, so
re-running is fast after the first computation.

The wrapper runs the lower-level scripts in this order:

1. `collate_inf_figures.py` to generate Figure 7 and the non-RAG Figure 6 caches.
2. `rag_composition.py` to generate `rag_composition.json` for Figure 6k.
3. `collate_inf_figures.py --skip-fig6` to refresh Figure 6 with panel k included.
4. `export_figure6_panels.py` to write Figure 6g/h/i/k CSV tables.
5. `build_figure6_full.py` to embed the schematic PNGs from `figures/diagrams/`, assemble the full Figure 6 layout, and export loss CSVs from `trainer_state.json`.

To export the Figure 6g/h/i/k source-data tables:

```bash
python inference/export_figure6_panels.py \
  --config inference/inference_config.json
```

This command requires generated caches:

- `inference/data/rel_inf_cache/aggregated_inf.json`
- `inference/data/rel_inf_cache/grid_generalisation.json`
- `inference/data/rel_inf_cache/imagination.json`
- `inference/data/rel_inf_cache/rag_composition.json`

The local panel/input tables are written under `inference/data/` as
`Figure_6g_aggregated_inference.csv`, `Figure_6h_grid_generalisation.csv`,
`Figure_6i_imagination_validity.csv`, and
`Figure_6k_rag_composition_summary.csv`. Copies for journal upload are also
written under the repository-level `source_data/` directory.

## Server run for reproducible inference data

On the A100 machine, from the repository root:

```bash
# 1. Edit inference/inference_config.json first.
#    Set spatial_model_dir and family_model_dir to the trained model paths.

python inference/generate_figures.py \
  --config inference/inference_config.json \
  --clear-cache
```

This computes and saves all generated caches for Figures 6 and 7, refreshes
`figures/Figure 6.pdf` after panel k data exists, exports separate vector
PDFs for panels 6g, 6h, 6i, and 6k with CSV tables derived from the caches,
then rebuilds the full Figure 6 layout with diagrams and loss panels.
Figure 7 source-data CSVs are also written to `source_data/`.

The representation-geometry caches in `inference/data/rel_inf_cache/` store the
PCA coordinates used for Figure 7 (`pc1`, `pc2`) plus plotting metadata, not the
full hidden-state vectors.

## Step 3: RAG composition experiment

This tests whether the model can **combine information from multiple retrieved memories** to solve a task that no single memory can solve alone.

Each N-hop closed walk is split into two memory fragments sharing a bridge entity:

```
Full sequence:   ab EAST cd SOUTH ef WEST gh NORTH ab
                 ├── mem1 ──────┤├── mem2 ─┤├ query ┤
Memory 1:        ab EAST cd SOUTH ef
Memory 2:                        ef WEST gh
Query:                                      gh NORTH → predict "ab"
```

The target (`ab`) is only in mem1; the query entity (`gh`) is only in mem2. Correct prediction requires composing both memories via the bridge (`ef`).

### Running

```bash
cd inference

# Full run (100 trials per template, ~30 min on CPU)
python rag_composition.py --n-per-template 100

# Quick smoke test
python rag_composition.py --n-per-template 10

# Load cached results and print the summary table only
python rag_composition.py --plot-only

# Force re-evaluation (ignore cache)
python rag_composition.py --clear-cache --n-per-template 100
```

### Outputs

- `inference/data/rel_inf_cache/rag_composition.json` — model-generated cache used by Figure 6k
- panel k in `figures/Figure 6.pdf` — paired bar chart (Spatial vs Family) comparing NC only, HPC only, RAG (single), RAG (multi)

Results are cached to `inference/data/rel_inf_cache/rag_composition.json` by default.

### Conditions evaluated

| Condition | Prompt format | What it tests |
|-----------|---------------|---------------|
| Full | Complete walk on one line | Upper bound — single-line replay |
| RAG (3-line) | `mem1\nmem2\nquery` | Two memories + query on separate lines |
| **RAG (2-line)** | `mem1\nmem2+query` | Mem2 merged with query onto one line |
| RAG-walk | Padded memories + query | Memories padded with reverse traversals |
| Mem-1 only | `mem1\nquery` | Single memory (first half) + query |
| Mem-2 only | `mem2\nquery` | Single memory (second half) + query |
| NC only | `query` | No context — neocortex-only baseline |
| HPC only | Random entity from mem1 ∪ mem2 | Random guess from stored entities |
