# Full Model

This directory contains the full hippocampal-neocortical narrative memory
simulation. It models episodic story encoding, consolidation into a LoRA-tuned
neocortical model, semantic memory, and forgetting after further training.
The plotting pipeline generates `figures/Figure 3.pdf` and
`figures/Figure 4.pdf`.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The full simulation uses Mistral-7B with LoRA and was run on a single A100.

## Reproduce Results From Cached Data

If `full_model/output/data/` already contains the saved `.pkl` and CSV artifacts,
regenerate Figures 3 and 4 without retraining:

```bash
cd full_model
python collate_all_figures.py
```

Outputs:

- `figures/Figure 3.pdf`
- `figures/Figure 4.pdf`
- `source_data/Figure_3_*.csv`
- `source_data/Figure_4_*.csv`

## Reproduce Results From Scratch

Use `--force` to clear this pipeline's cached data and saved models before
rerunning. Without `--force`, `memory_simulation.py` may reuse cached encoding,
consolidation, forgetting, or model artifacts.

```bash
cd full_model
python memory_simulation.py --force
python collate_all_figures.py
```

Figure 4 uses GPT-rated memory attributes stored in:

- `output/data/story_llm_ratings_simulated.csv`
- `output/data/forgetting_llm_ratings.csv`

To regenerate those ratings rather than reuse the cached CSVs:

```bash
export OPENAI_API_KEY=...
python rate_simulated_llm_attributes.py --force
python collate_all_figures.py
```

## Main Scripts

### `memory_simulation.py`

Runs the core simulation:

1. Encodes ROC stories at multiple compression/detail levels.
2. Fine-tunes a LoRA model on selected encoded stories.
3. Measures episodic recall and semantic Q&A accuracy across consolidation
   epochs.
4. Continues training on new stories and measures forgetting of the original
   stories.

Useful options:

```bash
python memory_simulation.py --num_stories 500 --num_epochs 10
python memory_simulation.py --consolidation_encoding 0
python memory_simulation.py --output_dir output
python memory_simulation.py --force
```

### `collate_all_figures.py`

Reads saved simulation outputs from `output/data/`, plots Figures 3 and 4, and
exports source-data CSVs to the repository-level `source_data/` directory.

### `semantic_memory_standalone.py`

Runs semantic-memory analysis using saved models from a previous
`memory_simulation.py` run.

```bash
python semantic_memory_standalone.py --output_dir output
python semantic_memory_standalone.py --num_stories 50
```

### `rate_simulated_llm_attributes.py`

Regenerates GPT ratings of simulated model memories used in Figure 4.

```bash
export OPENAI_API_KEY=...
python rate_simulated_llm_attributes.py --force
python rate_simulated_llm_attributes.py --dummy
```

## Subdirectories

- `narratives/`: Bartlett and Raykov prior-knowledge narrative simulations for
  Figure 5. See `narratives/README.md`.
- `xRAG/`: local copy of the xRAG implementation adapted from Cheng et al.
  (2024). See `xRAG/README.md`.
- `LLM ratings/`: notebooks/scripts and cached ratings used for Figure 4 human
  and model memory-attribute comparisons.
- `data/`: full-model input data, including HIPPOCORPUS/NFRD comparison data and
  Bartlett text used by the narrative pipeline.
