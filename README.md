## Hippocampo-neocortical interaction as compressive retrieval-augmented generation

Code for the paper "Hippocampo-neocortical
interaction as compressive retrieval-augmented generation".

The repository contains four main simulation pipelines:

- `full_model/`: narrative encoding, consolidation, forgetting, semantic memory
  analysis, and LLM-rated memory-content analyses.
- `full_model/narratives/`: Bartlett/Raykov prior-knowledge narrative
  simulations.
- `inference/`: spatial and family-tree relational inference, learned
  representation analyses, and RAG-composition experiments.
- `hpc_model/`: modern Hopfield network hippocampus simulations.

Generated figure PDFs are collected in `figures/`. Source-data CSVs for plotted
values are written to `source_data/`.

See `FIGURES.md` for instructions for reproducing the results.

> [!NOTE]
> This repo reproduces the results in the paper. See this repo for tools to use in other experiments: [llm-psychology](https://github.com/ellie-as/llm-psychology).

#### Installation

```bash
git clone https://github.com/ellie-as/hippocampal-neocortical-RAG.git
cd hippocampal-neocortical-RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The code was tested with Python 3.11.5. Training and full regeneration of the
Mistral-7B/LoRA narrative results requires GPU hardware; the full model
training runs were run on a single A100. (The inference models are GPT-2 based
so can run on more modest GPUs.)

#### Repository Layout

- `data/`: input CSV/text data used by the simulations.
- `figures/`: generated figure PDFs and Figure 6 schematic PNG inputs.
- `full_model/`: full narrative memory simulation and plotting.
- `full_model/narratives/`: Bartlett and Raykov narrative simulations and
  plotting.
- `full_model/xRAG/`: local copy of the xRAG code adapted from Cheng et al.
  (2024), kept here for reproducibility.
- `hpc_model/`: dual modern Hopfield network hippocampus pipeline.
- `inference/`: relational inference, representation, and RAG-composition
  pipeline.
- `scripts/`: shared helpers, source-data export utilities, tokenizer helpers,
  model wrappers, and training scripts adapted from Hugging Face examples.
- `source_data/`: journal source-data CSV exports written by figure scripts.
- `wordcloud_outputs/`: exploratory word-cloud sweeps used while tuning the
  word-cloud panel.

Saved LLM outputs can be inspected without retraining, for example with
`full_model/Inspect saved narratives.ipynb`.
