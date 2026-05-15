# Narratives: Bartlett And Raykov Experiments

This directory implements the prior-knowledge narrative simulations for
`figures/Figure 5.pdf`. The pipeline combines:

- Raykov-style pre/post-consolidation recall using xRAG.
- Bartlett two-stage training, where topic-specific LoRA models first learn a
  background topic corpus and then learn Bartlett's "War of the Ghosts".
- Encoding-vs-consolidation comparisons and checkpoint analyses used for the
  PCA, cosine-distance, new-words, word-cloud, and omissions/extensions panels.

## Reproduce Results From Cached Data

If the output folders already exist, regenerate Figure 5 and source-data CSVs
without rerunning training or sampling where caches are valid:

```bash
cd full_model/narratives
python collate_figures.py
```

Outputs:

- `figures/Figure 5.pdf`
- `source_data/Figure_5_*.csv`

Cached inputs are read mainly from:

- `output_raykov_xrag_fixed_firstline_50/`
- `output_twostage/`
- `output_twostage/_analysis/`
- `output_twostage/_ckpt_cache/`
- `bartlett_encoding_vs_consolidation/`

## Reproduce Results From Scratch

Clear or rename the cached output directories before rerunning from scratch.
This is required because `run_all.py` skips stages when their outputs already
exist, and `collate_figures.py` reuses cached CSVs and generation samples unless
`--overwrite` is passed.

Clear or rename:

- `output_raykov_xrag_fixed_firstline_50/`
- `output_twostage/`
- `bartlett_encoding_vs_consolidation/`

Then run:

```bash
cd full_model/narratives
python run_all.py
python collate_figures.py --overwrite
```

## End-To-End Runner

`run_all.py` orchestrates the simulation stages:

```bash
python run_all.py
```

What it does:

1. Runs the Raykov xRAG simulation with typical, incomplete, and updated story
   variants.
2. Runs Bartlett two-stage training for the default topics: `Nature`,
   `Politics`, `Universe`, `Sport`, and `Health`.
3. Runs the encoding-vs-consolidation experiment.
4. Runs lower-level plotting/sampling scripts unless `--skip_plots` is passed.

Useful stage controls:

```bash
python run_all.py --skip_raykov
python run_all.py --skip_bartlett
python run_all.py --skip_enc_con
python run_all.py --skip_plots
python run_all.py --force_raykov
python run_all.py --force_bartlett
python run_all.py --force_stage2
python run_all.py --force_enc_con
```

For a true scratch run, prefer clearing the output directories listed above
before invoking `run_all.py`; the force flags rerun selected stages but do not
serve as a single complete cache-cleaning command.

## Figure Collation

`collate_figures.py` reads configuration from `plot_config.py`, assembles
`figures/Figure 5.pdf`, and exports the plotted values to `source_data/`.

```bash
python collate_figures.py
python collate_figures.py --overwrite
python collate_figures.py --results_dir output_twostage
python collate_figures.py --raykov_dir output_raykov_xrag_fixed_firstline_50/data
python collate_figures.py --enc_con_dir bartlett_encoding_vs_consolidation
```

Use `--overwrite` when changing sampling settings or when reproducing from
scratch, because it ignores cached CSVs and per-topic generation caches.

## Other Scripts

- `stories.py`: Raykov xRAG story preparation, pre-consolidation recall,
  consolidation, and post-consolidation recall.
- `bartlett_twostage.py`: two-stage LoRA training on topic corpora followed by
  Bartlett story training.
- `bartlett_encoding_vs_consolidation.py`: comparison of compressed encoding and
  consolidated recall for Figure 5.
- `plot.py`: lower-level plotting and checkpoint sampling utilities used by
  `run_all.py`.
- `plot_embeddings_standalone.py`, `analyze_ckpt_pca.py`,
  `wordcloud_from_ckpts.py`, and `keyword_experiments.py`: standalone analysis
  helpers for cached Bartlett outputs.
