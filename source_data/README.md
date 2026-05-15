Nature Communications source data exports are written here when the figure
generation scripts run.

CSV filenames start with the figure and panel where possible, for example
`Figure_6h_grid_generalisation.csv` or
`Figure_7g_correlation_vs_layer.csv`.  The figure and panel are encoded in the
filename, so the tables avoid repeated `figure`/`panel` metadata columns.  Each
CSV contains the plotted values with axis-like columns and, where relevant,
legend or grouping columns such as `series`, `Task`, `Memory stage`, or
`Condition`.

For panels that show both summary bars and overlaid individual points, the
`Value type` column distinguishes the plotted mean rows from the individual
point rows.

Regenerate the inference source data from cached trained models with:

```bash
python inference/generate_figures.py --config inference/inference_config.json
```

To recompute inference caches before exporting source data, add `--clear-cache`.
Regenerate the other figure source data by rerunning the corresponding figure
scripts documented in `FIGURES.md`, which distinguishes cached-data
reproduction from scratch reproduction.
