"""Export manuscript Figure 6g/h/i/k source-data tables.

This script does not contain result values. It reads model-generated JSON caches
and writes derived CSV tables for auditability. It can optionally write separate
panel PDFs for editing.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

INFERENCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INFERENCE_DIR))

from run_config import load_config
import collate_inf_figures as figures


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_source_csv(config: dict, filename: str, rows: list[dict]) -> None:
    source_dir = Path(config["figures_dir"]).parent / "source_data"
    _write_csv(source_dir / filename, rows)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required cache: {path}\n"
            "Run collate_inf_figures.py and rag_composition.py first."
        )
    return json.loads(path.read_text())


def _hops(pattern: str) -> int:
    return len(pattern.split()) // 2


def _panel_label(ax, label: str) -> None:
    ax.text(-0.22, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=11)


def export_aggregated_inference(data: dict, out_path: Path, config: dict) -> None:
    rows = []
    for task, patterns in data.items():
        for pattern, accuracy in patterns.items():
            rows.append(
                {
                    "Task": task,
                    "Pattern": pattern,
                    "Number of transitions": _hops(pattern),
                    "Accuracy": float(accuracy),
                }
            )
    _write_csv(out_path, rows)
    _write_source_csv(config, out_path.name, rows)


def export_grid_generalisation(data: dict, out_path: Path, config: dict) -> None:
    rows = [
        {"Grid size": int(n) + 1, "Average accuracy": vals[0], "SEM": vals[1]}
        for n, vals in sorted(data.items(), key=lambda item: int(item[0]))
    ]
    _write_csv(out_path, rows)
    _write_source_csv(config, out_path.name, rows)


def export_imagination(data: dict, out_path: Path, config: dict) -> None:
    rows = []
    lengths = data["lengths"]
    for temp, fractions in sorted(data["validity"].items(), key=lambda item: float(item[0])):
        for transitions, fraction in zip(lengths, fractions):
            rows.append(
                {
                    "Temperature": float(temp),
                    "Number of transitions": int(transitions),
                    "Fraction valid": float(fraction),
                }
            )
    _write_csv(out_path, rows)
    _write_source_csv(config, out_path.name, rows)


def export_rag_summary(data: dict, out_path: Path, config: dict) -> None:
    summary_conds = [
        ("NC", "NC only"),
        ("HPC", "HPC only"),
        ("Mem-1", "RAG single"),
        ("RAG-2L", "RAG multi"),
    ]
    task_labels = {"Spatial": "Spatial", "Family": "Family tree"}
    task_agg: dict[str, dict[str, list[int]]] = {}

    for group_name, conds in data.items():
        if "4-hop" in group_name:
            continue
        task = group_name.split()[0]
        task_agg.setdefault(task, {cn: [] for cn, _ in summary_conds})
        for cn, _ in summary_conds:
            task_agg[task][cn].extend(conds.get(cn, []))

    rows = []
    for task in ("Spatial", "Family"):
        if task not in task_agg:
            continue
        for cache_name, label in summary_conds:
            vals = task_agg[task][cache_name]
            mean = figures.np.mean(vals) if vals else 0.0
            sem = figures.np.std(vals) / figures.np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            rows.append(
                {
                    "Task": task_labels[task],
                    "Condition": label,
                    "Average accuracy": float(mean),
                    "SEM": float(sem),
                    "n": len(vals),
                }
            )
    _write_csv(out_path, rows)
    _write_source_csv(config, out_path.name, rows)


def save_panel(label: str, plotter, data: dict, out_path: Path, figsize: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    plotter(ax, data)
    _panel_label(ax, label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export Figure 6g/h/i/k source data from generated caches.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference_config.json with trained model and output paths.",
    )
    parser.add_argument(
        "--pdfs",
        action="store_true",
        help="Also export separate Figure 6g/h/i/k panel PDFs.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    figures.configure(args.config)
    cache_dir = config["cache_dir"]
    data_dir = config["data_dir"]
    figures_dir = config["figures_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    aggregated = _load_json(cache_dir / "aggregated_inf.json")
    grid = _load_json(cache_dir / "grid_generalisation.json")
    imagination = _load_json(cache_dir / "imagination.json")
    rag = _load_json(cache_dir / "rag_composition.json")

    export_aggregated_inference(aggregated, data_dir / "Figure_6g_aggregated_inference.csv", config)
    export_grid_generalisation(grid, data_dir / "Figure_6h_grid_generalisation.csv", config)
    export_imagination(imagination, data_dir / "Figure_6i_imagination_validity.csv", config)
    export_rag_summary(rag, data_dir / "Figure_6k_rag_composition_summary.csv", config)

    if args.pdfs:
        save_panel("g", figures.plot_aggregated_inf, aggregated, figures_dir / "Figure 6g.pdf", (2.3, 2.0))
        save_panel("h", figures.plot_grid_generalisation, grid, figures_dir / "Figure 6h.pdf", (2.3, 2.0))
        save_panel("i", figures.plot_temp_validity, imagination, figures_dir / "Figure 6i.pdf", (2.3, 2.0))
        save_panel("k", figures.plot_rag_summary, rag, figures_dir / "Figure 6k.pdf", (2.55, 2.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
