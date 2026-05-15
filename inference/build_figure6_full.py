"""Build manuscript Figure 6 from diagrams, loss logs, and inference panels.

The loss panels are generated from Hugging Face ``trainer_state.json`` files.
Panels g/h/i/k are redrawn from ``source_data/Figure_6*.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image, ImageChops

INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
sys.path.insert(0, str(INFERENCE_DIR))

from run_config import load_config

SOURCE_DATA_DIR = REPO_ROOT / "source_data"
DIAGRAMS_DIR = REPO_ROOT / "figures" / "diagrams"
DEFAULT_TOP_ROW_IMAGE = DIAGRAMS_DIR / "Fig 6 parts a to d.png"
DEFAULT_RAG_IMAGE = DIAGRAMS_DIR / "Figure 6 part j.png"
LEGACY_TOP_ROW_IMAGE = INFERENCE_DIR / "assets" / "figure6_top_row.png"

SPATIAL_COLOR = "#ff8c8c"
FAMILY_COLOR = "#8d8cf2"
TRAIN_COLOR = "#ff4a4a"
VAL_COLOR = "#5b5bff"
NODE_RED = "#a55369"
NODE_BLUE = "#edf3fb"
EDGE = "#222222"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_loss_rows(trainer_state_path: Path) -> list[dict[str, Any]]:
    state = json.loads(trainer_state_path.read_text())
    rows: list[dict[str, Any]] = []
    for entry in state.get("log_history", []):
        epoch = entry.get("epoch")
        step = entry.get("step")
        if epoch is None or step is None:
            continue
        if "loss" in entry:
            rows.append(
                {
                    "Epoch": float(epoch),
                    "Step": int(step),
                    "Split": "Train",
                    "Loss": float(entry["loss"]),
                }
            )
        if "eval_loss" in entry:
            rows.append(
                {
                    "Epoch": float(epoch),
                    "Step": int(step),
                    "Split": "Validation",
                    "Loss": float(entry["eval_loss"]),
                }
            )
    return rows


def _plot_loss(ax, rows: list[dict[str, Any]], title: str | None = None) -> None:
    df = pd.DataFrame(rows)
    for split, color, label in [
        ("Train", TRAIN_COLOR, "Train loss"),
        ("Validation", VAL_COLOR, "Val loss"),
    ]:
        part = df[df["Split"] == split].sort_values("Epoch")
        ax.plot(
            part["Epoch"],
            part["Loss"],
            label=label,
            color=color,
            lw=1.6,
            alpha=0.55,
            marker=".",
            markersize=2,
        )
    if title:
        ax.set_title(title, loc="center", fontsize=17)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0.25)
    ax.legend(frameon=True, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def _export_loss_panel(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(2.35, 2.35))
    _plot_loss(ax, rows)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _arrow(ax, xy1, xy2, *, color=EDGE, lw=1.0, rad=0.0, alpha=1.0, mutation_scale=8):
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=mutation_scale,
            connectionstyle=f"arc3,rad={rad}",
            alpha=alpha,
        )
    )


def _node(ax, xy, label: str, *, active: bool = False, radius: float = 0.056) -> None:
    face = NODE_RED if active else NODE_BLUE
    edge = "#633042" if active else "#8aa2ce"
    ax.add_patch(Circle(xy, radius, facecolor=face, edgecolor=edge, lw=1.2))
    ax.text(
        xy[0],
        xy[1],
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="white" if active else "black",
        fontweight="bold" if active else "normal",
    )


def _panel_heading(ax, label: str, title: str) -> None:
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=13, ha="left", va="bottom")
    ax.text(0.18, 1.02, title, transform=ax.transAxes, fontsize=13, ha="left", va="bottom")


def draw_spatial_diagram(ax, *, inference: bool = False) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = [
        ["hb", "vc", "id"],
        ["us", "er", "ns"],
        ["zu", "ko", "we"],
    ] if not inference else [
        ["jn", "xu", "ue"],
        ["sm", "hw", "iv"],
        ["em", "qa", "le"],
    ]
    active = {"hb", "vc", "er", "ns", "we"} if not inference else {"jn", "xu", "hw", "sm"}
    xs = [0.18, 0.48, 0.78]
    ys = [0.82, 0.62, 0.42]
    pos = {}
    for r, row in enumerate(labels):
        for c, lab in enumerate(row):
            pos[lab] = (xs[c], ys[r])
    # Grid transition edges.
    for r in range(3):
        for c in range(2):
            a, b = labels[r][c], labels[r][c + 1]
            _arrow(ax, (pos[a][0] + 0.055, pos[a][1]), (pos[b][0] - 0.055, pos[b][1]), lw=0.9, alpha=0.85)
            _arrow(ax, (pos[b][0] - 0.055, pos[b][1] - 0.015), (pos[a][0] + 0.055, pos[a][1] - 0.015), lw=0.9, alpha=0.85)
    for c in range(3):
        for r in range(2):
            a, b = labels[r][c], labels[r + 1][c]
            _arrow(ax, (pos[a][0], pos[a][1] - 0.055), (pos[b][0], pos[b][1] + 0.055), lw=0.9, alpha=0.85)
            _arrow(ax, (pos[b][0] - 0.018, pos[b][1] + 0.055), (pos[a][0] - 0.018, pos[a][1] - 0.055), lw=0.9, alpha=0.85)
    path = ["hb", "vc", "er", "ns", "we"] if not inference else ["jn", "xu", "hw", "sm"]
    for a, b in zip(path, path[1:]):
        _arrow(ax, pos[a], pos[b], color="#aa3541", lw=1.4, mutation_scale=10, alpha=0.95, rad=0.12)
    for lab, xy in pos.items():
        _node(ax, xy, lab, active=lab in active)
    seq = "EAST   SOUTH   EAST   SOUTH\nhb \u27f6 vc \u27f6 er \u27f6 ns \u27f6 we" if not inference else \
        "EAST   SOUTH   WEST   NORTH\njn \u27f6 xu \u27f6 hw \u27f6 sm \u27f6 ?"
    ax.text(0.5, 0.22, seq, ha="center", va="center", fontsize=10)
    ax.text(
        0.5,
        0.06,
        "'hb EAST vc SOUTH er EAST ns\nSOUTH we'" if not inference else "Correct output: jn",
        ha="center",
        va="center",
        fontsize=9.5,
    )


def draw_family_diagram(ax, *, inference: bool = False) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if not inference:
        nodes = {
            "nd": (0.18, 0.84), "oq": (0.43, 0.84), "nr": (0.10, 0.64), "ew": (0.48, 0.64),
            "oc": (0.40, 0.46), "wn": (0.67, 0.64), "xy": (0.78, 0.46),
            "ye": (0.56, 0.84), "nw": (0.77, 0.84), "aw": (0.93, 0.64),
        }
        active = {"nd", "oc", "wn", "nw"}
        seq_top = "GRANDPARENT_OF  CHILD_OF  CHILD_OF"
        seq_mid = "nd \u27f6 oc \u27f6 wn \u27f6 nw"
        seq_bottom = "'nd GRANDPARENT_OF oc CHILD_OF wn\nCHILD_OF nw'"
    else:
        nodes = {
            "be": (0.10, 0.84), "un": (0.30, 0.84), "rn": (0.08, 0.64), "ye": (0.43, 0.64),
            "fg": (0.38, 0.45), "em": (0.65, 0.64), "bt": (0.65, 0.45),
            "so": (0.72, 0.84), "rt": (0.90, 0.84), "jk": (0.88, 0.64),
        }
        active = {"ye", "fg", "em"}
        seq_top = "PARENT_OF  CHILD_OF  SPOUSE_OF"
        seq_mid = "ye \u27f6 fg \u27f6 em \u27f6 ?"
        seq_bottom = "Correct output: ye"
    # Pedigree-like structure.
    for x1, x2, y in [(0.18, 0.43, 0.77), (0.56, 0.77, 0.77), (0.48, 0.72, 0.57)]:
        ax.plot([x1, x2], [y, y], color=EDGE, lw=1.0)
    for x, y1, y2 in [(0.31, 0.77, 0.68), (0.67, 0.77, 0.68), (0.60, 0.57, 0.50)]:
        ax.plot([x, x], [y1, y2], color=EDGE, lw=1.0)
    # Highlighted inference path.
    if not inference:
        _arrow(ax, nodes["nd"], nodes["oc"], color="#aa3541", lw=1.0, rad=0.15)
        _arrow(ax, nodes["oc"], nodes["wn"], color="#aa3541", lw=1.0, rad=0.18)
        _arrow(ax, nodes["wn"], nodes["nw"], color="#aa3541", lw=1.0, rad=0.22)
    else:
        _arrow(ax, nodes["ye"], nodes["fg"], color="#aa3541", lw=1.0, rad=0.35)
        _arrow(ax, nodes["fg"], nodes["em"], color="#aa3541", lw=1.0, rad=0.18)
        _arrow(ax, nodes["em"], nodes["ye"], color="#aa3541", lw=1.0, rad=0.35)
    for lab, xy in nodes.items():
        _node(ax, xy, lab, active=lab in active, radius=0.052)
    ax.text(0.5, 0.30, seq_top, ha="center", va="center", fontsize=9.5)
    ax.text(0.5, 0.22, seq_mid, ha="center", va="center", fontsize=11)
    ax.text(0.5, 0.07, seq_bottom, ha="center", va="center", fontsize=9)


def draw_rag_diagram(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.95, "j)  Retrieval-augmented generation:", fontsize=13, ha="left", va="top")
    ax.text(0.52, 0.82, "Given a new task, e.g. ", fontsize=10, ha="right", va="center")
    ax.text(0.53, 0.82, "cd SIBLING_OF:", fontsize=10, color="#c3322b", ha="left", va="center")

    def box(x, y, w, h, title, rows, highlight=None):
        ax.text(x + w / 2, y + h + 0.035, title, ha="center", va="bottom", fontsize=10)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#eef3fa", edgecolor="#9aafd8", lw=1.0))
        for i, row in enumerate(rows):
            ry = y + h - 0.06 - i * 0.08
            face = "#b7d7f1" if highlight == row else "#eef3fa"
            ax.add_patch(Rectangle((x + 0.04, ry - 0.035), w - 0.08, 0.055, facecolor=face, edgecolor="#2d6da5", lw=0.8))
            ax.text(x + w / 2, ry - 0.007, row, ha="center", va="center", fontsize=9)

    box(0.03, 0.34, 0.20, 0.30, "HPC", ["X", "Y", "Z"], highlight="Z")
    box(0.29, 0.34, 0.20, 0.30, "NC", [], highlight=None)
    ax.add_patch(Rectangle((0.325, 0.475), 0.13, 0.06, facecolor="#fff4f4", edgecolor="#c3322b", lw=0.9))
    ax.text(0.39, 0.505, "New task", ha="center", va="center", fontsize=8.5)
    _arrow(ax, (0.22, 0.56), (0.29, 0.56), color="#b8c5d8", lw=2, mutation_scale=16)
    _arrow(ax, (0.29, 0.43), (0.23, 0.43), color="#b8c5d8", lw=2, mutation_scale=16, rad=-0.18)

    box(0.58, 0.34, 0.20, 0.30, "HPC", ["X", "Y", "Z"], highlight=None)
    ax.text(0.91, 0.675, "NC", ha="center", va="bottom", fontsize=10)
    ax.add_patch(Rectangle((0.82, 0.34), 0.18, 0.30, facecolor="#eef3fa", edgecolor="#9aafd8", lw=1.0))
    ax.text(0.91, 0.59, "Do", ha="center", va="center", fontsize=8.5)
    ax.add_patch(Rectangle((0.845, 0.515), 0.13, 0.055, facecolor="#fff4f4", edgecolor="#c3322b", lw=0.9))
    ax.text(0.91, 0.543, "New task", ha="center", va="center", fontsize=8.0)
    ax.text(0.91, 0.475, "given", ha="center", va="center", fontsize=8.5)
    ax.add_patch(Rectangle((0.855, 0.375), 0.11, 0.055, facecolor="#b7d7f1", edgecolor="#2d6da5", lw=0.8))
    ax.text(0.91, 0.402, "Z", ha="center", va="center", fontsize=8.5)

    ax.text(0.13, 0.22, "Retrieve relevant\nsequences from HPC", ha="center", va="center", fontsize=10, style="italic")
    ax.text(0.13, 0.06, "e.g. cd CHILD_OF bh\nPARENT_OF zs ...", ha="center", va="center", fontsize=10, color="#2d79c7")
    ax.text(0.72, 0.22, "Generate solution given\nsequences", ha="center", va="center", fontsize=10, style="italic")
    ax.text(0.58, 0.06, "Prompt: ", ha="right", va="center", fontsize=9.5)
    ax.text(0.59, 0.06, "cd CHILD_OF\nbh ... ", ha="left", va="center", fontsize=9.5, color="#2d79c7")
    ax.text(0.75, 0.06, "cd SIBLING_OF", ha="left", va="center", fontsize=9.5, color="#c3322b")
    _arrow(ax, (0.47, 0.08), (0.56, 0.08), color="#bcbcbc", lw=3, mutation_scale=18)


def _find_csv(candidates: list[str]) -> Path | None:
    for name in candidates:
        for base in [INFERENCE_DIR / "data", SOURCE_DATA_DIR, REPO_ROOT / "figures" / "data"]:
            path = base / name
            if path.exists():
                return path
    return None


def draw_panel_g(ax) -> bool:
    path = _find_csv(["Figure_6g_aggregated_inference.csv", "Figure_6_g.csv"])
    if path is None:
        return False
    df = pd.read_csv(path)
    x_col = "Number of transitions" if "Number of transitions" in df.columns else "transitions"
    task_col = "Task" if "Task" in df.columns else "task"
    if "Average accuracy" in df.columns:
        grouped = df.rename(columns={"Average accuracy": "mean", "SD": "err", "SEM": "err"})
    else:
        grouped = (
            df.groupby([x_col, task_col])["Accuracy"]
            .agg(mean="mean", err=lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0)
            .reset_index()
        )
    transitions = [2, 4, 6]
    tasks = ["Spatial", "Family tree"]
    x = np.arange(len(transitions))
    width = 0.35
    rng = np.random.default_rng(0)
    raw_points = df if "Accuracy" in df.columns else None
    for j, task in enumerate(tasks):
        part = grouped[grouped[task_col].astype(str) == task]
        means = [float(part[part[x_col] == t]["mean"].iloc[0]) if not part[part[x_col] == t].empty else np.nan for t in transitions]
        errs = [float(part[part[x_col] == t]["err"].iloc[0]) if "err" in part and not part[part[x_col] == t].empty else 0.0 for t in transitions]
        positions = x - width / 2 + j * width
        color = SPATIAL_COLOR if task == "Spatial" else FAMILY_COLOR
        ax.bar(positions, means, width, yerr=errs, capsize=3, label=task,
               color=color, edgecolor="none", alpha=0.9, zorder=2)
        if raw_points is not None:
            raw_task = raw_points[raw_points[task_col].astype(str) == task].copy()
            raw_task[x_col] = pd.to_numeric(raw_task[x_col], errors="coerce")
            for x0, transition in zip(positions, transitions):
                values = raw_task.loc[raw_task[x_col] == transition, "Accuracy"].dropna().astype(float)
                if values.empty:
                    continue
                jitter = rng.uniform(-width * 0.18, width * 0.18, size=len(values))
                ax.scatter(
                    np.full(len(values), x0) + jitter,
                    values,
                    s=18,
                    facecolors="white",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.95,
                    zorder=4,
                )
    ax.set_title("g)", loc="center")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in transitions])
    ax.set_xlabel("Number of transitions")
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0.8, 1.01)
    ax.legend(loc="lower left")
    return True


def draw_panel_h(ax) -> bool:
    path = _find_csv(["Figure_6h_grid_generalisation.csv", "Figure_6_h.csv"])
    if path is None:
        return False
    df = pd.read_csv(path)
    x = df["Grid size"] if "Grid size" in df.columns else df["grid_size"]
    y = df["Average accuracy"] if "Average accuracy" in df.columns else df["mean_accuracy"]
    err = df["SEM"] if "SEM" in df.columns else df.get("sem", 0)
    ax.errorbar(x, y, yerr=err, fmt="o-", color="blue", capsize=4, lw=1.7)
    ax.set_title("h)", loc="center")
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.05)
    return True


def draw_panel_i(ax) -> bool:
    path = _find_csv(["Figure_6i_imagination_validity.csv", "Figure_6_i.csv"])
    if path is None:
        return False
    df = pd.read_csv(path)
    t_col = "Temperature" if "Temperature" in df.columns else "temperature"
    x_col = "Number of transitions" if "Number of transitions" in df.columns else "transitions"
    y_col = "Fraction valid" if "Fraction valid" in df.columns else "fraction_valid"
    colors = plt.get_cmap("magma")(np.linspace(0.15, 0.75, df[t_col].nunique()))
    for color, temp in zip(colors, sorted(df[t_col].unique())):
        part = df[df[t_col] == temp].sort_values(x_col)
        ax.plot(part[x_col], part[y_col], marker="o", lw=1.7, label=f"{temp:g}", color=color)
    ax.set_title("i)", loc="center")
    ax.set_xlabel("Number of transitions")
    ax.set_ylabel("Fraction valid")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Temp.", loc="lower left")
    return True


def draw_panel_k(ax) -> bool:
    path = _find_csv(["Figure_6k_rag_composition_summary.csv", "Figure_6_k.csv"])
    if path is None:
        return False
    df = pd.read_csv(path)
    task_col = "Task" if "Task" in df.columns else "task"
    cond_col = "Condition" if "Condition" in df.columns else "condition"
    y_col = "Average accuracy" if "Average accuracy" in df.columns else "mean_accuracy"
    err_col = "SEM" if "SEM" in df.columns else "sem"
    conditions = ["NC only", "HPC only", "RAG single", "RAG multi"]
    labels = ["NC only", "HPC only", "RAG\n(single)", "RAG\n(multi)"]
    x = np.arange(len(conditions))
    width = 0.35
    for j, task in enumerate(["Spatial", "Family tree"]):
        part = df[df[task_col].astype(str) == task]
        means = []
        errs = []
        for cond in conditions:
            rows = part[part[cond_col].astype(str).str.replace("\n", " ", regex=False) == cond]
            means.append(float(rows[y_col].iloc[0]) if not rows.empty else np.nan)
            errs.append(float(rows[err_col].iloc[0]) if err_col in rows and not rows.empty else 0.0)
        ax.bar(x - width / 2 + j * width, means, width, yerr=errs, capsize=3, label=task,
               color=SPATIAL_COLOR if task == "Spatial" else FAMILY_COLOR, edgecolor="none")
    ax.set_title("k) RAG vs. baselines", loc="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left")
    return True


def _load_top_row_image(path: str | Path | None) -> Image.Image:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(
        [
            DEFAULT_TOP_ROW_IMAGE,
            LEGACY_TOP_ROW_IMAGE,
            REPO_ROOT / "figures" / "Figure 6 top row.png",
            REPO_ROOT / "figures" / "figure6_top_row.png",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return Image.open(candidate).convert("RGB")

    raise FileNotFoundError(
        "No Figure 6 top-row PNG found. Provide --top-row-image or add "
        f"{DEFAULT_TOP_ROW_IMAGE.relative_to(REPO_ROOT)}."
    )


def _load_rag_image(path: str | Path | None) -> Image.Image:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(DEFAULT_RAG_IMAGE)
    for candidate in candidates:
        if candidate.exists():
            return _crop_whitespace(Image.open(candidate).convert("RGBA"))
    raise FileNotFoundError(
        "No Figure 6j diagram PNG found. Provide --panel-j-image or add "
        f"{DEFAULT_RAG_IMAGE.relative_to(REPO_ROOT)}."
    )


def _crop_whitespace(image: Image.Image, *, threshold: int = 8, pad: int = 8) -> Image.Image:
    rgb = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", image.size, "white")).convert("L")
    mask = diff.point(lambda value: 255 if value > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.size[0], right + pad)
    bottom = min(image.size[1], bottom + pad)
    return image.crop((left, top, right, bottom))


def _draw_top_row_image(ax, image: Image.Image) -> None:
    ax.set_axis_off()
    image_aspect = image.size[1] / image.size[0]
    ax.imshow(image, extent=(0, 1, 0, image_aspect), interpolation="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, image_aspect)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("S")
    y = image_aspect + 0.022
    for x, label in [
        (0.142, "a)   Spatial training"),
        (0.374, "b)   Spatial inference"),
        (0.626, "c)   Family tree training"),
        (0.872, "d)   Family tree inference"),
    ]:
        ax.text(x, y, label, fontsize=17, ha="center", va="center", clip_on=False)


def _draw_rag_image(ax, image: Image.Image) -> None:
    ax.set_axis_off()
    image_aspect = image.size[1] / image.size[0]
    ax.imshow(image, extent=(0, 1, 0, image_aspect), interpolation="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, image_aspect)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("S")
    y = image_aspect + 0.01
    ax.text(
        0.5,
        y,
        "j)  Retrieval-augmented generation:",
        fontsize=17,
        ha="center",
        va="center",
        clip_on=False,
    )


def _match_rag_axis_height(fig, ax, left_ax, right_ax, image: Image.Image) -> None:
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()
    height = left_pos.height * 1.18
    width = height * (fig.get_figheight() / fig.get_figwidth()) * (image.size[0] / image.size[1])
    gap_left = left_pos.x1
    gap_right = right_pos.x0
    x0 = gap_left + (gap_right - gap_left - width) / 2
    y0 = max(0.02, left_pos.y0 - left_pos.height * 0.16)
    ax.set_position([x0, y0, width, height])


def _widen_side_axes(left_ax, center_ax, right_ax) -> None:
    center_pos = center_ax.get_position()
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()
    gap = 0.014

    left_x0 = max(0.030, left_pos.x0 - 0.010)
    left_x1 = min(center_pos.x0 - gap, left_pos.x1 + 0.035)
    left_ax.set_position([left_x0, left_pos.y0, left_x1 - left_x0, left_pos.height])

    right_x0 = max(center_pos.x1 + gap, right_pos.x0 - 0.035)
    right_x1 = min(0.990, right_pos.x1 + 0.010)
    right_ax.set_position([right_x0, right_pos.y0, right_x1 - right_x0, right_pos.height])


def _missing_panel(panel: str) -> None:
    raise FileNotFoundError(
        f"Missing source-data CSV for Figure 6{panel}. "
        "Run inference/generate_figures.py first."
    )


def build_figure(
    config_path: str | None,
    *,
    top_row_image: str | Path | None = None,
    panel_j_image: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    figures_dir = config["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    spatial_rows = _load_loss_rows(config["spatial_model_dir"] / "trainer_state.json")
    family_rows = _load_loss_rows(config["family_model_dir"] / "trainer_state.json")
    _write_csv(SOURCE_DATA_DIR / "Figure_6e_spatial_loss.csv", spatial_rows)
    _write_csv(SOURCE_DATA_DIR / "Figure_6f_family_tree_loss.csv", family_rows)

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    top_row = _load_top_row_image(top_row_image)
    rag_image = _load_rag_image(panel_j_image)

    fig = plt.figure(figsize=(16.0, 14.0), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.982, bottom=0.055)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[2.35, 1.0, 1.0],
        hspace=0.24,
    )
    middle = outer[1].subgridspec(1, 4, wspace=0.38)
    bottom = outer[2].subgridspec(1, 36, wspace=0.26)

    ax = fig.add_subplot(outer[0])
    top_pos = ax.get_position()
    ax.set_position([0.005, top_pos.y0 - 0.043, 0.99, top_pos.height])
    _draw_top_row_image(ax, top_row)

    ax = fig.add_subplot(middle[0, 0])
    _plot_loss(ax, spatial_rows, "e)   Spatial loss")
    ax = fig.add_subplot(middle[0, 1])
    _plot_loss(ax, family_rows, "f)   Family tree loss")
    ax = fig.add_subplot(middle[0, 2])
    if not draw_panel_g(ax):
        _missing_panel("g")
    ax = fig.add_subplot(middle[0, 3])
    if not draw_panel_h(ax):
        _missing_panel("h")

    ax_i = fig.add_subplot(bottom[0, 0:8])
    if not draw_panel_i(ax_i):
        _missing_panel("i")
    ax_j = fig.add_subplot(bottom[0, 9:27])
    ax_k = fig.add_subplot(bottom[0, 28:36])
    if not draw_panel_k(ax_k):
        _missing_panel("k")
    _match_rag_axis_height(fig, ax_j, ax_i, ax_k, rag_image)
    _widen_side_axes(ax_i, ax_j, ax_k)
    i_pos = ax_i.get_position()
    ax_i.set_position([i_pos.x0 + 0.025, i_pos.y0, i_pos.width, i_pos.height])
    _draw_rag_image(ax_j, rag_image)

    out_path = figures_dir / "Figure 6.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build full manuscript Figure 6.")
    parser.add_argument("--config", default=None, help="Path to inference_config.json.")
    parser.add_argument(
        "--top-row-image",
        default=None,
        help="PNG/JPEG to embed for Figure 6 panels a-d. Defaults to figures/diagrams/Fig 6 parts a to d.png.",
    )
    parser.add_argument(
        "--panel-j-image",
        default=None,
        help="PNG/JPEG to embed for Figure 6j. Defaults to figures/diagrams/Figure 6 part j.png.",
    )
    args = parser.parse_args(argv)
    out = build_figure(
        args.config,
        top_row_image=args.top_row_image,
        panel_j_image=args.panel_j_image,
    )
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
