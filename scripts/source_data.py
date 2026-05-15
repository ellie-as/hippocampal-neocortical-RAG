"""Utilities for exporting plotted Matplotlib values as source-data CSVs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATA_DIR = REPO_ROOT / "source_data"


def _clean(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_-]+", "", text)
    return text[:80] or "untitled"


def _panel_from_title(title: str, fallback: int) -> str:
    match = re.match(r"\s*([A-Za-z][A-Za-z0-9]?)\)", title or "")
    if match:
        return match.group(1).lower()
    return f"panel_{fallback:02d}"


def _title_without_panel(title: str) -> str:
    return re.sub(r"^\s*[A-Za-z][A-Za-z0-9]?\)\s*", "", title or "").strip()


def _axis_column(label: str, fallback: str) -> str:
    label = str(label or "").strip()
    return label if label else fallback


def _tick_label(ax, value: float, axis: str) -> str:
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    if len(ticks) == 0 or len(labels) == 0:
        return ""
    idx = int(np.argmin(np.abs(np.asarray(ticks, dtype=float) - float(value))))
    if abs(float(ticks[idx]) - float(value)) > 1e-6:
        return ""
    return labels[idx].get_text()


def _artist_label(label: Any) -> str:
    label = "" if label is None else str(label)
    return "" if label.startswith("_") else label


def _rows_from_custom_source(figure: str, panel: str, subplot_index: int, ax) -> list[dict[str, Any]]:
    rows = getattr(ax, "_source_data_rows", None)
    if not rows:
        return []
    return [dict(row) for row in rows]


def _rows_from_lines(figure: str, panel: str, subplot_index: int, ax) -> list[dict[str, Any]]:
    rows = []
    x_col = _axis_column(ax.get_xlabel(), "x")
    y_col = _axis_column(ax.get_ylabel(), "y")
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(orig=False))
        ydata = np.asarray(line.get_ydata(orig=False))
        if xdata.size == 0 or ydata.size == 0 or xdata.size != ydata.size:
            continue
        label = _artist_label(line.get_label())
        for i, (x, y) in enumerate(zip(xdata, ydata)):
            row: dict[str, Any] = {}
            if label:
                row["series"] = label
            x_label = _tick_label(ax, float(x), "x")
            row[x_col] = x_label if x_label else float(x)
            row[y_col] = float(y)
            rows.append(row)
    return rows


def _rows_from_bars(figure: str, panel: str, subplot_index: int, ax) -> list[dict[str, Any]]:
    rows = []
    x_col = _axis_column(ax.get_xlabel(), "x")
    y_col = _axis_column(ax.get_ylabel(), "y")
    patch_labels: dict[int, str] = {}
    for container in getattr(ax, "containers", []):
        label = _artist_label(getattr(container, "get_label", lambda: "")())
        if not label:
            continue
        for patch in getattr(container, "patches", []):
            patch_labels[id(patch)] = label
    for i, patch in enumerate(ax.patches):
        if patch is ax.patch:
            continue
        if not hasattr(patch, "get_x") or not hasattr(patch, "get_width") or not hasattr(patch, "get_height"):
            continue
        width = float(patch.get_width())
        height = float(patch.get_height())
        x_left = float(patch.get_x())
        y_bottom = float(patch.get_y())
        if not np.isfinite(width) or not np.isfinite(height):
            continue
        x = x_left + width / 2
        row: dict[str, Any] = {}
        label = patch_labels.get(id(patch), _artist_label(patch.get_label()))
        if label:
            row["series"] = label
        x_label = _tick_label(ax, x, "x")
        row[x_col] = x_label if x_label else x
        row[y_col] = height
        rows.append(row)
    return rows


def _rows_from_collections(figure: str, panel: str, subplot_index: int, ax) -> list[dict[str, Any]]:
    rows = []
    x_col = _axis_column(ax.get_xlabel(), "x")
    y_col = _axis_column(ax.get_ylabel(), "y")
    for coll_idx, collection in enumerate(ax.collections):
        label = _artist_label(collection.get_label())
        offsets = getattr(collection, "get_offsets", lambda: np.empty((0, 2)))()
        offsets = np.asarray(offsets)
        if offsets.ndim == 2 and offsets.shape[1] >= 2 and offsets.size:
            for i, (x, y) in enumerate(offsets[:, :2]):
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                row: dict[str, Any] = {}
                if label:
                    row["series"] = label
                x_label = _tick_label(ax, float(x), "x")
                row[x_col] = x_label if x_label else float(x)
                row[y_col] = float(y)
                rows.append(row)
            continue
    return rows


def _rows_from_images(figure: str, panel: str, subplot_index: int, ax) -> list[dict[str, Any]]:
    rows = []
    for image_idx, image in enumerate(ax.images):
        arr = np.asarray(image.get_array())
        if arr.ndim == 2:
            for row_idx in range(arr.shape[0]):
                for col_idx in range(arr.shape[1]):
                    rows.append(
                        {
                            "image_index": image_idx,
                            "row": row_idx,
                            "column": col_idx,
                            "value": float(arr[row_idx, col_idx]),
                        }
                    )
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_figure_source_data(fig, figure: str, *, out_dir: str | Path | None = None) -> None:
    """Export plotted values for each axis in *fig* to ``source_data/*.csv``."""
    target = Path(out_dir) if out_dir is not None else SOURCE_DATA_DIR
    figure_clean = _clean(figure)
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob(f"{figure_clean}_*.csv"):
        stale.unlink()

    for subplot_index, ax in enumerate(fig.get_axes(), start=1):
        if str(ax.get_label()).startswith("<colorbar"):
            continue
        if getattr(ax, "_skip_source_data", False):
            continue
        title = ax.get_title()
        panel = getattr(ax, "_source_data_panel", None) or _panel_from_title(title, subplot_index)
        rows = []
        rows.extend(_rows_from_custom_source(figure, panel, subplot_index, ax))
        if not rows:
            rows.extend(_rows_from_lines(figure, panel, subplot_index, ax))
            rows.extend(_rows_from_bars(figure, panel, subplot_index, ax))
            rows.extend(_rows_from_collections(figure, panel, subplot_index, ax))
            rows.extend(_rows_from_images(figure, panel, subplot_index, ax))
        if not rows:
            continue
        source_title = getattr(ax, "_source_data_title", None)
        title_clean = _clean(source_title if source_title is not None else _title_without_panel(title))
        filename = f"{figure_clean}_{panel}.csv" if title_clean == "untitled" else f"{figure_clean}_{panel}_{title_clean}.csv"
        _write_rows(target / filename, rows)
