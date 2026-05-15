from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scripts.source_data import export_figure_source_data


def _savefig(out_path: str | Path, *, dpi: int = 300) -> None:
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()


def plot_query_vs_doc_heatmap(
    dist2: np.ndarray,
    *,
    out_path: str | Path,
    title: str = "Query → Doc distance (squared L2)",
    max_n: int = 120,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    D = np.asarray(dist2, dtype=np.float32)
    n = int(D.shape[0])
    if n > max_n:
        idx = np.linspace(0, n - 1, num=max_n).round().astype(int)
        idx = np.unique(idx)
        D = D[np.ix_(idx, idx)]

    plt.figure(figsize=(3.0, 3.0))
    sns.heatmap(D, cmap="mako", square=True)
    plt.title(title)
    plt.xlabel("Doc id (index)")
    plt.ylabel("Query id (index)")
    plt.tight_layout()
    _savefig(out_path)


def plot_sweep(
    rows: List[Dict[str, Any]],
    *,
    x_key: str,
    out_path: str | Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    x = [float(r[x_key]) for r in rows]
    rec = [float(r["recall_rate"]) for r in rows]

    plt.figure(figsize=(3.0, 3.0))
    plt.plot(x, rec, "-o")
    plt.ylim(0.0, 1.01)
    plt.grid(True, alpha=0.25)
    plt.xlabel(x_key)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    _savefig(out_path)


def plot_three_panel(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    *,
    decay_rows: List[Dict[str, Any]],
    beta_rows: List[Dict[str, Any]],
    out_path: str | Path,
    max_n: int = 120,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    panel_title_fs = 16
    axis_label_fs = 12
    tick_fs = 9

    Q = np.asarray(query_emb, dtype=np.float32)
    D = np.asarray(doc_emb, dtype=np.float32)
    n = int(min(Q.shape[0], D.shape[0]))
    if n > max_n:
        idx = np.linspace(0, n - 1, num=max_n).round().astype(int)
        idx = np.unique(idx)
        Q = Q[idx]
        D = D[idx]

    # Normalized dot-product (cosine) similarity in the model-hippocampus space.
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    S = (Qn @ Dn.T).astype(np.float32)
    smin = float(np.min(S))
    smax = float(np.max(S))
    center = 0.5 * (smin + smax)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    # Slightly increase spacing between panels.
    try:
        fig.set_constrained_layout_pads(wspace=0.08)
    except Exception:
        pass

    # a) Query → memory similarity heatmap
    ax = axes[0]
    sns.heatmap(
        S,
        cmap="vlag",
        square=True,
        ax=ax,
        cbar=True,
        vmin=smin,
        vmax=smax,
        center=center,
        cbar_kws={"label": "Projected cosine similarity"},
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title("a)", loc="center", fontsize=panel_title_fs)
    ax.set_xlabel("Memory", fontsize=axis_label_fs)
    ax.set_ylabel("Query", fontsize=axis_label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax._source_data_rows = [
        {
            "query_index": int(i),
            "memory_index": int(j),
            "Projected cosine similarity": float(S[i, j]),
        }
        for i in range(S.shape[0])
        for j in range(S.shape[1])
    ]

    # b) Recall vs decay
    ax = axes[1]
    x = [float(r["decay_rate"]) for r in decay_rows]
    y = [float(r["recall_rate"]) for r in decay_rows]
    ax.plot(x, y, "-o", label="Recall")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("b)", loc="center", fontsize=panel_title_fs)
    ax.set_xlabel("Decay constant", fontsize=axis_label_fs)
    ax.set_ylabel("Fraction perfectly recalled", fontsize=axis_label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax._source_data_rows = [
        {
            "Decay constant": float(r["decay_rate"]),
            "Fraction perfectly recalled": float(r["recall_rate"]),
        }
        for r in decay_rows
    ]

    # c) Recall vs beta
    ax = axes[2]
    x = [float(r["beta"]) for r in beta_rows]
    y = [float(r["recall_rate"]) for r in beta_rows]
    ax.plot(x, y, "-o", label="Recall")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("c)", loc="center", fontsize=panel_title_fs)
    ax.set_xlabel("Beta", fontsize=axis_label_fs)
    ax.set_ylabel("Fraction perfectly recalled", fontsize=axis_label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax._source_data_rows = [
        {
            "Beta": float(r["beta"]),
            "Fraction perfectly recalled": float(r["recall_rate"]),
        }
        for r in beta_rows
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_figure_source_data(fig, "Figure S2")
    fig.savefig(out, dpi=300)
    plt.close(fig)
