#!/usr/bin/env python3
"""
Standalone script to generate PCA/UMAP/t-SNE embedding plots from cached Bartlett data.
Uses the same style as narratives_gpt2/Bartlett embedding analysis.ipynb.

Usage:
    python plot_embeddings_standalone.py --output_dir .
"""
from __future__ import annotations
import argparse
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from utils import BARTLETT_FALLBACK

# Color palette matching the notebook
BASE_COLORS = ['#6a00a8', '#e16462', '#b12a90', '#0d0887', '#f0f921', '#fca636']

# Bartlett story: single source of truth is utils.BARTLETT_FALLBACK
BARTLETT_STORY = BARTLETT_FALLBACK


def load_recalled_stories(
    pkl_dir: Path,
    categories: List[str],
    temp: float = 0.5,
    epoch: int = 5,
) -> pd.DataFrame:
    """Load recalled stories from cached pickle files."""
    records = []
    
    for filename in os.listdir(pkl_dir):
        if not filename.endswith('.pkl'):
            continue
        filepath = pkl_dir / filename
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        for category in categories:
            if category not in data or not data[category]:
                continue
            ckpts = sorted(data[category].keys(), key=lambda name: int(name.split('-')[-1]))
            epoch_map = {ck: i + 1 for i, ck in enumerate(ckpts)}
            
            for ckpt in data[category]:
                for t in [0, 0.5, 1, 1.5]:
                    texts = data[category][ckpt].get(t, [])
                    if isinstance(texts, str):
                        texts = [texts]
                    for story in texts:
                        records.append({
                            'topic': category,
                            'epoch': epoch_map[ckpt],
                            'temp': t,
                            'text': story
                        })
    
    return pd.DataFrame(records)


def load_background_texts(
    topics: List[str],
    max_chars: int,
    n_per_topic: int = 1000,
    sample_seed: int = 42,
) -> Dict[str, List[str]]:
    """Load background texts from Wikipedia dataset."""
    print("Loading Wikipedia topics dataset...")
    dataset = load_dataset('tarekziade/wikipedia-topics', split='train')
    wiki_df = dataset.to_pandas()
    
    def has_category(cats, target):
        """Check if target category is in the categories list."""
        if cats is None:
            return False
        if isinstance(cats, (list, tuple, np.ndarray)):
            return target in cats
        return False
    
    # Filter out People articles
    wiki_df = wiki_df[~wiki_df['categories'].apply(lambda x: has_category(x, 'People'))]
    
    topic_map = {
        'Universe': 'Universe',
        'Politics': 'Politics', 
        'Health': 'Health',
        'Sport': 'Sports',  # Wikipedia uses 'Sports'
        'Technology': 'Technology',
        'Nature': 'Nature'
    }
    
    result = {}
    for topic in topics:
        wiki_cat = topic_map.get(topic, topic)
        filtered = wiki_df[wiki_df['categories'].apply(lambda x: has_category(x, wiki_cat))]
        texts = filtered['text'].sample(frac=1, random_state=sample_seed).tolist()[:n_per_topic]
        texts = [t[:max_chars] for t in texts]
        result[topic] = texts
        print(f"  {topic}: {len(texts)} articles")
    
    return result


def embed_texts(model: SentenceTransformer, texts: List[str], max_chars: int = 800) -> np.ndarray:
    """Embed texts using sentence transformer."""
    texts = [t[:max_chars] for t in texts]
    return model.encode(texts, show_progress_bar=False)


def plot_2d_embedding(
    all_bg_points: np.ndarray,
    bg_means: np.ndarray,
    rec_means: np.ndarray,
    bart_emb: np.ndarray,
    topics: List[str],
    group_sizes: List[int],
    out_png: Path,
    method: str = "pca",
    random_state: int = 42,
):
    """Create 2D embedding plot with inset zoom."""
    from sklearn.decomposition import PCA
    
    colors = BASE_COLORS * ((len(topics) // len(BASE_COLORS)) + 1)
    color_map = {t: colors[i] for i, t in enumerate(topics)}
    
    # Stack all points for dimensionality reduction
    all_points = np.vstack([
        all_bg_points,
        bg_means,
        rec_means,
        bart_emb.reshape(1, -1)
    ])
    
    # Apply dimensionality reduction
    method_lower = method.lower()
    if method_lower == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        all_2d = reducer.fit_transform(all_points)
    elif method_lower == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            all_2d = reducer.fit_transform(all_points)
        except ImportError:
            print("[Warning] umap-learn not installed, falling back to PCA")
            reducer = PCA(n_components=2, random_state=random_state)
            all_2d = reducer.fit_transform(all_points)
    elif method_lower == "tsne":
        from sklearn.manifold import TSNE
        perp = min(30, len(all_points) - 1)
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perp)
        all_2d = reducer.fit_transform(all_points)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Split back
    n_bg = all_bg_points.shape[0]
    n_bg_means = bg_means.shape[0]
    n_rec_means = rec_means.shape[0]
    
    embeddings_2d = all_2d[:n_bg]
    rec_means_2d = all_2d[n_bg + n_bg_means:n_bg + n_bg_means + n_rec_means]
    bart_2d = all_2d[-1].reshape(1, -1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5.8, 5))
    
    # Plot background clouds
    start = 0
    for i, (topic, sz) in enumerate(zip(topics, group_sizes)):
        end = start + sz
        ax.scatter(
            embeddings_2d[start:end, 0],
            embeddings_2d[start:end, 1],
            color=color_map[topic],
            alpha=0.35,
            s=25,
        )
        start = end
    
    # Plot recalled means with arrows
    for i, mean in enumerate(rec_means_2d):
        topic = topics[i] if i < len(topics) else f"rec_{i}"
        ax.scatter(
            mean[0], mean[1],
            color=color_map.get(topic, colors[i]),
            marker="o", s=25, edgecolors="black",
            label=f"Recalled ({topic})",
        )
        ax.arrow(
            bart_2d[0, 0], bart_2d[0, 1],
            mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
            color="black", lw=0.5, length_includes_head=True, head_width=0.01,
        )
    
    # Plot original Bartlett
    ax.scatter(
        bart_2d[0, 0], bart_2d[0, 1],
        color="black", marker="o", s=25, edgecolors="black",
        label="Original story",
    )
    ax.legend(fontsize=10, ncol=1, loc="upper left", markerscale=2)
    
    # Create inset zoom
    if rec_means_2d.size > 0:
        xmin, ymin = np.min(rec_means_2d, axis=0)
        xmax, ymax = np.max(rec_means_2d, axis=0)
        xmin = min(xmin, bart_2d[0, 0])
        ymin = min(ymin, bart_2d[0, 1])
        xmax = max(xmax, bart_2d[0, 0])
        ymax = max(ymax, bart_2d[0, 1])
        
        w = max(xmax - xmin, 1e-6)
        h = max(ymax - ymin, 1e-6)
        margin = 0.15
        
        axins = ax.inset_axes(
            [0.02, 0.02, 0.27, 0.4],
            xlim=(xmin - margin * w, xmax + margin * w),
            ylim=(ymin - margin * h, ymax + margin * h),
            xticks=[], yticks=[],
        )
        
        # Replot in inset
        start = 0
        for i, (topic, sz) in enumerate(zip(topics, group_sizes)):
            end = start + sz
            axins.scatter(
                embeddings_2d[start:end, 0], embeddings_2d[start:end, 1],
                color=color_map[topic], alpha=0.25, s=130,
            )
            start = end
        
        for i, mean in enumerate(rec_means_2d):
            topic = topics[i] if i < len(topics) else f"rec_{i}"
            axins.scatter(
                mean[0], mean[1],
                color=color_map.get(topic, colors[i]),
                marker="o", s=130, edgecolors="black",
            )
            axins.arrow(
                bart_2d[0, 0], bart_2d[0, 1],
                mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
                color="black", lw=0.5, length_includes_head=True, head_width=0.015,
            )
        
        axins.scatter(
            bart_2d[0, 0], bart_2d[0, 1],
            color="black", marker="o", s=130, edgecolors="black", linewidth=2,
        )
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate PCA/UMAP/t-SNE plots from cached Bartlett data")
    parser.add_argument("--pkl_dir", type=str, 
                        default="../../narratives_gpt2/bartlett_data",
                        help="Directory with cached pickle files")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for plots")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="Temperature to use for recalled stories")
    parser.add_argument("--epoch", type=int, default=5,
                        help="Epoch to use for recalled stories")
    parser.add_argument(
        "--temps",
        type=str,
        default="",
        help="Comma-separated temps (overrides --temp), e.g. '0,0.5,1,1.5'.",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="",
        help="Comma-separated epochs (overrides --epoch), e.g. '1,3,5'.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pca,umap,tsne",
        help="Comma-separated: pca, umap, tsne.",
    )
    parser.add_argument("--n_bg", type=int, default=1000,
                        help="Number of background articles per topic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude_topics",
        type=str,
        default="",
        help="Comma-separated topic names to skip (e.g. 'Technology').",
    )
    args = parser.parse_args()
    
    pkl_dir = Path(args.pkl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_topics = {t.strip() for t in args.exclude_topics.split(",") if t.strip()}
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one of: pca, umap, tsne")
    for m in methods:
        if m not in {"pca", "umap", "tsne"}:
            raise ValueError(f"Unknown method in --methods: {m}")

    temps = [args.temp]
    if args.temps.strip():
        temps = [float(x.strip()) for x in args.temps.split(",") if x.strip()]

    epochs = [args.epoch]
    if args.epochs.strip():
        epochs = [int(x.strip()) for x in args.epochs.split(",") if x.strip()]
    
    # Load data
    print("Loading recalled stories from pickle files...")
    topics = ['Universe', 'Politics', 'Sport', 'Technology', 'Health', 'Nature']
    if exclude_topics:
        topics = [t for t in topics if t not in exclude_topics]
    df = load_recalled_stories(pkl_dir, categories=topics)
    print(f"  Loaded {len(df)} records")
    
    max_chars = len(BARTLETT_STORY)
    
    # Load background texts
    bg_texts = load_background_texts(topics, max_chars, args.n_bg, sample_seed=args.seed)
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    unique_texts = df["text"].dropna().unique().tolist() if len(df) else []
    text_to_emb = {}
    if unique_texts:
        print(f"Embedding recalled texts ({len(unique_texts)} unique)...")
        rec_embs = model.encode(unique_texts, show_progress_bar=False)
        text_to_emb = {t: e for t, e in zip(unique_texts, rec_embs)}
    
    # Embed everything
    print("Embedding texts...")
    bartlett_emb = model.encode([BARTLETT_STORY])[0]
    
    all_bg_embs = []
    bg_means = []
    group_sizes = []
    rec_means = []
    
    for topic in topics:
        # Background embeddings
        if bg_texts[topic]:
            bg_emb = embed_texts(model, bg_texts[topic])
            all_bg_embs.append(bg_emb)
            bg_means.append(np.mean(bg_emb, axis=0))
            group_sizes.append(len(bg_emb))
        else:
            print(f"  [Warning] No background texts for {topic}")
            group_sizes.append(0)
            bg_means.append(np.zeros(384))  # MiniLM dimension
        
        rec_means.append(np.zeros(384))
    
    all_bg_points = np.vstack(all_bg_embs) if all_bg_embs else np.zeros((0, 384))
    bg_means = np.vstack(bg_means)

    def recalled_mean_for(topic: str, temp: float, epoch: int) -> np.ndarray:
        rows = df[(df["topic"] == topic) & (df["temp"] == temp) & (df["epoch"] == epoch)] if len(df) else df
        texts = rows["text"].tolist() if len(rows) else []
        if not texts:
            return np.zeros(384)
        embs = [text_to_emb.get(t) for t in texts if t in text_to_emb]
        if not embs:
            return np.zeros(384)
        return np.mean(np.vstack(embs), axis=0)

    # Generate plots (single or grid)
    print("\nGenerating plots...")
    is_grid = len(temps) > 1 or len(epochs) > 1
    for epoch in epochs:
        for temp in temps:
            rec_means = np.vstack([recalled_mean_for(topic, temp, epoch) for topic in topics])
            if is_grid:
                combo_dir = output_dir / f"t{temp}_e{epoch}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Combo: temp={temp}, epoch={epoch} -> {combo_dir}")
                for method in methods:
                    out_path = combo_dir / f"embedding_2d_{method}.png"
                    plot_2d_embedding(
                        all_bg_points, bg_means, rec_means, bartlett_emb,
                        topics, group_sizes, out_path,
                        method=method, random_state=args.seed
                    )
            else:
                for method in methods:
                    out_path = output_dir / f"embedding_2d_{method}.png"
                    plot_2d_embedding(
                        all_bg_points, bg_means, rec_means, bartlett_emb,
                        topics, group_sizes, out_path,
                        method=method, random_state=args.seed
                    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
