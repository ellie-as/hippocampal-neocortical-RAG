#!/usr/bin/env python3
"""
Standalone plotting script for Raykov + Bartlett experiments.

Subcommands:
  raykov        -- Omission/extension bar plots + optional wordfreq analysis
  bartlett      -- Final-model sampling (or --skip_final for checkpoint new-word curves only)
  bartlett_ckpt -- Checkpoint-grid plots (epoch x temp): bars, wordclouds, PCA/UMAP/t-SNE

Prompt convention (matches training in bartlett_twostage.py):
  <s>[INST] One night two men from Egulac... What happened (in detail)? [/INST]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import re
import string as _string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as t_dist, chi2_contingency, fisher_exact
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from wordcloud import WordCloud

from utils import (
    BARTLETT_TXT,
    EXCLUDE_OFFENSIVE_WORDS,
    load_bartlett,
    load_topic_corpus_wiki,
    recall_prefix,
)

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
COLORS = ['#6a00a8', '#e16462', '#b12a90', '#0d0887', '#f0f921', '#fca636']

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _word_count(s: str) -> int:
    return len((s or "").strip().split())


def _t95_ci(x) -> float:
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 2:
        return 0.0
    se = x.std(ddof=1) / (n ** 0.5)
    return float(t_dist.ppf(0.975, df=n - 1) * se)


def _sem(x) -> float:
    x = np.asarray(x, dtype=float); n = len(x)
    return float(x.std(ddof=1) / (n ** 0.5)) if n >= 2 else 0.0


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
# Bartlett prompt (matches training format)
# --------------------------------------------------------------------------- #

def _make_prompt(bartlett_text: str) -> str:
    """Build recall prompt from the shortened Bartlett cue."""
    return f"<s>[INST] {recall_prefix()} What happened (in detail)? [/INST]"


def _first_sentence(bartlett_text: str) -> str:
    """Prompt cue used to prefix recalled text for embedding comparisons."""
    return recall_prefix()


# --------------------------------------------------------------------------- #
# Model loading & sampling
# --------------------------------------------------------------------------- #

def _ensure_pad(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _embedder() -> SentenceTransformer:
    try:
        from plot_config import embedding_model as _model_name
    except (ImportError, AttributeError):
        _model_name = "all-MiniLM-L6-v2"
    return SentenceTransformer(_model_name)


def _resolve_embed_trim() -> Optional[int]:
    """Resolve the embedding truncation limit from plot_config, matching collate_figures."""
    try:
        from plot_config import embed_trim_chars as _raw
    except (ImportError, AttributeError):
        return None
    if _raw is None:
        return None
    if isinstance(_raw, int):
        return _raw
    if isinstance(_raw, str) and _raw.lower() == "bartlett":
        return len(load_bartlett())
    return int(_raw)


_EMBED_TRIM: Optional[int] = _resolve_embed_trim()


def _embed_texts(st: SentenceTransformer, texts: List[str], trim_to: Optional[int] = None) -> np.ndarray:
    if trim_to is None:
        trim_to = _EMBED_TRIM
    if trim_to is not None:
        texts = [t[:trim_to] for t in texts]
    return st.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def _cosdist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b))


def _mean_vec(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if len(x) == 0:
        return np.zeros((x.shape[1],), dtype=np.float32)
    return x.mean(axis=0)


def _load_final_model_4bit(topic_dir: Path, offload_root: Path, torch_dtype=None):
    final_dir = topic_dir / "model" / "final"
    if not final_dir.exists():
        raise FileNotFoundError(f"No final model found at {final_dir}")
    cfg_path = final_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"adapter_config.json missing in {final_dir}")
    base_name = json.loads(cfg_path.read_text())["base_model_name_or_path"]

    offload_folder = offload_root / f"_offload_{topic_dir.name}"
    offload_folder.mkdir(parents=True, exist_ok=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=quant_cfg, device_map="auto",
        offload_folder=str(offload_folder), low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    tok = _ensure_pad(tok)
    model = PeftModel.from_pretrained(base, str(final_dir), device_map="auto",
                                      offload_folder=str(offload_folder))
    model.eval()
    return model, tok


def _find_checkpoints(topic_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted list of (epoch, checkpoint_path)."""
    from math import gcd
    from functools import reduce

    checkpoints = []
    stage2_root = topic_dir / "stage2_bartlett"
    if stage2_root.exists():
        for ckpt_dir in stage2_root.glob("checkpoint-*"):
            if ckpt_dir.is_dir() and (ckpt_dir / "adapter_config.json").exists():
                try:
                    step = int(ckpt_dir.name.split("-")[1])
                    checkpoints.append((step, ckpt_dir))
                except (ValueError, IndexError):
                    continue
    if not checkpoints:
        for ckpt_dir in topic_dir.glob("checkpoint-*"):
            if ckpt_dir.is_dir() and (ckpt_dir / "adapter_config.json").exists():
                try:
                    step = int(ckpt_dir.name.split("-")[1])
                    checkpoints.append((step, ckpt_dir))
                except (ValueError, IndexError):
                    continue

    checkpoints = sorted(checkpoints, key=lambda x: x[0])
    if checkpoints:
        steps = [s for s, _ in checkpoints]
        spe = reduce(gcd, steps) if len(steps) >= 2 else steps[0]
        checkpoints = [(s // spe, p) for s, p in checkpoints]

    final_dir = topic_dir / "model" / "final"
    if not checkpoints and final_dir.exists() and (final_dir / "adapter_config.json").exists():
        checkpoints.append((1, final_dir))
    return checkpoints


def _load_checkpoint_model(checkpoint_path: Path, offload_root: Path, torch_dtype=None):
    cfg_path = checkpoint_path / "adapter_config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"adapter_config.json missing in {checkpoint_path}")
    base_name = json.loads(cfg_path.read_text())["base_model_name_or_path"]

    offload_folder = offload_root / f"_offload_{checkpoint_path.parent.name}"
    offload_folder.mkdir(parents=True, exist_ok=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=quant_cfg, device_map="auto",
        offload_folder=str(offload_folder), low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    tok = _ensure_pad(tok)
    model = PeftModel.from_pretrained(base, str(checkpoint_path), device_map="auto",
                                      offload_folder=str(offload_folder))
    model.eval()
    return model, tok


def _resolve_auto_min_tokens(
    tok, bartlett_text: str, min_new_tokens: int, max_new_tokens: int, prompt: str | None = None,
) -> tuple:
    """Resolve ``-1`` (auto) to the prompt-conditioned Bartlett token count.

    Returns ``(resolved_min, resolved_max)``.
    """
    if min_new_tokens != -1:
        return max(0, min_new_tokens), max_new_tokens
    if prompt:
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tok(f"{prompt} {bartlett_text}", add_special_tokens=False)["input_ids"]
        n = len(target_ids) - len(prompt_ids)
    else:
        ids = tok(bartlett_text, add_special_tokens=False)["input_ids"]
        n = len(ids) if isinstance(ids, list) else int(ids.shape[-1])
    n = max(0, n)
    print(f"  [length guard] Auto min/max_new_tokens = {n} (from prompt-conditioned Bartlett token count)")
    return n, n


@torch.no_grad()
def _sample_n(model, tok, prompt: str, n: int, temp: float, max_new: int,
              batch_size: int = 100, *, min_new_tokens: int = 0) -> List[str]:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    seq_len = input_ids.shape[1]

    texts: List[str] = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        in_ids = input_ids.expand(b, -1).contiguous()
        attn = attention_mask.expand(b, -1).contiguous() if attention_mask is not None else None
        gen_kwargs = dict(
                input_ids=in_ids, attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=(temp > 0), temperature=temp if temp > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id, use_cache=True,
            )
        if min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = int(min_new_tokens)
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        new_tokens = out[:, seq_len:]
        texts.extend(t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True))
        del out, in_ids, attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return texts


def _load_epoch_logs_greedy(topic_dir: Path) -> Dict[int, str]:
    result = {}
    log_path = topic_dir / "stage2_bartlett" / "epoch_logs.json"
    if log_path.exists():
        try:
            logs = json.loads(log_path.read_text())
            if isinstance(logs, list):
                for entry in logs:
                    if isinstance(entry, dict) and "epoch" in entry and "greedy" in entry:
                        result[int(entry["epoch"])] = entry["greedy"].strip()
        except Exception:
            pass
    return result


# --------------------------------------------------------------------------- #
# Wordcloud / histogram helpers
# --------------------------------------------------------------------------- #

WORDCLOUD_RENDER_SCALE = 2
WORDCLOUD_MAX_FONT_SIZE = 54
WORDCLOUD_MIN_FONT_SIZE = 8
WORDCLOUD_RELATIVE_SCALING = 0.30


def _exclusion_from_bartlett(txt: str) -> set:
    trans = str.maketrans(_string.punctuation, ' ' * len(_string.punctuation))
    tokens = txt.translate(trans).lower().split()
    return set(tokens + ['s']) | EXCLUDE_OFFENSIVE_WORDS


def _pre_wcloud(text: str, excl: set) -> str:
    trans = str.maketrans(_string.punctuation, ' ' * len(_string.punctuation))
    tokens = text.translate(trans).lower().split()
    return ' '.join(w for w in tokens if w not in excl)


def _wordcloud(joined_text: str, excl: set, out_path: Path, title: str):
    proc = _pre_wcloud(joined_text, excl)
    if not proc.strip():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No intrusion words", ha='center', va='center')
        ax.axis('off'); ax.set_title(title)
        fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)
        return
    wc = WordCloud(width=400, height=400, scale=WORDCLOUD_RENDER_SCALE,
                   relative_scaling=WORDCLOUD_RELATIVE_SCALING, normalize_plurals=True,
                   max_font_size=WORDCLOUD_MAX_FONT_SIZE,
                   min_font_size=WORDCLOUD_MIN_FONT_SIZE,
                   background_color='white', colormap='plasma').generate(proc)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)


def _topic_hist(cos_dists: List[float], bart_to_bg: float, out_path: Path, title: str):
    plt.figure(figsize=(5, 4))
    plt.hist(cos_dists, bins=12, alpha=0.85, edgecolor='black')
    plt.axvline(bart_to_bg, linestyle='--', linewidth=2, label=f"Bartlett→BG = {bart_to_bg:.3f}")
    plt.xlabel("Cosine distance to background centroid")
    plt.ylabel("Count"); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()


def _paired_bars(means_left, means_right, sem_left, sem_right,
                 xlabels, left_label, right_label, ylabel, title, out_path, *, ylim=None, hline0=False):
    x = np.arange(len(xlabels)); width = 0.35
    if sem_left is None:
        sem_left = np.zeros_like(means_left, dtype=float)
    if sem_right is None:
        sem_right = np.zeros_like(means_right, dtype=float)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(x - width / 2, means_left, width, yerr=sem_left, capsize=5, alpha=0.6,
           label=left_label, color=COLORS[0])
    ax.bar(x + width / 2, means_right, width, yerr=sem_right, capsize=5, alpha=0.6,
           label=right_label, color=COLORS[5])
    if hline0:
        ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower center", ncols=2)
    fig.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)


def _line_with_sem(x_vals, means, sems, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.errorbar(x_vals, means, yerr=sems, fmt='-o', linewidth=2, markersize=5, capsize=4, color=COLORS[0])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] Saved {out_path}")


# --------------------------------------------------------------------------- #
# New-words helpers
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-zA-Z']+")
_DEFAULT_STOP = frozenset({
    "a","an","the","and","or","but","if","then","else","when","while","for","to","from",
    "of","in","on","at","by","with","about","as","into","like","through","after","over",
    "between","out","against","during","without","before","under","around","among","is",
    "am","are","was","were","be","been","being","do","does","did","done","doing","have",
    "has","had","having","will","would","can","could","should","may","might","must","i",
    "you","he","she","it","we","they","me","him","her","us","them","my","your","his",
    "its","our","their","mine","yours","hers","ours","theirs","this","that","these",
    "those","there","here","who","whom","which","what","where","why","how","not","no",
    "so","too","very","just","also","than","such",
})


def _tok(texts: Iterable[str], lower=True, min_len=2, remove_stop=True) -> List[str]:
    stop = set(_DEFAULT_STOP)
    out: List[str] = []
    for t in texts:
        if not isinstance(t, str):
            continue
        if lower:
            t = t.lower()
        for w in _WORD_RE.findall(t):
            w = w.strip("'")
            if len(w) < min_len:
                continue
            if remove_stop and w in stop:
                continue
            if w:
                out.append(w)
    return out


def _tok1(text: str, **kw) -> List[str]:
    return _tok([text], **kw)


def _frac_new_words(original_text: str, generated_text: str, min_len: int = 2) -> float:
    orig_set = set(_tok1(original_text or "", lower=True, min_len=min_len, remove_stop=True))
    gen_tokens = _tok1(generated_text or "", lower=True, min_len=min_len, remove_stop=True)
    denom = len(gen_tokens)
    if denom == 0:
        return float("nan")
    new_count = sum(1 for w in gen_tokens if w not in orig_set)
    return new_count / denom


# --------------------------------------------------------------------------- #
# Checkpoint sampling for new-words curves
# --------------------------------------------------------------------------- #

def _sample_checkpoints_with_temps(
    topic_dir: Path, prompt: str, temps: List[float],
    offload_root: Path, cache_dir: Path,
    n_samples: int = 10, max_new_tokens: int = 500,
    min_new_tokens: int = -1,
    bartlett_text: str = "",
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{topic_dir.name}_checkpoint_samples.json"

    cached = {}
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            print(f"  [Cache] Loaded {len(cached)} cached entries from {cache_file.name}")
        except Exception:
            cached = {}

    greedy_from_training = _load_epoch_logs_greedy(topic_dir)
    if greedy_from_training:
        print(f"  [Cache] Found {len(greedy_from_training)} greedy samples from training logs")

    checkpoints = _find_checkpoints(topic_dir)
    if not checkpoints:
        print(f"  [Warning] No checkpoints found for {topic_dir.name}")
        return pd.DataFrame(columns=["epoch", "temp", "text"])

    print(f"  Found {len(checkpoints)} epoch checkpoints for {topic_dir.name}")
    rows = []
    model, tok = None, None

    for epoch, ckpt_path in checkpoints:
        for temp in temps:
            cache_key = f"{epoch}_{temp}"
            if cache_key in cached:
                for txt in cached[cache_key]:
                    rows.append({"epoch": epoch, "temp": temp, "text": txt})
            elif temp == 0.0 and epoch in greedy_from_training:
                txt = greedy_from_training[epoch]
                rows.append({"epoch": epoch, "temp": temp, "text": txt})
                cached[cache_key] = [txt]
            else:
                if model is None or str(ckpt_path) != getattr(model, '_loaded_from', ''):
                    if model is not None:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    print(f"    Loading checkpoint {ckpt_path.name}...")
                    model, tok = _load_checkpoint_model(ckpt_path, offload_root)
                    model._loaded_from = str(ckpt_path)
                    if min_new_tokens == -1 and bartlett_text:
                        min_new_tokens, max_new_tokens = _resolve_auto_min_tokens(
                            tok, bartlett_text, min_new_tokens, max_new_tokens, prompt)

                print(f"    Sampling epoch={epoch}, temp={temp}...")
                samples = _sample_n(model, tok, prompt, n=n_samples, temp=temp, max_new=max_new_tokens,
                                    min_new_tokens=min_new_tokens)
                cached[cache_key] = samples
                for txt in samples:
                    rows.append({"epoch": epoch, "temp": temp, "text": txt})

    cache_file.write_text(json.dumps(cached, indent=2))
    print(f"  [Cache] Saved to {cache_file}")

    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return pd.DataFrame(rows)


def _build_newword_curves_from_checkpoints(
    results_root: Path, original_text: str, prompt: str, temps: List[float],
    offload_root: Path, cache_dir: Path,
    n_samples: int = 10, max_new_tokens: int = 500, min_len: int = 2,
    epoch_fixed_for_temp: int = 5, temp_fixed_for_epoch: float = 0.5,
    min_new_tokens: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for topic_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if topic_dir.name.startswith("_"):
            continue
        print(f"\n[NewWords] Processing {topic_dir.name}...")
        df_samples = _sample_checkpoints_with_temps(
            topic_dir, prompt, temps, offload_root, cache_dir,
            n_samples=n_samples, max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            bartlett_text=original_text,
        )
        if df_samples.empty:
            continue
        for _, row in df_samples.iterrows():
            frac = _frac_new_words(original_text, row["text"], min_len=min_len)
            if np.isfinite(frac):
                all_rows.append({
                    "topic": topic_dir.name, "epoch": int(row["epoch"]),
                    "temp": float(row["temp"]), "frac_new_words": float(frac),
                })

    if not all_rows:
        return (pd.DataFrame(columns=["epoch", "mean", "sem", "n"]),
                pd.DataFrame(columns=["temp", "mean", "sem", "n"]))

    df = pd.DataFrame(all_rows)

    # Curve A: frac vs epoch (temp fixed)
    df_a = df[np.isclose(df["temp"], float(temp_fixed_for_epoch))]
    if not df_a.empty:
        g = df_a.groupby("epoch")["frac_new_words"]
        df_epoch = pd.DataFrame({
            "epoch": g.mean().index.astype(int), "mean": g.mean().values.astype(float),
            "sem": g.std(ddof=1).div(np.sqrt(g.count())).fillna(0.0).values.astype(float),
            "n": g.count().values.astype(int),
        }).sort_values("epoch")
    else:
        df_epoch = pd.DataFrame(columns=["epoch", "mean", "sem", "n"])

    # Curve B: frac vs temp (epoch fixed)
    df_b = df[df["epoch"] == int(epoch_fixed_for_temp)]
    if not df_b.empty:
        g2 = df_b.groupby("temp")["frac_new_words"]
        df_temp = pd.DataFrame({
            "temp": g2.mean().index.astype(float), "mean": g2.mean().values.astype(float),
            "sem": g2.std(ddof=1).div(np.sqrt(g2.count())).fillna(0.0).values.astype(float),
            "n": g2.count().values.astype(int),
        }).sort_values("temp")
    else:
        df_temp = pd.DataFrame(columns=["temp", "mean", "sem", "n"])

    return df_epoch, df_temp


# --------------------------------------------------------------------------- #
# 2-D embedding plot (PCA / UMAP / t-SNE)
# --------------------------------------------------------------------------- #

def _plot_2d_embedding(
    all_bg_points, bg_means, rec_means, bart_emb,
    labels_bg, labels_rec, group_sizes, out_png,
    method="pca", random_state=42, *,
    rec_points=None, rec_group_sizes=None,
):
    from sklearn.decomposition import PCA

    def _topic_from_bg(s):
        return re.sub(r"\s*data\s*$", "", str(s)).strip()

    def _topic_from_rec(s):
        m = re.search(r"\((.*?)\)", str(s))
        return m.group(1).strip() if m else str(s).strip()

    topics_bg = [_topic_from_bg(x) for x in labels_bg]
    topics_rec = [_topic_from_rec(x) for x in labels_rec]
    num_bg = len(topics_bg)
    colors = COLORS * ((num_bg // len(COLORS)) + 1)
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(topics_bg)}

    rec_points = rec_points if rec_points is not None and len(rec_points) else None
    stack = [all_bg_points]
    if rec_points is not None:
        stack.append(rec_points)
    stack.extend([bg_means, rec_means, bart_emb.reshape(1, -1)])
    all_points = np.vstack(stack)

    method_lower = method.lower()
    if method_lower == "pca":
        all_2d = PCA(n_components=2, random_state=random_state).fit_transform(all_points)
    elif method_lower == "umap":
        try:
            from umap import UMAP
            all_2d = UMAP(n_components=2, random_state=random_state, n_neighbors=15,
                          min_dist=0.1).fit_transform(all_points)
        except ImportError:
            print("[Warning] umap-learn not installed, falling back to PCA")
            all_2d = PCA(n_components=2, random_state=random_state).fit_transform(all_points)
    elif method_lower == "tsne":
        try:
            from sklearn.manifold import TSNE
            all_2d = TSNE(n_components=2, random_state=random_state,
                          perplexity=min(30, len(all_points) - 1)).fit_transform(all_points)
        except ImportError:
            all_2d = PCA(n_components=2, random_state=random_state).fit_transform(all_points)
    else:
        raise ValueError(f"Unknown method: {method}")

    n_bg = all_bg_points.shape[0]
    n_rec = rec_points.shape[0] if rec_points is not None else 0
    n_bg_means = bg_means.shape[0]
    n_rec_means = rec_means.shape[0]

    embeddings_2d = all_2d[:n_bg]
    rec_points_2d = all_2d[n_bg:n_bg + n_rec] if n_rec else None
    rec_means_2d = all_2d[n_bg + n_rec + n_bg_means:n_bg + n_rec + n_bg_means + n_rec_means]
    bart_2d = all_2d[-1].reshape(1, -1)

    fig, ax = plt.subplots(figsize=(5.8, 5))
    start = 0
    for i, (topic, sz) in enumerate(zip(topics_bg, group_sizes[:num_bg])):
        end = start + sz
        ax.scatter(embeddings_2d[start:end, 0], embeddings_2d[start:end, 1],
                   color=color_map.get(topic, colors[i]), alpha=0.35, s=25)
        start = end

    if rec_points_2d is not None and rec_group_sizes:
        start = 0
        for i, (topic, sz) in enumerate(zip(topics_bg, rec_group_sizes[:num_bg])):
            end = start + sz
            ax.scatter(rec_points_2d[start:end, 0], rec_points_2d[start:end, 1],
                       color=color_map.get(topic, colors[i]), alpha=0.10, s=10)
            start = end

    for i, mean in enumerate(rec_means_2d):
        topic = topics_rec[i] if i < len(topics_rec) else f"rec_{i}"
        col = color_map.get(topic, colors[i % len(colors)])
        ax.scatter(mean[0], mean[1], color=col, marker="o", s=25, edgecolors="black",
                   label=f"Recalled ({topic})")
        ax.arrow(bart_2d[0, 0], bart_2d[0, 1], mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
                 color="black", lw=0.5, length_includes_head=True, head_width=0.01)

    ax.scatter(bart_2d[0, 0], bart_2d[0, 1], color="black", marker="o", s=25,
               edgecolors="black", label="Original story")
    ax.legend(fontsize=10, ncol=1, loc="upper left", markerscale=2)

    if rec_means_2d.size > 0:
        if rec_points_2d is not None and rec_points_2d.shape[0] > 0:
            xmin, xmax = np.quantile(rec_points_2d[:, 0], [0.05, 0.95])
            ymin, ymax = np.quantile(rec_points_2d[:, 1], [0.05, 0.95])
        else:
            xmin, ymin = np.min(rec_means_2d, axis=0)
            xmax, ymax = np.max(rec_means_2d, axis=0)
        xmin = float(min(xmin, bart_2d[0, 0])); ymin = float(min(ymin, bart_2d[0, 1]))
        xmax = float(max(xmax, bart_2d[0, 0])); ymax = float(max(ymax, bart_2d[0, 1]))
        w = max(xmax - xmin, 1e-6); h = max(ymax - ymin, 1e-6); m = 0.15
        axins = ax.inset_axes([0.02, 0.02, 0.27, 0.4],
                              xlim=(xmin - m * w, xmax + m * w),
                              ylim=(ymin - m * h, ymax + m * h), xticks=[], yticks=[])
        start = 0
        for i, (topic, sz) in enumerate(zip(topics_bg, group_sizes[:num_bg])):
            end = start + sz
            axins.scatter(embeddings_2d[start:end, 0], embeddings_2d[start:end, 1],
                          color=color_map.get(topic, colors[i]), alpha=0.25, s=130)
            start = end
        if rec_points_2d is not None and rec_group_sizes:
            start = 0
            for i, (topic, sz) in enumerate(zip(topics_bg, rec_group_sizes[:num_bg])):
                end = start + sz
                axins.scatter(rec_points_2d[start:end, 0], rec_points_2d[start:end, 1],
                              color=color_map.get(topic, colors[i]), alpha=0.08, s=120)
                start = end
        for i, mean in enumerate(rec_means_2d):
            topic = topics_rec[i] if i < len(topics_rec) else f"rec_{i}"
            col = color_map.get(topic, colors[i % len(colors)])
            axins.scatter(mean[0], mean[1], color=col, marker="o", s=130, edgecolors="black")
            axins.arrow(bart_2d[0, 0], bart_2d[0, 1], mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
                        color="black", lw=0.5, length_includes_head=True, head_width=0.015)
        axins.scatter(bart_2d[0, 0], bart_2d[0, 1], color="black", marker="o", s=130,
                      edgecolors="black", linewidth=2)
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1)

    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"[Plot] Saved {out_png}")


def _plot_ckpt_pca_like_collate(
    *,
    all_bg_points: np.ndarray,
    group_sizes_bg: List[int],
    topics: List[str],
    rec_means: np.ndarray,
    bart_emb: np.ndarray,
    out_png: Path,
    random_state: int = 42,
) -> None:
    """
    Checkpoint PCA plot formatted like collate_figures panel a:
      - Fit PCA on background points only
      - Plot only: background cloud + recalled means + original Bartlett point
      - Recalled samples are NOT plotted (and are not used in PCA fit); they
        should already be folded into `rec_means` upstream.
    """
    from sklearn.decomposition import PCA

    if all_bg_points is None or getattr(all_bg_points, "size", 0) == 0:
        raise ValueError("No background points for PCA.")

    if rec_means is None:
        rec_means = np.zeros((0, all_bg_points.shape[1]), dtype=float)

    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(all_bg_points)

    bg_2d = pca.transform(all_bg_points)
    rec_means_2d = pca.transform(rec_means) if rec_means.size else np.zeros((0, 2))
    bart_2d = pca.transform(bart_emb.reshape(1, -1))

    num_topics = len(topics)
    colors = COLORS * ((max(num_topics, 1) // len(COLORS)) + 1)
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(topics)}

    fig, ax = plt.subplots(figsize=(5.8, 5))

    # Background clouds (one color per topic)
    start = 0
    for i, (topic, sz) in enumerate(zip(topics, group_sizes_bg)):
        end = start + int(sz)
        if end > start:
            ax.scatter(
                bg_2d[start:end, 0],
                bg_2d[start:end, 1],
                color=color_map.get(topic, colors[i]),
                alpha=0.35,
                s=25,
            )
        start = end

    # Recalled means + arrows from Bartlett
    for i, mean in enumerate(rec_means_2d):
        topic = topics[i] if i < len(topics) else f"rec_{i}"
        col = color_map.get(topic, colors[i % len(colors)])
        ax.scatter(mean[0], mean[1], color=col, marker="o", s=25, edgecolors="black",
                   label=f"Recalled ({topic})")
        ax.arrow(
            bart_2d[0, 0],
            bart_2d[0, 1],
            mean[0] - bart_2d[0, 0],
            mean[1] - bart_2d[0, 1],
            color="black",
            lw=0.5,
            length_includes_head=True,
            head_width=0.01,
        )

    # Original Bartlett
    ax.scatter(bart_2d[0, 0], bart_2d[0, 1], color="black", marker="o", s=25,
               edgecolors="black", label="Original story")
    ax.legend(fontsize=10, ncol=1, loc="upper left", markerscale=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", bottom=False, left=False)

    # Inset zoom around recalled means (and Bartlett), replotting only bg + means.
    if rec_means_2d.size > 0:
        xmin, ymin = np.min(rec_means_2d, axis=0)
        xmax, ymax = np.max(rec_means_2d, axis=0)
        xmin = float(min(xmin, bart_2d[0, 0])); ymin = float(min(ymin, bart_2d[0, 1]))
        xmax = float(max(xmax, bart_2d[0, 0])); ymax = float(max(ymax, bart_2d[0, 1]))
        w = max(xmax - xmin, 1e-6); h = max(ymax - ymin, 1e-6); m = 0.15

        axins = ax.inset_axes(
            [0.02, 0.02, 0.27, 0.4],
            xlim=(xmin - m * w, xmax + m * w),
            ylim=(ymin - m * h, ymax + m * h),
            xticks=[],
            yticks=[],
        )

        start = 0
        for i, (topic, sz) in enumerate(zip(topics, group_sizes_bg)):
            end = start + int(sz)
            if end > start:
                axins.scatter(
                    bg_2d[start:end, 0],
                    bg_2d[start:end, 1],
                    color=color_map.get(topic, colors[i]),
                    alpha=0.25,
                    s=90,
                )
            start = end

        for i, mean in enumerate(rec_means_2d):
            topic = topics[i] if i < len(topics) else f"rec_{i}"
            col = color_map.get(topic, colors[i % len(colors)])
            axins.scatter(mean[0], mean[1], color=col, marker="o", s=90, edgecolors="black")
            axins.arrow(
                bart_2d[0, 0],
                bart_2d[0, 1],
                mean[0] - bart_2d[0, 0],
                mean[1] - bart_2d[0, 1],
                color="black",
                lw=0.5,
                length_includes_head=True,
                head_width=0.01,
            )

        axins.scatter(
            bart_2d[0, 0],
            bart_2d[0, 1],
            color="black",
            marker="o",
            s=90,
            edgecolors="black",
            linewidth=2,
        )
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved {out_png}")


# --------------------------------------------------------------------------- #
# PCA inset plot (legacy style – recalled means/original are larger than bg)
# --------------------------------------------------------------------------- #

def _pca_inset(
    all_bg_points, bg_means, rec_means, bart_emb,
    labels_bg, labels_rec, group_sizes, out_png, *,
    rec_points=None, rec_group_sizes=None,
):
    """PCA plot with inset zoom.  Recalled means (s=45) and original (s=40)
    are drawn larger than background points (s=25) so they stand out."""
    from sklearn.decomposition import PCA

    def _topic_from_bg(s):
        return re.sub(r"\s*data\s*$", "", str(s)).strip()

    def _topic_from_rec(s):
        m = re.search(r"\((.*?)\)", str(s))
        return m.group(1).strip() if m else str(s).strip()

    topics_bg = [_topic_from_bg(x) for x in labels_bg]
    topics_rec = [_topic_from_rec(x) for x in labels_rec]
    num_bg = len(topics_bg)
    color_map = {t: f"C{i % 10}" for i, t in enumerate(topics_bg)}

    rec_points = rec_points if rec_points is not None and len(rec_points) else None

    fit_parts = [all_bg_points]
    if rec_points is not None:
        fit_parts.append(rec_points)
    fit_parts.extend([bart_emb.reshape(1, -1), rec_means])

    pca_fit = PCA(n_components=2, random_state=123).fit(np.vstack(fit_parts))
    embeddings_2d = pca_fit.transform(all_bg_points)
    rec_points_2d = pca_fit.transform(rec_points) if rec_points is not None else None
    reduced_means = pca_fit.transform(np.vstack([bg_means, rec_means]))
    rec_means_2d = reduced_means[len(bg_means):]
    bart_2d = pca_fit.transform(bart_emb.reshape(1, -1))

    fig, ax = plt.subplots(figsize=(5.8, 5))

    # Background clouds
    start = 0
    for i, (topic, sz) in enumerate(zip(topics_bg, group_sizes[:num_bg])):
        end = start + sz
        col = color_map.get(topic, f"C{i % 10}")
        ax.scatter(embeddings_2d[start:end, 0], embeddings_2d[start:end, 1],
                   alpha=0.35, s=25, color=col, label=None)
        start = end

    # Recalled sample clouds (if provided)
    if rec_points_2d is not None and rec_group_sizes:
        start = 0
        for i, (topic, sz) in enumerate(zip(topics_bg, rec_group_sizes[:num_bg])):
            end = start + sz
            col = color_map.get(topic, f"C{i % 10}")
            ax.scatter(rec_points_2d[start:end, 0], rec_points_2d[start:end, 1],
                       alpha=0.12, s=10, color=col, label=None)
            start = end

    # Recalled means – larger (s=45) with black edge
    for i, mean in enumerate(rec_means_2d):
        topic = topics_rec[i] if i < len(topics_rec) else f"rec_{i}"
        col = color_map.get(topic, f"C{i % 10}")
        ax.scatter(mean[0], mean[1], marker="o", s=45, edgecolors="black",
                   linewidths=0.8, color=col, label=f"Recalled ({topic})")
        ax.arrow(bart_2d[0, 0], bart_2d[0, 1],
                 mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
                 color="black", lw=0.6, length_includes_head=True, head_width=0.02)

    # Original story – slightly larger (s=40)
    ax.scatter(bart_2d[0, 0], bart_2d[0, 1], color="black", marker="o", s=40,
               edgecolors="black", label="Original story")
    ax.legend(fontsize=9, ncol=1, loc="upper left", markerscale=1.4)

    # Inset zoom (fixed 0.32 x 0.32)
    if rec_means_2d.size > 0:
        if rec_points_2d is not None and rec_points_2d.shape[0] > 0:
            xmin, xmax = np.quantile(rec_points_2d[:, 0], [0.05, 0.95])
            ymin, ymax = np.quantile(rec_points_2d[:, 1], [0.05, 0.95])
        else:
            xmin, ymin = np.min(rec_means_2d, axis=0)
            xmax, ymax = np.max(rec_means_2d, axis=0)

        xmin = float(min(xmin, bart_2d[0, 0]))
        ymin = float(min(ymin, bart_2d[0, 1]))
        xmax = float(max(xmax, bart_2d[0, 0]))
        ymax = float(max(ymax, bart_2d[0, 1]))
        w = max(xmax - xmin, 1e-6); h = max(ymax - ymin, 1e-6)
        m = 0.08

        axins = ax.inset_axes([0.02, 0.02, 0.32, 0.32],
                              xlim=(xmin - m * w, xmax + m * w),
                              ylim=(ymin - m * h, ymax + m * h),
                              xticks=[], yticks=[])

        # Replot background in inset
        start = 0
        for i, (topic, sz) in enumerate(zip(topics_bg, group_sizes[:num_bg])):
            end = start + sz
            col = color_map.get(topic, f"C{i % 10}")
            axins.scatter(embeddings_2d[start:end, 0], embeddings_2d[start:end, 1],
                          alpha=0.22, s=120, color=col)
            start = end

        # Replot recalled samples in inset
        if rec_points_2d is not None and rec_group_sizes:
            start = 0
            for i, (topic, sz) in enumerate(zip(topics_bg, rec_group_sizes[:num_bg])):
                end = start + sz
                col = color_map.get(topic, f"C{i % 10}")
                axins.scatter(rec_points_2d[start:end, 0], rec_points_2d[start:end, 1],
                              alpha=0.10, s=80, color=col)
                start = end

        # Replot recalled means in inset
        for i, mean in enumerate(rec_means_2d):
            topic = topics_rec[i] if i < len(topics_rec) else f"rec_{i}"
            col = color_map.get(topic, f"C{i % 10}")
            axins.scatter(mean[0], mean[1], marker="o", s=120, edgecolors="black",
                          linewidths=1.0, color=col)
            axins.arrow(bart_2d[0, 0], bart_2d[0, 1],
                        mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
                        color="black", lw=0.5, length_includes_head=True, head_width=0.003)

        axins.scatter(bart_2d[0, 0], bart_2d[0, 1], color="black", marker="o",
                      s=120, edgecolors="black", linewidth=2)
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved {out_png}")


# =========================================================================== #
# RAYKOV SUBCOMMAND
# =========================================================================== #

def _raykov_load_story_map(datadir: Path) -> Dict[str, str]:
    pkl_path = datadir / "stories_prepared.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Run the simulation first.")
    sets = pickle.load(open(pkl_path, "rb"))
    story_map: Dict[str, str] = {}
    for cat in ("typical", "incomplete", "updated"):
        for i, s in enumerate(sets.get(cat, [])):
            story_map[f"{cat}_{i:04d}"] = s
    return story_map


def _raykov_extract_pre_text(row):
    return row.get("generation", "")


def _raykov_extract_post_text(row):
    if "generations" in row and isinstance(row["generations"], dict):
        gens = row["generations"]
        if "0.0" in gens and isinstance(gens["0.0"], str):
            return gens["0.0"]
        for k in sorted(gens.keys()):
            if isinstance(gens[k], str):
                return gens[k]
    return row.get("generation", "")


def _raykov_compute_diffs(rows, story_map, mode):
    recs = []
    for r in rows:
        rid = r.get("id"); cat = r.get("category", "unknown")
        if mode == "post":
            input_text = r.get("input_text") or story_map.get(rid, "")
            out_text = _raykov_extract_post_text(r)
        else:
            input_text = story_map.get(rid, "")
            out_text = _raykov_extract_pre_text(r)
        recs.append({"id": rid, "category": cat, "in_len": _word_count(input_text),
                      "out_len": _word_count(out_text),
                      "diff": _word_count(out_text) - _word_count(input_text)})
    return pd.DataFrame(recs)


def _raykov_summarize_om_ex(df):
    recs = []
    for _, r in df.iterrows():
        d = float(r["diff"])
        recs.append({"id": r["id"], "category": r["category"],
                      "shorter": 1.0 if d < 0 else 0.0, "longer": 1.0 if d > 0 else 0.0, "diff": d})
    return pd.DataFrame(recs)


def _raykov_plot_om_ex(df, title, out_path):
    cats = ["incomplete", "updated"]; labels = ["Omission errors", "Extension errors"]
    data = {}
    for c in cats:
        sub = df[df["category"] == c]
        data[c] = {"om": sub["shorter"].values, "ex": sub["longer"].values}
    x = np.arange(len(labels)); bar_w = 0.35

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for c, offset, label in [("incomplete", -bar_w / 2, "Incomplete"), ("updated", bar_w / 2, "Updated")]:
        col = "skyblue" if c == "incomplete" else "purple"
        alpha = 1.0 if c == "incomplete" else 0.6
        y = np.array([data[c]["om"].mean() if len(data[c]["om"]) else 0,
                       data[c]["ex"].mean() if len(data[c]["ex"]) else 0])
        e = np.array([_t95_ci(data[c]["om"]) if len(data[c]["om"]) else 0,
                       _t95_ci(data[c]["ex"]) if len(data[c]["ex"]) else 0])
        ax.bar(x + offset, y, bar_w, yerr=e, capsize=5, color=col, alpha=alpha, label=label)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion of errors"); ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=500, bbox_inches="tight"); plt.close()


def _raykov_plot_len_diff(df_pre, df_post, out_path):
    cats = ["incomplete", "typical", "updated"]; labs = ["Incomplete", "Typical", "Updated"]
    pre = [df_pre[df_pre["category"] == c]["diff"].values for c in cats]
    post = [df_post[df_post["category"] == c]["diff"].values for c in cats]
    means_pre = [np.mean(v) if len(v) else 0.0 for v in pre]
    sems_pre = [_sem(v) for v in pre]
    means_post = [np.mean(v) if len(v) else 0.0 for v in post]
    sems_post = [_sem(v) for v in post]

    x = np.arange(len(labs)); bar_w = 0.35
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.bar(x - bar_w / 2, means_pre, bar_w, yerr=sems_pre, capsize=5, alpha=0.55, label="Pre (short delay)")
    ax.bar(x + bar_w / 2, means_post, bar_w, yerr=sems_post, capsize=5, alpha=0.75, label="Post (long delay)")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("Length difference (output - input)")
    ax.set_title("Length differences: pre vs post consolidation")
    ax.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=500, bbox_inches="tight"); plt.close()


def run_raykov_main(rargs):
    outdir = Path(rargs.raykov_dir)
    datadir = outdir / "data"
    plotsdir = outdir / "plots"; plotsdir.mkdir(parents=True, exist_ok=True)
    pre_path = datadir / "generations_pre.json"
    post_path = datadir / "generations_post.json"
    if not pre_path.exists():
        raise FileNotFoundError(f"Missing {pre_path}")
    if not post_path.exists():
        raise FileNotFoundError(f"Missing {post_path}")

    pre_rows = json.loads(pre_path.read_text())
    post_rows = json.loads(post_path.read_text())
    story_map = _raykov_load_story_map(datadir)

    df_pre_all = _raykov_compute_diffs(pre_rows, story_map, mode="pre")
    df_post_all = _raykov_compute_diffs(post_rows, story_map, mode="post")
    df_pre = _raykov_summarize_om_ex(df_pre_all)
    df_post = _raykov_summarize_om_ex(df_post_all)

    _raykov_plot_om_ex(df_pre, "Pre (short delay)", plotsdir / "omissions_vs_extensions_pre.png")
    _raykov_plot_om_ex(df_post, "Post (long delay)", plotsdir / "omissions_vs_extensions_post.png")
    _raykov_plot_len_diff(df_pre_all, df_post_all, plotsdir / "length_diff_sem_pre_post.png")
    print(f"[Raykov] Plots saved to {plotsdir}")


# =========================================================================== #
# BARTLETT SUBCOMMAND
# =========================================================================== #

def run_bartlett_main(bargs):
    _set_seed(bargs.bg_seed)
    results_root = Path(bargs.results_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"--results_dir not found: {results_root}")

    bartlett_full = load_bartlett(bargs.bartlett_path)
    prompt = _make_prompt(bartlett_full)
    excl = _exclusion_from_bartlett(bartlett_full)

    offload_root = Path(bargs.offload_dir) if bargs.offload_dir else (results_root / "_offload")
    offload_root.mkdir(parents=True, exist_ok=True)

    if bargs.topics:
        topic_dirs = [results_root / t for t in bargs.topics]
    else:
        topic_dirs = [p for p in results_root.iterdir()
                      if p.is_dir() and (p / "model" / "final").exists()]
    if not topic_dirs:
        raise SystemExit(f"No topics with model/final found under {results_root}")
    topics = [td.name for td in topic_dirs]

    # --skip_final: only checkpoint new-word curves
    if getattr(bargs, "skip_final", False):
        analysis_dir = results_root / "_analysis"
        plots_dir = analysis_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cache = Path(bargs.ckpt_cache_dir) if bargs.ckpt_cache_dir else (results_root / "_ckpt_cache")

        df_epoch, df_temp = _build_newword_curves_from_checkpoints(
            results_root=results_root, original_text=bartlett_full, prompt=prompt,
            temps=bargs.ckpt_temps, offload_root=offload_root, cache_dir=ckpt_cache,
            n_samples=bargs.ckpt_n_samples, max_new_tokens=bargs.max_new_tokens,
            temp_fixed_for_epoch=bargs.ckpt_temp_fixed_for_epoch,
            epoch_fixed_for_temp=bargs.ckpt_epoch_fixed_for_temp,
            min_new_tokens=getattr(bargs, "min_new_tokens", -1),
        )
        efp = bargs.ckpt_epoch_fixed_for_temp
        tfe = bargs.ckpt_temp_fixed_for_epoch
        df_epoch.to_csv(plots_dir / f"new_words_curve_vs_epoch_temp{tfe}.csv", index=False)
        df_temp.to_csv(plots_dir / f"new_words_curve_vs_temp_epoch{efp}.csv", index=False)
        if not df_epoch.empty:
            _line_with_sem(df_epoch["epoch"].tolist(), df_epoch["mean"].to_numpy(float),
                           df_epoch["sem"].to_numpy(float), "Epoch", "Frac. new words",
                           "New words vs Epoch", plots_dir / f"new_words_vs_epoch_temp{tfe}.png")
        if not df_temp.empty:
            _line_with_sem(df_temp["temp"].tolist(), df_temp["mean"].to_numpy(float),
                           df_temp["sem"].to_numpy(float), "Temperature", "Frac. new words",
                           "New words vs Temperature", plots_dir / f"new_words_vs_temp_epoch{efp}.png")
        print("\n[Bartlett] Done (checkpoint new-words only).")
        return

    # Full final-model sampling + analysis
    temp_tag = str(bargs.temp)
    emb = _embedder()
    bart_emb = _embed_texts(emb, [bartlett_full])[0]
    topic_docs = load_topic_corpus_wiki(
        topics=topics, seed=bargs.bg_seed,
        articles_per_topic=bargs.articles_per_topic,
        chars_per_article=bargs.chars_per_article,
        use_tfidf_filter=not bargs.no_tfidf_filter,
    )

    all_rows: List[Dict] = []; stats_rows: List[Dict] = []
    bg_emb_groups = []; rec_mean_groups = []; rec_emb_groups = []
    pca_all_points = []; bg_labels_kept = []; centers = {}; topic_to_index = {}

    for tdir in topic_dirs:
        topic = tdir.name
        print(f"\n→ Sampling final model for topic: {topic}")

        bg_txts = topic_docs.get(topic, [])
        gen_dir = tdir / "generations"; gen_dir.mkdir(parents=True, exist_ok=True)
        samples_path = gen_dir / f"final_temp{temp_tag}_samples.json"

        texts: List[str] = []
        if samples_path.exists():
            try:
                payload = json.loads(samples_path.read_text())
                saved_temp = payload.get("temperature", None)
                saved_prompt = payload.get("prompt", None)
                if saved_prompt is not None and saved_prompt != prompt:
                    print(f"[{topic}] Ignoring cached samples (prompt mismatch)")
                elif saved_temp is not None and float(saved_temp) != float(bargs.temp):
                    print(f"[{topic}] Ignoring cached samples (temp mismatch)")
                else:
                    cached = payload.get("samples", [])
                    texts = [t for t in cached if isinstance(t, str)]
                    if texts:
                        print(f"[{topic}] Loaded {len(texts)} saved samples")
            except Exception:
                texts = []

        if len(texts) < bargs.num_samples:
            need = bargs.num_samples - len(texts)
            model, tok = _load_final_model_4bit(tdir, offload_root=offload_root)
            _min, _max = _resolve_auto_min_tokens(
                tok, bartlett_full, getattr(bargs, "min_new_tokens", -1), bargs.max_new_tokens, prompt)
            new_texts = _sample_n(model, tok, prompt, n=need, temp=bargs.temp, max_new=_max,
                                  min_new_tokens=_min)
            texts = texts + new_texts
            samples_path.write_text(json.dumps({
                "prompt": prompt, "temperature": bargs.temp,
                "num_samples": bargs.num_samples, "max_new_tokens": bargs.max_new_tokens,
                "samples": texts,
            }, indent=2))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif len(texts) > bargs.num_samples:
            texts = texts[:bargs.num_samples]

        embs = _embed_texts(emb, texts)
        joined = " ".join(texts)
        _wordcloud(joined, excl, gen_dir / f"wordcloud_temp{temp_tag}.png", title=f"{topic} — temp={bargs.temp}")

        if not bg_txts:
            continue

        bg_embs = _embed_texts(emb, bg_txts)
        bg_center = _mean_vec(bg_embs); centers[topic] = bg_center
        cos_each = [_cosdist(e, bg_center) for e in embs]

        for i, txt in enumerate(texts):
            all_rows.append({"topic": topic, "sample_idx": i + 1, "temperature": bargs.temp,
                             "text": txt, "cos_to_bg": float(cos_each[i])})

        cos_mean = float(np.mean(cos_each)) if cos_each else float("nan")
        cos_sem_val = float(np.std(cos_each, ddof=1) / np.sqrt(len(cos_each))) if len(cos_each) > 1 else 0.0
        mean_emb = _mean_vec(embs)
        cos_bart_to_bg = _cosdist(bart_emb, bg_center)
        stats_rows.append({
            "topic": topic, "n_samples": len(texts), "n_bg_docs": len(bg_txts),
            "cos_bart_to_bg": float(cos_bart_to_bg),
            "cos_samples_to_bg_mean": float(cos_mean), "cos_samples_to_bg_sem": float(cos_sem_val),
        })

        bg_emb_groups.append(bg_embs); pca_all_points.append(bg_embs)
        bg_labels_kept.append(topic)
        rec_mean_groups.append(mean_emb.reshape(1, -1)); rec_emb_groups.append(embs)
        _topic_hist(cos_each, float(cos_bart_to_bg), gen_dir / f"hist_cos_to_bg_temp{temp_tag}.png",
                    title=f"{topic}: samples→BG (n={len(cos_each)})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    analysis_dir = results_root / "_analysis"; plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(analysis_dir / f"final_temp{temp_tag}_all_samples.csv", index=False)
    df_stats = pd.DataFrame(stats_rows).sort_values("topic")
    df_stats.to_csv(analysis_dir / f"final_temp{temp_tag}_stats.csv", index=False)

    if pca_all_points:
        all_bg_points = np.vstack(pca_all_points)
        bg_means = np.vstack([_mean_vec(g) for g in bg_emb_groups])
        rec_means = np.vstack(rec_mean_groups) if rec_mean_groups else np.zeros((0, bart_emb.shape[0]))
        rec_points = np.vstack(rec_emb_groups) if rec_emb_groups else None
        rec_group_sizes = [g.shape[0] for g in rec_emb_groups] if rec_emb_groups else None
        labels_bg = [f"{t} data" for t in bg_labels_kept]
        labels_rec = [f"Final mean ({t})" for t in bg_labels_kept]
        group_sizes = [g.shape[0] for g in bg_emb_groups]

        for method in ["pca", "umap", "tsne"]:
            _plot_2d_embedding(all_bg_points, bg_means, rec_means, bart_emb,
                               labels_bg, labels_rec, group_sizes,
                               plots_dir / f"embedding_2d_{method}_temp{temp_tag}.png",
                               method=method, random_state=bargs.bg_seed,
                               rec_points=rec_points, rec_group_sizes=rec_group_sizes)

        # Legacy-style PCA with larger recalled-mean markers
        _pca_inset(all_bg_points, bg_means, rec_means, bart_emb,
                   labels_bg, labels_rec, group_sizes,
                   plots_dir / f"global_pca_inset_final_temp{temp_tag}.png",
                   rec_points=rec_points, rec_group_sizes=rec_group_sizes)

    if not df_stats.empty:
        _paired_bars(
            df_stats["cos_bart_to_bg"].to_numpy(float),
            df_stats["cos_samples_to_bg_mean"].to_numpy(float),
            np.zeros(len(df_stats)),
            df_stats["cos_samples_to_bg_sem"].to_numpy(float),
            list(df_stats["topic"]), "Original", "Recalled", "Cosine distance",
            f"Distance to background centroid (final model @ temp {bargs.temp})",
            plots_dir / f"bar_distances_final_temp{temp_tag}.png",
        )

    # Also run checkpoint new-word curves
    ckpt_cache = Path(bargs.ckpt_cache_dir) if bargs.ckpt_cache_dir else (results_root / "_ckpt_cache")
    df_epoch, df_temp = _build_newword_curves_from_checkpoints(
        results_root=results_root, original_text=bartlett_full, prompt=prompt,
        temps=bargs.ckpt_temps, offload_root=offload_root, cache_dir=ckpt_cache,
        n_samples=bargs.ckpt_n_samples, max_new_tokens=bargs.max_new_tokens,
        temp_fixed_for_epoch=bargs.ckpt_temp_fixed_for_epoch,
        epoch_fixed_for_temp=bargs.ckpt_epoch_fixed_for_temp,
    )
    efp = bargs.ckpt_epoch_fixed_for_temp; tfe = bargs.ckpt_temp_fixed_for_epoch
    df_epoch.to_csv(plots_dir / f"new_words_curve_vs_epoch_temp{tfe}.csv", index=False)
    df_temp.to_csv(plots_dir / f"new_words_curve_vs_temp_epoch{efp}.csv", index=False)
    if not df_epoch.empty:
        _line_with_sem(df_epoch["epoch"].tolist(), df_epoch["mean"].to_numpy(float),
                       df_epoch["sem"].to_numpy(float), "Epoch", "Frac. new words",
                       "New words vs Epoch", plots_dir / f"new_words_vs_epoch_temp{tfe}.png")
    if not df_temp.empty:
        _line_with_sem(df_temp["temp"].tolist(), df_temp["mean"].to_numpy(float),
                       df_temp["sem"].to_numpy(float), "Temperature", "Frac. new words",
                       "New words vs Temperature", plots_dir / f"new_words_vs_temp_epoch{efp}.png")

    print("\n[Bartlett] Done.")


# =========================================================================== #
# BARTLETT_CKPT SUBCOMMAND
# =========================================================================== #

def run_bartlett_ckpt_main(bargs):
    _set_seed(bargs.bg_seed)
    results_root = Path(bargs.results_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"--results_dir not found: {results_root}")

    if bargs.topics:
        topic_dirs = [results_root / t for t in bargs.topics]
    else:
        topic_dirs = [p for p in results_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not topic_dirs:
        raise SystemExit(f"No topics found under {results_root}")
    topics = [td.name for td in topic_dirs]

    bartlett_full = load_bartlett(bargs.bartlett_path)
    prompt = _make_prompt(bartlett_full)
    excl = _exclusion_from_bartlett(bartlett_full)

    offload_root = Path(bargs.offload_dir) if bargs.offload_dir else (results_root / "_offload")
    offload_root.mkdir(parents=True, exist_ok=True)

    emb = _embedder()
    bart_emb = _embed_texts(emb, [bartlett_full])[0]
    topic_docs = load_topic_corpus_wiki(
        topics=topics, seed=bargs.bg_seed,
        articles_per_topic=bargs.articles_per_topic,
        chars_per_article=bargs.chars_per_article,
        use_tfidf_filter=not bargs.no_tfidf_filter,
    )

    bg_by_topic = {}; bg_emb_groups = []; bg_labels_kept = []; pca_all_points = []
    for tdir in topic_dirs:
        topic = tdir.name
        bg_txts = topic_docs.get(topic, [])
        if not bg_txts:
            continue
        bg_embs = _embed_texts(emb, bg_txts)
        bg_center = _mean_vec(bg_embs)
        bg_by_topic[topic] = {"bg_txts": bg_txts, "bg_embs": bg_embs, "bg_center": bg_center}
        bg_emb_groups.append(bg_embs); bg_labels_kept.append(topic); pca_all_points.append(bg_embs)

    if not bg_by_topic:
        raise SystemExit("No background documents available.")

    all_bg_points = np.vstack(pca_all_points)
    bg_means = np.vstack([_mean_vec(g) for g in bg_emb_groups])
    labels_bg = [f"{t} data" for t in bg_labels_kept]
    group_sizes_bg = [g.shape[0] for g in bg_emb_groups]

    analysis_root = results_root / "_analysis" / "checkpoints"
    ckpt_cache_dir = Path(bargs.ckpt_cache_dir) if getattr(bargs, "ckpt_cache_dir", None) else (results_root / "_ckpt_cache")

    for epoch in list(bargs.epochs):
        for temp in list(bargs.temps):
            temp_tag = str(temp)
            print(f"\n[Bartlett-ckpt] epoch={epoch} temp={temp_tag}")
            analysis_dir = analysis_root / f"epoch{epoch}" / f"temp{temp_tag}"
            plots_dir = analysis_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)

            all_rows = []; stats_rows = []; rec_mean_groups = []

            for tdir in topic_dirs:
                topic = tdir.name
                if topic not in bg_by_topic:
                    continue
                print(f"  → {topic}")

                gen_dir = tdir / "generations" / f"epoch{epoch}" / f"temp{temp_tag}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                samples_path = gen_dir / "samples.json"

                texts: List[str] = []
                if samples_path.exists():
                    try:
                        payload = json.loads(samples_path.read_text())
                        saved_prompt = payload.get("prompt", None)
                        if saved_prompt is not None and saved_prompt != prompt:
                            print("    [Cache] Ignoring samples (prompt mismatch)")
                        elif int(payload.get("epoch", -1)) == int(epoch) and float(payload.get("temperature", -1)) == float(temp):
                            cached = payload.get("samples", [])
                            texts = [t for t in cached if isinstance(t, str)]
                            if texts:
                                print(f"    [Cache] Loaded {len(texts)} samples")
                    except Exception:
                        pass

                if len(texts) < bargs.num_samples:
                    need = bargs.num_samples - len(texts)
                    ckpts = dict(_find_checkpoints(tdir))
                    ckpt_path = ckpts.get(int(epoch))

                    # Try _ckpt_cache fallback
                    if ckpt_path is None:
                        cache_file = ckpt_cache_dir / f"{topic}_checkpoint_samples.json"
                        cache_key = f"{int(epoch)}_{float(temp)}"
                        if cache_file.exists():
                            try:
                                co = json.loads(cache_file.read_text())
                                ct = [t for t in co.get(cache_key, []) if isinstance(t, str)]
                            except Exception:
                                ct = []
                            if ct:
                                texts = texts + ct[:need]
                                need = bargs.num_samples - len(texts)
                        if need > 0:
                            raise SystemExit(f"Missing checkpoint for {topic} epoch={epoch}")

                    if need > 0:
                        model, tok = _load_checkpoint_model(ckpt_path, offload_root)
                        _min, _max = _resolve_auto_min_tokens(
                            tok, bartlett_full, getattr(bargs, "min_new_tokens", -1), bargs.max_new_tokens, prompt)
                        new = _sample_n(model, tok, prompt, n=need, temp=float(temp), max_new=_max,
                                        min_new_tokens=_min)
                        texts = texts + new
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                    samples_path.write_text(json.dumps({
                        "prompt": prompt, "epoch": int(epoch), "temperature": float(temp),
                        "num_samples": int(bargs.num_samples), "max_new_tokens": int(bargs.max_new_tokens),
                        "samples": texts,
                    }, indent=2))
                elif len(texts) > bargs.num_samples:
                    texts = texts[:bargs.num_samples]

                _wordcloud(" ".join(texts), excl, gen_dir / "wordcloud.png",
                           title=f"{topic} — epoch={epoch} temp={temp}")

                bg_center = bg_by_topic[topic]["bg_center"]
                embs = _embed_texts(emb, texts)
                cos_each = [_cosdist(e, bg_center) for e in embs]

                for i, txt in enumerate(texts):
                    all_rows.append({"topic": topic, "epoch": int(epoch), "temperature": float(temp),
                                     "sample_idx": i + 1, "text": txt, "cos_to_bg": float(cos_each[i])})

                cos_mean = float(np.mean(cos_each)) if cos_each else float("nan")
                cos_sem_val = float(np.std(cos_each, ddof=1) / np.sqrt(len(cos_each))) if len(cos_each) > 1 else 0.0
                mean_emb = _mean_vec(embs)
                rec_mean_groups.append(mean_emb.reshape(1, -1))
                cos_bart_to_bg = _cosdist(bart_emb, bg_center)
                stats_rows.append({
                    "topic": topic, "epoch": int(epoch), "temperature": float(temp),
                    "n_samples": len(texts), "n_bg_docs": int(bg_by_topic[topic]["bg_embs"].shape[0]),
                    "cos_bart_to_bg": float(cos_bart_to_bg),
                    "cos_samples_to_bg_mean": float(cos_mean), "cos_samples_to_bg_sem": float(cos_sem_val),
                })
                _topic_hist(cos_each, float(cos_bart_to_bg), gen_dir / "hist_cos_to_bg.png",
                            title=f"{topic}: samples→BG (n={len(cos_each)})")

            pd.DataFrame(all_rows).to_csv(analysis_dir / "all_samples.csv", index=False)
            df_stats = pd.DataFrame(stats_rows).sort_values("topic")
            df_stats.to_csv(analysis_dir / "stats.csv", index=False)

            rec_means = np.vstack(rec_mean_groups) if rec_mean_groups else np.zeros((0, bart_emb.shape[0]))
            topics_kept = list(bg_labels_kept)

            # PCA formatted like collate_figures panel a:
            # fit on background only, and plot only background + recalled means + Bartlett.
            _plot_ckpt_pca_like_collate(
                all_bg_points=all_bg_points,
                group_sizes_bg=group_sizes_bg,
                topics=topics_kept,
                rec_means=rec_means,
                bart_emb=bart_emb,
                out_png=plots_dir / "embedding_2d_pca.png",
                random_state=bargs.bg_seed,
            )
            _plot_ckpt_pca_like_collate(
                all_bg_points=all_bg_points,
                group_sizes_bg=group_sizes_bg,
                topics=topics_kept,
                rec_means=rec_means,
                bart_emb=bart_emb,
                out_png=plots_dir / "global_pca_inset.png",
                random_state=bargs.bg_seed,
            )

            if not df_stats.empty:
                _paired_bars(
                    df_stats["cos_bart_to_bg"].to_numpy(float),
                    df_stats["cos_samples_to_bg_mean"].to_numpy(float),
                    np.zeros(len(df_stats)),
                    df_stats["cos_samples_to_bg_sem"].to_numpy(float),
                    list(df_stats["topic"]), "Original", "Recalled", "Cosine distance",
                    f"Distance to background centroid (epoch {epoch} @ temp {temp})",
                    plots_dir / "bar_distances.png",
                )
            print(f"  [CSV] {analysis_dir / 'all_samples.csv'}")
            print(f"  [CSV] {analysis_dir / 'stats.csv'}")


# =========================================================================== #
# ARGPARSE + MAIN
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    bartlett_path_default = str(HERE.parent / "data" / "bartlett.txt")
    p = argparse.ArgumentParser(description="Plotting for Raykov + Bartlett experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Raykov
    pr = sub.add_parser("raykov", help="Raykov omissions/extensions plots.")
    pr.add_argument("--raykov_dir", type=str, default="output_raykov_xrag_fixed_firstline_50")
    pr.add_argument("--post_temps", type=str, default="0.0")

    # Bartlett
    pb = sub.add_parser("bartlett", help="Bartlett final-model sampling + PCA + bars + wordclouds.")
    pb.add_argument("--results_dir", type=str, default="output_twostage")
    pb.add_argument("--bartlett_path", type=str, default=bartlett_path_default)
    pb.add_argument("--topics", type=str, nargs="*", default=None)
    pb.add_argument("--num_samples", type=int, default=5)
    pb.add_argument("--temp", type=float, default=0.5)
    pb.add_argument("--max_new_tokens", type=int, default=500)
    pb.add_argument("--bg_seed", type=int, default=42)
    pb.add_argument("--articles_per_topic", type=int, default=1000)
    pb.add_argument("--chars_per_article", type=int, default=1000)
    pb.add_argument("--no_tfidf_filter", action="store_true",
                    help="Disable TF-IDF centrality filtering for Wikipedia topic articles")
    pb.add_argument("--use_mps", action="store_true")
    pb.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    pb.add_argument("--offload_dir", type=str, default=None)
    pb.add_argument("--skip_final", action="store_true",
                    help="Only compute checkpoint-based new-word curves.")
    pb.add_argument("--ckpt_temps", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    pb.add_argument("--ckpt_n_samples", type=int, default=10)
    pb.add_argument("--ckpt_temp_fixed_for_epoch", type=float, default=0.5)
    pb.add_argument("--ckpt_epoch_fixed_for_temp", type=int, default=5)
    pb.add_argument("--ckpt_cache_dir", type=str, default=None)
    pb.add_argument("--min_new_tokens", type=int, default=-1,
                    help="Minimum generated tokens (-1 = auto from Bartlett, 0 = off)")

    # Bartlett checkpoint grid
    pc = sub.add_parser("bartlett_ckpt", help="Bartlett checkpoint grid: bars, wordclouds, embeddings.")
    pc.add_argument("--results_dir", type=str, default="output_twostage")
    pc.add_argument("--bartlett_path", type=str, default=bartlett_path_default)
    pc.add_argument("--topics", type=str, nargs="*", default=None)
    pc.add_argument("--epochs", type=int, nargs="+", default=[5, 10])
    pc.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.1])
    pc.add_argument("--num_samples", type=int, default=5)
    pc.add_argument("--max_new_tokens", type=int, default=500)
    pc.add_argument("--bg_seed", type=int, default=42)
    pc.add_argument("--articles_per_topic", type=int, default=1000)
    pc.add_argument("--chars_per_article", type=int, default=1000)
    pc.add_argument("--no_tfidf_filter", action="store_true",
                    help="Disable TF-IDF centrality filtering for Wikipedia topic articles")
    pc.add_argument("--use_mps", action="store_true")
    pc.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    pc.add_argument("--offload_dir", type=str, default=None)
    pc.add_argument("--ckpt_cache_dir", type=str, default=None)
    pc.add_argument("--min_new_tokens", type=int, default=-1,
                    help="Minimum generated tokens (-1 = auto from Bartlett, 0 = off)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "raykov":
        run_raykov_main(args)
    elif args.cmd == "bartlett":
        run_bartlett_main(args)
    elif args.cmd == "bartlett_ckpt":
        run_bartlett_ckpt_main(args)


if __name__ == "__main__":
    main()
