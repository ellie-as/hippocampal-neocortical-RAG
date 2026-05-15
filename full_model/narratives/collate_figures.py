"""
Collate narrative simulation figures into a single combined plot.
Regenerates all plots from underlying data for visual consistency.

Layout:
- a) PCA plot (left, square)
- Right of PCA, two rows:
  - Row 1: b) Bergman/Roediger data, c) Cosine distances to topic centroid
  - Row 2: d) Frac new words by temp, e) Frac new words by epoch
- f) Row of 5 word clouds
- g) Human (Raykov): long vs short retention
- h) Model: before vs after consolidation
"""

from __future__ import annotations
from pathlib import Path
import gc
import hashlib
import json
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from utils import EXCLUDE_OFFENSIVE_WORDS, load_bartlett, recall_prefix
import plot_config as CFG

# torch is only needed when sampling (not for pure plotting from cached CSVs)
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Increase default font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})
from scipy.stats import t
from scipy.spatial.distance import cosine as cos_dist
import string as stringp
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.source_data import export_figure_source_data

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

WORDCLOUD_RENDER_SCALE = 2
WORDCLOUD_MAX_FONT_SIZE = 45
WORDCLOUD_MIN_FONT_SIZE = 8
WORDCLOUD_RELATIVE_SCALING = 0.10
WORDCLOUD_BASE_COLORMAP = plt.get_cmap("plasma")
WORDCLOUD_COLORMAP = LinearSegmentedColormap.from_list(
    "plasma_without_light_yellow",
    WORDCLOUD_BASE_COLORMAP(np.linspace(0.02, 0.78, 256)),
)

try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except ImportError:
    HAS_PCA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False


# ============================================================================
# PATHS & CONFIG  (driven by plot_config.py)
# ============================================================================
BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPO_ROOT = BASE.parent.parent
FIGURES_DIR = REPO_ROOT / "figures"

# Data paths
RAYKOV_DIR = BASE / "output_raykov_xrag_fixed_firstline_50" / "data"
BARTLETT_DIR = BASE / CFG.results_dir
BARTLETT_ANALYSIS = BARTLETT_DIR / "_analysis"
RAYKOV_HUMAN_CSV = REPO_ROOT / "data" / "Exp_5b_vs_Exp_4.csv"

# Color scheme
COLORS = ['#6a00a8', '#e16462', '#b12a90', '#0d0887', '#f0f921', '#fca636']
COLOR_INCOMPLETE = 'skyblue'
COLOR_UPDATED = 'purple'


def _maybe_float(value):
    return None if pd.isna(value) else float(value)

# Bartlett story: single source of truth is utils
BARTLETT_STORY = load_bartlett()
_BARTLETT_WORD_COUNT = len(BARTLETT_STORY.split())

# Embedding truncation limit (resolved once at import time)
_raw = CFG.embed_trim_chars
if _raw is None:
    EMBED_TRIM: int | None = None
elif isinstance(_raw, int):
    EMBED_TRIM = _raw
elif isinstance(_raw, str) and _raw.lower() == "bartlett":
    EMBED_TRIM = len(BARTLETT_STORY)
else:
    EMBED_TRIM = int(_raw)


def _trim(texts: list[str]) -> list[str]:
    """Truncate texts to ``EMBED_TRIM`` characters (no-op when None)."""
    if EMBED_TRIM is None:
        return texts
    return [t[:EMBED_TRIM] for t in texts]


# Cosine aggregation method
COSINE_AGG: str = getattr(CFG, "cosine_aggregation", "mean_of_distances")


def _aggregate_cos(text_embs: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    """Compute (cos_mean, cos_sem) from sample embeddings and a centroid.

    Respects ``COSINE_AGG``:
      * ``"mean_of_distances"`` – average per-sample cosine distances.
      * ``"distance_of_mean"``  – cosine distance of the mean embedding.
    SEM is always computed over per-sample cosine distances.
    """
    from scipy.spatial.distance import cosine as _cos
    cos_each = np.array([float(_cos(e, center)) for e in text_embs])
    cos_sem = (float(np.std(cos_each, ddof=1) / np.sqrt(len(cos_each)))
               if len(cos_each) > 1 else 0.0)
    if COSINE_AGG == "distance_of_mean":
        mean_emb = np.mean(text_embs, axis=0)
        cos_mean = float(_cos(mean_emb, center))
    else:
        cos_mean = float(np.mean(cos_each))
    return cos_mean, cos_sem


# ============================================================================
# SAMPLING INFRASTRUCTURE
# ============================================================================
def _make_prompt(bartlett_text: str) -> str:
    """Build the recall prompt from the shortened Bartlett cue."""
    return f"<s>[INST] {recall_prefix()} What happened (in detail)? [/INST]"


def _first_sentence(bartlett_text: str) -> str:
    """Prompt cue used to prefix recalled text for embedding comparisons."""
    return recall_prefix()


def _find_checkpoints(topic_dir: Path) -> dict[int, Path]:
    """Return {epoch: checkpoint_path} for a topic directory."""
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

    return dict(checkpoints)


def _load_model(ckpt_path: Path, offload_root: Path):
    """Load a LoRA checkpoint (4-bit) and return (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    cfg_path = ckpt_path / "adapter_config.json"
    base_name = json.loads(cfg_path.read_text())["base_model_name_or_path"]

    offload_folder = offload_root / f"_offload_{ckpt_path.parent.name}"
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(
        base, str(ckpt_path), device_map="auto", offload_folder=str(offload_folder),
    )
    model.eval()
    return model, tok


def _resolve_min_new_tokens(tok, bartlett_text: str, max_new: int, prompt: str | None = None) -> tuple[int | None, int]:
    """Return ``(min_new_tokens, max_new_tokens)`` for generation.

    When ``enforce_mean_length`` is enabled both are set to the prompt-conditioned
    Bartlett token count so generated recalls match the original length.
    """
    if not bool(getattr(CFG, "enforce_mean_length", False)):
        return None, max_new
    if prompt:
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tok(f"{prompt} {bartlett_text}", add_special_tokens=False)["input_ids"]
        bart_tok_len = len(target_ids) - len(prompt_ids)
    else:
        tok_ids = tok(bartlett_text, add_special_tokens=False)["input_ids"]
        bart_tok_len = len(tok_ids) if isinstance(tok_ids, list) else int(tok_ids.shape[-1])
    if bart_tok_len <= 0:
        return None, max_new
    print(f"  [length guard] Auto min/max_new_tokens = {bart_tok_len} (from prompt-conditioned Bartlett token count)")
    return int(bart_tok_len), int(bart_tok_len)


def _filter_short_texts(texts: list[str]) -> list[str]:
    """Drop cached texts that are shorter than the Bartlett story when enforce_mean_length is on."""
    if not bool(getattr(CFG, "enforce_mean_length", False)):
        return texts
    threshold = int(_BARTLETT_WORD_COUNT * 0.5)
    kept = [t for t in texts if len(t.split()) >= threshold]
    n_dropped = len(texts) - len(kept)
    if n_dropped:
        print(f"    [length guard] Dropped {n_dropped}/{len(texts)} cached texts "
              f"shorter than {threshold} words")
    return kept


def _sample_n(
    model,
    tok,
    prompt: str,
    n: int,
    temp: float,
    max_new: int,
    *,
    min_new_tokens: int | None = None,
) -> list[str]:
    """Generate *n* samples from *model* at *temp*."""
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    seq_len = input_ids.shape[1]

    texts: list[str] = []
    batch_size = min(n, 100)
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        in_ids = input_ids.expand(b, -1).contiguous()
        att = attn.expand(b, -1).contiguous() if attn is not None else None
        with torch.inference_mode():
            gen_kwargs = dict(
                input_ids=in_ids, attention_mask=att,
                max_new_tokens=max_new, do_sample=(temp > 0),
                temperature=temp if temp > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id, use_cache=True,
            )
            if min_new_tokens is not None and min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = int(min_new_tokens)
            out = model.generate(**gen_kwargs)
        new_tokens = out[:, seq_len:]
        texts.extend(t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True))
        del out, in_ids, att
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return texts


def _ensure_samples(
    results_dir: Path,
    topics: list[str],
    epoch: int,
    temp: float,
    num_samples: int,
    max_new_tokens: int,
    bartlett_text: str,
    *,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (samples_df, stats_df) for the requested epoch/temp condition.

    Uses cached files under ``results_dir/_analysis/`` when they exist and match
    the requested epoch/temp; otherwise samples from checkpoints on the fly and
    caches the results.

    When *overwrite* is True, cached CSVs and per-topic generation caches are
    ignored and data is regenerated from models/checkpoints.
    """
    analysis_dir = results_dir / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tag = f"epoch{epoch}_temp{temp}"
    samples_csv = analysis_dir / f"{tag}_all_samples.csv"
    stats_csv = analysis_dir / f"{tag}_stats.csv"

    # Try loading cached CSVs (skip when overwriting)
    if not overwrite:
        if samples_csv.exists() and stats_csv.exists():
            df_s = pd.read_csv(samples_csv)
            df_st = pd.read_csv(stats_csv)
            if len(df_s) > 0 and len(df_st) > 0:
                print(f"  [Cache] Loaded {samples_csv.name} ({len(df_s)} rows)")
                return df_s, df_st
    else:
        print(f"  [Overwrite] Skipping cached CSVs for {tag}")

    # --- Need to sample from models (or use _ckpt_cache fallback) ---
    print(f"  Sampling from checkpoints: epoch={epoch} temp={temp} n={num_samples}")
    from sentence_transformers import SentenceTransformer
    _model_name = getattr(CFG, "embedding_model", "all-MiniLM-L6-v2")
    embedder = SentenceTransformer(_model_name)
    prompt = _make_prompt(bartlett_text)
    bartlett_emb = embedder.encode(_trim([bartlett_text]), show_progress_bar=False)[0]

    # Load background data for topic centroids
    from utils import load_topic_corpus_wiki
    topic_docs = load_topic_corpus_wiki(
        topics=topics, seed=42,
        articles_per_topic=CFG.articles_per_topic,
        chars_per_article=CFG.chars_per_article,
        use_tfidf_filter=getattr(CFG, "use_tfidf_filter", True),
    )

    offload_root = results_dir / "_offload"
    offload_root.mkdir(parents=True, exist_ok=True)

    # Pre-load _ckpt_cache so we can fall back to it without GPU
    _ckpt_data = _load_ckpt_cache()

    all_rows: list[dict] = []
    stats_rows: list[dict] = []

    for topic in topics:
        topic_dir = results_dir / topic
        texts: list[str] = []

        # Try loading from per-topic generation cache first (skip when overwriting)
        if not overwrite and topic_dir.exists():
            gen_dir = topic_dir / "generations" / f"epoch{epoch}" / f"temp{temp}"
            gen_cache = gen_dir / "samples.json"
            if gen_cache.exists():
                try:
                    payload = json.loads(gen_cache.read_text())
                    saved_prompt = payload.get("prompt", None)
                    if saved_prompt is not None and saved_prompt != prompt:
                        print(f"    [{topic}] Ignoring cached generation samples (prompt mismatch)")
                    else:
                        cached = payload.get("samples", [])
                        texts = _filter_short_texts(
                            [t for t in cached if isinstance(t, str)]
                        )[:num_samples]
                        if texts:
                            print(f"    [{topic}] Loaded {len(texts)} cached generation samples")
                except Exception:
                    texts = []

        # Fallback: _ckpt_cache (skip when overwriting)
        if not overwrite and not texts and topic in _ckpt_data:
            cache_key = f"{int(epoch)}_{float(temp)}"
            cached_texts = _ckpt_data[topic].get(cache_key, [])
            if cached_texts:
                texts = _filter_short_texts(
                    [t for t in cached_texts if isinstance(t, str)]
                )[:num_samples]
                print(f"    [{topic}] Using {len(texts)} texts from _ckpt_cache ({cache_key})")

        # Fallback: sample from model checkpoints (requires GPU)
        if len(texts) < num_samples and topic_dir.exists():
            ckpt_map = _find_checkpoints(topic_dir)
            ckpt_path = ckpt_map.get(epoch)
            if ckpt_path is None:
                final = topic_dir / "model" / "final"
                if final.exists() and (final / "adapter_config.json").exists():
                    ckpt_path = final
            if ckpt_path is not None:
                need = num_samples - len(texts)
                print(f"    [{topic}] Sampling {need} texts from {ckpt_path.name}...")
                model, tok = _load_model(ckpt_path, offload_root)
                min_new_tokens, eff_max_new = _resolve_min_new_tokens(tok, bartlett_text, max_new_tokens, prompt)
                new_texts = _sample_n(
                    model,
                    tok,
                    prompt,
                    need,
                    temp,
                    eff_max_new,
                    min_new_tokens=min_new_tokens,
                )
                texts.extend(new_texts)
                del model, tok
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                gen_dir = topic_dir / "generations" / f"epoch{epoch}" / f"temp{temp}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                gen_cache = gen_dir / "samples.json"
                gen_cache.write_text(json.dumps({
                    "prompt": prompt, "epoch": epoch, "temperature": temp,
                    "num_samples": num_samples, "max_new_tokens": max_new_tokens,
                    "samples": texts,
                }, indent=2))

        if not texts:
            print(f"    [{topic}] No texts available for epoch={epoch} temp={temp}, skipping")
            continue

        # Compute stats
        bg_txts = topic_docs.get(topic, [])
        if not bg_txts:
            continue
        bg_embs = embedder.encode(_trim(bg_txts), show_progress_bar=False)
        bg_center = np.mean(bg_embs, axis=0)

        text_embs = embedder.encode(_trim(texts), show_progress_bar=False)
        from scipy.spatial.distance import cosine as _cos
        cos_each = [float(_cos(e, bg_center)) for e in text_embs]

        for i, txt in enumerate(texts):
            all_rows.append({"topic": topic, "sample_idx": i + 1, "temperature": temp,
                             "text": txt, "cos_to_bg": cos_each[i]})

        cos_mean, cos_sem = _aggregate_cos(text_embs, bg_center)
        bart_to_bg = float(_cos(bartlett_emb, bg_center))
        stats_rows.append({
            "topic": topic, "n_samples": len(texts), "n_bg_docs": len(bg_txts),
            "cos_bart_to_bg": bart_to_bg,
            "cos_samples_to_bg_mean": cos_mean,
            "cos_samples_to_bg_sem": cos_sem,
        })

    df_samples = pd.DataFrame(all_rows)
    df_stats = pd.DataFrame(stats_rows).sort_values("topic") if stats_rows else pd.DataFrame()

    # Cache for next time
    if len(df_samples) > 0:
        df_samples.to_csv(samples_csv, index=False)
        print(f"  [Cache] Wrote {samples_csv.name}")
    if len(df_stats) > 0:
        df_stats.to_csv(stats_csv, index=False)
        print(f"  [Cache] Wrote {stats_csv.name}")

    return df_samples, df_stats


def _ensure_wordcloud_samples(
    results_dir: Path,
    topics: list[str],
    epoch: int,
    temp: float,
    num_samples: int,
    max_new_tokens: int,
    bartlett_text: str,
    *,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """Return {topic: [text, ...]} for word-cloud generation, sampling if needed.

    When *overwrite* is True, per-topic generation caches are ignored and data is
    regenerated from model checkpoints.
    """
    prompt = _make_prompt(bartlett_text)
    offload_root = results_dir / "_offload"
    offload_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[str]] = {}
    for topic in topics:
        topic_dir = results_dir / topic
        gen_dir = topic_dir / "generations" / f"epoch{epoch}" / f"temp{temp}"
        gen_cache = gen_dir / "samples.json"

        texts: list[str] = []
        if not overwrite and gen_cache.exists():
            try:
                payload = json.loads(gen_cache.read_text())
                texts = _filter_short_texts(
                    [t for t in payload.get("samples", []) if isinstance(t, str)]
                )[:num_samples]
            except Exception:
                pass

        if len(texts) < num_samples and topic_dir.exists():
            ckpt_map = _find_checkpoints(topic_dir)
            ckpt_path = ckpt_map.get(epoch)
            if ckpt_path is None:
                final = topic_dir / "model" / "final"
                if final.exists() and (final / "adapter_config.json").exists():
                    ckpt_path = final
            if ckpt_path is not None:
                need = num_samples - len(texts)
                print(f"    [{topic}] Sampling {need} texts for wordcloud (epoch={epoch}, temp={temp})...")
                model, tok = _load_model(ckpt_path, offload_root)
                min_new_tokens, eff_max_new = _resolve_min_new_tokens(tok, bartlett_text, max_new_tokens, prompt)
                texts.extend(
                    _sample_n(
                        model,
                        tok,
                        prompt,
                        need,
                        temp,
                        eff_max_new,
                        min_new_tokens=min_new_tokens,
                    )
                )
                del model, tok
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                gen_dir.mkdir(parents=True, exist_ok=True)
                gen_cache.write_text(json.dumps({
                    "prompt": prompt, "epoch": epoch, "temperature": temp,
                    "num_samples": num_samples, "max_new_tokens": max_new_tokens,
                    "samples": texts,
                }, indent=2))

        if texts:
            result[topic] = texts
    return result


# ============================================================================
# DATA LOADING
# ============================================================================
def load_raykov_data():
    """Load Raykov simulation data."""
    if not RAYKOV_DIR.exists():
        print(f"Raykov data not found at {RAYKOV_DIR}")
        return None
    
    try:
        stories = pickle.load(open(RAYKOV_DIR / "stories_prepared.pkl", "rb"))
        pre = json.loads((RAYKOV_DIR / "generations_pre.json").read_text())
        post = json.loads((RAYKOV_DIR / "generations_post.json").read_text())
        return {"stories": stories, "pre": pre, "post": post}
    except Exception as e:
        print(f"Error loading Raykov data: {e}")
        return None


def load_raykov_human_data():
    """Load Raykov human omission/extension data (Exp_5b_vs_Exp_4.csv), if available."""
    if not RAYKOV_HUMAN_CSV.exists():
        return None
    try:
        df = pd.read_csv(RAYKOV_HUMAN_CSV)
    except Exception as e:
        print(f"Error loading Raykov human data: {e}")
        return None

    required = {"Study", "Condition", "Omission", "Extension"}
    if not required.issubset(set(df.columns)):
        print(f"Raykov human CSV missing columns: {sorted(required - set(df.columns))}")
        return None

    cond_map = {0: "incomplete", 1: "updated", "0": "incomplete", "1": "updated"}
    df = df.copy()
    df["Condition"] = df["Condition"].map(cond_map)
    df["Study"] = df["Study"].astype(str).str.lower()
    df = df[df["Condition"].isin(["incomplete", "updated"])]
    df = df[df["Study"].isin(["immediate", "delayed"])]

    out = {}
    for study in ["immediate", "delayed"]:
        sub = df[df["Study"] == study]
        out[study] = {
            "incomplete": {"om": [], "ex": []},
            "updated": {"om": [], "ex": []},
        }
        for cond in ["incomplete", "updated"]:
            subc = sub[sub["Condition"] == cond]
            if "Sub_num" in subc.columns:
                g = subc.groupby("Sub_num")[["Omission", "Extension"]].mean(numeric_only=True)
                out[study][cond]["om"] = g["Omission"].dropna().astype(float).tolist()
                out[study][cond]["ex"] = g["Extension"].dropna().astype(float).tolist()
            else:
                out[study][cond]["om"] = subc["Omission"].dropna().astype(float).tolist()
                out[study][cond]["ex"] = subc["Extension"].dropna().astype(float).tolist()
    return out


def load_bartlett_stats():
    """Load Bartlett analysis stats (placeholder - real loading in main via _ensure_samples)."""
    tag = f"epoch{CFG.pca_epoch}_temp{CFG.pca_temp}"
    stats_path = BARTLETT_ANALYSIS / f"{tag}_stats.csv"
    if stats_path.exists():
        return pd.read_csv(stats_path)
    return None


def load_bartlett_epoch_logs():
    """Load epoch logs for all topics."""
    logs = {}
    for topic in CFG.topics:
        log_path = BARTLETT_DIR / topic / "stage2_bartlett" / "epoch_logs.json"
        if log_path.exists():
            try:
                logs[topic] = json.loads(log_path.read_text())
            except:
                pass
    return logs


# ============================================================================
# HELPERS
# ============================================================================
def _t95_ci(x):
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return 0.0
    se = x.std(ddof=1) / (n ** 0.5)
    return float(t.ppf(0.975, df=n-1) * se)


def _word_count(s):
    return len((s or "").strip().split())


def _compute_raykov_omex(rows, story_map, mode):
    """Compute omission/extension data from Raykov generations."""
    results = {"incomplete": {"om": [], "ex": []}, "updated": {"om": [], "ex": []}}
    
    for r in rows:
        cat = r.get("category", "")
        if cat not in ["incomplete", "updated"]:
            continue
        
        rid = r.get("id")
        if mode == "post":
            input_text = r.get("input_text") or story_map.get(rid, "")
            out_text = r.get("generations", {}).get("0.0", r.get("generation", ""))
        else:
            input_text = story_map.get(rid, "")
            out_text = r.get("generation", "")
        
        in_len = _word_count(input_text)
        out_len = _word_count(out_text)
        diff = out_len - in_len
        
        if diff < 0:
            results[cat]["om"].append(1.0)
            results[cat]["ex"].append(0.0)
        else:
            results[cat]["om"].append(0.0)
            results[cat]["ex"].append(1.0)
    
    return results


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================
def plot_bergman_roediger(ax):
    """Plot Bergman & Roediger data showing distortion at different retention intervals."""
    intervals = ["15 min", "1 week", "6 months"]
    accurate = np.array([0.19, 0.09, 0.04])
    minor = np.array([0.21, 0.18, 0.07])
    major = np.array([0.15, 0.18, 0.16])
    
    accurate_sd = np.array([0.10, 0.07, 0.03])
    minor_sd = np.array([0.12, 0.07, 0.04])
    major_sd = np.array([0.06, 0.08, 0.08])
    
    n = 8
    accurate_sem = accurate_sd / np.sqrt(n)
    minor_sem = minor_sd / np.sqrt(n)
    major_sem = major_sd / np.sqrt(n)
    
    total = accurate + minor + major
    no_frac = accurate / total
    minor_frac = minor / total
    major_frac = major / total
    
    no_sem = accurate_sem / total
    minor_sem_n = minor_sem / total
    major_sem_n = major_sem / total
    
    x = np.arange(len(intervals))
    width = 0.25
    
    # Three shades of blue
    ax.bar(x - width, no_frac, width, yerr=no_sem, capsize=3,
           label="No distortion", color="#08306b")
    ax.bar(x, minor_frac, width, yerr=minor_sem_n, capsize=3,
           label="Minor distortion", color="#4292c6")
    ax.bar(x + width, major_frac, width, yerr=major_sem_n, capsize=3,
           label="Major distortion", color="#c6dbef")
    
    ax.set_xticks(x)
    ax.set_xticklabels(intervals)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Fraction of recalled")
    ax.set_xlabel("Retention interval")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, edgecolor="lightgray",
              fontsize=10, loc='upper left')
    ax._source_data_rows = [
        {
            "Retention interval": interval,
            "Distortion type": distortion,
            "Fraction of recalled": float(value),
            "SEM": float(error),
        }
        for distortion, values, errors in [
            ("No distortion", no_frac, no_sem),
            ("Minor distortion", minor_frac, minor_sem_n),
            ("Major distortion", major_frac, major_sem_n),
        ]
        for interval, value, error in zip(intervals, values, errors)
    ]


def plot_cosine_distances(ax, stats_df=None):
    """Plot cosine distance to the background centroid."""
    if stats_df is None:
        ax.text(0.5, 0.5, "No Bartlett stats", ha='center', va='center', transform=ax.transAxes)
        return
    
    topics = stats_df['topic'].tolist()
    original_dists = stats_df['cos_bart_to_bg'].values
    recalled_dists = stats_df['cos_samples_to_bg_mean'].values
    recalled_sems = stats_df['cos_samples_to_bg_sem'].values

    x = np.arange(len(topics))
    width = 0.35
    
    # Use distinct red shades for part c
    ax.bar(x - width/2, original_dists, width, label="Original", color="#b22222", alpha=0.8)
    ax.bar(x + width/2, recalled_dists, width, yerr=recalled_sems, capsize=4,
           label="Recalled", color="#ff6b6b", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=20, ha='right')
    ax.set_ylabel("Cosine distance")
    ax.legend(loc='lower right', fontsize=10)
    ax._source_data_rows = [
        {
            "Topic": topic,
            "series": "Original",
            "Cosine distance": float(dist),
            "SEM": 0.0,
        }
        for topic, dist in zip(topics, original_dists)
    ] + [
        {
            "Topic": topic,
            "series": "Recalled",
            "Cosine distance": float(dist),
            "SEM": float(error),
        }
        for topic, dist, error in zip(topics, recalled_dists, recalled_sems)
    ]


def _load_ckpt_cache(cache_dir: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Load checkpoint sample caches written by ``plot.py bartlett --skip_final``.

    Returns ``{topic: {"epoch_temp": [text, ...], ...}, ...}``.
    The cache lives in ``{cache_dir}/{topic}_checkpoint_samples.json``
    with keys like ``"5_0.5"`` → list of sampled texts.

    When *cache_dir* is None the function checks (in order):
      1. ``pca_ckpt_dir`` from plot_config (external checkpoint samples)
      2. ``{BARTLETT_DIR}/_ckpt_cache`` (local default)
    """
    if cache_dir is None:
        ext = getattr(CFG, "pca_ckpt_dir", None)
        if ext and Path(ext).exists():
            cache_dir = Path(ext)
        else:
            cache_dir = BARTLETT_DIR / "_ckpt_cache"

    result: dict[str, dict[str, list[str]]] = {}
    if not cache_dir.exists():
        return result
    for f in cache_dir.glob("*_checkpoint_samples.json"):
        topic = f.name.replace("_checkpoint_samples.json", "")
        try:
            result[topic] = json.loads(f.read_text())
        except Exception:
            pass
    return result


def _load_all_ckpt_texts(
    ckpt_cache_dir: Path | None = None,
    topics: list[str] | None = None,
) -> dict[str, list[str]]:
    """Load ALL recalled texts across ALL epoch-temp combos from the checkpoint cache.

    Parameters
    ----------
    ckpt_cache_dir : Path or None
        Directory containing ``{Topic}_checkpoint_samples.json`` files.
        Defaults to ``{BARTLETT_DIR}/_ckpt_cache``.
    topics : list[str] or None
        Topic names.  Defaults to ``CFG.topics``.

    Returns ``{topic: [text, text, ...], ...}`` with all texts from every key.
    """
    if ckpt_cache_dir is None:
        ckpt_cache_dir = BARTLETT_DIR / "_ckpt_cache"
    if topics is None:
        topics = CFG.topics

    result: dict[str, list[str]] = {}
    for topic in topics:
        f = ckpt_cache_dir / f"{topic}_checkpoint_samples.json"
        if not f.exists():
            continue
        try:
            cached = json.loads(f.read_text())
        except Exception:
            continue
        texts: list[str] = []
        for key, samples in cached.items():
            if isinstance(samples, list):
                for t in samples:
                    if isinstance(t, str) and t.strip():
                        texts.append(t)
        texts = _filter_short_texts(texts)
        if texts:
            result[topic] = texts
            print(f"  [all-ckpt] {topic}: {len(texts)} texts (all ep-temp)")
    return result


_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _tokenize_words(text: str) -> list[str]:
    """Lowercase alphabetic tokens, keeping contractions (e.g. 'don't')."""
    return _WORD_RE.findall(text.lower())


def _frac_new(text: str, bartlett_words: set[str]) -> float:
    """Fraction of words in *text* that are not in *bartlett_words*."""
    words = _tokenize_words(text)
    if not words:
        return float("nan")
    return sum(1 for w in words if w not in bartlett_words) / len(words)


def plot_new_words_vs_temp(ax, epoch_logs=None, *, epoch: int | None = None):
    """Plot fraction of new words vs temperature at a fixed epoch.

    Data sources (checked in order):
      1. ``epoch_logs`` — ``entry["temps"]`` dict logged during training
      2. ``_ckpt_cache/`` — samples gathered by ``plot.py bartlett --skip_final``
      3. Placeholder data (so the panel is never empty)
    """
    epoch = epoch if epoch is not None else CFG.newwords_vs_temp_epoch
    bartlett_words = set(_tokenize_words(BARTLETT_STORY))

    temp_fracs: dict[float, list[float]] = {}  # temp -> [frac per topic]

    # Source 1: epoch_logs (includes multi-temp if available)
    if epoch_logs:
        for _topic, logs in epoch_logs.items():
            for entry in logs:
                if int(entry.get("epoch", -1)) != epoch:
                    continue
                temps_dict = entry.get("temps", {})
                greedy = entry.get("greedy", "")
                if greedy:
                    temps_dict = {"0.0": greedy, **temps_dict}
                for t_str, text in temps_dict.items():
                    if not text:
                        continue
                    try:
                        t_val = float(t_str)
                    except ValueError:
                        continue
                    frac = _frac_new(text, bartlett_words)
                    if np.isfinite(frac):
                        temp_fracs.setdefault(t_val, []).append(frac)

    # Source 2: _ckpt_cache (only if epoch_logs gave ≤ 1 temperature)
    if len(temp_fracs) <= 1:
        ckpt_cache = _load_ckpt_cache()
        if ckpt_cache:
            print(f"  [new-words-vs-temp] Using _ckpt_cache data for epoch={epoch}")
            for topic, cache in ckpt_cache.items():
                for key, texts in cache.items():
                    try:
                        ep_str, t_str = key.split("_", 1)
                        if int(ep_str) != epoch:
                            continue
                        t_val = float(t_str)
                    except (ValueError, IndexError):
                        continue
                    for text in _filter_short_texts(list(texts)):
                        if not text:
                            continue
                        frac = _frac_new(text, bartlett_words)
                        if np.isfinite(frac):
                            temp_fracs.setdefault(t_val, []).append(frac)

    if temp_fracs:
        temps_sorted = sorted(temp_fracs)
        means = np.array([np.mean(temp_fracs[t]) for t in temps_sorted])
        sems = np.array([
            np.std(temp_fracs[t], ddof=1) / np.sqrt(len(temp_fracs[t]))
            if len(temp_fracs[t]) > 1 else 0.0
            for t in temps_sorted
        ])
        ax.errorbar(temps_sorted, means, yerr=sems, fmt='-o', linewidth=2, markersize=6,
                    capsize=4, color=COLORS[0])
        ax._source_data_rows = [
            {
                "Temperature": float(temp_value),
                "Frac. new words": float(mean),
                "SEM": float(error),
            }
            for temp_value, mean, error in zip(temps_sorted, means, sems)
        ]
    else:
        ax.text(0.5, 0.5, "No data\n(run plot.py bartlett --skip_final\nto generate multi-temp samples)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='gray', style='italic')
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Frac. new words")


def plot_new_words_vs_epoch(ax, epoch_logs=None, *, temp: float | None = None):
    """Plot fraction of new words vs training epoch at a fixed temperature.

    Data sources (checked in order):
      1. ``epoch_logs`` — greedy or ``entry["temps"]`` dict
      2. ``_ckpt_cache/`` — samples gathered by ``plot.py bartlett --skip_final``
      3. Placeholder data
    """
    temp = temp if temp is not None else CFG.newwords_vs_epoch_temp
    bartlett_words = set(_tokenize_words(BARTLETT_STORY))

    epoch_new_fracs: dict[int, list[float]] = {}

    # Source 1: epoch_logs
    found_nongreedy = False
    if epoch_logs:
        for _topic, logs in epoch_logs.items():
            for entry in logs:
                ep = int(entry.get("epoch", 0))
                if temp == 0.0 or temp is None:
                    text = entry.get("greedy", "") or entry.get("greedy_text", "")
                else:
                    temps_dict = entry.get("temps", {})
                    text = ""
                    for k, v in temps_dict.items():
                        try:
                            if abs(float(k) - temp) < 1e-6:
                                text = v
                                found_nongreedy = True
                                break
                        except ValueError:
                            pass
                    if not text:
                        text = entry.get("greedy", "") or entry.get("greedy_text", "")
                if not text:
                    continue
                frac = _frac_new(text, bartlett_words)
                if np.isfinite(frac):
                    epoch_new_fracs.setdefault(ep, []).append(frac)

    # Source 2: _ckpt_cache (use if we want a non-greedy temp but epoch_logs
    # only had greedy, OR if epoch_logs were empty)
    use_cache = (not epoch_new_fracs) or (temp not in (0.0, None) and not found_nongreedy)
    if use_cache:
        ckpt_cache = _load_ckpt_cache()
        if ckpt_cache:
            print(f"  [new-words-vs-epoch] Using _ckpt_cache data for temp={temp}")
            epoch_new_fracs.clear()  # replace greedy fallback with real temp data
            for topic, cache in ckpt_cache.items():
                for key, texts in cache.items():
                    try:
                        ep_str, t_str = key.split("_", 1)
                        if abs(float(t_str) - temp) > 1e-6:
                            continue
                        ep = int(ep_str)
                    except (ValueError, IndexError):
                        continue
                    for text in _filter_short_texts(list(texts)):
                        if not text:
                            continue
                        frac = _frac_new(text, bartlett_words)
                        if np.isfinite(frac):
                            epoch_new_fracs.setdefault(ep, []).append(frac)

    if epoch_new_fracs:
        epochs = sorted(epoch_new_fracs.keys())
        means = np.array([np.mean(epoch_new_fracs[e]) for e in epochs])
        sems = np.array([
            np.std(epoch_new_fracs[e], ddof=1) / np.sqrt(len(epoch_new_fracs[e]))
            if len(epoch_new_fracs[e]) > 1 else 0.0
            for e in epochs
        ])
        ax.errorbar(epochs, means, yerr=sems, fmt='-o', linewidth=2, markersize=6,
                    capsize=4, color=COLORS[0])
        ax._source_data_rows = [
            {
                "Epoch": int(epoch_value),
                "Frac. new words": float(mean),
                "SEM": float(error),
            }
            for epoch_value, mean, error in zip(epochs, means, sems)
        ]
    else:
        ax.text(0.5, 0.5, "No data\n(run plot.py bartlett --skip_final\nto generate multi-epoch samples)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='gray', style='italic')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frac. new words")


def plot_cosine_distance_encoded_consolidated(ax, *, embedder=None):
    """
    Bar chart: cosine distance to original for xRAG-encoded vs LoRA-consolidated recalls.

    Uses artifacts from ``bartlett_encoding_vs_consolidation.py``
    (directory set by ``plot_config.enc_vs_con_dir``):
      - ``statistics.json``          -- pre-computed distances / means / SEMs
      - ``encoded_samples.json``     -- fallback: raw encoded texts
      - ``consolidated_samples.json``-- fallback: raw consolidated texts
    """
    enc_con_dir = BASE / CFG.enc_vs_con_dir
    stats_path = enc_con_dir / "statistics.json"
    enc_samples_path = enc_con_dir / "encoded_samples.json"
    con_samples_path = enc_con_dir / "consolidated_samples.json"

    encoded_mean = encoded_sem = consolidated_mean = consolidated_sem = None
    enc_dists = con_dists = None

    # ---- Try pre-computed statistics first --------------------------------
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
            enc_dists = [float(v) for v in stats["encoded"].get("distances", [])]
            con_dists = [float(v) for v in stats["consolidated"].get("distances", [])]
            encoded_mean = float(stats["encoded"]["mean"])
            encoded_sem = float(stats["encoded"]["sem"])
            consolidated_mean = float(stats["consolidated"]["mean"])
            consolidated_sem = float(stats["consolidated"]["sem"])
            print(f"  [Panel f] Loaded pre-computed stats from {stats_path.name}")
        except Exception as e:
            print(f"  [Panel f] Could not parse {stats_path}: {e}")

    # ---- Fallback: recompute from sample JSONs + SBERT --------------------
    if encoded_mean is None:
        if not enc_samples_path.exists() or not con_samples_path.exists():
            msg = (f"Run bartlett_encoding_vs_consolidation.py first\n"
                   f"(expected {enc_con_dir.relative_to(BASE)}/)")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            return
        if embedder is None or not HAS_SBERT:
            ax.text(0.5, 0.5, "SBERT not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            return

        bartlett_emb = embedder.encode(_trim([BARTLETT_STORY]), show_progress_bar=False)[0]
        def _dists_from_json(path):
            payload = json.loads(path.read_text())
            texts = [payload.get("greedy_temp0", "")] + payload.get("sampled", [])
            texts = [t for t in texts if t]
            embs = embedder.encode(_trim(texts), show_progress_bar=False)
            return [float(cos_dist(e, bartlett_emb)) for e in embs]

        enc_dists = _dists_from_json(enc_samples_path)
        con_dists = _dists_from_json(con_samples_path)
        encoded_mean = float(np.mean(enc_dists))
        encoded_sem = float(np.std(enc_dists, ddof=1) / np.sqrt(len(enc_dists))) if len(enc_dists) > 1 else 0.0
        consolidated_mean = float(np.mean(con_dists))
        consolidated_sem = float(np.std(con_dists, ddof=1) / np.sqrt(len(con_dists))) if len(con_dists) > 1 else 0.0
        print(f"  [Panel f] Recomputed stats from sample JSONs")

    # ---- Plot -------------------------------------------------------------
    labels = ["Encoded", "Consolidated"]
    means = [encoded_mean, consolidated_mean]
    sems = [encoded_sem, consolidated_sem]

    x = np.arange(2)
    ax.bar(x, means, yerr=sems, capsize=5, color=COLORS[0], alpha=0.9,
           edgecolor="none")
    if enc_dists and con_dists:
        rng = np.random.default_rng(0)
        jitter = 0.08
        for x0, vals in zip(x, [enc_dists, con_dists]):
            vals = np.asarray(vals, dtype=float)
            xs = x0 + rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(
                xs,
                vals,
                color=COLORS[0],
                edgecolors="white",
                linewidth=0.8,
                s=22,
                alpha=0.9,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine distance")
    y_max = max(means) + max(sems)
    if enc_dists and con_dists:
        y_max = max(y_max, max(enc_dists), max(con_dists))
    y_max += 0.05
    ax.set_ylim(0, min(y_max, 1.0))
    ax._source_data_rows = [
        {
            "Memory stage": label,
            "Value type": "Mean",
            "Cosine distance": float(mean),
            "SEM": float(error),
        }
        for label, mean, error in zip(labels, means, sems)
    ]
    if enc_dists and con_dists:
        ax._source_data_rows.extend(
            {
                "Memory stage": label,
                "Value type": "Individual",
                "Cosine distance": float(value),
                "SEM": "",
            }
            for label, values in zip(labels, [enc_dists, con_dists])
            for value in values
        )


def plot_omissions_extensions(
    ax,
    title,
    data=None,
    *,
    show_ylabel=True,
    show_legend=True,
    show_points=False,
):
    """Plot omission vs extension errors for incomplete and updated stories (mean ± 95% CI)."""
    if data is None:
        # Example data
        data = {
            "incomplete": {"om": [0.65] * 10, "ex": [0.35] * 10},
            "updated": {"om": [0.30] * 10, "ex": [0.70] * 10},
        }

    labels = ["Omission errors", "Extension errors"]
    x = np.arange(len(labels))
    bar_w = 0.35

    inc_om = np.array(data["incomplete"]["om"], dtype=float)
    inc_ex = np.array(data["incomplete"]["ex"], dtype=float)
    upd_om = np.array(data["updated"]["om"], dtype=float)
    upd_ex = np.array(data["updated"]["ex"], dtype=float)

    y_in = np.array([np.mean(inc_om), np.mean(inc_ex)])
    e_in = np.array([_t95_ci(inc_om), _t95_ci(inc_ex)])
    y_up = np.array([np.mean(upd_om), np.mean(upd_ex)])
    e_up = np.array([_t95_ci(upd_om), _t95_ci(upd_ex)])

    ax.bar(x - bar_w / 2, y_in, bar_w, yerr=e_in, capsize=5,
           color=COLOR_INCOMPLETE, alpha=1.0, label="Incomplete")
    ax.bar(x + bar_w / 2, y_up, bar_w, yerr=e_up, capsize=5,
           color=COLOR_UPDATED, alpha=0.6, label="Updated")

    if show_points:
        # Participant-level jittered points (matches the Raykov human plotting notebook style).
        rng = np.random.default_rng(0)
        jitter = 0.08

        def _scatter(vals, x0, color, alpha):
            vals = np.asarray(vals, dtype=float)
            if len(vals) == 0:
                return
            xs = x0 + rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(
                xs,
                vals,
                color=color,
                s=30,
                edgecolors="darkgrey",
                linewidth=1.0,
                alpha=alpha,
                zorder=3,
                rasterized=True,
            )

        _scatter(inc_om, x[0] - bar_w / 2, COLOR_INCOMPLETE, 0.8)
        _scatter(inc_ex, x[1] - bar_w / 2, COLOR_INCOMPLETE, 0.8)
        _scatter(upd_om, x[0] + bar_w / 2, COLOR_UPDATED, 0.8)
        _scatter(upd_ex, x[1] + bar_w / 2, COLOR_UPDATED, 0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.05)
    if show_ylabel:
        ax.set_ylabel("Proportion")
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", labelleft=False)
    ax.set_title(title, fontsize=9, pad=3)
    if show_legend:
        ax.legend(fontsize=10)
    summary = [
        ("Incomplete", "Omission errors", inc_om, y_in[0], e_in[0]),
        ("Incomplete", "Extension errors", inc_ex, y_in[1], e_in[1]),
        ("Updated", "Omission errors", upd_om, y_up[0], e_up[0]),
        ("Updated", "Extension errors", upd_ex, y_up[1], e_up[1]),
    ]
    rows = []
    for story_type, error_type, values, mean, ci95 in summary:
        rows.append(
            {
                "Error type": error_type,
                "Story type": story_type,
                "Value type": "Mean",
                "Proportion": float(mean),
                "95% CI": float(ci95),
            }
        )
        if show_points:
            for value in values:
                rows.append(
                    {
                        "Error type": error_type,
                        "Story type": story_type,
                        "Value type": "Individual",
                        "Proportion": float(value),
                        "95% CI": None,
                    }
                )
    ax._source_data_title = title
    ax._source_data_rows = rows


_BASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_pca_embeddings(ax, samples_df=None, embedder=None):
    """
    PCA of Wikipedia background clouds + recalled-story means (arrows from
    original Bartlett point).  Background data loaded via ``utils.load_topic_corpus_wiki``
    with sizes set in ``plot_config``.
    """
    if not HAS_PCA or not HAS_SBERT:
        ax.text(0.5, 0.5, "PCA/SBERT not available", ha='center', va='center', transform=ax.transAxes)
        return

    if samples_df is None or len(samples_df) == 0:
        ax.text(0.5, 0.5, "No Bartlett samples", ha='center', va='center', transform=ax.transAxes)
        return

    from utils import load_topic_corpus_wiki

    topics = sorted(samples_df['topic'].unique().tolist())
    color_map = {t: _BASE_COLORS[i % len(_BASE_COLORS)] for i, t in enumerate(topics)}

    if embedder is None:
        print("  Loading embedder for PCA...")
        _model_name = getattr(CFG, "embedding_model", "all-MiniLM-L6-v2")
        embedder = SentenceTransformer(_model_name)

    print("  Loading Wikipedia background data for PCA...")
    bg_texts = load_topic_corpus_wiki(
        topics, seed=42,
        articles_per_topic=CFG.articles_per_topic,
        chars_per_article=CFG.chars_per_article,
        use_tfidf_filter=getattr(CFG, "use_tfidf_filter", True),
    )
    
    # Embed background texts
    print("  Embedding background texts...")
    all_bg_embs = []
    bg_means = []
    group_sizes = []
    for topic in topics:
        texts = bg_texts.get(topic, [])
        if texts:
            embs = embedder.encode(_trim(texts), show_progress_bar=False)
            all_bg_embs.append(np.asarray(embs))
            bg_means.append(np.mean(embs, axis=0))
            group_sizes.append(len(texts))
        else:
            group_sizes.append(0)
            bg_means.append(np.zeros(384))
    
    all_bg_points = np.vstack(all_bg_embs) if all_bg_embs else np.zeros((0, 384))
    bg_means_arr = np.vstack(bg_means)
    
    # Compute recalled story means (one mean per topic)
    print("  Computing recalled story means...")
    bartlett_emb = embedder.encode(_trim([BARTLETT_STORY]))[0]
    rec_means = []
    for topic in topics:
        topic_texts = samples_df[samples_df['topic'] == topic]['text'].dropna().tolist()
        if topic_texts:
            embs = embedder.encode(_trim(topic_texts[:100]), show_progress_bar=False)
            rec_means.append(np.mean(embs, axis=0))
        else:
            rec_means.append(np.zeros(384))
    rec_means_arr = np.vstack(rec_means)
    
    # Dimensionality reduction — strategy driven by plot_config.pca_projection
    mode = getattr(CFG, "pca_projection", "pca_all")
    all_points = np.vstack([all_bg_points, bg_means_arr, rec_means_arr, bartlett_emb.reshape(1, -1)])
    n_bg = all_bg_points.shape[0]
    n_bg_means = bg_means_arr.shape[0]
    n_rec_means = rec_means_arr.shape[0]

    if mode == "umap_all":
        if not HAS_UMAP:
            ax.text(0.5, 0.5, "umap not installed", ha='center', va='center', transform=ax.transAxes)
            return
        reducer = UMAP(
            n_components=2, random_state=42,
            n_neighbors=getattr(CFG, "pca_umap_n_neighbors", 30),
            min_dist=getattr(CFG, "pca_umap_min_dist", 0.1),
        )
        all_2d = reducer.fit_transform(all_points)
        print(f"  [PCA] UMAP projection ({all_points.shape[0]} points)")

    elif mode == "tsne_all":
        if not HAS_TSNE:
            ax.text(0.5, 0.5, "sklearn TSNE not available", ha='center', va='center', transform=ax.transAxes)
            return
        perp = min(getattr(CFG, "pca_tsne_perplexity", 30), all_points.shape[0] - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
        all_2d = reducer.fit_transform(all_points)
        print(f"  [PCA] t-SNE projection ({all_points.shape[0]} points, perplexity={perp})")

    elif mode == "pca_background":
        pca = PCA(n_components=2, random_state=42)
        pca.fit(all_bg_points)
        all_2d = pca.transform(all_points)
        ev = pca.explained_variance_ratio_
        print(f"  [PCA] Fit on background only — var explained: {ev[0]:.1%}, {ev[1]:.1%}")

    elif mode == "pca_centroids_bartlett":
        fit_pts = np.vstack([bg_means_arr, bartlett_emb.reshape(1, -1)])
        pca = PCA(n_components=2, random_state=42)
        pca.fit(fit_pts)
        all_2d = pca.transform(all_points)
        ev = pca.explained_variance_ratio_
        print(f"  [PCA] Fit on centroids+Bartlett — var explained: {ev[0]:.1%}, {ev[1]:.1%}")

    elif mode == "pca_centroids_recalled":
        fit_pts = np.vstack([bg_means_arr, rec_means_arr])
        pca = PCA(n_components=2, random_state=42)
        pca.fit(fit_pts)
        all_2d = pca.transform(all_points)
        ev = pca.explained_variance_ratio_
        print(f"  [PCA] Fit on centroids+recalled — var explained: {ev[0]:.1%}, {ev[1]:.1%}")

    else:  # "pca_all" or fallback
        pca = PCA(n_components=2, random_state=42)
        all_2d = pca.fit_transform(all_points)
        ev = pca.explained_variance_ratio_
        print(f"  [PCA] Fit on all points — var explained: {ev[0]:.1%}, {ev[1]:.1%}")

    embeddings_2d = all_2d[:n_bg]
    rec_means_2d = all_2d[n_bg + n_bg_means:n_bg + n_bg_means + n_rec_means]
    bart_2d = all_2d[-1].reshape(1, -1)
    
    # Plot background clouds
    start = 0
    for i, (topic, sz) in enumerate(zip(topics, group_sizes)):
        end = start + sz
        ax.scatter(embeddings_2d[start:end, 0], embeddings_2d[start:end, 1],
                   color=color_map[topic], alpha=0.35, s=25)
        start = end
    
    # Plot recalled means with arrows
    for i, mean in enumerate(rec_means_2d):
        topic = topics[i]
        ax.scatter(mean[0], mean[1], color=color_map[topic], marker="o", s=25,
                   edgecolors="black", label=f"Recalled ({topic})")
        ax.arrow(bart_2d[0, 0], bart_2d[0, 1], mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
                 color="black", lw=0.5, length_includes_head=True, head_width=0.01)
    
    # Plot original Bartlett
    ax.scatter(bart_2d[0, 0], bart_2d[0, 1], color="black", marker="o", s=25,
               edgecolors="black", label="Original story")
    ax.legend(fontsize=10, ncol=1, loc="upper right", markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", bottom=False, left=False)
    
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
        
        # Calculate inset dimensions to match data region aspect ratio
        data_aspect = h / w  # height / width of data region
        inset_width = 0.27 * 1.5  # 1.5x scale
        inset_height = inset_width * data_aspect
        # Clamp height to reasonable range
        inset_height = min(max(inset_height, 0.15), 0.75)
        
        axins = ax.inset_axes([0.02, 0.02, inset_width, inset_height],
                              xlim=(xmin - margin * w, xmax + margin * w),
                              ylim=(ymin - margin * h, ymax + margin * h),
                              xticks=[], yticks=[])
        
        # Replot in inset
        start = 0
        for i, (topic, sz) in enumerate(zip(topics, group_sizes)):
            end = start + sz
            axins.scatter(embeddings_2d[start:end, 0], embeddings_2d[start:end, 1],
                          color=color_map[topic], alpha=0.25, s=130)
            start = end
        
        for i, mean in enumerate(rec_means_2d):
            topic = topics[i]
            axins.scatter(mean[0], mean[1], color=color_map[topic], marker="o", s=130,
                          edgecolors="black")
            axins.arrow(bart_2d[0, 0], bart_2d[0, 1], mean[0] - bart_2d[0, 0], mean[1] - bart_2d[0, 1],
                        color="black", lw=0.5, length_includes_head=True, head_width=0.006)
        
        axins.scatter(bart_2d[0, 0], bart_2d[0, 1], color="black", marker="o", s=130,
                      edgecolors="black", linewidth=2)
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1)

    rows = []
    start = 0
    for topic, sz in zip(topics, group_sizes):
        end = start + sz
        for x_value, y_value in embeddings_2d[start:end]:
            rows.append(
                {
                    "PC1": float(x_value),
                    "PC2": float(y_value),
                    "series": topic,
                    "Point type": "Background",
                }
            )
        start = end
    for topic, coord in zip(topics, rec_means_2d):
        rows.append(
            {
                "PC1": float(coord[0]),
                "PC2": float(coord[1]),
                "series": f"Recalled ({topic})",
                "Point type": "Recalled mean",
            }
        )
    rows.append(
        {
            "PC1": float(bart_2d[0, 0]),
            "PC2": float(bart_2d[0, 1]),
            "series": "Original story",
            "Point type": "Original",
        }
    )
    ax._source_data_rows = rows


def _wc_tokenize(text: str, bartlett_excl: set, min_len: int) -> list[str]:
    """Tokenize text, strip punctuation, remove Bartlett words and short words."""
    trans = str.maketrans(stringp.punctuation, ' ' * len(stringp.punctuation))
    words = text.translate(trans).lower().split()
    return [w for w in words if w not in bartlett_excl and len(w) >= min_len]


def _wc_pos_filter(words: list[str], mode: str) -> list[str]:
    """Optionally filter words by POS tag using spaCy."""
    if mode == "all":
        return words
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("[wordcloud] spaCy model en_core_web_sm not found; skipping POS filter.")
            return words
    except ImportError:
        print("[wordcloud] spaCy not installed; skipping POS filter.")
        return words

    if mode == "nouns":
        allowed = {"NOUN", "PROPN"}
    elif mode == "nouns_adjs":
        allowed = {"NOUN", "PROPN", "ADJ"}
    else:
        return words

    # Process in batch for speed
    doc = nlp(" ".join(words))
    return [tok.text.lower() for tok in doc if tok.pos_ in allowed and len(tok.text) >= 2]


def _wordcloud_exclusion_set() -> set[str]:
    """Words excluded before noun-only wordcloud scoring."""
    try:
        from wordcloud import STOPWORDS
    except Exception:
        stopwords = set()
    else:
        stopwords = {str(w).lower() for w in STOPWORDS}
    bartlett_excl = set(BARTLETT_STORY.translate(
        str.maketrans(stringp.punctuation, ' ' * len(stringp.punctuation))
    ).lower().split())
    return bartlett_excl | stopwords | EXCLUDE_OFFENSIVE_WORDS


def _wordcloud_cache_key(wc_texts: dict[str, list[str]], *, label: str, excluded: set[str] | None = None) -> str:
    """Stable cache key for expensive wordcloud preprocessing."""
    h = hashlib.sha1()
    h.update(label.encode("utf-8"))
    h.update(str(len(BARTLETT_STORY)).encode("ascii"))
    if excluded:
        for term in sorted(str(x).lower() for x in excluded):
            h.update(term.encode("utf-8"))
            h.update(b"\0")
    for topic in sorted(wc_texts):
        h.update(topic.encode("utf-8"))
        h.update(str(len(wc_texts[topic])).encode("ascii"))
        for txt in wc_texts[topic]:
            h.update(str(txt).encode("utf-8", errors="ignore"))
            h.update(b"\0")
    return h.hexdigest()[:16]


def _load_or_build_wordcloud_noun_docs(
    wc_texts: dict[str, list[str]],
    excluded: set[str],
) -> dict[str, list[str]]:
    """Return per-sample noun-only documents, cached because spaCy is expensive."""
    cache_dir = BARTLETT_ANALYSIS / "_wordcloud_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _wordcloud_cache_key(wc_texts, label="noun_docs_no_custom_stopwords_v2", excluded=excluded)
    cache_path = cache_dir / f"noun_docs_{key}.json"

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            docs = payload.get("docs", {})
            if isinstance(docs, dict):
                print(f"  [wordcloud] Loaded cached noun docs: {cache_path.name}")
                return {
                    str(topic): [str(x) for x in topic_docs]
                    for topic, topic_docs in docs.items()
                    if isinstance(topic_docs, list)
                }
        except Exception:
            pass

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception as e:
        print(f"  [wordcloud] Could not load spaCy noun filter ({type(e).__name__}: {e}); falling back to token docs")
        docs = {
            topic: [
                " ".join(_wc_tokenize(str(txt), excluded, 4))
                for txt in texts
            ]
            for topic, texts in wc_texts.items()
        }
    else:
        excluded_lower = {w.lower() for w in excluded}
        docs = {}
        for topic, texts in wc_texts.items():
            topic_docs = []
            print(f"  [wordcloud] Extracting noun docs for {topic} ({len(texts)} recalls)")
            for doc in nlp.pipe((str(txt) for txt in texts), batch_size=100):
                nouns = [
                    tok.text.lower()
                    for tok in doc
                    if tok.pos_ == "NOUN"
                    and tok.is_alpha
                    and len(tok.text) >= 4
                    and tok.text.lower() not in excluded_lower
                ]
                topic_docs.append(" ".join(nouns))
            docs[topic] = topic_docs

    cache_path.write_text(json.dumps({"docs": docs}, indent=2))
    print(f"  [wordcloud] Wrote cached noun docs: {cache_path.name}")
    return docs


def _load_or_build_wordcloud_bg_vocab(excluded: set[str]) -> set[str]:
    """Union vocabulary from all five background corpora, cached on disk."""
    cache_dir = BARTLETT_ANALYSIS / "_wordcloud_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "background_vocab_union_no_custom_stopwords_v2.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            terms = payload.get("terms", [])
            if isinstance(terms, list):
                print(f"  [wordcloud] Loaded cached background vocab: {cache_path.name}")
                return {str(t) for t in terms}
        except Exception:
            pass

    from utils import load_topic_corpus_wiki

    print("  [wordcloud] Loading background corpora for vocabulary filter...")
    bg_docs = load_topic_corpus_wiki(
        CFG.topics,
        seed=42,
        articles_per_topic=getattr(CFG, "articles_per_topic", 1000),
        chars_per_article=getattr(CFG, "chars_per_article", 1000),
        use_tfidf_filter=getattr(CFG, "use_tfidf_filter", True),
    )
    terms: set[str] = set()
    for docs in bg_docs.values():
        for doc in docs:
            terms.update(_wc_tokenize(str(doc), excluded, 4))

    cache_path.write_text(json.dumps({"terms": sorted(terms)}, indent=2))
    print(f"  [wordcloud] Wrote cached background vocab: {cache_path.name}")
    return terms


def _prepare_noun_tfidf_contrast_wordcloud_data(
    wc_texts: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Noun-only per-topic TF-IDF, contrasted across topics and filtered."""
    import math
    from sklearn.feature_extraction.text import TfidfVectorizer

    excluded = _wordcloud_exclusion_set()
    noun_docs = _load_or_build_wordcloud_noun_docs(wc_texts, excluded)

    weights_by_topic: dict[str, dict[str, float]] = {}
    stopwords = sorted(excluded)
    for topic in CFG.topics:
        docs = [d for d in noun_docs.get(topic, []) if d.strip()]
        if not docs:
            weights_by_topic[topic] = {}
            continue
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=stopwords,
            token_pattern=r"(?u)\b[a-z][a-z][a-z]+\b",
            ngram_range=(1, 1),
            min_df=0.0025,
            max_df=0.80,
            sublinear_tf=True,
        )
        try:
            x_mat = vectorizer.fit_transform(docs)
        except ValueError:
            weights_by_topic[topic] = {}
            continue
        scores = x_mat.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        weights_by_topic[topic] = dict(zip(terms, scores))

    # Cross-topic contrast: keep terms over-represented in the topic.
    contrasted: dict[str, dict[str, float]] = {}
    eps = 1e-9
    topics = list(CFG.topics)
    for topic in topics:
        topic_weights = weights_by_topic.get(topic, {})
        out: dict[str, float] = {}
        for word, score in topic_weights.items():
            other_scores = [
                weights_by_topic.get(other, {}).get(word, 0.0)
                for other in topics
                if other != topic
            ]
            other_mean = sum(other_scores) / max(len(other_scores), 1)
            ratio = math.log2((score + eps) / (other_mean + eps))
            if ratio > 0:
                out[word] = score * ratio
        contrasted[topic] = out

    bg_vocab = _load_or_build_wordcloud_bg_vocab(excluded)

    try:
        from wordfreq import zipf_frequency
    except ImportError:
        def _english_ok(_: str) -> bool:
            return True
        print("  [wordcloud] wordfreq not installed; skipping English-frequency filter.")
    else:
        def _english_ok(word: str) -> bool:
            return zipf_frequency(word, "en") >= 2.5

    filtered: dict[str, dict[str, float]] = {}
    for topic, weights in contrasted.items():
        topic_weights = {
            word: math.sqrt(weight)
            for word, weight in weights.items()
            if word in bg_vocab and _english_ok(word)
        }
        filtered[topic] = dict(sorted(topic_weights.items(), key=lambda kv: (-kv[1], kv[0])))

    return filtered


def prepare_wordcloud_data(
    wc_texts: dict[str, list[str]],
) -> dict[str, dict[str, float] | str]:
    """Process raw per-topic texts into word-frequency dicts ready for WordCloud.

    Reads options from ``plot_config``:
      - ``wordcloud_weighting`` ("raw" | "tfidf_topics" | "tfidf_english" | "topic_contrast")
      - ``wordcloud_pos_filter`` ("all" | "nouns" | "nouns_adjs")
      - ``wordcloud_exclude_shared`` (bool)
      - ``wordcloud_min_word_len`` (int)
      - ``wordcloud_min_freq`` / ``wordcloud_max_freq`` (optional int)
      - ``wordcloud_topic_boost`` (bool) + ``wordcloud_topic_boost_strength`` (float)

    Returns ``{topic: {word: weight, ...}}`` or ``{topic: filtered_text_string}``
    depending on weighting mode.
    """
    from collections import Counter
    import math

    def _topic_training_counts(topics: list[str], min_len: int) -> dict[str, dict[str, int]]:
        """
        Load per-topic word counts from the topic training corpus (Wikipedia topics dataset).
        Used only when wordcloud_topic_boost is enabled.
        """
        topics_key = tuple(sorted(topics))
        cache_key = (
            topics_key,
            int(getattr(CFG, "articles_per_topic", 1000)),
            int(getattr(CFG, "chars_per_article", 1000)),
            int(min_len),
        )
        cache = getattr(prepare_wordcloud_data, "_topic_boost_cache", None)
        if cache is None:
            cache = {}
            setattr(prepare_wordcloud_data, "_topic_boost_cache", cache)
        if cache_key in cache:
            return cache[cache_key]

        try:
            from utils import load_topic_corpus_wiki
        except Exception:
            print("[wordcloud] Could not import topic corpus loader; skipping topic_boost.")
            cache[cache_key] = {}
            return {}

        try:
            topic_docs = load_topic_corpus_wiki(
                topics=list(topics_key),
                seed=42,
                articles_per_topic=getattr(CFG, "articles_per_topic", 1000),
                chars_per_article=getattr(CFG, "chars_per_article", 1000),
                use_tfidf_filter=getattr(CFG, "use_tfidf_filter", True),
            )
        except Exception as e:
            print(f"[wordcloud] Could not load topic corpus ({type(e).__name__}: {e}); skipping topic_boost.")
            cache[cache_key] = {}
            return {}

        counts: dict[str, dict[str, int]] = {}
        for topic in topics_key:
            c = Counter()
            for doc in topic_docs.get(topic, []) or []:
                for w in _wc_tokenize(str(doc), bartlett_excl, min_len):
                    c[w] += 1
            counts[topic] = dict(c)

        cache[cache_key] = counts
        return counts

    def _apply_topic_boost(
        weights_by_topic: dict[str, dict[str, float]],
        training_counts: dict[str, dict[str, int]],
        strength: float,
    ) -> dict[str, dict[str, float]]:
        """Multiply weights by a topic-corpus-derived multiplier in [1, 1+strength]."""
        if strength <= 0:
            return weights_by_topic
        boosted: dict[str, dict[str, float]] = {}
        for topic, weights in weights_by_topic.items():
            counts = training_counts.get(topic) or {}
            if not weights or not counts:
                boosted[topic] = weights
                continue
            max_c = max(counts.values()) if counts else 0
            denom = math.log1p(max_c) if max_c > 0 else 1.0
            topic_boosted: dict[str, float] = {}
            for w, base in weights.items():
                c = counts.get(w, 0)
                mult = 1.0 + strength * (math.log1p(c) / denom if denom > 0 else 0.0)
                topic_boosted[w] = base * mult
            boosted[topic] = topic_boosted
        return boosted

    min_len = getattr(CFG, "wordcloud_min_word_len", 3)
    weighting = getattr(CFG, "wordcloud_weighting", "raw")
    if weighting == "tfidf":
        weighting = "tfidf_topics"  # backward compat
    pos_filter = getattr(CFG, "wordcloud_pos_filter", "all")
    exclude_shared = getattr(CFG, "wordcloud_exclude_shared", False)
    min_freq = getattr(CFG, "wordcloud_min_freq", None)
    max_freq = getattr(CFG, "wordcloud_max_freq", None)
    topic_boost = getattr(CFG, "wordcloud_topic_boost", False)
    topic_boost_strength = float(getattr(CFG, "wordcloud_topic_boost_strength", 1.0))

    if min_freq is not None:
        min_freq = int(min_freq)
        if min_freq < 1:
            raise ValueError("plot_config.wordcloud_min_freq must be >= 1 or None")
    if max_freq is not None:
        max_freq = int(max_freq)
        if max_freq < 1:
            raise ValueError("plot_config.wordcloud_max_freq must be >= 1 or None")
    if (min_freq is not None) and (max_freq is not None) and (max_freq < min_freq):
        raise ValueError("plot_config.wordcloud_max_freq must be >= plot_config.wordcloud_min_freq")

    if weighting == "noun_tfidf_contrast_bg_english":
        print("  [wordcloud] Using noun-only TF-IDF contrast with background + English filters")
        return _prepare_noun_tfidf_contrast_wordcloud_data(wc_texts)

    # Build Bartlett exclusion set
    bartlett_excl = set(BARTLETT_STORY.translate(
        str.maketrans(stringp.punctuation, ' ' * len(stringp.punctuation))
    ).lower().split()) | EXCLUDE_OFFENSIVE_WORDS

    # Tokenize each topic
    topic_tokens: dict[str, list[str]] = {}
    for topic, texts in wc_texts.items():
        joined = " ".join(texts) if texts else ""
        tokens = _wc_tokenize(joined, bartlett_excl, min_len)
        tokens = _wc_pos_filter(tokens, pos_filter)
        topic_tokens[topic] = tokens

    # Cross-topic exclusion: remove words present in ALL topics
    if exclude_shared and len(topic_tokens) > 1:
        word_sets = [set(tokens) for tokens in topic_tokens.values()]
        shared = set.intersection(*word_sets) if word_sets else set()
        if shared:
            print(f"  [wordcloud] Excluding {len(shared)} words shared across all topics")
            topic_tokens = {
                t: [w for w in tokens if w not in shared]
                for t, tokens in topic_tokens.items()
            }

    # Per-topic frequency filtering: drop words that are too rare/common within each topic.
    if (min_freq is not None) or (max_freq is not None):
        print(f"  [wordcloud] Frequency filter: min={min_freq}, max={max_freq}")
        filtered: dict[str, list[str]] = {}
        for topic, tokens in topic_tokens.items():
            if not tokens:
                filtered[topic] = tokens
                continue
            tf = Counter(tokens)
            filtered[topic] = [
                w for w in tokens
                if (min_freq is None or tf[w] >= min_freq)
                and (max_freq is None or tf[w] <= max_freq)
            ]
        topic_tokens = filtered

    # Build output
    result: dict[str, dict[str, float]] | dict[str, str]
    if weighting == "tfidf_topics":
        # TF-IDF wrt 5 categories: IDF = how rare across our topic recalls
        n_topics = len(topic_tokens)
        all_words = set()
        for tokens in topic_tokens.values():
            all_words.update(tokens)
        doc_freq: dict[str, int] = {}
        for word in all_words:
            doc_freq[word] = sum(
                1 for tokens in topic_tokens.values() if word in set(tokens)
            )

        result = {}
        for topic, tokens in topic_tokens.items():
            if not tokens:
                result[topic] = {}
                continue
            tf = Counter(tokens)
            max_tf = max(tf.values())
            weights = {}
            for word, count in tf.items():
                idf = math.log(1 + n_topics / (1 + doc_freq.get(word, 0)))
                weights[word] = (count / max_tf) * idf
            result[topic] = weights

    elif weighting == "tfidf_english":
        # TF-IDF wrt English language: IDF from global word frequency (wordfreq)
        try:
            from wordfreq import word_frequency
        except ImportError:
            print("[wordcloud] wordfreq not installed; falling back to raw weighting.")
            result = {t: " ".join(tokens) for t, tokens in topic_tokens.items()}
        else:
            eps = 1e-12

            def _idf_english(w: str) -> float:
                f = max(word_frequency(w, "en"), eps)
                return math.log(1.0 / f)

            result = {}
            for topic, tokens in topic_tokens.items():
                if not tokens:
                    result[topic] = {}
                    continue
                tf = Counter(tokens)
                max_tf = max(tf.values())
                weights = {}
                for word, count in tf.items():
                    idf = _idf_english(word)
                    weights[word] = (count / max_tf) * idf
                result[topic] = weights

    elif weighting == "topic_contrast":
        # Log-ratio: how over-represented a word is in this topic vs others.
        # p(w|topic) / p(w|all_topics)  — words with ratio > 1 are distinctive.
        # We use log2 and clamp negatives to 0 so only over-represented words appear.
        all_tokens_flat = []
        for tokens in topic_tokens.values():
            all_tokens_flat.extend(tokens)
        global_counts = Counter(all_tokens_flat)
        global_total = len(all_tokens_flat) or 1

        result = {}
        for topic, tokens in topic_tokens.items():
            if not tokens:
                result[topic] = {}
                continue
            tf = Counter(tokens)
            topic_total = len(tokens)
            weights = {}
            for word, count in tf.items():
                p_topic = count / topic_total
                p_global = global_counts[word] / global_total
                ratio = math.log2(p_topic / p_global) if p_global > 0 else 0.0
                if ratio > 0:
                    weights[word] = ratio * count  # scale by count for visual weight
            result[topic] = weights

    elif weighting == "contrastive_bg":
        # Contrastive background: keep only recall words that ALSO exist in the
        # topic's Wikipedia background corpus, then weight by
        # recall_count × log(1 + bg_this / (bg_other_mean + 1)).
        # This surfaces words that genuinely leaked from background training.
        from utils import load_topic_corpus_wiki

        print("  [wordcloud] Loading background corpora for contrastive_bg weighting...")
        topics = list(topic_tokens.keys())
        bg_docs = load_topic_corpus_wiki(
            topics,
            seed=42,
            articles_per_topic=getattr(CFG, "articles_per_topic", 1000),
            chars_per_article=getattr(CFG, "chars_per_article", 1000),
            use_tfidf_filter=getattr(CFG, "use_tfidf_filter", True),
        )

        # Build background vocab per topic
        bg_vocabs: dict[str, Counter] = {}
        for topic in topics:
            c: Counter = Counter()
            for doc in bg_docs.get(topic, []):
                c.update(_wc_tokenize(str(doc), bartlett_excl, min_len))
            bg_vocabs[topic] = c

        n_topics = len(topics)
        result = {}
        for topic, tokens in topic_tokens.items():
            if not tokens:
                result[topic] = {}
                continue
            tf = Counter(tokens)
            bg_this = bg_vocabs.get(topic, Counter())
            weights = {}
            for word, recall_count in tf.items():
                if recall_count < 2:
                    continue  # need at least 2 occurrences
                if word not in bg_this:
                    continue  # word not in this topic's background — skip
                this_freq = bg_this[word]
                other_mean = sum(bg_vocabs.get(t, Counter()).get(word, 0)
                                 for t in topics if t != topic) / max(n_topics - 1, 1)
                if this_freq <= other_mean + 1:
                    continue  # not over-represented in this background
                contrast = this_freq / (other_mean + 1)
                weights[word] = recall_count * math.log(1 + contrast)
            result[topic] = weights

    else:
        # Raw frequency
        result = {t: " ".join(tokens) for t, tokens in topic_tokens.items()}

    if topic_boost:
        topics = list(topic_tokens.keys())
        training_counts = _topic_training_counts(topics, min_len)
        if isinstance(result, dict) and result and all(isinstance(v, str) for v in result.values()):
            # Raw-text mode: convert to per-topic counts so we can apply multiplicative boosting.
            base_weights: dict[str, dict[str, float]] = {}
            for t in topics:
                tf = Counter(topic_tokens.get(t, []))
                base_weights[t] = {w: float(c) for w, c in tf.items()}
        else:
            base_weights = result  # type: ignore[assignment]

        boosted = _apply_topic_boost(base_weights, training_counts, topic_boost_strength)
        return boosted

    return result


def plot_wordcloud(ax, topic, text_or_freqs=None):
    """Plot word cloud for semantic intrusions.

    ``text_or_freqs`` can be:
      - a ``str`` (space-joined words; WordCloud counts them)
      - a ``dict[str, float]`` (pre-computed weights for TF-IDF mode)
      - ``None`` (placeholder)
    """
    if not HAS_WORDCLOUD or text_or_freqs is None:
        ax.text(0.5, 0.5, topic, ha='center', va='center', transform=ax.transAxes,
                fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
        return

    wc_obj = WordCloud(width=300, height=300, scale=WORDCLOUD_RENDER_SCALE,
                       relative_scaling=WORDCLOUD_RELATIVE_SCALING, normalize_plurals=True,
                       max_font_size=WORDCLOUD_MAX_FONT_SIZE,
                       min_font_size=WORDCLOUD_MIN_FONT_SIZE, background_color='white',
                       colormap=WORDCLOUD_COLORMAP, collocations=False, random_state=7,
                       prefer_horizontal=0.9)

    if isinstance(text_or_freqs, dict):
        # TF-IDF mode: generate from pre-computed frequencies
        if not text_or_freqs:
            ax.text(0.5, 0.5, f'{topic}\n(no intrusions)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            return
        ax._source_data_rows = [
            {"topic": topic, "word": word, "weight": float(weight)}
            for word, weight in sorted(text_or_freqs.items(), key=lambda item: item[1], reverse=True)
        ]
        wc_obj.generate_from_frequencies(text_or_freqs)
    else:
        # Raw text mode
        if not text_or_freqs.strip():
            ax.text(0.5, 0.5, f'{topic}\n(no intrusions)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            return
        wc_obj.generate(text_or_freqs)
        ax._source_data_rows = [
            {"topic": topic, "word": word, "weight": float(weight)}
            for word, weight in sorted(wc_obj.words_.items(), key=lambda item: item[1], reverse=True)
        ]

    ax.imshow(wc_obj, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(topic, fontsize=12, pad=0)
    ax._source_data_panel = "g"
    ax._source_data_title = f"Wordcloud {topic}"


# ============================================================================
# MAIN
# ============================================================================
def main(
    *,
    overwrite: bool = False,
    raykov_dir: str | None = None,
    results_dir: str | None = None,
    enc_con_dir: str | None = None,
    figures_dir: str | None = None,
):
    global RAYKOV_DIR, BARTLETT_DIR, BARTLETT_ANALYSIS, FIGURES_DIR

    if raykov_dir is not None:
        RAYKOV_DIR = Path(raykov_dir)
    if results_dir is not None:
        BARTLETT_DIR = Path(results_dir)
        BARTLETT_ANALYSIS = BARTLETT_DIR / "_analysis"
    if enc_con_dir is not None:
        CFG.enc_vs_con_dir = str(Path(enc_con_dir).resolve())
    if figures_dir is not None:
        FIGURES_DIR = Path(figures_dir)

    print(f"Config: results_dir={BARTLETT_DIR}")
    print(f"  PCA / cosine bar: epoch={CFG.pca_epoch}, temp={CFG.pca_temp}, n={CFG.pca_num_samples}")
    print(f"  Word clouds:      epoch={CFG.wordcloud_epoch}, temp={CFG.wordcloud_temp}, n={CFG.wordcloud_num_samples}")
    if overwrite:
        print("  ** OVERWRITE mode: ignoring all cached CSVs and generation caches **")
    print()

    # ------------------------------------------------------------------
    # Non-Bartlett data (Raykov, human)
    # ------------------------------------------------------------------
    print("Loading Raykov & human data...")
    raykov_data = load_raykov_data()
    raykov_humans = load_raykov_human_data()
    epoch_logs = load_bartlett_epoch_logs()

    pre_omex = None
    post_omex = None
    if raykov_data:
        story_map = {}
        for cat in ("typical", "incomplete", "updated"):
            for i, s in enumerate(raykov_data["stories"].get(cat, [])):
                story_map[f"{cat}_{i:04d}"] = s
        pre_omex = _compute_raykov_omex(raykov_data["pre"], story_map, "pre")
        post_omex = _compute_raykov_omex(raykov_data["post"], story_map, "post")
        print(f"  Raykov: {len(raykov_data['pre'])} pre, {len(raykov_data['post'])} post")
    if raykov_humans:
        pre_n = len(raykov_humans.get("immediate", {}).get("incomplete", {}).get("om", []))
        post_n = len(raykov_humans.get("delayed", {}).get("incomplete", {}).get("om", []))
        print(f"  Raykov humans: immediate n={pre_n}, delayed n={post_n}")
    if epoch_logs:
        print(f"  Epoch logs: {list(epoch_logs.keys())}")

    # ------------------------------------------------------------------
    # Bartlett samples for PCA + cosine bar  (panel a, c)
    # ------------------------------------------------------------------
    pca_ckpt_dir = getattr(CFG, "pca_ckpt_dir", None)

    if pca_ckpt_dir:
        # Load from external checkpoint_samples dir — compute stats using
        # mean-embedding approach (consistent with check_recall_vs_category_mean.py)
        pca_ckpt_path = Path(pca_ckpt_dir)
        cache_key = f"{int(CFG.pca_epoch)}_{float(CFG.pca_temp)}"
        print(f"Loading PCA / cosine bar samples from {pca_ckpt_path} (key={cache_key})...")
        from sentence_transformers import SentenceTransformer
        from utils import load_topic_corpus_wiki
        _model_name = getattr(CFG, "embedding_model", "all-MiniLM-L6-v2")
        _pca_embedder = SentenceTransformer(_model_name)
        bartlett_emb = _pca_embedder.encode(_trim([BARTLETT_STORY]), show_progress_bar=False)[0]

        _pca_topic_docs = load_topic_corpus_wiki(
            CFG.topics, seed=42,
            articles_per_topic=CFG.articles_per_topic,
            chars_per_article=CFG.chars_per_article,
            use_tfidf_filter=getattr(CFG, "use_tfidf_filter", True),
        )

        all_rows: list[dict] = []
        stats_rows: list[dict] = []
        for topic in CFG.topics:
            f = pca_ckpt_path / f"{topic}_checkpoint_samples.json"
            if not f.exists():
                print(f"    [{topic}] No file in ckpt dir, skipping")
                continue
            cached = json.loads(f.read_text())
            texts = [t for t in cached.get(cache_key, []) if isinstance(t, str) and t.strip()]
            if not texts:
                print(f"    [{topic}] No texts for key {cache_key}")
                continue
            print(f"    [{topic}] {len(texts)} texts from {cache_key}")

            bg_docs = _pca_topic_docs.get(topic, [])
            bg_embs = _pca_embedder.encode(_trim(bg_docs), show_progress_bar=False)
            bg_center = np.mean(bg_embs, axis=0)

            text_embs = _pca_embedder.encode(_trim(texts), show_progress_bar=False)
            cos_each = [float(cos_dist(e, bg_center)) for e in text_embs]

            for i, txt in enumerate(texts):
                all_rows.append({"topic": topic, "sample_idx": i + 1,
                                 "temperature": CFG.pca_temp, "text": txt,
                                 "cos_to_bg": cos_each[i]})

            cos_mean, cos_sem = _aggregate_cos(text_embs, bg_center)
            bart_to_bg = float(cos_dist(bartlett_emb, bg_center))
            stats_rows.append({
                "topic": topic, "n_samples": len(texts), "n_bg_docs": len(bg_docs),
                "cos_bart_to_bg": bart_to_bg,
                "cos_samples_to_bg_mean": cos_mean,
                "cos_samples_to_bg_sem": cos_sem,
            })

        samples_df = pd.DataFrame(all_rows) if all_rows else None
        bartlett_stats = pd.DataFrame(stats_rows).sort_values("topic") if stats_rows else None
    else:
        print("Ensuring Bartlett samples for PCA / cosine bar chart...")
        samples_df, bartlett_stats = _ensure_samples(
            BARTLETT_DIR, CFG.topics,
            epoch=CFG.pca_epoch, temp=CFG.pca_temp,
            num_samples=CFG.pca_num_samples,
            max_new_tokens=CFG.max_new_tokens,
            bartlett_text=BARTLETT_STORY,
            overwrite=overwrite,
        )

    if bartlett_stats is not None and len(bartlett_stats) > 0:
        print(f"  Bartlett stats: {len(bartlett_stats)} topics")
        for _, r in bartlett_stats.iterrows():
            orig_sim = 1 - r['cos_bart_to_bg']
            rec_sim = 1 - r['cos_samples_to_bg_mean']
            print(f"    {r['topic']}: orig_sim={orig_sim:.4f} rec_sim={rec_sim:.4f} "
                  f"{'closer' if rec_sim > orig_sim else 'FARTHER'}")
    else:
        bartlett_stats = None

    # ------------------------------------------------------------------
    # Bartlett samples for word clouds  (panel g)
    # ------------------------------------------------------------------
    wc_mode = getattr(CFG, "wordcloud_weighting", "raw")
    wc_use_all_ckpts = getattr(CFG, "wordcloud_all_ckpts", False)
    wc_ckpt_dir = getattr(CFG, "wordcloud_ckpt_dir", None)

    if wc_use_all_ckpts:
        # Use ALL epoch-temp recalls from the checkpoint cache
        ckpt_dir = Path(wc_ckpt_dir) if wc_ckpt_dir else None
        print(f"Loading ALL checkpoint recalls for word clouds (ckpt_dir={ckpt_dir or 'default'})...")
        wc_texts = _load_all_ckpt_texts(ckpt_cache_dir=ckpt_dir, topics=CFG.topics)
    else:
        print("Ensuring Bartlett samples for word clouds...")
        wc_texts = _ensure_wordcloud_samples(
            BARTLETT_DIR, CFG.topics,
            epoch=CFG.wordcloud_epoch, temp=CFG.wordcloud_temp,
            num_samples=CFG.wordcloud_num_samples,
            max_new_tokens=CFG.max_new_tokens,
            bartlett_text=BARTLETT_STORY,
            overwrite=overwrite,
        )

    wc_max_chars = max(1, len(BARTLETT_STORY))
    wc_texts = {t: [str(x)[:wc_max_chars] for x in xs] for t, xs in wc_texts.items()}
    print(f"  [wordcloud] Truncating recalls to {wc_max_chars} chars, len(Bartlett)")
    print("Processing word cloud data...")
    wc_pos = getattr(CFG, "wordcloud_pos_filter", "all")
    wc_excl = getattr(CFG, "wordcloud_exclude_shared", False)
    wc_boost = getattr(CFG, "wordcloud_topic_boost", False)
    print(f"  weighting={wc_mode}, pos_filter={wc_pos}, exclude_shared={wc_excl}, topic_boost={wc_boost}")
    wc_processed = prepare_wordcloud_data(wc_texts)

    # ------------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------------
    print("Creating figure...")
    fig = plt.figure(figsize=(14, 9.5))

    width_ratios = [0.7, 0.7, 1, 1, 1]
    h2, h3 = 0.018, 0.004
    left_frac_w = (width_ratios[0] + width_ratios[1]) / sum(width_ratios)
    fig_w, fig_h = fig.get_size_inches()
    fig_aspect = fig_w / fig_h if fig_h else 1.0
    denom = 1 - left_frac_w * fig_aspect
    top_h = (left_frac_w * fig_aspect * (h2 + h3)) / denom if denom > 1e-6 else 1.0
    height_ratios = [top_h / 2, top_h / 2, h2, h3]

    gs = gridspec.GridSpec(
        4, 5, figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.0025, wspace=0.4,
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)

    # Load embedder once for PCA
    embedder = None
    if HAS_SBERT and samples_df is not None and len(samples_df) > 0:
        print("  Loading sentence embedder...")
        from sentence_transformers import SentenceTransformer
        _model_name = getattr(CFG, "embedding_model", "all-MiniLM-L6-v2")
        embedder = SentenceTransformer(_model_name)
    
    # ========================================================================
    # ROW 1-2: PCA (left) + 2x3 grid on the right
    # ========================================================================
    ax_pca = fig.add_subplot(gs[0:2, 0:2])  # PCA spans first 2 cols, 2 rows
    plot_pca_embeddings(ax_pca, samples_df, embedder)
    ax_pca.set_title('a)', fontsize=12, loc='center')

    gs_right_outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2, 2:], hspace=0.55)
    gs_right_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_right_outer[0, 0], wspace=0.30, width_ratios=[1.15, 1.1])
    gs_right_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right_outer[1, 0], wspace=0.45)

    ax_br = fig.add_subplot(gs_right_top[0, 0])
    plot_bergman_roediger(ax_br)
    ax_br.set_title('b)', fontsize=12, loc='center')

    ax_cos = fig.add_subplot(gs_right_top[0, 1])
    plot_cosine_distances(ax_cos, bartlett_stats)
    ax_cos.set_title('c)', fontsize=12, loc='center')

    # ========================================================================
    # ROW 2 (right): New words + cosine-to-original pre/post
    # ========================================================================
    ax_temp = fig.add_subplot(gs_right_bottom[0, 0])  # New words temp
    plot_new_words_vs_temp(ax_temp, epoch_logs)
    ax_temp.set_title('d)', fontsize=12, loc='center')

    ax_epoch = fig.add_subplot(gs_right_bottom[0, 1])  # New words epoch
    plot_new_words_vs_epoch(ax_epoch, epoch_logs)
    ax_epoch.set_title('e)', fontsize=12, loc='center')

    ax_cos_prepost = fig.add_subplot(gs_right_bottom[0, 2])
    plot_cosine_distance_encoded_consolidated(ax_cos_prepost, embedder=embedder)
    ax_cos_prepost.set_title('f)', fontsize=12, loc='center')
    
    # ========================================================================
    # ROW 3: Word clouds (using config wordcloud_epoch / wordcloud_temp)
    # ========================================================================
    topics = CFG.topics
    wc_row_bbox = gs[2, :].get_position(fig)
    bottom_row_bbox = gs[3, :].get_position(fig)

    wc_left = 0.04
    wc_right = 0.96
    n_wc = len(topics)
    wc_wspace = 0.02
    wc_total_w = wc_right - wc_left
    wc_ax_w = wc_total_w / (n_wc + (n_wc - 1) * wc_wspace)
    wc_gap = wc_ax_w * wc_wspace
    wc_height = wc_ax_w * 1.32
    wc_gap_to_bottom = 0.05
    wc_bottom = float(bottom_row_bbox.y1) + wc_gap_to_bottom

    fig.text(0.5, wc_bottom + wc_height + 0.025, 'g)',
             fontsize=12, ha='center', va='bottom')

    for i, topic in enumerate(topics):
        ax_wc = fig.add_axes([
            wc_left + i * (wc_ax_w + wc_gap),
            wc_bottom,
            wc_ax_w,
            wc_height,
        ])
        wc_data = wc_processed.get(topic)
        # Fallback to raw text from samples_df if no processed data
        if not wc_data and samples_df is not None and len(samples_df) > 0:
            if topic in samples_df['topic'].values:
                topic_texts = samples_df[samples_df['topic'] == topic]['text'].dropna().tolist()
                if topic_texts:
                    wc_data = " ".join(topic_texts[:50])
        plot_wordcloud(ax_wc, topic, wc_data)
    
    # ========================================================================
    # ROW 4: Human (Raykov) + Model (pre/post consolidation), 4 equal-width subplots
    # ========================================================================
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[3, :], wspace=0.10)

    # g) Human: short (left) vs long (right) retention intervals
    ax_g_short = fig.add_subplot(gs_bottom[0, 0])
    plot_omissions_extensions(
        ax_g_short,
        "Short delay",
        data=(raykov_humans.get("immediate") if raykov_humans else None),
        show_ylabel=True,
        show_legend=True,
        show_points=True,
    )
    ax_g_short._source_data_panel = "h"
    ax_g_short._source_data_title = "Human data Short delay"

    ax_g_long = fig.add_subplot(gs_bottom[0, 1])
    plot_omissions_extensions(
        ax_g_long,
        "Long delay",
        data=(raykov_humans.get("delayed") if raykov_humans else None),
        show_ylabel=False,
        show_legend=False,
        show_points=True,
    )
    ax_g_long._source_data_panel = "h"
    ax_g_long._source_data_title = "Human data Long delay"

    # i) Model: before (short delay) vs after (long delay) consolidation
    ax_h_short = fig.add_subplot(gs_bottom[0, 2])
    plot_omissions_extensions(
        ax_h_short,
        "Short delay",
        data=pre_omex,
        show_ylabel=False,
        show_legend=True,
        show_points=False,
    )
    ax_h_short._source_data_panel = "i"
    ax_h_short._source_data_title = "Model data Short delay"

    ax_h_long = fig.add_subplot(gs_bottom[0, 3])
    plot_omissions_extensions(
        ax_h_long,
        "Long delay",
        data=post_omex,
        show_ylabel=False,
        show_legend=False,
        show_points=False,
    )
    ax_h_long._source_data_panel = "i"
    ax_h_long._source_data_title = "Model data Long delay"
    
    # Adjust bottom row: shift right to align margins, and increase height by 1.5x
    for ax in [ax_g_short, ax_g_long, ax_h_short, ax_h_long]:
        pos = ax.get_position()
        new_height = pos.height * 1.95
        # Extend downward (keep top edge, move bottom down)
        new_y0 = pos.y0 - (new_height - pos.height)
        ax.set_position([pos.x0 + 0.02, new_y0, pos.width, new_height])
    
    # Add "h) Human data" and "i) Model data" headings above the subplot pairs
    # Get positions after adjustment
    pos_g_short = ax_g_short.get_position()
    pos_g_long = ax_g_long.get_position()
    pos_h_short = ax_h_short.get_position()
    pos_h_long = ax_h_long.get_position()
    
    # Human data heading - centered above first two plots
    human_center_x = (pos_g_short.x0 + pos_g_long.x1) / 2
    human_top_y = pos_g_short.y1 + 0.02
    fig.text(human_center_x, human_top_y, 'h) Human data', fontsize=12, ha='center', va='bottom')
    
    # Model data heading - centered above last two plots
    model_center_x = (pos_h_short.x0 + pos_h_long.x1) / 2
    model_top_y = pos_h_short.y1 + 0.02
    fig.text(model_center_x, model_top_y, 'i) Model data', fontsize=12, ha='center', va='bottom')
    
    # Save
    out_path = FIGURES_DIR / "Figure 5.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_figure_source_data(fig, "Figure 5")
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collate narrative simulation figures.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Ignore cached CSVs and generation caches; regenerate all data from models/checkpoints.",
    )
    parser.add_argument(
        "--raykov_dir", type=str, default=None,
        help="Override Raykov data directory (default: output_raykov_.../data)",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override Bartlett results directory (default from plot_config.py)",
    )
    parser.add_argument(
        "--enc_con_dir", type=str, default=None,
        help="Override encoding-vs-consolidation directory",
    )
    parser.add_argument(
        "--figures_dir", type=str, default=None,
        help="Override output figure directory (default: ../../figures)",
    )
    args = parser.parse_args()
    main(
        overwrite=args.overwrite,
        raykov_dir=args.raykov_dir,
        results_dir=args.results_dir,
        enc_con_dir=args.enc_con_dir,
        figures_dir=args.figures_dir,
    )
