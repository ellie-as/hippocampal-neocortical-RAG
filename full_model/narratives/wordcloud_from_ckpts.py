#!/usr/bin/env python3
r"""
Generate wordclouds from checkpoint samples (e.g. Downloads/latest ckpts).

Default: one wordcloud per topic, aggregating all recalled text across epochs and temps.

Usage:
    python wordcloud_from_ckpts.py --ckpt_dir ~/Downloads/latest\ ckpts
    python wordcloud_from_ckpts.py --per_epoch_temp  # per epoch/temp instead of aggregate
"""
from __future__ import annotations

import argparse
import json
import string as _string
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Ensure narratives is on path when run as script
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from utils import EXCLUDE_OFFENSIVE_WORDS

WORDCLOUD_RENDER_SCALE = 2
WORDCLOUD_MAX_FONT_SIZE = 54
WORDCLOUD_MIN_FONT_SIZE = 8
WORDCLOUD_RELATIVE_SCALING = 0.30
WORDCLOUD_SAVE_DPI = 300


def _exclusion_from_bartlett(txt: str) -> set:
    trans = str.maketrans(_string.punctuation, " " * len(_string.punctuation))
    tokens = txt.translate(trans).lower().split()
    return set(tokens + ["s"]) | EXCLUDE_OFFENSIVE_WORDS


def _pre_wcloud(text: str, excl: set, english_only: bool = False) -> str:
    trans = str.maketrans(_string.punctuation, " " * len(_string.punctuation))
    tokens = [w for w in text.translate(trans).lower().split() if w not in excl]
    if english_only:
        try:
            from wordfreq import word_frequency
            tokens = [w for w in tokens if word_frequency(w, "en") > 0]
        except ImportError:
            pass
    return " ".join(tokens)


def _wordcloud(joined_text: str, excl: set, out_path: Path, title: str, english_only: bool = False):
    proc = _pre_wcloud(joined_text, excl, english_only=english_only)
    if not proc.strip():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No intrusion words", ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=WORDCLOUD_SAVE_DPI)
        plt.close(fig)
        return
    wc = WordCloud(
        width=400,
        height=400,
        scale=WORDCLOUD_RENDER_SCALE,
        relative_scaling=WORDCLOUD_RELATIVE_SCALING,
        normalize_plurals=True,
        max_font_size=WORDCLOUD_MAX_FONT_SIZE,
        min_font_size=WORDCLOUD_MIN_FONT_SIZE,
        background_color="white",
        colormap="plasma",
    ).generate(proc)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=WORDCLOUD_SAVE_DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Wordclouds from checkpoint samples")
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=str(Path.home() / "Downloads" / "latest ckpts"),
        help="Directory containing {Topic}_checkpoint_samples.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: ckpt_dir/wordclouds)",
    )
    ap.add_argument(
        "--bartlett_path",
        type=str,
        default=str(HERE.parent / "data" / "bartlett.txt"),
        help="Path to Bartlett story for exclusion set",
    )
    ap.add_argument(
        "--per_epoch_temp",
        action="store_true",
        help="Generate per epoch/temp wordclouds (default: aggregate all)",
    )
    ap.add_argument(
        "--english-only",
        action="store_true",
        help="Keep only words in the English dictionary (requires wordfreq)",
    )
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else ckpt_dir / "wordclouds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Bartlett for exclusion
    from utils import load_bartlett
    bartlett = load_bartlett(args.bartlett_path)
    excl = _exclusion_from_bartlett(bartlett)

    # Find checkpoint sample files (handle "Nature (1)" style names)
    sample_files = list(ckpt_dir.glob("*checkpoint_samples*.json"))
    if not sample_files:
        print(f"No checkpoint sample JSONs found in {ckpt_dir}")
        return

    topics_data: dict[str, dict[str, list[str]]] = {}
    for f in sample_files:
        name = f.name
        if "checkpoint_samples" in name:
            topic = name.split("checkpoint_samples")[0].strip("_ ").rstrip(" (1)")
            try:
                data = json.loads(f.read_text())
                topics_data[topic] = data
                print(f"Loaded {topic}: {len(data)} epoch-temp keys")
            except Exception as e:
                print(f"Skip {f.name}: {e}")

    if not topics_data:
        print("No valid data loaded")
        return

    # Collect all epoch_temp keys and sort
    all_keys = set()
    for data in topics_data.values():
        all_keys.update(k for k in data if "_" in k)
    # Sort by (epoch, temp)
    def sort_key(k: str):
        parts = k.split("_")
        if len(parts) != 2:
            return (0, 0.0)
        try:
            return (int(parts[0]), float(parts[1]))
        except ValueError:
            return (0, 0.0)

    if args.per_epoch_temp:
        sorted_keys = sorted(all_keys, key=sort_key)
        epochs = sorted({sort_key(k)[0] for k in sorted_keys})
        temps = sorted({sort_key(k)[1] for k in sorted_keys})
        print(f"Epochs: {epochs}, Temps: {temps}")
        for topic, data in topics_data.items():
            topic_out = out_dir / topic
            topic_out.mkdir(parents=True, exist_ok=True)
            for key in sorted_keys:
                if key not in data:
                    continue
                samples = data[key]
                if not isinstance(samples, list):
                    continue
                texts = [t for t in samples if isinstance(t, str) and t.strip()]
                if not texts:
                    continue
                joined = " ".join(texts)
                out_path = topic_out / f"wordcloud_ep{sort_key(key)[0]}_temp{sort_key(key)[1]}.png"
                title = f"{topic} — epoch={sort_key(key)[0]} temp={sort_key(key)[1]}"
                _wordcloud(joined, excl, out_path, title, english_only=args.english_only)
                print(f"  {out_path.relative_to(out_dir)}")
    else:
        # Aggregate all epochs and temps per topic — one wordcloud per topic
        for topic, data in topics_data.items():
            all_texts = []
            for key, samples in data.items():
                if "_" not in key or not isinstance(samples, list):
                    continue
                all_texts.extend(t for t in samples if isinstance(t, str) and t.strip())
            joined = " ".join(all_texts)
            out_path = out_dir / f"wordcloud_{topic}.png"
            _wordcloud(joined, excl, out_path, title=f"{topic} (all epochs & temps)", english_only=args.english_only)
            print(f"  {out_path.name}")

    print(f"\nWordclouds saved to {out_dir}")


if __name__ == "__main__":
    main()
