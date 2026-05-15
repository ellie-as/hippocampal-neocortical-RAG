import os
import json
import time
import argparse
import pandas as pd
import openai
from pathlib import Path
from scipy.stats import f_oneway, ttest_rel, ttest_ind
import matplotlib.pyplot as plt

# Script directory for output paths
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_NFRD_ROOTS = [
    Path(os.environ["NFRD_ROOT"]) if os.environ.get("NFRD_ROOT") else None,
    SCRIPT_DIR.parents[2] / "data" / "Naturalistic-Free-Recall-Dataset",
    Path("/Users/eleanorspens/PycharmProjects/Naturalistic-Free-Recall-Dataset"),
]
client = None

SYSTEM_PROMPT = """Your task is score text on three metrics: how concrete (vs abstract) it is, how rich in detail it is, and how specific (vs general) it is.

Return ONLY a JSON dictionary with 3 keys, each a float 0-1:

{
  "concrete_vs_abstract": 0-1,
  "rich_vs_poor_details": 0-1,
  "specific_vs_general":  0-1
}

A higher score corresponds to more concrete, richer in detail, or more specific text."""

def read_file_with_encoding(filepath):
    """Read file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try with error handling
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return ""

def llm_scores(text: str,
               model: str = "gpt-4o-mini",
               max_retries: int = 5) -> dict:
    """Get LLM scores for a text using OpenAI API."""
    global client
    if client is None:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text[:16_000]}
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except openai.RateLimitError:
            time.sleep(2 + attempt)          # back-off then retry
        except openai.OpenAIError as e:
            print("OpenAI error:", e)
            time.sleep(3)
    # if still failing:
    return {"concrete_vs_abstract": None,
            "rich_vs_poor_details": None,
            "specific_vs_general":  None}


def resolve_nfrd_root(nfrd_root: str | Path | None = None) -> Path:
    """Find the Naturalistic-Free-Recall-Dataset root directory."""
    candidates = [Path(nfrd_root).expanduser()] if nfrd_root else []
    candidates.extend(p for p in DEFAULT_NFRD_ROOTS if p is not None)
    for root in candidates:
        story_dir = root / "story_transcript"
        recall_dir = root / "recall_transcripts"
        if story_dir.exists() and recall_dir.exists():
            return root
    raise FileNotFoundError(
        "Could not find Naturalistic-Free-Recall-Dataset. Pass --nfrd-root "
        "or set NFRD_ROOT to a directory containing story_transcript/ and recall_transcripts/."
    )


def load_nfrd_data(limit_per_story: int = None, nfrd_root: str | Path | None = None):
    """Load NFRD original and recalled stories."""
    stories_data = []
    root = resolve_nfrd_root(nfrd_root)
    
    # Load original stories
    story_files = {
        'baseball': root / 'story_transcript' / 'baseball_transcript.txt',
        'eyespy': root / 'story_transcript' / 'eyespy_transcript.txt',
        'oregontrail': root / 'story_transcript' / 'oregontrail_transcript.txt',
        'pieman': root / 'story_transcript' / 'pieman_transcript.txt'
    }
    
    for story_name, file_path in story_files.items():
        try:
            original_text = read_file_with_encoding(file_path)
            stories_data.append({
                'story_name': story_name,
                'text': original_text,
                'type': 'original',
                'participant': 'ORIGINAL'
            })
        except Exception as e:
            print(f"Error loading {story_name} original: {e}")
    
    # Load recalled stories
    recall_dirs = {
        'baseball': root / 'recall_transcripts' / 'baseball',
        'eyespy': root / 'recall_transcripts' / 'eyespy',
        'oregontrail': root / 'recall_transcripts' / 'oregontrail',
        'pieman': root / 'recall_transcripts' / 'pieman'
    }
    
    for story_name, recall_dir in recall_dirs.items():
        if os.path.exists(recall_dir):
            story_files = sorted([f for f in os.listdir(recall_dir) if f.endswith('.txt')])
            if limit_per_story is not None and limit_per_story > 0:
                story_files = story_files[:limit_per_story]
            for filename in story_files:
                if filename.endswith('.txt'):
                    participant_id = Path(filename).name.replace(f'_{story_name}.txt', '')
                    try:
                        recall_text = read_file_with_encoding(recall_dir / filename)
                        stories_data.append({
                            'story_name': story_name,
                            'text': recall_text,
                            'type': 'recalled',
                            'participant': participant_id
                        })
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
    
    return pd.DataFrame(stories_data)


def analyze_nfrd_stories(
    limit_per_story: int = None,
    nfrd_root: str | Path | None = None,
    dry_run: bool = False,
    model: str = "gpt-4o-mini",
):
    """Main analysis function for NFRD stories."""
    print("Loading NFRD stories...")
    df = load_nfrd_data(limit_per_story=limit_per_story, nfrd_root=nfrd_root)
    
    print(f"Loaded {len(df)} stories:")
    print(f"  Original: {len(df[df['type'] == 'original'])}")
    print(f"  Recalled: {len(df[df['type'] == 'recalled'])}")
    
    # Check if we already have scores
    scores_file = SCRIPT_DIR / 'nfrd_llm_scores.csv'
    if os.path.exists(scores_file):
        print(f"Loading existing scores from {scores_file}")
        score_df = pd.read_csv(scores_file)
        # Merge with original data
        df = df.merge(score_df, on=['story_name', 'type', 'participant'], how='left')

        # Identify rows missing any scores and compute only for those
        needed_cols = ['concrete_vs_abstract', 'rich_vs_poor_details', 'specific_vs_general']
        missing_mask = df[needed_cols].isna().any(axis=1)
        missing_df = df[missing_mask]

        if len(missing_df) > 0:
            print(f"No. of items missing scores: {len(missing_df)}")
            print(missing_df.groupby(["story_name", "type"]).size())
            if dry_run:
                return df
            print("This will take a while and cost money! Running LLM analysis for missing items...")
            records = []
            total = len(missing_df)
            for i, (idx, row) in enumerate(missing_df.iterrows(), start=1):
                print(f"Processing missing {i}/{total}: {row['type']} - {row['story_name']} - {row['participant']}")
                js = llm_scores(row['text'], model=model)
                js.update({
                    'story_name': row['story_name'],
                    'type': row['type'],
                    'participant': row['participant']
                })
                records.append(js)
                time.sleep(0.5)

            if records:
                new_scores = pd.DataFrame(records)
                score_df = pd.concat([score_df, new_scores], ignore_index=True)
                score_df = score_df.drop_duplicates(subset=['story_name', 'type', 'participant'], keep='last')
                score_df.to_csv(scores_file, index=False)
                print(f"Updated scores saved to {scores_file}")

                # Re-merge with refreshed scores
                df = df.drop(columns=needed_cols, errors='ignore')
                df = df.merge(score_df, on=['story_name', 'type', 'participant'], how='left')
    else:
        print("No existing scores found. Running LLM analysis...")
        print("This will take a while and cost money!")
        if dry_run:
            return df
        
        # Get LLM scores for all stories
        records = []
        total = len(df)
        
        for idx, row in df.iterrows():
            print(f"Processing {idx+1}/{total}: {row['type']} - {row['story_name']}")
            js = llm_scores(row['text'], model=model)
            js.update({
                'story_name': row['story_name'],
                'type': row['type'],
                'participant': row['participant']
            })
            records.append(js)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        score_df = pd.DataFrame(records)
        score_df.to_csv(scores_file, index=False)
        print(f"Saved scores to {scores_file}")
        
        # Merge scores with original data
        df = df.merge(score_df, on=['story_name', 'type', 'participant'], how='left')
    
    return df

def create_visualization(df):
    """Create the visualization comparing original vs recalled stories."""
    # Filter to only include stories with valid scores
    valid_df = df.dropna(subset=['concrete_vs_abstract', 'rich_vs_poor_details', 'specific_vs_general'])
    
    print(f"Valid scores for visualization: {len(valid_df)} stories")
    print(f"  Original: {len(valid_df[valid_df['type'] == 'original'])}")
    print(f"  Recalled: {len(valid_df[valid_df['type'] == 'recalled'])}")
    
    metrics = [
        ("concrete_vs_abstract", "Concreteness"),
        ("rich_vs_poor_details", "Richness in detail"),
        ("specific_vs_general",  "Specificity")
    ]
    
    # Only plot original vs recalled
    groups = ["original", "recalled"]
    colors = ["tomato", "#4C78A8"]  # Using your preferred colors
    
    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharey=True)
    
    # Calculate common y-axis range across all metrics
    all_means = []
    all_sems = []
    for col, title in metrics:
        means = valid_df.groupby("type")[col].mean().reindex(groups)
        sems = valid_df.groupby("type")[col].sem().reindex(groups)
        all_means.extend(means.values)
        all_sems.extend(sems.values)
    
    # Calculate common limits
    y_min_data = float(min([m - s for m, s in zip(all_means, all_sems)]))
    y_max_data = float(max([m + s for m, s in zip(all_means, all_sems)]))
    data_range = max(0.01, y_max_data - y_min_data)
    low_pad = 0.10 * data_range
    sig_gap = 0.08 * data_range
    text_gap = 0.015 * data_range
    base_sig = y_max_data + sig_gap
    extra_headroom = 0.12 * data_range
    y_top = base_sig + text_gap + extra_headroom
    
    # Set common y-limits for all subplots
    for ax in axes:
        ax.set_ylim(y_min_data - low_pad, y_top)
    
    for ax, (col, title) in zip(axes, metrics):
        # means & SEMs
        means = valid_df.groupby("type")[col].mean().reindex(groups)
        sems  = valid_df.groupby("type")[col].sem().reindex(groups)
        
        # plot bars -------------------------------------------------------
        ax.bar(groups, means, yerr=sems, color=colors, capsize=4, alpha=.9)
        ax.set_title(title)
        if ax == axes[0]:  # Only set ylabel on the leftmost subplot
            ax.set_ylabel("Score")
        ax.tick_params(axis='x', labelrotation=20)
        
        # significance line for original vs recalled ------------------------
        g1, g2 = "original", "recalled"
        x1, x2 = groups.index(g1), groups.index(g2)
        y  = base_sig
        g1v = valid_df[valid_df.type==g1][col].dropna()
        g2v = valid_df[valid_df.type==g2][col].dropna()
        t, p = ttest_ind(g1v, g2v, equal_var=False)
        
        sym = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
        # Clamp positions to sit just below the top of the axis
        margin = 0.02 * data_range
        y_text = min(base_sig + 1.5*text_gap, y_top - margin)
        bracket_h = 0.40 * text_gap
        y_bracket_base = min(base_sig, y_text - 1.8*text_gap)
        # Draw bracket with little tails
        tail_h =  2*text_gap
        ax.plot([x1, x1, x2, x2], [y_bracket_base+bracket_h, y_bracket_base+bracket_h+tail_h, y_bracket_base+bracket_h+tail_h, y_bracket_base+bracket_h], lw=1.2, c="k")
        ax.plot([x1, x1], [y_bracket_base+bracket_h, y_bracket_base+bracket_h-tail_h], lw=1.2, c="k")
        ax.plot([x2, x2], [y_bracket_base+bracket_h, y_bracket_base+bracket_h-tail_h], lw=1.2, c="k")
        ax.text((x1+x2)/2, y_text, sym, ha="center", va="bottom")
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(SCRIPT_DIR / 'nfrd_llm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Filter to valid scores
    valid_df = df.dropna(subset=['concrete_vs_abstract', 'rich_vs_poor_details', 'specific_vs_general'])
    
    metrics = ['concrete_vs_abstract', 'rich_vs_poor_details', 'specific_vs_general']
    metric_names = ['Concreteness', 'Richness in Detail', 'Specificity']
    
    for metric, name in zip(metrics, metric_names):
        print(f"\n{name}:")
        print("-" * 30)
        
        original_scores = valid_df[valid_df['type'] == 'original'][metric]
        recalled_scores = valid_df[valid_df['type'] == 'recalled'][metric]
        
        print(f"Original (n={len(original_scores)}): {original_scores.mean():.3f} ± {original_scores.sem():.3f}")
        print(f"Recalled (n={len(recalled_scores)}): {recalled_scores.mean():.3f} ± {recalled_scores.sem():.3f}")
        
        # Statistical test
        t, p = ttest_ind(original_scores, recalled_scores, equal_var=False)
        print(f"t-test: t={t:.3f}, p={p:.4f}")
        
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"Significance: {sig}")

def main():
    """Main function."""
    print("NFRD LLM Specificity Analysis")
    print("=" * 40)
    global client
    
    parser = argparse.ArgumentParser(description="LLM-based specificity analysis for NFRD.")
    parser.add_argument("-n", "--limit-per-story", type=int, default=None,
                        help="If provided, only process the first N recall transcripts per story. Default: all.")
    parser.add_argument("--nfrd-root", type=Path, default=None,
                        help="Path to Naturalistic-Free-Recall-Dataset.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Environment variable containing the OpenAI API key.")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model used for ratings.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and report missing cached scores without making API calls.")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Update the score CSV without opening/saving the diagnostic plot.")
    args = parser.parse_args()

    if not args.dry_run:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"ERROR: Please set {args.api_key_env} before scoring missing NFRD rows.")
            return
        client = openai.OpenAI(api_key=api_key)
    
    try:
        # Load and analyze data
        df = analyze_nfrd_stories(
            limit_per_story=args.limit_per_story,
            nfrd_root=args.nfrd_root,
            dry_run=args.dry_run,
            model=args.model,
        )
        if args.dry_run:
            return
        
        # Create visualization
        if not args.skip_plot:
            fig = create_visualization(df)
        
        # Print summary statistics
        print_summary_stats(df)
        
        print(f"\nAnalysis complete! Results saved to:")
        print(f"  - nfrd_llm_scores.csv (raw scores)")
        print(f"  - nfrd_llm_analysis.png (visualization)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
