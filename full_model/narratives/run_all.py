from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PY = sys.executable

STORIES_PY   = HERE / "stories.py"
BARTLETT_PY  = HERE / "bartlett_twostage.py"
ENC_CON_PY   = HERE / "bartlett_encoding_vs_consolidation.py"
PLOT_PY      = HERE / "plot.py"

RAYKOV_DIR_DEFAULT   = "output_raykov_xrag_fixed_firstline_50"
BARTLETT_DIR_DEFAULT = "output_twostage"
BARTLETT_TXT_DEFAULT = str(HERE.parent / "data" / "bartlett.txt")
TOPICS_DEFAULT = ["Nature", "Politics", "Universe", "Sport", "Health"]


def sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)


def raykov_outputs_present(raykov_dir: Path) -> bool:
    d = raykov_dir / "data"
    return all((d / name).exists() for name in [
        "stories_prepared.pkl", "generations_pre.json", "generations_post.json"
    ])


def bartlett_models_present(results_dir: Path, topics: list[str] | None) -> bool:
    # If topics not specified, consider presence of at least one topic with model/final
    if topics:
        return all((results_dir / t / "model" / "final").exists() for t in topics)
    else:
        have = [p for p in results_dir.iterdir() if p.is_dir() and (p / "model" / "final").exists()]
        return len(have) > 0


def main():
    ap = argparse.ArgumentParser(description="End-to-end runner: simulations + plots")
    # Where to put/find things
    ap.add_argument("--raykov_dir", default=RAYKOV_DIR_DEFAULT,
                    help="Raykov output root (stories.py writes here)")
    ap.add_argument("--results_dir", default=BARTLETT_DIR_DEFAULT,
                    help="Bartlett results root (bartlett.py writes per-topic here)")
    ap.add_argument("--bartlett_path", default=BARTLETT_TXT_DEFAULT,
                    help="Path to bartlett.txt")
    ap.add_argument("--topics", nargs="*", default=TOPICS_DEFAULT,
                    help="Topics for Bartlett (default: Nature, Politics, Universe, Sport, Health)")

    # Control which stages to do
    ap.add_argument("--skip_raykov", action="store_true", help="Skip Raykov simulation")
    ap.add_argument("--skip_bartlett", action="store_true", help="Skip Bartlett/IMDB simulation")
    ap.add_argument("--skip_enc_con", action="store_true", help="Skip encoding-vs-consolidation experiment")
    ap.add_argument("--skip_plots", action="store_true", help="Skip plotting")
    ap.add_argument("--force_raykov", action="store_true",
                    help="Run Raykov even if outputs already exist")
    ap.add_argument("--force_bartlett", action="store_true",
                    help="Run Bartlett even if model/final exists")
    ap.add_argument("--force_stage2", action="store_true",
                    help="Re-run only Stage 2 (Bartlett) while reusing existing Stage 1 models")
    ap.add_argument("--force_enc_con", action="store_true",
                    help="Run encoding-vs-consolidation even if outputs exist")

    ap.add_argument("--raykov_n_typical", type=int, default=100) #50)
    ap.add_argument("--raykov_n_variants", type=int, default=100) #50)
    ap.add_argument("--raykov_seed", type=int, default=123)
    ap.add_argument("--raykov_max_new_tokens", type=int, default=300)

    ap.add_argument("--bart_stage1_epochs", type=int, default=10)
    ap.add_argument("--bart_stage2_epochs", type=int, default=10)
    ap.add_argument("--bart_batch_size", type=int, default=1)
    ap.add_argument("--bart_stage1_lr", type=float, default=2e-4)
    ap.add_argument("--bart_stage2_lr", type=float, default=5e-5)
    ap.add_argument("--bart_print_steps", type=int, default=20)
    ap.add_argument("--bart_max_new_tokens", type=int, default=500)
    ap.add_argument("--bart_use_mps", action="store_true")
    ap.add_argument("--bart_use_4bit", default=False)
    ap.add_argument("--bart_stage2_bg_replay", type=int, default=0,
                    help="Background docs to replay per epoch in Stage 2 (default 50, 0 to disable)")
    ap.add_argument("--min_new_tokens", type=int, default=-1,
                    help="Minimum generated tokens for recall (-1 = auto from Bartlett, 0 = off)")

    ap.add_argument("--articles_per_topic", type=int, default=1000)
    ap.add_argument("--chars_per_article", type=int, default=5000)
    ap.add_argument("--no_tfidf_filter", action="store_true",
                    help="Disable TF-IDF centrality filtering for Wikipedia topic articles")
    # Encoding-vs-consolidation experiment
    ap.add_argument("--enc_con_output_dir", default="bartlett_encoding_vs_consolidation",
                    help="Output directory for encoding-vs-consolidation experiment")
    ap.add_argument("--enc_con_n_samples", type=int, default=10,
                    help="Samples per condition for encoding-vs-consolidation")
    ap.add_argument("--enc_con_epochs", type=int, default=5,
                    help="Consolidation epochs for encoding-vs-consolidation")

    # Plot-specific
    ap.add_argument("--post_temps", default="0.0", help="Temps to try for Raykov post generations (plot.py)")
    ap.add_argument("--num_samples", type=int, default=10, help="Bartlett final-model samples per topic for plots")
    ap.add_argument("--temp", type=float, default=0.5, help="Bartlett sampling temperature for plots")
    ap.add_argument("--dtype", default="auto", choices=["auto","fp16","bf16","fp32"])
    ap.add_argument("--offload_dir", default=None, help="Bartlett offload dir for plot sampling (default: results/_offload)")
    # Checkpoint sampling for new words vs epoch
    ap.add_argument("--ckpt_temps", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0, 1.5],
                    help="Temperatures to sample from each checkpoint")
    ap.add_argument("--ckpt_n_samples", type=int, default=10, help="Samples per checkpoint/temp combo")
    ap.add_argument("--ckpt_temp_fixed_for_epoch", type=float, default=0.5, help="Fixed temp for epoch curve")
    ap.add_argument("--ckpt_epoch_fixed_for_temp", type=int, default=5, help="Fixed epoch for temp curve")

    # Bartlett checkpoint plotting grid + new-words defaults
    ap.add_argument("--skip_bartlett_ckpt_plots", action="store_true",
                    help="Skip Bartlett checkpoint grid plots (epoch x temp).")
    ap.add_argument("--skip_bartlett_newwords", action="store_true",
                    help="Skip Bartlett checkpoint new-word curves.")
    ap.add_argument("--bart_ckpt_epochs", nargs="+", type=int, default=[5, 10],
                    help="Epochs for Bartlett checkpoint grid plots.")
    ap.add_argument("--bart_ckpt_temps", nargs="+", type=float, default=[0.5, 0.1],
                    help="Temperatures for Bartlett checkpoint grid plots.")
    ap.add_argument("--newwords_temp_epochs", nargs="+", type=int, default=[5, 10],
                    help="Epochs to test for new-words vs temp plots.")
    ap.add_argument("--newwords_epoch_temps", nargs="+", type=float, default=[0.1, 0.5],
                    help="Temperatures to test for new-words vs epoch plots.")

    args = ap.parse_args()

    for p in (STORIES_PY, BARTLETT_PY, ENC_CON_PY, PLOT_PY):
        if not p.exists():
            print(f"ERROR: required script not found: {p}", file=sys.stderr); sys.exit(2)

    raykov_dir = Path(args.raykov_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    need_raykov = (not raykov_outputs_present(raykov_dir)) or args.force_raykov
    if args.skip_raykov:
        print("→ Skipping Raykov simulation (per flag).")
    elif need_raykov:
        print("→ Running Raykov simulation (stories.py)…")
        cmd = [
            PY, str(STORIES_PY),
            "--output_dir", str(raykov_dir),
            "--n_typical", str(args.raykov_n_typical),
            "--n_variants", str(args.raykov_n_variants),
            "--rng_seed", str(args.raykov_seed),
            "--max_new_tokens", str(args.raykov_max_new_tokens),
        ]
        sh(cmd)
    else:
        print("✓ Raykov artifacts already present — skipping (use --force_raykov to re-run).")

    need_bartlett = (not bartlett_models_present(results_dir, args.topics)) or args.force_bartlett or args.force_stage2
    if args.skip_bartlett:
        print("→ Skipping Bartlett two-stage simulation (per flag).")
    elif need_bartlett:
        print("→ Running Bartlett two-stage simulation (bartlett_twostage.py)…")
        cmd = [
            PY, str(BARTLETT_PY),
            "--output_dir", str(results_dir),
            "--bartlett_path", str(Path(args.bartlett_path).resolve()),
            "--stage1_epochs", str(args.bart_stage1_epochs),
            "--stage2_epochs", str(args.bart_stage2_epochs),
            "--stage1_learning_rate", str(args.bart_stage1_lr),
            "--stage2_learning_rate", str(args.bart_stage2_lr),
            "--articles_per_topic", str(args.articles_per_topic),
            "--chars_per_article", str(args.chars_per_article),
            "--stage2_bg_replay", str(args.bart_stage2_bg_replay),
        ]
        if args.no_tfidf_filter:
            cmd += ["--no_tfidf_filter"]
        if args.min_new_tokens != -1:
            cmd += ["--min_new_tokens", str(args.min_new_tokens)]
        if args.topics:
            cmd += ["--topics"] + list(args.topics)
        if args.bart_use_mps:
            cmd += ["--use_mps"]
        if args.bart_use_4bit:
            cmd += ["--use_4bit"]
        if args.force_stage2:
            cmd += ["--force_stage2"]
        sh(cmd)
    else:
        print("✓ Bartlett models already present — skipping (use --force_bartlett to re-run).")

    # ---- Encoding-vs-consolidation experiment (panel f) ----
    enc_con_dir = Path(args.enc_con_output_dir).resolve()
    enc_con_done = (enc_con_dir / "statistics.json").exists()
    if args.skip_enc_con:
        print("→ Skipping encoding-vs-consolidation (per flag).")
    elif enc_con_done and not args.force_enc_con:
        print("✓ Encoding-vs-consolidation outputs already present — skipping (use --force_enc_con to re-run).")
    else:
        print("→ Running encoding-vs-consolidation experiment (bartlett_encoding_vs_consolidation.py)…")
        cmd = [
            PY, str(ENC_CON_PY),
            "--output_dir", str(enc_con_dir),
            "--bartlett_path", str(Path(args.bartlett_path).resolve()),
            "--n_samples", str(args.enc_con_n_samples),
            "--consolidation_epochs", str(args.enc_con_epochs),
        ]
        sh(cmd)

    if args.skip_plots:
        print("→ Skipping plots (per flag).")
        sys.exit(0)

    if raykov_outputs_present(raykov_dir):
        print("→ Plotting Raykov figures …")
        cmd_raykov = [
            PY, str(PLOT_PY), "raykov",
            "--raykov_dir", str(raykov_dir),
            "--post_temps", str(args.post_temps),
        ]
        sh(cmd_raykov)
    else:
        print("⚠ Raykov data not found — skipping Raykov plots.")

    if bartlett_models_present(results_dir, args.topics):
        # (1) Checkpoint grid plots (epoch x temp): bar plots, wordclouds, PCA/UMAP/t-SNE
        if not args.skip_bartlett_ckpt_plots:
            print("→ Plotting Bartlett checkpoint-grid figures …")
            cmd_ckpt = [
                PY, str(PLOT_PY), "bartlett_ckpt",
                "--results_dir", str(results_dir),
                "--bartlett_path", str(Path(args.bartlett_path).resolve()),
                "--epochs", *[str(e) for e in args.bart_ckpt_epochs],
                "--temps", *[str(t) for t in args.bart_ckpt_temps],
                "--num_samples", str(args.num_samples),
                "--max_new_tokens", str(args.bart_max_new_tokens),
                "--articles_per_topic", str(args.articles_per_topic),
                "--chars_per_article", str(args.chars_per_article),
            ]
            if args.no_tfidf_filter:
                cmd_ckpt += ["--no_tfidf_filter"]
            if args.min_new_tokens != -1:
                cmd_ckpt += ["--min_new_tokens", str(args.min_new_tokens)]
            if args.offload_dir:
                cmd_ckpt += ["--offload_dir", str(args.offload_dir)]
            if args.bart_use_mps:
                cmd_ckpt += ["--use_mps"]
            if args.topics:
                cmd_ckpt += ["--topics"] + list(args.topics)
            sh(cmd_ckpt)

        # (2) Checkpoint new-word curves:
        # - frac new words vs temp at epochs (default: 5,10)
        # - frac new words vs epoch at temps (default: 0.1,0.5)
        # Use two runs that each produce one of each plot; filenames include epoch/temp, so they won't overwrite.
        if not args.skip_bartlett_newwords:
            epochs = list(args.newwords_temp_epochs)
            temps = list(args.newwords_epoch_temps)
            if len(epochs) < 2 or len(temps) < 2:
                print("⚠ Need at least 2 epochs and 2 temps for default new-words suite; skipping.")
            else:
                pairs = [(epochs[0], temps[0]), (epochs[1], temps[1])]
                for epoch_fixed_for_temp, temp_fixed_for_epoch in pairs:
                    print(f"→ Plotting new-word curves (epoch={epoch_fixed_for_temp}, temp={temp_fixed_for_epoch}) …")
                    cmd_newwords = [
                        PY, str(PLOT_PY), "bartlett",
                        "--skip_final",
                        "--results_dir", str(results_dir),
                        "--bartlett_path", str(Path(args.bartlett_path).resolve()),
                        "--max_new_tokens", str(args.bart_max_new_tokens),
                        "--articles_per_topic", str(args.articles_per_topic),
                        "--chars_per_article", str(args.chars_per_article),
                        "--dtype", str(args.dtype),
                        "--ckpt_temps", *[str(t) for t in args.ckpt_temps],
                        "--ckpt_n_samples", str(args.ckpt_n_samples),
                        "--ckpt_temp_fixed_for_epoch", str(temp_fixed_for_epoch),
                        "--ckpt_epoch_fixed_for_temp", str(epoch_fixed_for_temp),
                    ]
                    if args.no_tfidf_filter:
                        cmd_newwords += ["--no_tfidf_filter"]
                    if args.min_new_tokens != -1:
                        cmd_newwords += ["--min_new_tokens", str(args.min_new_tokens)]
                    if args.offload_dir:
                        cmd_newwords += ["--offload_dir", str(args.offload_dir)]
                    if args.bart_use_mps:
                        cmd_newwords += ["--use_mps"]
                    if args.topics:
                        cmd_newwords += ["--topics"] + list(args.topics)
                    sh(cmd_newwords)
    else:
        print("⚠ Bartlett models not found — skipping Bartlett plots. (Run without --skip_bartlett or with --force_bartlett.)")

    print("\n✓ All done.")
    print(f"Raykov → {raykov_dir / 'plots'}")
    print(f"Bartlett (checkpoint grid) → {results_dir / '_analysis' / 'checkpoints'}")
    print(f"Bartlett (new words) → {results_dir / '_analysis' / 'plots'}")
    print(f"Encoding vs consolidation → {enc_con_dir}")


if __name__ == "__main__":
    main()
