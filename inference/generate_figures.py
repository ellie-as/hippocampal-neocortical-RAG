"""Generate manuscript inference Figures 6 and 7 from trained models.

This wrapper preserves the separate reproducible data-generation steps while
providing one command for normal use.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80, flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate inference Figure 6, Figure 7, caches, and source-data tables.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(INFERENCE_DIR / "inference_config.json"),
        help="Path to inference_config.json with trained model and output paths.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Recompute generated caches instead of reusing existing cache files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smaller collate settings for a quick end-to-end test.",
    )
    parser.add_argument(
        "--n-per-template",
        type=int,
        default=None,
        help="RAG-composition trials per template. Defaults to 100, or 10 with --smoke.",
    )
    parser.add_argument(
        "--skip-panel-export",
        action="store_true",
        help="Do not export Figure 6g/h/i/k CSV tables from generated caches.",
    )
    args = parser.parse_args(argv)

    n_per_template = args.n_per_template if args.n_per_template is not None else (10 if args.smoke else 100)
    py = sys.executable
    config_args = ["--config", args.config]
    clear_args = ["--clear-cache"] if args.clear_cache else []
    smoke_args = ["--smoke"] if args.smoke else []

    # First pass: generate Figure 7 and all non-RAG Figure 6 caches.
    _run(
        [
            py,
            str(INFERENCE_DIR / "collate_inf_figures.py"),
            *config_args,
            *clear_args,
            *smoke_args,
        ]
    )

    # Generate the RAG-composition cache used by Figure 6k.
    _run(
        [
            py,
            str(INFERENCE_DIR / "rag_composition.py"),
            *config_args,
            *clear_args,
            "--n-per-template",
            str(n_per_template),
        ]
    )

    # Second pass: rebuild manuscript Figure 6 now that panel k data exists.
    _run(
        [
            py,
            str(INFERENCE_DIR / "collate_inf_figures.py"),
            *config_args,
            "--skip-fig6",
            *smoke_args,
        ]
    )

    if not args.skip_panel_export:
        _run(
            [
                py,
                str(INFERENCE_DIR / "export_figure6_panels.py"),
                *config_args,
            ]
        )

    _run(
        [
            py,
            str(INFERENCE_DIR / "build_figure6_full.py"),
            *config_args,
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
