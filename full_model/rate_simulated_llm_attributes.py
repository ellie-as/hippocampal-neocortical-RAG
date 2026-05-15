from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI, OpenAIError, RateLimitError


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "output" / "data"

METRICS = [
    "concrete_vs_abstract",
    "rich_vs_poor_details",
    "specific_vs_general",
]

SYSTEM_PROMPT = """Your task is score text on three metrics: how concrete (vs abstract) it is, how rich in detail it is, and how specific (vs general) it is.

Return ONLY a JSON dictionary with 3 keys, each a float 0-1:

{
  "concrete_vs_abstract": 0-1,
  "rich_vs_poor_details": 0-1,
  "specific_vs_general":  0-1
}

A higher score corresponds to more concrete, richer in detail, or more specific text."""


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected {path} to contain a dict, got {type(obj).__name__}")
    return obj


def _parse_encoding_key(raw: str) -> int | str:
    try:
        return int(raw)
    except ValueError:
        return raw


def _coerce_score_dict(raw: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for metric in METRICS:
        value = raw.get(metric)
        if value is None:
            scores[metric] = None
            continue
        value = float(value)
        scores[metric] = min(1.0, max(0.0, value))
    return scores


def llm_scores(
    client: OpenAI,
    text: str,
    *,
    model: str,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, float | None]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text[:16_000]},
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            return _coerce_score_dict(json.loads(content))
        except (json.JSONDecodeError, ValueError) as err:
            print(f"[Score] Invalid JSON/score response ({err}); retrying...")
            time.sleep(retry_sleep)
        except RateLimitError:
            time.sleep(retry_sleep + attempt)
        except OpenAIError as err:
            print(f"[Score] OpenAI error: {err}")
            time.sleep(retry_sleep + attempt)

    return {metric: None for metric in METRICS}


def _validate_equal_lengths(groups: dict[str, list[str]]) -> int:
    lengths = {name: len(texts) for name, texts in groups.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Model-rating groups have mismatched lengths: {lengths}")
    return next(iter(lengths.values()))


def load_model_rating_texts(data_dir: Path, encoded_key: int | str) -> dict[str, list[str]]:
    enc = _load_pickle(data_dir / "recalled_stories.pkl")
    con = _load_pickle(data_dir / "consolidation_recall.pkl")

    recalled = enc.get("recalled_stories")
    if not isinstance(recalled, dict):
        raise KeyError("recalled_stories.pkl is missing recalled_stories")

    epoch_recalls = con.get("epoch_recalls") or []
    if not epoch_recalls:
        raise KeyError("consolidation_recall.pkl is missing epoch_recalls")

    originals = list(recalled["full"])
    encoded = list(recalled[encoded_key])
    consolidated = list(epoch_recalls[-1])
    imagined = list(recalled["imagined"])
    raw_groups = {
        "original": originals,
        "encoded": encoded,
        "consolidated": consolidated,
        "imagined": imagined,
    }
    _validate_equal_lengths(raw_groups)

    # Match the legacy Figure 4 rating code: generated variants are clipped to
    # the character length of their paired original story before scoring.
    original_lengths = [len(orig) for orig in originals]
    groups = {
        "original": originals,
        "encoded": [txt[:n] for txt, n in zip(encoded, original_lengths)],
        "consolidated": [txt[:n] for txt, n in zip(consolidated, original_lengths)],
        "imagined": [txt[:n] for txt, n in zip(imagined, original_lengths)],
    }
    return groups


def load_forgetting_texts(data_dir: Path) -> list[list[str]]:
    forg = _load_pickle(data_dir / "forgetting_multi.pkl")
    recalls = forg.get("recalls")
    if not isinstance(recalls, list) or not recalls:
        raise KeyError("forgetting_multi.pkl is missing recalls")
    lengths = [len(stage) for stage in recalls]
    if len(set(lengths)) != 1:
        raise ValueError(f"Forgetting recall stages have mismatched lengths: {lengths}")
    return [list(stage) for stage in recalls]


def _read_existing(path: Path, key_cols: list[str], force: bool) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    if force or not path.exists():
        return [], set()

    df = pd.read_csv(path)
    if not set(key_cols).issubset(df.columns):
        return [], set()
    if not set(METRICS).issubset(df.columns):
        return [], set()

    complete = df.dropna(subset=METRICS)
    rows = df.to_dict("records")
    done = {tuple(row[col] for col in key_cols) for row in complete.to_dict("records")}
    print(f"[Resume] Loaded {len(done)} completed rows from {path}")
    return rows, done


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _score_text(
    client: OpenAI,
    text: str,
    *,
    model: str,
    max_retries: int,
    retry_sleep: float,
    dry_run: bool,
) -> dict[str, float | None]:
    if dry_run:
        return {metric: None for metric in METRICS}
    return llm_scores(
        client,
        text,
        model=model,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )


def score_model_ratings(
    *,
    client: OpenAI,
    data_dir: Path,
    output_path: Path,
    encoded_key: int | str,
    model: str,
    max_retries: int,
    retry_sleep: float,
    request_sleep: float,
    force: bool,
    dry_run: bool,
    limit_items: int | None,
) -> None:
    groups = load_model_rating_texts(data_dir, encoded_key)
    versions = ["original", "encoded", "consolidated", "imagined"]
    n_items = _validate_equal_lengths(groups)
    if limit_items is not None:
        n_items = min(n_items, limit_items)

    rows, done = _read_existing(output_path, ["item_id", "version"], force)
    columns = METRICS + ["item_id", "version"]
    total = n_items * len(versions)
    processed = 0

    for item_id in range(n_items):
        for version in versions:
            key = (item_id, version)
            if key in done:
                continue

            processed += 1
            print(f"[Model] {processed}/{total}: item_id={item_id} version={version}")
            scores = _score_text(
                client,
                groups[version][item_id],
                model=model,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                dry_run=dry_run,
            )
            rows.append({**scores, "item_id": item_id, "version": version})
            _write_rows(output_path, rows, columns)
            if request_sleep:
                time.sleep(request_sleep)

    _write_rows(output_path, rows, columns)
    print(f"[Model] Saved {len(rows)} rows to {output_path}")


def score_forgetting_ratings(
    *,
    client: OpenAI,
    data_dir: Path,
    output_path: Path,
    model: str,
    max_retries: int,
    retry_sleep: float,
    request_sleep: float,
    force: bool,
    dry_run: bool,
    limit_items: int | None,
) -> None:
    recalls = load_forgetting_texts(data_dir)
    n_stages = len(recalls)
    n_items = len(recalls[0])
    if limit_items is not None:
        n_items = min(n_items, limit_items)

    rows, done = _read_existing(output_path, ["episode", "item_id"], force)
    columns = METRICS + ["episode", "item_id"]
    total = n_stages * n_items
    processed = 0

    for episode, stage_recalls in enumerate(recalls):
        for item_id in range(n_items):
            key = (episode, item_id)
            if key in done:
                continue

            processed += 1
            print(f"[Forgetting] {processed}/{total}: episode={episode} item_id={item_id}")
            scores = _score_text(
                client,
                stage_recalls[item_id],
                model=model,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                dry_run=dry_run,
            )
            rows.append({**scores, "episode": episode, "item_id": item_id})
            _write_rows(output_path, rows, columns)
            if request_sleep:
                time.sleep(request_sleep)

    _write_rows(output_path, rows, columns)
    print(f"[Forgetting] Saved {len(rows)} rows to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate GPT ratings for simulated model memories used in Figure 4."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_output", type=Path, default=None)
    parser.add_argument("--forgetting_output", type=Path, default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--encoded_key", default="0", help="Key from recalled_stories.pkl to rate as the encoded version.")
    parser.add_argument("--which", choices=["all", "model", "forgetting"], default="all")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSVs instead of resuming them.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and write empty-score rows without API calls.")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help=(
            "Rate only the first 5 stories and write *_dummy.csv outputs by default, "
            "leaving the Figure 4 CSVs untouched."
        ),
    )
    parser.add_argument("--limit_items", type=int, default=None, help="Only process the first N item IDs for testing.")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=3.0)
    parser.add_argument("--request_sleep", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir

    if args.dummy:
        args.limit_items = 5
        model_output = args.model_output or data_dir / "story_llm_ratings_simulated_dummy.csv"
        forgetting_output = args.forgetting_output or data_dir / "forgetting_llm_ratings_dummy.csv"
        print(f"[Dummy] Writing first-5-story outputs to {model_output} and {forgetting_output}")
    else:
        model_output = args.model_output or data_dir / "story_llm_ratings_simulated.csv"
        forgetting_output = args.forgetting_output or data_dir / "forgetting_llm_ratings.csv"

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise SystemExit(f"Set {args.api_key_env} before running, or use --dry_run.")

    client = OpenAI(api_key=api_key) if api_key else OpenAI(api_key="dry-run")
    encoded_key = _parse_encoding_key(args.encoded_key)

    if args.which in {"all", "model"}:
        score_model_ratings(
            client=client,
            data_dir=data_dir,
            output_path=model_output,
            encoded_key=encoded_key,
            model=args.model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            request_sleep=args.request_sleep,
            force=args.force,
            dry_run=args.dry_run,
            limit_items=args.limit_items,
        )

    if args.which in {"all", "forgetting"}:
        score_forgetting_ratings(
            client=client,
            data_dir=data_dir,
            output_path=forgetting_output,
            model=args.model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            request_sleep=args.request_sleep,
            force=args.force,
            dry_run=args.dry_run,
            limit_items=args.limit_items,
        )


if __name__ == "__main__":
    main()
