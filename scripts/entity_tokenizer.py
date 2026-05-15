"""
Utilities to make 2-letter entity strings and relation names tokenize as single tokens.

Motivation: in the relational inference tasks, entities are generated as 2 lowercase
letters (e.g. "ab") and relations like NORTH, CHILD_OF appear in walks. Under GPT-2 BPE
these often split into multiple tokens, which makes variable binding across multiple
memories harder.

This module builds a GPT-2 tokenizer with:
- all "aa".."zz" as atomic tokens (and optionally space-prefixed variants);
- relation names (NORTH, SOUTH, EAST, WEST, CHILD_OF, PARENT_OF, etc.) as single tokens.

If you add relation tokens, the tokenizer vocab size increases; models trained with
an older tokenizer (entities only) will have a different embedding size and need
retraining or embedding resize when switching to this tokenizer.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

# Relation types used in spatial and family graph walks (single tokens each).
DEFAULT_RELATION_TOKENS: list[str] = [
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "CHILD_OF",
    "PARENT_OF",
    "GRANDCHILD_OF",
    "GRANDPARENT_OF",
    "SIBLING_OF",
    "SPOUSE_OF",
]


@dataclass(frozen=True)
class EntityTokenizerSpec:
    base_tokenizer_name: str = "gpt2-medium"
    include_space_prefixed: bool = False
    alphabet: str = string.ascii_lowercase
    # If True, add DEFAULT_RELATION_TOKENS so NORTH, CHILD_OF, etc. are single tokens.
    include_relation_tokens: bool = True
    # Override default relations (e.g. [] to add none, or a custom list).
    relation_tokens: list[str] | None = None


def two_letter_entities(alphabet: str = string.ascii_lowercase) -> list[str]:
    return [a + b for a in alphabet for b in alphabet]


def _relation_tokens_to_add(spec: EntityTokenizerSpec) -> list[str]:
    relations = spec.relation_tokens if spec.relation_tokens is not None else DEFAULT_RELATION_TOKENS
    if not spec.include_relation_tokens or not relations:
        return []
    out = list(relations)
    if spec.include_space_prefixed:
        out.extend([" " + t for t in relations])
    return out


def build_and_save_entity_tokenizer(
    *,
    output_dir: str | Path,
    spec: EntityTokenizerSpec = EntityTokenizerSpec(),
) -> Path:
    """
    Create a tokenizer (based on `spec.base_tokenizer_name`) with 2-letter entities
    and (optionally) relation names added as atomic tokens, then save it to `output_dir`.

    Returns the resolved output directory path.
    """
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Missing dependency `transformers`. Install with `pip install -r requirements.txt`.") from e

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.base_tokenizer_name, use_fast=True)

    entities = two_letter_entities(spec.alphabet)
    tokens_to_add: list[str] = list(entities)
    if spec.include_space_prefixed:
        tokens_to_add.extend([" " + t for t in entities])

    tokens_to_add.extend(_relation_tokens_to_add(spec))

    # `add_tokens` ignores tokens that already exist.
    tokenizer.add_tokens(tokens_to_add, special_tokens=False)
    tokenizer.save_pretrained(str(output_dir))

    return output_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a GPT-2 tokenizer with 2-letter entities and relation names as single tokens."
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write the tokenizer to.")
    parser.add_argument("--base", default="gpt2-medium", help="Base tokenizer to start from.")
    parser.add_argument("--no-space-prefixed", action="store_true", help="Do not add space-prefixed entity tokens.")
    parser.add_argument(
        "--no-relation-tokens",
        action="store_true",
        help="Do not add relation tokens (NORTH, CHILD_OF, etc.); entities only.",
    )
    parser.add_argument(
        "--relation-tokens",
        type=str,
        default=None,
        help="Comma-separated relation tokens to add (default: NORTH,SOUTH,EAST,WEST,CHILD_OF,...). Overrides --no-relation-tokens if set.",
    )
    args = parser.parse_args(argv)

    relation_tokens: list[str] | None = None
    if args.relation_tokens is not None:
        relation_tokens = [t.strip() for t in args.relation_tokens.split(",") if t.strip()]
    include_relation_tokens = not args.no_relation_tokens if relation_tokens is None else True

    spec = EntityTokenizerSpec(
        base_tokenizer_name=args.base,
        include_space_prefixed=not args.no_space_prefixed,
        include_relation_tokens=include_relation_tokens,
        relation_tokens=relation_tokens,
    )
    out = build_and_save_entity_tokenizer(output_dir=args.output_dir, spec=spec)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

