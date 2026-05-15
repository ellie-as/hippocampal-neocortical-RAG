#!/usr/bin/env python3
"""
Probe Stage 1 models to check whether topic knowledge was absorbed.

Loads each topic's Stage 1 LoRA checkpoint from output_twostage/{topic}/
stage1_background/model/ and generates text from a set of topic-relevant
prompts.  Useful for diagnosing weak topic effects before running Stage 2.

Usage:
    python probe_stage1.py
    python probe_stage1.py --output_dir /path/to/output_twostage
    python probe_stage1.py --topics Sport Nature
    python probe_stage1.py --max_new_tokens 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


TOPIC_PROMPTS = {
    "Nature": [
        "The forest ecosystem is",
        "Rivers and lakes are important because",
        "Many species of wildlife",
    ],
    "Politics": [
        "The government decided to",
        "In a democratic society, elections",
        "The political debate focused on",
    ],
    "Universe": [
        "The stars in the night sky",
        "Astronomers recently discovered",
        "The origin of the universe",
    ],
    "Sport": [
        "The championship game was",
        "Athletes train hard because",
        "The team scored a goal",
    ],
    "Health": [
        "The patient was diagnosed with",
        "Medical research has shown that",
        "A healthy lifestyle includes",
    ],
}

GENERIC_PROMPTS = [
    "One night two young men went down to the river",
    "Tell me about",
    "Once upon a time"
]


def load_stage1_model(model_path: Path, use_4bit: bool = True):
    cfg_path = model_path / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No adapter_config.json at {model_path}")

    base_name = json.loads(cfg_path.read_text())["base_model_name_or_path"]

    if use_4bit and torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_name, quantization_config=quant_cfg, device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="auto",
        )

    tok = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = PeftModel.from_pretrained(base, str(model_path), device_map="auto")
    model.eval()
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 150) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Probe Stage 1 topic models")
    parser.add_argument("--output_dir", type=str, default="output_twostage")
    parser.add_argument("--topics", nargs="*",
                        default=["Nature", "Politics", "Universe", "Sport", "Health"])
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--include_generic", action="store_true",
                        help="Also test generic prompts (e.g. Bartlett opening)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for topic in args.topics:
        model_path = output_dir / topic / "stage1_background" / "model"
        if not model_path.exists():
            print(f"\n[{topic}] No Stage 1 model at {model_path} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  TOPIC: {topic}")
        print(f"  Model: {model_path}")
        print(f"{'='*60}")

        model, tok = load_stage1_model(model_path, use_4bit=not args.no_4bit)

        prompts = TOPIC_PROMPTS.get(topic, GENERIC_PROMPTS[:1])
        if args.include_generic:
            prompts = GENERIC_PROMPTS

        for prompt in prompts:
            text = generate(model, tok, prompt, args.max_new_tokens)
            print(f"\n  Prompt: {prompt!r}")
            print(f"  Output: {text[:300]}")

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
