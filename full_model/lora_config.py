"""
Global LoRA Configuration

This file defines the LoRA settings used across all memory simulation and 
consolidation scripts. Change settings here to affect all experiments.

Three presets are provided:
- CURRENT: Original aggressive settings (high capacity, higher forgetting risk)
- OPTION_A: Conservative settings (minimal forgetting, may reduce memorization)
- OPTION_B: Moderate balance between memorization and forgetting

For parameter sweeps, set LORA_SWEEP_R (and optionally LORA_SWEEP_ALPHA,
LORA_SWEEP_DROPOUT, LORA_SWEEP_MODULES) as environment variables to
override the ACTIVE config without editing this file.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """LoRA hyperparameters for fine-tuning."""
    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.target_modules:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# DEFAULT: Original settings - high rank but low effective scaling (alpha/r = 0.25)
# Trains all modules but LoRA updates are heavily attenuated
DEFAULT = LoRAConfig(
    r=64,
    alpha=16,
    dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # FFN (stores factual knowledge)
    ],
)

# # CURRENT: Pervasive settings - high rank with standard scaling (alpha/r = 1.0)
# # Trains all modules with full LoRA effective learning rate
# # Used for Bartlett base models.
# CURRENT = LoRAConfig(
#     r=32,
#     alpha=64,
#     dropout=0.05,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
#         "gate_proj", "up_proj", "down_proj",      # FFN (stores factual knowledge)
#     ],
# )

# # OPTION A: Conservative - minimal forgetting
# # Lower rank + attention-only = preserves world knowledge in FFN layers
# # May reduce episodic memorization capacity
# OPTION_A = LoRAConfig(
#     r=8,
#     alpha=16,         # alpha/r = 2.0 (stronger per-parameter updates)
#     dropout=0.1,      # More regularization
#     target_modules=[
#         "q_proj", "v_proj",  # Only Q and V attention (minimal intervention)
#     ],
# )

# # OPTION B: Moderate balance
# # Medium rank, attention-only, moderate regularization
# # Balances memorization with knowledge preservation
# OPTION_B = LoRAConfig(
#     r=16,
#     alpha=32,         # alpha/r = 2.0
#     dropout=0.1,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",  # Full attention, no FFN
#     ],
# )


# =============================================================================
# ACTIVE CONFIGURATION
# =============================================================================

# Change this to switch between presets:
#   ACTIVE = CURRENT    # Original aggressive settings
#   ACTIVE = OPTION_A   # Conservative (uncomment OPTION_A above first)
#   ACTIVE = OPTION_B   # Moderate (uncomment OPTION_B above first)

ACTIVE = DEFAULT

# =============================================================================
# ENVIRONMENT VARIABLE OVERRIDES (for parameter sweeps)
# =============================================================================
# Setting LORA_SWEEP_R triggers override mode. Other LORA_SWEEP_* vars are
# optional and fall back to the DEFAULT preset values.

_sweep_r = os.environ.get("LORA_SWEEP_R")
if _sweep_r is not None:
    _sweep_modules_raw = os.environ.get("LORA_SWEEP_MODULES")
    ACTIVE = LoRAConfig(
        r=int(_sweep_r),
        alpha=int(os.environ.get("LORA_SWEEP_ALPHA", DEFAULT.alpha)),
        dropout=float(os.environ.get("LORA_SWEEP_DROPOUT", DEFAULT.dropout)),
        target_modules=(
            _sweep_modules_raw.split(",")
            if _sweep_modules_raw
            else DEFAULT.target_modules.copy()
        ),
    )
    print(
        f"[LoRA] ENV override: r={ACTIVE.r}, alpha={ACTIVE.alpha}, "
        f"dropout={ACTIVE.dropout}, modules={ACTIVE.target_modules}"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_lora_config():
    """Get the active LoRA configuration."""
    return ACTIVE


def get_peft_lora_config():
    """
    Get a peft LoraConfig object from the active configuration.
    
    Returns:
        peft.LoraConfig: Ready to use with get_peft_model()
    """
    from peft import LoraConfig as PeftLoraConfig
    
    cfg = get_lora_config()
    return PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
