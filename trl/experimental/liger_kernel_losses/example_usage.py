#!/usr/bin/env python3
"""
Example script demonstrating how to use experimental GRPO loss variants.

This script shows three different ways to use the GRPO trainer:
1. Default base loss
2. Variant 1 with damped exponential k3_loss_fn
3. Variant 2 with squared approximation k3_loss_fn
"""

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from trl.experimental import (
    LigerFusedLinearGRPOLossVariant1,
    LigerFusedLinearGRPOLossVariant2,
)
from trl.rewards import accuracy_reward


def train_with_base_loss():
    """Train using the default base GRPO loss."""
    print("=" * 80)
    print("Training with BASE GRPO Loss")
    print("=" * 80)

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train[:100]")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        args=GRPOConfig(
            use_liger_kernel=True,
            output_dir="./grpo_base",
            max_steps=10,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    return trainer


def train_with_variant1():
    """Train using Variant 1 (damped exponential k3_loss_fn)."""
    print("=" * 80)
    print("Training with VARIANT 1 GRPO Loss (Damped Exponential)")
    print("=" * 80)

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train[:100]")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        args=GRPOConfig(
            use_liger_kernel=True,
            output_dir="./grpo_variant1",
            max_steps=10,
        ),
        train_dataset=dataset,
        loss_class=LigerFusedLinearGRPOLossVariant1,  # Use Variant 1
    )
    trainer.train()
    return trainer


def train_with_variant2():
    """Train using Variant 2 (squared approximation k3_loss_fn)."""
    print("=" * 80)
    print("Training with VARIANT 2 GRPO Loss (Squared Approximation)")
    print("=" * 80)

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train[:100]")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        args=GRPOConfig(
            use_liger_kernel=True,
            output_dir="./grpo_variant2",
            max_steps=10,
        ),
        train_dataset=dataset,
        loss_class=LigerFusedLinearGRPOLossVariant2,  # Use Variant 2
    )
    trainer.train()
    return trainer


if __name__ == "__main__":
    # Example 1: Train with base loss
    # base_trainer = train_with_base_loss()

    # Example 2: Train with Variant 1
    # variant1_trainer = train_with_variant1()

    # Example 3: Train with Variant 2
    # variant2_trainer = train_with_variant2()

    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    print("""
To use these experimental loss variants in your own training:

1. Import the variant you want to use:
   from trl.experimental import LigerFusedLinearGRPOLossVariant1

2. Pass it to GRPOTrainer via the loss_class parameter:
   trainer = GRPOTrainer(
       model="your-model",
       reward_funcs=your_reward_func,
       args=GRPOConfig(use_liger_kernel=True),
       train_dataset=dataset,
       loss_class=LigerFusedLinearGRPOLossVariant1,
   )

3. Compare results across different variants to find the best one!

Available variants:
- LigerFusedLinearGRPOLoss (base - default)
- LigerFusedLinearGRPOLossVariant1 (damped exponential)
- LigerFusedLinearGRPOLossVariant2 (squared approximation)
    """)
