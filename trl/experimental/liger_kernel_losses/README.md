# Experimental Liger Kernel Losses for GRPO

This directory contains vendored and experimental versions of the Liger-Kernel GRPO loss implementations. The code allows you to experiment with different variants of the k3_loss_fn used in GRPO training.

## Directory Structure

```
trl/experimental/liger_kernel_losses/
├── __init__.py                    # Exports base loss classes
├── README.md                      # This file
├── fused_linear_ppo.py           # Base PPO functionality from Liger-Kernel
├── grpo_loss_base.py             # Base GRPO loss (vendored from Liger-Kernel)
├── grpo_loss_variant1.py         # Example variant 1 with modified k3_loss_fn
└── grpo_loss_variant2.py         # Example variant 2 with different k3_loss_fn
```

## Usage

### Using the Base Loss (Default)

By default, GRPOTrainer uses the base implementation from `grpo_loss_base.py`:

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=your_reward_func,
    args=GRPOConfig(use_liger_kernel=True),
    train_dataset=dataset,
)
```

### Using Experimental Variants

To use an experimental variant with a modified k3_loss_fn:

```python
from trl import GRPOTrainer, GRPOConfig
from trl.experimental import LigerFusedLinearGRPOLossVariant1

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=your_reward_func,
    args=GRPOConfig(use_liger_kernel=True),
    train_dataset=dataset,
    loss_class=LigerFusedLinearGRPOLossVariant1,  # Use your custom variant
)
```

Or variant 2:

```python
from trl.experimental import LigerFusedLinearGRPOLossVariant2

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=your_reward_func,
    args=GRPOConfig(use_liger_kernel=True),
    train_dataset=dataset,
    loss_class=LigerFusedLinearGRPOLossVariant2,
)
```

## Creating Your Own Variants

To create your own experimental k3_loss_fn variant:

### Step 1: Create a new variant file

Create a new file like `grpo_loss_variant3.py`:

```python
from typing import Optional
import torch
from .grpo_loss_base import clip_coef_fn
from liger_kernel_losses.fused_linear_ppo import LigerFusedLinearPPOBase


def k3_loss_fn_variant3(log_p, log_q):
    """
    Your custom k3_loss_fn implementation.

    Args:
        log_p: Log probabilities from reference model
        log_q: Log probabilities from current policy

    Returns:
        KL divergence estimate
    """
    # Your custom implementation here
    log_ratio = log_p - log_q
    # Example: Your own KL approximation
    return your_custom_kl_calculation(log_ratio)


class LigerFusedLinearGRPOFunctionVariant3(LigerFusedLinearPPOBase):
    @staticmethod
    def ppo_loss_fn(
        log_probs,
        selected_token_ids,
        attention_mask,
        advantages,
        full_attention_mask,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_log_probs=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        **kwargs,
    ):
        # Copy the implementation from variant1.py or variant2.py
        # But replace k3_loss_fn with k3_loss_fn_variant3
        per_token_logps = log_probs.gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(-1)

        # ... rest of the implementation ...

        if beta != 0.0:
            # Use YOUR variant k3_loss_fn
            kl_div = k3_loss_fn_variant3(ref_per_token_logps, per_token_logps)
            per_token_loss = per_token_loss + beta * kl_div

        # ... rest of the implementation ...
        return loss, metrics

    # Include forward and backward methods (copy from variant1.py)


class LigerFusedLinearGRPOLossVariant3(torch.nn.Module):
    """Your variant 3 description."""

    def __init__(
        self,
        beta: float = 0.04,
        compiled: bool = True,
        use_ref_model: bool = True,
        chunk_size: int = 1,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        loss_type: str = "dapo",
        max_completion_length: Optional[int] = None,
        importance_sampling_level: str = "token",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.importance_sampling_level = importance_sampling_level
        self.temperature = temperature

    def forward(self, _input, lin_weight, selected_token_ids, attention_mask,
                advantages, bias=None, ref_per_token_logps=None,
                old_per_token_logps=None, ref_input=None, ref_weight=None,
                ref_bias=None):
        return LigerFusedLinearGRPOFunctionVariant3.apply(
            _input, lin_weight, selected_token_ids, attention_mask, advantages,
            bias, ref_per_token_logps, old_per_token_logps, ref_input,
            ref_weight, ref_bias, self.beta, self.epsilon_low, self.epsilon_high,
            self.loss_type, self.max_completion_length,
            self.importance_sampling_level, self.temperature, self.compiled,
            self.use_ref_model, self.chunk_size,
        )
```

### Step 2: Update exports

Add your variant to `trl/experimental/__init__.py`:

```python
from .liger_kernel_losses.grpo_loss_variant3 import (
    LigerFusedLinearGRPOFunctionVariant3,
    LigerFusedLinearGRPOLossVariant3,
    k3_loss_fn_variant3,
)
```

### Step 3: Use your variant

```python
from trl.experimental import LigerFusedLinearGRPOLossVariant3

trainer = GRPOTrainer(
    ...,
    loss_class=LigerFusedLinearGRPOLossVariant3,
)
```

## Key Modifications

The main modification point is the `k3_loss_fn` function. The original implementation from Liger-Kernel is:

```python
def k3_loss_fn(log_p, log_q):
    # Computes k3 estimate of KL[q, p]
    # ref: http://joschu.net/blog/kl-approx.html
    return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0
```

You can replace this with any other KL divergence approximation or regularization term you want to experiment with.

## Example Variants Included

### Variant 1: Bottom 20% Percentile Filtering
This variant uses an O(n) quick-select algorithm to find the 20th percentile threshold in log_p, then applies the KL penalty only to the bottom 20% of tokens (those with lowest log probabilities from the reference model).

**Implementation:**
```python
def k3_loss_fn_variant1(log_p, log_q):
    # Flatten to 1D for percentile calculation
    log_p_flat = log_p.reshape(-1)

    # Calculate 20th percentile using O(n) quick-select
    n = log_p_flat.numel()
    k = max(1, int(n * 0.20))
    threshold_value, _ = torch.kthvalue(log_p_flat, k)

    # Create mask for bottom 20%
    mask = (log_p <= threshold_value).float()

    # Calculate standard k3 loss only on masked values
    log_ratio = log_p - log_q
    k3_loss = torch.exp(log_ratio) - log_ratio - 1.0

    # Apply mask and scale to maintain magnitude
    return mask * k3_loss * 5.0
```

**Intuition:** Focus KL regularization on tokens where the reference model is least confident (lowest log probs), allowing more exploration on high-confidence tokens.

### Variant 2: Squared Approximation
This is a placeholder example variant using a simple squared approximation. Replace with your own experimental variant.

```python
def k3_loss_fn_variant2(log_p, log_q):
    log_ratio = log_p - log_q
    return 0.5 * log_ratio ** 2
```

## Benefits of This Approach

1. **Easy Experimentation**: Switch between different k3_loss_fn implementations by just passing a different class
2. **Minimal Code Duplication**: Only override the k3_loss_fn, inherit everything else
3. **Side-by-side Comparison**: Keep all variants in one place for easy comparison
4. **Clean Implementation**: Each variant file only shows what changed from the base
5. **Type Safety**: All variants have the same interface as the base class

## Source Attribution

The base implementation in `grpo_loss_base.py` and `fused_linear_ppo.py` are vendored from:
- [Liger-Kernel GRPO Loss](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/grpo_loss.py)

All vendored code retains the original Apache 2.0 license.
