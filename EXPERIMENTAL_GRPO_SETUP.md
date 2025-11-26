# Experimental GRPO Loss Setup - Implementation Summary

This document describes the implementation of experimental GRPO loss variants for the TRL library.

## Overview

The implementation allows you to experiment with different variants of the `k3_loss_fn` used in GRPO training without forking both TRL and Liger-Kernel. All experimental code is contained within the TRL repository in the `trl/experimental/liger_kernel_losses/` directory.

## Changes Made

### 1. Directory Structure Created

```
trl/experimental/liger_kernel_losses/
├── __init__.py                    # Exports base loss classes
├── README.md                      # Detailed documentation
├── example_usage.py               # Example usage script
├── fused_linear_ppo.py           # Base PPO functionality (already existed)
├── grpo_loss_base.py             # Base GRPO loss (renamed from grpo_loss.py)
├── grpo_loss_variant1.py         # Example variant 1 with damped exponential
└── grpo_loss_variant2.py         # Example variant 2 with squared approximation
```

### 2. Files Modified

#### `trl/trainer/grpo_trainer.py`
- **Line 94**: Updated import to use `trl.experimental.liger_kernel_losses.grpo_loss_base`
- **Line 251**: Added `loss_class` parameter to `__init__` method
- **Line 220-225**: Added documentation for `loss_class` parameter
- **Line 522-534**: Modified loss instantiation to support custom loss classes

#### `trl/experimental/__init__.py`
- **Line 37-70**: Added exports for experimental loss classes

### 3. Files Created

#### `trl/experimental/liger_kernel_losses/__init__.py`
Exports the base loss classes and functions.

#### `trl/experimental/liger_kernel_losses/grpo_loss_base.py`
Renamed from `grpo_loss.py`. Contains the base GRPO loss implementation.

#### `trl/experimental/liger_kernel_losses/grpo_loss_variant1.py`
Bottom 20% percentile filtering variant. Uses O(n) quick-select to apply KL penalty only to tokens where the reference model has lowest confidence:
```python
def k3_loss_fn_variant1(log_p, log_q):
    # Find 20th percentile threshold using quick-select
    log_p_flat = log_p.reshape(-1)
    n = log_p_flat.numel()
    k = max(1, int(n * 0.20))
    threshold_value, _ = torch.kthvalue(log_p_flat, k)

    # Mask for bottom 20% and apply k3 loss
    mask = (log_p <= threshold_value).float()
    log_ratio = log_p - log_q
    k3_loss = torch.exp(log_ratio) - log_ratio - 1.0
    return mask * k3_loss * 5.0
```

**Intuition:** Focus KL regularization on low-confidence tokens, allowing more exploration on high-confidence tokens.

#### `trl/experimental/liger_kernel_losses/grpo_loss_variant2.py`
Placeholder example variant with squared approximation k3_loss_fn (replace with your own):
```python
def k3_loss_fn_variant2(log_p, log_q):
    log_ratio = log_p - log_q
    return 0.5 * log_ratio ** 2
```

#### `trl/experimental/liger_kernel_losses/README.md`
Comprehensive documentation on usage and creating new variants.

#### `trl/experimental/liger_kernel_losses/example_usage.py`
Example script showing how to use each variant.

## Usage

### Using the Base Loss (Default Behavior)

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=GRPOConfig(use_liger_kernel=True),
    train_dataset=dataset,
)
```

### Using Experimental Variant 1

```python
from trl import GRPOTrainer, GRPOConfig
from trl.experimental import LigerFusedLinearGRPOLossVariant1

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=GRPOConfig(use_liger_kernel=True),
    train_dataset=dataset,
    loss_class=LigerFusedLinearGRPOLossVariant1,
)
```

### Using Experimental Variant 2

```python
from trl import GRPOTrainer, GRPOConfig
from trl.experimental import LigerFusedLinearGRPOLossVariant2

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=GRPOConfig(use_liger_kernel=True),
    train_dataset=dataset,
    loss_class=LigerFusedLinearGRPOLossVariant2,
)
```

## Creating Your Own Variants

To create your own experimental k3_loss_fn variant:

1. **Create a new variant file** (e.g., `grpo_loss_variant3.py`) following the pattern in variant1 or variant2
2. **Implement your custom k3_loss_fn**:
   ```python
   def k3_loss_fn_variant3(log_p, log_q):
       # Your custom KL divergence approximation
       return your_custom_calculation(log_p, log_q)
   ```
3. **Create the Function and Loss classes** that use your k3_loss_fn
4. **Export your variant** in `trl/experimental/__init__.py`
5. **Use your variant** by passing it to `GRPOTrainer(loss_class=YourVariant)`

See [trl/experimental/liger_kernel_losses/README.md](trl/experimental/liger_kernel_losses/README.md) for detailed instructions.

## Benefits of This Approach

1. **Single Repository**: Only need to maintain a TRL fork, not both TRL and Liger-Kernel
2. **Easy Experimentation**: Switch between variants with a single parameter
3. **Minimal Code Duplication**: Inherit from base class, only override k3_loss_fn
4. **Side-by-Side Comparison**: All variants organized in one directory
5. **Clean Implementation**: Each variant file shows only what changed
6. **Easy Installation**: Users just install your TRL fork

## Implementation Details

### Key Modification Point

The main modification is the `k3_loss_fn` function used in the GRPO loss calculation. The original implementation from Liger-Kernel is:

```python
def k3_loss_fn(log_p, log_q):
    # Computes k3 estimate of KL[q, p]
    # ref: http://joschu.net/blog/kl-approx.html
    return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0
```

This function is called in the loss calculation when `beta != 0.0`:

```python
if beta != 0.0:
    kl_div = k3_loss_fn(ref_per_token_logps, per_token_logps)
    per_token_loss = per_token_loss + beta * kl_div
```

### Inheritance Pattern

Each variant:
1. Defines a custom `k3_loss_fn_variantN` function
2. Creates a `LigerFusedLinearGRPOFunctionVariantN` class inheriting from `LigerFusedLinearPPOBase`
3. Overrides `ppo_loss_fn` to use the custom k3_loss_fn
4. Creates a `LigerFusedLinearGRPOLossVariantN` module class that applies the custom function

## Source Attribution

- Base implementation vendored from [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- All vendored code retains the original Apache 2.0 license
- Modifications and variants are part of the TRL library

## Testing

To verify the implementation works:

1. **Check imports**:
   ```python
   from trl.experimental import (
       LigerFusedLinearGRPOLoss,
       LigerFusedLinearGRPOLossVariant1,
       LigerFusedLinearGRPOLossVariant2,
   )
   ```

2. **Run example script**:
   ```bash
   python trl/experimental/liger_kernel_losses/example_usage.py
   ```

3. **Train with different variants** and compare results

## Next Steps

1. **Replace the example k3_loss_fn implementations** in variant1.py and variant2.py with your actual experimental variants
2. **Create additional variant files** as needed for different experiments
3. **Run experiments** to compare performance across variants
4. **Iterate** based on results

## Questions or Issues?

See the detailed README in [trl/experimental/liger_kernel_losses/README.md](trl/experimental/liger_kernel_losses/README.md) for more information.
