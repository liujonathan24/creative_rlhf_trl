# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GRPO Loss Variant 1: Bottom 20% Percentile Filtering k3_loss_fn.

This variant filters for the bottom 20% of log_p values using a quick-select
approach, then calculates the standard k3 loss only on those filtered values.
The intuition is to focus the KL penalty on tokens where the reference model
has lower confidence (lower log probabilities).
"""

from typing import Optional

import torch

from .grpo_loss_base import LigerFusedLinearGRPOLoss, clip_coef_fn
from liger_kernel_losses.fused_linear_ppo import LigerFusedLinearPPOBase


def k3_loss_fn_variant1(log_p, log_q):
    """
    Variant 1: Bottom 20% percentile filtering k3 estimate of KL[q, p].

    Uses quick-select (via torch.kthvalue) to find the 20th percentile threshold
    in log_p, then applies the standard k3 loss calculation only to the bottom 20%
    of values. For the remaining 80%, returns zero KL penalty.

    This focuses the KL regularization on tokens where the reference model has
    lower confidence, potentially allowing more exploration elsewhere.

    Args:
        log_p: Log probabilities from reference model (shape: [batch_size, seq_len])
        log_q: Log probabilities from current policy (shape: [batch_size, seq_len])

    Returns:
        KL divergence estimate with same shape as input, but only non-zero for
        bottom 20% of log_p values.
    """
    # Flatten to 1D for percentile calculation
    log_p_flat = log_p.reshape(-1)

    # Calculate the index for 20th percentile (k-th smallest value)
    # For 20th percentile, we want the value at position: n * 0.20
    n = log_p_flat.numel()
    k = max(1, int(n * 0.20))  # Ensure k is at least 1

    # Use torch.kthvalue for O(n) average-case quick-select
    # kthvalue returns (values, indices) where values is the k-th smallest
    threshold_value, _ = torch.kthvalue(log_p_flat, k)

    # Create mask for bottom 20% (values <= threshold)
    mask = (log_p <= threshold_value).float()

    # Calculate standard k3 loss
    log_ratio = log_p - log_q
    k3_loss = torch.exp(log_ratio) - log_ratio - 1.0

    # Apply mask: only keep loss for bottom 20% of log_p values
    # Scale by 5.0 to maintain similar magnitude (since we're using 20% of values)
    return mask * k3_loss * 5.0


class LigerFusedLinearGRPOFunctionVariant1(LigerFusedLinearPPOBase):
    """GRPO Function with Variant 1 of k3_loss_fn."""

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
        """GRPO Loss Function with Variant 1 k3_loss_fn."""
        per_token_logps = log_probs.gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(-1)

        # Get reference model probabilities
        if ref_per_token_logps is None:
            if ref_log_probs is not None:
                with torch.no_grad():
                    ref_per_token_logps = ref_log_probs.gather(
                        dim=-1, index=selected_token_ids.unsqueeze(-1)
                    ).squeeze(-1)
            else:
                ref_per_token_logps = per_token_logps.detach()

        # Compute policy gradient loss with importance sampling ratio
        old_per_token_logps = old_per_token_logps if old_per_token_logps is not None else per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps

        if importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * attention_mask).sum(-1) / attention_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = clip_coef_fn(coef_1, epsilon_low, epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if beta != 0.0:
            # Use the variant k3_loss_fn
            kl_div = k3_loss_fn_variant1(ref_per_token_logps, per_token_logps)
            per_token_loss = per_token_loss + beta * kl_div

        # Loss normalization (same as base implementation)
        if loss_type == "grpo":
            loss = (
                (per_token_loss * attention_mask).sum(-1) / torch.clamp(attention_mask.sum(-1), min=1.0)
            ).sum() / full_attention_mask.shape[0]
        elif loss_type == "bnpo":
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)
        elif loss_type == "dr_grpo":
            if max_completion_length is None:
                raise ValueError("max_completion_length must be provided for loss_type 'dr_grpo'")
            loss = (per_token_loss * attention_mask).sum() / (full_attention_mask.shape[0] * max_completion_length)
        elif loss_type == "dapo":
            loss_normalizer = LigerFusedLinearPPOBase._compute_dapo_normalizer(full_attention_mask)
            loss = (per_token_loss * attention_mask).sum() / loss_normalizer
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Calculate metrics
        metrics = []
        if beta != 0.0:
            metrics.append(((kl_div * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)))

        # Clipping metrics
        if importance_sampling_level == "token":
            is_clipped = ((coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
                (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
            )
        else:
            is_clipped = ((coef_1.squeeze(-1) < 1 - epsilon_low) & (advantages < 0)) | (
                (coef_1.squeeze(-1) > 1 + epsilon_high) & (advantages > 0)
            )
            is_clipped = is_clipped.unsqueeze(1).expand_as(attention_mask)

        metrics.append((is_clipped * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0))
        return loss, metrics

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.04,
        epsilon_low=0.2,
        epsilon_high=0.2,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        temperature=1.0,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
    ):
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            selected_token_ids=selected_token_ids,
            attention_mask=attention_mask,
            advantages=advantages,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            temperature=temperature,
            compiled=compiled,
            use_ref_model=use_ref_model,
            chunk_size=chunk_size,
            importance_sampling_level=importance_sampling_level,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        grads = LigerFusedLinearPPOBase.backward(ctx, grad_output)
        return (
            *grads[:6],
            None,  # grad_ref_per_token_logps
            None,  # grad_old_per_token_logps
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_epsilon_low
            None,  # grad_epsilon_high
            None,  # grad_loss_type
            None,  # grad_max_completion_length
            None,  # grad_importance_sampling_level
            None,  # grad_temperature
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_chunk_size
        )


class LigerFusedLinearGRPOLossVariant1(torch.nn.Module):
    """
    GRPO Loss Variant 1: Bottom 20% Percentile Filtering.

    This variant applies the KL penalty only to tokens where the reference model
    has the lowest 20% of log probabilities. It uses an O(n) quick-select algorithm
    (torch.kthvalue) to find the 20th percentile threshold, then masks the k3 loss
    to only apply to those low-confidence tokens.

    The intuition is that focusing KL regularization on tokens where the reference
    model is least confident may allow more exploration on high-confidence tokens
    while still maintaining alignment on uncertain predictions.

    Usage:
        from trl.experimental.liger_kernel_losses.grpo_loss_variant1 import LigerFusedLinearGRPOLossVariant1

        trainer = GRPOTrainer(
            model="Qwen/Qwen2-0.5B-Instruct",
            reward_funcs=accuracy_reward,
            args=GRPOConfig(use_liger_kernel=True),
            train_dataset=dataset,
            loss_class=LigerFusedLinearGRPOLossVariant1,
        )
    """

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

    def forward(
        self,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return LigerFusedLinearGRPOFunctionVariant1.apply(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias,
            ref_per_token_logps,
            old_per_token_logps,
            ref_input,
            ref_weight,
            ref_bias,
            self.beta,
            self.epsilon_low,
            self.epsilon_high,
            self.loss_type,
            self.max_completion_length,
            self.importance_sampling_level,
            self.temperature,
            self.compiled,
            self.use_ref_model,
            self.chunk_size,
        )
