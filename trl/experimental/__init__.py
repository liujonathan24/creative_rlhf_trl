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
Experimental submodule for TRL.

This submodule contains unstable or incubating features. Anything here may change (or be removed) in any release
without deprecation. Use at your own risk.

To silence this notice set environment variable TRL_EXPERIMENTAL_SILENCE=1.
"""

import os
import warnings


if not os.environ.get("TRL_EXPERIMENTAL_SILENCE"):
    warnings.warn(
        "You are importing from 'trl.experimental'. APIs here are unstable and may change or be removed without "
        "notice. Silence this warning by setting environment variable TRL_EXPERIMENTAL_SILENCE=1.",
        UserWarning,
        stacklevel=2,
    )


# Experimental Liger Kernel Losses
try:
    from .liger_kernel_losses import (
        LigerFusedLinearGRPOFunction,
        LigerFusedLinearGRPOLoss,
        clip_coef_fn,
        k3_loss_fn,
    )
    from .liger_kernel_losses.grpo_loss_variant1 import (
        LigerFusedLinearGRPOFunctionVariant1,
        LigerFusedLinearGRPOLossVariant1,
        k3_loss_fn_variant1,
    )
    from .liger_kernel_losses.grpo_loss_variant2 import (
        LigerFusedLinearGRPOFunctionVariant2,
        LigerFusedLinearGRPOLossVariant2,
        k3_loss_fn_variant2,
    )

    __all__ = [
        "LigerFusedLinearGRPOFunction",
        "LigerFusedLinearGRPOLoss",
        "k3_loss_fn",
        "clip_coef_fn",
        "LigerFusedLinearGRPOFunctionVariant1",
        "LigerFusedLinearGRPOLossVariant1",
        "k3_loss_fn_variant1",
        "LigerFusedLinearGRPOFunctionVariant2",
        "LigerFusedLinearGRPOLossVariant2",
        "k3_loss_fn_variant2",
    ]
except ImportError:
    # Liger kernel losses are optional
    __all__ = []
