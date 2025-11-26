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
Experimental Liger Kernel Losses - Vendored from Liger-Kernel with modifications.

This module contains vendored components from the Liger-Kernel library for GRPO loss calculations.
The base implementations are kept in grpo_loss_base.py, and experimental variants can be created
by inheriting from the base classes and overriding specific methods (e.g., k3_loss_fn).
"""

from .grpo_loss_base import (
    LigerFusedLinearGRPOFunction,
    LigerFusedLinearGRPOLoss,
    clip_coef_fn,
    k3_loss_fn,
)

__all__ = [
    "LigerFusedLinearGRPOFunction",
    "LigerFusedLinearGRPOLoss",
    "k3_loss_fn",
    "clip_coef_fn",
]
