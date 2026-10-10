# Adapted from NVIDIA Isaac-GR00T embodiment_conditioned_mlp.py (Apache-2.0).
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T two-layer category-specific decoder, 768 -> 768 -> velocity125."""

from torch import nn
from torch.nn import functional as F

from .reference_encoder import CategorySpecificLinear


class ReferenceDecoder(nn.Module):
    def __init__(self, width=768, reference_dim=125, num_embodiments=1):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_embodiments, width, width)
        self.layer2 = CategorySpecificLinear(num_embodiments, width, reference_dim)

    def forward(self, features, category_ids):
        return self.layer2(F.relu(self.layer1(features, category_ids)), category_ids)
