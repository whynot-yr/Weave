"""PointNet2RLSegmenter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule

from .pointnet2_rl_encoder import PointNet2RLEncoder

# ShapeNetPart constants (from ShapeNetPartLoader): 16 object categories, 50 parts.
NUM_OBJECT_CLASSES = 16
NUM_PART_CLASSES = 50


class PointNet2RLSegmenter(nn.Module):
    r"""Per-point part-segmentation head built on the same SA hierarchy as
    ``PointNet2RLEncoder`` (reuses its SA_modules directly), with a matching
    FeaturePropagation decoder to restore per-point resolution.
    """

    def __init__(self, input_channels=0, output_dim=128, num_classes=NUM_OBJECT_CLASSES, num_part_classes=NUM_PART_CLASSES):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.encoder = PointNet2RLEncoder(input_channels=input_channels, output_dim=output_dim)

        sa1_channels = 64  # PointNet2RLEncoder SA1 mlp[-1]
        self.FP_modules = nn.ModuleList(
            [
                PointnetFPModule(mlp=[output_dim + sa1_channels, 128, 128]),
                PointnetFPModule(mlp=[128 + input_channels, 128, 128]),
            ]
        )

        self.head = nn.Sequential(
            nn.Conv1d(128 + num_classes, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(128, num_part_classes, kernel_size=1),
        )

    def extract_point_features(self, pointcloud):
        r"""Per-point features from the SA+FP backbone, before the category
        one-hot conditioning and final classifier head.

        Returns
        -------
        torch.Tensor
            (B, N, 128) per-point feature.
        """
        xyz, features = self.encoder._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for module in self.encoder.SA_modules:
            li_xyz, li_features = module(l_xyz[-1], l_features[-1])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        l_features[1] = self.FP_modules[0](l_xyz[1], l_xyz[2], l_features[1], l_features[2])
        l_features[0] = self.FP_modules[1](l_xyz[0], l_xyz[1], l_features[0], l_features[1])

        return l_features[0].transpose(1, 2).contiguous()  # (B, N, 128)

    def forward(self, pointcloud, cls_label):
        point_features = self.extract_point_features(pointcloud).transpose(1, 2)  # (B, 128, N)

        one_hot = F.one_hot(cls_label, self.num_classes).float()  # (B, num_classes)
        one_hot = one_hot.unsqueeze(-1).expand(-1, -1, point_features.size(-1))
        point_features = torch.cat([point_features, one_hot], dim=1)

        logits = self.head(point_features)  # (B, num_part_classes, N)
        return logits.transpose(1, 2).contiguous()  # (B, N, num_part_classes)
