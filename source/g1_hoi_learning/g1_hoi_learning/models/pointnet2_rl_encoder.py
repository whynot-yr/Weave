from typing import Optional, Tuple

import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2RLEncoder(nn.Module):
    r"""Per-point part-segmentation head built on the same SA hierarchy as
    ``PointNet2RLEncoder`` (reuses its SA_modules directly, so the encoder
    part is identical to the RL classification/embedding encoder), with a
    matching FeaturePropagation decoder added to restore per-point
    resolution. The predicted object category (one-hot) is concatenated in
    before the final per-point classifier, following the standard
    ShapeNetPart part-segmentation protocol.

    Parameters
    ----------
    input_channels : int
        Number of extra per-point feature channels beyond xyz.
    output_dim : int
        Size of the output embedding fed to the policy/value heads.
    use_xyz : bool
        Concatenate xyz into the per-point features used by each SA stage.
    """

    def __init__(
        self,
        input_channels: int = 0,
        output_dim: int = 128,
        use_xyz: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        c_in = input_channels
        self.SA_modules = nn.ModuleList(
            [
                PointnetSAModule(
                    npoint=64,
                    radius=0.2,
                    nsample=16,
                    mlp=[c_in, 32, 32, 64],
                    use_xyz=use_xyz,
                ),
                PointnetSAModule(
                    mlp=[64, 64, output_dim],
                    use_xyz=use_xyz,
                ),
            ]
        )

    def _break_up_pc(
        self, pc: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        pointcloud : torch.Tensor
            (B, N, 3 + input_channels) tensor, points formatted as
            (x, y, z, features...)

        Returns
        -------
        torch.Tensor
            (B, output_dim) embedding of the input point cloud.
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return features.squeeze(-1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = PointNet2RLEncoder(input_channels=0, output_dim=128).to(device)

    rollout_batch, minibatch, num_points = 4096, 512, 512

    with torch.no_grad():
        pc = torch.randn(rollout_batch, num_points, 3, device=device)
        out = encoder(pc)
        print(f"[rollout] input {tuple(pc.shape)} -> output {tuple(out.shape)}")

    pc = torch.randn(minibatch, num_points, 3, device=device)
    out = encoder(pc)
    out.sum().backward()
    print(f"[update]  input {tuple(pc.shape)} -> output {tuple(out.shape)}, backward ok")
