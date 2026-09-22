from __future__ import annotations

import torch
import omni.usd
from pxr import Gf, UsdLux

from isaaclab.envs import ManagerBasedEnv


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    light_path: str,
    intensity_range: tuple[float, float],
    color_temperature_range: tuple[float, float] | None = None,
    color_range: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None,
) -> None:
    """Randomize the global DomeLight shared by all environments."""

    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath(light_path)

    if not light_prim.IsValid():
        raise RuntimeError(f"Dome light not found at: {light_path}")

    dome_light = UsdLux.DomeLight(light_prim)
    if not dome_light:
        raise RuntimeError(f"Prim is not a DomeLight: {light_path}")

    intensity = torch.empty(1).uniform_(
        intensity_range[0],
        intensity_range[1],
    ).item()
    light_prim.GetAttribute("inputs:intensity").Set(float(intensity))

    if color_temperature_range is not None:
        temperature = torch.empty(1).uniform_(
            color_temperature_range[0],
            color_temperature_range[1],
        ).item()

        light_prim.GetAttribute(
            "inputs:enableColorTemperature"
        ).Set(True)
        light_prim.GetAttribute(
            "inputs:colorTemperature"
        ).Set(float(temperature))

    if color_range is not None:
        color_min = torch.tensor(color_range[0])
        color_max = torch.tensor(color_range[1])
        color = color_min + torch.rand(3) * (color_max - color_min)

        light_prim.GetAttribute("inputs:color").Set(
            Gf.Vec3f(*color.tolist())
        )
