"""Read IsaacLab config snapshots, including tuple and slice values."""

from pathlib import Path

import yaml


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def read_yaml(path: str) -> dict:
    result = yaml.load(Path(path).read_text(), Loader=ConfigLoader)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return result
