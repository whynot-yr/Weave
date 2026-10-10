"""Allow these scripts to run from a source checkout without Isaac imports."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/weave_policy"))

from weave_policy.config import ReferencePolicyConfig


def read_config(path=None):
    return ReferencePolicyConfig.from_dict(json.loads(Path(path).read_text())) if path else ReferencePolicyConfig()


def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
