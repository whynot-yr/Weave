"""Optional numerical comparison with the user's installed GR00T sources."""

import importlib.util
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from weave_policy.config import ReferencePolicyConfig
from weave_policy.models.dit import ReferenceDiT
from weave_policy.models.reference_decoder import ReferenceDecoder
from weave_policy.models.reference_encoder import ReferenceEncoder


def original(name):
    root = Path(os.environ.get("GR00T_SOURCE", "/home/whynot/Isaac-GR00T"))
    path = root / "gr00t/model/modules" / (name + ".py")
    if not path.is_file():
        pytest.skip("Set GR00T_SOURCE for source parity checks")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_time_mlp_and_decoder_match_gr00t():
    groot = original("embodiment_conditioned_mlp")
    ours, theirs = ReferenceEncoder(), groot.MultiEmbodimentActionEncoder(125, 768, 1)
    theirs.load_state_dict(ours.state_dict())
    value, time, ids = torch.randn(2, 10, 125), torch.tensor([0, 999]), torch.zeros(2, dtype=torch.long)
    torch.testing.assert_close(ours(value, time, ids), theirs(value, time, ids), atol=2e-5, rtol=2e-5)
    ours, theirs = ReferenceDecoder(), groot.CategorySpecificMLP(1, 768, 768, 125)
    theirs.load_state_dict(ours.state_dict())
    value = torch.randn(2, 10, 768)
    torch.testing.assert_close(ours(value, ids), theirs(value, ids), atol=0, rtol=0)


def test_dit_blocks_time_and_output_match_gr00t():
    pytest.importorskip("diffusers")
    groot = original("dit")
    config = replace(ReferencePolicyConfig(), width=32, num_heads=2, head_dim=16, num_layers=4, dropout=0)
    ours = ReferenceDiT(config).eval()
    theirs = groot.DiT(
        num_attention_heads=2,
        attention_head_dim=16,
        output_dim=32,
        num_layers=4,
        dropout=0,
        cross_attention_dim=768,
        positional_embeddings=None,
        interleave_self_attention=True,
    ).eval()
    theirs.load_state_dict(ours.state_dict(), strict=True)
    value, context, time = torch.randn(2, 10, 32), torch.randn(2, 257, 768), torch.tensor([250, 999])
    with torch.no_grad():
        torch.testing.assert_close(ours.timestep_encoder(time), theirs.timestep_encoder(time), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(ours(value, context, time), theirs(value, context, time), atol=2e-5, rtol=2e-5)
