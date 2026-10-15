"""Compatibility import for the restored 3904770 PPO+BC training algorithm."""

from g1_hoi_learning.algorithms.distillation.algorithm import MuonPPODistill

DaggerAlgorithm = MuonPPODistill

__all__ = ["DaggerAlgorithm"]
