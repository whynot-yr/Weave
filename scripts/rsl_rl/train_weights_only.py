"""Train from checkpoint weights with a fresh optimizer and iteration counter.

This wrapper uses the regular training entry point and changes only checkpoint
loading semantics. Model parameters, observation normalizers, and policy noise
parameters are loaded; optimizer state and the previous iteration are not.
"""

import os
import runpy
import sys
from pathlib import Path


os.environ["G1_HOI_WEIGHTS_ONLY"] = "1"

if "--resume" not in sys.argv:
    sys.argv.append("--resume")

train_script = Path(__file__).with_name("train.py")
runpy.run_path(train_script, run_name="__main__")
