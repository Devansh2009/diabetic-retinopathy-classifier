"""Central configuration for the diabetic retinopathy classifier.

All tunable knobs live here so that scripts, the training loop and the notebook
share a single source of truth. Override any field from the command line via
``src/train.py`` flags (see ``Config.from_args``).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# --- Project layout ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # original full-resolution fundus images
PROCESSED_DIR = DATA_DIR / "processed"  # circle-cropped, resized cache
OUTPUT_DIR = ROOT / "outputs"        # models, metrics, split files
ASSETS_DIR = ROOT / "assets"         # plots/figures used by the README

# --- Task definition --------------------------------------------------------
# Class indices follow the APTOS 2019 clinical grading scale (ordinal).
CLASS_NAMES: List[str] = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class Config:
    """Hyper-parameters and run settings for one training experiment."""

    # Reproducibility
    seed: int = 42

    # Image / data pipeline
    img_size: int = 224                # EfficientNetB0 native resolution
    batch_size: int = 32
    val_split: float = 0.15            # fraction held out for validation
    test_split: float = 0.15           # fraction held out for final test
    circle_crop: bool = True           # mask out corners outside the retinal disc

    # Model
    dropout: float = 0.4
    backbone: str = "EfficientNetB0"

    # Imbalance handling: "class_weights" (weighted cross-entropy) or "focal"
    imbalance: str = "class_weights"
    focal_gamma: float = 2.0

    # Phase 1 — train the classification head with a frozen backbone
    head_epochs: int = 25
    head_lr: float = 1e-3
    head_patience: int = 6             # early-stopping patience (epochs)

    # Phase 2 — fine-tune the top layers of the backbone
    finetune_epochs: int = 20
    finetune_lr: float = 1e-4
    finetune_unfreeze: int = 40        # number of top backbone layers to unfreeze
    finetune_patience: int = 6
    freeze_batchnorm: bool = True      # keep BN statistics fixed while fine-tuning

    # Bookkeeping
    monitor: str = "val_qwk"           # metric that drives checkpoint/early-stop
    monitor_mode: str = "max"

    def as_dict(self) -> dict:
        return asdict(self)


# A ready-to-use default instance for interactive / notebook use.
CFG = Config()
