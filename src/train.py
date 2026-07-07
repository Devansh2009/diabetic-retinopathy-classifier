"""Two-phase transfer-learning training for APTOS 2019 DR grading.

Run from the repository root::

    python -m src.train                       # sensible defaults
    python -m src.train --imbalance focal      # try focal loss instead
    python -m src.train --head-epochs 30 --finetune-unfreeze 60

Phase 1 trains a fresh head on a frozen EfficientNetB0 backbone; Phase 2
unfreezes the top layers and fine-tunes at a low learning rate. The checkpoint is
selected on validation Quadratic Weighted Kappa, the competition metric.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import tensorflow as tf

from src.config import CLASS_NAMES, OUTPUT_DIR, Config
from src.data import (build_cache, compute_class_weights, download_dataset,
                      labels_from_df, make_dataset, stratified_split)
from src.losses import sparse_categorical_focal_loss
from src.metrics import QWKCallback
from src.model import build_model, freeze_backbone, unfreeze_top


def _loss_for(cfg: Config):
    if cfg.imbalance == "focal":
        return sparse_categorical_focal_loss(cfg.focal_gamma)
    return tf.keras.losses.SparseCategoricalCrossentropy()


def _callbacks(cfg: Config, val_ds, val_labels, patience: int
               ) -> List[tf.keras.callbacks.Callback]:
    return [
        QWKCallback(val_ds, val_labels),  # must run first — populates val_qwk
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor=cfg.monitor, mode=cfg.monitor_mode, patience=patience,
            restore_best_weights=True, verbose=1),
    ]


def _run_phase(model: tf.keras.Model, cfg: Config, train_ds, val_ds, val_labels,
               epochs: int, lr: float, patience: int,
               class_weight: Dict[int, float] | None) -> dict:
    """Compile and fit one phase; returns the Keras history dict."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=_loss_for(cfg),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=class_weight,
        callbacks=_callbacks(cfg, val_ds, val_labels, patience),
        verbose=1,
    )
    return history.history


def train(cfg: Config) -> Tuple[tf.keras.Model, dict]:
    """Execute the full pipeline and persist the best model + artifacts."""
    tf.keras.utils.set_random_seed(cfg.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    csv_path, img_dir = download_dataset()
    df = build_cache(csv_path, img_dir, cfg)
    train_df, val_df, test_df = stratified_split(df, cfg)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split[["id_code", "diagnosis"]].to_csv(
            OUTPUT_DIR / f"split_{name}.csv", index=False)
    print(f"[train] splits — train:{len(train_df)} val:{len(val_df)} "
          f"test:{len(test_df)}")

    class_weight = compute_class_weights(train_df) \
        if cfg.imbalance == "class_weights" else None
    if class_weight:
        print(f"[train] class weights: "
              f"{{{', '.join(f'{CLASS_NAMES[k]}:{v:.2f}' for k, v in class_weight.items())}}}")

    train_ds = make_dataset(train_df, cfg, training=True)
    val_ds = make_dataset(val_df, cfg, training=False)
    val_labels = labels_from_df(val_df)

    # --- Model ---
    model = build_model(cfg)

    # --- Phase 1: frozen backbone, train head ---
    print("\n[train] Phase 1 — training head on frozen backbone")
    freeze_backbone(model)
    hist1 = _run_phase(model, cfg, train_ds, val_ds, val_labels,
                       cfg.head_epochs, cfg.head_lr, cfg.head_patience,
                       class_weight)
    qwk1 = max(hist1["val_qwk"])
    model.save(OUTPUT_DIR / "phase1_best.keras")

    # --- Phase 2: fine-tune top layers ---
    print("\n[train] Phase 2 — fine-tuning top backbone layers")
    n_trainable = unfreeze_top(model, cfg.finetune_unfreeze, cfg.freeze_batchnorm)
    print(f"[train] trainable layers: {n_trainable}")
    hist2 = _run_phase(model, cfg, train_ds, val_ds, val_labels,
                       cfg.finetune_epochs, cfg.finetune_lr,
                       cfg.finetune_patience, class_weight)
    qwk2 = max(hist2["val_qwk"])

    # --- Select overall best across phases ---
    if qwk2 >= qwk1:
        best_qwk, best_phase = qwk2, 2
        model.save(OUTPUT_DIR / "best_model.keras")
    else:
        best_qwk, best_phase = qwk1, 1
        model = tf.keras.models.load_model(
            OUTPUT_DIR / "phase1_best.keras", compile=False)
        model.save(OUTPUT_DIR / "best_model.keras")

    history = {
        "phase1": hist1,
        "phase2": hist2,
        "best_val_qwk": best_qwk,
        "best_phase": best_phase,
        "config": cfg.as_dict(),
    }
    with open(OUTPUT_DIR / "history.json", "w") as fh:
        json.dump(history, fh, indent=2, default=float)
    print(f"\n[train] Best model: phase {best_phase}, val_qwk={best_qwk:.4f}")
    print(f"[train] Saved → {OUTPUT_DIR / 'best_model.keras'}")
    return model, history


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train the APTOS 2019 DR classifier.")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--img-size", type=int, default=Config.img_size)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--imbalance", choices=["class_weights", "focal"],
                   default=Config.imbalance)
    p.add_argument("--head-epochs", type=int, default=Config.head_epochs)
    p.add_argument("--finetune-epochs", type=int, default=Config.finetune_epochs)
    p.add_argument("--finetune-unfreeze", type=int,
                   default=Config.finetune_unfreeze)
    args = p.parse_args()
    return Config(
        seed=args.seed, img_size=args.img_size, batch_size=args.batch_size,
        imbalance=args.imbalance, head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_unfreeze=args.finetune_unfreeze,
    )


if __name__ == "__main__":
    train(_parse_args())
