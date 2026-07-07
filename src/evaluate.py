"""Evaluate the trained DR classifier on the held-out test split.

Produces the numbers and figures reported in the README:

* Accuracy and Quadratic Weighted Kappa
* Per-class precision / recall / F1 (``classification_report``)
* Confusion matrix (raw + row-normalised)
* One-vs-rest ROC-AUC (macro and per class)
* Training-history curves
* Grad-CAM overlays (one confident example per severity grade)

Run from the repository root::

    python -m src.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from src.config import ASSETS_DIR, CLASS_NAMES, NUM_CLASSES, OUTPUT_DIR, Config
from src.data import build_cache, download_dataset, stratified_split
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.metrics import quadratic_weighted_kappa

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _predict(model: tf.keras.Model, paths: np.ndarray, img_size: int,
             batch_size: int = 32) -> np.ndarray:
    """Batched softmax predictions for a list of cached image paths."""
    probs = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        imgs = np.stack([
            cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype("float32")
            for p in batch])
        probs.append(model.predict(imgs, verbose=0))
    return np.concatenate(probs, axis=0)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title, fmt in [
        (axes[0], cm, "Confusion Matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion Matrix (row-normalised)", ".2f")]:
        im = ax.imshow(mat, cmap="Blues", vmin=0,
                       vmax=mat.max() if fmt == "d" else 1.0)
        ax.set_title(title)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        thresh = (mat.max() if fmt == "d" else 1.0) / 2
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path
             ) -> Dict[str, float]:
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        per_class[name] = float(auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves"); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return per_class


def plot_history(history: dict, out_path: Path) -> None:
    def _concat(key: str):
        return history["phase1"].get(key, []) + history["phase2"].get(key, [])

    boundary = len(history["phase1"].get("loss", []))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [("loss", "val_loss", "Loss"),
              ("accuracy", "val_accuracy", "Accuracy"),
              (None, "val_qwk", "Validation QWK")]
    for ax, (train_key, val_key, title) in zip(axes, panels):
        if train_key:
            ax.plot(_concat(train_key), label="train")
        ax.plot(_concat(val_key), label="val")
        ax.axvline(boundary - 0.5, color="grey", ls=":", label="fine-tune start")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_gradcam_grid(model: tf.keras.Model, test_df, img_size: int,
                      out_path: Path) -> None:
    """One confident, correctly-classified example per grade with Grad-CAM."""
    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 8))
    for cls in range(NUM_CLASSES):
        subset = test_df[test_df["diagnosis"] == cls]
        picked = None
        for _, row in subset.iterrows():
            img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
            arr = img.astype("float32")[None]
            heatmap, pred = make_gradcam_heatmap(arr, model)
            if pred == cls:  # prefer a correctly-classified, confident example
                picked = (img, heatmap); break
        if picked is None and len(subset):  # fall back to first available
            row = subset.iloc[0]
            img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
            heatmap, _ = make_gradcam_heatmap(img.astype("float32")[None], model, class_index=cls)
            picked = (img, heatmap)
        img, heatmap = picked
        axes[0, cls].imshow(img); axes[0, cls].set_title(CLASS_NAMES[cls])
        axes[1, cls].imshow(overlay_heatmap(img, heatmap))
        for r in (0, 1):
            axes[r, cls].axis("off")
    axes[0, 0].set_ylabel("original", rotation=90)
    fig.suptitle("Grad-CAM: regions driving each severity prediction", y=0.98)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def evaluate(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(OUTPUT_DIR / "best_model.keras",
                                       compile=False)

    csv_path, img_dir = download_dataset()
    df = build_cache(csv_path, img_dir, cfg)
    _, _, test_df = stratified_split(df, cfg)

    y_true = test_df["diagnosis"].values.astype(int)
    y_prob = _predict(model, test_df["path"].values, cfg.img_size, cfg.batch_size)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    plot_confusion_matrix(cm, ASSETS_DIR / "confusion_matrix.png")
    per_class_auc = plot_roc(y_true, y_prob, ASSETS_DIR / "roc_curves.png")
    macro_auc = float(np.mean(list(per_class_auc.values())))
    if (OUTPUT_DIR / "history.json").exists():
        with open(OUTPUT_DIR / "history.json") as fh:
            plot_history(json.load(fh), ASSETS_DIR / "training_history.png")
    plot_gradcam_grid(model, test_df, cfg.img_size, ASSETS_DIR / "gradcam.png")

    metrics = {
        "test_size": int(len(test_df)),
        "accuracy": float(acc),
        "quadratic_weighted_kappa": float(qwk),
        "macro_roc_auc_ovr": macro_auc,
        "per_class_roc_auc": per_class_auc,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n===== TEST-SET RESULTS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"QWK      : {qwk:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))
    print(f"[eval] figures → {ASSETS_DIR}")
    print(f"[eval] metrics → {OUTPUT_DIR / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    evaluate()
