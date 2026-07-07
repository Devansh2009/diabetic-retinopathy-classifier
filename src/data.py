"""Data acquisition, fundus preprocessing and ``tf.data`` input pipelines.

The APTOS 2019 Blindness Detection training set ships as ~3,662 high-resolution
retinal fundus photographs plus a ``train.csv`` mapping each ``id_code`` to an
ordinal ``diagnosis`` in {0..4}. This module:

1. Downloads the data — Kaggle (via ``kagglehub``) is the primary source, with a
   credential-free Hugging Face mirror as an automatic fallback.
2. Preprocesses each image once (crop the black border, optional circular mask,
   resize) and caches the result so training I/O stays cheap.
3. Builds stratified train/val/test splits and augmented ``tf.data`` pipelines.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import NUM_CLASSES, PROCESSED_DIR, RAW_DIR, Config
from src.preprocessing import process_one_image

AUTOTUNE = tf.data.AUTOTUNE
HF_REPO = "Tejaswini628/aptos-fundus-images"  # public mirror of the APTOS 2019 files


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def _locate_csv_and_images(root: Path) -> Tuple[Path, Path]:
    """Find ``train.csv`` and the folder holding ``train_images`` under ``root``."""
    csv_path = next(root.rglob("train.csv"))
    img_dir = next(p for p in root.rglob("train_images") if p.is_dir())
    return csv_path, img_dir


def download_dataset(raw_dir: Path = RAW_DIR) -> Tuple[Path, Path]:
    """Ensure the APTOS 2019 train set is present locally.

    Returns a ``(train_csv, train_images_dir)`` tuple. Tries Kaggle first (the
    canonical competition source) and falls back to a public Hugging Face mirror
    when Kaggle credentials are unavailable.
    """
    raw_dir = Path(raw_dir)
    if raw_dir.exists():
        try:
            return _locate_csv_and_images(raw_dir)
        except StopIteration:
            pass  # partial/empty download — re-fetch below

    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Primary: Kaggle via kagglehub ---
    try:
        import kagglehub

        print("[data] Downloading APTOS 2019 via kagglehub (Kaggle competition)…")
        path = Path(
            kagglehub.competition_download("aptos2019-blindness-detection")
        )
        csv_path, img_dir = _locate_csv_and_images(path)
        print(f"[data] Kaggle download ready at {path}")
        return csv_path, img_dir
    except Exception as exc:  # noqa: BLE001 — any auth/network issue triggers fallback
        print(f"[data] kagglehub unavailable ({exc}). Falling back to Hugging Face mirror…")

    # --- Fallback: public Hugging Face mirror (no credentials required) ---
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    snap = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(raw_dir / "hf"),
        allow_patterns=["data/train.csv", "data/train_images/*.png"],
        max_workers=16,
    )
    csv_path, img_dir = _locate_csv_and_images(Path(snap))
    print(f"[data] Hugging Face mirror ready at {snap}")
    return csv_path, img_dir


# ---------------------------------------------------------------------------
# 2. Preprocess once into a resized cache
# ---------------------------------------------------------------------------
def build_cache(csv_path: Path, img_dir: Path, cfg: Config,
                processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Preprocess every image once (in parallel) into a resized cache.

    Returns a dataframe with columns ``id_code``, ``diagnosis`` and ``path``
    pointing at the cached, model-ready images.
    """
    processed_dir = Path(processed_dir)
    img_out = processed_dir / f"images_{cfg.img_size}"
    img_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    tasks, paths = [], []
    for id_code in df["id_code"]:
        src = str(Path(img_dir) / f"{id_code}.png")
        dst = str(img_out / f"{id_code}.png")
        tasks.append((src, dst, cfg.img_size, cfg.circle_crop))
        paths.append(dst)

    pending = [t for t in tasks if not os.path.exists(t[1])]
    if pending:
        print(f"[data] Preprocessing {len(pending)} images → {img_out} …")
        # OpenCV releases the GIL in its native calls, so threads parallelise the
        # decode/resize well while avoiding process fork/spawn hazards on macOS.
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            list(ex.map(process_one_image, pending))
    else:
        print(f"[data] Using cached preprocessed images at {img_out}")

    df["path"] = paths
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Splits, class weights and tf.data pipelines
# ---------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, cfg: Config
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic, class-stratified train/val/test split."""
    test_frac = cfg.test_split
    val_frac = cfg.val_split
    train_df, temp_df = train_test_split(
        df, test_size=test_frac + val_frac, stratify=df["diagnosis"],
        random_state=cfg.seed)
    rel_val = val_frac / (test_frac + val_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - rel_val, stratify=temp_df["diagnosis"],
        random_state=cfg.seed)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def compute_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """Inverse-frequency class weights for the (imbalanced) training split."""
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["diagnosis"].values)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_augmenter() -> tf.keras.Sequential:
    """Fundus-safe augmentation.

    Flips and rotations are label-preserving for retinal images (there is no
    canonical orientation), and mild brightness/contrast jitter mimics
    acquisition variability. We deliberately avoid shear/elastic warps that would
    distort the morphology of microaneurysms, haemorrhages and exudates.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.10),
    ], name="fundus_augmentation")


def _decode(path: tf.Tensor, label: tf.Tensor, size: int):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (size, size))  # no-op if already cached at size
    return tf.cast(img, tf.float32), label


def make_dataset(df: pd.DataFrame, cfg: Config, training: bool) -> tf.data.Dataset:
    """Build a batched, prefetched ``tf.data.Dataset`` from a split dataframe.

    Decoded images are cached in memory (the resized set is small), so only the
    first epoch pays decode cost; augmentation is applied per-epoch after the
    cache so every epoch still sees fresh transforms.
    """
    paths = df["path"].values
    labels = df["diagnosis"].values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _decode(p, y, cfg.img_size),
                num_parallel_calls=AUTOTUNE).cache()

    if training:
        augment = build_augmenter()
        ds = ds.shuffle(len(df), seed=cfg.seed, reshuffle_each_iteration=True)
        ds = ds.batch(cfg.batch_size)
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.batch(cfg.batch_size)
    return ds.prefetch(AUTOTUNE)


def labels_from_df(df: pd.DataFrame) -> np.ndarray:
    """Return the integer label vector for a split (used by the QWK callback)."""
    return df["diagnosis"].values.astype("int32")
