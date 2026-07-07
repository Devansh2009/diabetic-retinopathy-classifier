"""Fundus image preprocessing (OpenCV/NumPy only — no TensorFlow).

Kept dependency-light on purpose: ``build_cache`` fans these functions out across
worker processes, and importing TensorFlow in every worker would be wasteful.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def crop_to_retina(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the uninformative black border surrounding the circular retina.

    Fundus photographs are a bright disc on a black background, often with wide
    letterboxing. We threshold on luminance and crop to the bounding box of the
    retina so the informative region fills the frame after resizing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img  # degenerate (all-black) image — leave untouched
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]


def apply_circle_mask(img: np.ndarray) -> np.ndarray:
    """Zero out the corners outside the inscribed circle of the (square) image.

    Removes bright rectangular scanner artefacts in the corners that would
    otherwise distract the network and Grad-CAM.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)


def preprocess_fundus(img: np.ndarray, size: int, circle_crop: bool) -> np.ndarray:
    """Full preprocessing for a single RGB fundus image → ``size×size`` uint8.

    Note: no intensity normalisation is applied here. EfficientNet performs its
    own normalisation internally and expects pixel values in ``[0, 255]``.
    """
    img = crop_to_retina(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if circle_crop:
        img = apply_circle_mask(img)
    return img


def process_one_image(args) -> None:
    """Worker: read one raw image, preprocess it and write the cached PNG.

    ``args`` is ``(src_path, dst_path, size, circle_crop)`` — a single tuple so
    the function is trivially usable with ``Executor.map``.
    """
    src_path, dst_path, size, circle_crop = args
    if os.path.exists(dst_path):
        return
    bgr = cv2.imread(src_path)
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = preprocess_fundus(rgb, size, circle_crop)
    cv2.imwrite(dst_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
