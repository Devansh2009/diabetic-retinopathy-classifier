"""Grad-CAM visual explanations for the DR classifier.

Grad-CAM highlights the image regions most responsible for a prediction by
weighting the final convolutional feature maps with the gradient of the target
class score. On fundus images this should localise the lesions a clinician looks
for — microaneurysms, haemorrhages and exudates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.model import last_conv_layer_name


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model,
                         layer_name: Optional[str] = None,
                         class_index: Optional[int] = None
                         ) -> Tuple[np.ndarray, int]:
    """Compute a Grad-CAM heatmap for one image.

    Args:
        img_array: Batch of shape ``(1, H, W, 3)``, float32 in ``[0, 255]``.
        model: Trained classifier (flat graph).
        layer_name: Target conv layer; defaults to the last Conv2D.
        class_index: Class to explain; defaults to the predicted class.

    Returns:
        ``(heatmap, class_index)`` where ``heatmap`` is a ``[0, 1]`` array the
        size of the target feature map.
    """
    layer_name = layer_name or last_conv_layer_name(model)
    grad_model = tf.keras.Model(
        model.input, [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(class_index)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4,
                    mask_background: bool = True) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an RGB image and return an RGB uint8 array.

    When ``mask_background`` is set, activation outside the retinal disc (the
    black circle-crop border) is suppressed and the heatmap is renormalised
    within the retina, so the explanation only highlights informative pixels.
    """
    img = img.astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    retina = img.sum(axis=2) > 10  # True inside the (non-black) retinal disc
    if mask_background:
        heatmap = heatmap * retina
        peak = heatmap.max()
        if peak > 0:
            heatmap = heatmap / peak

    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    out = np.uint8(colored * alpha + img * (1 - alpha))
    if mask_background:
        out[~retina] = img[~retina]  # leave the background as-is
    return out
