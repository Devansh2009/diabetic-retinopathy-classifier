"""EfficientNetB0 transfer-learning model and freeze/unfreeze helpers.

The model is built as a single *flat* graph (the backbone is created with
``input_tensor=`` rather than being nested as a sub-model). This keeps every
convolutional layer directly addressable by name, which matters for both
selective fine-tuning and Grad-CAM.
"""
from __future__ import annotations

from typing import List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from src.config import NUM_CLASSES, Config

# Layers that make up our custom classification head (always trainable).
HEAD_LAYERS = ("head_gap", "head_dropout", "predictions")


def build_model(cfg: Config) -> tf.keras.Model:
    """Build EfficientNetB0 (ImageNet) with a fresh softmax classification head.

    EfficientNet includes its own rescaling/normalisation layers, so the model
    consumes raw ``[0, 255]`` RGB inputs directly.
    """
    inputs = tf.keras.Input((cfg.img_size, cfg.img_size, 3), name="input_image")
    backbone = EfficientNetB0(include_top=False, weights="imagenet",
                              input_tensor=inputs)
    backbone.trainable = False  # Phase 1: frozen feature extractor

    x = layers.GlobalAveragePooling2D(name="head_gap")(backbone.output)
    x = layers.Dropout(cfg.dropout, name="head_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                           name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="efficientnetb0_dr")


def _backbone_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    """All backbone layers (everything except the input and custom head)."""
    return [l for l in model.layers
            if l.name not in HEAD_LAYERS
            and not isinstance(l, tf.keras.layers.InputLayer)]


def freeze_backbone(model: tf.keras.Model) -> None:
    """Freeze the entire backbone (Phase 1 — head-only training)."""
    for layer in _backbone_layers(model):
        layer.trainable = False


def unfreeze_top(model: tf.keras.Model, n_layers: int,
                 freeze_batchnorm: bool = True) -> int:
    """Unfreeze the top ``n_layers`` of the backbone for fine-tuning.

    BatchNormalization layers are optionally kept frozen so their ImageNet
    running statistics are preserved — the standard recipe for stable fine-tuning
    on a small, domain-shifted dataset. Returns the number of trainable layers.
    """
    backbone = _backbone_layers(model)
    for layer in backbone:
        layer.trainable = False
    for layer in backbone[-n_layers:]:
        layer.trainable = True
    if freeze_batchnorm:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return sum(1 for l in model.layers if l.trainable)


def last_conv_layer_name(model: tf.keras.Model) -> str:
    """Name of the final convolutional layer (Grad-CAM target).

    EfficientNetB0's is ``top_conv``; we search defensively so the code also
    works for other backbones.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")
