"""Loss functions for imbalanced ordinal classification.

Two complementary strategies for the heavy No_DR skew are supported:

* **Class-weighted cross-entropy** (default) — pass inverse-frequency weights to
  ``model.fit(class_weight=...)`` with a standard cross-entropy loss.
* **Focal loss** — down-weights easy, well-classified majority examples so the
  gradient focuses on the rare, hard severe/proliferative cases.
"""
from __future__ import annotations

import tensorflow as tf


def sparse_categorical_focal_loss(gamma: float = 2.0):
    """Return a focal-loss function for integer labels and softmax outputs.

    Args:
        gamma: Focusing parameter. ``gamma=0`` reduces to plain cross-entropy;
            larger values place more emphasis on misclassified examples.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.gather(y_pred, y_true, batch_dims=1)      # prob of true class
        cross_entropy = -tf.math.log(p_t)
        modulating = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(modulating * cross_entropy)

    loss.__name__ = f"focal_loss_gamma{gamma}"
    return loss
