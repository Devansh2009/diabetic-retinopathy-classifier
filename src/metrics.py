"""Quadratic Weighted Kappa (QWK) — the metric that matters for DR grading.

DR severity is *ordinal*: predicting "Severe" for a "Proliferative" case is a far
smaller error than predicting "No_DR". Plain accuracy treats every mistake
equally; QWK penalises predictions by the **squared distance** from the true
grade, so it rewards a model that stays close on the severity scale. It is the
official APTOS 2019 competition metric.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score

from src.config import NUM_CLASSES


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa with quadratic weights over the full label range."""
    return float(cohen_kappa_score(
        y_true, y_pred, weights="quadratic", labels=list(range(NUM_CLASSES))))


class QWKCallback(tf.keras.callbacks.Callback):
    """Compute validation QWK at each epoch and expose it as ``val_qwk``.

    Placed *first* in the callback list so that ``ModelCheckpoint`` and
    ``EarlyStopping`` (which run afterwards) can monitor ``val_qwk`` directly —
    letting us select the model that best matches the competition objective
    rather than the one with the lowest cross-entropy.
    """

    def __init__(self, val_ds: tf.data.Dataset, y_true: np.ndarray):
        super().__init__()
        self.val_ds = val_ds
        self.y_true = y_true

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        y_prob = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        qwk = quadratic_weighted_kappa(self.y_true, y_pred)
        logs["val_qwk"] = qwk
        print(f" — val_qwk: {qwk:.4f}")
