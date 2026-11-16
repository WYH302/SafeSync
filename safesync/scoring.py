"""One change score, one known-only novelty score, and a fixed evaluator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def change_score(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.clip(1 - np.sum(normalize(current) * normalize(reference), axis=-1), 0, 2)


@dataclass
class KnownDistribution:
    mean: np.ndarray
    variance: np.ndarray
    calibration_scores: np.ndarray

    @classmethod
    def fit(cls, known_features: np.ndarray, variance_floor=1e-4):
        values = normalize(known_features)
        if len(values) < 100:
            raise ValueError(f"only {len(values)} labelled-known calibration tokens; need >=100")
        mean = values.mean(axis=0)
        variance = np.maximum(values.var(axis=0), variance_floor)
        scores = np.mean((values - mean)**2 / variance, axis=-1)
        return cls(mean, variance, np.sort(scores))

    def score(self, features: np.ndarray) -> np.ndarray:
        return np.mean((normalize(features) - self.mean)**2 / self.variance, axis=-1)

    def known_quantile(self, q: float) -> float:
        return float(np.quantile(self.calibration_scores, q))


def safety_gate(change, unknown, change_limit, unknown_limit, confidence=None, drift=None, acknowledgement_known=True):
    """Budget starvation never enters the stable set. This is empirical gating."""
    c, u = np.asarray(change), np.asarray(unknown)
    q = np.ones_like(c) if confidence is None else np.asarray(confidence)
    d = np.zeros_like(c) if drift is None else np.asarray(drift)
    return (c <= change_limit) & (u <= unknown_limit) & (q >= 0.95) & (d <= 0.1) & acknowledgement_known


def choose_change_threshold(scores: np.ndarray, changed: np.ndarray, valid: np.ndarray):
    """Only calibration labels enter this function. Tie rule is deterministic."""
    x, y = np.asarray(scores)[valid], np.asarray(changed)[valid]
    if not len(x) or not y.any() or y.all():
        raise ValueError("calibration must contain both changed and unchanged valid units")
    thresholds = np.unique(np.r_[0.0, np.quantile(x, np.linspace(0, 1, 201)), 2.0])
    def f1(threshold):
        pred = x > threshold
        tp, fp, fn = np.sum(pred & y), np.sum(pred & ~y), np.sum(~pred & y)
        return 2 * tp / max(1, 2 * tp + fp + fn)
    return float(max(thresholds, key=lambda t: (f1(t), t)))
