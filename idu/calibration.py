"""Stage H: calibration and selective-prediction metrics.

Without a held-out labelled set nothing here runs and every calibrated_confidence stays None.
"""
from __future__ import annotations

import json

import numpy as np

MIN_SAMPLES = 50


def ece(probs, labels, n_bins=10):
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append({"bin": [float(lo), float(hi)], "n": int(m.sum()), "conf": float(p[m].mean()), "acc": float(y[m].mean())})
    return float(e), rows


def risk_coverage(scores, correct):
    """Sort by descending score; risk = error rate among the retained top-k. Returns curve + AURC."""
    s, c = np.asarray(scores, float), np.asarray(correct, float)
    if len(s) == 0:
        return {"coverage": [], "risk": [], "aurc": None}
    o = np.argsort(-s)
    cum_err = np.cumsum(1 - c[o])
    k = np.arange(1, len(s) + 1)
    risk, cov = cum_err / k, k / len(s)
    return {"coverage": cov.tolist(), "risk": risk.tolist(), "aurc": float((getattr(np, "trapezoid", None) or np.trapz)(risk, cov)) if len(s) > 1 else float(risk[0])}


class Calibrator:
    """Maps a raw score to P(claim correct). method: platt | isotonic | temperature (for logits)."""

    def __init__(self, method="platt"):
        self.method, self.params = method, {}

    def fit(self, scores, labels):
        x, y = np.asarray(scores, float), np.asarray(labels, int)
        if len(x) < MIN_SAMPLES or len(set(y.tolist())) < 2:
            raise ValueError(f"need >= {MIN_SAMPLES} labelled samples with both classes, got {len(x)}")
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e4).fit(x[:, None], y)
            self.params = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, y)
            self.params = {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}
        elif self.method == "temperature":
            from scipy.optimize import minimize_scalar

            def nll(t):
                p = 1 / (1 + np.exp(-x / t))
                return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            self.params = {"T": float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)}
        else:
            raise ValueError(self.method)
        self.params["n_fit"] = int(len(x))
        return self

    def predict(self, scores):
        x = np.asarray(scores, float)
        if self.method == "platt":
            return 1 / (1 + np.exp(-(self.params["a"] * x + self.params["b"])))
        if self.method == "isotonic":
            return np.interp(x, self.params["x"], self.params["y"])
        return 1 / (1 + np.exp(-x / self.params["T"]))

    def save(self, path):
        json.dump({"method": self.method, "params": self.params}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        c = cls(d["method"])
        c.params = d["params"]
        return c


def apply_calibrator(claims, path):
    if not path:
        return "not_available (no calibrator_path; scores are uncalibrated heuristics)"
    cal = Calibrator.load(path)
    for c in claims:
        if c.raw_support_score is not None:
            c.calibrated_confidence = round(float(cal.predict([c.raw_support_score])[0]), 4)
    return f"applied {cal.method} calibrator fitted on n={cal.params.get('n_fit')} ({path})"
