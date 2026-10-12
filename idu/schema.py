"""Typed records shared by every stage, plus geometry helpers and the adapter contract.

Coordinate convention (used everywhere):
  * boxes are [x1, y1, x2, y2] floats in ORIGINAL image pixels, after EXIF orientation
    correction; origin top-left, x right, y down; x2/y2 are exclusive.
  * `M_ba` is a 3x3 matrix mapping Image-B pixel coords into Image-A pixel coords.
  * Image A is the reference frame for aligned views and difference maps.
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

DECISIONS = ("accepted", "needs_review", "abstained")


@dataclass
class Observation:
    observation_id: str
    source: str                      # model or algorithm name
    source_version: Optional[str]    # checkpoint id / package version / git revision
    observation_type: str            # e.g. lowlevel_region, ocr_line, text_diff, vlm_hypothesis
    family: str                      # evidence family used for independence accounting
    image: str                       # "A", "B" or "AB"
    bbox_a: Optional[list] = None
    bbox_b: Optional[list] = None
    mask_ref: Optional[str] = None
    raw: Any = None                  # raw model output / measurements
    raw_score: Optional[float] = None
    score_kind: Optional[str] = None  # what raw_score means; never implies calibration
    calibrated_confidence: Optional[float] = None
    calibration_status: str = "uncalibrated"
    context: dict = field(default_factory=dict)   # alignment + quality context
    supports: list = field(default_factory=list)
    contradicts: list = field(default_factory=list)
    depends_on: list = field(default_factory=list)


@dataclass
class Claim:
    claim_id: str
    claim_type: str                  # text_change | visual_change | semantic_hypothesis
    change_type: str                 # added | removed | color_change | text_substitution | ...
    object_or_region: Optional[str] = None
    bbox_a: Optional[list] = None
    bbox_b: Optional[list] = None
    bbox_b_method: Optional[str] = None
    mask_ref: Optional[str] = None
    location_semantics: str = "approximate_hint"  # changed_pixels | text_region | containing_object | approximate_hint
    localization_method: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None
    supporting: list = field(default_factory=list)      # observation ids
    contradicting: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)      # per-dimension raw / categorical values
    raw_support_score: Optional[float] = None           # heuristic ranking score, NOT a probability
    calibrated_confidence: Optional[float] = None
    nuisance_explanations: list = field(default_factory=list)
    decision: str = "needs_review"
    decision_reasons: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


class IdGen:
    def __init__(self):
        self._c = {}

    def __call__(self, prefix: str) -> str:
        c = self._c.setdefault(prefix, itertools.count(1))
        return f"{prefix}_{next(c):03d}"


class Adapter:
    """Contract for every specialist. `name`/`version` end up in observation provenance."""
    name = "adapter"
    version: Optional[str] = None
    heavy = False   # heavy adapters are swapped out of VRAM before another heavy one loads

    def check_environment(self) -> dict:
        return {"ok": True}

    def load_model(self, cfg) -> None:
        pass

    def predict(self, image_a, image_b, context: dict):
        raise NotImplementedError

    def normalize_output(self, raw):
        return raw

    def release_resources(self) -> None:
        for k in list(vars(self)):
            if k not in ("name", "version", "heavy"):
                setattr(self, k, None)
        free_cuda()


def free_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------- geometry ----------------
def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = box_area(a) + box_area(b) - inter
    return inter / u if u > 0 else 0.0


def box_containment(small, big) -> float:
    """Fraction of `small` inside `big`."""
    ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    a = box_area(small)
    return ix * iy / a if a > 0 else 0.0


def boxes_related(a, b, iou=0.3, contain=0.7) -> bool:
    if a is None or b is None:
        return False
    sa, sb = (a, b) if box_area(a) <= box_area(b) else (b, a)
    return box_iou(a, b) >= iou or box_containment(sa, sb) >= contain


def clip_box(b, w, h):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    return [float(x1), float(y1), float(x2), float(y2)]


def pad_box(b, pad, w, h):
    return clip_box([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], w, h)


def transform_box(b, M):
    """Apply a 3x3 projective transform to a box's corners and return the enclosing box."""
    pts = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[2], b[3], 1], [b[0], b[3], 1]], float)
    q = pts @ np.asarray(M, float).T
    q = q[:, :2] / np.clip(q[:, 2:3], 1e-9, None)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def to_jsonable(o):
    if dataclasses.is_dataclass(o):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(o).items()}
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<ndarray shape={o.shape} dtype={o.dtype}>"
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
