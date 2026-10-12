"""Stage F: candidate-region merging with provenance, and the AnyChange learned localizer.

Object masks are not change masks: AnyChange/SAM masks are recorded as `learned_change`
evidence with location_semantics='containing_object' unless the pixel branch confirms
which pixels changed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .schema import Adapter, boxes_related


def merge_candidates(cands, iou=0.3, contain=0.7, image_area=None, max_container_frac=0.5):
    """cands: [{'bbox_a', 'source', 'obs_id', ...}] -> merged groups with union box + provenance.
    A box covering > max_container_frac of the image never absorbs others by containment
    (one huge, sparse component must not swallow every unrelated change)."""
    from .schema import box_area, box_iou
    parent = list(range(len(cands)))
    big = [image_area is not None and box_area(c["bbox_a"]) > max_container_frac * image_area for c in cands]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            rel = (box_iou(cands[i]["bbox_a"], cands[j]["bbox_a"]) >= iou if (big[i] or big[j])
                   else boxes_related(cands[i]["bbox_a"], cands[j]["bbox_a"], iou, contain))
            if rel:
                parent[find(i)] = find(j)
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault(find(i), []).append(c)
    out = []
    for members in groups.values():
        bs = np.array([m["bbox_a"] for m in members])
        out.append({"bbox_a": [float(bs[:, 0].min()), float(bs[:, 1].min()), float(bs[:, 2].max()), float(bs[:, 3].max())],
                    "members": members, "sources": sorted({m["source"] for m in members})})
    return out


class AnyChangeAdapter(Adapter):
    name = "anychange"
    heavy = True

    def __init__(self, cfg, device):
        self.model_type, self.ckpt, self.device = cfg.anychange_model_type, cfg.anychange_sam_checkpoint, device
        self.max_side, self.thr = cfg.anychange_max_side, cfg.anychange_conf_threshold

    def check_environment(self):
        import os
        try:
            import torchange  # noqa: F401
        except ImportError as e:
            return {"ok": False, "error": str(e), "fix": "pip install torchange"}
        if not os.path.exists(self.ckpt):
            return {"ok": False, "error": f"SAM checkpoint missing: {self.ckpt}",
                    "fix": "download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"}
        return {"ok": True}

    def load_model(self, cfg=None):
        import torchange
        from torchange.models.segment_any_change import AnyChange
        self.m = AnyChange(self.model_type, sam_checkpoint=self.ckpt)
        # upstream README defaults
        self.m.make_mask_generator(points_per_side=32, stability_score_thresh=0.95)
        self.m.set_hyperparameters(change_confidence_threshold=self.thr, use_normalized_feature=True,
                                   bitemporal_match=True)
        try:
            self.m.to(self.device)
        except Exception:
            pass
        self.version = f"torchange-{getattr(torchange, '__version__', '?')} AnyChange/{self.model_type}"

    @staticmethod
    def _rle_to_mask(rle):
        try:
            from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask
        except ImportError:
            from segment_anything.utils.amg import rle_to_mask
        return rle_to_mask(rle)

    def predict(self, image_a, aligned_b, context=None):
        """Inputs must be co-registered (aligned B in A's frame). Returns boxes/masks in A coords."""
        H, W = image_a.shape[:2]
        s = min(1.0, self.max_side / max(H, W))
        a = cv2.resize(image_a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image_a
        b = cv2.resize(aligned_b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        changemasks, _, _ = self.m.forward(a, b)
        out = []
        rles = changemasks["rles"]
        has_boxes = "boxes" in getattr(changemasks, "_stats", {})
        for i, rle in enumerate(rles):
            m = self._rle_to_mask(rle).astype(bool)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            box = [xs.min() / s, ys.min() / s, (xs.max() + 1) / s, (ys.max() + 1) / s]
            conf = None
            for key in ("change_confidence", "confidence"):
                if key in getattr(changemasks, "_stats", {}):
                    conf = float(np.asarray(changemasks[key][i]).ravel()[0])
            out.append({"bbox_a": [float(v) for v in box], "area_px": int(m.sum() / (s * s)), "raw_score": conf,
                        "mask_analysis": m, "mask_scale": s, "boxes_reported": has_boxes})
        return out
