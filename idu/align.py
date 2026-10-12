"""Stage B: correspondence, geometric-model selection, alignment quality and uncertainty.

Alignment is evidence, not truth: the aligned view is an additional view of B in A's frame,
originals are never modified, and every downstream claim carries the alignment status.
"""
from __future__ import annotations

import cv2
import numpy as np

from .schema import Adapter, transform_box, clip_box


def _to_tensor(img, device):
    import torch
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)


class ORBMatcher(Adapter):
    name = "orb_bf_ratio"

    def load_model(self, cfg):
        self.version = f"opencv-{cv2.__version__}"
        self.orb = cv2.ORB_create(nfeatures=max(cfg.max_keypoints, 5000), scaleFactor=1.2, nlevels=8)

    def predict(self, a, b, context=None):
        res = []
        for img in (a, b):
            s = min(1.0, 1600 / max(img.shape[:2]))
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
            kp, des = self.orb.detectAndCompute(g, None)
            pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2) / s
            res.append((pts, des))
        (p0, d0), (p1, d1) = res
        if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
        good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < 0.8 * n.distance]
        k0 = np.array([p0[m.queryIdx] for m in good]).reshape(-1, 2)
        k1 = np.array([p1[m.trainIdx] for m in good]).reshape(-1, 2)
        sc = np.array([1 - m.distance / 256 for m in good])
        return k0, k1, sc


class LightGlueMatcher(Adapter):
    name = "lightglue"

    def __init__(self, device):
        self.device = device

    def check_environment(self):
        try:
            import lightglue  # noqa: F401
            return {"ok": True}
        except ImportError as e:
            return {"ok": False, "error": str(e), "fix": "pip install git+https://github.com/cvg/LightGlue.git"}

    def load_model(self, cfg):
        import lightglue
        from lightglue import LightGlue, SuperPoint, DISK, ALIKED
        feats = cfg.lightglue_features
        ext = {"superpoint": SuperPoint, "disk": DISK, "aliked": ALIKED}[feats]
        self.extractor = ext(max_num_keypoints=cfg.max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features=feats).eval().to(self.device)
        self.version = f"lightglue+{feats} ({getattr(lightglue, '__version__', 'git')})"

    def predict(self, a, b, context=None):
        import torch
        from lightglue.utils import rbd
        with torch.inference_mode():
            f0 = self.extractor.extract(_to_tensor(a, self.device))   # keypoints returned in input px
            f1 = self.extractor.extract(_to_tensor(b, self.device))
            m = self.matcher({"image0": f0, "image1": f1})
            f0, f1, m = [rbd(x) for x in (f0, f1, m)]
            idx = m["matches"]
            k0 = f0["keypoints"][idx[:, 0]].cpu().numpy()
            k1 = f1["keypoints"][idx[:, 1]].cpu().numpy()
            sc = m["scores"].cpu().numpy()
        return k0, k1, sc


class LoFTRMatcher(Adapter):
    name = "loftr_kornia_outdoor"

    def __init__(self, device):
        self.device = device

    def check_environment(self):
        try:
            import kornia  # noqa: F401
            return {"ok": True}
        except ImportError as e:
            return {"ok": False, "error": str(e), "fix": "pip install kornia"}

    def load_model(self, cfg):
        import kornia
        self.model = kornia.feature.LoFTR(pretrained="outdoor").eval().to(self.device)
        self.version = f"kornia-{kornia.__version__}/LoFTR-outdoor"

    def predict(self, a, b, context=None):
        import torch
        ims, scales = [], []
        for img in (a, b):
            s = min(1.0, 840 / max(img.shape[:2]))
            h, w = int(img.shape[0] * s) // 8 * 8, int(img.shape[1] * s) // 8 * 8
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA)
            ims.append(torch.from_numpy(g).float().div(255)[None, None].to(self.device))
            scales.append((img.shape[1] / w, img.shape[0] / h))
        with torch.inference_mode():
            out = self.model({"image0": ims[0], "image1": ims[1]})
        k0 = out["keypoints0"].cpu().numpy() * np.array(scales[0])
        k1 = out["keypoints1"].cpu().numpy() * np.array(scales[1])
        return k0, k1, out["confidence"].cpu().numpy()


# ---------------- geometric model fitting ----------------
def _project(M, pts):
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(M, float).T
    return p[:, :2] / np.clip(p[:, 2:3], 1e-9, None)


def _fit(kind, kb, ka, thr):
    """Return 3x3 matrix mapping B->A, or None."""
    if kind == "identity":
        return np.eye(3)
    if len(ka) < 4:
        return None
    if kind == "similarity":
        A, _ = cv2.estimateAffinePartial2D(kb, ka, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                           maxIters=5000, confidence=0.999)
    elif kind == "affine":
        A, _ = cv2.estimateAffine2D(kb, ka, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                    maxIters=5000, confidence=0.999)
    else:
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, _ = cv2.findHomography(kb, ka, method, thr, maxIters=10000, confidence=0.999)
        return H
    return None if A is None else np.vstack([A, [0, 0, 1]])


def _sane(M, wb, hb, wa, ha):
    corners = np.array([[0, 0], [wb, 0], [wb, hb], [0, hb]], float)
    q = _project(M, corners).astype(np.float32)
    area = abs(cv2.contourArea(q))
    return bool(cv2.isContourConvex(q.reshape(-1, 1, 2))) and 0.05 < area / (wa * ha) < 20


def _ncc(a, b):
    ga, gb = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (256, 256)).astype(np.float32) for x in (a, b)]
    ga, gb = ga - ga.mean(), gb - gb.mean()
    return float((ga * gb).sum() / (np.sqrt((ga ** 2).sum() * (gb ** 2).sum()) + 1e-6))


DOMAIN_ORDER = {
    "document": ["identity", "homography", "similarity", "affine"],
    "screenshot": ["identity", "similarity", "homography"],
    "fixed_camera": ["identity", "similarity", "affine"],
}
DEFAULT_ORDER = ["identity", "similarity", "affine", "homography"]


def estimate_alignment(img_a, img_b, k_a, k_b, scores, cfg) -> dict:
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    same_size = (ha, wa) == (hb, wb)
    thr = cfg.ransac_thresh_px * max(1.0, max(ha, wa) / 2000)  # scale threshold with resolution
    k_a, k_b = np.asarray(k_a, np.float32).reshape(-1, 2), np.asarray(k_b, np.float32).reshape(-1, 2)
    scores = np.asarray(scores, float).reshape(-1)
    out = {"n_matches": int(len(k_a)), "ransac_thresh_px": thr, "candidates": {}, "reasons": []}

    order = DOMAIN_ORDER.get(cfg.domain, DEFAULT_ORDER)
    if not same_size:
        order = [k for k in order if k != "identity"]
    fits = {}
    for kind in order:
        M = _fit(kind, k_b, k_a, thr)
        if M is None or not np.all(np.isfinite(M)) or not _sane(M, wb, hb, wa, ha):
            continue
        err = np.linalg.norm(_project(M, k_b) - k_a, axis=1) if len(k_a) else np.zeros(0)
        inl = err < thr
        fits[kind] = (M, err, inl)
        out["candidates"][kind] = {"n_inliers": int(inl.sum()),
                                   "median_err": float(np.median(err[inl])) if inl.any() else None}
    best = max((f[2].sum() for f in fits.values()), default=0)
    chosen = next((k for k in order if k in fits and fits[k][2].sum() >= max(cfg.min_inliers, 0.95 * best)), None)

    if chosen is None:
        if same_size and _ncc(img_a, img_b) > 0.8:
            chosen, out["assumption"] = "identity", "few matches; identity assumed (global NCC > 0.8)"
            fits.setdefault("identity", (np.eye(3), np.zeros(len(k_a)), np.zeros(len(k_a), bool)))
            out["reasons"].append(out["assumption"])
        else:
            M = np.diag([wa / wb, ha / hb, 1.0])
            out.update(model="scale_only_assumed", M_ba=M.tolist(), status="failed",
                       n_inliers=0, inlier_ratio=0.0, overlap=1.0, coverage=0.0, inliers_a=np.zeros((0, 2)))
            out["reasons"].append("no geometric model with enough inliers; comparison uses naive rescale")
            return out

    M, err, inl = fits[chosen]
    n_in = int(inl.sum())
    out.update(model=chosen, M_ba=M.tolist(), n_inliers=n_in,
               inlier_ratio=float(n_in / max(len(k_a), 1)),
               median_err=float(np.median(err[inl])) if n_in else None,
               p90_err=float(np.percentile(err[inl], 90)) if n_in else None)
    pts = k_a[inl]
    grid = np.zeros((8, 8), bool)
    if n_in:
        gx = np.clip((pts[:, 0] / wa * 8).astype(int), 0, 7)
        gy = np.clip((pts[:, 1] / ha * 8).astype(int), 0, 7)
        grid[gy, gx] = True
    out["coverage"] = float(grid.mean())
    valid = cv2.warpPerspective(np.ones((hb, wb), np.uint8), M, (wa, ha), flags=cv2.INTER_NEAREST)
    out["overlap"] = float(valid.mean())
    conf_out = (scores > np.quantile(scores, 0.5)) & ~inl if len(scores) == len(inl) and len(scores) else np.zeros(0, bool)
    out["confident_outlier_frac"] = float(conf_out.mean()) if len(conf_out) else 0.0
    out["inliers_a"] = pts

    assumed = "assumption" in out
    if (not assumed and n_in >= 2 * cfg.min_inliers and out["inlier_ratio"] >= 0.5
            and (out["median_err"] or 0) <= 1.5 * thr / cfg.ransac_thresh_px and out["coverage"] >= 0.2
            and out["overlap"] >= 0.6):
        out["status"] = "reliable"
    elif (assumed or (n_in >= cfg.min_inliers and out["inlier_ratio"] >= 0.25)) and out["overlap"] >= 0.3:
        out["status"] = "approximate"
    else:
        out["status"] = "unreliable"
    if out["confident_outlier_frac"] > 0.3:
        out["reasons"].append("many confident outlier matches: parallax, independent motion, or real change")
    if out["coverage"] < 0.2:
        out["reasons"].append("inliers cover <20% of the image; alignment far from inliers is extrapolated")
    if chosen == "homography" and cfg.domain in ("scene", "nonrigid"):
        out["reasons"].append("global homography on a 3D/non-rigid scene: valid only near the dominant plane")
    return out


def warp_b_to_a(img_b, M_ba, shape_a):
    ha, wa = shape_a[:2]
    M = np.asarray(M_ba, float)
    aligned = cv2.warpPerspective(img_b, M, (wa, ha), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.ones(img_b.shape[:2], np.uint8), M, (wa, ha), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
    return aligned, valid


def local_uncertainty(shape_hw, inliers_a, scale, status, diag_frac=0.15):
    """0 = next to a geometric inlier, 1 = far from any (alignment extrapolated)."""
    h, w = shape_hw
    if status == "failed":
        return np.ones((h, w), np.float32)
    if inliers_a is None or len(inliers_a) == 0:
        return np.full((h, w), 0.5, np.float32)
    canvas = np.full((h, w), 255, np.uint8)
    p = np.clip(np.round(np.asarray(inliers_a) * scale).astype(int), 0, [w - 1, h - 1])
    canvas[p[:, 1], p[:, 0]] = 0
    d = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
    return np.clip(d / (diag_frac * np.hypot(h, w)), 0, 1).astype(np.float32)


def residual_flow(gray_a, gray_b_aligned):
    """DIS optical flow field (HxWx2, A->B) between A and aligned B: misregistration or real motion."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(gray_a, gray_b_aligned, None).astype(np.float32)


def map_box_a_to_b(box_a, align, shape_b):
    M_ab = np.linalg.inv(np.asarray(align["M_ba"], float))
    hb, wb = shape_b[:2]
    return clip_box(transform_box(box_a, M_ab), wb, hb), f"{align['model']}:{align['status']}"


def map_box_b_to_a(box_b, align, shape_a):
    ha, wa = shape_a[:2]
    return clip_box(transform_box(box_b, align["M_ba"]), wa, ha)
