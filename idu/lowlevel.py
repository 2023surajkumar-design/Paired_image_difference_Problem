"""Stage C: independent low-level evidence maps, candidate regions, nuisance signatures,
and native-resolution re-inspection.

Maps are robust z-scores (median/MAD units with a noise floor). They rank pixels; they are
NOT probabilities and are never reported as such.
"""
from __future__ import annotations

import cv2
import numpy as np

from .align import residual_flow
from .schema import Adapter, pad_box

# Fused maps are displacement-TOLERANT: a difference counts only if no shift within
# `tol` px explains it (checked in both directions, so thin structures that appear or
# disappear survive). SSIM / strict Lab are kept as reported evidence but not fused,
# because they fire on sub-pixel misregistration and resampling blur at every edge.
MAP_KEYS = ("lab", "edge", "texture")
AUX_KEYS = ("ssim", "lab_strict", "rgb", "lum", "lum_raw", "chroma", "unaligned")


# ---------------- DINOv2 (compatibility + texture) ----------------
class DinoAdapter(Adapter):
    name = "dinov2"

    def __init__(self, model_id, device):
        self.model_id, self.device = model_id, device

    def check_environment(self):
        try:
            import transformers  # noqa: F401
            return {"ok": True}
        except ImportError as e:
            return {"ok": False, "error": str(e)}

    def load_model(self, cfg=None):
        import torch
        from transformers import AutoModel
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=dt).eval().to(self.device)
        self.dtype, self.version = dt, self.model_id

    def _tokens(self, img, max_side):
        import torch
        s = max_side / max(img.shape[:2])
        h = max(14, int(round(img.shape[0] * s / 14)) * 14)
        w = max(14, int(round(img.shape[1] * s / 14)) * 14)
        x = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)[None].to(self.device, self.dtype)
        with torch.inference_mode():
            hs = self.model(pixel_values=t).last_hidden_state[0]
        return hs[0], hs[1:].reshape(h // 14, w // 14, -1)

    def similarity(self, a, b):
        import torch
        ca, _ = self._tokens(a, 448)
        cb, _ = self._tokens(b, 448)
        return float(torch.nn.functional.cosine_similarity(ca.float(), cb.float(), dim=0))

    def texture_distance(self, a, b_aligned, max_side):
        import torch
        _, ta = self._tokens(a, max_side)
        _, tb = self._tokens(b_aligned, max_side)
        d = 1 - torch.nn.functional.cosine_similarity(ta.float(), tb.float(), dim=-1)
        return d.clamp(0, 2).cpu().numpy()


# ---------------- map primitives (torch, GPU when available) ----------------
def _t(x, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None].to(device)


def _gauss(x, sigma):
    import torch
    import torch.nn.functional as F
    r = int(round(3 * sigma))
    k = torch.exp(-(torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def ssim_dissim(ya, yb, sigma, device):
    """(1 - SSIM)/2 per pixel on luminance in [0,1]."""
    x, y = _t(ya, device), _t(yb, device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = _gauss(x, sigma), _gauss(y, sigma)
    vx = _gauss(x * x, sigma) - mx ** 2
    vy = _gauss(y * y, sigma) - my ** 2
    cxy = _gauss(x * y, sigma) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return ((1 - s) / 2).clamp(0, 1)[0, 0].cpu().numpy()


def grad_mag(y, device):
    import torch
    import torch.nn.functional as F
    x = F.pad(_t(y, device), (1, 1, 1, 1), mode="reflect")
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8
    gx, gy = F.conv2d(x, kx), F.conv2d(x, kx.transpose(2, 3))
    return torch.sqrt(gx ** 2 + gy ** 2)[0, 0].cpu().numpy()


def _lum(img):
    return (img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255


def _lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)


def photometric_normalize(a, b, valid):
    """Global per-channel robust gain/offset mapping B's intensity distribution onto A's."""
    out, gains = b.astype(np.float32).copy(), []
    for c in range(3):
        va, vb = a[..., c][valid].astype(np.float32), b[..., c][valid].astype(np.float32)
        if va.size < 100:
            gains.append((1.0, 0.0))
            continue
        pa, pb = np.percentile(va[::7], [5, 50, 95]), np.percentile(vb[::7], [5, 50, 95])
        g = float(np.clip((pa[2] - pa[0]) / max(pb[2] - pb[0], 1e-3), 0.5, 2.0))
        o = float(pa[1] - g * pb[1])
        out[..., c] = out[..., c] * g + o
        gains.append((g, o))
    return np.clip(out, 0, 255).astype(np.uint8), gains


def robust_z(m, valid, floor, cond=None, nbins=8):
    """Robust z = (m - median) / max(1.4826*MAD, floor).
    With `cond` (local contrast), median/MAD are estimated per contrast bin: compression and
    resampling errors grow with local contrast, so a residual on a strong edge is judged
    against other strong edges, not against flat areas. Scale is forced non-decreasing in
    contrast (conservative). Changes are sparse, so each bin is dominated by unchanged pixels."""
    vm = valid if valid.any() else np.ones_like(valid)
    v = m[vm][::3]
    med = float(np.median(v))
    scale = max(1.4826 * float(np.median(np.abs(v - med))), floor)
    stats = {"median": med, "scale": scale}
    z = None
    if cond is not None:
        c = cond[vm][::3]
        edges = np.unique(np.quantile(c, np.linspace(0, 1, nbins + 1)))
        if len(edges) >= 3:
            bi = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, len(edges) - 2)
            meds, scs = [], []
            for i in range(len(edges) - 1):
                d = v[bi == i]
                mi = float(np.median(d)) if d.size else med
                meds.append(mi)
                scs.append(max(1.4826 * float(np.median(np.abs(d - mi))), floor) if d.size else scale)
            stats.update(cond_edges=edges.tolist(), cond_median=meds,
                         cond_scale=np.maximum.accumulate(np.array(scs)).tolist())
            z = apply_z(m, stats, cond)
    if z is None:
        z = (m - med) / scale
    z[~valid] = 0
    return z.astype(np.float32), stats


def apply_z(m, stats, cond=None):
    if cond is None or "cond_edges" not in stats:
        return (m - stats["median"]) / stats["scale"]
    e = np.asarray(stats["cond_edges"])
    bi = np.clip(np.searchsorted(e, cond, side="right") - 1, 0, len(e) - 2)
    return (m - np.asarray(stats["cond_median"])[bi]) / np.asarray(stats["cond_scale"])[bi]


def contrast_cond(ga, gb):
    """min of both images' smoothed gradient magnitude: high only where BOTH have an edge
    (misregistration / compression territory), low where structure appears or disappears."""
    return np.minimum(cv2.GaussianBlur(ga, (0, 0), 1.5), cv2.GaussianBlur(gb, (0, 0), 1.5))


def _tolerant(x, y, r):
    """Per-pixel difference not explainable by any displacement within r px (both directions)."""
    if r <= 0:
        return np.abs(x - y)
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    xmax, xmin, ymax, ymin = cv2.dilate(x, k), cv2.erode(x, k), cv2.dilate(y, k), cv2.erode(y, k)
    return np.maximum.reduce([x - ymax, ymin - x, y - xmax, xmin - y, np.zeros_like(x)])


def _lab_chroma_lowpass(img, sigma=1.2):
    """Lab with a/b low-passed: compares colour at chroma resolution (JPEG 4:2:0 subsampling)."""
    lab = _lab(img)
    lab[..., 1] = cv2.GaussianBlur(lab[..., 1], (0, 0), sigma)
    lab[..., 2] = cv2.GaussianBlur(lab[..., 2], (0, 0), sigma)
    return lab


def lab_tolerant(a, b, r):
    la, lb = _lab_chroma_lowpass(a), _lab_chroma_lowpass(b)
    return np.sqrt(sum(_tolerant(np.ascontiguousarray(la[..., c]), np.ascontiguousarray(lb[..., c]), r) ** 2
                       for c in range(3)))


def edge_tolerant(ga, gb, r):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return np.maximum.reduce([ga - cv2.dilate(gb, k), gb - cv2.dilate(ga, k), np.zeros_like(ga)])


def compute_maps(a, b, valid, device, cfg, dino=None, b_naive=None, tol=1):
    """All inputs at the same resolution in A's frame. Returns raw maps, z maps, stats, fused."""
    raw = {}
    la, lb = _lab(a), _lab(b)
    raw["lab"] = lab_tolerant(a, b, tol)
    raw["lab_strict"] = np.linalg.norm(la - lb, axis=2)
    raw["chroma"] = np.linalg.norm(la[..., 1:] - lb[..., 1:], axis=2)
    ya, yb = _lum(a), _lum(b)
    raw["lum"] = np.abs(ya - yb)
    raw["rgb"] = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(2) / 255
    raw["ssim"] = ssim_dissim(ya, yb, cfg.ssim_sigma, device)
    gya, gyb = grad_mag(ya, device), grad_mag(yb, device)
    raw["edge"] = edge_tolerant(gya, gyb, tol)
    cond = contrast_cond(gya, gyb)
    if dino is not None:
        t = dino.texture_distance(a, b, cfg.texture_max_side)
        raw["texture"] = cv2.resize(t, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    if b_naive is not None:
        raw["unaligned"] = np.linalg.norm(la - _lab(b_naive), axis=2)
    floors = dict(cfg.noise_floor, chroma=cfg.noise_floor["lab"], unaligned=cfg.noise_floor["lab"],
                  lab_strict=cfg.noise_floor["lab"], lum_raw=cfg.noise_floor["lum"])
    z, stats = {}, {}
    for k, m in raw.items():
        vmask = valid if k != "unaligned" else np.ones_like(valid)
        z[k], stats[k] = robust_z(m, vmask, floors[k], cond if k in ("lab", "edge") else None)
    fused = np.max(np.stack([z[k] for k in MAP_KEYS if k in z]), axis=0)
    return raw, z, stats, fused, cond


def hysteresis_regions(fused, valid, z_hi, z_lo, min_area, close_px):
    weak = ((fused >= z_lo) & valid).astype(np.uint8)
    if close_px > 1:
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    strong = (fused >= z_hi) & valid
    n, lab, st, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = set(np.unique(lab[strong]).tolist()) - {0}
    regs = []
    for i in keep:
        x, y, w, h, area = st[i]
        if area >= min_area:
            regs.append({"bbox": [float(x), float(y), float(x + w), float(y + h)], "area": int(area),
                         "mask": lab[y:y + h, x:x + w] == i})
    return regs


def _region_mask(shape, regs):
    m = np.zeros(shape, bool)
    for r in regs:
        x1, y1, x2, y2 = map(int, r["bbox"])
        m[y1:y2, x1:x2] |= r["mask"]
    return m


def _region_nuisance(st, cfg, q_a, q_b):
    flags = []
    zl = cfg.z_low
    if ((st["structure_corr"] > 0.9 and st["chroma_z_mean"] < zl and st["edge_z_mean"] < zl
         and st["lum_raw_z_mean"] >= zl)
            or (st.get("lab_z_mean", 0) >= zl and st["illum_explained_ratio"] < 0.35 and st["structure_corr"] > 0.8)):
        flags.append("illumination_or_shadow")
    if st["clip_frac"] > 0.3:
        flags.append("exposure_clipping")
    if st["flow_mean"] > 1.0 and st["flow_explained_ratio"] < 0.3:
        flags.append("local_displacement_explains_residual(misregistration or moved content)")
    if st["uncert_mean"] > 0.5:
        flags.append("far_from_alignment_inliers")
    if st["saturated_frac"] > 0.3:
        flags.append("specular_or_saturation")
    if (st["blur_ratio"] > 4 or st["blur_ratio"] < 0.25) and st["chroma_z_mean"] < zl:
        flags.append("focus_blur_difference")
    if (max(q_a["jpeg_blockiness"], q_b["jpeg_blockiness"]) > 1.3 and st["peak_z"] < 3 * cfg.z_high
            and st["area_px"] < 2000):
        flags.append("compression_artifact")
    if st["valid_frac"] < 0.8:
        flags.append("near_invalid_overlap")
    return flags


def analyze(img_a, aligned_b, valid, b_naive, uncert_full_fn, align, cfg, device, quality, dino=None):
    """Run the low-level branch at analysis resolution. Returns dict with maps + regions."""
    H, W = img_a.shape[:2]
    s = min(1.0, cfg.analysis_max_side / max(H, W))

    def rs(x, interp=cv2.INTER_AREA):
        return cv2.resize(x, (int(W * s), int(H * s)), interpolation=interp) if s < 1 else x

    a = rs(img_a)
    v = rs(valid.astype(np.uint8), cv2.INTER_NEAREST)
    # filters below (tolerance window, chroma low-pass) must not read the black warp border
    guard = 2 * (cfg.tolerance_px + 1) + 5
    v = cv2.erode(v, np.ones((guard, guard), np.uint8)) > 0
    b_raw = rs(aligned_b)
    if cfg.photometric_normalize:
        b, gains = photometric_normalize(a, b_raw, v)
    else:
        b, gains = b_raw, [(1.0, 0.0)] * 3
    bn = None
    if b_naive is not None and align.get("model") != "identity":
        bn = rs(cv2.resize(b_naive, (W, H), interpolation=cv2.INTER_AREA))
        bn, _ = photometric_normalize(a, bn, np.ones_like(v))

    # resampling (any non-identity warp) adds interpolation blur + sub-pixel error: widen tolerance by 1 px
    tol = cfg.tolerance_px + (0 if align.get("model") == "identity" else 1)
    raw, z, stats, fused, cond = compute_maps(a, b, v, device, cfg, dino=dino, b_naive=bn, tol=tol)
    z["lum_raw"], stats["lum_raw"] = robust_z(np.abs(_lum(a) - _lum(b_raw)), v, cfg.noise_floor["lum"])

    diag = np.hypot(*a.shape[:2])
    min_area = max(2, int(cfg.min_region_px * s * s))
    close = max(3, int(0.004 * diag)) | 1
    regs = hysteresis_regions(fused, v, cfg.z_high, cfg.z_low, min_area, close)
    # coarse scale: catches diffuse changes whose per-pixel z is diluted at full res
    a2, b2, v2 = (cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for x in (a, b, v.astype(np.uint8)))
    f2 = compute_maps(a2, b2, v2 > 0, device, cfg, tol=1)[3]
    f2 = cv2.resize(f2, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    for r in hysteresis_regions(f2, v, cfg.z_high, cfg.z_low, min_area * 4, close):
        r["scale"] = "coarse"
        regs.append(r)
    # merge overlapping fine/coarse components into one mask and re-label
    union = _region_mask(fused.shape, regs).astype(np.uint8)
    n, lab, st_cc, _ = cv2.connectedComponentsWithStats(union, connectivity=8)

    uncert = uncert_full_fn(a.shape[:2], s)
    # nuisance signature: does dense flow (local displacement) remove the residual?
    flow, z_flow = None, None
    if cfg.use_flow and align.get("status") != "failed":
        fl = residual_flow(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY))
        flow = cv2.GaussianBlur(np.linalg.norm(fl, axis=2), (0, 0), 2)
        gx, gy = np.meshgrid(np.arange(a.shape[1], dtype=np.float32), np.arange(a.shape[0], dtype=np.float32))
        b_fc = cv2.remap(b, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        z_flow = apply_z(lab_tolerant(a, b_fc, tol), stats["lab"], cond)
    # nuisance signature: does a smooth LOCAL luminance gain (shadow / light falloff) explain the residual?
    sig = max(4.0, 0.03 * diag)
    gain = np.clip((cv2.GaussianBlur(_lum(a), (0, 0), sig) + 0.02) /
                   (cv2.GaussianBlur(_lum(b_raw), (0, 0), sig) + 0.02), 0.4, 2.5)
    b_loc = np.clip(b_raw.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    z_loc = apply_z(lab_tolerant(a, b_loc, tol), stats["lab"], cond)
    clip_xor = (a.max(2) >= 250) ^ (b_raw.max(2) >= 250)
    hsv_a, hsv_b = cv2.cvtColor(a, cv2.COLOR_RGB2HSV), cv2.cvtColor(b_raw, cv2.COLOR_RGB2HSV)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

    regions = []
    for i in range(1, n):
        x, y, w, h, area = st_cc[i]
        if area < min_area:
            continue
        m = lab[y:y + h, x:x + w] == i
        sl = (slice(y, y + h), slice(x, x + w))
        st = {"area_px": int(area / (s * s)), "peak_z": float(fused[sl][m].max()),
              "mean_z": float(fused[sl][m].mean())}
        for k in MAP_KEYS + ("ssim", "lab_strict", "rgb", "lum", "lum_raw", "chroma"):
            if k in z:
                st[f"{k}_z_mean"] = float(z[k][sl][m].mean())
                st[f"{k}_z_peak"] = float(z[k][sl][m].max())
        st.setdefault("texture_z_mean", 0.0)
        bx = [int(c) for c in pad_box([x, y, x + w, y + h], 3, a.shape[1], a.shape[0])]
        pa = ga[bx[1]:bx[3], bx[0]:bx[2]].astype(np.float32).ravel()
        pb = gb[bx[1]:bx[3], bx[0]:bx[2]].astype(np.float32).ravel()
        st["structure_corr"] = float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 1 and pb.std() > 1 else 0.0
        st["illum_explained_ratio"] = float(max(z_loc[sl][m].mean(), 0) / max(st.get("lab_z_mean", 0), 1e-3))
        st["clip_frac"] = float(clip_xor[sl][m].mean())
        st["flow_mean"] = float(flow[sl][m].mean()) if flow is not None else 0.0
        st["flow_explained_ratio"] = (float(max(z_flow[sl][m].mean(), 0) / max(st.get("lab_z_mean", 0), 1e-3))
                                      if z_flow is not None else 1.0)
        st["uncert_mean"] = float(uncert[sl][m].mean())
        st["valid_frac"] = float(v[sl].mean())
        sat = lambda hsv: (hsv[..., 2][sl][m] > 247) & (hsv[..., 1][sl][m] < 25)
        st["saturated_frac"] = float(np.maximum(sat(hsv_a).mean(), sat(hsv_b).mean()))
        la_ = cv2.Laplacian(ga[sl], cv2.CV_32F).var() + 1
        lb_ = cv2.Laplacian(gb[sl], cv2.CV_32F).var() + 1
        st["blur_ratio"] = float(la_ / lb_)
        dominant = max((k for k in MAP_KEYS if k in z), key=lambda k: st.get(f"{k}_z_mean", -1))
        regions.append({
            "bbox_a": [x / s, y / s, (x + w) / s, (y + h) / s], "stats": st, "dominant_map": dominant,
            "nuisance": _region_nuisance(st, cfg, quality["a"], quality["b"]),
            "mask_analysis": m, "mask_origin_analysis": [int(x), int(y)],
        })
    regions.sort(key=lambda r: -r["stats"]["peak_z"] * np.sqrt(r["stats"]["area_px"]))
    dropped = max(0, len(regions) - 60)
    regions = regions[:60]  # ponytail: hard cap; raise if dense-change scenes need more

    # regions that differ only before alignment: displacement explained by registration
    unaligned_only = []
    if "unaligned" in z:
        um = ((z["unaligned"] >= cfg.z_high) & ~(fused >= cfg.z_low) & v).astype(np.uint8)
        um = cv2.morphologyEx(um, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        frac = float(um.mean())
        unaligned_only.append({"fraction_of_image": frac,
                               "interpretation": "residual present before alignment but not after: "
                                                 "consistent with camera/viewpoint displacement (plausible, not proven)"})

    full_area = float(sum(r["mask_analysis"].sum() for r in regions) / fused.size)
    return {
        "scale": s, "analysis_shape": list(a.shape[:2]), "photometric_gains": gains,
        "map_stats": stats, "regions": regions, "regions_dropped": dropped,
        "changed_area_fraction": full_area, "unaligned_only": unaligned_only,
        "maps": {"fused": fused, **{f"z_{k}": z[k] for k in z}, **({"flow": flow} if flow is not None else {}),
                 "uncertainty": uncert, "valid": v},
        "params": {"z_high": cfg.z_high, "z_low": cfg.z_low, "displacement_tolerance_px": tol, "min_area_analysis_px": min_area,
                   "closing_px": close, "noise_floor": cfg.noise_floor, "ssim_sigma": cfg.ssim_sigma,
                   "fused_maps": [k for k in MAP_KEYS if k in z], "coordinate_frame": "image A (analysis px = orig*scale)",
                   "photometric_normalize": cfg.photometric_normalize},
        "_b_norm_analysis": b, "_a_analysis": a,
    }


# ---------------- native-resolution re-inspection ----------------
def residual_blockiness(d, x0, y0, block=8):
    """Gradient of the residual on the absolute 8-px JPEG grid vs elsewhere. Compression drift is
    piecewise-constant per block (ratio >> 1); a genuine change is smooth inside (ratio ~ 1)."""
    out = 1.0
    for axis, off in ((1, x0), (0, y0)):
        g = np.abs(np.diff(d, axis=axis))
        if g.shape[axis] < 3 * block:
            continue
        pos = (np.arange(g.shape[axis]) + off) % block == block - 1
        on = g[:, pos].mean() if axis == 1 else g[pos, :].mean()
        off_ = g[:, ~pos].mean() if axis == 1 else g[~pos, :].mean()
        out = max(out, float((on + 1e-3) / (off_ + 1e-3)))
    return out


def add_native_flags(region, q_a, q_b):
    """Nuisance flags that need the native-resolution re-inspection."""
    nat = region.get("native") or {}
    if (nat.get("residual_blockiness", 1.0) > 1.5 and max(q_a["jpeg_blockiness"], q_b["jpeg_blockiness"]) > 1.2
            and "compression_artifact" not in region["nuisance"]):
        region["nuisance"].append("compression_artifact")

def _ssim_np(x, y, sigma=1.5):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    g = lambda t: cv2.GaussianBlur(t, (0, 0), sigma)
    mx, my = g(x), g(y)
    vx, vy, cxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return np.clip((1 - s) / 2, 0, 1)


def native_reinspect(img_a, img_b, M_ba, box_a, low, cfg, pad=24):
    """Re-examine a candidate at native resolution in A's frame; refine its box to changed pixels."""
    H, W = img_a.shape[:2]
    x1, y1, x2, y2 = [int(round(c)) for c in pad_box(box_a, pad, W, H)]
    if x2 - x1 < 4 or y2 - y1 < 4:
        return {"confirmed": False, "reason": "degenerate crop"}
    T = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], float)
    Mc = T @ np.asarray(M_ba, float)
    ca = img_a[y1:y2, x1:x2]
    cb = cv2.warpPerspective(img_b, Mc, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR)
    vc = cv2.warpPerspective(np.ones(img_b.shape[:2], np.uint8), Mc, (x2 - x1, y2 - y1), flags=cv2.INTER_NEAREST) > 0
    cbf = cb.astype(np.float32)
    for c, (g, o) in enumerate(low["photometric_gains"]):
        cbf[..., c] = cbf[..., c] * g + o
    cb = np.clip(cbf, 0, 255).astype(np.uint8)
    st, tol = low["map_stats"], low["params"]["displacement_tolerance_px"]
    gm = lambda im: cv2.magnitude(cv2.Sobel(_lum(im), cv2.CV_32F, 1, 0, ksize=3),
                                  cv2.Sobel(_lum(im), cv2.CV_32F, 0, 1, ksize=3)) / 8
    ga_, gb_ = gm(ca), gm(cb)
    cond = contrast_cond(ga_, gb_)
    zl = apply_z(lab_tolerant(ca, cb, tol), st["lab"], cond)
    ze = apply_z(edge_tolerant(ga_, gb_, tol), st["edge"], cond)
    zf = np.where(vc, np.maximum(zl, ze), 0)
    vc = cv2.erode(vc.astype(np.uint8), np.ones((2 * tol + 5, 2 * tol + 5), np.uint8)) > 0
    zf = np.where(vc, zf, 0)
    rb = residual_blockiness(np.linalg.norm(_lab(ca) - _lab(cb), axis=2), x1, y1)
    strong = zf >= cfg.z_high
    if not strong.any():
        return {"confirmed": False, "peak_z_native": float(zf.max()), "crop_box": [x1, y1, x2, y2],
                "residual_blockiness": rb}
    weak = (zf >= cfg.z_low).astype(np.uint8)
    n, lab, stc, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    ids = set(np.unique(lab[strong]).tolist()) - {0}
    mask = np.isin(lab, list(ids))
    ys, xs = np.nonzero(mask)
    return {"confirmed": True, "peak_z_native": float(zf.max()), "residual_blockiness": rb,
            "refined_box": [float(xs.min() + x1), float(ys.min() + y1), float(xs.max() + 1 + x1), float(ys.max() + 1 + y1)],
            "changed_px_native": int(mask.sum()), "crop_box": [x1, y1, x2, y2], "mask_crop": mask}
