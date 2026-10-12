"""Stage A: robust loading, image-quality measurements and pair validation."""
from __future__ import annotations

import hashlib
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # we enforce our own limit (cfg.max_pixels)


def load_image(src, max_pixels=40_000_000):
    """src: path, bytes, PIL image or HxWx3 uint8 array. Returns (rgb uint8, metadata)."""
    if isinstance(src, np.ndarray):
        arr = src if src.ndim == 3 else np.stack([src] * 3, -1)
        arr = np.ascontiguousarray(arr[..., :3].astype(np.uint8))
        return arr, {"source": "ndarray", "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                     "width": arr.shape[1], "height": arr.shape[0], "exif_transposed": False}
    if isinstance(src, Image.Image):
        im, raw = src, None
    else:
        raw = src if isinstance(src, (bytes, bytearray)) else open(src, "rb").read()
        im = Image.open(io.BytesIO(raw))
    meta = {"source": src if isinstance(src, str) else type(src).__name__,
            "format": im.format, "mode": im.mode, "raw_size": list(im.size),
            "icc_profile": bool(im.info.get("icc_profile")),
            "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None}
    if im.size[0] * im.size[1] > max_pixels:
        raise ValueError(f"image has {im.size[0] * im.size[1]} px > max_pixels={max_pixels}")
    orient = im.getexif().get(0x0112, 1) if hasattr(im, "getexif") else 1
    im = ImageOps.exif_transpose(im)
    meta["exif_orientation"], meta["exif_transposed"] = int(orient), orient not in (None, 1)
    meta["has_alpha"] = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    if meta["has_alpha"]:  # composite on white so transparent pixels compare deterministically
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    meta["width"], meta["height"] = arr.shape[1], arr.shape[0]
    return arr, meta


def _gray_small(img, side=1024):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g


def blockiness(gray) -> float:
    """Ratio of mean |gradient| on 8-px block boundaries to elsewhere (~1 = no JPEG blocking)."""
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    if dx.shape[1] < 16:
        return 1.0
    cols = np.arange(dx.shape[1])
    on = dx[:, cols % 8 == 7].mean()
    off = dx[:, cols % 8 != 7].mean()
    return float(on / (off + 1e-6))


def image_quality(img) -> dict:
    g = _gray_small(img)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    q = {
        "width": img.shape[1], "height": img.shape[0],
        "megapixels": round(img.shape[0] * img.shape[1] / 1e6, 3),
        "sharpness_laplacian_var": float(lap.var()),
        "rms_contrast": float(g.std() / 255.0),
        "clipped_dark_frac": float((g <= 2).mean()),
        "clipped_bright_frac": float((g >= 253).mean()),
        "jpeg_blockiness": blockiness(g),
    }
    w = []
    if q["sharpness_laplacian_var"] < 30:
        w.append("possible_blur")
    if q["rms_contrast"] < 0.05:
        w.append("low_contrast")
    if q["clipped_bright_frac"] > 0.2 or q["clipped_dark_frac"] > 0.2:
        w.append("heavy_clipping")
    if q["jpeg_blockiness"] > 1.3:
        w.append("visible_compression_blocking")
    if min(img.shape[:2]) < 256:
        w.append("low_resolution")
    q["warnings"] = w
    return q


def validate_pair(img_a, img_b, meta_a, meta_b, cfg, compat=None) -> dict:
    """compat: optional callable (img_a, img_b) -> cosine similarity of global embeddings."""
    qa, qb = image_quality(img_a), image_quality(img_b)
    rec = {"image_a": {"meta": meta_a, "quality": qa}, "image_b": {"meta": meta_b, "quality": qb},
           "warnings": [], "status": "comparable"}
    for tag, q in (("A", qa), ("B", qb)):
        rec["warnings"] += [f"{tag}:{w}" for w in q["warnings"]]
        if min(q["width"], q["height"]) < cfg.min_side_px:
            rec["status"] = "not_comparable"
            rec["warnings"].append(f"{tag}:too_small(<{cfg.min_side_px}px)")
    ar_a, ar_b = qa["width"] / qa["height"], qb["width"] / qb["height"]
    rec["aspect_ratio_a"], rec["aspect_ratio_b"] = round(ar_a, 4), round(ar_b, 4)
    if abs(np.log(ar_a / ar_b)) > 0.15:
        rec["warnings"].append("aspect_ratio_differs(crop/viewpoint change or different capture)")
    if meta_a.get("sha256") and meta_a.get("sha256") == meta_b.get("sha256"):
        rec["warnings"].append("byte_identical_files")
    rec["identical_pixels"] = img_a.shape == img_b.shape and bool(np.array_equal(img_a, img_b))
    sharp = (qa["sharpness_laplacian_var"] + 1) / (qb["sharpness_laplacian_var"] + 1)
    rec["sharpness_ratio_a_over_b"] = round(float(sharp), 3)
    if sharp > 3 or sharp < 1 / 3:
        rec["warnings"].append("focus_or_blur_differs_between_images")
    rec["compat_similarity"] = None
    if compat is not None:
        s = float(compat(img_a, img_b))
        rec["compat_similarity"] = round(s, 4)
        rec["compat_note"] = ("coarse DINOv2 CLS cosine; high does not prove same instance, "
                              "low does not invalidate a comparative task")
        if s < 0.3:
            rec["warnings"].append("low_global_similarity(may depict different subjects)")
    if rec["status"] == "comparable" and rec["warnings"]:
        rec["status"] = "comparable_with_warnings"
    return rec
