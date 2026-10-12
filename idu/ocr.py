"""Stage E: OCR, text-region correspondence, exact edit operations, re-reading and
OCR-independent visual verification of text crops.

An OCR mismatch is never by itself a verified change: each edit carries (a) OCR confidence,
(b) correspondence quality, (c) re-read stability and (d) a direct pixel comparison verdict.
"""
from __future__ import annotations

import difflib

import cv2
import numpy as np

from .align import map_box_a_to_b, map_box_b_to_a
from .lowlevel import _ssim_np
from .schema import Adapter, box_iou, pad_box


class EasyOCRAdapter(Adapter):
    name = "easyocr"

    def __init__(self, langs, device):
        self.langs, self.device = list(langs), device

    def check_environment(self):
        try:
            import easyocr  # noqa: F401
            return {"ok": True}
        except ImportError as e:
            return {"ok": False, "error": str(e), "fix": "pip install easyocr"}

    def load_model(self, cfg=None):
        import easyocr
        self.reader = easyocr.Reader(self.langs, gpu=self.device == "cuda", verbose=False)
        self.version = f"easyocr-{getattr(easyocr, '__version__', '?')} langs={self.langs}"

    def read(self, img_rgb, mag_ratio=1.0, offset=(0, 0), scale=1.0):
        res = self.reader.readtext(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), detail=1, paragraph=False,
                                   mag_ratio=mag_ratio)
        out = []
        for quad, text, conf in res:
            q = np.asarray(quad, float) / scale + np.asarray(offset, float)
            out.append({"quad": q.tolist(), "bbox": [float(q[:, 0].min()), float(q[:, 1].min()),
                                                     float(q[:, 0].max()), float(q[:, 1].max())],
                        "text": " ".join(str(text).split()), "conf": float(conf)})
        return out

    def predict(self, image_a, image_b, context=None):
        return self.read(image_a), self.read(image_b)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_sim(a, b):
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def char_span_box(box, text, i0, i1):
    """Approximate character-span box by proportional split of the line width.
    ponytail: assumes monospace-ish horizontal text; replace with a char-level recognizer for exact boxes."""
    n = max(len(text), 1)
    x1, y1, x2, y2 = box
    w = (x2 - x1) / n
    return [x1 + i0 * w, y1, x1 + max(i1, i0 + 1) * w, y2]


def _crop(img, box, pad=4):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(c)) for c in pad_box(box, pad, w, h)]
    return img[y1:y2, x1:x2], (x1, y1)


def reread(adapter, img, box, upscale):
    crop, (ox, oy) = _crop(img, box, pad=6)
    if crop.size == 0:
        return None
    up = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    lines = adapter.read(up, offset=(ox, oy), scale=upscale)
    lines.sort(key=lambda l: l["bbox"][0])
    return {"text": " ".join(l["text"] for l in lines), "conf": min((l["conf"] for l in lines), default=0.0)}


def visual_check(img_a, img_b, aligned_b, box_a, box_b, align, edit_spans_a=(), min_char_h=10):
    """Compare text crops directly in pixel space, independent of OCR."""
    ca, _ = _crop(img_a, box_a)
    if align["status"] in ("reliable", "approximate"):
        cb, _ = _crop(aligned_b, box_a)
        how = "aligned_b_same_coords"
    else:
        cb, _ = _crop(img_b, box_b)
        cb = cv2.resize(cb, (ca.shape[1], ca.shape[0]), interpolation=cv2.INTER_AREA) if cb.size else cb
        how = "b_box_resized"
    if ca.size == 0 or cb.size == 0 or ca.shape != cb.shape:
        return {"verdict": "inconclusive", "reason": "empty or mismatched crop"}
    ga = cv2.cvtColor(ca, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(cb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F) if min(ga.shape) > 8 else None
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, win)
    if abs(dx) < ga.shape[1] / 4 and abs(dy) < ga.shape[0] / 4:   # sub-crop residual alignment
        gb = cv2.warpAffine(gb, np.float32([[1, 0, -dx], [0, 1, -dy]]), ga.shape[::-1], borderMode=cv2.BORDER_REPLICATE)

    def norm(g):
        lo, hi = np.percentile(g, [2, 98])
        return np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1)
    na, nb = norm(ga), norm(gb)
    diff = np.abs(na - nb) > 0.35
    out = {"method": how, "residual_shift_px": [float(dx), float(dy)],
           "diff_frac": float(diff.mean()), "ssim_dissim": float(_ssim_np(na, nb).mean()),
           "char_height_px": float(box_a[3] - box_a[1])}
    if edit_spans_a:
        ox, oy = int(max(box_a[0] - 4, 0)), int(max(box_a[1] - 4, 0))
        inside = np.zeros_like(diff)
        for sb in edit_spans_a:
            inside[:, max(int(sb[0]) - ox, 0):max(int(sb[2]) - ox, 0)] = True
        if inside.any() and (~inside).any():
            out["diff_frac_in_edit_span"] = float(diff[inside].mean())
            out["edit_span_ratio"] = float((diff[inside].mean() + 1e-4) / (diff[~inside].mean() + 1e-4))
    span = out.get("diff_frac_in_edit_span", 0.0)
    # a one-character edit is a small fraction of a line: judge it inside the OCR-estimated edit span
    localized = span > 0.04 and out.get("edit_span_ratio", 0) > 3
    if out["char_height_px"] < min_char_h:
        out["verdict"] = "low_resolution"
    elif out["diff_frac"] > 0.015 or localized:
        out["verdict"] = "differs"
    elif out["diff_frac"] < 0.005 and span < 0.02:
        out["verdict"] = "same"
    else:
        out["verdict"] = "inconclusive"
    return out


def match_lines(lines_a, lines_b, align, shape_a):
    from scipy.optimize import linear_sum_assignment
    if not lines_a or not lines_b:
        return [], list(range(len(lines_a))), list(range(len(lines_b)))
    b_in_a = [map_box_b_to_a(l["bbox"], align, shape_a) for l in lines_b]
    C = np.full((len(lines_a), len(lines_b)), 10.0)
    for i, la in enumerate(lines_a):
        for j, bb in enumerate(b_in_a):
            iou = box_iou(la["bbox"], bb)
            h = (la["bbox"][3] - la["bbox"][1] + bb[3] - bb[1]) / 2
            ca = np.array([(la["bbox"][0] + la["bbox"][2]) / 2, (la["bbox"][1] + la["bbox"][3]) / 2])
            cb = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
            if iou >= 0.2 or np.linalg.norm(ca - cb) < 0.6 * h:
                C[i, j] = (1 - iou) + 0.3 * (1 - text_sim(la["text"], lines_b[j]["text"]))
    r, c = linear_sum_assignment(C)
    pairs = [(i, j, float(box_iou(lines_a[i]["bbox"], b_in_a[j]))) for i, j in zip(r, c) if C[i, j] < 10]
    ua = sorted(set(range(len(lines_a))) - {p[0] for p in pairs})
    ub = sorted(set(range(len(lines_b))) - {p[1] for p in pairs})
    return pairs, ua, ub


def run_ocr_branch(adapter, img_a, img_b, aligned_b, align, cfg, do_reread=True):
    lines_a, lines_b = adapter.read(img_a), adapter.read(img_b)
    pairs, ua, ub = match_lines(lines_a, lines_b, align, img_a.shape)
    diffs, unchanged = [], []
    for i, j, iou in pairs:
        la, lb = lines_a[i], lines_b[j]
        if la["text"] == lb["text"]:
            unchanged.append({"a": i, "b": j, "text": la["text"], "iou": iou})
            continue
        ops = [o for o in difflib.SequenceMatcher(None, la["text"], lb["text"], autojunk=False).get_opcodes()
               if o[0] != "equal"]
        edits = [{"op": {"replace": "substitution", "delete": "deletion", "insert": "insertion"}[t],
                  "a_index": [i1, i2], "b_index": [j1, j2], "a_str": la["text"][i1:i2], "b_str": lb["text"][j1:j2],
                  "a_char_box_est": char_span_box(la["bbox"], la["text"], i1, i2),
                  "b_char_box_est": char_span_box(lb["bbox"], lb["text"], j1, j2)} for t, i1, i2, j1, j2 in ops]
        d = {"a": i, "b": j, "iou_in_a_frame": iou, "before": la["text"], "after": lb["text"],
             "conf_a": la["conf"], "conf_b": lb["conf"], "edits": edits,
             "char_box_method": "proportional_estimate", "index_convention": "0-based, end-exclusive"}
        d["visual"] = visual_check(img_a, img_b, aligned_b, la["bbox"], lb["bbox"], align,
                                   [e["a_char_box_est"] for e in edits], cfg.ocr_min_char_height_px)
        ra = reread(adapter, img_a, la["bbox"], cfg.ocr_upscale) if do_reread else None
        rb = reread(adapter, img_b, lb["bbox"], cfg.ocr_upscale) if do_reread else None
        d["reread"] = {"a": ra, "b": rb}
        if not do_reread:
            d["reread_status"] = "not_run"
        elif ra and rb and ra["text"] == rb["text"]:
            d["reread_status"] = "reread_agrees_texts_equal"       # first pass likely misread
        elif ra and rb and ra["text"] == la["text"] and rb["text"] == lb["text"]:
            d["reread_status"] = "stable"
        else:
            d["reread_status"] = "unstable"
        diffs.append(d)

    def presence(idx_list, lines, other_img, other_lines_idx, other_lines, to_other):
        out = []
        for k in idx_list:
            l = lines[k]
            if l["conf"] < cfg.ocr_min_conf * 0.6 or len(l["text"]) == 0:
                continue
            box_other, how = to_other(l["bbox"])
            rr = reread(adapter, other_img, box_other, cfg.ocr_upscale) if do_reread else None
            moved_to = next((m for m in other_lines_idx if text_sim(other_lines[m]["text"], l["text"]) > 0.9), None)
            if rr and rr["text"] and text_sim(rr["text"], l["text"]) > 0.8:
                status = "present_in_other_image(detection_miss)"
            elif moved_to is not None:
                status = "moved"
            else:
                status = "absent_in_other_image"
            out.append({"line": k, "text": l["text"], "conf": l["conf"], "bbox": l["bbox"], "other_bbox": box_other,
                        "other_bbox_method": how, "status": status, "reread_other": rr,
                        "moved_to": None if moved_to is None else other_lines[moved_to]["bbox"]})
        return out

    removed = presence(ua, lines_a, img_b, ub, lines_b, lambda b: map_box_a_to_b(b, align, img_b.shape))
    added = presence(ub, lines_b, img_a, ua, lines_a,
                     lambda b: (map_box_b_to_a(b, align, img_a.shape), f"{align['model']}:{align['status']}"))
    return {"lines_a": lines_a, "lines_b": lines_b, "pairs": pairs, "diffs": diffs, "unchanged": unchanged,
            "removed": removed, "added": added, "engine": adapter.version}
