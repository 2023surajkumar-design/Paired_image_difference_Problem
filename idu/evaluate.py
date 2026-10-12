"""Evaluation utilities: controlled synthetic pairs with exact ground truth, detection /
localization / text / reliability / efficiency metrics, ablations, calibration data.

Synthetic pairs are for smoke tests, regression and calibration plumbing. They are NOT a
substitute for an annotated benchmark from your target domain; never report research claims
from them alone.

Dataset format (pairs.jsonl, one JSON per line):
  {"id", "category", "image_a", "image_b", "nuisance_only": bool,
   "changes": [{"bbox_a": [x1,y1,x2,y2], "type": "...", "before": str|null, "after": str|null}]}
"""
from __future__ import annotations

import io
import json
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .calibration import ece, risk_coverage
from .ocr import levenshtein
from .schema import box_iou

# ---------------- synthetic scenes ----------------


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_scene(rng, w=960, h=720):
    """Deterministic list of draw ops over a textured background (feature-rich for matching)."""
    bg = rng.integers(0, 255, (h // 16 + 1, w // 16 + 1, 3)).astype(np.uint8)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = np.clip(bg.astype(np.int16) + rng.integers(-12, 12, (h, w, 3)), 0, 255).astype(np.uint8)
    ops = []
    for _ in range(14):
        x, y = int(rng.integers(0, w - 120)), int(rng.integers(0, h - 120))
        s = int(rng.integers(30, 110))
        ops.append({"kind": rng.choice(["rect", "ellipse"]), "box": [x, y, x + s, y + int(s * rng.uniform(0.6, 1.4))],
                    "color": tuple(int(c) for c in rng.integers(0, 255, 3))})
    for _ in range(10):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        ops.append({"kind": "checker", "box": [x, y, x + 48, y + 48], "color": (20, 20, 20)})
    lx, ly = int(w * 0.55), int(h * 0.08)
    ops.append({"kind": "rect", "box": [lx, ly, lx + 380, ly + 190], "color": (245, 245, 240), "label": True})
    texts = [f"LOT AB-{rng.integers(1000, 9999)}", f"EXP 20{rng.integers(26, 31)}-{rng.integers(1, 12):02d}",
             f"NET {rng.integers(100, 900)} ml"]
    for i, t in enumerate(texts):
        ops.append({"kind": "text", "xy": [lx + 18, ly + 18 + i * 56], "text": t, "size": 38, "color": (15, 15, 15)})
    return {"bg": bg, "ops": ops}


def _text_box(op):
    f = _font(op["size"])
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    b = d.textbbox(tuple(op["xy"]), op["text"], font=f)
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]


def render(scene):
    im = Image.fromarray(scene["bg"].copy())
    d = ImageDraw.Draw(im)
    for op in scene["ops"]:
        if op["kind"] == "rect":
            d.rectangle(op["box"], fill=op["color"])
        elif op["kind"] == "ellipse":
            d.ellipse(op["box"], fill=op["color"])
        elif op["kind"] == "checker":
            x1, y1, x2, y2 = op["box"]
            for i in range(0, x2 - x1, 8):
                for j in range(0, y2 - y1, 8):
                    if (i // 8 + j // 8) % 2 == 0:
                        d.rectangle([x1 + i, y1 + j, x1 + i + 7, y1 + j + 7], fill=op["color"])
        elif op["kind"] == "text":
            d.text(tuple(op["xy"]), op["text"], font=_font(op["size"]), fill=op["color"])
    return np.asarray(im)


def _copy(scene):
    return {"bg": scene["bg"], "ops": [dict(o) for o in scene["ops"]]}


def _jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=q)
    return np.asarray(Image.open(buf).convert("RGB"))


def _shift(img, rng, max_t=14, max_r=1.5):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-max_r, max_r)), 1.0)
    M[:, 2] += rng.uniform(-max_t, max_t, 2)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def make_pair(category, seed):
    rng = np.random.default_rng(seed)
    sc = make_scene(rng)
    a = render(sc)
    b_sc, gts = _copy(sc), []
    shapes = [i for i, o in enumerate(sc["ops"]) if o["kind"] in ("rect", "ellipse") and not o.get("label")]
    texts = [i for i, o in enumerate(sc["ops"]) if o["kind"] == "text"]
    cats = category.split("+")
    for cat in cats:
        if cat == "text_edit":
            i = int(rng.choice(texts))
            t = b_sc["ops"][i]["text"]
            pos = int(rng.choice([k for k, ch in enumerate(t) if ch.isdigit()]))
            new = str((int(t[pos]) + int(rng.integers(1, 9))) % 10)
            b_sc["ops"][i] = dict(b_sc["ops"][i], text=t[:pos] + new + t[pos + 1:])
            gts.append({"bbox_a": _text_box(sc["ops"][i]), "type": "text_substitution", "before": t,
                        "after": b_sc["ops"][i]["text"]})
        elif cat == "color_change":
            i = int(rng.choice(shapes))
            old = np.array(b_sc["ops"][i]["color"])
            b_sc["ops"][i] = dict(b_sc["ops"][i], color=tuple(int(c) for c in (old + 128) % 256))
            gts.append({"bbox_a": [float(v) for v in sc["ops"][i]["box"]], "type": "color_change", "before": None, "after": None})
        elif cat == "remove_object":
            i = int(rng.choice(shapes))
            b_sc["ops"][i] = dict(b_sc["ops"][i], kind="none")
            gts.append({"bbox_a": [float(v) for v in sc["ops"][i]["box"]], "type": "removed", "before": None, "after": None})
        elif cat == "add_object":
            x, y = int(rng.integers(40, 500)), int(rng.integers(300, 600))
            box = [x, y, x + 70, y + 55]
            b_sc["ops"].append({"kind": "ellipse", "box": box, "color": (230, 30, 30)})
            gts.append({"bbox_a": [float(v) for v in box], "type": "added", "before": None, "after": None})
    b = render(b_sc)
    if "scratch" in cats:
        x, y = int(rng.integers(60, 860)), int(rng.integers(60, 660))
        dx, dy = int(rng.integers(25, 70)), int(rng.integers(-15, 15))
        b = b.copy()
        cv2.line(b, (x, y), (x + dx, y + dy), (35, 35, 35), 2, cv2.LINE_AA)
        gts.append({"bbox_a": [float(x - 2), float(min(y, y + dy) - 2), float(x + dx + 2), float(max(y, y + dy) + 2)],
                    "type": "surface_defect", "before": None, "after": None})
    if "lighting" in cats:
        h, w = b.shape[:2]
        gx = np.linspace(0.78, 1.2, w)[None, :, None] * np.linspace(1.08, 0.92, h)[:, None, None]
        b = np.clip(b.astype(np.float32) * gx, 0, 255).astype(np.uint8)
    if "jpeg" in cats:
        b = _jpeg(b, 30)
    if "shift" in cats:
        b = _shift(b, rng)
    if "noise" in cats:
        b = np.clip(b.astype(np.int16) + rng.normal(0, 2, b.shape).astype(np.int16), 0, 255).astype(np.uint8)
    # crop a margin from both so a simulated camera move never exposes synthetic (reflected) border
    m = 32
    a, b = a[m:-m, m:-m].copy(), b[m:-m, m:-m].copy()
    h, w = a.shape[:2]
    for g in gts:
        x1, y1, x2, y2 = g["bbox_a"]
        g["bbox_a"] = [max(0.0, x1 - m), max(0.0, y1 - m), min(float(w), x2 - m), min(float(h), y2 - m)]
    return a, b, gts


DEFAULT_CATEGORIES = ["text_edit", "color_change", "add_object", "remove_object", "scratch",
                      "noise", "lighting", "jpeg", "shift",
                      "text_edit+shift", "color_change+add_object", "scratch+lighting"]
NUISANCE_ONLY = {"noise", "lighting", "jpeg", "shift"}


def make_synthetic_suite(out_dir, n_per_category=3, categories=DEFAULT_CATEGORIES, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ci, cat in enumerate(categories):
        for k in range(n_per_category):
            s = seed * 100_000 + ci * 1000 + k
            a, b, gts = make_pair(cat, s)
            pid = f"{cat.replace('+', '_')}_{k}"
            pa, pb = os.path.join(out_dir, f"{pid}_A.png"), os.path.join(out_dir, f"{pid}_B.png")
            Image.fromarray(a).save(pa)
            Image.fromarray(b).save(pb)
            rows.append({"id": pid, "category": cat, "image_a": pa, "image_b": pb,
                         "nuisance_only": cat in NUISANCE_ONLY, "changes": gts})
    path = os.path.join(out_dir, "pairs.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def load_pairs(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------- metrics ----------------
def _center_in(b, box):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def match(preds, gts, iou_thr=0.1):
    """Greedy one-to-one matching. A pair matches if IoU >= thr or either centre lies in the other box
    (tiny changes make IoU unstable). Returns [(pi, gi, iou)]."""
    cand = []
    for pi, p in enumerate(preds):
        if p.get("bbox_a") is None:
            continue
        for gi, g in enumerate(gts):
            iou = box_iou(p["bbox_a"], g["bbox_a"])
            if iou >= iou_thr or _center_in(p["bbox_a"], g["bbox_a"]) or _center_in(g["bbox_a"], p["bbox_a"]):
                cand.append((iou, pi, gi))
    used_p, used_g, out = set(), set(), []
    for iou, pi, gi in sorted(cand, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi), used_g.add(gi)
            out.append((pi, gi, iou))
    return out


COARSE = {"text": "text", "color": "color", "added": "added", "removed": "removed", "defect": "defect",
          "structure": "structure"}


def coarse_type(t):
    t = (t or "").lower()
    for k, v in COARSE.items():
        if k in t:
            return v
    return "other"


def evaluate(pipe, pairs, include_review=False, iou_thr=0.1, progress=True):
    rows, cal, rc_scores, rc_correct = [], [], [], []
    for n, p in enumerate(pairs):
        t0 = time.time()
        try:
            res, _ = pipe.run(p["image_a"], p["image_b"], do_export=False)
        except Exception as e:
            rows.append({"id": p["id"], "category": p["category"], "error": repr(e)})
            continue
        ok_dec = {"accepted", "needs_review"} if include_review else {"accepted"}
        claims = res["claims"]
        preds = [c for c in claims if c["decision"] in ok_dec]
        m = match(preds, p["changes"], iou_thr)
        matched_p = {pi for pi, _, _ in m}
        all_m = {pi for pi, _, _ in match(claims, p["changes"], iou_thr)}
        for i, c in enumerate(claims):
            if c.get("raw_support_score") is not None:
                cal.append({"score": c["raw_support_score"], "correct": int(i in all_m), "decision": c["decision"]})
        for i, c in enumerate(c for c in claims if c["decision"] in ("accepted", "needs_review")):
            rc_scores.append(c.get("raw_support_score") or 0)
            rc_correct.append(int(any(box_iou(c["bbox_a"], g["bbox_a"]) >= iou_thr or _center_in(c["bbox_a"], g["bbox_a"])
                                      for g in p["changes"])) if c.get("bbox_a") else 0)
        cer, exact, types, text_gt, typed = [], [], [], 0, 0
        for pi, gi, _ in m:
            g, c = p["changes"][gi], preds[pi]
            if c["change_type"] not in ("unexplained_visual_change", "semantic_hypothesis_only"):
                typed += 1     # only claims that assert a type are scored for type accuracy
                types.append(coarse_type(c["change_type"]) == coarse_type(g["type"]))
            if g.get("after") and g["type"].startswith("text"):
                text_gt += 1
                if c.get("after"):  # only claims that assert a reading are scored for CER
                    cer.append(levenshtein(c["after"], g["after"]) / max(len(g["after"]), 1))
                    exact.append(c["after"] == g["after"])
        rows.append({
            "id": p["id"], "category": p["category"], "nuisance_only": p["nuisance_only"],
            "tp": len(m), "fp": len(preds) - len(m), "fn": len(p["changes"]) - len(m),
            "mean_iou": float(np.mean([x[2] for x in m])) if m else None,
            "cer": cer, "exact": exact, "type_correct": types, "text_gt_matched": text_gt, "typed": typed,
            "n_review": sum(c["decision"] == "needs_review" for c in claims),
            "n_abstained": sum(c["decision"] == "abstained" for c in claims),
            "false_change": p["nuisance_only"] and any(c["decision"] == "accepted" for c in claims),
            "comparison_status": res["comparison_status"], "latency_s": time.time() - t0,
            "peak_gpu_gb": res["efficiency"]["gpu"].get("peak_gb"), "model_calls": res["efficiency"]["model_calls"],
        })
        if progress:
            print(f"[{n + 1}/{len(pairs)}] {p['id']}: tp={rows[-1]['tp']} fp={rows[-1]['fp']} fn={rows[-1]['fn']}")
    return {"aggregate": aggregate(rows), "per_category": per_category(rows), "per_pair": rows,
            "calibration_samples": cal, "risk_coverage": risk_coverage(rc_scores, rc_correct),
            "include_review": include_review, "iou_thr": iou_thr, "n_pairs": len(pairs)}


def aggregate(rows):
    ok = [r for r in rows if "error" not in r]
    tp, fp, fn = (sum(r[k] for r in ok) for k in ("tp", "fp", "fn"))
    P = tp / (tp + fp) if tp + fp else None
    R = tp / (tp + fn) if tp + fn else None
    nuis = [r for r in ok if r["nuisance_only"]]
    flat = lambda k: [x for r in ok for x in r[k]]
    return {
        "n": len(ok), "errors": len(rows) - len(ok), "tp": tp, "fp": fp, "fn": fn,
        "precision": P, "recall": R, "f1": (2 * P * R / (P + R)) if P and R else None,
        "missed_change_rate": fn / (tp + fn) if tp + fn else None,
        "unsupported_claim_rate": fp / (tp + fp) if tp + fp else None,
        "false_change_rate_nuisance_pairs": (sum(r["false_change"] for r in nuis) / len(nuis)) if nuis else None,
        "mean_box_iou": float(np.mean([r["mean_iou"] for r in ok if r["mean_iou"] is not None])) if any(r["mean_iou"] is not None for r in ok) else None,
        "text_cer": float(np.mean(flat("cer"))) if flat("cer") else None,
        "text_read_coverage": (len(flat("cer")) / sum(r["text_gt_matched"] for r in ok)) if sum(r["text_gt_matched"] for r in ok) else None,
        "typed_fraction_of_tp": (sum(r["typed"] for r in ok) / tp) if tp else None,
        "text_exact_field_acc": float(np.mean(flat("exact"))) if flat("exact") else None,
        "change_type_acc": float(np.mean(flat("type_correct"))) if flat("type_correct") else None,
        "review_rate": float(np.mean([r["n_review"] > 0 for r in ok])) if ok else None,
        "mean_latency_s": float(np.mean([r["latency_s"] for r in ok])) if ok else None,
        "max_peak_gpu_gb": max([r["peak_gpu_gb"] or 0 for r in ok], default=None),
    }


def per_category(rows):
    cats = sorted({r["category"] for r in rows})
    return {c: aggregate([r for r in rows if r["category"] == c]) for c in cats}


def calibration_report(samples, decisions=("accepted", "needs_review", "abstained")):
    s = [x for x in samples if x["decision"] in decisions]
    if not s:
        return None
    e, bins = ece([x["score"] for x in s], [x["correct"] for x in s])
    return {"n": len(s), "ece_of_raw_heuristic_score": e, "bins": bins,
            "note": "raw_support_score is a heuristic; this ECE shows how miscalibrated it is before fitting"}


DEFAULT_ABLATIONS = [
    ("full_reference", {}),
    ("no_registration", {"use_registration": False}),
    ("no_lowlevel", {"use_lowlevel": False}),
    ("no_ocr", {"use_ocr": False}),
    ("no_semantic", {"use_vlm": False, "use_m3diff": False}),
    ("no_learned_localization", {"use_anychange": False}),
    ("no_fusion", {"use_fusion": False}),
    ("no_abstention", {"use_abstention": False}),
    ("no_reinspection", {"use_reinspection": False}),
    ("vlm_only_baseline", {"use_lowlevel": False, "use_ocr": False, "use_anychange": False, "use_fusion": False}),
]


def run_ablations(pipe, pairs, ablations=DEFAULT_ABLATIONS, include_review=False):
    """Re-uses `pipe`'s model cache; each row reports the aggregate metrics of one variant."""
    import dataclasses
    base = pipe.cfg
    table = []
    try:
        for name, ov in ablations:
            cfg = dataclasses.replace(base, **ov)
            if not cfg.use_vlm and not cfg.use_lowlevel and not cfg.use_ocr:
                continue
            pipe.reconfigure(cfg)
            print(f"=== ablation: {name} {ov}")
            ev = evaluate(pipe, pairs, include_review=include_review, progress=False)
            table.append({"variant": name, **ov, **ev["aggregate"], "aurc": ev["risk_coverage"]["aurc"]})
    finally:
        pipe.reconfigure(base)
    return table
