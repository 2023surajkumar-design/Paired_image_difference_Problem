"""Stage D: semantic difference reasoning. Every output here is a HYPOTHESIS until Stage G
verifies it against visual evidence.

* HFVLMAdapter: any transformers image-text-to-text chat model (default Qwen3-VL-4B-Instruct).
  Used for (1) global hypotheses, (2) neutral crop re-inspection, (3) grounding free-text
  hypotheses to boxes, (4) the controlled verbalizer in Stage I (text-only).
* M3DiffAdapter: the official M3Diff eval script, run in its own conda env via subprocess.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

from .schema import Adapter, clip_box

CHANGE_TYPES = ["added", "removed", "moved", "replaced", "color_change", "attribute_change",
                "shape_or_structure_change", "text_change", "count_change", "surface_defect", "other"]
NUISANCE = ["none", "lighting", "shadow", "reflection", "blur", "compression", "misalignment", "viewpoint"]


def parse_json(text):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    s = t[i:j + 1]
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _coord_text(fmt):
    return ("integers 0-1000 relative to the width and height of that image" if fmt == "rel1000"
            else "pixel coordinates of that image as you see it")


class HFVLMAdapter(Adapter):
    heavy = True

    def __init__(self, model_id, device, use_4bit=False, max_pixels=1024 * 1024, bbox_format="rel1000",
                 max_new_tokens=1024):
        self.model_id, self.device, self.use_4bit = model_id, device, use_4bit
        self.max_pixels, self.bbox_format, self.max_new_tokens = max_pixels, bbox_format, max_new_tokens
        self.name = f"vlm:{model_id.split('/')[-1]}"

    def check_environment(self):
        try:
            import transformers
            from packaging.version import Version
            v = Version(transformers.__version__)
            if "qwen3-vl" in self.model_id.lower() and v < Version("4.57.0"):
                return {"ok": False, "error": f"transformers {v} < 4.57 lacks Qwen3-VL"}
            if self.use_4bit:
                import bitsandbytes  # noqa: F401
            return {"ok": True, "transformers": str(v)}
        except ImportError as e:
            return {"ok": False, "error": str(e)}

    def load_model(self, cfg=None):
        import torch
        import transformers
        from transformers import AutoProcessor
        dt = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device == "cuda" else torch.float32)
        kw = dict(torch_dtype=dt, device_map=self.device if self.device == "cuda" else None)
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
        cls = getattr(transformers, "AutoModelForImageTextToText", None) or transformers.AutoModelForVision2Seq
        try:
            self.model = cls.from_pretrained(self.model_id, attn_implementation="sdpa", **kw).eval()
        except (ValueError, ImportError):
            self.model = cls.from_pretrained(self.model_id, **kw).eval()
        if self.device != "cuda":
            self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.version = f"{self.model_id} (transformers {transformers.__version__}, {'nf4' if self.use_4bit else str(dt)})"

    def _fit(self, img):
        h, w = img.shape[:2]
        s = min(1.0, (self.max_pixels / (h * w)) ** 0.5)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img), s

    def generate(self, images, prompt, max_new_tokens=None):
        import torch
        content = [{"type": "image", "image": self._fit(im)[0]} for im in images] + [{"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _to_px(self, box, img):
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            return None
        try:
            box = [float(v) for v in box]
        except (TypeError, ValueError):
            return None
        h, w = img.shape[:2]
        if self.bbox_format == "rel1000":
            box = [box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000]
        else:
            s = self._fit(img)[1]
            box = [v / s for v in box]
        b = clip_box(box, w, h)
        return b if (b[2] - b[0]) >= 2 and (b[3] - b[1]) >= 2 else None

    # ---- (1) global hypotheses ----
    def hypotheses(self, img_a, img_b, scope=""):
        prompt = (
            "Image 1 is A (before). Image 2 is B (after). They show the same or corresponding subject.\n"
            + (f"Comparison scope: {scope}\n" if scope else "")
            + "List only GENUINE content changes: objects added/removed/moved/replaced, colour or attribute "
            "changes, shape/structure changes, text edits, count changes, surface defects (scratches, dents, stains).\n"
            "Do NOT list differences explained only by lighting, shadows, reflections, exposure, blur, compression "
            "or camera position; put those under \"nuisance\". If unsure, use \"uncertainty\":\"high\". "
            "Never guess text you cannot read.\n"
            f"change_type must be one of {CHANGE_TYPES}. Boxes are [x1,y1,x2,y2] in {_coord_text(self.bbox_format)}; "
            "use null if the object is not visible in that image.\n"
            'Return ONLY JSON: {"changes":[{"change_type":"","object":"","before":"","after":"","bbox_a":null,'
            '"bbox_b":null,"visual_evidence":"","uncertainty":"low|medium|high"}],"nuisance":[],"no_change":false}')
        raw = self.generate([img_a, img_b], prompt)
        js = parse_json(raw)
        hyps = []
        for c in (js or {}).get("changes", []) or []:
            if not isinstance(c, dict):
                continue
            hyps.append({"change_type": c.get("change_type") if c.get("change_type") in CHANGE_TYPES else "other",
                         "object": c.get("object"), "before": c.get("before"), "after": c.get("after"),
                         "bbox_a": self._to_px(c.get("bbox_a"), img_a), "bbox_b": self._to_px(c.get("bbox_b"), img_b),
                         "visual_evidence": c.get("visual_evidence"), "uncertainty": c.get("uncertainty")})
        return {"raw": raw, "parsed": js is not None, "hypotheses": hyps,
                "nuisance": (js or {}).get("nuisance", []), "no_change": (js or {}).get("no_change")}

    def predict(self, image_a, image_b, context=None):
        return self.hypotheses(image_a, image_b, (context or {}).get("scope", ""))

    # ---- (2) neutral crop re-inspection ----
    def reinspect(self, crop_a, crop_b):
        def up(c):
            s = max(1.0, 336 / max(1, min(c.shape[:2])))
            return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else c
        prompt = ("Image 1 and Image 2 are crops of the SAME region from image A (before) and image B (after), "
                  "aligned as well as possible. Is there a genuine content change between them? If the only "
                  "difference is brightness, shadow, reflection, blur, compression or slight misalignment, answer "
                  "changed=no and name it in nuisance.\n"
                  f"change_type one of {CHANGE_TYPES + ['none']}; nuisance one of {NUISANCE}.\n"
                  'Return ONLY JSON: {"changed":"yes|no|uncertain","change_type":"","object":"","before":"",'
                  '"after":"","nuisance":"none","explanation":""}')
        raw = self.generate([up(crop_a), up(crop_b)], prompt, max_new_tokens=256)
        js = parse_json(raw) or {}
        ch = str(js.get("changed", "uncertain")).lower()
        js["changed"] = ch if ch in ("yes", "no", "uncertain") else "uncertain"
        return {"raw": raw, **js}

    # ---- (3) grounding a free-text hypothesis ----
    def localize(self, img, description):
        prompt = (f'Locate the region described as: "{description}". Return ONLY JSON {{"bbox":[x1,y1,x2,y2]}} '
                  f'in {_coord_text(self.bbox_format)}, or {{"bbox":null}} if it is not visible.')
        raw = self.generate([img], prompt, max_new_tokens=64)
        return self._to_px((parse_json(raw) or {}).get("bbox"), img), raw

    # ---- (4) text-only generation for the verbalizer ----
    def text_only(self, prompt, max_new_tokens=700):
        import torch
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


M3DIFF_PROMPT = "What modifications can be observed between these two pictures?  <image><image>Difference:"
WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workers", "m3diff_worker.py")


class M3DiffAdapter(Adapter):
    """Runs upstream llava/eval/model_vqa_qwen_OmniDiff_Real.py unchanged (prompt format copied from
    the official data/evaluation/prompt files). GPU memory of this kernel must be released first."""
    name = "m3diff"
    heavy = True

    def __init__(self, cfg):
        self.py, self.repo, self.model_path = cfg.m3diff_python, os.path.abspath(cfg.m3diff_repo), cfg.m3diff_model_path
        self.load_4bit, self.max_side, self.timeout = cfg.m3diff_load_4bit, cfg.m3diff_max_side, cfg.m3diff_timeout_s

    def check_environment(self):
        probs = []
        if not self.py or not os.path.exists(self.py):
            probs.append("cfg.m3diff_python must point to python inside the m3diff conda env")
        if not os.path.exists(os.path.join(self.repo, "llava", "eval", "model_vqa_qwen_OmniDiff_Real.py")):
            probs.append(f"official repo not found at {self.repo} (git clone https://github.com/Yuan-Liu-01/M3Diff)")
        return {"ok": not probs, "problems": probs}

    def load_model(self, cfg=None):
        try:
            rev = subprocess.run(["git", "-C", self.repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
        except Exception:
            rev = None
        self.version = f"{self.model_path} @ repo {rev or 'unknown-rev'}{' (4-bit nf4)' if self.load_4bit else ''}"

    def predict(self, image_a, image_b, context=None):
        with tempfile.TemporaryDirectory() as td:
            for nm, im in (("a.png", image_a), ("b.png", image_b)):
                s = min(1.0, self.max_side / max(im.shape[:2]))
                im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
                Image.fromarray(im).save(os.path.join(td, nm))
            q = os.path.join(td, "q.json")
            json.dump([{"id": "pair", "image": ["a.png", "b.png"],
                        "conversations": [{"from": "human", "value": M3DIFF_PROMPT}]}], open(q, "w"))
            ans = os.path.join(td, "ans.jsonl")
            cmd = [self.py, WORKER, "--repo", self.repo] + (["--load-4bit"] if self.load_4bit else []) + [
                "--", "--model-path", self.model_path, "--question-file", q, "--image-folder", td,
                "--answers-file", ans, "--temperature", "0", "--no-flash-attention",
                "--torch-dtype", "float16" if self.load_4bit else "bfloat16"]
            p = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True, timeout=self.timeout)
            if p.returncode != 0 or not os.path.exists(ans):
                raise RuntimeError(f"m3diff worker failed (rc={p.returncode}): {p.stderr[-2000:]}")
            caption = json.loads(open(ans).readline())["caption"]
        return {"raw": caption, "hypotheses": self.normalize_output(caption)}

    def normalize_output(self, caption):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption or "") if len(s.strip()) > 3]
        return [{"change_type": "other", "object": None, "description": s, "before": None, "after": None,
                 "bbox_a": None, "bbox_b": None, "uncertainty": None} for s in sents]
