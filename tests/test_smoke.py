"""Smoke / regression checks. No model downloads: OCR and VLM are replaced by fakes injected
into the model cache, so the text, semantic, re-inspection and verbalizer code paths still run.

    python -m pytest tests -q        or        python tests/test_smoke.py
"""
import json
import os
import re
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idu import Pipeline, make_config  # noqa: E402
from idu.evaluate import _text_box, make_pair, make_scene, match  # noqa: E402
from idu.evidence import decide_text, decide_visual  # noqa: E402
from idu.report import validate_verbal  # noqa: E402
from idu.schema import Adapter  # noqa: E402


def _pipe(**kw):
    return Pipeline(make_config("classical", device="cpu", **kw))


def _accepted(res):
    return [c for c in res["claims"] if c["decision"] == "accepted"]


# ---------------- classical baseline ----------------
def test_true_changes_accepted_and_localized():
    p = _pipe()
    for cat in ["color_change", "add_object", "remove_object", "scratch", "color_change+add_object"]:
        a, b, gt = make_pair(cat, 7)
        res, _ = p.run(a, b, do_export=False)
        m = match(_accepted(res), gt)
        assert len(m) == len(gt), (cat, res["summary_counts"], gt)


def test_nuisance_only_pairs_produce_no_accepted_change():
    p = _pipe()
    for cat in ["noise", "lighting", "jpeg", "shift"]:
        a, b, _ = make_pair(cat, 7)
        res, _ = p.run(a, b, do_export=False)
        assert not _accepted(res), (cat, [(c["bbox_a"], c["decision_reasons"]) for c in _accepted(res)])


def test_identical_images():
    a, _, _ = make_pair("noise", 3)
    res, _ = _pipe().run(a, a.copy(), do_export=False)
    assert res["validation"]["identical_pixels"] and not res["claims"]
    assert "not proof" in res["report"]["text"]


def test_export_and_json_roundtrip():
    a, b, _ = make_pair("color_change", 7)
    with tempfile.TemporaryDirectory() as td:
        res, _ = _pipe(output_dir=td).run(a, b, out_dir=td)
        d = json.load(open(os.path.join(td, "result.json")))
        assert d["claims"] and d["claims"][0]["mask_ref"] and os.path.exists(os.path.join(td, d["claims"][0]["mask_ref"]))
        assert {"overview.png", "maps_grid.png", "report.md"} <= set(os.listdir(td))
        for c in d["claims"]:                       # every claim traces back to existing observations
            ids = {o["observation_id"] for o in d["evidence_graph"]["observations"]}
            assert set(c["supporting"]) <= ids


def test_not_comparable_tiny_image():
    a = np.zeros((20, 20, 3), np.uint8)
    res, _ = _pipe().run(a, a, do_export=False)
    assert res["comparison_status"] == "not_comparable" and not res["claims"]


# ---------------- policy unit checks ----------------
def test_decide_visual_policy():
    cfg = make_config("full")
    base = dict(pixel_peak_z=40, native_confirmed=True, nuisance=[], learned=[], vlm_global=[], m3diff=[],
                vlm_check="not_run", vlm_nuisance=None, align_ok=True, degraded=False)
    assert decide_visual(base, cfg)[0] == "accepted"
    assert decide_visual({**base, "vlm_check": "no"}, cfg)[0] == "needs_review"
    assert decide_visual({**base, "vlm_check": "no", "nuisance": ["illumination_or_shadow"]}, cfg)[0] == "abstained"
    assert decide_visual({**base, "align_ok": False, "pixel_peak_z": 8}, cfg)[0] == "abstained"
    assert decide_visual({**base, "pixel_peak_z": 0, "native_confirmed": False, "vlm_global": [1]}, cfg)[0] == "abstained"
    assert decide_visual({**base, "pixel_peak_z": 8, "vlm_check": "yes"}, cfg)[0] == "accepted"
    assert decide_visual({**base, "pixel_peak_z": 13, "degraded": True}, cfg)[0] == "needs_review"


def test_decide_text_policy():
    cfg = make_config("full")
    d = dict(conf_a=0.9, conf_b=0.9, iou_in_a_frame=0.8, reread_status="stable")
    assert decide_text(d, {"verdict": "differs"}, cfg)[0] == "accepted"
    assert decide_text(d, {"verdict": "same"}, cfg)[0] == "needs_review"
    assert decide_text({**d, "reread_status": "reread_agrees_texts_equal"}, {"verdict": "same"}, cfg)[0] == "abstained"
    assert decide_text(d, {"verdict": "low_resolution"}, cfg)[0] == "needs_review"
    assert decide_text({**d, "conf_b": 0.2}, {"verdict": "differs"}, cfg)[0] == "needs_review"


def test_verbalizer_validation():
    packet = {"accepted": [{"claim_id": "claim_001", "before": "AB-7831", "after": "AB-7837"}], "needs_review": []}
    ok = {"summary": "One text change was verified.", "items": [{"claim_id": "claim_001", "sentence": 'Text "AB-7831" became "AB-7837".'}]}
    assert validate_verbal(ok, packet) == []
    bad = {"summary": "Two changes, including a new logo.", "items": [{"claim_id": "claim_002", "sentence": 'Price "9.99" changed.'}]}
    probs = validate_verbal(bad, packet)
    assert any("unknown claim" in p for p in probs) and any("9.99" in p for p in probs) and any("missing" in p for p in probs)


# ---------------- fakes: OCR + VLM code paths without models ----------------
class FakeOCR(Adapter):
    """Returns ground-truth text lines of whichever source image the (possibly upscaled) crop came from."""
    name, version = "fake_ocr", "fake"

    def __init__(self, a, b, lines_a, lines_b):
        self.a, self.b, self.la, self.lb = a, b, lines_a, lines_b

    def read(self, img, mag_ratio=1.0, offset=(0, 0), scale=1.0):
        ox, oy = int(offset[0]), int(offset[1])
        w, h = int(round(img.shape[1] / scale)), int(round(img.shape[0] / scale))
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        da = np.abs(self.a[oy:oy + h, ox:ox + w].astype(np.float32) - small[:self.a[oy:oy + h, ox:ox + w].shape[0], :self.a[oy:oy + h, ox:ox + w].shape[1]]).mean()
        db = np.abs(self.b[oy:oy + h, ox:ox + w].astype(np.float32) - small[:self.b[oy:oy + h, ox:ox + w].shape[0], :self.b[oy:oy + h, ox:ox + w].shape[1]]).mean()
        lines = self.la if da <= db else self.lb
        out = []
        for l in lines:
            cx, cy = (l["bbox"][0] + l["bbox"][2]) / 2, (l["bbox"][1] + l["bbox"][3]) / 2
            if ox <= cx < ox + w and oy <= cy < oy + h:
                out.append(dict(l, quad=[[l["bbox"][0], l["bbox"][1]], [l["bbox"][2], l["bbox"][1]],
                                         [l["bbox"][2], l["bbox"][3]], [l["bbox"][0], l["bbox"][3]]]))
        return out


class FakeVLM(Adapter):
    name, version, heavy = "vlm:fake", "fake", True

    def __init__(self, gt, hallucinate=False):
        self.gt, self.hallucinate, self.bbox_format = gt, hallucinate, "rel1000"

    def hypotheses(self, a, b, scope=""):
        hy = [{"change_type": "color_change", "object": "square", "before": "blue", "after": "orange",
               "bbox_a": g["bbox_a"], "bbox_b": None, "visual_evidence": "", "uncertainty": "low"} for g in self.gt]
        hy.append({"change_type": "added", "object": "phantom logo", "before": None, "after": "logo",
                   "bbox_a": [5, 5, 60, 60], "bbox_b": None, "visual_evidence": "", "uncertainty": "high"})
        return {"raw": "fake", "parsed": True, "hypotheses": hy, "nuisance": [], "no_change": False}

    def reinspect(self, ca, cb):
        diff = np.abs(ca.astype(int) - cb.astype(int)).mean()
        return {"changed": "yes" if diff > 3 else "no", "change_type": "color_change", "object": "square",
                "before": "blue", "after": "orange", "nuisance": "none" if diff > 3 else "lighting"}

    def localize(self, img, desc):
        return None, "fake"

    def text_only(self, prompt, max_new_tokens=700):
        packet = json.loads(prompt.split("EVIDENCE:", 1)[1])
        items = [{"claim_id": c["claim_id"], "sentence": "A verified change."} for c in packet["accepted"] + packet["needs_review"]]
        if self.hallucinate:
            items.append({"claim_id": "claim_999", "sentence": 'The serial "XYZ-0001" was added.'})
        return json.dumps({"summary": "Verified changes are listed below.", "items": items})


def _text_lines(scene, ops_texts):
    return [{"bbox": _text_box(o), "text": t, "conf": 0.95} for o, t in ops_texts]


def test_ocr_text_path_with_fake_ocr():
    rng = np.random.default_rng(11)
    a, b, gt = make_pair("text_edit", 11)
    sc = make_scene(np.random.default_rng(11))
    tops = [o for o in sc["ops"] if o["kind"] == "text"]
    m = 32  # make_pair crops a 32 px margin
    la = [{"bbox": [v - m for v in _text_box(o)], "text": o["text"], "conf": 0.95} for o in tops]
    lb = [dict(l, text=gt[0]["after"]) if l["text"] == gt[0]["before"] else dict(l) for l in la]
    p = Pipeline(make_config("ocr", device="cpu", use_compat_model=False, use_texture_features=False, matcher="orb"))
    p.mm.cache["ocr"] = FakeOCR(a, b, la, lb)
    res, _ = p.run(a, b, do_export=False)
    text = [c for c in res["claims"] if c["claim_type"] == "text_change"]
    assert len(text) == 1 and text[0]["decision"] == "accepted", [(c["decision"], c["decision_reasons"]) for c in text]
    assert text[0]["before"] == gt[0]["before"] and text[0]["after"] == gt[0]["after"]
    assert text[0]["details"]["edits"][0]["op"] == "substitution"
    del rng


def test_semantic_path_with_fake_vlm():
    a, b, gt = make_pair("color_change", 7)
    for halluc in (False, True):
        p = Pipeline(make_config("semantic", device="cpu", use_compat_model=False, use_texture_features=False,
                                 matcher="orb"))
        p.mm.cache["vlm"] = FakeVLM(gt, hallucinate=halluc)
        res, _ = p.run(a, b, do_export=False)
        acc = _accepted(res)
        assert len(match(acc, gt)) == 1 and len(acc) == 1, [(c["bbox_a"], c["decision"]) for c in res["claims"]]
        assert acc[0]["object_or_region"] == "square"                       # description from VLM, verified by pixels
        assert "vlm" in acc[0]["confidence"]["claim"]["independent_support_families"]
        phantom = [c for c in res["claims"] if c.get("object_or_region") == "phantom logo"]
        assert phantom and phantom[0]["decision"] == "abstained"           # ungrounded hypothesis withheld
        assert res["report"]["status"] == ("llm_rejected -> template" if halluc else "llm_validated")
        assert "XYZ-0001" not in res["report"]["text"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, "->", repr(e)[:800])
    sys.exit(1 if fails else 0)
