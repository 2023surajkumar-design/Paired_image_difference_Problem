"""Orchestrator. Runs stages A-I, records exactly what ran / failed / was skipped, and never
lets an optional specialist's failure take down the classical baseline.

Inference order (chosen for a 16 GB GPU; heavy models never co-reside):
  A validate -> B align -> C low-level (+native re-inspection) -> E OCR -> F AnyChange
  -> D M3Diff (subprocess, GPU freed first) -> D VLM (global, grounding, crop re-inspection)
  -> G evidence graph + claims -> H calibration -> I verbalize (same VLM, text-only) -> export
"""
from __future__ import annotations

import hashlib
import os
import time
import traceback

import numpy as np

from . import align as AL
from .calibration import apply_calibrator
from .config import Config, make_config
from .env import ModelManager, StatusBoard, gpu_mem_snapshot, resolve_device
from .evidence import (EvidenceGraph, add_anychange_obs, add_lowlevel_obs, add_ocr_obs, add_semantic_obs,
                       build_claims, candidate_groups)
from .io_quality import load_image, validate_pair
from .lowlevel import DinoAdapter, add_native_flags, analyze, native_reinspect
from .report import evidence_packet, export, verbalize
from .schema import pad_box, to_jsonable


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or make_config()
        self.device = resolve_device(self.cfg.device)
        self.status = StatusBoard()
        self.mm = ModelManager(self.cfg, self.status)
        c = self.cfg
        for comp, on in (("dinov2", c.use_compat_model or (c.use_texture_features and c.use_lowlevel)),
                         ("matcher", c.use_registration), ("lowlevel", c.use_lowlevel), ("ocr", c.use_ocr),
                         ("anychange", c.use_anychange), ("m3diff", c.use_m3diff), ("vlm", c.use_vlm),
                         ("verbalizer", c.use_vlm and c.use_llm_report)):
            self.status.set(comp, "not_started" if on else "skipped", "" if on else "disabled by config")

    # ------------------------------------------------------------------
    def _stage(self, name, fn, timings):
        t0 = time.time()
        try:
            out = fn()
            self.status.set(name, "tested", f"ok in {time.time() - t0:.1f}s",
                            version=getattr(self.mm.cache.get(name), "version", None))
            return out
        except Exception as e:
            self.status.set(name, "failed", f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
            return None
        finally:
            timings[name] = round(time.time() - t0, 3)

    def _matcher(self, name):
        dev = self.device
        return {"lightglue": lambda: AL.LightGlueMatcher(dev), "loftr": lambda: AL.LoFTRMatcher(dev),
                "orb": AL.ORBMatcher}[name]

    def _vlm_factory(self):
        from .semantic import HFVLMAdapter
        c = self.cfg
        return HFVLMAdapter(c.vlm_model, self.device, c.vlm_4bit, c.vlm_max_image_pixels, c.vlm_bbox_format,
                            c.vlm_max_new_tokens)

    # ------------------------------------------------------------------
    def run(self, src_a, src_b, scope=None, out_dir=None, do_export=True):
        cfg, T = self.cfg, {}
        np.random.seed(cfg.seed)
        try:
            import torch
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        t_start = time.time()
        img_a, meta_a = load_image(src_a, cfg.max_pixels)
        img_b, meta_b = load_image(src_b, cfg.max_pixels)
        run_id = time.strftime("%Y%m%d-%H%M%S-") + hashlib.sha1(
            (str(meta_a.get("sha256")) + str(meta_b.get("sha256"))).encode()).hexdigest()[:6]
        out_dir = out_dir or os.path.join(cfg.output_dir, run_id)
        g = EvidenceGraph()
        calls = {"vlm": 0, "m3diff": 0}
        art = {"img_a": img_a, "img_b": img_b, "masks": {}}

        # ---- A: validation ----
        dino = None
        if cfg.use_compat_model or (cfg.use_texture_features and cfg.use_lowlevel):
            dino = self._stage("dinov2", lambda: self.mm.get("dinov2", lambda: DinoAdapter(cfg.dino_model, self.device)), T)
        validation = validate_pair(img_a, img_b, meta_a, meta_b, cfg,
                                   compat=dino.similarity if (dino and cfg.use_compat_model) else None)

        # ---- B: correspondence ----
        align, subst = None, []
        if cfg.use_registration and validation["status"] != "not_comparable":
            chain = [cfg.matcher] + ([m for m in ("lightglue", "loftr", "orb") if m != cfg.matcher]
                                     if cfg.allow_matcher_fallback else [])
            for name in chain:
                def do(name=name):
                    m = self.mm.get(f"matcher:{name}", self._matcher(name))
                    k_a, k_b, sc = m.predict(img_a, img_b)
                    al = AL.estimate_alignment(img_a, img_b, k_a, k_b, sc, cfg)
                    al["matcher"] = m.version or name
                    return al, k_a, k_b
                r = self._stage(f"matcher:{name}", do, T)
                if r is not None:
                    align, art["k_a"], art["k_b"] = r
                    break
                subst.append(f"{name} failed: {self.status.rows[f'matcher:{name}']['detail']}")
            if subst and align is not None:
                align["substitution"] = subst
                align["reasons"].append(f"matcher substitution: {subst}; correspondence quality may differ")
            self.status.set("matcher", "tested" if align else "failed", "; ".join(subst) or "ok")
        if align is None:
            ha, wa = img_a.shape[:2]
            hb, wb = img_b.shape[:2]
            align = {"model": "naive_resize", "M_ba": np.diag([wa / wb, ha / hb, 1.0]).tolist(), "n_matches": 0,
                     "n_inliers": 0, "inliers_a": np.zeros((0, 2)), "reasons": [],
                     "status": "approximate" if not cfg.use_registration else "failed"}
            align["reasons"].append("registration disabled (ablation): naive resize assumed aligned"
                                    if not cfg.use_registration else "all matchers failed")
        aligned_b, valid = AL.warp_b_to_a(img_b, align["M_ba"], img_a.shape)
        art["aligned_b"] = aligned_b
        if "k_a" in art and len(art["k_a"]):
            err = np.linalg.norm(AL._project(align["M_ba"], np.asarray(art["k_b"], float)) - art["k_a"], axis=1)
            art["inlier_mask"] = err < align.get("ransac_thresh_px", cfg.ransac_thresh_px)
        art["align_title"] = f"{align.get('matcher', '')} / {align['model']} / {align['status']}"

        # ---- C: low-level evidence + native re-inspection ----
        low = None
        if cfg.use_lowlevel and validation["status"] != "not_comparable":
            uf = lambda shape, s: AL.local_uncertainty(shape, align.get("inliers_a"), s, align["status"])

            def do_low():
                lo = analyze(img_a, aligned_b, valid, img_b, uf, align, cfg, self.device,
                             {"a": validation["image_a"]["quality"], "b": validation["image_b"]["quality"]},
                             dino=dino if cfg.use_texture_features else None)
                if cfg.use_reinspection:
                    for r in lo["regions"]:
                        r["native"] = native_reinspect(img_a, img_b, align["M_ba"], r["bbox_a"], lo, cfg)
                        add_native_flags(r, validation["image_a"]["quality"], validation["image_b"]["quality"])
                return lo
            low = self._stage("lowlevel", do_low, T)
            if low:
                add_lowlevel_obs(g, low, align, "idu.lowlevel")
                art["low"] = low

        # ---- E: OCR ----
        ocr = None
        if cfg.use_ocr and validation["status"] != "not_comparable":
            def do_ocr():
                from .ocr import EasyOCRAdapter, run_ocr_branch
                ad = self.mm.get("ocr", lambda: EasyOCRAdapter(cfg.ocr_langs, self.device))
                return run_ocr_branch(ad, img_a, img_b, aligned_b, align, cfg, do_reread=cfg.use_reinspection)
            ocr = self._stage("ocr", do_ocr, T)
            if ocr:
                add_ocr_obs(g, ocr, align, self.mm.cache["ocr"].version)

        # ---- F: AnyChange ----
        anych = None
        if cfg.use_anychange and validation["status"] != "not_comparable":
            def do_any():
                from .localize import AnyChangeAdapter
                ad = self.mm.get("anychange", lambda: AnyChangeAdapter(cfg, self.device))
                return ad.predict(img_a, aligned_b), ad.version
            r = self._stage("anychange", do_any, T)
            if r:
                anych = r[0]
                add_anychange_obs(g, anych, r[1])
            self.mm.release("anychange")

        # ---- D: M3Diff (own env; free this kernel's GPU first) ----
        m3 = None
        if cfg.use_m3diff and validation["status"] != "not_comparable":
            def do_m3():
                from .semantic import M3DiffAdapter
                self.mm.release_all()
                ad = M3DiffAdapter(cfg)
                env = ad.check_environment()
                if not env["ok"]:
                    raise RuntimeError("; ".join(env["problems"]))
                ad.load_model()
                calls["m3diff"] += 1
                return ad.predict(img_a, img_b), ad.version
            r = self._stage("m3diff", do_m3, T)
            if r:
                m3 = {"raw": r[0]["raw"], "hypotheses": r[0]["hypotheses"], "version": r[1]}

        # ---- D: VLM global hypotheses + grounding ----
        vlm, vlm_global = None, None
        if cfg.use_vlm and validation["status"] != "not_comparable":
            vlm = self._stage("vlm", lambda: self.mm.get("vlm", self._vlm_factory), T)
            if vlm:
                def do_hyp():
                    calls["vlm"] += 1
                    return vlm.hypotheses(img_a, img_b, scope if scope is not None else cfg.scope)
                vlm_global = self._stage("vlm_hypotheses", do_hyp, T)
                if vlm_global:
                    for h in vlm_global["hypotheses"]:
                        if h["bbox_a"] is None and h["bbox_b"] is not None:
                            h["bbox_a"] = AL.map_box_b_to_a(h["bbox_b"], align, img_a.shape)
                            h["bbox_a_method"] = "mapped_from_b"
                    add_semantic_obs(g, vlm_global, vlm.name, "vlm", vlm.version)
        if m3:
            if vlm:  # ground each M3Diff sentence with the VLM (iterative: caption -> region -> verification)
                for h in m3["hypotheses"][:6]:
                    try:
                        calls["vlm"] += 1
                        h["bbox_a"], h["grounding_raw"] = vlm.localize(img_a, h["description"])
                        h["grounded_by"] = vlm.name
                    except Exception as e:
                        h["grounding_error"] = str(e)
            add_semantic_obs(g, m3, "m3diff", "m3diff", m3["version"])

        # ---- F (iterative): candidate groups, native pixel check, VLM crop re-inspection ----
        groups = candidate_groups(g, image_area=img_a.shape[0] * img_a.shape[1])
        if cfg.use_reinspection and low:
            for gr in groups:
                if not any(m["source"] == "lowlevel_region" for m in gr["members"]):
                    n = native_reinspect(img_a, img_b, align["M_ba"], gr["bbox_a"], low, cfg)
                    o = g.add_obs(source="native_pixel_check", source_version="idu.lowlevel.native_reinspect",
                                  observation_type="native_pixel_check", family="appearance", image="AB",
                                  bbox_a=n.get("refined_box", gr["bbox_a"]), raw=n, raw_score=n.get("peak_z_native"),
                                  score_kind="robust z peak at native resolution (uncalibrated)",
                                  depends_on=[m["obs_id"] for m in gr["members"]])
                    gr["checks"].append(o.observation_id)
        routing = {"ocr": "run whenever enabled; never skipped because of a small coarse difference score",
                   "vlm_reinspection": f"top {cfg.vlm_reinspect_top_k} candidate groups by priority "
                                       "(pixel peak z, +5 if proposed by a non-pixel branch)",
                   "native_reinspection": "every low-level region and every non-pixel proposal" if cfg.use_reinspection else "disabled"}
        if vlm and cfg.use_reinspection:
            H, W = img_a.shape[:2]
            for gr in groups[:cfg.vlm_reinspect_top_k]:
                b = gr["bbox_a"]
                pad = max(16, 0.25 * max(b[2] - b[0], b[3] - b[1]))
                x1, y1, x2, y2 = [int(v) for v in pad_box(b, pad, W, H)]
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                ca = img_a[y1:y2, x1:x2]
                if align["status"] in ("reliable", "approximate"):
                    cb = aligned_b[y1:y2, x1:x2]
                else:
                    bb, _ = AL.map_box_a_to_b([x1, y1, x2, y2], align, img_b.shape)
                    cb = img_b[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                if cb.size == 0:
                    continue
                try:
                    calls["vlm"] += 1
                    rr = vlm.reinspect(ca, cb)
                except Exception as e:
                    rr = {"changed": "uncertain", "error": str(e)}
                o = g.add_obs(source=vlm.name, source_version=vlm.version, observation_type="vlm_region_check",
                              family="vlm", image="AB", bbox_a=[x1, y1, x2, y2], raw=rr,
                              depends_on=[m["obs_id"] for m in gr["members"]],
                              context={"crop_source_b": "aligned_b" if align["status"] in ("reliable", "approximate") else "mapped_b_box"})
                gr["checks"].append(o.observation_id)

        # ---- G: claims ----
        ctx = {"validation": validation, "align": align, "low": low,
               "compression_warning": any("compression" in w for w in validation["warnings"]),
               "map_a_to_b": lambda box: AL.map_box_a_to_b(box, align, img_b.shape)}
        claims, comp_status = build_claims(g, groups, ctx, cfg)
        if validation["status"] == "not_comparable":
            comp_status = "not_comparable"

        # masks for changed-pixel claims
        regions_by_obs = {r.get("obs_id"): r for r in (low or {}).get("regions", [])}
        for c in claims:
            for oid in c.supporting:
                r = regions_by_obs.get(oid)
                if r is None:
                    continue
                nat = r.get("native") or {}
                if nat.get("confirmed"):
                    art["masks"][c.claim_id] = (nat["mask_crop"], nat["crop_box"][:2], 1.0)
                else:
                    art["masks"][c.claim_id] = (r["mask_analysis"], r["mask_origin_analysis"], low["scale"])
                break

        # ---- H: calibration ----
        cal_status = apply_calibrator(claims, cfg.calibrator_path)

        result = {
            "run_id": run_id, "config": cfg.to_dict(), "device": self.device,
            "inputs": {"a": meta_a, "b": meta_b},
            "comparison_status": comp_status,
            "coordinate_convention": "xyxy, original pixels after EXIF orientation, origin top-left, x2/y2 exclusive; "
                                     "M_ba maps B->A; A is the reference frame",
            "validation": validation,
            "alignment": {k: v for k, v in align.items() if k != "inliers_a"},
            "lowlevel": None if not low else {k: v for k, v in low.items() if not k.startswith("_") and k != "maps"} | {
                "regions": [{"bbox_a": r["bbox_a"], "stats": r["stats"], "nuisance": r["nuisance"],
                             "dominant_map": r["dominant_map"], "obs_id": r.get("obs_id"),
                             "native": {k: v for k, v in (r.get("native") or {}).items() if k != "mask_crop"}}
                            for r in low["regions"]]},
            "ocr": ocr and {k: v for k, v in ocr.items()},
            "semantic": {"vlm": vlm_global, "m3diff": m3},
            "anychange": anych and [{k: v for k, v in it.items() if k != "mask_analysis"} for it in anych],
            "routing": routing,
            "calibration": cal_status,
            "summary_counts": {d: sum(c.decision == d for c in claims) for d in ("accepted", "needs_review", "abstained")},
        }
        result["component_status"] = self.status.table()
        packet = evidence_packet(result, claims)
        use_llm = vlm is not None and cfg.use_llm_report
        rep = self._stage("verbalizer", lambda: verbalize(packet, vlm), T) if use_llm else None
        if use_llm:
            calls["vlm"] += 1
        result["report"] = rep or verbalize(packet, None)
        result["evidence_packet"] = packet
        result["evidence_graph"] = g.to_json()
        if not cfg.keep_models_loaded:
            self.mm.release_all()
        result["efficiency"] = {"latency_s": round(time.time() - t_start, 2), "stage_s": T, "model_calls": calls,
                                "gpu": gpu_mem_snapshot()}
        result["component_status"] = self.status.table()
        result["claims"] = [to_jsonable(c) for c in claims]
        art["claims"] = claims
        if do_export:
            result["artifacts"] = export(result, art, claims, out_dir)
            result["artifacts_dir"] = out_dir
        return result, art

    def reconfigure(self, cfg: Config):
        """Swap config (e.g. for ablations) while keeping the loaded-model cache."""
        self.cfg = self.mm.cfg = cfg
        self.device = resolve_device(cfg.device)

    def close(self):
        self.mm.release_all()


def component_self_test(cfg: Config, run_m3diff=False):
    """Load every enabled specialist and run ONE minimal inference on a synthetic pair.
    Returns rows {component, state, detail, load_s, test_s, peak_gpu_gb}. Nothing is marked
    'tested' unless the inference call returned."""
    from .evaluate import make_pair
    a, b, _ = make_pair("color_change", 0)
    dev = resolve_device(cfg.device)
    tests = []
    if cfg.use_compat_model or cfg.use_texture_features:
        tests.append(("dinov2", lambda: DinoAdapter(cfg.dino_model, dev), lambda ad: f"cosine(A,B)={ad.similarity(a, b):.3f}"))
    for m in dict.fromkeys([cfg.matcher, "orb"]):
        fac = {"lightglue": lambda: AL.LightGlueMatcher(dev), "loftr": lambda: AL.LoFTRMatcher(dev), "orb": AL.ORBMatcher}[m]
        tests.append((f"matcher:{m}", fac, lambda ad: f"{len(ad.predict(a, b)[0])} matches"))
    if cfg.use_ocr:
        from .ocr import EasyOCRAdapter
        tests.append(("ocr", lambda: EasyOCRAdapter(cfg.ocr_langs, dev),
                      lambda ad: f"lines: {[l['text'] for l in ad.read(a)][:4]}"))
    if cfg.use_anychange:
        from .localize import AnyChangeAdapter
        tests.append(("anychange", lambda: AnyChangeAdapter(cfg, dev), lambda ad: f"{len(ad.predict(a, b))} change masks"))
    if cfg.use_vlm:
        from .semantic import HFVLMAdapter
        tests.append(("vlm", lambda: HFVLMAdapter(cfg.vlm_model, dev, cfg.vlm_4bit, cfg.vlm_max_image_pixels,
                                                  cfg.vlm_bbox_format, 32),
                      lambda ad: repr(ad.generate([a], "Describe this image in five words.", 16))))
    if cfg.use_m3diff:
        from .semantic import M3DiffAdapter
        tests.append(("m3diff", lambda: M3DiffAdapter(cfg),
                      (lambda ad: ad.predict(a, b)["raw"][:200]) if run_m3diff else None))
    rows = []
    for name, fac, fn in tests:
        row = {"component": name, "state": "not_started", "detail": "", "load_s": None, "test_s": None, "peak_gpu_gb": None}
        ad = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        try:
            ad = fac()
            env = ad.check_environment()
            if not env.get("ok"):
                row.update(state="failed", detail=f"environment: {env}")
                rows.append(row)
                continue
            t0 = time.time()
            ad.load_model(cfg)
            row.update(state="installed", load_s=round(time.time() - t0, 1), version=ad.version)
            if fn is None:
                row["detail"] = "environment ok; inference not run (pass run_m3diff=True)"
            else:
                t0 = time.time()
                row.update(state="tested", detail=str(fn(ad))[:300], test_s=round(time.time() - t0, 1))
        except Exception as e:
            row.update(state="failed", detail=f"{type(e).__name__}: {e}"[:500])
        finally:
            row["peak_gpu_gb"] = gpu_mem_snapshot().get("peak_gb")
            if ad is not None:
                ad.release_resources()
        rows.append(row)
    return rows
