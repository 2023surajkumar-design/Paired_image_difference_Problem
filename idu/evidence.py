"""Stage G: evidence graph + claim verification.

Independence accounting: evidence is grouped into FAMILIES that fail independently.
  appearance      - every pixel/edge/SSIM/texture map AND OCR-free crop comparisons. These
                    share the same images and the same alignment, so they count ONCE.
  ocr             - OCR readings / string alignment
  vlm             - all calls to the same VLM (global, crop re-inspection, grounding): shared blind spots
  m3diff          - M3Diff captions (after grounding)
  learned_change  - AnyChange masks
A claim is never accepted by counting agreeing maps; see decide_visual / decide_text for the policy.
"""
from __future__ import annotations

import math

from .schema import Claim, IdGen, Observation, to_jsonable


class EvidenceGraph:
    def __init__(self):
        self.ids = IdGen()
        self.obs: dict[str, Observation] = {}
        self.claims: dict[str, Claim] = {}

    def add_obs(self, **kw) -> Observation:
        o = Observation(observation_id=self.ids("obs"), **kw)
        self.obs[o.observation_id] = o
        return o

    def add_claim(self, claim: Claim):
        self.claims[claim.claim_id] = claim
        for oid in claim.supporting:
            self.obs[oid].supports.append(claim.claim_id)
        for oid in claim.contradicting:
            self.obs[oid].contradicts.append(claim.claim_id)

    def of_type(self, t):
        return [o for o in self.obs.values() if o.observation_type == t]

    def to_json(self):
        return {"observations": [to_jsonable(o) for o in self.obs.values()],
                "claims": [to_jsonable(c) for c in self.claims.values()],
                "edges": [{"from": o, "to": c.claim_id, "relation": "supports"} for c in self.claims.values() for o in c.supporting]
                + [{"from": o, "to": c.claim_id, "relation": "contradicts"} for c in self.claims.values() for o in c.contradicting]}


# ---------------- observation builders ----------------
def add_lowlevel_obs(g, low, align, version):
    out = []
    for r in low["regions"]:
        st = r["stats"]
        out.append(g.add_obs(
            source="lowlevel_maps", source_version=version, observation_type="lowlevel_region", family="appearance",
            image="AB", bbox_a=r["bbox_a"], raw={"stats": st, "dominant_map": r["dominant_map"], "native": r.get("native")},
            raw_score=st["peak_z"], score_kind="robust z peak (median/MAD units, uncalibrated)",
            context={"alignment_status": align["status"], "alignment_model": align["model"],
                     "local_uncertainty": st["uncert_mean"], "nuisance_signatures": r["nuisance"]}))
        r["obs_id"] = out[-1].observation_id
    return out


def add_ocr_obs(g, ocr, align, version):
    la = [g.add_obs(source="ocr", source_version=version, observation_type="ocr_line", family="ocr", image="A",
                    bbox_a=l["bbox"], raw=l, raw_score=l["conf"], score_kind="ocr line confidence (uncalibrated)")
          for l in ocr["lines_a"]]
    lb = [g.add_obs(source="ocr", source_version=version, observation_type="ocr_line", family="ocr", image="B",
                    bbox_b=l["bbox"], raw=l, raw_score=l["conf"], score_kind="ocr line confidence (uncalibrated)")
          for l in ocr["lines_b"]]
    for d in ocr["diffs"]:
        td = g.add_obs(source="ocr_string_alignment", source_version=version, observation_type="text_diff",
                       family="ocr", image="AB", bbox_a=ocr["lines_a"][d["a"]]["bbox"], bbox_b=ocr["lines_b"][d["b"]]["bbox"],
                       raw={k: v for k, v in d.items() if k != "visual"}, raw_score=min(d["conf_a"], d["conf_b"]),
                       score_kind="min OCR confidence of the two lines",
                       depends_on=[la[d["a"]].observation_id, lb[d["b"]].observation_id],
                       context={"alignment_status": align["status"], "correspondence_iou": d["iou_in_a_frame"]})
        vc = g.add_obs(source="text_crop_pixel_compare", source_version="phase-corr + normalized residual",
                       observation_type="text_visual_check", family="appearance", image="AB",
                       bbox_a=td.bbox_a, bbox_b=td.bbox_b, raw=d["visual"], raw_score=d["visual"].get("diff_frac"),
                       score_kind="fraction of crop pixels with normalized residual > 0.35",
                       depends_on=[td.observation_id])
        d["obs_id"], d["visual_obs_id"] = td.observation_id, vc.observation_id
    for kind, items in (("text_removed", ocr["removed"]), ("text_added", ocr["added"])):
        for p in items:
            o = g.add_obs(source="ocr_presence_check", source_version=version, observation_type=kind, family="ocr",
                          image="AB", bbox_a=p["bbox"] if kind == "text_removed" else p["other_bbox"],
                          bbox_b=p["other_bbox"] if kind == "text_removed" else p["bbox"], raw=p,
                          raw_score=p["conf"], score_kind="ocr line confidence (uncalibrated)")
            p["obs_id"] = o.observation_id
    for u in ocr["unchanged"]:
        g.add_obs(source="ocr_string_alignment", source_version=version, observation_type="text_unchanged",
                  family="ocr", image="AB", bbox_a=ocr["lines_a"][u["a"]]["bbox"], bbox_b=ocr["lines_b"][u["b"]]["bbox"],
                  raw=u, depends_on=[la[u["a"]].observation_id, lb[u["b"]].observation_id])


def add_semantic_obs(g, result, source, family, version):
    out = []
    for h in result.get("hypotheses", []):
        out.append(g.add_obs(source=source, source_version=version, observation_type="semantic_hypothesis",
                             family=family, image="AB", bbox_a=h.get("bbox_a"), bbox_b=h.get("bbox_b"), raw=h,
                             score_kind="self-reported uncertainty (text)" if h.get("uncertainty") else None,
                             context={"region_is_hint_only": True}))
    return out


def add_anychange_obs(g, items, version):
    return [g.add_obs(source="anychange", source_version=version, observation_type="learned_change_mask",
                      family="learned_change", image="AB", bbox_a=it["bbox_a"],
                      raw={"area_px": it["area_px"]}, raw_score=it["raw_score"],
                      score_kind="AnyChange change confidence (uncalibrated)" if it["raw_score"] is not None else None,
                      context={"location_semantics": "containing_object_or_segment"}) for it in items]


# ---------------- candidate groups ----------------
REGION_TYPES = ("lowlevel_region", "text_diff", "semantic_hypothesis", "learned_change_mask", "text_removed", "text_added")


def candidate_groups(g, image_area=None):
    from .localize import merge_candidates
    cands = [{"bbox_a": o.bbox_a, "source": o.observation_type, "obs_id": o.observation_id}
             for o in g.obs.values() if o.observation_type in REGION_TYPES and o.bbox_a is not None]
    groups = merge_candidates(cands, image_area=image_area)
    for gr in groups:
        gr["checks"] = []   # ids of re-inspection observations attached later
        peak = max((g.obs[m["obs_id"]].raw_score or 0) for m in gr["members"] if m["source"] == "lowlevel_region") \
            if any(m["source"] == "lowlevel_region" for m in gr["members"]) else 0.0
        gr["priority"] = peak + (5 if any(m["source"] != "lowlevel_region" for m in gr["members"]) else 0)
    groups.sort(key=lambda x: -x["priority"])
    return groups


# ---------------- decision policy ----------------
def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def decide_text(d, vis, cfg):
    """d: text_diff raw; vis: visual check raw. Returns (decision, reasons, reading_uncertain)."""
    rr = d.get("reread_status", "not_run")
    conf_ok = min(d["conf_a"], d["conf_b"]) >= cfg.ocr_min_conf
    corr_ok = d["iou_in_a_frame"] >= 0.3
    v = vis.get("verdict")
    if v == "low_resolution":
        return "needs_review", ["text height below legibility threshold: exact characters cannot be established"], True
    if rr == "reread_agrees_texts_equal":
        if v == "differs":
            return "needs_review", ["upscaled re-read gives identical strings but pixels differ (style/colour/font change?)"], True
        return "abstained", ["OCR mismatch not reproduced on upscaled re-read and text pixels match: likely OCR error"], True
    if v == "same":
        return ("needs_review" if conf_ok else "abstained"), ["OCR strings differ but the text crops match pixel-wise: likely misread"], True
    if v == "differs" and conf_ok and corr_ok and rr in ("stable", "not_run"):
        return "accepted", ["OCR edit + independent pixel difference in the text region + stable re-read"], False
    if v == "differs":
        why = []
        if not conf_ok:
            why.append(f"OCR confidence below {cfg.ocr_min_conf}")
        if rr == "unstable":
            why.append("re-read changes the transcription")
        if not corr_ok:
            why.append("weak text-region correspondence")
        return "needs_review", ["text region visibly changed; exact reading uncertain: " + "; ".join(why)], True
    return "needs_review", ["pixel comparison of the text crops is inconclusive"], True


def decide_visual(f, cfg):
    """f: features of a candidate group. Returns (decision, reasons, support_families, contradictions)."""
    zh = cfg.z_high
    peak, native = f["pixel_peak_z"], f["native_confirmed"]
    strong = peak >= zh and (native or not cfg.use_reinspection)
    bar = cfg.pixel_only_accept_factor * zh * (cfg.degraded_factor if f.get("degraded") else 1.0)
    very_strong = strong and peak >= bar
    support = set()
    if peak >= cfg.z_low:
        support.add("appearance")
    if f["learned"]:
        support.add("learned_change")
    if f["vlm_check"] == "yes" or (f["vlm_global"] and f["vlm_check"] != "no"):
        support.add("vlm")
    if f["m3diff"]:
        support.add("m3diff")
    contra = ["vlm_region_check:no_change"] if f["vlm_check"] == "no" else []
    other = support - {"appearance"}
    nuis = list(f["nuisance"])
    if f["vlm_check"] == "no" and f["vlm_nuisance"] not in (None, "", "none"):
        nuis.append(f"vlm_says:{f['vlm_nuisance']}")

    if "appearance" not in support:
        return "abstained", ["no pixel evidence at this location even after native-resolution re-inspection; "
                             "semantic/learned proposal withheld"], support, contra
    if not strong:
        if other and not contra:
            return "needs_review", [f"weak pixel evidence corroborated by {sorted(other)}"], support, contra
        return "abstained", ["weak pixel evidence only"], support, contra
    if not f["align_ok"]:
        if other and not contra:
            return "needs_review", ["alignment unreliable; corroborated by " + ", ".join(sorted(other))], support, contra
        return ("needs_review" if very_strong else "abstained"), \
            ["alignment unreliable: residual may be registration error"], support, contra
    if nuis:
        if contra:
            return "abstained", ["likely nuisance variation: " + ", ".join(nuis)], support, contra
        return "needs_review", ["nuisance explanation plausible: " + ", ".join(nuis)
                                + (f"; independent support from {sorted(other)}" if other else "")], support, contra
    if contra:
        return "needs_review", ["strong persistent pixel change but VLM crop re-inspection reports no content change"], support, contra
    if other:
        return "accepted", ["strong persistent pixel change under usable alignment + independent support from "
                            + ", ".join(sorted(other))], support, contra
    if very_strong:
        return "accepted", ["very strong persistent pixel change under usable alignment, no nuisance signature; "
                            "reported as a localized change WITHOUT semantic explanation"], support, contra
    return "needs_review", ["pixel-only evidence of moderate strength"], support, contra


def _score(peak, n_other, n_contra, nuis, align_ok, text_verified, cfg):
    x = (0.8 * min(peak / cfg.z_high, 3.0) + 1.0 * n_other + (1.5 if text_verified else 0)
         - 1.5 * n_contra - 0.8 * bool(nuis) - 1.0 * (not align_ok) - 2.0)
    return round(_sigmoid(x), 4)


# ---------------- claim construction ----------------
def build_claims(g, groups, ctx, cfg):
    """ctx: validation, align, shapes, map_a_to_b (fn), quality. Mutates g; returns list of claims."""
    align = ctx["align"]
    align_ok = align["status"] in ("reliable", "approximate")
    used_text = set()

    def mk(**kw):
        c = Claim(claim_id=g.ids("claim"), **kw)
        g.add_claim(c)
        return c

    for gr in groups:
        members = [g.obs[m["obs_id"]] for m in gr["members"]]
        checks = [g.obs[i] for i in gr["checks"]]
        low = [o for o in members if o.observation_type == "lowlevel_region"]
        native_checks = [o for o in checks if o.observation_type == "native_pixel_check"]
        vlm_checks = [o for o in checks if o.observation_type == "vlm_region_check"]

        # -------- text claims --------
        for td in [o for o in members if o.observation_type == "text_diff" and o.observation_id not in used_text]:
            used_text.add(td.observation_id)
            d = td.raw
            vis_o = next(o for o in g.obs.values() if o.observation_type == "text_visual_check" and td.observation_id in o.depends_on)
            dec, reasons, uncertain = decide_text(d, vis_o.raw, cfg)
            sup = [td.observation_id] + list(td.depends_on)
            contra = []
            if vis_o.raw.get("verdict") == "differs":
                sup.append(vis_o.observation_id)
            elif vis_o.raw.get("verdict") == "same":
                contra.append(vis_o.observation_id)
            sup += [o.observation_id for o in low]
            sup += [o.observation_id for o in vlm_checks if o.raw.get("changed") == "yes"]
            ops = sorted({e["op"] for e in d["edits"]})
            peak = max((o.raw_score for o in low), default=0.0)
            mk(claim_type="text_change", change_type="text_" + (ops[0] if len(ops) == 1 else "edit"),
               object_or_region="text line", bbox_a=td.bbox_a, bbox_b=td.bbox_b, bbox_b_method="ocr_detection_in_b",
               location_semantics="text_region", localization_method="ocr_line_box (+ proportional char boxes)",
               before=d["before"], after=d["after"], supporting=sup, contradicting=contra,
               confidence={"ocr": {"conf_a": d["conf_a"], "conf_b": d["conf_b"], "reread": d.get("reread_status"),
                                   "reading_uncertain": uncertain},
                           "correspondence": {"global": align["status"], "text_region_iou": d["iou_in_a_frame"]},
                           "detection": {"text_crop_pixel_verdict": vis_o.raw.get("verdict"),
                                         "diff_frac": vis_o.raw.get("diff_frac"), "pixel_peak_z": peak},
                           "localization": {"char_boxes": "proportional_estimate"}},
               raw_support_score=_score(peak, 1, len(contra), [], align_ok, dec == "accepted", cfg),
               decision=dec, decision_reasons=reasons,
               details={"edits": d["edits"], "reading_note": "exact characters uncertain" if uncertain else None})

        for kind in ("text_removed", "text_added"):
            for o in [m for m in members if m.observation_type == kind]:
                p = o.raw
                if p["status"].startswith("present_in_other_image"):
                    continue
                peak = max((x.raw_score for x in low), default=0.0)
                if p["status"] == "moved":
                    dec, why = "needs_review", ["same text detected at another location: possible text move"]
                elif p["conf"] >= cfg.ocr_min_conf and peak >= cfg.z_high:
                    dec, why = "accepted", ["text absent in the other image (re-read) and pixel residual confirms"]
                else:
                    dec, why = "needs_review", ["text presence differs per OCR; pixel or confidence support insufficient"]
                mk(claim_type="text_change", change_type="text_moved" if p["status"] == "moved" else kind,
                   object_or_region="text line", bbox_a=o.bbox_a, bbox_b=o.bbox_b, bbox_b_method=p.get("other_bbox_method"),
                   location_semantics="text_region", localization_method="ocr_line_box",
                   before=p["text"] if kind == "text_removed" else None, after=p["text"] if kind == "text_added" else None,
                   supporting=[o.observation_id] + [x.observation_id for x in low],
                   confidence={"ocr": {"conf": p["conf"]}, "detection": {"pixel_peak_z": peak}},
                   raw_support_score=_score(peak, 1, 0, [], align_ok, dec == "accepted", cfg), decision=dec, decision_reasons=why)

        if any(m.observation_type in ("text_diff", "text_removed", "text_added") for m in members):
            continue  # region explained by its text claim(s)

        # -------- visual / semantic claims --------
        nat = [o.raw for o in native_checks] + [o.raw["native"] for o in low if o.raw.get("native")]
        f = {
            "pixel_peak_z": max([o.raw_score for o in low] + [n.get("peak_z_native", 0) for n in nat if n.get("confirmed")], default=0.0),
            "native_confirmed": any(n.get("confirmed") for n in nat),
            "nuisance": sorted({fl for o in low for fl in o.context["nuisance_signatures"]}),
            "learned": [o for o in members if o.observation_type == "learned_change_mask"],
            "vlm_global": [o for o in members if o.observation_type == "semantic_hypothesis" and o.family == "vlm"],
            "m3diff": [o for o in members if o.observation_type == "semantic_hypothesis" and o.family == "m3diff"],
            "vlm_check": vlm_checks[-1].raw.get("changed") if vlm_checks else "not_run",
            "vlm_nuisance": vlm_checks[-1].raw.get("nuisance") if vlm_checks else None,
            "align_ok": align_ok and not any(o.context["local_uncertainty"] > 0.8 for o in low),
            "degraded": ctx.get("compression_warning", False) or (
                align.get("model") not in ("identity",) and
                max((o.raw["stats"]["area_px"] for o in low), default=0) < cfg.tiny_region_px),
        }
        if not low and not nat and not f["learned"] and not f["vlm_global"] and not f["m3diff"]:
            continue
        dec, reasons, support, contra = decide_visual(f, cfg)

        # location: most specific verified box wins
        refined = [n for n in nat if n.get("confirmed")]
        if refined:
            bbox, sem, how = refined[0]["refined_box"], "changed_pixels", "native_resolution_residual"
        elif low:
            bbox, sem, how = gr["bbox_a"], "changed_pixels", "analysis_resolution_residual"
        elif f["learned"]:
            bbox, sem, how = gr["bbox_a"], "containing_object", "anychange_mask"
        else:
            bbox, sem, how = gr["bbox_a"], "approximate_hint", "semantic_model_box"
        bbox_b, bmeth = ctx["map_a_to_b"](bbox)

        sem_src = (vlm_checks[-1].raw if vlm_checks and vlm_checks[-1].raw.get("changed") == "yes"
                   else f["vlm_global"][0].raw if f["vlm_global"] else None)
        if sem_src:
            ctype = sem_src.get("change_type") or "other"
            obj, before, after = sem_src.get("object"), sem_src.get("before"), sem_src.get("after")
        else:
            ctype = "unexplained_visual_change" if low else "semantic_hypothesis_only"
            obj, before, after = None, None, None
        sup_ids = [o.observation_id for o in members] + [o.observation_id for o in native_checks if o.raw.get("confirmed")]
        sup_ids += [o.observation_id for o in vlm_checks if o.raw.get("changed") == "yes"]
        con_ids = [o.observation_id for o in vlm_checks if o.raw.get("changed") == "no"]
        con_ids += [o.observation_id for o in native_checks if not o.raw.get("confirmed")]
        mk(claim_type="visual_change" if low or refined else "semantic_hypothesis", change_type=ctype,
           object_or_region=obj, bbox_a=bbox, bbox_b=bbox_b, bbox_b_method=bmeth, location_semantics=sem,
           localization_method=how, before=before, after=after, supporting=sup_ids, contradicting=con_ids,
           nuisance_explanations=f["nuisance"] + ([f"vlm:{f['vlm_nuisance']}"] if f["vlm_nuisance"] not in (None, "", "none") else []),
           confidence={"correspondence": {"global": align["status"],
                                          "local_uncertainty": max((o.context["local_uncertainty"] for o in low), default=None)},
                       "detection": {"pixel_peak_z": f["pixel_peak_z"], "native_confirmed": f["native_confirmed"],
                                     "level": "strong" if f["pixel_peak_z"] >= cfg.z_high else "weak" if f["pixel_peak_z"] >= cfg.z_low else "none"},
                       "localization": {"method": how, "semantics": sem},
                       "semantic": {"vlm_region_check": f["vlm_check"], "vlm_global_overlap": bool(f["vlm_global"]),
                                    "m3diff_grounded": bool(f["m3diff"]),
                                    "description_source": "vlm" if sem_src else None},
                       "claim": {"independent_support_families": sorted(support), "contradictions": contra}},
           raw_support_score=_score(f["pixel_peak_z"], len(support - {"appearance"}), len(contra), f["nuisance"],
                                    f["align_ok"], False, cfg),
           decision=dec, decision_reasons=reasons,
           details={"dominant_signal": low[0].raw["dominant_map"] if low else None, "sources": gr["sources"]})

    claims = list(g.claims.values())
    status = comparison_status(ctx, claims)
    for c in claims:
        if status == "inconclusive" and c.decision == "accepted" and \
                c.confidence.get("claim", {}).get("independent_support_families", ["appearance"]) == ["appearance"]:
            c.decision = "needs_review"
            c.decision_reasons.append("comparison inconclusive (alignment failed with widespread residual)")
        if not cfg.use_fusion:
            c.decision_reasons.append(f"ABLATION use_fusion=False: naive union, original decision={c.decision}")
            c.decision = "accepted"
        elif not cfg.use_abstention and c.decision != "accepted":
            c.decision_reasons.append(f"ABLATION use_abstention=False: original decision={c.decision}")
            c.decision = "accepted"
    return claims, status


def comparison_status(ctx, claims):
    v = ctx["validation"]["status"]
    if v == "not_comparable":
        return "not_comparable"
    align, low = ctx["align"], ctx.get("low") or {}
    widespread = low.get("changed_area_fraction", 0) > 0.3 or len(low.get("regions", [])) >= 40
    corroborated = any(c.decision == "accepted" and set(c.confidence.get("claim", {}).get("independent_support_families", [])) - {"appearance"}
                       for c in claims) or any(c.decision == "accepted" and c.claim_type == "text_change" for c in claims)
    if align["status"] in ("failed", "unreliable") and widespread and not corroborated:
        return "inconclusive"
    if align["status"] in ("failed", "unreliable"):
        return "comparable_with_warnings"
    return v
