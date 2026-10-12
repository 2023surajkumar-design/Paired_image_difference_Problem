"""Stage I: evidence packet, controlled verbalization (validated), template report,
visualizations and artifact export."""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np

from .schema import to_jsonable

COLORS = {"accepted": (0, 200, 0), "needs_review": (255, 165, 0), "abstained": (150, 150, 150)}


def _brief(c):
    return {k: getattr(c, k) for k in ("claim_id", "claim_type", "change_type", "object_or_region", "before", "after",
                                       "bbox_a", "bbox_b", "location_semantics", "decision_reasons",
                                       "nuisance_explanations", "raw_support_score", "calibrated_confidence")} | {
        "support_families": c.confidence.get("claim", {}).get("independent_support_families"),
        "reading_uncertain": c.confidence.get("ocr", {}).get("reading_uncertain")}


def evidence_packet(result, claims):
    by = {d: [_brief(c) for c in claims if c.decision == d] for d in ("accepted", "needs_review", "abstained")}
    return to_jsonable({
        "comparison_status": result["comparison_status"],
        "alignment": {k: result["alignment"].get(k) for k in ("model", "status", "reasons")},
        "pair_warnings": result["validation"]["warnings"],
        "components_ran": [r["component"] for r in result["component_status"] if r["state"] == "tested"],
        "components_failed": [f"{r['component']}: {r['detail'][:120]}" for r in result["component_status"]
                              if r["state"] == "failed" and not r["component"].startswith("matcher:")],
        "substitutions": result["alignment"].get("substitution", []),
        "accepted": by["accepted"], "needs_review": by["needs_review"],
        "abstained_count": len(by["abstained"]),
        "abstained_semantic_hypotheses": [{"id": c["claim_id"], "object": c["object_or_region"], "change_type": c["change_type"]}
                                          for c in by["abstained"] if c["change_type"] != "unexplained_visual_change"][:10],
    })


VERBALIZER_PROMPT = """You write the final report of an image-difference verification system.
Use ONLY the JSON evidence below. Do not add any change, object, value, colour, or text that is not in it.
Accepted claims are established; needs_review claims must be described as uncertain; never present abstained items as changes.
If a claim has no object/before/after, describe it only as a localized visual change at its location.
If there are no accepted claims, say that no change was verified (which is not proof that nothing changed).
Return ONLY JSON: {{"summary": "<2-4 sentences>", "items": [{{"claim_id": "<id from evidence>", "sentence": "<one sentence>"}}]}}
Include one item for every accepted and every needs_review claim.

EVIDENCE:
{packet}"""


def validate_verbal(js, packet):
    problems = []
    if not isinstance(js, dict) or not isinstance(js.get("summary"), str):
        return ["output is not the requested JSON"]
    allowed = {c["claim_id"] for c in packet["accepted"] + packet["needs_review"]}
    items = [i for i in js.get("items", []) if isinstance(i, dict)]
    ids = {i.get("claim_id") for i in items}
    problems += [f"unknown claim id {i}" for i in ids - allowed]
    problems += [f"missing claim {i}" for i in allowed - ids]
    text = js["summary"] + " " + " ".join(str(i.get("sentence", "")) for i in items)
    corpus = json.dumps(packet).lower()
    quoted = [q for t in re.findall(r'"([^"]{1,80})"|“([^”]{1,80})”', text) for q in t if q]
    problems += [f"quoted string not in evidence: {q!r}" for q in quoted if q.lower() not in corpus]
    for tok in re.findall(r"\b[\w\-./]*\d[\w\-./]*\b", text):
        if tok.lower() not in corpus and not re.fullmatch(r"(claim_)?\d{1,3}", tok.lower()):
            problems.append(f"value not in evidence: {tok!r}")
    return problems


def template_report(packet) -> str:
    L = [f"**Comparison status:** {packet['comparison_status']}  ",
         f"**Alignment:** {packet['alignment']['model']} ({packet['alignment']['status']})"]
    if packet["pair_warnings"]:
        L.append(f"**Pair warnings:** {', '.join(packet['pair_warnings'])}  ")
    L.append(f"**Components that ran:** {', '.join(packet['components_ran']) or 'none'}  ")
    if packet["components_failed"]:
        L.append(f"**Failed (their evidence is absent):** {'; '.join(packet['components_failed'])}  ")
    if packet["substitutions"]:
        L.append(f"**Substitutions:** {'; '.join(s[:90] for s in packet['substitutions'])}")

    def line(c):
        loc = f"A{[round(v) for v in c['bbox_a']]}" if c["bbox_a"] else "no location"
        if c["claim_type"] == "text_change":
            s = f"text {c['change_type'].replace('text_', '')}: \"{c['before'] or ''}\" -> \"{c['after'] or ''}\" at {loc}"
            if c.get("reading_uncertain"):
                s += " (exact characters uncertain)"
        elif c["object_or_region"] or c["before"] or c["after"]:
            s = f"{c['change_type']}: {c['object_or_region'] or 'region'}"
            if c["before"] or c["after"]:
                s += f" ({c['before'] or '?'} -> {c['after'] or '?'})"
            s += f" at {loc}"
        else:
            s = f"localized visual change (no semantic explanation) at {loc}"
        return f"- `{c['claim_id']}` {s}. Evidence: {', '.join(c['support_families'] or ['ocr'])}. {'; '.join(c['decision_reasons'])}"
    L.append(f"\n### Accepted changes ({len(packet['accepted'])})")
    L += [line(c) for c in packet["accepted"]] or ["- none verified (this is not proof that nothing changed)"]
    L.append(f"\n### Needs review ({len(packet['needs_review'])})")
    L += [line(c) for c in packet["needs_review"]] or ["- none"]
    if packet["abstained_semantic_hypotheses"]:
        L.append("\n### Unverified semantic observations (withheld)")
        L += [f"- `{c['id']}` {c['change_type']}: {c['object'] or '?'}" for c in packet["abstained_semantic_hypotheses"]]
    nuis = sorted({n for c in packet["accepted"] + packet["needs_review"] for n in (c["nuisance_explanations"] or [])})
    if nuis:
        L.append(f"\n### Plausible nuisance factors (not established facts)\n- " + "\n- ".join(nuis))
    L.append(f"\n_{packet['abstained_count']} candidate(s) abstained. Scores are uncalibrated unless a calibrator was applied._")
    return "\n".join(L)


def verbalize(packet, vlm=None, retries=1):
    if vlm is None:
        return {"status": "template (no LLM)", "text": template_report(packet)}
    from .semantic import parse_json
    problems = []
    for _ in range(retries + 1):
        raw = vlm.text_only(VERBALIZER_PROMPT.format(packet=json.dumps(packet, indent=1)))
        js = parse_json(raw)
        problems = validate_verbal(js, packet)
        if not problems:
            text = js["summary"] + "\n\n" + "\n".join(f"- `{i['claim_id']}` {i['sentence']}" for i in js["items"])
            return {"status": "llm_validated", "text": text + "\n\n---\n" + template_report(packet), "raw": raw}
    return {"status": "llm_rejected -> template", "problems": problems, "text": template_report(packet)}


# ---------------- visualization ----------------
def _plt():
    import matplotlib
    if not os.environ.get("DISPLAY") and "ipykernel" not in __import__("sys").modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_claims(img, claims, which="a"):
    out = img.copy()
    t = max(2, int(0.003 * max(img.shape[:2])))
    for c in claims:
        b = c.bbox_a if which == "a" else c.bbox_b
        if b is None:
            continue
        col = COLORS[c.decision]
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        cv2.putText(out, c.claim_id.replace("claim_", "#"), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + 0.0004 * max(img.shape[:2]), col, max(1, t // 2), cv2.LINE_AA)
    return out


def colorize(z, vmax):
    z = np.clip(np.nan_to_num(z) / vmax, 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def figure_overview(art, claims):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(draw_claims(art["img_a"], claims, "a")); ax[0].set_title("Image A + claims")
    ax[1].imshow(draw_claims(art["img_b"], claims, "b")); ax[1].set_title("Image B + claims (mapped)")
    ax[2].imshow(art["aligned_b"]); ax[2].set_title("B aligned to A (evidence view)")
    for a in ax:
        a.axis("off")
    fig.suptitle("green=accepted  orange=needs_review  grey=abstained")
    fig.tight_layout()
    return fig


def figure_matches(art, max_lines=200):
    plt = _plt()
    a, b = art["img_a"], art["img_b"]
    h = max(a.shape[0], b.shape[0])
    canvas = np.full((h, a.shape[1] + b.shape[1], 3), 255, np.uint8)
    canvas[:a.shape[0], :a.shape[1]], canvas[:b.shape[0], a.shape[1]:] = a, b
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(canvas)
    ka, kb, inl = art.get("k_a"), art.get("k_b"), art.get("inlier_mask")
    if ka is not None and len(ka):
        idx = np.random.default_rng(0).permutation(len(ka))[:max_lines]
        for i in idx:
            ax.plot([ka[i, 0], kb[i, 0] + a.shape[1]], [ka[i, 1], kb[i, 1]], lw=0.6,
                    color="lime" if inl is not None and inl[i] else "red", alpha=0.7)
    ax.set_title(f"correspondences (green inlier / red outlier) - {art.get('align_title', '')}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def figure_maps(low, z_high):
    plt = _plt()
    keys = [k for k in ("fused", "z_lab", "z_ssim", "z_edge", "z_texture", "z_chroma", "z_lum_raw", "z_unaligned",
                        "flow", "uncertainty") if k in low["maps"]]
    n = len(keys)
    cols = 4
    fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(20, 5 * ((n + cols - 1) // cols)))
    ax = np.atleast_1d(ax).ravel()
    for a, k in zip(ax, keys):
        m = low["maps"][k]
        vmax = 1.0 if k == "uncertainty" else 5.0 if k == "flow" else 2 * z_high
        im = a.imshow(np.clip(m, 0, vmax), cmap="inferno", vmin=0, vmax=vmax)
        a.set_title(k + (" (robust z, NOT probability)" if k.startswith(("z_", "fused")) else ""))
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    for a in ax[n:]:
        a.axis("off")
    fig.tight_layout()
    return fig


def export(result, art, claims, out_dir):
    os.makedirs(os.path.join(out_dir, "maps"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    paths = {}
    save = lambda name, rgb: cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    save("overlay_a.png", draw_claims(art["img_a"], claims, "a"))
    save("overlay_b.png", draw_claims(art["img_b"], claims, "b"))
    save("aligned_b.png", art["aligned_b"])
    low = art.get("low")
    if low:
        for k, m in low["maps"].items():
            if m.dtype == bool:
                m = m.astype(np.float32)
            vmax = 1.0 if k in ("uncertainty", "valid") else 5.0 if k == "flow" else 2 * result["config"]["z_high"]
            save(f"maps/{k}.png", colorize(m, vmax))
    for c in claims:
        mk = art.get("masks", {}).get(c.claim_id)
        if mk is None:
            continue
        mask, (ox, oy), scale = mk
        full = np.zeros(art["img_a"].shape[:2], np.uint8)
        if scale != 1.0:
            mask = cv2.resize(mask.astype(np.uint8), None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST) > 0
            ox, oy = int(ox / scale), int(oy / scale)
        h, w = min(mask.shape[0], full.shape[0] - oy), min(mask.shape[1], full.shape[1] - ox)
        full[oy:oy + h, ox:ox + w] = mask[:h, :w].astype(np.uint8) * 255
        rel = f"masks/{c.claim_id}_imageA.png"
        cv2.imwrite(os.path.join(out_dir, rel), full)
        c.mask_ref = rel
    for name, fn in (("overview.png", lambda: figure_overview(art, claims)),
                     ("matches.png", lambda: figure_matches(art)),
                     ("maps_grid.png", lambda: figure_maps(low, result["config"]["z_high"]) if low else None)):
        fig = fn()
        if fig is not None:
            fig.savefig(os.path.join(out_dir, name), dpi=110)
            _plt().close(fig)
    result["claims"] = [to_jsonable(c) for c in claims]
    if "evidence_graph" in result:
        result["evidence_graph"]["claims"] = result["claims"]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# Image difference report `{result['run_id']}`\n\n{result['report']['text']}\n")
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, indent=2)
    paths.update({k: os.path.join(out_dir, k) for k in ("result.json", "report.md", "overview.png", "overlay_a.png")})
    return paths
