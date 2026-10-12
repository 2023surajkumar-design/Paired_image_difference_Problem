"""Explicit pipeline configuration. Every knob that changes a result lives here and is
serialized into the report, so a run can be reproduced from its JSON."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class Config:
    preset: str = "full"              # classical | ocr | semantic | full  (see make_config)
    device: str = "auto"              # auto | cuda | cpu
    domain: str = "auto"              # auto | document | screenshot | product | scene | fixed_camera | nonrigid
    scope: str = ""                   # optional natural-language comparison scope, passed to the VLM
    output_dir: str = "outputs"
    seed: int = 0

    # ---- ablation switches (Section 9) ----
    use_registration: bool = True
    use_lowlevel: bool = True
    use_ocr: bool = True
    use_vlm: bool = True
    use_m3diff: bool = False          # needs its own conda env, see README
    use_anychange: bool = False       # experimental, remote-sensing origin
    use_reinspection: bool = True     # native-resolution + VLM crop re-inspection
    use_fusion: bool = True           # False = naive union of every branch's proposals
    use_abstention: bool = True       # False = every candidate becomes "accepted"

    # ---- Stage A: validation ----
    use_compat_model: bool = True
    dino_model: str = "facebook/dinov2-small"  # shared by compatibility check and texture map
    min_side_px: int = 64
    max_pixels: int = 40_000_000      # refuse absurdly large inputs

    # ---- Stage B: correspondence ----
    matcher: str = "lightglue"        # lightglue | loftr | orb
    lightglue_features: str = "superpoint"  # superpoint (non-commercial weights) | disk | aliked
    max_keypoints: int = 4096
    allow_matcher_fallback: bool = True     # recorded as a substitution when used
    ransac_thresh_px: float = 3.0
    min_inliers: int = 15
    use_flow: bool = True             # DIS optical flow on the aligned pair as local-misregistration evidence

    # ---- Stage C: low-level evidence ----
    analysis_max_side: int = 2048
    photometric_normalize: bool = True
    tolerance_px: int = 1             # displacement tolerance of fused maps (+1 automatically after a warp)
    ssim_sigma: float = 1.5
    z_high: float = 6.0               # robust z (median/MAD units, NOT a probability)
    z_low: float = 3.0
    # pixel-only acceptance (no independent family) needs peak z >= factor * z_high, raised by
    # degraded_factor under compression warnings or for tiny regions on a resampled (warped) B.
    # ponytail: hand-set policy knobs; tune with evaluate.run_ablations on labelled pairs.
    pixel_only_accept_factor: float = 2.0
    degraded_factor: float = 1.5
    tiny_region_px: int = 64
    min_region_px: int = 12           # in original-image pixels
    # Noise floors stop the robust scale collapsing to ~0 on near-identical images.
    # ponytail: hand-set per-map floors; recalibrate on your sensor/domain with evaluate.py.
    noise_floor: dict = field(default_factory=lambda: {
        "lab": 2.0, "ssim": 0.03, "edge": 0.03, "texture": 0.03, "rgb": 0.02, "lum": 0.02})
    use_texture_features: bool = True
    texture_max_side: int = 644       # multiple of 14

    # ---- Stage E: OCR ----
    ocr_langs: list = field(default_factory=lambda: ["en"])
    ocr_min_conf: float = 0.5
    ocr_upscale: float = 2.5
    ocr_min_char_height_px: int = 10

    # ---- Stage D: semantic ----
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_4bit: bool = False            # 4B in bf16 ~9 GB; set True for 8B models on 16 GB
    vlm_bbox_format: str = "rel1000"  # Qwen3-VL emits 0..1000 relative boxes; Qwen2.5-VL: "abs_resized"
    vlm_max_image_pixels: int = 1024 * 1024
    vlm_max_new_tokens: int = 1024
    vlm_reinspect_top_k: int = 8
    m3diff_python: str = ""           # python executable inside the m3diff conda env
    m3diff_repo: str = "third_party/M3Diff"
    m3diff_model_path: str = "IVC-liuyuan/M3Diff"
    m3diff_load_4bit: bool = True     # 7B bf16 does not fit next to two images in 16 GB
    m3diff_max_side: int = 768
    m3diff_timeout_s: int = 1800

    # ---- Stage F: learned localization ----
    anychange_model_type: str = "vit_h"
    anychange_sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth"
    anychange_max_side: int = 1024
    anychange_conf_threshold: float = 145.0   # upstream README default (angle, degrees)

    # ---- Stage H/I ----
    calibrator_path: str = ""
    use_llm_report: bool = True
    keep_models_loaded: bool = True   # cache light models across pairs; heavy ones are swapped

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


PRESETS = {
    # lightweight classical baseline: no model downloads at all (ORB matcher, no DINOv2)
    "classical": dict(use_ocr=False, use_vlm=False, use_m3diff=False, use_anychange=False,
                      use_compat_model=False, use_texture_features=False, matcher="orb",
                      use_llm_report=False),
    "ocr": dict(use_ocr=True, use_vlm=False, use_m3diff=False, use_anychange=False,
                use_llm_report=False),
    "semantic": dict(use_ocr=False, use_vlm=True, use_anychange=False),
    "full": dict(use_ocr=True, use_vlm=True),
}


def make_config(preset: str = "full", **overrides) -> Config:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    cfg = Config(preset=preset, **PRESETS[preset])
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Config has no field {k!r}")
        setattr(cfg, k, v)
    return cfg
