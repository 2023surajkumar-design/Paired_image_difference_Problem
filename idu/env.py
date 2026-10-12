"""Environment report, component status board, VRAM-aware model cache, model inventory."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import sys
import time
import urllib.request

from .schema import free_cuda

LIBS = ["torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "opencv-python",
        "opencv-python-headless", "numpy", "Pillow", "scipy", "scikit-learn", "kornia",
        "lightglue", "easyocr", "torchange", "matplotlib", "ipywidgets", "qwen-vl-utils"]


def _ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _reachable(url, timeout=5):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        return True
    except Exception as e:  # HTTP errors still mean the host answered
        return hasattr(e, "code")


def environment_report(check_network=True) -> dict:
    rep = {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "runtime": ("colab" if "google.colab" in sys.modules
                    else "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                    else "local_or_cloud"),
        "libraries": {k: _ver(k) for k in LIBS},
        "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "hf_home": os.environ.get("HF_HOME"),
    }
    try:
        import psutil
        rep["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        rep["ram_gb"] = None
    try:
        import torch
        rep["torch_cuda_build"] = torch.version.cuda
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            rep["gpu"] = p.name
            rep["vram_gb"] = round(p.total_memory / 1e9, 1)
            rep["compute_capability"] = f"{p.major}.{p.minor}"
            rep["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        rep["cuda_available"] = False
    if check_network:
        rep["huggingface_reachable"] = _reachable("https://huggingface.co")
        rep["github_reachable"] = _reachable("https://github.com")
    return rep


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


STATES = ("not_started", "installing", "installed", "tested", "failed", "deferred", "skipped")


class StatusBoard:
    """What actually ran. Nothing is reported as tested unless the call returned."""

    def __init__(self):
        self.rows = {}

    def set(self, component, state, detail="", **extra):
        assert state in STATES, state
        self.rows[component] = {"component": component, "state": state, "detail": str(detail)[:500],
                                "time": time.strftime("%H:%M:%S"), **extra}

    def table(self):
        return list(self.rows.values())


class ModelManager:
    """Caches adapters. Loading a heavy adapter first evicts other heavy ones so the
    16 GB budget is never shared by two big models."""

    def __init__(self, cfg, status: StatusBoard):
        self.cfg, self.status, self.cache = cfg, status, {}

    def get(self, key, factory):
        if key in self.cache:
            return self.cache[key]
        ad = factory()
        if ad.heavy:
            for k in [k for k, a in self.cache.items() if a.heavy]:
                self.release(k)
        env = ad.check_environment()
        if not env.get("ok"):
            raise RuntimeError(f"{key}: environment check failed: {env}")
        t0 = time.time()
        ad.load_model(self.cfg)
        self.status.set(key, "installed", f"loaded in {time.time() - t0:.1f}s", version=ad.version)
        self.cache[key] = ad
        return ad

    def release(self, key):
        ad = self.cache.pop(key, None)
        if ad is not None:
            ad.release_resources()

    def release_all(self, heavy_only=False):
        for k in [k for k, a in self.cache.items() if a.heavy or not heavy_only]:
            self.release(k)
        free_cuda()


def gpu_mem_snapshot() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    except ImportError:
        pass
    return {}


# Verified against upstream pages on 2026-10-12 (see README "Model inventory").
# "status" here is upstream verification, not installation; installation status is
# produced at runtime by StatusBoard.
MODEL_INVENTORY = [
    dict(component="Pair compatibility + texture features", name="DINOv2",
         repo="https://github.com/facebookresearch/dinov2", paper="Oquab et al., arXiv:2304.07193 (TMLR 2024)",
         checkpoint="facebook/dinov2-base, facebook/dinov2-small (HF)", license="Apache-2.0 (code+weights)",
         deps="transformers", hardware="<1 GB VRAM", io="image -> CLS + patch tokens",
         status="verified_upstream", limitations="High similarity does not prove same physical instance."),
    dict(component="Correspondence (primary)", name="SuperPoint/DISK/ALIKED + LightGlue",
         repo="https://github.com/cvg/LightGlue", paper="Lindenberger et al., ICCV 2023",
         checkpoint="auto-downloaded from GitHub releases on first use",
         license="LightGlue code Apache-2.0; SuperPoint weights Magic Leap NON-COMMERCIAL; DISK Apache-2.0; ALIKED BSD-3",
         deps="torch, kornia>=0.6.11", hardware="~1-2 GB VRAM", io="2 images -> matched keypoints + scores",
         status="verified_upstream", limitations="Sparse; weak on textureless screenshots/documents."),
    dict(component="Correspondence (alternative)", name="LoFTR (kornia)",
         repo="https://github.com/zju3dv/LoFTR", paper="Sun et al., CVPR 2021",
         checkpoint="kornia.feature.LoFTR(pretrained='outdoor'|'indoor')", license="Apache-2.0",
         deps="kornia", hardware="~2-4 GB VRAM at 840px", io="2 gray images -> dense-ish matches + confidence",
         status="verified_upstream", limitations="Fixed coarse resolution; slower than LightGlue."),
    dict(component="Correspondence (no-download fallback)", name="ORB + RANSAC (OpenCV)",
         repo="https://github.com/opencv/opencv", paper="Rublee et al., ICCV 2011", checkpoint="none",
         license="Apache-2.0", deps="opencv", hardware="CPU", io="keypoint matches",
         status="verified_upstream", limitations="Least robust to viewpoint/illumination."),
    dict(component="OCR", name="EasyOCR (CRAFT + CRNN)", repo="https://github.com/JaidedAI/EasyOCR",
         paper="CRAFT: Baek et al., CVPR 2019", checkpoint="auto-downloaded per language",
         license="Apache-2.0", deps="easyocr", hardware="~1-2 GB VRAM",
         io="image -> [(quad, text, conf)]", status="verified_upstream",
         limitations="Line-level confidence only; no per-character confidence or boxes."),
    dict(component="Semantic reasoner (general MLLM) + controlled verbalizer", name="Qwen3-VL-4B-Instruct",
         repo="https://github.com/QwenLM/Qwen3-VL", paper="Qwen3-VL technical report (2025)",
         checkpoint="Qwen/Qwen3-VL-4B-Instruct (HF)", license="Apache-2.0",
         deps="transformers>=4.57", hardware="~9-11 GB VRAM bf16; ~4 GB in 4-bit",
         io="images + prompt -> text (we request JSON, boxes in 0..1000 relative coords)",
         status="verified_upstream", limitations="Can hallucinate; boxes are hints until verified."),
    dict(component="Semantic reasoner (difference-specialised baseline)", name="M3Diff (OmniDiff, ICCV 2025)",
         repo="https://github.com/Yuan-Liu-01/M3Diff",
         paper="Liu et al., OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning, ICCV 2025",
         checkpoint="IVC-liuyuan/M3Diff (HF), base lmms-lab/llava-onevision-qwen2-7b-ov",
         license="Code Apache-2.0; weights subject to their own/base-model terms",
         deps="SEPARATE env: python 3.10, torch 2.2.2, transformers 4.40.0 (pinned upstream)",
         hardware="7B: ~15 GB bf16 (does not fit with images on 16 GB) -> 4-bit ~6-8 GB",
         io="official eval script llava/eval/model_vqa_qwen_*.py: question JSON + image folder -> JSONL captions",
         status="verified_upstream", limitations="Free-text captions, no boxes; 4-bit may shift outputs vs paper."),
    dict(component="Learned change localization (experimental)", name="AnyChange (Segment Any Change)",
         repo="https://github.com/Z-Zheng/pytorch-change-models", paper="Zheng et al., NeurIPS 2024",
         checkpoint="SAM ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
         license="Apache-2.0 (torchange, SAM)", deps="torchange", hardware="~6-8 GB VRAM (ViT-H)",
         io="2 co-registered HxWx3 uint8 -> MaskData with 'rles' (+ optional 'boxes')",
         status="verified_upstream", limitations="Designed for remote sensing; masks may be object-level."),
    dict(component="Text-conditioned scene change", name="ViewDelta (ICCVW 2025)", repo="not verified",
         paper="Varghese et al., ICCVW 2025", checkpoint="not verified", license="unknown", deps="", hardware="",
         io="", status="deferred", limitations="Public code/weights not verified in this build."),
    dict(component="Joint registration + change detection", name="DiffRegCD (WACV 2026)", repo="not verified",
         paper="Madani et al., WACV 2026", checkpoint="not verified", license="unknown", deps="", hardware="",
         io="", status="deferred", limitations="Not integrated; aerial/street domain."),
    dict(component="Change captioning + segmentation", name="CCRC", repo="not verified",
         paper="FAIA250884 (2026)", checkpoint="not verified", license="unknown", deps="", hardware="", io="",
         status="deferred", limitations="Inference code/checkpoints not verified."),
    dict(component="Difference captioning", name="Find and Perceive (IJCAI 2025) / Multi-grained (TPAMI 2026)",
         repo="not verified", paper="Lv et al. IJCAI 2025; Tu et al. TPAMI 2026", checkpoint="not verified",
         license="unknown", deps="", hardware="", io="", status="deferred",
         limitations="Add via semantic.Adapter once weights are public."),
    dict(component="Image-based text change detection", name="Park et al. (WACV 2025)", repo="not verified",
         paper="Text Change Detection in Multilingual Documents using Image Comparison, WACV 2025",
         checkpoint="not verified", license="unknown", deps="", hardware="", io="", status="deferred",
         limitations="Replaced here by classical crop-level visual verification (ocr.visual_check)."),
]
