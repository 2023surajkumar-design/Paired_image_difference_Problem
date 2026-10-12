"""Runs INSIDE the m3diff conda env (python 3.10, torch 2.2.2, transformers 4.40.0).

Executes the official, unmodified llava/eval/model_vqa_qwen_OmniDiff_Real.py. The only
intervention is optional 4-bit loading, done through the upstream builder's own
`load_4bit` argument so a 7B model fits on a 16 GB GPU next to two images.

usage: python m3diff_worker.py --repo <M3Diff checkout> [--load-4bit] -- <official eval args>
"""
import argparse
import functools
import runpy
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--load-4bit", action="store_true")
args, rest = ap.parse_known_args()
rest = [a for a in rest if a != "--"]

sys.path.insert(0, args.repo)
import llava.model.builder as builder  # noqa: E402

if args.load_4bit:
    builder.load_pretrained_model = functools.partial(builder.load_pretrained_model, load_4bit=True)

sys.argv = ["model_vqa_qwen_OmniDiff_Real.py"] + rest
# run_module re-imports the eval module fresh, so its `from llava.model.builder import
# load_pretrained_model` binds to the patched function above.
runpy.run_module("llava.eval.model_vqa_qwen_OmniDiff_Real", run_name="__main__", alter_sys=True)
