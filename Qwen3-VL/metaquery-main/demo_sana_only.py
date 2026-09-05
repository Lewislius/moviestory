#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure Sana text-to-image demo for isolation testing.

Use this script to verify whether black images come from:
1) Sana base model / runtime environment, or
2) MetaQuery + Qwen3-VL conditioning path.
"""

import argparse
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

try:
    from diffusers import SanaPipeline as _SanaPipeline
except Exception:
    _SanaPipeline = None

from diffusers import DiffusionPipeline


DEFAULT_SANA_MODEL = "/home/liuzhirui/model/Qwen3-VL-main/Sana_1600M_512px_diffusers"
DEFAULT_PROMPT = "A cinematic portrait of a young traveler standing in neon rain, ultra-detailed, 35mm film style"
DEFAULT_NEG_PROMPT = "low quality, blurry, artifacts, watermark"


def _resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    target = dtype_map[dtype_name]
    if device == "cpu" and target != torch.float32:
        print("[WARN] CPU only supports stable inference with float32; switching dtype to float32.")
        return torch.float32
    return target


def _select_pipeline_class():
    if _SanaPipeline is not None:
        return _SanaPipeline, "SanaPipeline"
    return DiffusionPipeline, "DiffusionPipeline"


def _supported_call_kwargs(pipe) -> Dict[str, Any]:
    sig = inspect.signature(pipe.__call__)
    return {k: v for k, v in sig.parameters.items()}


def _image_stats(img: Image.Image) -> str:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}"


def _is_almost_black(img: Image.Image, threshold: float = 0.02) -> bool:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return float(arr.mean()) < threshold and float(arr.max()) < threshold * 2.0


def load_sana_pipeline(model_id: str, device: str, dtype: torch.dtype, local_files_only: bool):
    pipeline_cls, pipeline_name = _select_pipeline_class()
    print(f"[LOAD] pipeline_class={pipeline_name}")

    load_kwargs = {
        "torch_dtype": dtype,
    }
    if local_files_only:
        load_kwargs["local_files_only"] = True

    pipe = pipeline_cls.from_pretrained(model_id, **load_kwargs)

    if device == "cuda" and hasattr(pipe, "to"):
        pipe = pipe.to("cuda")
    elif hasattr(pipe, "to"):
        pipe = pipe.to(device)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    return pipe


def main():
    parser = argparse.ArgumentParser(description="Pure Sana text-to-image demo")
    parser.add_argument("--sana_model_id", type=str, default=DEFAULT_SANA_MODEL, help="Sana model path or HF repo id")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_dir", type=str, default="./demo_outputs_sana_only")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    target_dtype = _resolve_dtype(args.dtype, args.device)
    print(f"[ARGS] device={args.device}, dtype={target_dtype}, steps={args.num_inference_steps}, guidance={args.guidance_scale}")
    print(f"[ARGS] sana_model_id={args.sana_model_id}")

    pipe = load_sana_pipeline(
        model_id=args.sana_model_id,
        device=args.device,
        dtype=target_dtype,
        local_files_only=args.local_files_only,
    )

    call_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "num_images_per_prompt": args.num_images,
        "generator": torch.Generator(device=args.device).manual_seed(args.seed),
    }

    supported = _supported_call_kwargs(pipe)
    filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in supported}
    unsupported = sorted(set(call_kwargs.keys()) - set(filtered_kwargs.keys()))
    if unsupported:
        print(f"[INFO] ignored unsupported kwargs for this pipeline: {unsupported}")

    result = pipe(**filtered_kwargs)
    images = result.images

    out_dir = Path(args.output_dir) / datetime.now().strftime("sana_only_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        out_path = out_dir / f"sana_only_{i:02d}.png"
        img.save(out_path)
        stats = _image_stats(img)
        black_flag = _is_almost_black(img)
        print(f"[SAVE] {out_path}")
        print(f"[STAT] image[{i}] {stats}")
        if black_flag:
            print("[WARN] image appears almost black.")

    print(f"[DONE] outputs in {out_dir}")


if __name__ == "__main__":
    main()

