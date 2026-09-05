#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick quality probe for MetaQuery checkpoints.

Purpose:
1) Rapidly detect black-image collapse / low-contrast collapse.
2) Compare multiple checkpoints with fixed prompts and same seed policy.
3) Produce a JSON summary for checkpoint selection.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from pipeline_metaquery import MetaQueryPipeline
from trainer_utils import find_newest_checkpoint


DEFAULT_PROMPTS = [
    "A cinematic portrait of a traveler in neon rain, ultra-detailed, 35mm film style",
    "A cozy wooden cabin in snowy mountains at sunset, warm light, high detail",
    "A futuristic city skyline at night with flying vehicles and reflections",
    "A macro photo of a dew-covered red rose in soft morning light",
    "A fantasy castle above clouds, dramatic volumetric lighting",
    "A street photography scene of a busy market in Tokyo, realistic colors",
    "A watercolor illustration of a cat reading a book near a window",
    "A product photo of a minimalist white sneaker on studio background",
    "An astronaut riding a horse on Mars, surreal but realistic",
    "A serene lake with mirror reflections and distant mountains",
]


def _to_dtype(name: str, device: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = mapping[name]
    if device == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def _img_stats(img: Image.Image) -> Dict[str, float]:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    vmax = float(arr.max())
    vmin = float(arr.min())
    mean = float(arr.mean())
    std = float(arr.std())
    # average channel saturation proxy
    sat = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    # black collapse heuristic
    is_black = bool(mean < 0.03 and vmax < 0.12 and std < 0.03)
    # low-contrast collapse heuristic
    is_flat = bool(std < 0.035)
    # dynamic range proxy
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    dr = p99 - p1
    return {
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": std,
        "sat": sat,
        "p1": p1,
        "p99": p99,
        "dr": float(dr),
        "is_black": float(is_black),
        "is_flat": float(is_flat),
    }


def _aggregate(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["min", "max", "mean", "std", "sat", "p1", "p99", "dr", "is_black", "is_flat"]
    out = {}
    for k in keys:
        vals = [s[k] for s in stats_list]
        out[f"{k}_avg"] = float(np.mean(vals))
        out[f"{k}_med"] = float(np.median(vals))
    out["num_samples"] = len(stats_list)
    out["black_ratio"] = out["is_black_avg"]
    out["flat_ratio"] = out["is_flat_avg"]
    # simple quality score in [0, 100], higher is better
    # penalize black/flat strongly, encourage moderate contrast and saturation
    contrast_score = np.clip(out["std_avg"] / 0.20, 0.0, 1.0)
    range_score = np.clip(out["dr_avg"] / 0.75, 0.0, 1.0)
    sat_score = np.clip(out["sat_avg"] / 0.35, 0.0, 1.0)
    collapse_penalty = 1.0 - np.clip(0.8 * out["black_ratio"] + 0.5 * out["flat_ratio"], 0.0, 1.0)
    out["quick_quality_score"] = float(100.0 * (0.45 * contrast_score + 0.35 * range_score + 0.20 * sat_score) * collapse_penalty)
    return out


def _load_pipeline(ckpt_path: str, dtype: torch.dtype, device: str) -> MetaQueryPipeline:
    pipeline = MetaQueryPipeline.from_pretrained(
        ckpt_path,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device=device, dtype=dtype)
    pipeline.eval()
    return pipeline


def _run_once(
    pipeline: MetaQueryPipeline,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
) -> Image.Image:
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    out = pipeline(
        image=None,
        caption=prompt,
        negative_prompt="low quality, blurry, artifacts, watermark",
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator,
        enable_progress_bar=False,
    ).images[0]
    return out


def evaluate_checkpoint(
    ckpt_path: str,
    prompts: List[str],
    seeds: List[int],
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    dtype: torch.dtype,
    device: str,
    save_dir: Path,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    pipeline = _load_pipeline(ckpt_path, dtype=dtype, device=device)
    sample_stats: List[Dict[str, float]] = []
    save_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for p in prompts:
        for s in seeds:
            img = _run_once(
                pipeline,
                prompt=p,
                seed=s,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
            )
            out_path = save_dir / f"{idx:04d}_seed{s}.png"
            img.save(out_path)
            st = _img_stats(img)
            st["prompt"] = p
            st["seed"] = float(s)
            st["file"] = str(out_path)
            sample_stats.append(st)
            idx += 1
    agg = _aggregate(sample_stats)
    return agg, sample_stats


def main():
    parser = argparse.ArgumentParser(description="Quick quality probe for inst_small checkpoint(s)")
    parser.add_argument("--checkpoint_paths", nargs="+", required=True, help="One or more checkpoint/run dirs")
    parser.add_argument("--output_dir", type=str, default="./quick_eval_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--prompts_file", type=str, default="")
    parser.add_argument("--max_prompts", type=int, default=10)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        prompts = [x.strip() for x in lines if x.strip()][: args.max_prompts]
    else:
        prompts = DEFAULT_PROMPTS[: args.max_prompts]

    dtype = _to_dtype(args.dtype, args.device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ckpt in args.checkpoint_paths:
        resolved = find_newest_checkpoint(ckpt)
        print(f"[EVAL] checkpoint={resolved}")
        ckpt_name = Path(resolved).name
        save_dir = out_root / ckpt_name
        agg, sample_stats = evaluate_checkpoint(
            ckpt_path=resolved,
            prompts=prompts,
            seeds=seeds,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            dtype=dtype,
            device=args.device,
            save_dir=save_dir,
        )
        result = {"checkpoint": resolved, "aggregate": agg, "samples": sample_stats}
        all_results.append(result)
        print(
            f"[RESULT] {ckpt_name} score={agg['quick_quality_score']:.2f} "
            f"black_ratio={agg['black_ratio']:.3f} flat_ratio={agg['flat_ratio']:.3f} "
            f"mean={agg['mean_avg']:.3f} std={agg['std_avg']:.3f}"
        )

        (save_dir / "summary.json").write_text(
            json.dumps({"checkpoint": resolved, "aggregate": agg}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report_path = out_root / "all_results.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report={report_path}")


if __name__ == "__main__":
    main()

