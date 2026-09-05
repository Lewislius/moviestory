#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MetaQuery + Qwen3-VL + Sana (transformers 5.2 compatible) non-Gradio demo.

用途:
1) 直接在代码中给出示例文本与参考图示例
2) 一次运行自动保存结果图
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from pipeline_metaquery import MetaQueryPipeline
from trainer_utils import find_newest_checkpoint

DEFAULT_CHECKPOINT_PATH = "/home/liuzhirui/model/Qwen3-VL-main/metaquery-main/checkpoints/output/qwen3vl2b_inst_small/checkpoint-370"
DEFAULT_REF_IMAGE = "/home/liuzhirui/model/SCAIL/examples/004/ref.jpg"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_VAE_DTYPE = "auto"
DEFAULT_NUM_INFERENCE_STEPS = 30


def _force_default_device_cpu() -> None:
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass


def _sanitize_meta_env() -> None:
    # 某些启动脚本会设置该变量，触发 accelerate 的 meta 初始化路径。
    if os.environ.get("ACCELERATE_USE_META_DEVICE") == "1":
        os.environ.pop("ACCELERATE_USE_META_DEVICE", None)
        print("[LOAD] unset ACCELERATE_USE_META_DEVICE=1")


def _is_meta_default_device() -> bool:
    try:
        return torch.empty(1).device.type == "meta"
    except Exception:
        return False


def _read_checkpoint_config(ckpt_dir: str) -> Dict[str, Any]:
    cfg_file = Path(ckpt_dir) / "config.json"
    if not cfg_file.exists():
        return {}
    try:
        return json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _check_qwen3_sana(cfg: Dict[str, Any]) -> None:
    ckpt_format = str(cfg.get("format", ""))
    mllm_id = str(cfg.get("mllm_id", ""))
    diffusion_model_id = str(cfg.get("diffusion_model_id", ""))
    vae_id = str(cfg.get("vae_id", ""))
    print(f"[CHECK] format={ckpt_format}")
    print(f"[CHECK] mllm_id={mllm_id}")
    print(f"[CHECK] diffusion_model_id={diffusion_model_id}")
    print(f"[CHECK] vae_id={vae_id}")
    if mllm_id and "Qwen3-VL" not in mllm_id:
        print(f"[WARN] mllm_id 不是 Qwen3-VL: {mllm_id}")
    if (diffusion_model_id and "Sana" not in diffusion_model_id) and (vae_id and "Sana" not in vae_id):
        print("[WARN] 看起来不是 Sana 路线 checkpoint，生成质量/可用性可能受影响。")


def _validate_checkpoint_config(cfg: Dict[str, Any], allow_non_sana_checkpoint: bool) -> None:
    """
    默认拒绝明显非 Sana/MetaQuery 的 checkpoint，避免“加载成功但全黑”的静默错配。
    """
    ckpt_format = str(cfg.get("format", ""))
    diffusion_model_id = str(cfg.get("diffusion_model_id", ""))
    vae_id = str(cfg.get("vae_id", ""))

    # Wan 训练产物的典型格式标记：format=wan_metaquery_encoder
    if ckpt_format == "wan_metaquery_encoder":
        raise ValueError(
            "检测到 Wan checkpoint (format=wan_metaquery_encoder)。"
            "该 demo 走的是 Qwen3-VL + Sana 的 MetaQueryPipeline，二者不兼容。"
        )

    looks_like_sana = ("Sana" in diffusion_model_id) or ("Sana" in vae_id)
    if not looks_like_sana and not allow_non_sana_checkpoint:
        raise ValueError(
            "checkpoint 看起来不是 Sana 路线（diffusion_model_id / vae_id 未包含 Sana）。"
            "如确需强行尝试，可加 --allow_non_sana_checkpoint。"
        )


def _resolve_vae_dtype(vae_dtype_name: str, default_dtype: torch.dtype, device: str) -> torch.dtype:
    if vae_dtype_name == "auto":
        # 经验上 VAE 用 fp32 更稳，能显著降低黑图/数值塌陷风险。
        return torch.float32

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    target = dtype_map[vae_dtype_name]
    if device == "cpu" and target != torch.float32:
        print("[WARN] CPU 下 VAE 自动切换为 float32")
        return torch.float32
    return target


def _try_load_pipeline(
    ckpt_path: str,
    dtype: torch.dtype,
    allow_mismatched_sizes: bool = False,
) -> MetaQueryPipeline:
    # 兼容 transformers 参数差异
    kwarg_candidates = [
        {
            "ignore_mismatched_sizes": allow_mismatched_sizes,
            "_gradient_checkpointing": False,
            "torch_dtype": dtype,
            "_fast_init": False,
            "low_cpu_mem_usage": False,
            "device_map": None,
        },
        {
            "ignore_mismatched_sizes": allow_mismatched_sizes,
            "_gradient_checkpointing": False,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": False,
            "device_map": None,
        },
        {
            "ignore_mismatched_sizes": allow_mismatched_sizes,
            "_gradient_checkpointing": False,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": False,
        },
        {
            "ignore_mismatched_sizes": allow_mismatched_sizes,
            "_gradient_checkpointing": False,
            "torch_dtype": dtype,
        },
    ]

    last_err: Optional[Exception] = None
    for i, kwargs in enumerate(kwarg_candidates, start=1):
        try:
            _sanitize_meta_env()
            _force_default_device_cpu()
            print(f"[LOAD] try #{i}: {list(kwargs.keys())}")
            return MetaQueryPipeline.from_pretrained(ckpt_path, **kwargs)
        except Exception as e:
            print(f"[LOAD] try #{i} failed: {e}")
            if "meta device context manager" in str(e) or _is_meta_default_device():
                print("[LOAD] detected meta default device context, will retry with CPU default device.")
            last_err = e
    raise RuntimeError(f"Failed to load pipeline from {ckpt_path}: {last_err}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_images(images: List[Image.Image], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        out = output_dir / f"{prefix}_{i:02d}.png"
        img.save(out)
        print(f"[SAVE] {out}")


def run_demo(
    pipeline: MetaQueryPipeline,
    output_dir: Path,
    ref_image_path: Optional[str],
    seed: int,
    num_inference_steps: int,
) -> None:
    _seed_everything(seed)
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    # 示例1：文本生图
    t2i_prompt = "A cinematic portrait of a young traveler standing in neon rain, ultra-detailed, 35mm film style"
    t2i_neg = "low quality, blurry, artifacts, deformed face, watermark"
    print("[DEMO] text-to-image")
    t2i_images = pipeline(
        image=None,
        caption=t2i_prompt,
        negative_prompt=t2i_neg,
        guidance_scale=4.5,
        image_guidance_scale=1.5,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=2,
        generator=gen,
        enable_progress_bar=True,
    ).images
    _save_images(t2i_images, output_dir, "demo_t2i")

    # 示例2：参考图 + 文本
    if ref_image_path and Path(ref_image_path).exists():
        print(f"[DEMO] image+text, ref={ref_image_path}")
        ref = Image.open(ref_image_path).convert("RGB")
        i2i_prompt = "Keep the main subject identity, change scene to a warm golden-hour street, rich bokeh"
        i2i_neg = "low quality, blurry, extra fingers, artifacts, watermark"
        i2i_images = pipeline(
            image=[[ref]],
            caption=i2i_prompt,
            negative_prompt=i2i_neg,
            guidance_scale=4.5,
            image_guidance_scale=1.8,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=2,
            generator=gen,
            enable_progress_bar=True,
        ).images
        _save_images(i2i_images, output_dir, "demo_i2i")
    else:
        print("[DEMO] skip image+text (未提供或未找到参考图)")


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaQuery Qwen3-VL + Sana non-Gradio demo")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="checkpoint dir or run dir",
    )
    parser.add_argument("--ref_image", type=str, default=DEFAULT_REF_IMAGE, help="optional reference image path")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vae_dtype", type=str, default=DEFAULT_VAE_DTYPE, choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--allow_mismatched_sizes", action="store_true",
                        help="允许尺寸不匹配权重加载（默认关闭，避免静默错配）")
    parser.add_argument("--allow_non_sana_checkpoint", action="store_true",
                        help="允许使用看起来非 Sana 路线的 checkpoint（默认关闭）")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    target_dtype = dtype_map[args.dtype]
    if args.device == "cpu" and target_dtype != torch.float32:
        print("[WARN] CPU 下自动使用 float32")
        target_dtype = torch.float32
    vae_dtype = _resolve_vae_dtype(args.vae_dtype, target_dtype, args.device)

    _sanitize_meta_env()
    _force_default_device_cpu()
    print(f"[LOAD] default_device_is_meta={_is_meta_default_device()}")
    ckpt = find_newest_checkpoint(args.checkpoint_path)
    print(f"[LOAD] resolved checkpoint: {ckpt}")
    cfg = _read_checkpoint_config(ckpt)
    _check_qwen3_sana(cfg)
    _validate_checkpoint_config(cfg, allow_non_sana_checkpoint=args.allow_non_sana_checkpoint)
    pipeline = _try_load_pipeline(
        ckpt,
        target_dtype,
        allow_mismatched_sizes=args.allow_mismatched_sizes,
    )
    pipeline = pipeline.to(device=args.device, dtype=target_dtype)
    if hasattr(pipeline, "vae") and pipeline.vae is not None and hasattr(pipeline.vae, "to"):
        pipeline.vae = pipeline.vae.to(device=args.device, dtype=vae_dtype)
    pipeline.eval()
    print(f"[READY] device={args.device}, dtype={target_dtype}, vae_dtype={vae_dtype}, allow_mismatched_sizes={args.allow_mismatched_sizes}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"demo_{ts}"
    run_demo(
        pipeline=pipeline,
        output_dir=out_dir,
        ref_image_path=args.ref_image if args.ref_image else None,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
    )
    print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
