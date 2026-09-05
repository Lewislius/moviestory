#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for Qwen3-VL by the project contributors.
#
# Inference script for MetaQuery + Qwen3-VL.
# Supports: text-to-image, image-to-image, instruction-guided generation.

import argparse
import os
import torch
import numpy as np
from PIL import Image

from pipeline_metaquery import MetaQueryPipeline
from trainer_utils import find_newest_checkpoint


def load_pipeline(checkpoint_path, device="cuda", dtype=torch.bfloat16):
    """Load the MetaQuery pipeline from a checkpoint."""
    ckpt = find_newest_checkpoint(checkpoint_path)
    print(f"Loading checkpoint: {ckpt}")
    pipeline = MetaQueryPipeline.from_pretrained(
        ckpt,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device=device, dtype=dtype)
    pipeline.eval()
    return pipeline


def text_to_image(
    pipeline,
    prompt,
    negative_prompt="",
    guidance_scale=4.5,
    num_inference_steps=30,
    num_images=1,
    seed=None,
    output_dir="outputs",
):
    """Generate images from text prompt."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    images = pipeline(
        caption=prompt,
        image=None,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        generator=generator,
        enable_progress_bar=True,
    ).images

    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"t2i_{i:04d}.png")
        img.save(path)
        print(f"Saved: {path}")
    return images


def image_to_image(
    pipeline,
    input_image_paths,
    prompt="",
    negative_prompt="",
    guidance_scale=4.5,
    image_guidance_scale=1.5,
    num_inference_steps=30,
    num_images=1,
    seed=None,
    output_dir="outputs",
):
    """Generate images from input images + optional text prompt."""
    input_images = []
    for path in input_image_paths:
        img = Image.open(path).convert("RGB")
        input_images.append(img)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    images = pipeline(
        caption=prompt,
        image=[input_images],
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        generator=generator,
        enable_progress_bar=True,
    ).images

    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"i2i_{i:04d}.png")
        img.save(path)
        print(f"Saved: {path}")
    return images


def main():
    parser = argparse.ArgumentParser(
        description="MetaQuery + Qwen3-VL Inference"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained checkpoint directory"
    )
    parser.add_argument(
        "--mode", type=str, default="t2i", choices=["t2i", "i2i"],
        help="Generation mode: t2i (text-to-image) or i2i (image-to-image)"
    )
    parser.add_argument(
        "--prompt", type=str, default="A beautiful sunset over the ocean",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="",
        help="Negative prompt"
    )
    parser.add_argument(
        "--input_images", nargs="*", default=None,
        help="Paths to input images (for i2i mode)"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=4.5,
        help="Text guidance scale"
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.5,
        help="Image guidance scale (for i2i mode)"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=30,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--num_images", type=int, default=1,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (None for random)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    args = parser.parse_args()

    pipeline = load_pipeline(args.checkpoint_path, device=args.device)

    if args.mode == "t2i":
        text_to_image(
            pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images=args.num_images,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.mode == "i2i":
        if not args.input_images:
            raise ValueError("--input_images is required for i2i mode")
        image_to_image(
            pipeline,
            input_image_paths=args.input_images,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images=args.num_images,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
