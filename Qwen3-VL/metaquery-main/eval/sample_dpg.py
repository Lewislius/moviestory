# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from pipeline_metaquery import MetaQueryPipeline
import torch
from tqdm import tqdm
from trainer_utils import find_newest_checkpoint, seed_everything
import glob


def sample_metaquery(pipeline, prompt, args):
    return pipeline(
        caption=prompt,
        image=None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=4,
        enable_progress_bar=False,
    ).images


def main(args):
    args.checkpoint_path = find_newest_checkpoint(args.checkpoint_path)
    pipeline = MetaQueryPipeline.from_pretrained(
        args.checkpoint_path,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    sample_fn = sample_metaquery
    os.makedirs(args.output_dir, exist_ok=True)

    data_files = sorted(glob.glob(os.path.join(args.dataset_folder, "*.txt")))
    data_files = data_files[args.start_idx : args.end_idx]

    dataset = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            dataset.append((data_file.split("/")[-1].split(".")[0], f.read().strip()))

    for file_name, prompt in tqdm(dataset):
        seed_everything(args.seed)
        images = sample_fn(pipeline, [prompt], args)

        grid_size = 2
        try:
            lanczos = Image.Resampling.LANCZOS
        except AttributeError:
            lanczos = Image.LANCZOS

        images = [
            img.resize((512, 512), resample=lanczos) if img.size != (512, 512) else img
            for img in images
        ]
        grid = Image.new("RGB", (512 * grid_size, 512 * grid_size))

        for idx, img in enumerate(images):
            x = (idx % grid_size) * 512
            y = (idx // grid_size) * 512
            grid.paste(img, (x, y))

        grid.save(f"{args.output_dir}/{file_name}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/path/to/dataset_folder")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    parser.add_argument("--output_dir", type=str, default="/path/to/output_dir")
    parser.add_argument("--checkpoint_path", type=str, default="/path/to/checkpoint_path")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
