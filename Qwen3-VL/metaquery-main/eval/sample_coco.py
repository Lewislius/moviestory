# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from pipeline_metaquery import MetaQueryPipeline
import torch
from datasets import load_dataset
from tqdm import tqdm
from trainer_utils import find_newest_checkpoint


def sample_metaquery(pipeline, prompt, args):
    return pipeline(
        caption=prompt,
        image=None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
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

    dataset = load_dataset(
        "sayakpaul/coco-30-val-2014",
        split="train",
        cache_dir=args.dataset_folder,
        trust_remote_code=True,
        num_proc=12,
    )
    dataset = dataset.select(
        range(args.start_idx, args.end_idx if args.end_idx != -1 else len(dataset))
    )

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        data = dataset[i : i + args.batch_size]
        prompt = data["caption"]
        images = sample_fn(pipeline, prompt, args)

        for j, image in enumerate(images):
            image.save(f"{args.output_dir}/{args.start_idx+i+j:05d}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/path/to/dataset_folder")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    parser.add_argument("--output_dir", type=str, default="/path/to/output_dir")
    parser.add_argument("--checkpoint_path", type=str, default="/path/to/checkpoint_path")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
