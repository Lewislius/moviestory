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
from tqdm import tqdm
from trainer_utils import find_newest_checkpoint, seed_everything
import json


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

    if args.category == "all":
        categories = [
            "cultural_common_sense",
            "natural_science",
            "spatio-temporal_reasoning",
        ]
    else:
        categories = [args.category]
    for category in categories:
        os.makedirs(os.path.join(args.output_dir, category), exist_ok=True)
        if args.rewrite:
            print(
                f"Using rewrite dataset: {args.dataset_folder}/{category}_rewrite.json"
            )
            with open(
                os.path.join(args.dataset_folder, f"{category}_rewrite.json"), "r"
            ) as f:
                dataset = json.load(f)
        else:
            print(f"Using original dataset: {args.dataset_folder}/{category}.json")
            with open(os.path.join(args.dataset_folder, f"{category}.json"), "r") as f:
                dataset = json.load(f)

        for data in tqdm(dataset):
            seed_everything(args.seed)
            image = sample_fn(pipeline, data["Prompt"], args)[0]
            image.save(
                os.path.join(args.output_dir, category, f"{data['prompt_id']}.png")
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/path/to/dataset_folder")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="/path/to/output_dir")
    parser.add_argument("--checkpoint_path", type=str, default="/path/to/checkpoint_path")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rewrite", action="store_true", dest="rewrite")
    args = parser.parse_args()
    main(args)
