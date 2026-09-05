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

    if not os.path.exists(args.dataset_file):
        import urllib.request

        url = "https://raw.githubusercontent.com/djghosh13/geneval/refs/heads/main/prompts/evaluation_metadata.jsonl"
        print(f"Downloading prompts from {url}")
        urllib.request.urlretrieve(url, args.dataset_file)
    with open(args.dataset_file) as fp:
        dataset = [json.loads(line) for line in fp]
    dataset = dataset[args.start_idx : args.end_idx]

    for i, data in tqdm(enumerate(dataset)):
        seed_everything(args.seed)
        outpath = os.path.join(args.output_dir, f"{i:0>5}")
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(data, fp)

        prompt = data["prompt"]
        samples = sample_fn(pipeline, prompt, args)
        for j, sample in enumerate(samples):
            sample.save(os.path.join(sample_path, f"{j:05}.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="/path/to/dataset_file")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    parser.add_argument("--output_dir", type=str, default="/path/to/output_dir")
    parser.add_argument("--checkpoint_path", type=str, default="/path/to/checkpoint_path")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
