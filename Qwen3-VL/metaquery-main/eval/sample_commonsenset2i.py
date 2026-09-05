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
from PIL import Image
from datasets import load_dataset


def sample_metaquery(pipeline, prompt, args, negative_prompt=""):
    return pipeline(
        caption=prompt,
        image=None,
        negative_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=4,
        enable_progress_bar=False,
    ).images


def get_grid(imgs):
    rows, cols = 2, 2
    w, h = imgs[0].size
    grid_img = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(imgs):
        grid_img.paste(img, ((idx % cols) * w, (idx // cols) * h))
    return grid_img


def generate_images(args, pipe, sample_fn, prompts, prompt_name, neg_prompts=None):
    for index, (global_id, prompt) in tqdm(enumerate(prompts)):
        seed_everything(args.seed)
        grid_image_path = (
            f"{args.output_dir}/{prompt_name}/{str(global_id + 1).zfill(4)}.jpg"
        )
        print(f"The prompt for image {global_id + 1} is: {prompt}")

        if neg_prompts is None:
            images = sample_fn(pipe, prompt, args)
        else:
            neg_prompt = neg_prompts[index][1]
            images = sample_fn(pipe, prompt, args, neg_prompt)
        for j, image in enumerate(images):
            image.save(
                f"{args.output_dir}/{prompt_name}/original/{str(global_id + 1).zfill(4)}-{j + 1}.jpg"
            )

        grid_img = get_grid(images)
        grid_img.save(grid_image_path)


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
    os.makedirs(f"{args.output_dir}/prompt1_img/original", exist_ok=True)
    os.makedirs(f"{args.output_dir}/prompt2_img/original", exist_ok=True)

    dataset = load_dataset(
        "CommonsenseT2I/CommonsensenT2I",
        cache_dir=args.dataset_folder,
        split="train",
        trust_remote_code=True,
        num_proc=12,
    )
    prompts1 = [(i, d["prompt1"]) for i, d in enumerate(dataset)][
        args.start_idx : args.end_idx
    ]
    prompts2 = [(i, d["prompt2"]) for i, d in enumerate(dataset)][
        args.start_idx : args.end_idx
    ]

    if args.use_negative_prompt:
        generate_images(args, pipeline, sample_fn, prompts1, "prompt1_img", prompts2)
        generate_images(args, pipeline, sample_fn, prompts2, "prompt2_img", prompts1)
    else:
        generate_images(args, pipeline, sample_fn, prompts1, "prompt1_img")
        generate_images(args, pipeline, sample_fn, prompts2, "prompt2_img")


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
    parser.add_argument(
        "--use_negative_prompt", action="store_true", dest="use_negative_prompt"
    )
    args = parser.parse_args()
    main(args)
