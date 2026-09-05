# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from argparse import ArgumentParser
import torch
from datasets import load_dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    LearnedPerceptualImagePatchSimilarity,
)
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
from glob import glob

preprocess_image = v2.Compose(
    [
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(256),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

sample_preprocess_image = v2.Compose(
    [
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(256),
    ]
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid = fid.to(device)
    fid.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    gt_images = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        cache_dir=args.dataset_folder,
        trust_remote_code=True,
        num_proc=12,
    )
    gt_images = gt_images.remove_columns("label")
    print("Number of real images: ", len(gt_images))

    generated_images = sorted(
        glob(os.path.join(args.image_folder, "*")),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    print("Number of fake images: ", len(generated_images))

    assert len(gt_images) == len(
        generated_images
    ), f"Number of real images ({len(gt_images)}) and fake images ({len(generated_images)}) must be the same"

    for i in tqdm(range(0, len(gt_images), args.batch_size), desc="Evaluating metrics"):
        start_idx = i
        end_idx = min(i + args.batch_size, len(gt_images))
        real_images = torch.stack(
            [
                preprocess_image(img)
                for img in gt_images.select(range(start_idx, end_idx))["image"]
            ],
            dim=0,
        ).to(device)
        fake_images = torch.stack(
            [
                preprocess_image(Image.open(generated_images[j]).convert("RGB"))
                for j in range(start_idx, end_idx)
            ],
            dim=0,
        ).to(device)

        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
        ssim_metric.update(fake_images, real_images)
        psnr_metric.update(fake_images, real_images)
        lpips_metric.update(fake_images, real_images)

    print("Final Metrics:")
    print("FID: ", fid.compute().item())
    print("SSIM: ", ssim_metric.compute().item())
    print("PSNR: ", psnr_metric.compute().item())
    print("LPIPS: ", lpips_metric.compute().item())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/path/to/dataset_folder")
    parser.add_argument("--image_folder", type=str, default="/path/to/image_folder")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()
    main(args)
