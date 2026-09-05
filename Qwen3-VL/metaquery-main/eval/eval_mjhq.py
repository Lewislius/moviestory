# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from argparse import ArgumentParser
import torch
from glob import glob
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import v2
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


preprocess_image = v2.Compose(
    [
        v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(256),
        v2.PILToTensor(),
    ]
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance()
    fid = fid.to(device)
    fid.eval()

    gt_images = SimpleImageDataset(
        image_paths=glob(os.path.join(args.dataset_folder, "*", "*")),
        transform=preprocess_image,
    )

    print("Number of real images: ", len(gt_images))
    gt_images_loader = torch.utils.data.DataLoader(
        gt_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=lambda x: torch.stack(x, dim=0),
    )

    # Process real images
    for batch in tqdm(gt_images_loader):
        batch = batch.to(device)
        fid.update(batch, real=True)

    # Load generated images using the custom dataset
    generated_images = SimpleImageDataset(
        image_paths=glob(os.path.join(args.image_folder, "*.jpg")),
        transform=preprocess_image,
    )
    print("Number of fake images: ", len(generated_images))
    generated_images_loader = torch.utils.data.DataLoader(
        generated_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=lambda x: torch.stack(x, dim=0),
    )

    # Process generated images
    for batch in tqdm(generated_images_loader):
        batch = batch.to(device)
        fid.update(batch, real=False)

    fid_score = fid.compute().item()
    print("FID: ", fid_score)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/path/to/dataset_folder")
    parser.add_argument("--image_folder", type=str, default="/path/to/image_folder")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()
    main(args)
