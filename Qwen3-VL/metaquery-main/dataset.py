# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datasets import load_dataset, load_from_disk, Image, Dataset
import PIL
import io
import torch
from torchvision.transforms import v2
import random
from torch.utils.data.dataset import ConcatDataset
from functools import partial


def delete_keys_except(batch, except_keys):
    keys_to_delete = [key for key in list(batch.keys()) if key not in except_keys]
    for key in keys_to_delete:
        del batch[key]
    return batch


def _i2i_process_fn(batch, target_transform):
    images = batch["image"]
    captions = ["" for _ in range(len(images))]
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(
                io.BytesIO(images[i]["bytes"])
                if images[i]["bytes"] is not None
                else images[i]["path"]
            ).convert("RGB")
        except:
            images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    rand_probs = torch.rand((len(images), 1))
    null_image_mask = rand_probs <= 0.1
    images = [
        (
            PIL.Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(images)
    ]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in images
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def i2i_eval_process_fn(batch):
    images = batch["image"]
    captions = ["" for _ in range(len(images))]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in images
    ]
    delete_keys_except(batch, ["input_images", "caption"])
    return batch


def _t2i_process_fn(batch, target_transform):
    images = batch["image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption for caption in captions]
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(
                io.BytesIO(images[i]["bytes"])
                if images[i]["bytes"] is not None
                else images[i]["path"]
            ).convert("RGB")
        except:
            images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    rand_probs = torch.rand((len(images), 1))
    null_caption_mask = rand_probs < 0.1
    captions = [
        caption if not null_caption_mask[i] else ""
        for i, caption in enumerate(captions)
    ]
    batch["caption"] = captions
    delete_keys_except(batch, ["target", "caption"])
    return batch


def t2i_eval_process_fn(batch):
    captions = batch["caption"]
    batch["caption"] = captions
    delete_keys_except(batch, ["caption"])
    return batch


def _inst_process_fn(batch, target_transform):
    source_images = batch["source_images"]
    caption = batch["caption"]
    rand_probs = torch.rand((len(batch["target_image"]), 1))
    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)
    caption = [
        caption if not null_caption_mask[i] else "" for i, caption in enumerate(caption)
    ]
    source_images = (
        [
            (
                image
                if not null_image_mask[i]
                else [PIL.Image.new("RGB", (img.width, img.height)) for img in image]
            )
            for i, image in enumerate(source_images)
        ]
        if source_images is not None
        else None
    )
    batch["caption"], batch["input_images"] = caption, source_images
    batch["target"] = [
        target_transform(img.convert("RGB")) for img in batch["target_image"]
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def inst_eval_process_fn(batch):
    source_images = batch["source_images"]
    caption = batch["caption"]
    batch["caption"], batch["input_images"] = caption, source_images
    delete_keys_except(batch, ["caption", "input_images"])
    return batch


def _editing_process_fn(batch, target_transform, ground_truth_transform):
    source_images = batch["source_image"]
    target_images = batch["target_image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption[-1] for caption in captions]
    for i in range(len(source_images)):
        try:
            source_images[i] = PIL.Image.open(
                io.BytesIO(source_images[i]["bytes"])
                if source_images[i]["bytes"] is not None
                else source_images[i]["path"]
            ).convert("RGB")
            target_images[i] = PIL.Image.open(
                io.BytesIO(target_images[i]["bytes"])
                if target_images[i]["bytes"] is not None
                else target_images[i]["path"]
            ).convert("RGB")
        except:
            source_images[i] = None
            target_images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None
        for image in target_images
    ]
    rand_probs = torch.rand((len(target_images), 1))
    null_image_mask = rand_probs <= 0.1
    source_images = [
        (
            PIL.Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(source_images)
    ]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in source_images
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def editing_eval_process_fn(batch):
    source_images = batch["source_image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption[-1] for caption in captions]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in source_images
    ]
    delete_keys_except(batch, ["input_images", "caption"])
    return batch


def _collate_fn(batch, tokenize_func, tokenizer):
    none_idx = [i for i, example in enumerate(batch) if example["target"] is None]
    if len(none_idx) > 0:
        batch = [example for i, example in enumerate(batch) if i not in none_idx]
    return_dict = {"target": torch.stack([example["target"] for example in batch])}
    input_images = [
        example["input_images"] if "input_images" in example else None
        for example in batch
    ]

    if any(input_images):
        (
            return_dict["input_ids"],
            return_dict["attention_mask"],
            return_dict["pixel_values"],
            return_dict["image_sizes"],
        ) = tokenize_func(
            tokenizer, [example["caption"] for example in batch], input_images
        )
    else:
        return_dict["input_ids"], return_dict["attention_mask"] = tokenize_func(
            tokenizer, [example["caption"] for example in batch]
        )
    return return_dict


def _get_num_samples(subset_value, is_test=False):
    """根据数据子集比例值计算需要的样本数。

    Args:
        subset_value: yaml 配置中的数据集值，如 0.12 表示 12万条，-1 表示全量
        is_test: 是否为 test 模式
    Returns:
        需要的样本数，None 表示全量
    """
    if is_test:
        return 10000
    if subset_value > 0:
        return int(subset_value * 1000000)
    return None


def _load_dataset_efficient(dataset_name, cache_dir, num_proc, num_samples=None):
    """高效加载数据集。

    当需要子集时，使用 streaming 模式只下载所需的数据分片，
    避免下载完整数据集（如 CC12M 全量 2176 个分片 ~300GB）。
    下载后缓存到本地磁盘，重复运行时直接加载缓存。

    Args:
        dataset_name: HuggingFace 数据集名称
        cache_dir: 缓存目录
        num_proc: 并行加载进程数
        num_samples: 需要的样本数，None 表示全量
    Returns:
        datasets.Dataset 对象
    """
    if num_samples is None:
        # 全量加载：正常流程
        print(f"[数据集] {dataset_name}: 加载全量数据")
        return load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            split="train",
            num_proc=num_proc,
        )

    # 子集加载：使用 streaming 模式，只下载所需的数据分片
    safe_name = dataset_name.replace("/", "__")
    subset_dir = os.path.join(cache_dir, f"{safe_name}_subset_{num_samples}")

    # 检查本地缓存是否已存在
    if os.path.exists(subset_dir):
        print(f"[数据集] {dataset_name}: 从本地缓存加载 ({subset_dir})")
        return load_from_disk(subset_dir)

    # 使用 streaming 模式仅下载所需数据
    print(f"[数据集] {dataset_name}: 使用 streaming 模式下载前 {num_samples} 条")
    print(f"[数据集] (只下载包含所需数据的分片，不会下载完整数据集)")

    stream = load_dataset(dataset_name, split="train", streaming=True)
    count = 0

    def take_generator():
        nonlocal count
        for item in stream:
            if count >= num_samples:
                break
            count += 1
            if count % 10000 == 0:
                print(f"[数据集]   已下载 {count}/{num_samples} 条...")
            yield item

    dataset = Dataset.from_generator(take_generator)
    dataset.save_to_disk(subset_dir)
    print(f"[数据集] {dataset_name}: 子集已缓存到 {subset_dir} ({len(dataset)} 条)")
    return dataset


def get_train_datasets(data_args, training_args, model_args, tokenize_func, tokenizer):
    train_datasets = {}
    is_test = (training_args.run_name == "test")

    if "cc12m_i2i" in data_args.train_datasets:
        subset_val = data_args.train_datasets["cc12m_i2i"]
        num_samples = _get_num_samples(subset_val, is_test)

        train_dataset = _load_dataset_efficient(
            "pixparse/cc12m-wds",
            cache_dir=training_args.data_dir,
            num_proc=training_args.datasets_num_proc,
            num_samples=num_samples,
        )

        if num_samples is not None and not is_test:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)

        # 列名统一（streaming 缓存和原始加载均适用）
        if "jpg" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("jpg", "image")
        if "txt" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns(
            [
                col
                for col in train_dataset.column_names
                if col not in ["image", "caption"]
            ]
        )
        print(f"[数据集] cc12m_i2i: 就绪，共 {len(train_dataset)} 条")
        train_datasets["cc12m_i2i"] = train_dataset

    if "cc12m_t2i" in data_args.train_datasets:
        subset_val = data_args.train_datasets["cc12m_t2i"]
        num_samples = _get_num_samples(subset_val, is_test)

        train_dataset = _load_dataset_efficient(
            "pixparse/cc12m-wds",
            cache_dir=training_args.data_dir,
            num_proc=training_args.datasets_num_proc,
            num_samples=num_samples,
        )

        if num_samples is not None and not is_test:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)

        if "jpg" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("jpg", "image")
        if "txt" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns(
            [
                col
                for col in train_dataset.column_names
                if col not in ["image", "caption"]
            ]
        )
        print(f"[数据集] cc12m_t2i: 就绪，共 {len(train_dataset)} 条")
        train_datasets["cc12m_t2i"] = train_dataset

    if "inst2m" in data_args.train_datasets:
        subset_val = data_args.train_datasets["inst2m"]
        num_samples = _get_num_samples(subset_val, is_test)

        train_dataset = _load_dataset_efficient(
            "xcpan/MetaQuery_Instruct_2.4M_512res",
            cache_dir=training_args.data_dir,
            num_proc=training_args.datasets_num_proc,
            num_samples=num_samples,
        )

        if num_samples is not None and not is_test:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)

        if "prompt" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("prompt", "caption")
        print(f"[数据集] inst2m: 就绪，共 {len(train_dataset)} 条")
        train_datasets["inst2m"] = train_dataset

    if "ominiedit" in data_args.train_datasets:
        subset_val = data_args.train_datasets["ominiedit"]
        num_samples = _get_num_samples(subset_val, is_test)

        train_dataset = _load_dataset_efficient(
            "TIGER-Lab/OmniEdit-Filtered-1.2M",
            cache_dir=training_args.data_dir,
            num_proc=training_args.datasets_num_proc,
            num_samples=num_samples,
        )

        if num_samples is not None and not is_test:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)

        if "src_img" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("src_img", "source_image")
        if "edited_img" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("edited_img", "target_image")
        if "edited_prompt_list" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("edited_prompt_list", "caption")
        train_dataset = train_dataset.remove_columns(
            [
                col
                for col in train_dataset.column_names
                if col not in ["source_image", "target_image", "caption"]
            ]
        )
        print(f"[数据集] ominiedit: 就绪，共 {len(train_dataset)} 条")
        train_datasets["ominiedit"] = train_dataset

    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )

    ground_truth_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
        ]
    )

    i2i_process_fn = partial(_i2i_process_fn, target_transform=target_transform)
    t2i_process_fn = partial(_t2i_process_fn, target_transform=target_transform)
    inst_process_fn = partial(_inst_process_fn, target_transform=target_transform)
    editing_process_fn = partial(
        _editing_process_fn,
        target_transform=target_transform,
        ground_truth_transform=ground_truth_transform,
    )
    collate_fn = partial(_collate_fn, tokenize_func=tokenize_func, tokenizer=tokenizer)

    eval_dataset = train_datasets[data_args.eval_dataset].select(
        range(training_args.world_size)
    )
    if "source_images" in eval_dataset.column_names:
        eval_dataset = eval_dataset.cast_column("target_image", Image(decode=True))
    elif "source_image" in eval_dataset.column_names:
        eval_dataset = eval_dataset.cast_column("source_image", Image(decode=True))
        eval_dataset = eval_dataset.cast_column("target_image", Image(decode=True))
    else:
        eval_dataset = eval_dataset.cast_column("image", Image(decode=True))
    gt_images = (
        eval_dataset["target_image"]
        if "target_image" in eval_dataset.column_names
        else eval_dataset["image"]
    )
    gt_images = [ground_truth_transform(image.convert("RGB")) for image in gt_images]

    if data_args.eval_dataset in ["cc12m_i2i"]:
        eval_dataset.set_transform(i2i_eval_process_fn)
    elif data_args.eval_dataset in ["cc12m_t2i"]:
        eval_dataset.set_transform(t2i_eval_process_fn)
    elif data_args.eval_dataset in ["inst2m"]:
        eval_dataset.set_transform(inst_eval_process_fn)
    elif data_args.eval_dataset in ["ominiedit"]:
        eval_dataset.set_transform(editing_eval_process_fn)
    else:
        raise ValueError(f"Unknown eval_dataset: {data_args.eval_dataset}")

    for dataset_name, train_dataset in train_datasets.items():
        if dataset_name in ["cc12m_i2i"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "image", Image(decode=False)
            )
            train_datasets[dataset_name].set_transform(i2i_process_fn)
        elif dataset_name in ["cc12m_t2i"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "image", Image(decode=False)
            )
            train_datasets[dataset_name].set_transform(t2i_process_fn)
        elif dataset_name in ["inst2m"]:
            train_datasets[dataset_name].set_transform(inst_process_fn)
        elif dataset_name in ["ominiedit"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "source_image", Image(decode=False)
            )
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "target_image", Image(decode=False)
            )
            train_datasets[dataset_name].set_transform(editing_process_fn)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        train_datasets[dataset_name] = train_datasets[dataset_name].shuffle(
            seed=training_args.data_seed
        )

    # if more than one dataset in the dict, concatenate them
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(list(train_datasets.values()))
    else:
        train_dataset = train_datasets[list(train_datasets.keys())[0]]

    return train_dataset, eval_dataset, gt_images, collate_fn
