# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for Qwen3-VL by the project contributors.
#
# This script curates MetaQuery instruction-tuning data using Qwen3-VL
# instead of Qwen2-VL. The pipeline:
# 1. Load image-caption pairs from mmc4 web corpus
# 2. Use SigLIP to cluster semantically similar captions (max-clique grouping)
# 3. Select target image (lowest image similarity to others)
# 4. Use Qwen3-VL to generate instruction prompts via few-shot demonstration
# 5. Save as HuggingFace Dataset

import json
import PIL
import requests
import torch
from tqdm import tqdm
import networkx as nx
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModel
from datasets.features import Sequence, Image
import argparse
import numpy as np
import os


# get all the images from assets folder and save it into a dict
assets_dict = {}
assets_folder = "assets"
if os.path.exists(assets_folder):
    for file in os.listdir(assets_folder):
        try:
            assets_dict[file.split(".")[0]] = PIL.Image.open(
                os.path.join(assets_folder, file)
            ).convert("RGB")
        except Exception:
            pass


FEW_SHOT_SYSTEM_PROMPT = """Based on the provided of one or multiple source images, one target image, and their captions, create an interesting text prompt which can be used with the source images to generate the target image.

This prompt should include:
1) one general and unspecific similarity shared with the source images (same jersey top, similar axe, similar building, etc).
2) all differences that only the target image has.

This prompt should NOT include:
1) any specific details that would allow generating the target image independently without referencing the source images.

Remember the prompt should be concise and short. The generation has to be done by combining the source images and text prompt, not only the prompt itself."""


def build_few_shot_messages(source_images, source_captions, target_image, target_caption):
    """Build the few-shot prompt messages for Qwen3-VL."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FEW_SHOT_SYSTEM_PROMPT},
                {
                    "type": "text",
                    "text": "Now first think then generate a prompt following the demonstration\nSource Images: CAPTION ["
                    + ",".join([f'"{cap}"' for cap in source_captions])
                    + "]. PIXELS [",
                },
                *[
                    {"type": "image", "image": source_image}
                    for source_image in source_images
                ],
                {
                    "type": "text",
                    "text": f']\nTarget Image: CAPTION ["{target_caption}"]. PIXELS [',
                },
                {"type": "image", "image": target_image},
                {"type": "text", "text": "]\n"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Think: "}],
        },
    ]
    return messages


@torch.inference_mode()
def main(file_name):
    with open(file_name, "r") as f:
        dataset = f.readlines()
    all_grouped_pairs = []
    for data in tqdm(dataset, leave=True):
        data = json.loads(data)
        image_caption_pairs = []
        for image_info in data["image_info"]:
            image_url = image_info["raw_url"]
            try:
                image = PIL.Image.open(
                    requests.get(image_url, stream=True, timeout=5).raw
                ).convert("RGB")
                if image.size[0] < 256 or image.size[1] < 256:
                    continue
            except Exception:
                continue
            image_caption_pairs.append(
                (
                    image,
                    data["text_list"][image_info["matched_text_index"]]
                    .replace("\n", " ")
                    .replace("\t", " ")
                    .replace('"', "")
                    .strip(),
                )
            )
        if len(image_caption_pairs) < 2:
            continue

        try:
            # calculate the similarity matrix between all captions using siglip
            captions = [caption for _, caption in image_caption_pairs]
            inputs = siglip_processor(
                text=captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to("cuda")
            with torch.no_grad():
                caption_embeddings = siglip.get_text_features(**inputs)

            caption_embeddings = caption_embeddings / caption_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )

            similarity_matrix = (
                torch.matmul(caption_embeddings, caption_embeddings.t())
                * siglip.logit_scale.exp()
                + siglip.logit_bias
            )
            similarities = similarity_matrix.detach().cpu().float().numpy()

            # Construct adjacency matrix with threshold 50
            adjacency = (similarities > 50).astype(int)
            G = nx.from_numpy_array(adjacency)

            groups = []
            while G.number_of_nodes() > 0:
                cliques = list(nx.find_cliques(G))
                cliques.sort(key=lambda x: len(x), reverse=True)
                largest_clique = cliques[0]
                groups.append(largest_clique)
                G.remove_nodes_from(largest_clique)

            grouped_pairs = [
                [image_caption_pairs[idx] for idx in group]
                for group in groups
                if len(group) > 1
            ]
            if len(grouped_pairs) == 0:
                continue
            # split groups with more than 6 captions
            splited_grouped_captions = []
            for group in grouped_pairs:
                if len(group) > 6:
                    if len(group) % 6 < 2:
                        splited_grouped_captions.append(group[-2:])
                        group = group[:-2]
                    for i in range(0, len(group), 6):
                        splited_grouped_captions.append(group[i : i + 6])
                else:
                    splited_grouped_captions.append(group)

            # for each group, find the target image (lowest similarity to others)
            for group in splited_grouped_captions:
                images = [image for image, _ in group]
                inputs = siglip_processor(
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                ).to("cuda", torch.bfloat16)
                image_embeddings = siglip.get_image_features(**inputs)
                image_embeddings = image_embeddings / image_embeddings.norm(
                    p=2, dim=-1, keepdim=True
                )
                similarity_matrix = (
                    torch.matmul(image_embeddings, image_embeddings.t())
                    * siglip.logit_scale.exp()
                    + siglip.logit_bias
                )
                similarities = similarity_matrix.detach().cpu().float().numpy()
                min_similarity = np.min(similarities, axis=1)
                target_image_idx = np.argmin(min_similarity)
                source_images = [
                    images[i] for i in range(len(images)) if i != target_image_idx
                ]
                source_captions = [
                    group[i][1] for i in range(len(group)) if i != target_image_idx
                ]
                target_image = images[target_image_idx]
                target_caption = group[target_image_idx][1]

                # Use Qwen3-VL to generate instruction prompt
                messages = build_few_shot_messages(
                    source_images, source_captions, target_image, target_caption
                )

                # Process with Qwen3-VL
                text = qwen_vl_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Qwen3-VL uses process_vision_info from qwen_vl_utils
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = qwen_vl_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                generated_ids = qwen_vl_model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                prompt = qwen_vl_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                if "[" not in prompt or "]" not in prompt:
                    print(f"Skipping malformed prompt: {prompt}")
                    continue
                prompt = prompt.split("[")[1].split("]")[0]

                all_grouped_pairs.append(
                    (source_images, source_captions, prompt, target_image, target_caption)
                )
        except Exception as e:
            print(f"Error processing group: {e}")
            continue

    # Save as HuggingFace dataset
    dataset_dict = {
        "source_images": [group[0] for group in all_grouped_pairs],
        "source_captions": [group[1] for group in all_grouped_pairs],
        "prompt": [group[2] for group in all_grouped_pairs],
        "target_image": [group[3] for group in all_grouped_pairs],
        "target_caption": [group[4] for group in all_grouped_pairs],
    }
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("source_images", Sequence(Image(decode=True)))
    dataset = dataset.cast_column("target_image", Image(decode=True))
    output_path = os.path.join(args.output_dir, file_name.split("/")[-1].split("\\")[-1].split(".")[0])
    dataset.save_to_disk(output_path)
    print(f"Saved {len(all_grouped_pairs)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate MetaQuery instruction data using Qwen3-VL"
    )
    parser.add_argument(
        "--file_name", type=str, required=True,
        help="Path to mmc4 JSONL file (e.g., docs_shard_0_v2.jsonl)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./metaquery_instruct_qwen3vl",
        help="Output directory for the curated dataset"
    )
    parser.add_argument(
        "--qwen3vl_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model ID for prompt generation"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SigLIP model...")
    siglip = AutoModel.from_pretrained(
        "google/siglip-large-patch16-256", torch_dtype=torch.bfloat16
    )
    siglip.to("cuda")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-256")

    print(f"Loading Qwen3-VL model: {args.qwen3vl_model}...")
    qwen_vl_model = AutoModelForImageTextToText.from_pretrained(
        args.qwen3vl_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # Qwen3-VL uses patch_size=16, factor=28
    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28
    qwen_vl_processor = AutoProcessor.from_pretrained(
        args.qwen3vl_model, min_pixels=min_pixels, max_pixels=max_pixels
    )
    print("Models loaded. Starting data curation...")
    main(args.file_name.strip())
