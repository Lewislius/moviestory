# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import PIL
import requests
import torch
from tqdm import tqdm
import networkx as nx
from datasets import Dataset

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info
from datasets.features import Sequence, Image
import argparse
import numpy as np
import os
from huggingface_hub import login


# get all the images from assets folder and save it into a dict
assets_dict = {}
assets_folder = "assets"
for file in os.listdir(assets_folder):
    assets_dict[file.split(".")[0]] = PIL.Image.open(
        os.path.join(assets_folder, file)
    ).convert("RGB")


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
            except:
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
            # calculate the similarity matrix between all captions in image_caption_pairs use siglip
            captions = [caption for _, caption in image_caption_pairs]
            inputs = siglip_processor(
                text=captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to("cuda")
            with torch.no_grad():
                caption_embeddings = siglip.get_text_features(**inputs)

            # Normalize the embeddings
            caption_embeddings = caption_embeddings / caption_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )

            # Calculate similarity matrix with logit scale
            similarity_matrix = (
                torch.matmul(caption_embeddings, caption_embeddings.t())
                * siglip.logit_scale.exp()
                + siglip.logit_bias
            )
            # Convert the Torch tensor to a NumPy array
            similarities = similarity_matrix.detach().cpu().float().numpy()

            # Construct the adjacency matrix with a threshold of 50
            adjacency = (similarities > 50).astype(int)

            # Create an undirected graph
            G = nx.from_numpy_array(adjacency)

            # Initialize the result list
            groups = []

            # Iterate to find the maximum cliques
            while G.number_of_nodes() > 0:
                # Find all the maximum cliques
                cliques = list(nx.find_cliques(G))
                # Sort the cliques by size in descending order
                cliques.sort(key=lambda x: len(x), reverse=True)
                # Select the largest clique
                largest_clique = cliques[0]
                # Add the largest clique to the result list
                groups.append(largest_clique)
                # Remove the grouped nodes from the graph
                G.remove_nodes_from(largest_clique)

            grouped_pairs = [
                [image_caption_pairs[idx] for idx in group]
                for group in groups
                if len(group) > 1
            ]
            if len(grouped_pairs) == 0:
                continue
            # split the group contains more than 6 captions into multiple groups
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
            # for each of the group, calculate the similarity matrix between all images use siglip
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
                # choose the one with the lowest similarity to the rest of the images as the target image, others are the source images
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
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Based on the provided of one or multiple source images, one target image, and their captions, create an interesting text prompt which can be used with the source images to generate the target image.

This prompt should include:
1) one general and unspecific similarity shared with the source images (same jersey top, similar axe, similar building, etc).
2) all differences that only the target image has.

This prompt should NOT include:
1) any specific details that would allow generating the target image independently without referencing the source images.

Remember the prompt should be concise and short. The generation has to be done by combining the source images and text prompt, not only the prompt itself.""",
                            },
                            {
                                "type": "text",
                                "text": """Source Images: CAPTION ["Blatche averaged 10.1 points and 5.3 rebounds per game in 2012-13 for the Nets, one of two seasons in which he did not miss a game."]. PIXELS [""",
                            },
                            {
                                "type": "image",
                                "image": assets_dict["nets0"],
                            },
                            {
                                "type": "text",
                                "text": """]
Target Image: CAPTION ["It is photo-matched to the Nets' 111-93 victory over the Portland Trail Blazers on March 27, 2013 at Rose Garden Arena in Portland, OR."]. PIXELS [""",
                            },
                            {
                                "type": "image",
                                "image": assets_dict["nets1"],
                            },
                            {
                                "type": "text",
                                "text": "]",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": """Think: Source image shows the upper part of the jersey top of Blatche, a basketball player in the Nets. The target image shows the complete front view of the same jersey top. Therefore, the prompt should focus on the content of image (same jersey top), but specify the different view (complete front view).
Prompt: [The complete front view of the same jersey top]""",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Source Images: CAPTION ["The recipe I would like to share this time is 'Sweet 'n' spicy Sriracha wings', with the recipe from taste.com.au magazine and using Marian Gasby's recipe."]. PIXELS [""",
                            },
                            {
                                "type": "image",
                                "image": assets_dict["wings0"],
                            },
                            {
                                "type": "text",
                                "text": """]
Target Image: CAPTION ["The recipe said to BBQ but I grilled in the oven for around 20 mins, saving excess marinade."]. PIXELS [""",
                            },
                            {
                                "type": "image",
                                "image": assets_dict["wings1"],
                            },
                            {
                                "type": "text",
                                "text": "]",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": """Think: The source image shows a close-up photo of the sweet 'n' spicy Sriracha wings. The target image shows all the steps of the recipe to cook the same dish. Therefore, the prompt should focus on the topic of the image (same dish), but specify the different content (all the steps of the recipe).
Prompt: [Show all the steps of the recipe to cook the same dish]""",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Source Images: CAPTION [ "Hand-forged Wooden Handle Axe | BUDK.com - Knives & Swords At The Lowest Prices!", "Crafted of one solid piece of iron that is 100% hand-forged, this axe also features a Sheesham wood handle."]. PIXELS [""",
                            },
                            {
                                "type": "image",
                                "image": assets_dict["axe0"],
                            },
                            {
                                "type": "image",
                                "image": assets_dict["axe1"],
                            },
                            {
                                "type": "text",
                                "text": """]
Target Image: CAPTION ["Travel back in time with this hand-forged axe."]. PIXELS [""",
                            },
                            {
                                "type": "image",
                                "image": assets_dict["axe2"],
                            },
                            {
                                "type": "text",
                                "text": "]",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": """Think: The source images are product photos of an axe against a white background. The target image shows a similar axe placed in front of a rough, weathered tree stump, with visible textures and uneven bark. The background features a green grassy area, indicating an outdoor setting. Therefore, the prompt should focus on the content of the image (similar axe), but specify the different background (rough, weathered tree stump, green grassy area).
Prompt: [Take an outdoor photo of a similar axe with a slightly faded wooden handle. The axe is positioned at an angle with blade embedded downward into a rough, weathered tree stump, with green grassy area in background]""",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Now first think then generate a prompt following the demonstration\nSource Images: CAPTION ["
                                + ",".join([f'"{cap}"' for cap in source_captions])
                                + "]. PIXELS [",
                            },
                            *[
                                {
                                    "type": "image",
                                    "image": source_image,
                                }
                                for source_image in source_images
                            ],
                            {
                                "type": "text",
                                "text": f"""]
Target Image: CAPTION ["{target_caption}"]. PIXELS [""",
                            },
                            {
                                "type": "image",
                                "image": target_image,
                            },
                            {"type": "text", "text": "]\n"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Think: ",
                            }
                        ],
                    },
                ]

            # Preparation for inference
            text = qwen_vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = qwen_vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output with adjusted hyperparameters
            generated_ids = qwen_vl_model.generate(
                **inputs,
                max_new_tokens=256,
            )
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
                print(prompt)
                continue
            prompt = prompt.split("[")[1].split("]")[0]

            all_grouped_pairs.append(
                (source_images, source_captions, prompt, target_image, target_caption)
            )
        except:
            continue
    # Prepare data for Hugging Face dataset
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

    dataset.save_to_disk(
        os.path.join(args.output_dir, file_name.split("/")[-1].split(".")[0])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name",
        type=str,
        default="/path/to/mmc4/docs_shard_0_v2.jsonl",
        help="File to process",
    )
    parser.add_argument("--output_dir", type=str, default="/path/to/metaquery_instruct")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    siglip = AutoModel.from_pretrained(
        "google/siglip-large-patch16-256", torch_dtype=torch.bfloat16
    )
    siglip.to("cuda")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-256")

    # # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    qwen_vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28
    qwen_vl_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )
    main(args.file_name.strip())
