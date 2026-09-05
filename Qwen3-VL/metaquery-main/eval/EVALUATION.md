# Evaluation

## General Instructions

We provide sample evaluation scripts for the following datasets:
- COCO FID
- MJHQ-30k FID
- ImageNet Reconstruction
- GenEval
- DPG Bench
- CommonsenseT2I
- WISE

For COCO, MJHQ, and ImageNet Reconstruction, we provide the sample scripts to generate images. The scripts has arguments `start_idx` and `end_idx` to specify the range of the dataset to evaluate, users can use it for multiprocessing sampling on multiple GPUs. After sampling, users can run the eval scripts to get the numbers on single GPU.

For GenEval, DPG Bench, CommonsenseT2I, and WISE, we only provide the sample scripts to generate images. Users can use the corresponding eval scripts in these repos to get the numbers.

## COCO
The dataset will be automatically downloaded from [here](https://huggingface.co/datasets/sayakpaul/coco-30-val-2014) into the `dataset_folder`.
```bash
python sample_coco.py \
    --dataset_folder /path/to/cache_coco_dataset \
    --start_idx 0 \
    --end_idx -1 \
    --output_dir /path/to/output \
    --checkpoint_path /path/to/checkpoint \
    --guidance_scale 3.0 \
    --batch_size 1 \
    --num_inference_steps 30 \
```

```bash
python eval_coco.py \
    --dataset_folder /path/to/cache_coco_dataset \
    --image_folder /path/to/output \
```

## MJHQ

The dataset need to be manually downloaded from [here](https://huggingface.co/datasets/sayakpaul/MJHQ-30K):
```bash
cd /path/to/mjhq_dataset
git clone https://huggingface.co/datasets/playgroundai/MJHQ-30K
unzip mjhq30k_imgs.zip
```

```bash
python sample_mjhq.py \
    --dataset_folder /path/to/mjhq_dataset/MJHQ-30K \
    --start_idx 0 \
    --end_idx -1 \
    --output_dir /path/to/output \
    --checkpoint_path /path/to/checkpoint \
    --guidance_scale 3.0 \
    --batch_size 1 \
    --num_inference_steps 30 \
```

```bash
python eval_mjhq.py \
    --dataset_folder /path/to/mjhq_dataset/MJHQ-30K \
    --image_folder /path/to/output \
```

## ImageNet Reconstruction
The dataset will be automatically downloaded from [here](https://huggingface.co/datasets/ILSVRC/imagenet-1k) into the `dataset_folder`.
```bash
python sample_reconstruction.py \
    --dataset_folder /path/to/cache_imagenet_dataset \
    --start_idx 0 \
    --end_idx -1 \
    --output_dir /path/to/output \
    --checkpoint_path /path/to/checkpoint \
    --guidance_scale 3.0 \
    --image_guidance_scale 3.0 \
    --batch_size 1 \
    --num_inference_steps 30 \
```

```bash
python eval_reconstruction.py \
    --dataset_folder /path/to/cache_imagenet_dataset \
    --image_folder /path/to/output \
```

## GenEval
The dataset will be automatically downloaded from [here](https://github.com/djghosh13/geneval/blob/main/prompts/evaluation_metadata.jsonl) into the `dataset_file`.
```bash
python sample_geneval.py \
    --dataset_file /path/to/geneval_dataset/evaluation_metadata.jsonl \
    --start_idx 0 \
    --end_idx -1 \
    --output_dir /path/to/output \
    --checkpoint_path /path/to/checkpoint \
    --guidance_scale 7.5 \
    --num_inference_steps 30 \
    --seed 42 \
```

For evaluation, users can use the corresponding eval scripts in [here](https://github.com/djghosh13/geneval).

## DPG Bench

The dataset need to be manually downloaded from [here](https://github.com/TencentQQGYLab/ELLA/):
```bash
cd /path/to/dpg_bench_dataset
git clone https://github.com/TencentQQGYLab/ELLA.git
```

```bash
python sample_dpg.py \
    --dataset_folder /path/to/dpg_bench_dataset/ELLA/dpg_bench/prompts \
    --start_idx 0 \
    --end_idx -1 \
    --output_dir /path/to/output \
    --checkpoint_path /path/to/checkpoint \
    --guidance_scale 7.5 \
    --batch_size 1 \
    --num_inference_steps 30 \
    --seed 42 \
```

For evaluation, users can use the corresponding eval scripts in [here](https://github.com/TencentQQGYLab/ELLA).

## CommonsenseT2I

The dataset will be automatically downloaded from [here](https://huggingface.co/datasets/CommonsenseT2I/CommonsensenT2I) into the `dataset_folder`.
```bash
python sample_commonsenset2i.py \
    --dataset_folder /path/to/cache_commonsense_t2i_dataset \
    --start_idx 0 \
    --end_idx -1 \
    --output_dir /path/to/output \
    --checkpoint_path /path/to/checkpoint \
    --guidance_scale 7.5 \
    --num_inference_steps 30 \
    --seed 42 \
```

For evaluation, users can use the corresponding eval scripts in [here](https://github.com/CommonsenseT2I/CommonsenseT2I).

## WISE

The dataset need to be manually downloaded from [here](https://github.com/PKU-YuanGroup/WISE):
```bash
cd /path/to/wise_dataset
git clone https://github.com/PKU-YuanGroup/WISE.git
```

```bash
python sample_wise.py \
    --dataset_folder /path/to/wise_dataset/WISE/data \
    --start_idx 0 \
    --end_idx -1 \
    --output_dir /path/to/output \
    --checkpoint_path /path/to/checkpoint \
    --guidance_scale 7.5 \
    --num_inference_steps 30 \
    --seed 42 \
```

For evaluation, users can use the corresponding eval scripts in [here](https://github.com/PKU-YuanGroup/WISE).
