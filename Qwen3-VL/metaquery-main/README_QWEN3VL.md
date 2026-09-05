# MetaQuery + Qwen3-VL: 完整集成指南

本项目将 [MetaQuery](https://github.com/facebookresearch/MetaQuery)（Meta 提出的跨模态迁移框架）与 **Qwen3-VL**（通义千问第三代视觉语言模型）相结合，实现基于 Qwen3-VL 的图像生成、编辑和指令引导生成。

## 架构概览

```
输入文本/图像
      │
      ▼
┌─────────────────┐
│   Qwen3-VL      │  ← 冻结的 MLLM 骨干网络
│  (2B/4B/8B)     │     提取多模态语义特征
└────────┬────────┘
         │ MetaQuery Tokens (256个)
         ▼
┌─────────────────┐
│  Connector       │  ← 可训练的 24层双向 Qwen2 Encoder
│  (Qwen2Encoder)  │     + Linear + GELU + Linear + RMSNorm
└────────┬────────┘
         │ 条件向量
         ▼
┌─────────────────┐
│  Sana 1.6B      │  ← 冻结/可训练的扩散模型
│  Transformer     │     Flow Matching 生成
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AutoencoderDC  │  ← 冻结的 VAE 解码器
│  (Sana VAE)     │
└────────┬────────┘
         │
         ▼
    生成图像 (512×512)
```

## 环境配置

### 方式一：Conda（推荐）

```bash
conda env create -f environment_qwen3vl.yml
conda activate metaquery-qwen3vl
```

### 方式二：Pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.52.0 accelerate diffusers datasets
pip install wandb gradio tabulate networkx qwen-vl-utils deepspeed
pip install torchmetrics[image] piq sentencepiece bitsandbytes
```

### 关键依赖版本

| 库 | 最低版本 | 说明 |
|---|---------|------|
| transformers | >= 4.52.0 | 支持 `Qwen3VLForConditionalGeneration` |
| diffusers | >= 0.32.0 | 支持 Sana 1.6B / AutoencoderDC |
| qwen-vl-utils | >= 0.0.14 | Qwen3-VL 视觉处理工具 |
| PyTorch | >= 2.3.0 | CUDA 12.x 支持 |

## 全流程指南

### 第一阶段：数据准备

#### 1.1 使用预构建数据集（推荐）

MetaQuery 官方提供了预构建的指令微调数据集：

```python
# 在 HuggingFace 上可直接加载
from datasets import load_dataset
dataset = load_dataset("xcpan/MetaQuery_Instruct_2.4M_512res")
```

#### 1.2 使用 Qwen3-VL 构建自定义数据集

如需从 mmc4 网页语料构建自定义指令数据，使用我们适配的脚本：

```bash
python curate_dataset_qwen3vl.py \
    --file_name /path/to/mmc4/docs_shard_0_v2.jsonl \
    --output_dir ./metaquery_instruct_qwen3vl \
    --qwen3vl_model Qwen/Qwen3-VL-8B-Instruct
```

数据集构建流程：
1. 从 mmc4 JSONL 中提取图文对
2. 使用 SigLIP 计算 caption 相似度，进行最大团分组
3. 选择组内最不相似的图像作为目标图像
4. 使用 **Qwen3-VL** 生成 few-shot 指令 prompt
5. 保存为 HuggingFace Dataset 格式

### 第二阶段：训练

MetaQuery 采用两阶段训练：

#### Stage 1: 文本到图像预训练（在 CC12M 上）

训练 connector 的基础文本-图像对齐能力：

```bash
# Linux (8 GPU)
bash scripts/train_qwen3vl.sh qwen3vl2b_t2i qwen3vl2b_sana.yaml /path/to/base_dir

# Windows (单 GPU)
scripts\train_qwen3vl.bat qwen3vl2b_t2i qwen3vl2b_sana.yaml E:\data\metaquery 1
```

#### Stage 2: 指令微调（在 MetaQuery-Instruct-2.4M 上）

从 Stage 1 checkpoint 继续训练，学习多图像指令引导：

```bash
# Linux
bash scripts/train_qwen3vl.sh qwen3vl2b_inst qwen3vl2b_sana_inst.yaml /path/to/base_dir

# Windows
scripts\train_qwen3vl.bat qwen3vl2b_inst qwen3vl2b_sana_inst.yaml E:\data\metaquery 1
```

从 checkpoint 恢复训练：

```bash
torchrun --nproc-per-node=8 train.py \
    --run_name qwen3vl2b_inst \
    --config_file qwen3vl2b_sana_inst.yaml \
    --base_dir /path/to/base_dir \
    --resume_from_checkpoint /path/to/checkpoint
```

#### Stage 2 (替代): 图像编辑微调

```bash
bash scripts/train_qwen3vl.sh qwen3vl2b_edit qwen3vl2b_sana_edit.yaml /path/to/base_dir
```

#### SLURM 集群训练

```bash
bash scripts/train_qwen3vl_slurm.sh qwen3vl8b_t2i qwen3vl8b_sana.yaml 2 /path/to/base_dir
```

### 第三阶段：推理

#### 命令行推理

```bash
# 文本到图像
python inference_qwen3vl.py \
    --checkpoint_path /path/to/checkpoint \
    --mode t2i \
    --prompt "A cat wearing a top hat in a garden" \
    --guidance_scale 4.5 \
    --num_images 4 \
    --seed 42

# 图像到图像（指令引导）
python inference_qwen3vl.py \
    --checkpoint_path /path/to/checkpoint \
    --mode i2i \
    --input_images ref1.jpg ref2.jpg \
    --prompt "Same style but in a winter scene" \
    --guidance_scale 4.5 \
    --image_guidance_scale 1.5
```

#### Gradio 交互式 Demo

```bash
python app_qwen3vl.py --checkpoint_path /path/to/checkpoint --share
```

#### Python API

```python
import torch
from pipeline_metaquery import MetaQueryPipeline
from trainer_utils import find_newest_checkpoint

# 加载模型
pipeline = MetaQueryPipeline.from_pretrained(
    find_newest_checkpoint("/path/to/checkpoint"),
    ignore_mismatched_sizes=True,
    _gradient_checkpointing=False,
    torch_dtype=torch.bfloat16,
).to("cuda", torch.bfloat16)

# 文本生成图像
images = pipeline(
    caption="A serene mountain lake at sunset",
    guidance_scale=4.5,
    num_inference_steps=30,
).images

# 保存
images[0].save("output.png")
```

## 可用配置文件

| 配置文件 | MLLM | 扩散模型 | 任务 | 推荐 GPU |
|---------|------|---------|------|---------|
| `qwen3vl2b_sana.yaml` | Qwen3-VL-2B | Sana 1.6B | T2I 预训练 | 4×A100-40G |
| `qwen3vl2b_sana_inst.yaml` | Qwen3-VL-2B | Sana 1.6B | 指令微调 | 4×A100-40G |
| `qwen3vl2b_sana_edit.yaml` | Qwen3-VL-2B | Sana 1.6B | 图像编辑 | 4×A100-40G |
| `qwen3vl4b_sana.yaml` | Qwen3-VL-4B | Sana 1.6B | T2I 预训练 | 4×A100-80G |
| `qwen3vl4b_sana_inst.yaml` | Qwen3-VL-4B | Sana 1.6B | 指令微调 | 4×A100-80G |
| `qwen3vl8b_sana.yaml` | Qwen3-VL-8B | Sana 1.6B | T2I 预训练 | 8×A100-80G |
| `qwen3vl8b_sana_inst.yaml` | Qwen3-VL-8B | Sana 1.6B | 指令微调 | 8×A100-80G |

## 训练参数说明

### 关键超参数

| 参数 | Stage 1 (T2I) | Stage 2 (Inst) | 说明 |
|------|--------------|----------------|------|
| `learning_rate` | 1e-4 | 5e-5 | 指令微调阶段使用更小学习率 |
| `warmup_steps` | 5000 | 2000 | 指令微调热身更短 |
| `num_metaqueries` | 256 | 256 | MetaQuery token 数量 |
| `connector_num_hidden_layers` | 24 | 24 | Connector 层数 |
| `loss_type` | flow | flow | Flow Matching 训练 |

### 冻结策略

```yaml
modules_to_freeze:
  - "vae"                    # VAE 始终冻结
  - "model.mllm_backbone"   # Qwen3-VL 骨干网络冻结

modules_to_unfreeze:
  - "model.mllm_backbone.model.embed_tokens"  # 仅解冻 embedding 层（用于新增的 MetaQuery tokens）
```

可训练参数：
- **Connector**（24层 Qwen2 双向 Encoder + 投影层）
- **Sana Transformer**（扩散模型主干）
- **Embed Tokens**（仅新增的 MetaQuery 特殊 token）

## 与官方 MetaQuery 的差异

| 方面 | 官方 MetaQuery | 本项目 |
|------|---------------|--------|
| MLLM 骨干 | LLaVA-OV 0.5B / Qwen2.5-VL 3B/7B | **Qwen3-VL 2B/4B/8B** |
| 视觉编码器 | SigLIP (LLaVA) / Qwen2.5-VL ViT (patch=14) | **Qwen3-VL ViT (patch=16, DeepStack)** |
| 位置编码 | MRoPE | **Interleaved-MRoPE** |
| 数据构建 | Qwen2-VL-7B 生成 prompt | **Qwen3-VL-8B 生成 prompt** |
| 训练脚本 | 仅 Linux/SLURM | **Linux + Windows 支持** |

## 项目结构

```
metaquery-main/
├── models/
│   ├── model.py                    # MLLMInContext（已添加 Qwen3-VL 支持）
│   ├── metaquery.py                # MetaQuery 训练/推理模型
│   └── transformer_encoder.py      # Connector（Qwen2 双向 Encoder）
├── configs/
│   ├── qwen3vl2b_sana.yaml        # [新增] Qwen3-VL-2B T2I
│   ├── qwen3vl2b_sana_inst.yaml   # [新增] Qwen3-VL-2B 指令微调
│   ├── qwen3vl2b_sana_edit.yaml   # [新增] Qwen3-VL-2B 图像编辑
│   ├── qwen3vl4b_sana.yaml        # [新增] Qwen3-VL-4B T2I
│   ├── qwen3vl4b_sana_inst.yaml   # [新增] Qwen3-VL-4B 指令微调
│   ├── qwen3vl8b_sana.yaml        # [新增] Qwen3-VL-8B T2I
│   ├── qwen3vl8b_sana_inst.yaml   # [新增] Qwen3-VL-8B 指令微调
│   ├── zero1.json / zero2.json / zero3.json
│   └── ...（原有 LLaVA/Qwen2.5 配置）
├── scripts/
│   ├── train_qwen3vl.sh            # [新增] Linux 训练脚本
│   ├── train_qwen3vl.bat           # [新增] Windows 训练脚本
│   └── train_qwen3vl_slurm.sh      # [新增] SLURM 集群脚本
├── train.py                        # 训练入口（无需修改）
├── trainer.py                      # 自定义 Trainer（无需修改）
├── dataset.py                      # 数据加载（无需修改）
├── pipeline_metaquery.py           # 推理 Pipeline（无需修改）
├── curate_dataset_qwen3vl.py       # [新增] Qwen3-VL 数据构建
├── inference_qwen3vl.py            # [新增] 命令行推理
├── app_qwen3vl.py                  # [新增] Gradio Web Demo
├── environment_qwen3vl.yml         # [新增] Conda 环境
└── README_QWEN3VL.md               # [新增] 本文档
```

## 常见问题

### Q: 显存不够怎么办？

1. **减小 batch size**：修改配置文件中的 `per_device_train_batch_size`
2. **使用 DeepSpeed ZeRO-2/3**：将配置中 `deepspeed` 改为 `configs/zero2.json` 或 `configs/zero3.json`
3. **使用更小的模型**：选择 Qwen3-VL-2B 而非 8B
4. **减少 max_pixels**：在代码中调整 `max_pixels` 参数

### Q: transformers 版本不支持 Qwen3-VL？

Qwen3-VL 需要 transformers >= 4.52.0。如果安装的版本较旧：

```bash
pip install transformers>=4.52.0
```

### Q: 如何从 Stage 1 过渡到 Stage 2？

Stage 2 训练可以直接使用 `--resume_from_checkpoint` 从 Stage 1 的 checkpoint 开始：

```bash
torchrun --nproc-per-node=8 train.py \
    --run_name qwen3vl2b_inst \
    --config_file qwen3vl2b_sana_inst.yaml \
    --base_dir /path/to/base_dir \
    --resume_from_checkpoint /path/to/stage1/checkpoint
```

## License

本项目代码基于 Meta 的 MetaQuery (BSD-3-Clause License) 和 Qwen3-VL (Apache-2.0) 的实现。
