# MetaQuery + Wan2.2 联合视频生成 — 原理与使用指南

> **MetaQuery** 是 Meta 提出的视觉条件生成方法，通过在多模态大语言模型 (MLLM) 中植入可学习的查询 token，
> 从参考图像中提取细粒度语义特征，用于引导扩散模型的生成过程。
>
> 本项目将 **MetaQuery (基于 Qwen3-VL)** 的视觉语义条件注入 **Wan2.2** 视频生成模型，
> 实现 **"看图生视频"** — 给定参考图像 + 文本描述，生成高质量视频。

---

## 目录

1. [整体架构与原理](#1-整体架构与原理)
2. [核心模块说明](#2-核心模块说明)
3. [数据流详解](#3-数据流详解)
4. [环境准备](#4-环境准备)
5. [快速开始](#5-快速开始)
6. [高级用法](#6-高级用法)
7. [三种管线对比](#7-三种管线对比)
8. [代码验证体系](#8-代码验证体系)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [文件结构](#10-文件结构)

---

## 1. 整体架构与原理

### 1.1 为什么需要 MetaQuery？

Wan2.2 原生只接受 **T5 文本编码** 作为生成条件（T2V）或 **首帧图像 + T5 文本**（I2V）。
但是：
- T5 文本编码缺乏细粒度视觉语义（无法精确描述颜色、构图、美学风格）
- 有时用户有一张参考图，希望生成 **"像这张图的风格/内容的视频"**，纯文本描述力不从心

MetaQuery 解决的核心问题：**如何将参考图像的视觉语义，高效注入视频扩散模型？**

### 1.2 原理概览

```
参考图像 + 文本描述
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Qwen3-VL (多模态大语言模型)                          │
│                                                     │
│  输入:                                               │
│    [System Prompt] + [图像 token] + [文本]            │
│    + <begin_of_img> <img0> <img1> ... <img255>       │
│      <end_of_img>                                    │
│                                                     │
│  Qwen3-VL 前向推理                                    │
│     │                                               │
│     ▼                                               │
│  从 <img0>~<img255> 位置提取隐藏状态                    │
│  → 256 个 MetaQuery embedding                        │
│    (shape: [256, hidden_size])                       │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Connector (双向 Qwen2Encoder + 线性投影)              │
│                                                     │
│  Qwen2Encoder (24层 bidirectional Transformer)       │
│     → Linear(hidden → connector_out)                 │
│     → GELU                                           │
│     → Linear(connector_out → connector_out)           │
│     → RMSNorm                                        │
│                                                     │
│  输出: [256, connector_out_dim]                       │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  to_wan_proj (投影层)                                │
│                                                     │
│  Linear(connector_out_dim → 4096) + GELU             │
│  + Linear(4096 → 4096)                               │
│                                                     │
│  输出: [256, 4096]  (与 T5 维度对齐)                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Context 拼接                                        │
│                                                     │
│  augmented_context = cat([MQ_feat, T5_feat], dim=0)  │
│  → shape: [256 + L_t5, 4096]                        │
│                                                     │
│  同时扩展 WanModel.text_len:                          │
│    512 → 512 + 256 = 768                             │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Wan2.2 DiT (扩散 Transformer)                        │
│                                                     │
│  每个 WanAttentionBlock 中:                           │
│    self-attention(video latent)                       │
│    cross-attention(video latent, augmented_context)   │
│         ↑                                            │
│    MetaQuery token 在这里参与 key/value 计算            │
│    为视频 latent 提供视觉语义引导                        │
│                                                     │
│  + CFG (Classifier-Free Guidance):                    │
│    条件预测: model(latent, context=[MQ+T5])            │
│    无条件预测: model(latent, context=[MQ_null+T5_null]) │
│    最终: uncond + scale * (cond - uncond)              │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
              VAE 解码 → 视频
```

### 1.3 关键设计要点

| 设计选择 | 原因 |
|---|---|
| **MetaQuery token 前置拼接到 T5 context** | 不修改 Wan2.2 模型权重，零侵入式注入 |
| **256 个 MetaQuery token** | 与 MetaQuery 论文一致，提供足够的视觉信息容量 |
| **投影到 4096 维** | 与 Wan2.2 的 T5 text_dim 完全对齐 |
| **动态扩展 text_len** | 使 Wan2.2 内部 padding 机制正确处理增长后的 context |
| **try/finally 恢复 text_len** | 即使生成中途崩溃也不会污染 WanModel 状态 |
| **Connector 使用 24 层 Qwen2Encoder** | 深度双向注意力，充分融合 MetaQuery 特征 |

---

## 2. 核心模块说明

### 2.1 MetaQueryEncoder (`wan/metaquery/encoder.py`)

负责将参考图像 + 文本编码为 256 个 MetaQuery embedding (维度=4096)。

**初始化流程**:
1. 加载 MetaQuery checkpoint（包含 Qwen3-VL + Connector 权重）
2. 丢弃 Sana Transformer 部分（节省 ~3-6GB 显存）
3. 初始化 `to_wan_proj` 投影层 (Xavier 初始化)
4. 运行完整性验证（backbone 类型、BOI/EOI token、connector 参数量等）

**编码流程** (`encode()` 方法):
1. 分词：将文本 + 图像通过 Qwen3-VL 的 tokenizer 处理
   - 自动插入 `<begin_of_img> <img0>...<img255> <end_of_img>`
2. Qwen3-VL 前向推理 → 获取所有 token 的隐藏状态
3. 从 BOI/EOI 之间提取 256 个 MetaQuery 隐藏状态
4. 通过 Connector (24层 Qwen2Encoder) 映射
5. 通过 `to_wan_proj` 投影到 4096 维
6. 返回 `List[Tensor]`，每项 shape `[256, 4096]`

### 2.2 MetaQueryWanBridge (`wan/metaquery/bridge.py`)

T2V（文本到视频）增强管线的核心编排器。

**职责**:
- 协调 T5 文本编码 + MetaQuery 视觉编码
- 将两路 context 拼接
- 管理 `WanModel.text_len` 的动态扩展与恢复
- 执行完整的去噪循环 + VAE 解码

**`generate()` 方法完整流程**:

```
Step 1: T5 编码文本 → context [L_t5, 4096], context_null [L_t5, 4096]
Step 2: MetaQuery 编码图像+文本 → mq_context [256, 4096], mq_null [256, 4096]
Step 3: 应用 mq_guidance_scale 缩放 mq_context
Step 4: 拼接 → aug_context [256+L_t5, 4096]
Step 5: 扩展 text_len → 512+256=768
Step 6: try: 去噪循环 (50步) → latent [16, T, H, W]
Step 7: VAE 解码 → 视频 [3, F, H, W]
Step 8: finally: 恢复 text_len → 512
```

### 2.3 MetaQueryWanI2VBridge (`wan/metaquery/bridge_i2v.py`)

I2V（图生视频）增强管线，提供 **双重视觉条件**:

| 条件类型 | 注入方式 | 提供信息 |
|---|---|---|
| 首帧条件 (y) | Channel Concatenation（20ch + 16ch = 36ch） | 像素级结构一致性 |
| MetaQuery 条件 | Context Concatenation（256+L_t5 token） | 语义级内容引导 |

### 2.4 MetaQueryWanAnimateBridge (`wan/metaquery/bridge_animate.py`)

Animate（人物动画）增强管线，提供 **四重条件**:

| 条件类型 | 注入方式 | 提供信息 |
|---|---|---|
| 参考图条件 | Channel Concat (y, 20ch) | 像素级人物外观 |
| CLIP 视觉条件 | 独立 cross-attn (k_img/v_img) | 全局视觉语义 |
| MetaQuery 条件 | Context Concat (MQ+T5) | 细粒度语义引导 |
| 面部条件 | Face Adapter (每5层) | 表情/面部动作 |

---

## 3. 数据流详解

### 3.1 维度变换链

```
Qwen3-VL hidden_size (因模型而异)
  │  2B: 1536
  │  4B: 2560
  │  8B: 4096
  ▼
Connector (24层 Qwen2Encoder + 线性投影)
  │  → connector_out_dim (因 Sana 配置而异, 如 2240)
  ▼
to_wan_proj (本模块新增)
  │  → wan_text_dim = 4096
  ▼
与 T5 context 拼接
  │  shape: [256 + L_t5, 4096]
  ▼
WanModel.text_embedding (Linear 4096→dim + GELU + Linear dim→dim)
  │  → [256 + L_t5, dim]  (dim=5120 for A14B)
  ▼
WanAttentionBlock.cross_attn (每层)
  │  Q = video latent,  K/V = text_embedding(augmented_context)
  │  MetaQuery token 作为额外的 key/value 参与注意力计算
  ▼
视频 latent 被视觉条件增强
```

### 3.2 CFG (Classifier-Free Guidance) 双条件

```python
# 条件预测: 有参考图 + 有文本
noise_pred_cond = model(latent, context=[MQ_cond + T5_cond])

# 无条件预测: 无参考图 + 负面文本
noise_pred_uncond = model(latent, context=[MQ_null + T5_null])

# CFG 引导
noise_pred = uncond + guide_scale * (cond - uncond)
```

**关键**: `MQ_cond`（有参考图）和 `MQ_null`（无参考图）编码出的特征**必须不同**，
CFG 才能有效引导生成。代码中有断言验证两者的余弦相似度 < 0.99。

### 3.3 WanModel.text_len 动态扩展

```python
# Wan2.2 WanModel.forward 中的 context 处理:
context = self.text_embedding(
    torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ])
)
# text_len 决定了 zero-padding 的目标长度
# 原始 text_len=512，MetaQuery 增加 256 个 token 后
# 需要扩展到 768，否则 padding 会截断增强的 context
```

---

## 4. 环境准备

### 4.1 依赖项

```bash
# Wan2.2 基础依赖
pip install torch torchvision torchaudio
pip install transformers>=4.46.0  # Qwen3-VL 需要较新版本
pip install diffusers>=0.31.0
pip install easydict tqdm Pillow

# MetaQuery 依赖 (from metaquery-main)
pip install datasets accelerate

# Qwen3-VL 视觉处理
pip install qwen-vl-utils

# Animate 额外依赖
pip install decord einops peft opencv-python
```

### 4.2 模型权重准备

需要准备以下模型权重:

| 权重 | 说明 | 获取方式 |
|---|---|---|
| Wan2.2 checkpoint | 包含 T5、VAE、DiT (low/high noise) | [Wan2.2 官方发布](https://github.com/Wan-Video/Wan2.2) |
| MetaQuery + Qwen3-VL checkpoint | 包含 Qwen3-VL backbone + Connector | MetaQuery 训练输出或初始权重 |

**MetaQuery checkpoint 目录结构示例**:
```
metaquery_qwen3vl_output/
├── checkpoint-1000/       # 训练过的 checkpoint (推荐)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── ...
```

> **提示**: `find_newest_checkpoint()` 会自动查找目录下最新的 checkpoint 子文件夹。
> 如果目录直接包含 `.safetensors` / `.pt` 文件，则直接使用该目录。

---

## 5. 快速开始

### 5.1 使用 Demo 脚本（最简单）

编辑 `demo_metaquery_wan.py` 中的路径配置：

```python
# ====== 用户配置区 ======
WAN_CKPT_DIR = r"E:\models\Wan2.2"                    # Wan2.2 权重目录
METAQUERY_CKPT = r"E:\models\metaquery_qwen3vl_output" # MetaQuery checkpoint
REFERENCE_IMAGES = [r"E:\data\reference_scene.jpg"]     # 参考图像
PROMPT = "一只橘猫慵懒地躺在阳光洒落的窗台上，微风轻拂窗帘，光影流动"
```

运行：
```bash
cd Wan2.2
python demo_metaquery_wan.py
```

### 5.2 使用 CLI 脚本（灵活配置）

```bash
cd Wan2.2
python generate_with_metaquery.py \
    --task t2v-A14B \
    --ckpt_dir /path/to/wan2.2 \
    --metaquery_ckpt /path/to/metaquery_qwen3vl \
    --prompt "一只猫在沙发上打盹" \
    --input_images /path/to/ref1.jpg /path/to/ref2.jpg \
    --num_metaqueries 256 \
    --size 1280*720 \
    --frame_num 81 \
    --sampling_steps 50 \
    --guide_scale 5.0 \
    --seed 42 \
    --offload_model \
    --output_dir ./outputs
```

### 5.3 Python API（集成开发）

```python
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.metaquery import MetaQueryWanBridge

# 1. 初始化 Wan2.2
cfg = WAN_CONFIGS["t2v-A14B"]
pipeline = wan.WanT2V(config=cfg, checkpoint_dir="/path/to/wan2.2", device_id=0)

# 2. 初始化 Bridge
bridge = MetaQueryWanBridge(
    wan_pipeline=pipeline,
    metaquery_checkpoint="/path/to/metaquery_ckpt",
    num_metaqueries=256,
    mq_guidance_scale=1.0,     # MetaQuery 特征缩放 (>1 增强视觉影响)
    dtype=torch.bfloat16,
)

# 3. 生成
ref_img = Image.open("reference.jpg").convert("RGB")
video = bridge.generate(
    input_prompt="一只猫在沙发上打盹",
    input_images=[ref_img],    # 可传入多张参考图
    size=(1280, 720),
    frame_num=81,
    sampling_steps=50,
    guide_scale=5.0,
    seed=42,
)
# video shape: [3, 81, 720, 1280]
```

---

## 6. 高级用法

### 6.1 I2V (图生视频) + MetaQuery

```python
from wan.metaquery import MetaQueryWanI2VBridge

# 初始化 WanI2V
cfg = WAN_CONFIGS["i2v-A14B"]
pipeline = wan.WanI2V(config=cfg, checkpoint_dir="/path/to/wan2.2", device_id=0)

# 初始化 I2V Bridge
bridge = MetaQueryWanI2VBridge(
    wan_i2v_pipeline=pipeline,
    metaquery_checkpoint="/path/to/metaquery_ckpt",
    num_metaqueries=256,
)

# 生成: 首帧 + MetaQuery 参考图 + 文本
first_frame = Image.open("first_frame.jpg").convert("RGB")
style_ref = Image.open("style_reference.jpg").convert("RGB")

video = bridge.generate(
    input_prompt="人物走向镜头",
    first_frame=first_frame,                    # 首帧 (必须)
    mq_reference_images=[style_ref],            # MetaQuery 参考图 (可选)
    frame_num=81,
    sampling_steps=40,
    guide_scale=5.0,
)
```

> **注意**: 如果不传 `mq_reference_images`，I2V Bridge 会自动使用 `first_frame` 作为 MetaQuery 输入。

### 6.2 Animate (人物动画) + MetaQuery

```python
from wan.metaquery import MetaQueryWanAnimateBridge

bridge = MetaQueryWanAnimateBridge(
    wan_animate_pipeline=animate_pipeline,
    metaquery_checkpoint="/path/to/metaquery_ckpt",
    num_metaqueries=256,
)

video = bridge.generate(
    input_prompt="人物微笑并转头",
    ref_image=person_image,                # 参考人物图
    face_video_path="face_driving.mp4",    # 面部驱动视频
    mq_reference_images=[style_image],     # MetaQuery 风格引导
    frame_num=81,
)
```

### 6.3 参数调优

| 参数 | 默认值 | 作用 | 建议范围 |
|---|---|---|---|
| `num_metaqueries` | 256 | 语义信息容量 | 64~512，必须与训练配置一致 |
| `mq_guidance_scale` | 1.0 | MetaQuery 特征缩放 | 0.5~2.0，>1 增强视觉影响 |
| `guide_scale` | 5.0 | CFG 强度 | 3.0~7.0 |
| `sampling_steps` | 50 | 去噪步数 | 30~100，越多越精细 |
| `shift` | 5.0 | 噪声调度偏移 | Wan2.2 默认值即可 |
| `seed` | -1 | 随机种子 | 固定种子可复现结果 |

### 6.4 显存优化

```python
bridge = MetaQueryWanBridge(
    wan_pipeline=pipeline,
    metaquery_checkpoint=ckpt,
    dtype=torch.bfloat16,       # 使用 bf16 减少显存
)

video = bridge.generate(
    ...,
    offload_model=True,          # 不活跃模型卸载到 CPU
)
```

**显存估算** (A14B 模型, 1280×720, 81帧):
| 组件 | 显存占用 |
|---|---|
| Wan2.2 DiT (单模型) | ~28 GB |
| Qwen3-VL 2B + Connector | ~5 GB |
| T5 编码器 | ~10 GB |
| VAE | ~2 GB |
| 开启 offload 时峰值 | ~35 GB |
| 不开启 offload 时峰值 | ~45+ GB |

---

## 7. 三种管线对比

| 特性 | MetaQueryWanBridge (T2V) | MetaQueryWanI2VBridge (I2V) | MetaQueryWanAnimateBridge |
|---|---|---|---|
| 输入 | 文本 + 参考图 | 首帧 + 文本 + 参考图 | 人物图 + 面部视频 + 文本 + 参考图 |
| MetaQuery 注入 | ✅ Context Concat | ✅ Context Concat | ✅ Context Concat |
| 首帧条件 | ❌ | ✅ Channel Concat | ✅ Channel Concat |
| CLIP 视觉条件 | ❌ | ❌ | ✅ 独立 cross-attn |
| 面部条件 | ❌ | ❌ | ✅ Face Adapter |
| 骨架条件 | ❌ | ❌ | ❌ (传零 pose) |
| WanModel 类型 | t2v | i2v | animate (s2v) |
| 输入通道数 | 16 | 36 (16+20) | 36 (16+20) |

---

## 8. 代码验证体系

本项目在每个关键环节都有 **assert 断言 + 数值验证 print**，确保整个流程不是 "假运行"。

### 8.1 初始化验证 (`encoder.py __init__`)

| 检查项 | 验证方式 |
|---|---|
| Qwen3-VL backbone 类型 | `assert 'Qwen3VL' in class_name` |
| BOI/EOI token ID 注册 | `assert hasattr(model, 'boi_token_id')` |
| Connector 参数非零 | `sum(p.numel()) > 0` |
| Connector 内部结构 | 打印 `[Qwen2Encoder, Linear, GELU, Linear, RMSNorm]` |
| 投影层权重非零 | `p.data.abs().sum() > 0` |
| WanModel text_embedding 维度=4096 | `in_features == 4096` |
| WanModel 含 cross_attn | `hasattr(block, 'cross_attn')` |

### 8.2 编码验证 (`encoder.py encode()`)

| 检查项 | 验证方式 |
|---|---|
| input_ids 含 BOI/EOI | `(input_ids == boi_id).sum() > 0` |
| img token 数量 = num_metaqueries | `eoi_pos - boi_pos - 1 == 256` |
| Connector 输出维度正确 | `shape[1] == 256, shape[2] == connector_out_dim` |
| 特征非零/非NaN/非Inf | `torch.isnan()`, `.norm() > 0` |
| 投影前后余弦相似度 | 证明 `to_wan_proj` 确实函数变换了特征 |

### 8.3 生成验证 (`bridge.py generate()`)

| 检查项 | 验证方式 |
|---|---|
| T5 输出非零 | `.norm() > 0` |
| MQ cond ≠ MQ uncond | `cosine_similarity < 0.99` |
| 拼接后长度 = T5 + MQ | `aug.shape[0] == t5.shape[0] + 256` |
| MQ 部分 ≠ T5 部分 | 两部分余弦相似度 < 0.99 |
| text_len 扩展回读验证 | `model.text_len == aug_text_len` |
| mq_guidance_scale 实际应用 | 缩放前后打印 MQ 范数变化 |
| 去噪 step 1: cond ≠ uncond | `(cond - uncond).norm() > 1e-6` |
| 去噪中间步: CFG 持续有效 | 中间步再验证一次 |
| text_len 恢复回读验证 | finally 块中恢复后 `== orig_text_len` |
| 视频 tensor 非零/非NaN | `.norm() > 0, no NaN` |
| 视频文件写入成功 | `os.path.exists() and size > 0` |

---

## 9. 常见问题 FAQ

### Q1: `find_newest_checkpoint` 报错找不到 checkpoint？

确保目录结构正确：
```
# 方式 A: 目录直接包含模型文件
checkpoint_dir/
├── config.json
├── model.safetensors
└── ...

# 方式 B: 目录包含编号子文件夹 (训练输出)
checkpoint_dir/
├── checkpoint-500/
├── checkpoint-1000/   ← 自动选择最新的
└── ...
```

### Q2: 显存不够怎么办？

1. 开启 `offload_model=True` (最重要)
2. 开启 `t5_cpu=True` (T5 放 CPU)
3. 减小分辨率: `size=(640, 360)`
4. 减少帧数: `frame_num=33`
5. 使用更小的 Qwen3-VL (如 2B 而非 8B)

### Q3: 生成的视频没有体现参考图的内容？

1. 检查 `mq_guidance_scale` 是否太小 → 尝试增大到 1.5~2.0
2. 确认 MetaQuery checkpoint 是**训练过的** (而非初始权重)
3. 查看运行时日志中 `MQ cond vs uncond 余弦相似度` 是否 < 0.99
4. 检查参考图是否正确加载 (看 `pixel_values` 的非零率)

### Q4: `to_wan_proj` 是随机初始化的，不会出问题吗？

`to_wan_proj` 使用 Xavier 初始化。即使没有训练，它也能将 MetaQuery 特征
合理映射到 Wan2.2 的特征空间。但效果不如经过微调的版本。

如果你有训练过的 MetaQuery checkpoint:
- Connector 权重是训练好的 ✅
- `to_wan_proj` 目前是新增层，需要额外微调才能发挥最佳效果

### Q5: 支持哪些 Qwen3-VL 模型尺寸？

| 尺寸 | hidden_size | 支持 | 备注 |
|---|---|---|---|
| 2B | 1536 | ✅ | 推荐，显存友好 |
| 4B | 2560 | ✅ | 平衡 |
| 8B | 4096 | ✅ | 最佳效果 |

所有尺寸都通过 Connector 统一映射到 4096 维，兼容同一套 Wan2.2 模型。

### Q6: `tokenize()` 传参的 `input_images` 格式是什么？

```python
# encoder.encode() 的 input_images 参数:
# List[Optional[List[PIL.Image]]], 长度 = batch_size
# 每项是一个图像列表 (支持多图), 或 None (无图像)

# 示例:
encoder.encode(
    captions=["一只猫"],
    input_images=[[Image.open("cat.jpg")]],  # batch=1, 1张参考图
)

encoder.encode(
    captions=["一只猫"],
    input_images=None,  # 无参考图 (null 条件)
)
```

### Q7: I2V Bridge 中首帧和 MetaQuery 参考图有什么区别？

- **首帧** (`first_frame`): 通过 VAE 编码为 latent → channel concat 到噪声输入，为视频提供**像素级结构一致性**
- **MQ 参考图** (`mq_reference_images`): 通过 Qwen3-VL + Connector → context concat 到 T5，为视频提供**语义级美学/风格引导**
- 如果未指定 `mq_reference_images`，I2V Bridge 自动用 `first_frame` 同时作为两种条件

### Q8: 生成阶段报维度错误？

- 确认 Wan checkpoint 与任务配置一致（T2V 用 t2v 权重，I2V 用 i2v 权重）
- 确认图像尺寸可被 Wan2.2 的 VAE 下采样倍率整除
- I2V 必须使用 i2v-A14B 权重，Animate 必须使用 animate-14B 权重

### Q9: `mq_guidance_scale` 与 `guide_scale` 的区别？

| 参数 | 作用域 | 影响 |
|---|---|---|
| `mq_guidance_scale` | 仅 MetaQuery token | 缩放 MQ 特征向量的范数，调整视觉语义权重 |
| `guide_scale` | 整个 CFG 过程 | $\text{pred} = \text{uncond} + s \cdot (\text{cond} - \text{uncond})$，CFG 总强度 |

两者独立作用。建议先固定 `guide_scale=5.0`，调 `mq_guidance_scale` 在 0.5~2.0 区间。

---

## 10. 文件结构

```
Wan2.2/
├── wan/
│   ├── metaquery/
│   │   ├── __init__.py          # 模块入口，导出所有 Bridge 类
│   │   ├── encoder.py           # MetaQueryEncoder — Qwen3-VL + Connector + 投影
│   │   ├── bridge.py            # MetaQueryWanBridge — T2V 增强管线
│   │   ├── bridge_i2v.py        # MetaQueryWanI2VBridge — I2V 增强管线
│   │   └── bridge_animate.py    # MetaQueryWanAnimateBridge — Animate 增强管线
│   ├── modules/
│   │   └── model.py             # WanModel — 扩散 Transformer (原始 Wan2.2)
│   ├── text2video.py            # WanT2V — 原始 T2V 管线
│   ├── image2video.py           # WanI2V — 原始 I2V 管线
│   └── configs/                 # 模型配置
│       ├── shared_config.py     # text_len=512, num_train_timesteps=1000
│       ├── wan_t2v_A14B.py      # boundary=0.875
│       └── wan_i2v_A14B.py      # boundary=0.900
├── demo_metaquery_wan.py        # 交互式 Demo 脚本 (T2V)
├── generate_with_metaquery.py   # CLI 生成脚本
└── MetaQuery_Wan_Usage_CN.md    # 本文档
```

**依赖的外部模块** (需在 sys.path 中, 位于 `Qwen3-VL-main/metaquery-main/`):
```
Qwen3-VL-main/metaquery-main/
├── models/
│   ├── model.py                 # MLLMInContext — 持有 Qwen3-VL + Connector
│   ├── metaquery.py             # MetaQuery — 完整 pipeline (含 VAE/Scheduler)
│   └── transformer_encoder.py   # Qwen2Encoder — Connector 的双向 Transformer
├── trainer_utils.py             # find_newest_checkpoint()
└── ...
```

---

> **最后提示**: 运行时请关注控制台输出的 `[VERIFY]` 和 `[PASS]` 标记。
> 如果所有验证都通过（无 `[FATAL]` 或 `⚠️`），则整个流程确认正确运行。

---

## 附录：维度不适配问题深度分析

### 问题描述

`connector_out_dim`（约 2240）与 `wan_text_dim`（4096）之间存在维度不匹配，需要额外的 `to_wan_proj` 层进行投影。这个投影层**从未经过针对 Wan2.2 的训练**，是本方案最关键的、需要通过微调解决的技术债务。

---

### 一、为什么 connector 输出是 2240，而不是 4096？

这是由 **MetaQuery 原始训练目标（Sana）** 决定的，与 Wan2.2 完全无关。

#### 1.1 connector_out_dim 是如何确定的

在 `MLLMInContext.__init__`（`models/model.py`）中，`connector_out_dim` 的值**直接从 Sana Transformer 的配置里读取**：

```python
# models/model.py — MLLMInContext.__init__
if "Sana" in config.diffusion_model_id:
    self.transformer = SanaTransformer2DModel.from_pretrained(
        config.diffusion_model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    input_scale = math.sqrt(5.5)
    # 注释: 2304 --> 2240 （Sana caption_channels 经压缩后的实际值）

self.connector_out_dim = (
    getattr(self.transformer.config, "caption_channels", None)  # ← Sana: 2240
    or getattr(self.transformer.config, "encoder_hid_dim", None)
    or getattr(self.transformer.config, "cross_attention_dim", None)
)
```

Sana-1600M 模型的 `caption_channels = 2240`。这就是 connector 输出维度的直接来源。

#### 1.2 connector 的输出维度由训练目标决定

Connector 被设计为：**将 MLLM 隐藏状态映射到扩散模型的条件输入维度**。原始训练时，条件输入维度就是 Sana 的 `caption_channels = 2240`，因此：

```
Connector 输出: [B, 256, 2240]
                        ↓
     直接送入 Sana.transformer(encoder_hidden_states=prompt_embeds)
                        ↓
             Sana cross-attention: K, V 来自 2240 维特征
```

梯度从 Sana 的重建损失出发，反传经过 Sana Transformer → Connector → Qwen3-VL，整个链路都在 2240 这个维度上对齐优化。

---

### 二、训练流程解析：哪些参数被优化了？

完整训练循环在 `MetaQuery.forward()`（`models/metaquery.py`）中：

```
训练数据: 图像 x_0 + 参考图像 + 文本 caption
                    │
                    ▼
         VAE.encode(x_0) → latents
                    │
                    ▼ 加噪
         noisy_latents = (1 - σ) * latents + σ * noise
                    │
                    ▼ 条件提取
         encode_condition(input_ids, pixel_values, ...)
           ├── Qwen3-VL 前向推理
           ├── 提取 BOI~EOI 之间的隐藏状态
           └── Connector
                   → prompt_embeds: [B, 256, 2240]
                    │
                    ▼ 去噪预测
         Sana.transformer(
             hidden_states=noisy_latents,
             encoder_hidden_states=prompt_embeds,  # ← 2240 维
         ) → noise_pred
                    │
                    ▼ 计算损失
         loss = MSE(noise_pred, noise - latents)
                    │
                    ▼ 反向传播
         梯度流: Sana ← Connector ← Qwen3-VL
```

**被优化的参数**（通过损失梯度）：

| 参数 | 是否被训练 | 说明 |
|------|-----------|------|
| Qwen3-VL backbone | ✅ 部分训练（取决于冻结策略） | 提取视觉语义 |
| Connector（Qwen2Encoder + Linear + GELU + Linear + RMSNorm） | ✅ **核心训练目标** | 2240 维对齐 |
| Sana Transformer | ✅ 部分训练 | 接受 2240 维条件 |
| **`to_wan_proj`（2240 → 4096）** | ❌ **完全不存在于训练流程中** | 这是本项目新增的 |
| Wan2.2 的任何组件 | ❌ **完全不存在** | 与训练无关 |

---

### 三、to_wan_proj 的状态：随机初始化，从未被训练

`to_wan_proj` 层是在 `MetaQueryEncoder.__init__`（`encoder.py`）中**新建**的：

```python
# encoder.py — MetaQueryEncoder.__init__
self.to_wan_proj = nn.Sequential(
    nn.Linear(connector_out_dim, wan_text_dim, bias=True),  # 2240 → 4096
    nn.GELU(approximate="tanh"),
    nn.Linear(wan_text_dim, wan_text_dim, bias=True),        # 4096 → 4096
).to(device=self.device, dtype=dtype)

# Xavier 初始化
for m in self.to_wan_proj.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
```

**它不存在于 MetaQuery 的 checkpoint 文件中**，不会被 `MetaQuery.from_pretrained()` 加载。每次启动程序都是全新的随机初始化权重。

这意味着**第一次运行时，送入 Wan2.2 的 4096 维特征是随机变换后的结果**，与 Wan2.2 的分布没有任何对齐。

---

### 四、这两种"不适配"是完全不同性质的问题

你的问题触及了两种不同性质的不适配，必须区分清楚：

#### 情况 A：硬不兼容（维度无法对齐，直接报错）

```
connector 输出 2240 维  →  直接送入 WanModel.text_embedding（期望 4096 维输入）
                               ↓
                    RuntimeError: mat1 (N×2240) mat2 (4096×dim) 维度不匹配
```

**这是硬性错误**。没有 `to_wan_proj`，根本无法运行。`to_wan_proj` 解决的是这个问题——让程序能够跑起来。

#### 情况 B：软不适配（能运行，但效果不确定）

```
connector 输出 2240 维（已被对齐到 Sana 语义空间）
    │
    ▼ to_wan_proj（随机初始化，未经训练）
    ▼
4096 维特征（语义空间未知，与 Wan2.2 的 T5 编码分布不一致）
    │
    ▼ 送入 Wan2.2 cross-attention
```

**这是软性问题**。程序可以运行，会生成视频，但：
1. Connector 被训练对齐到 Sana 的 2240 维空间，这个对齐知识在 `to_wan_proj` 后被破坏
2. Wan2.2 的 cross-attention 期望接收类似 T5 编码的特征分布（4096 维），而现在接收的是随机变换的特征
3. 效果好坏**完全取决于 `to_wan_proj` 对 connector 输出的保留程度**

---

### 五、从信号保留角度理解：Xavier 初始化下的变换质量

Xavier 初始化的目标是保持输入/输出的方差稳定，对于线性层 `Linear(2240 → 4096)` 有：

$$\text{weight} \sim \mathcal{U}\left(-\sqrt{\frac{6}{2240+4096}},\ +\sqrt{\frac{6}{2240+4096}}\right) \approx \mathcal{U}(-0.0307, 0.0307)$$

这确保了：
- 输出方差 ≈ 输入方差（不会爆炸或消失）
- **但语义信息几乎全部丢失**——因为是随机投影，connector 学到的"哪个方向对应哪种视觉语义"被完全打乱

直觉理解：connector 训练到"第3个输出维度代表红色调"，但经过随机 `to_wan_proj` 后，这个信息被分散到 4096 维的噪声中。

---

### 六、正确的解决方案：联合微调 to_wan_proj + 部分 WanModel

要使整个链路真正有效，需要**以 Wan2.2 的去噪损失为目标做一次微调**，让梯度流貫通整个链路：

```
训练目标（应该是）:
    图像/视频 + 参考图像 + 文本 caption
                │
                ▼
    Wan VAE.encode → latents  
                │
                ▼ 加噪
    noisy_latents
                │
                ▼
    encode_condition → connector_out [256, 2240]
                │
                ▼ to_wan_proj（可学习）
    mq_features [256, 4096]
                │ cat with T5 context
                ▼
    WanModel(noisy_latents, context=aug_context)
                │
                ▼
    noise_pred
                │
                ▼
    loss = MSE(noise_pred, target)  ← 梯度反传到 to_wan_proj 和上游
```

**建议的冻结策略**（显存有限时）：

| 模块 | 推荐策略 | 原因 |
|------|---------|------|
| Qwen3-VL backbone | 冻结 | 参数量巨大（2B+），已预训练 |
| Connector（24层Qwen2Encoder） | 冻结 | 已在 Sana 上训练，语义提取能力已具备 |
| **`to_wan_proj`** | **必须训练** | 这是唯一未训练的瓶颈 |
| WanModel（部分） | 可选训练（LoRA） | 让 cross-attention 适应新的条件分布 |
| WanModel（text_embedding） | 可选训练 | 让输入投影适应混合 MQ+T5 格式 |

---

### 七、当前状态的实际效果预期

在没有微调的情况下（纯 `to_wan_proj` Xavier 初始化）：

| 场景 | 预期效果 |
|------|---------|
| 不提供参考图（`input_images=None`） | 仅文本条件生效，效果≈原始 Wan2.2 T2V |
| 提供参考图，`mq_guidance_scale=1.0` | 视觉条件经随机投影后信号微弱，效果不确定 |
| 提供参考图，`mq_guidance_scale=3.0+` | 放大了随机噪声，可能导致生成质量下降 |
| **微调 to_wan_proj 后** | connector 的视觉语义能真正传递到 Wan2.2，参考图效果显著 |

---

### 八、快速验证 to_wan_proj 是否真的随机

运行以下代码可以验证 `to_wan_proj` 不在 checkpoint 中：

```python
import torch
from models.metaquery import MetaQuery
from trainer_utils import find_newest_checkpoint

ckpt = find_newest_checkpoint("/path/to/metaquery_qwen3vl")
state_dict = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu")

# 检查是否有 to_wan_proj 的键
wan_proj_keys = [k for k in state_dict if "to_wan_proj" in k]
print(f"to_wan_proj 相关键: {wan_proj_keys}")
# 预期输出: []  ← 空列表，说明不在 checkpoint 中
```

---

### 九、总结：不适配的本质

| 类型 | 内容 | 是否可运行 | 解决方案 |
|------|------|-----------|---------|
| **硬不兼容** | 2240 ≠ 4096，直接维度错误 | ❌ 不可运行 | `to_wan_proj` 层（已实现） |
| **软不适配（训练）** | `to_wan_proj` never trained on Wan2.2 loss | ✅ 可运行，效果差 | 联合微调 `to_wan_proj` + LoRA |
| **软不适配（分布）** | Connector 对齐 Sana 空间，Wan2.2 期望 T5 空间 | ✅ 可运行，效果差 | 微调让 connector/proj 对齐 Wan2.2 |

**一句话总结**：
> 2240→4096 的 `to_wan_proj` 解决了**能不能跑**的问题（硬不兼容）；
> 但由于该层从未用 Wan2.2 的损失训练过，**跑起来效果好不好**（软不适配）还需要一次针对 Wan2.2 的微调才能解决。

---

## 附录 B：输入接收、注入方式与管线对比全解析

### 一、三条适配管线及其输入一览

本项目为 Wan2.2 的三种生成模式各实现了一条 MetaQuery 增强管线：

| 管线 | 类名 | 对应 Wan 基础管线 | 文件 |
|------|------|-----------------|------|
| **T2V** (文生视频) | `MetaQueryWanBridge` | `WanT2V` | `bridge.py` |
| **I2V** (图生视频) | `MetaQueryWanI2VBridge` | `WanI2V` | `bridge_i2v.py` |
| **Animate** (人物动画) | `MetaQueryWanAnimateBridge` | `WanAnimate` | `bridge_animate.py` |

---

### 二、各管线接收的输入

#### 2.1 MetaQueryWanBridge（T2V 文生视频）

```python
bridge.generate(
    input_prompt: str,                           # 文本提示词
    input_images: Optional[List[PIL.Image]],     # MetaQuery 参考图（可选）
    size=(1280, 720),                            # 输出视频分辨率
    frame_num=81,                                # 帧数
    shift=5.0, sample_solver="unipc",
    sampling_steps=50, guide_scale=5.0,
    n_prompt="", seed=-1, offload_model=True,
)
```

**输入信号**：

| 输入 | 类型 | 必须 | 用途 |
|------|------|:----:|------|
| `input_prompt` | str | ✅ | T5 编码为文本 context；同时也作为 Qwen3-VL 的文本输入 |
| `input_images` | List[PIL.Image] | ❌ | Qwen3-VL 的参考图像，提取 MetaQuery 视觉语义特征 |

> **无图时**：`input_images=None` → MetaQuery encoder 只接收文本，无视觉像素输入，MQ 特征退化为纯文本语义特征。

#### 2.2 MetaQueryWanI2VBridge（I2V 图生视频）

```python
bridge_i2v.generate(
    input_prompt: str,                                    # 文本提示词
    first_frame: PIL.Image,                               # 首帧图像（必须）
    mq_reference_images: Optional[List[PIL.Image]],       # MetaQuery 参考图（可选）
    max_area=720*1280,                                    # 最大像素面积
    frame_num=81, shift=5.0, sample_solver="unipc",
    sampling_steps=40, guide_scale=5.0,
    n_prompt="", seed=-1, offload_model=True,
)
```

**输入信号**：

| 输入 | 类型 | 必须 | 用途 |
|------|------|:----:|------|
| `input_prompt` | str | ✅ | T5 编码文本 context + Qwen3-VL 文本输入 |
| `first_frame` | PIL.Image | ✅ | VAE 编码为首帧像素条件 (y tensor, 20ch) |
| `mq_reference_images` | List[PIL.Image] | ❌ | Qwen3-VL MetaQuery 参考图；**若不传，默认使用 `first_frame`** |

> **关键设计**：`first_frame` 和 `mq_reference_images` **可以不同**。前者控制首帧像素结构，后者控制语义风格。例如可以传入一张风景首帧 + 一张油画风格参考图。

#### 2.3 MetaQueryWanAnimateBridge（Animate 人物动画）

```python
bridge_animate.generate(
    input_prompt: str,                                    # 文本提示词
    ref_image: PIL.Image,                                 # 参考人物图（必须）
    face_source=None,                                     # 面部视频/帧列表
    mq_reference_images: Optional[List[PIL.Image]],       # MetaQuery 参考图（可选）
    frame_num=77, clip_len=77, refert_num=1,
    shift=5.0, sample_solver="dpm++",
    sampling_steps=20, guide_scale=1.0,
    n_prompt="", seed=-1, offload_model=True,
)
```

**输入信号**：

| 输入 | 类型 | 必须 | 用途 |
|------|------|:----:|------|
| `input_prompt` | str | ✅ | T5 编码文本 context + Qwen3-VL 文本输入 |
| `ref_image` | PIL.Image | ✅ | ① VAE 编码为参考图条件 (y, 20ch)；② CLIP ViT-H/14 编码为全局视觉 token |
| `face_source` | str / List[np.ndarray] / List[PIL.Image] | ❌ | 面部视频帧，经 motion_encoder → face_encoder → face_adapter 注入面部动作 |
| `mq_reference_images` | List[PIL.Image] | ❌ | Qwen3-VL MetaQuery 参考图；**若不传，默认使用 `ref_image`** |

> **面部条件为空时**：`face_source=None` → 全零面部帧 → 归一化后为 -1 → 面部 adapter 不起实际引导作用。
> **骨架条件**：本版本**始终禁用**，`pose_latents` 传全零张量。

---

### 三、条件注入方式详解

本项目中出现了 **五种不同的条件注入机制**。以下逐一解析其原理和在代码中的实现位置。

#### 3.1 Context Concatenation（上下文拼接注入 — 文本 + MetaQuery）

**作用于**：T2V / I2V / Animate 全部三条管线  
**注入对象**：T5 文本特征 + MetaQuery 视觉语义特征  
**注入位置**：WanModel / WanAnimateModel 的 `cross_attn` 层

**原理**：

```
T5 文本编码:     [L_t5, 4096]    （L_t5 ≈ 几十~几百 token）
MetaQuery 编码:  [256,  4096]    （经 to_wan_proj 投影到 4096 维）
                      ↓ 前置拼接
拼接后 context:  [256 + L_t5, 4096]
                      ↓ WanModel.text_embedding (Linear(4096→dim))
投影后:          [256 + L_t5, dim]
                      ↓ 零填充到 text_len
context tensor:  [text_len, dim]
                      ↓ 送入每个 block 的 cross_attn
在 cross-attn 中作为 K, V
```

**代码位置** — `WanModel.forward()`：
```python
# wan/modules/model.py — forward()
context = self.text_embedding(
    torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context       # ← context 已经是 [MQ+T5, 4096]
    ])
)
# → context: [B, text_len, dim]
# 进入每个 WanAttentionBlock.cross_attn 作为 K/V
```

**代码位置** — 拼接操作 `_augment_context()`：
```python
# bridge.py / bridge_i2v.py / bridge_animate.py — _augment_context()
aug = torch.cat([mq_feat, t5_feat], dim=0)
# mq_feat: [256, 4096], t5_feat: [L_t5, 4096]
# → aug: [256 + L_t5, 4096]
```

**关键细节**：MetaQuery 特征被 **前置** 拼接（排在 T5 前面），确保在 cross-attention 中被优先关注。同时动态扩展 `WanModel.text_len` 以容纳更长的 context 序列（+256 token）。

---

#### 3.2 Channel Concatenation（通道拼接注入 — 首帧/参考图像素条件）

**作用于**：I2V / Animate 两条管线  
**注入对象**：首帧图像 / 参考人物图的 VAE latent  
**注入位置**：WanModel `patch_embedding` 之前，与噪声 latent 在通道维度拼接

**原理**：

```
噪声 latent:     [16, T_lat, H_lat, W_lat]   (16 通道 VAE latent)
首帧条件 y:      [20, T_lat, H_lat, W_lat]   (4ch mask + 16ch VAE latent)
                      ↓ 通道拼接
x = cat([latent, y]):  [36, T_lat, H_lat, W_lat]
                      ↓ patch_embedding (Conv3d(in=36, out=dim))
嵌入后:          [dim, ...] → 展平 → 自注意力序列
```

**代码位置** — `WanModel.forward()`：
```python
# wan/modules/model.py — forward()
if y is not None:
    x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
# x 从 [16, ...] 变为 [36, ...]
x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
# patch_embedding: Conv3d(in_channels=36, ...)
```

**首帧 y 的构建**（以 I2V bridge 为例）：
```python
# bridge_i2v.py — _encode_first_frame()
# 1. 构建掩码: 第1帧=1, 其余=0 → [4, T_lat, lat_h, lat_w]
# 2. VAE.encode(首帧 + 零填充后续帧) → [16, T_lat, lat_h, lat_w]
# 3. y = cat([mask, vae_latent]) → [20, T_lat, lat_h, lat_w]
```

**这种注入方式的特点**：
- 直接在像素/latent 空间操作，提供 **像素级结构约束**
- 掩码告诉模型"哪帧是已知的，哪帧需要生成"
- 对首帧的外观、构图、颜色有强约束力
- 不经过 cross-attention，而是与噪声直接混合

---

#### 3.3 CLIP Image Embedding（CLIP 图像嵌入注入）

**仅作用于**：Animate 管线  
**注入对象**：参考人物图的 CLIP ViT-H/14 全局特征  
**注入位置**：`WanAnimateModel` 的 `WanAnimateCrossAttention` 层中独立的 img_attn 分支

**原理**：

```
参考人物图 → CLIP ViT-H/14.visual()
         → clip_context: [B, 257, 1280]  (CLS + 256 patch tokens)
         ↓ img_emb (MLPProj: LayerNorm → Linear → GELU → Linear → LayerNorm)
         → context_clip: [B, 257, dim]
         ↓ 拼接到 context 前面
context = cat([context_clip(257), text_context(text_len)])
         ↓ 在 WanAnimateCrossAttention.forward 中分离:
    context_img = context[:, :257]    ← CLIP 部分
    context_txt = context[:, 257:]    ← 文本(+MQ) 部分

    # 文本 cross-attn
    k = self.k(context_txt);  v = self.v(context_txt)
    x_txt = flash_attention(q, k, v)

    # 图像 cross-attn（独立的 k_img/v_img 投影）
    k_img = self.k_img(context_img);  v_img = self.v_img(context_img)
    x_img = flash_attention(q, k_img, v_img)

    # 输出 = 文本注意力 + 图像注意力
    output = x_txt + x_img
```

**代码位置** — `WanAnimateModel.forward()`：
```python
# wan/modules/animate/model_animate.py — forward()
if self.use_img_emb:
    context_clip = self.img_emb(clip_fea)   # [B, 257, dim]
    context = torch.concat([context_clip, context], dim=1)
```

**与 Context Concatenation 的关键区别**：
- CLIP 特征使用 **独立的 K/V 投影矩阵** (`k_img`, `v_img`)，与文本 cross-attn 的 `k`, `v` 是 **不同的参数**
- 两路注意力的结果是 **相加** 融合，而非共享 K/V 空间
- 这意味着 CLIP 和文本条件在注意力层中是 **并行独立处理** 再合并的

---

#### 3.4 Face Adapter（面部适配器注入）

**仅作用于**：Animate 管线  
**注入对象**：面部视频帧的运动特征  
**注入位置**：`WanAnimateModel` 的 transformer blocks 中，每 5 个 block 一次

**原理**：

```
面部视频帧 [B, 3, T, 512, 512]
    ↓ motion_encoder.get_motion()  (StyleGAN2-like Generator)
motion_vec: [B*T, 512]
    ↓ reshape → [B, T, 512]
    ↓ face_encoder (Linear + MultiheadAttention)
face_features: [B, T+1, num_heads, head_dim]   (前置一个 pad token)
    ↓ 每 5 个 block 通过 face_adapter.fuser_blocks[i] 做交叉注意力
    ↓ residual_out = CrossAttn(x, face_features)
x = x + residual_out
```

**代码位置** — `WanAnimateModel.after_transformer_block()`：
```python
# wan/modules/animate/model_animate.py
def after_transformer_block(self, block_idx, x, motion_vec, motion_masks=None):
    if block_idx % 5 == 0:  # 每 5 个 block 注入一次
        adapter_args = [x, motion_vec, motion_masks, self.use_context_parallel]
        residual_out = self.face_adapter.fuser_blocks[block_idx // 5](*adapter_args)
        x = residual_out + x
    return x
```

**特点**：
- 稀疏注入（每 5 块一次），不是每层都注入
- 面部特征作为独立的 cross-attention K/V
- 可通过传入全零面部帧禁用（归一化后 = -1，motion_encoder 输出接近零）

---

#### 3.5 MetaQuery Encoder 内部注入（Qwen3-VL → Connector → to_wan_proj）

**作用于**：T2V / I2V / Animate 全部三条管线（共享同一个 `MetaQueryEncoder`）  
**注入对象**：参考图像经 Qwen3-VL 提取的视觉-语言语义特征  
**注入位置**：`MetaQueryEncoder.encode()` → 输出 [256, 4096] → 送入 3.1 的 Context Concatenation

**原理**：

```
文本 prompt + 参考图像
    ↓ Qwen3-VL tokenizer (注入 <begin_of_img> + 256个<imgN> + <end_of_img>)
input_ids: [1, seq_len]   pixel_values: [num_patches, 1176]
    ↓ Qwen3-VL backbone forward
hidden_states: [1, seq_len, mllm_hidden_size]
    ↓ 提取 BOI~EOI 区间的 256 个 token
metaquery_hidden: [1, 256, mllm_hidden_size]
    ↓ Connector (Qwen2Encoder_24层 → Linear → GELU → Linear → RMSNorm)
connector_out: [1, 256, 2240]       ← Sana 训练维度
    ↓ to_wan_proj (Linear(2240→4096) → GELU → Linear(4096→4096))
mq_features: [1, 256, 4096]         ← Wan 需要的维度
    ↓ 拆分为 List[Tensor]
[Tensor(256, 4096)]                  ← 送入 _augment_context 拼接
```

---

### 四、原版 MetaQuery（Sana 文生图）vs 适配后 MetaQuery（Wan 视频生成）的注入方式对比

#### 4.1 原版 MetaQuery (Sana) 的注入方式

```
Qwen3-VL → Connector → prompt_embeds: [B, 256, 2240]
                            ↓
            SanaTransformer2DModel.forward(
                hidden_states = noisy_latents,
                encoder_hidden_states = prompt_embeds,  ← 直接作为 cross-attn 的 KV
                encoder_attention_mask = attention_mask,
            )
```

**特点**：
1. **独占 cross-attention**：MetaQuery 的 256 个 token 是 Sana Transformer **唯一的** cross-attention 条件。没有其他文本编码器（如 T5）参与
2. **端到端训练**：Connector 直接输出 2240 维，直接送入 Sana cross-attn，训练梯度直通
3. **MetaQuery 同时承载文本和视觉语义**：因为 Qwen3-VL 同时接收文本 prompt 和图像，所以 MetaQuery token 里既包含文本理解也包含视觉理解
4. **2D 图像生成**：Sana 是图像扩散模型，latent 是 2D：[B, C, H, W]
5. **无其他条件注入**：整个管线就只有一种条件——MetaQuery 的 cross-attention

#### 4.2 适配后 MetaQuery (Wan2.2) 的注入方式

```
Qwen3-VL → Connector → [B, 256, 2240]
              ↓ to_wan_proj (新增，未训练)
         [B, 256, 4096]
              ↓ 前置拼接到 T5 context
    aug_context = cat([MQ(256, 4096), T5(L_t5, 4096)])
              ↓ WanModel.text_embedding(Linear(4096→dim))
              ↓ 进入 WanModel 的 cross_attn
              ↓ 与其他条件共同参与生成
```

**特点**：
1. **共享 cross-attention**：MetaQuery 和 T5 文本特征 **拼接在一起**，共享同一组 Q/K/V 投影矩阵
2. **需要额外投影**：`to_wan_proj` (2240→4096) 将 MetaQuery 特征对齐到 T5 的维度空间
3. **MetaQuery 是辅助条件**：T5 文本是 Wan2.2 的原生主条件，MetaQuery 是"额外拼接"的增强条件
4. **3D 视频生成**：Wan 处理的是 3D 时空 latent：[C, F, H, W]
5. **多条件体系**：除了 context cross-attn，还有 channel concat (I2V)、CLIP embedding (Animate)、face adapter (Animate) 等

#### 4.3 对比总表

| 维度 | 原版 MetaQuery (Sana) | 适配后 MetaQuery (Wan2.2) |
|------|----------------------|--------------------------|
| **扩散模型** | SanaTransformer2DModel (图像) | WanModel / WanAnimateModel (视频) |
| **MetaQuery 地位** | 唯一条件源（独占 cross-attn） | 辅助条件（与 T5 共享 cross-attn） |
| **输出维度** | 2240 (= Sana caption_channels) | 4096 (经 to_wan_proj 投影) |
| **cross-attn KV** | 仅 MetaQuery 256 tokens | MQ 256 + T5 L_t5 tokens (拼接) |
| **KV 投影权重** | Sana Transformer 自有的 K/V Linear | WanModel cross_attn 的 K/V Linear（与 T5 共用） |
| **是否端到端训练** | ✅ 完全端到端 | ❌ to_wan_proj 未训练 |
| **其他条件** | 无 | channel concat / CLIP / face adapter |
| **生成目标** | 2D 图像 | 3D 视频 (T2V/I2V/Animate) |
| **文本编码器** | 无独立文本编码器，由 Qwen3-VL 统一处理 | T5-XXL 独立编码文本 |

---

### 五、注入方式的合理性分析

#### 5.1 原版 MetaQuery (Sana) 的注入方式 — 合理

**合理性论据**：
1. **MetaQuery 承载全部语义**：Qwen3-VL 是多模态大语言模型，同时理解文本和图像。将其输出作为唯一的 cross-attention 条件，符合"统一多模态理解"的设计哲学
2. **端到端训练保证对齐**：Connector 的 2240 维输出直接用 Sana 的重建损失训练，语义空间完全对齐
3. **简洁高效**：单一条件源→单一 cross-attention，无需处理多条件融合的复杂性

**潜在局限**：
- 256 个 token 需要同时编码文本（可能很长的 prompt）和图像语义，信息瓶颈明显
- 没有独立文本编码器的补充，纯文本生成（无参考图）时表达力可能弱于 T5

#### 5.2 适配后 MetaQuery (Wan) 的注入方式 — Context Concatenation — 合理

**合理性论据**：
1. **最小侵入性**：不修改 WanModel 的任何内部结构，只在输入侧拼接新 token，保持 Wan 预训练权重完全不变
2. **天然兼容 Wan 的架构**：Wan 的 `text_embedding` 和 `cross_attn` 对输入 token 数量是透明的（只要 `text_len` 够大即可）。拼接新 token 只是相当于"更长的提示词"
3. **T5 作为基线保底**：即使 to_wan_proj 未训练（MQ 特征质量差），T5 文本条件仍然正常工作，不会因为添加 MetaQuery 而破坏原始文本引导能力
4. **灵活的多条件互补**：MetaQuery 提供语义级引导，channel concat 提供像素级引导，CLIP 提供全局风格引导，face adapter 提供面部动作引导 — 各层级分工明确

**潜在局限**：
- MQ 和 T5 共享同一组 K/V 投影权重，但两者来自完全不同的语义空间（MQ 来自 Qwen3-VL，T5 来自独立的 T5-XXL）。共享投影可能导致 attention 分数不平衡
- MQ 前置拼接 → 在 padding 后可能被截断或受 position bias 影响
- to_wan_proj 未训练的问题（详见附录 A）

#### 5.3 I2V Channel Concatenation — 合理

**合理性论据**：
1. **Wan 原生设计**：`WanModel(model_type='i2v')` 的 `in_dim=36`（16 noise + 20 条件）就是为 channel concat 设计的，`patch_embedding = Conv3d(in=36, ...)`
2. **像素级精确控制**：相比 cross-attention 的"语义引导"，channel concat 在 latent 空间直接提供像素级结构信息，对首帧的外观保真度更高
3. **掩码机制**：4 通道掩码明确告诉模型"哪帧是已知内容，哪帧需要生成"，符合 I2V 的天然需求

#### 5.4 CLIP Image Embedding（Animate 独立图像注意力）— 合理

**合理性论据**：
1. **独立 K/V 分支**：`WanAnimateCrossAttention` 为 CLIP 特征分配了独立的 `k_img`/`v_img` 投影，避免与文本共享投影带来的语义混淆
2. **全局语义互补**：CLIP ViT-H/14 提取的 257 token 是全局视觉语义（人物外观、风格），与 MetaQuery 的细粒度语义形成互补
3. **加法融合**：`output = text_attn + img_attn`，两路注意力独立计算后相加，数学上等价于多头注意力的多路并行，是成熟的融合策略

#### 5.5 Face Adapter（稀疏面部注入）— 合理

**合理性论据**：
1. **稀疏注入**：每 5 个 block 注入一次（共 8 次 / 40 blocks），避免面部条件过度主导，平衡面部保真与整体自由生成
2. **残差连接**：`x = x + face_adapter_output`，面部条件以残差方式注入，不会破坏主干特征
3. **可禁用**：传入全零面部帧 → 归一化后 = -1 → motion_encoder 输出接近零 → face_adapter 实质上不起作用。这保证了面部条件是"可选增量"

---

### 六、多条件信息流全景图

#### T2V (bridge.py) — 两条信息流

```
                    input_prompt
                    ┌─────┴─────┐
                    ▼           ▼
              T5-XXL 编码    Qwen3-VL + Connector + to_wan_proj
                    │           │
              [L_t5, 4096]  [256, 4096]
                    │           │
                    └─────┬─────┘
                    cat (前置 MQ)
                          │
                    [256+L_t5, 4096]
                          ↓
                text_embedding(4096→dim)
                          ↓
                每个 block 的 cross_attn (K, V)
                          ↓
                    去噪 latent
                          ↓
                    VAE decode → 视频
```

#### I2V (bridge_i2v.py) — 三条信息流

```
              input_prompt                first_frame
              ┌─────┴─────┐              │
              ▼           ▼              ▼
        T5-XXL 编码    Qwen3-VL      VAE.encode + mask
              │           │              │
        [L_t5, 4096]  [256, 4096]    y: [20, T, H, W]
              │           │              │
              └─────┬─────┘              │
              cat (前置 MQ)               │
                    │                    │
              [256+L_t5, 4096]           │
                    ↓                    ↓
          text_embedding → cross_attn   channel concat with noise [36ch]
                    ↓                    ↓
                    └────────┬───────────┘
                     WanModel (i2v) forward
                             ↓
                       去噪 → VAE decode → 视频
```

#### Animate (bridge_animate.py) — 五条信息流

```
        input_prompt      ref_image         face_source
        ┌─────┴─────┐    ┌────┴────┐        │
        ▼           ▼    ▼    ▼    ▼        ▼
  T5-XXL 编码   Qwen3-VL VAE  CLIP  ──   motion_encoder
        │           │    │    │           face_encoder
  [L_t5, 4096] [256,4096] │  [257,1280]   [T+1, H, C]
        │           │    │    │              │
        └─────┬─────┘    │    ↓ img_emb     │
        cat(前置MQ)       │  [257, dim]      │
              │           │    │             │
        [256+L_t5, 4096]  │    │             │
              ↓           │    │             │
        text_embedding    │    │             │
              ↓           │    │             │
        [text_len, dim]   │    │             │
              ↓           ↓    ↓             ↓
        ┌─ cross_attn ←┬─── cat(clip, text) │
        │               │                   │
        │         channel concat [36ch]      │
        │               │                   │
        │               ↓                   │
        │     patch_embedding               │
        │               ↓                   │
        │    ┌──── transformer blocks ──────┤
        │    │          ↓                   │
        │    │   每5个block: face_adapter ←─┘
        │    │          ↓
        │    └──── 去噪完成
        │               ↓
        └──────── VAE decode → 视频
```

---

### 七、总结与设计建议

| 对比项 | 原版 MetaQuery (Sana) | 适配后 MetaQuery (Wan) | 评价 |
|--------|----------------------|------------------------|------|
| 注入方式 | 独占 cross-attn encoder_hidden_states | Context 拼接进 T5 context | 两者都合理，但后者更灵活 |
| 条件层级 | 仅语义级 | 语义级 + 像素级 + CLIP + 面部 | 后者更丰富，多层级互补 |
| MetaQuery 角色 | 唯一条件源 | 辅助增强条件 | 后者更安全（有 T5 保底） |
| 训练对齐 | 端到端 | to_wan_proj 未训练 | 前者更好，后者需微调 |
| 架构侵入性 | N/A | 零侵入（仅拼接 token） | 后者设计优雅 |

**设计建议**：

1. **短期可用**：当前 Context Concatenation 方案可以直接使用，即使 to_wan_proj 未训练，T5 文本条件仍正常工作，MetaQuery 相当于"额外噪声token"对结果影响有限
2. **中期改进**：微调 to_wan_proj + 可选 WanModel cross-attn LoRA，让 MetaQuery 特征真正对齐 Wan 的语义空间
3. **长期优化**：考虑为 MetaQuery 分配 **独立的 K/V 投影**（类似 Animate 中 CLIP 的做法），避免与 T5 共享投影带来的注意力分数不平衡问题
