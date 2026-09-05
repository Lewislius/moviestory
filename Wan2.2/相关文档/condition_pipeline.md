# MetaQuery + Wan2.2 条件注入全流程详解

## 目录

1. [概述](#1-概述)
2. [两种初始化模式对比](#2-两种初始化模式对比)
3. [核心组件架构](#3-核心组件架构)
4. [模式 A：METAQUERY_CKPT = None（从预训练模型直接初始化）](#4-模式-ametaquery_ckpt--none从预训练模型直接初始化)
5. [模式 B：METAQUERY_CKPT = 训练好的路径（从 checkpoint 加载）](#5-模式-bmetaquery_ckpt--训练好的路径从-checkpoint-加载)
6. [T2V 全流程（demo_metaquery_wan.py）](#6-t2v-全流程demo_metaquery_wanpy)
7. [Animate 全流程（demo_metaquery_animate.py）](#7-animate-全流程demo_metaquery_animatepy)
8. [关键问答](#8-关键问答)

---

## 1. 概述

MetaQuery 是一种将多模态大语言模型（MLLM，此处为 Qwen3-VL）的视觉-语言理解能力注入扩散模型（Wan2.2）的方法。其核心思想是：

> **在 MLLM 的 prompt 中植入 N 个特殊 token（MetaQuery tokens），让 MLLM 在前向传播过程中将图像和文本的语义信息"蒸馏"到这 N 个 token 的隐藏状态中，再通过 Connector 投影到扩散模型的 context 空间，最终拼接到 T5 文本编码之后参与 cross-attention。**

```
                        ┌─────────────────────────────────────────────────┐
   参考图 + 文本 ──────►│  Qwen3-VL  →  BOI + <img0>~<img255> + EOI     │
                        │  提取 256 个 MetaQuery 隐藏状态                  │
                        │        ↓                                        │
                        │  Connector (24层 Qwen2Encoder + 线性投影)        │
                        │        ↓                                        │
                        │  to_wan_proj (connector_dim → 4096)             │
                        └──────────────────┬──────────────────────────────┘
                                           │ [256, 4096]
                                           ▼
                     ┌─────────────────────────────────────────┐
   文本 ────────────►│  T5 Encoder → [L_t5, 4096]              │
                     └──────────────────┬──────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │  Context = concat([MQ, T5], dim=0)       │
                     │  shape: [256 + L_t5, 4096]               │
                     └──────────────────┬───────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │  Wan2.2 DiT Cross-Attention              │
                     │  Key, Value 来自拼接后的 Context          │
                     │  每去噪步都使用此 Context                  │
                     └──────────────────────────────────────────┘
```

---

## 2. 两种初始化模式对比

| 特性 | 模式 A：`METAQUERY_CKPT = None` | 模式 B：`METAQUERY_CKPT = 路径` |
|------|--------------------------------|-------------------------------|
| **Qwen3-VL backbone** | ✅ 有，加载 HuggingFace 原始预训练权重 | ✅ 有，加载 checkpoint 中微调后的权重 |
| **MetaQuery tokens** | ✅ 有，256 个特殊 token（新增到词表） | ✅ 有，256 个特殊 token（训练过的 embedding） |
| **Connector** | ✅ 有，**随机初始化**的 24 层 Qwen2Encoder | ✅ 有，**训练好的** 24 层 Qwen2Encoder |
| **to_wan_proj 投影层** | ✅ 有，Xavier 随机初始化 | ✅ 有，Xavier 随机初始化 |
| **生成效果** | ⚠️ 可运行但效果差（Connector 未对齐） | ✅ 效果好（Connector 已学会对齐） |
| **适用场景** | 调试流程 / 验证管线完整性 | 正式生产使用 |
| **配置参数** | 需要设置 `QWEN3_VL_MODEL_ID` | 不需要（从 checkpoint 中自动读取） |

### 关键区别：Connector 是否经过训练

**两种模式都有完整的 MetaQuery + Connector 结构**，区别仅在于参数是否经过训练：

```
模式 A (无 checkpoint):
  Qwen3-VL [预训练权重] → MetaQuery 隐藏状态 → Connector [随机权重] → 投影 → Wan2.2
                                                    ↑
                                              参数未对齐，输出近似噪声

模式 B (有 checkpoint):
  Qwen3-VL [微调权重] → MetaQuery 隐藏状态 → Connector [训练权重] → 投影 → Wan2.2
                                                   ↑
                                             参数已对齐，输出有意义的视觉语义
```

---

## 3. 核心组件架构

### 3.1 MetaQueryEncoder（`wan/metaquery/encoder.py`）

负责将参考图像 + 文本编码为 MetaQuery 条件特征。

```
MetaQueryEncoder
├── mllm_model (MLLMInContext)
│   ├── mllm_backbone (Qwen3VLForConditionalGeneration)
│   │   ├── model (Qwen3VLModel)
│   │   │   ├── embed_tokens    ← 词表已扩展 +258 个 token (256 MQ + BOI + EOI)
│   │   │   ├── layers[0..27]   ← Transformer 层
│   │   │   └── visual          ← Qwen3-VL 视觉编码器 (ViT)
│   │   └── lm_head → nn.Identity()  ← 替换为恒等映射，直接取隐藏状态
│   │
│   ├── connector (nn.Sequential)
│   │   ├── Qwen2Encoder (24 层双向 Transformer)  ← 核心对齐模块
│   │   │   └── hidden_size = mllm_hidden_size (2B=1536)
│   │   ├── nn.Linear(1536 → connector_out_dim)
│   │   ├── nn.GELU(tanh)
│   │   ├── nn.Linear(connector_out_dim → connector_out_dim)
│   │   └── RMSNorm(connector_out_dim)
│   │
│   ├── boi_token_id  ← <begin_of_img> 的 token ID
│   └── eoi_token_id  ← <end_of_img> 的 token ID
│
├── to_wan_proj (nn.Sequential)  ← 额外投影层，bridge 初始化时创建
│   ├── nn.Linear(connector_out_dim → 4096)
│   ├── nn.GELU(tanh)
│   └── nn.Linear(4096 → 4096)
│
└── tokenizer (AutoProcessor)
```

**维度链：**
```
Qwen3-VL-2B hidden_size = 1536
      ↓  (Connector: Qwen2Encoder + Linear + GELU + Linear + RMSNorm)
connector_out_dim = 2240 (Sana 的 caption_channels)
      ↓  (to_wan_proj: Linear + GELU + Linear)
wan_text_dim = 4096 (Wan2.2 的 T5 embedding 维度)
```

### 3.2 MetaQueryWanBridge（`wan/metaquery/bridge.py`）

将 MetaQuery 编码器与 Wan2.2 T2V 管线连接的桥梁。

### 3.3 MetaQueryWanAnimateBridge（`wan/metaquery/bridge_animate.py`）

将 MetaQuery 编码器与 Wan2.2 Animate 管线连接的桥梁，额外支持参考图、CLIP、面部条件。

---

## 4. 模式 A：METAQUERY_CKPT = None（从预训练模型直接初始化）

### 4.1 初始化流程

```python
# demo_metaquery_wan.py 中:
METAQUERY_CKPT = None
QWEN3_VL_MODEL_ID = "E:\models\Qwen3-VL-2B-Instruct"   # ← 此时必须指定
```

**初始化调用链：**

```
MetaQueryWanBridge.__init__()
  │
  ├── MetaQueryEncoder.__init__()
  │     │
  │     ├── use_checkpoint = False  (因为 METAQUERY_CKPT = None)
  │     │
  │     ├── _init_from_pretrained()       ← 模式 A 走这条路
  │     │     │
  │     │     ├── 构建 MLLMInContextConfig
  │     │     │     mllm_id = "Qwen3-VL-2B-Instruct"
  │     │     │     diffusion_model_id = "Sana_1600M_512px_diffusers"  (仅用于确定 connector 输出维度)
  │     │     │     num_metaqueries = 256
  │     │     │     connector_num_hidden_layers = 24
  │     │     │
  │     │     ├── MLLMInContext(config)
  │     │     │     │
  │     │     │     ├── 加载 Qwen3VLForConditionalGeneration (预训练权重)
  │     │     │     ├── resize_token_embeddings(原词表 + 258)
  │     │     │     │     → 新增的 258 个 embedding 随机初始化
  │     │     │     ├── lm_head = nn.Identity()  (丢弃语言建模头)
  │     │     │     │
  │     │     │     ├── 加载 SanaTransformer2DModel (仅为了读取 connector_out_dim)
  │     │     │     │     connector_out_dim = caption_channels = 2240
  │     │     │     │
  │     │     │     ├── 构建 Connector:
  │     │     │     │     Qwen2Encoder(1536, 24层) + Linear(1536→2240) + GELU + Linear(2240→2240) + RMSNorm
  │     │     │     │     → 全部随机初始化 ⚠️
  │     │     │     │
  │     │     │     └── 注册 BOI/EOI/MQ 特殊 token
  │     │     │
  │     │     ├── 删除 MLLMInContext.transformer (Sana, 不需要, 释放显存)
  │     │     └── 设置 eval() + requires_grad_(False)
  │     │
  │     └── _init_projection_and_validate()
  │           ├── 创建 to_wan_proj: Linear(2240→4096) + GELU + Linear(4096→4096)
  │           │     → Xavier 初始化
  │           └── 完整性验证 (BOI/EOI token, connector 参数量, 维度链等)
  │
  └── 记录/扩展 text_len: 原始值 + 256
```

### 4.2 各组件参数状态

| 组件 | 参数来源 | 参数状态 |
|------|---------|---------|
| Qwen3-VL backbone | HuggingFace 预训练 | ✅ 有意义的预训练权重 |
| 原始词表 embedding | HuggingFace 预训练 | ✅ 有意义 |
| 新增 258 个 embedding (BOI, EOI, img0~img255) | 随机初始化 | ⚠️ 随机，但 backbone 可通过 attention 整合上下文 |
| Connector (Qwen2Encoder) | 随机初始化 | ❌ 未学习对齐映射 |
| Connector (Linear+GELU+Linear+RMSNorm) | 随机初始化 | ❌ 未学习对齐映射 |
| to_wan_proj | Xavier 初始化 | ⚠️ 近似合理的随机映射 |

### 4.3 为什么模式 A 还能运行？

虽然 Connector 是随机初始化的，但整个管线仍然可以端到端运行：

1. **Qwen3-VL backbone 的预训练权重可以产生有意义的隐藏状态**
   - 文本部分：预训练的 language model 自然会对文本 token 产出有语义的表示
   - 图像部分：Qwen3-VL 内置的 ViT 会对图像产出有意义的 patch embedding
   - MetaQuery token 部分：虽然这 256 个 token 的 embedding 是随机的，但经过 28 层 Transformer 后，通过 attention 机制，它们会从周围的文本和图像 token 中吸收部分信息

2. **Connector 虽然随机，但不会产生 NaN 或 Inf**
   - 输出是有界的浮点数，只是不是"对齐良好"的表示

3. **to_wan_proj 将 Connector 输出投影到 4096 维**
   - Xavier 初始化保证输出量级合理

4. **最终效果**
   - MetaQuery 条件**存在但质量差**，近似随机噪声条件
   - 生成的视频主要受 T5 文本条件驱动
   - MetaQuery 部分可能产生轻微干扰或无明显效果

---

## 5. 模式 B：METAQUERY_CKPT = 训练好的路径（从 checkpoint 加载）

### 5.1 初始化流程

```python
# demo_metaquery_wan.py 中:
METAQUERY_CKPT = "/home/.../checkpoints/output/qwen3vl2b_inst_small"
# QWEN3_VL_MODEL_ID 此时不生效
```

**初始化调用链：**

```
MetaQueryWanBridge.__init__()
  │
  ├── MetaQueryEncoder.__init__()
  │     │
  │     ├── use_checkpoint = True  (因为 METAQUERY_CKPT 非空)
  │     │
  │     ├── _init_from_checkpoint()       ← 模式 B 走这条路
  │     │     │
  │     │     ├── find_newest_checkpoint(checkpoint_path)
  │     │     │     → 在目录下查找最新的 checkpoint-XXXX/ 子文件夹
  │     │     │     → 例如: .../qwen3vl2b_inst_small/checkpoint-300/
  │     │     │
  │     │     ├── MetaQuery.from_pretrained(ckpt, ...)
  │     │     │     │
  │     │     │     ├── MetaQuery.__init__()
  │     │     │     │     └── self.model = MLLMInContext(config)
  │     │     │     │           ├── 加载 Qwen3VLForConditionalGeneration
  │     │     │     │           ├── resize_token_embeddings
  │     │     │     │           ├── 构建 Connector
  │     │     │     │           └── 加载 SanaTransformer2DModel
  │     │     │     │
  │     │     │     └── from_pretrained 覆盖权重
  │     │     │           → **所有参数被 checkpoint 中保存的权重替换** ✅
  │     │     │           → backbone (可能已微调)
  │     │     │           → embedding (包含训练过的 MQ token embedding)
  │     │     │           → connector (已学会从 mllm hidden → diffusion context 的对齐映射)
  │     │     │
  │     │     ├── 提取 mllm_model, tokenizer
  │     │     ├── 删除不需要的 VAE / Diffusion 部分
  │     │     └── 释放 MetaQuery 外壳对象
  │     │
  │     └── _init_projection_and_validate()
  │           ├── 创建 to_wan_proj: Linear(2240→4096) + GELU + Linear(4096→4096)
  │           │     → Xavier 初始化 (注意: 此层两种模式都是新初始化的!)
  │           └── 完整性验证
  │
  └── 记录/扩展 text_len
```

### 5.2 各组件参数状态

| 组件 | 参数来源 | 参数状态 |
|------|---------|---------|
| Qwen3-VL backbone | checkpoint (Stage 1+2 微调后) | ✅ embedding 层已微调，适配 MQ token |
| 原始词表 embedding | checkpoint | ✅ 原始 embedding 冻结 (freeze_hook 梯度归零) |
| 新增 258 个 embedding | checkpoint (训练过) | ✅ 已学习到如何"查询"信息的初始表示 |
| Connector (Qwen2Encoder) | checkpoint (训练过) | ✅ 已学习 mllm_hidden → diffusion context 的对齐 |
| Connector (Linear+GELU+Linear+RMSNorm) | checkpoint (训练过) | ✅ 已学习维度投影 |
| to_wan_proj | Xavier 初始化 | ⚠️ 新初始化 (两种模式相同) |

### 5.3 训练阶段说明

MetaQuery 训练分两个阶段：

#### Stage 1：Text-to-Image 预训练 (`qwen3vl2b_t2i_small`)
- **数据**：CC12M（文本-图像对）的 1/1000 子集 ≈ 1.2 万条
- **目标**：学习 Connector 的基础对齐能力
- **冻结策略**：
  - 冻结 VAE
  - 冻结 Qwen3-VL backbone（除 embed_tokens 外）
  - Connector 全部可训练
- **损失函数**：Flow Matching Loss（在 Sana diffusion model 上计算）
- **学习率**：1e-4

#### Stage 2：Instruction Tuning (`qwen3vl2b_inst_small`)
- **数据**：MetaQuery-Instruct-2.4M 的 1/1000 子集 ≈ 2400 条
- **目标**：让 Connector 学习更丰富的指令感知对齐
- **在 Stage 1 基础上继续训练**
- **学习率**：5e-5（更小，精细调整）

```
Stage 1 (t2i_small)  ──────►  Stage 2 (inst_small)
   Connector 学基础对齐            Connector 学指令对齐
   1.2万条 CC12M                  2400条 Instruct 数据
        ↓                              ↓
   中间 checkpoint               最终 checkpoint ← 应该用这个!
```

---

## 6. T2V 全流程（demo_metaquery_wan.py）

### 完整的输入→输出流程

```
输入:
  - PROMPT: "一只橘猫慵懒地躺在阳光洒落的窗台上..."
  - REFERENCE_IMAGES: [reference_scene.jpg]
  - 参数: SIZE=(1280,720), FRAME_NUM=81, SAMPLING_STEPS=50, ...

输出:
  - video.mp4: 1280×720, 81帧, 16fps ≈ 5秒视频
```

#### Step 1: 初始化 Wan2.2 T2V 管线

```python
cfg = WAN_CONFIGS["t2v-A14B"]     # 14B 参数的文本转视频配置
wan_pipeline = wan.WanT2V(
    config=cfg,
    checkpoint_dir=WAN_CKPT_DIR,  # 加载 T5, VAE, DiT 权重
    device_id=0,
)
```

加载的组件：
- **T5 文本编码器**：将文本编码为 [L, 4096] 的 context 向量
- **VAE**：视频编解码器，latent space ↔ pixel space
- **DiT (WanModel)**：扩散 Transformer，包含 cross-attention 层
  - `high_noise_model`：高噪声阶段模型
  - `low_noise_model`：低噪声阶段模型

#### Step 2: 初始化 MetaQuery Bridge

```python
bridge = MetaQueryWanBridge(
    wan_pipeline=wan_pipeline,
    metaquery_checkpoint=METAQUERY_CKPT,      # None 或训练路径
    num_metaqueries=256,
    dtype=torch.bfloat16,
    mllm_id=QWEN3_VL_MODEL_ID,               # 仅 CKPT=None 时生效
)
```

→ 内部初始化 MetaQueryEncoder（见第 4/5 节）

#### Step 3: 调用 bridge.generate()

**generate() 内部完整流程：**

```
bridge.generate(prompt, images, size, frame_num, ...)
│
├── [1] 形状预算
│     target_shape = (z_dim, T', H', W')  # latent 空间的尺寸
│     seq_len = ceil(T'×H'×W' / patch²) × sp_size
│
├── [2] T5 文本编码    ← 标准 Wan2.2 文本条件
│     context      = T5_encoder([prompt])       # → [L_t5, 4096]  有条件
│     context_null = T5_encoder([neg_prompt])    # → [L_t5, 4096]  无条件 (CFG)
│
├── [3] MetaQuery 编码 ← MetaQuery 视觉-语言条件
│     │
│     ├── mq_encoder.encode([prompt], [[ref_image]])
│     │     │
│     │     ├── tokenize()  构造 input:
│     │     │     "system_prompt... user: <image> prompt文本
│     │     │      <begin_of_img><img0><img1>...<img255><end_of_img>"
│     │     │     │
│     │     │     ├── 文本部分 → token IDs
│     │     │     ├── 图像部分 → Qwen3-VL vision encoder 处理为 pixel_values
│     │     │     └── MQ部分 → BOI + 256个<img> token + EOI
│     │     │
│     │     ├── Qwen3-VL 前向:
│     │     │     input_ids + pixel_values + attention_mask
│     │     │     → 28层 Transformer 前向传播
│     │     │     → lm_head (Identity) → hidden_states [B, L, 1536]
│     │     │
│     │     ├── 提取 MQ token 的隐藏状态:
│     │     │     找到 BOI 和 EOI 位置
│     │     │     提取 BOI 到 EOI 之间的 256 个 token 的 hidden states
│     │     │     → [B, 256, 1536]
│     │     │
│     │     ├── Connector 前向:
│     │     │     Qwen2Encoder(24层双向) → Linear → GELU → Linear → RMSNorm
│     │     │     [B, 256, 1536] → [B, 256, 2240]
│     │     │
│     │     └── to_wan_proj 投影:
│     │           Linear → GELU → Linear
│     │           [B, 256, 2240] → [B, 256, 4096]
│     │
│     ├── mq_context      = encode([prompt], [[ref_image]])   # [256, 4096]
│     └── mq_context_null = encode([neg_prompt], None)        # [256, 4096] 无图像
│
├── [4] Context 拼接
│     aug_context      = concat([mq_context, t5_context])     # [256+L_t5, 4096]
│     aug_context_null = concat([mq_null, t5_null])           # [256+L_t5, 4096]
│     │
│     └── MetaQuery 特征前置，优先被 cross-attention 看到
│
├── [5] 扩展 WanModel.text_len
│     text_len: 原始值 → 原始值 + 256
│     → 确保 DiT 内部 padding 能容纳加长后的 context
│
├── [6] 初始化噪声
│     noise = randn(z_dim, T', H', W')  # 纯高斯噪声
│
├── [7] 去噪循环 (50步)    ← 每步都使用增强后的 context
│     for t in timesteps:
│     │
│     │   选择模型 (高/低噪声)
│     │   │
│     │   noise_pred_cond   = model(latents, t, context=aug_context)
│     │   noise_pred_uncond = model(latents, t, context=aug_context_null)
│     │   │
│     │   │  WanModel.forward 内部:
│     │   │    context [256+L_t5, 4096]
│     │   │    → text_embedding (Linear) → [256+L_t5, dim]
│     │   │    → padding 到 text_len (= 原始 + 256)
│     │   │    → cross_attn: Q=latent, K/V=padded_context
│     │   │    → 每个 WanAttentionBlock 的 cross-attention 都参与
│     │   │
│     │   CFG: noise_pred = uncond + scale × (cond - uncond)
│     │   │
│     │   latents = scheduler.step(noise_pred, t, latents)
│     │
│     └── 50步后 latents 收敛为干净的 latent
│
├── [8] VAE 解码
│     video_tensor = vae.decode(latents)  # [C=3, F=81, H=720, W=1280]
│
├── [9] 恢复 WanModel.text_len → 原始值
│
└── [10] 保存视频
      save_video(video_tensor, "demo_xxx.mp4", fps=16)
```

---

## 7. Animate 全流程（demo_metaquery_animate.py）

Animate 流程比 T2V 更复杂，引入了**四重条件**：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    四重条件注入架构                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  条件1: 参考图条件 (Channel Concat)                                   │
│    ref_image → VAE encode → [4ch mask + 16ch latent] = 20ch         │
│    + 16ch noise = 36ch total → WanAnimateModel.in_dim = 36          │
│                                                                      │
│  条件2: CLIP 视觉条件 (Cross-Attention)                              │
│    ref_image → CLIP ViT-H/14 → [257, dim] → img_emb 投影            │
│    → 独立的 cross-attention (非 text context)                        │
│                                                                      │
│  条件3: MetaQuery 语义条件 (Context Concat)                          │
│    ref_image + text → Qwen3-VL → Connector → to_wan_proj            │
│    → [256, 4096] concat 到 T5 context 前面                           │
│    → text cross-attention 参与                                       │
│                                                                      │
│  条件4: 面部条件 (Face Adapter)                                      │
│    face_video → motion_encoder → face_encoder → face_adapter         │
│    → 每 5 层 transformer block 注入面部动作特征                       │
│                                                                      │
│  (骨架条件: 禁用，传零 pose_latents)                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 完整流程：

```
bridge.generate(prompt, ref_image, face_source, mq_reference_images, ...)
│
├── [Step 1/7] 准备参考人物图
│     ref_np = padding_resize(ref_image, H, W)
│
├── [Step 2/7] 准备面部视频帧
│     face_frames = load_face_video(face_source, frame_num)  # list[np.ndarray]
│     或 全零帧 (无面部条件)
│
├── [Step 3/7] T5 文本编码
│     context      = T5([prompt])       → [L_t5, 4096]
│     context_null = T5([neg_prompt])   → [L_t5, 4096]
│
├── [Step 4/7] MetaQuery 编码  (与 T2V 相同)
│     mq_context      = mq_encoder.encode([prompt], [mq_ref_images])
│     mq_context_null = mq_encoder.encode([neg_prompt], None)
│     → [256, 4096]
│
├── [Step 5/7] Context 拼接  (与 T2V 相同)
│     aug_context = concat([mq, t5])       → [256+L_t5, 4096]
│     aug_context_null = concat([mq_null, t5_null])
│
├── [Step 6/7] CLIP 编码参考图
│     clip_context = CLIP.visual(ref_image)  → [257, dim]
│
├── [Step 7/7] 逐 Clip 去噪循环
│     │
│     │  扩展 text_len = 原始 + 256
│     │
│     │  while 还有 clip:
│     │  │
│     │  ├── 构建面部张量
│     │  │     face_pixel_values = face_frames[start:end] → [1, C, T, 512, 512]
│     │  │
│     │  ├── 构建参考图条件 y
│     │  │     ref_image → VAE encode → ref_latents
│     │  │     mask + ref_latents → y_ref      (参考图掩码 + latent)
│     │  │     prev_clip_tail → VAE encode → y_reft  (时序引导帧)
│     │  │     y = concat([y_ref, y_reft])
│     │  │
│     │  ├── 零 pose latent (骨架禁用)
│     │  │     pose_latents = zeros(1, 16, T', H', W')
│     │  │
│     │  ├── 去噪循环 (20步):
│     │  │     noise_pred = WanAnimateModel(
│     │  │         latents, t,
│     │  │         context = aug_context,        ← MetaQuery + T5
│     │  │         clip_fea = clip_context,       ← CLIP 视觉
│     │  │         y = [y],                       ← 参考图条件 (channel concat)
│     │  │         pose_latents = pose_latents,   ← 零 (骨架禁用)
│     │  │         face_pixel_values = face_pv,   ← 面部条件
│     │  │     )
│     │  │
│     │  └── VAE 解码当前 clip → 拼接到 all_out_frames
│     │
│     └── 恢复 text_len
│
└── 拼接所有 clip → video [C=3, F, H, W]
```

### WanAnimateModel 内部条件注入点

```
WanAnimateModel.forward(x, t, context, clip_fea, y, pose_latents, face_pixel_values):
│
├── x: [16, T', H', W'] 噪声 latent
│
├── 准备 y 条件:
│     y_concat = concat([x, y], dim=channel)   # [16+20=36, T'+1, H', W']
│     → patch_embedding 处理                    # 条件1: 参考图 channel concat
│
├── 准备 CLIP 条件:
│     img_emb = self.img_emb(clip_fea)         # 条件2: CLIP 视觉编码
│     → 独立 cross-attention
│
├── 准备面部条件:
│     face_feat = motion_encoder(face_pv)       # 条件4: 面部运动特征
│     face_feat = face_encoder(face_feat)
│     → face_adapter 每 5 层注入
│
├── 文本 context 处理:
│     context_emb = text_embedding(context)     # 条件3: MetaQuery + T5
│     → padding 到 text_len                     #   context = [MQ:256 | T5:L_t5]
│     → 每个 block 的 cross_attn:
│         Q = latent, K/V = context_emb
│         → MQ 部分在前，T5 部分在后
│         → attention 自动分配权重
│
└── 经过 N 个 WanAttentionBlock 后输出 noise_pred
```

---

## 8. 关键问答

### Q1: 没有 checkpoint 时有 MetaQuery 和 Connector 吗？

**有！** 两种模式都完整构建 MetaQuery 架构：
- 256 个 MetaQuery token（`<img0>` ~ `<img255>`）
- BOI / EOI 特殊 token
- 24 层 Qwen2Encoder Connector
- to_wan_proj 投影层

区别在于：
- **无 checkpoint**：Connector 是随机初始化的，MetaQuery token embedding 也是随机的
- **有 checkpoint**：Connector 经过训练，MetaQuery token embedding 经过训练

### Q2: 无 checkpoint 时 MetaQuery 条件如何起作用？

```
1. 输入构造阶段:
   - 文本 + 图像 + 256 个 MQ token 一起送入 Qwen3-VL
   - MQ token 的 embedding 是随机的

2. Qwen3-VL 前向传播:
   - 经过 28 层 Transformer
   - 通过 attention 机制，MQ token 从周围的文本/图像 token 吸收信息
   - 输出：256 个 MQ token 位置的 hidden states [256, 1536]
   → 有一定语义信息（来自预训练 backbone 的能力），但不够精确

3. Connector 处理:
   - 24 层 Qwen2Encoder 处理 [256, 1536]
   - 但因为是随机初始化的，处理结果接近随机变换
   → 输出 [256, 2240]，有信号但不对齐

4. to_wan_proj:
   - 投影到 [256, 4096]
   → Xavier 初始化，量级合理但方向无意义

5. 效果:
   - 这 256 个 token 会参与 Wan2.2 的 cross-attention
   - 但因为信号不对齐，对生成结果的引导能力很弱
   - 生成质量主要由 T5 文本条件主导
```

### Q3: 该用 `t2i_small` 还是 `inst_small`？

**应该用 `inst_small`（Stage 2 的输出）。**

- `t2i_small` 是 Stage 1 的中间产物，仅学了基础对齐
- `inst_small` 在 `t2i_small` 基础上继续训练，学了更好的指令感知对齐
- **Stage 2 包含了 Stage 1 的全部成果**

```python
METAQUERY_CKPT = "/home/liuzhirui/model/Qwen3-VL-main/metaquery-main/checkpoints/output/qwen3vl2b_inst_small"
```

### Q4: to_wan_proj 为什么两种模式都是新初始化的？

`to_wan_proj` 是 `MetaQueryEncoder` 在 Bridge 初始化时额外创建的投影层，将 Connector 输出（2240 维，Sana 的 caption_channels）进一步投影到 Wan2.2 需要的 4096 维。

这个层**不在训练好的 checkpoint 中**（训练时用的是 Sana diffusion model，target dim=2240；而 Wan2.2 需要 4096），所以两种模式都是 Xavier 初始化。

如果需要更好的效果，可以考虑：
- 在 Wan2.2 上对 `to_wan_proj` 进行少量微调
- 或者直接扩展训练流程。在 Wan2.2 的 diffusion loss 下训练这一层

### Q5: CFG（Classifier-Free Guidance）在 MetaQuery 中如何起作用？

每一步去噪都计算两次预测：
- **有条件**：context = MQ(prompt + ref_image) ⊕ T5(prompt)
- **无条件**：context = MQ(neg_prompt, 无图像) ⊕ T5(neg_prompt)

```
noise_pred = noise_pred_uncond + guide_scale × (noise_pred_cond - noise_pred_uncond)
```

MetaQuery 有条件和无条件的差异体现在：
- 有条件：Qwen3-VL 看到了参考图像 + 正面文本
- 无条件：Qwen3-VL 只看到负面文本，无图像输入

这使得 CFG 能放大 MetaQuery 视觉条件的影响。

---

## 附录：文件对照表

| 文件 | 作用 |
|------|------|
| `wan/metaquery/encoder.py` | MetaQueryEncoder：MLLM + Connector + 投影，两种初始化 |
| `wan/metaquery/bridge.py` | MetaQueryWanBridge：T2V 管线的条件注入桥梁 |
| `wan/metaquery/bridge_animate.py` | MetaQueryWanAnimateBridge：Animate 管线的条件注入桥梁 |
| `metaquery-main/models/model.py` | MLLMInContext：封装 Qwen3-VL + Connector 的完整模型 |
| `metaquery-main/models/metaquery.py` | MetaQuery：训练时的完整模型（含 VAE + Diffusion） |
| `demo_metaquery_wan.py` | T2V Demo 脚本 |
| `demo_metaquery_animate.py` | Animate Demo 脚本 |
