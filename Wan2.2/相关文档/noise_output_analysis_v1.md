# MetaQuery + Wan2.2 Animate 噪声输出根因分析

## 目录

1. [问题现象](#1-问题现象)
2. [训练阶段分析：qwen3vl2b_sana_inst_small 到底训练了什么](#2-训练阶段分析)
3. [关键错配：训练 vs 推理的目标 Diffusion 模型完全不同](#3-关键错配)
4. [Connector 输出有图/无图几乎一致的原因](#4-connector-输出分析)
5. [为什么生成结果是固定长度的随机噪声雪花](#5-噪声输出根因)
6. [架构维度链完整对照表](#6-维度链对照)
7. [修复方案](#7-修复方案)

---

## 1. 问题现象

| 现象 | 描述 |
|------|------|
| 生成结果 | 固定长度视频，所有帧均为**随机噪声/雪花** |
| Connector 输出 | 有图像 vs 无图像的 L2_norm/mean/std **几乎完全一致** |
| CFG 验证 | MQ cond vs uncond 余弦相似度可能接近 1.0（CFG 无效） |

---

## 2. 训练阶段分析

### 2.1 配置文件内容 (`qwen3vl2b_sana_inst_small.yaml`)

```yaml
mllm_id: "./base_models/Qwen3-VL-2B-Instruct"
diffusion_model_id: "./base_models/Sana_1600M_512px_diffusers"   # ← Sana!
num_metaqueries: 256
modules_to_freeze:
  - "vae"                         # Sana 的 AutoencoderDC (冻结)
  - "model.mllm_backbone"         # Qwen3-VL 全部 (冻结)
modules_to_unfreeze:
  - "model.mllm_backbone.model.embed_tokens"  # 只解冻 embed_tokens
train_datasets:
  inst2m: 0.0024                  # 仅使用 ~2400 条样本 (240万的千分之一)
target_image_size: 512
connector_num_hidden_layers: 24
```

### 2.2 哪些部分得到了训练？

| 组件 | 是否训练 | 参数量(估) | 说明 |
|------|---------|-----------|------|
| **Connector** (Qwen2Encoder 24层 + Linear + GELU + Linear + RMSNorm) | **✅ 训练** | ~200M | 核心可训练部分，将 MLLM 隐藏状态映射到 Diffusion 条件空间 |
| **embed_tokens 的 MQ 部分** (<img0>~<img255> + <begin_of_img> + <end_of_img>) | **✅ 训练** | ~0.4M | 258 个新 token 的 embedding (原始 vocab 部分通过 `freeze_hook` 梯度清零) |
| **Sana 1.6B DiT** (SanaTransformer2DModel) | ❌ 冻结⭐ | ~1.6B | **Sana** 作为 diffusion backbone 全部冻结，只提供梯度信号 |
| **Qwen3-VL 2B backbone** (除 embed_tokens 外全部) | ❌ 冻结 | ~2B | 视觉编码器+LLM 全部冻结，仅提供特征 |
| **Sana VAE** (AutoencoderDC) | ❌ 冻结 | ~100M | 编解码器冻结 |

### 2.3 哪些部分完全没有训练？

| 组件 | 存在于训练中？ | 说明 |
|------|--------------|------|
| **to_wan_proj** (2240→4096 投影层) | **❌ 不存在！** | 这是 `MetaQueryEncoder` 推理时才创建的层，**训练 checkpoint 里完全没有这一层** |
| **Wan2.2 Animate 14B DiT** | **❌ 不存在！** | 训练时用的是 Sana 1.6B，完全没见过 Wan 的 DiT |
| **Wan VAE** | **❌ 不存在！** | 训练时用的是 Sana 的 AutoencoderDC |
| **T5 编码器** | **❌ 不存在！** | 原始 MetaQuery 训练不使用 T5 |
| **CLIP ViT-H/14** | **❌ 不存在！** | Animate 的参考图编码器，训练完全没涉及 |

### 2.4 训练时的信号流

```
训练时的完整路径 (qwen3vl2b_sana_inst_small):
═══════════════════════════════════════════════════════

输入: (caption + 参考图像) 
   │
   ▼
┌─────────────────────────┐
│ Qwen3-VL 2B (backbone)  │ ← 冻结 (除 MQ embed_tokens)
│   vision_encoder(图像)   │
│   + LLM(文本+图像+MQ)    │
└──────────┬──────────────┘
           │ logits at MQ positions → [B, 256, 1536]
           ▼
┌─────────────────────────────────────────────┐
│ Connector (Qwen2Encoder 24L + Linear×2)     │ ← ★可训练★
│   1536 → 2240 (Sana caption_channels)       │
└──────────┬──────────────────────────────────┘
           │ [B, 256, 2240]
           ▼
┌──────────────────────────┐
│ Sana 1.6B DiT (冻结)     │ ← Cross-Attention 消费条件
│   caption_channels=2240  │
│   2D 图像生成             │
└──────────┬───────────────┘
           │ model_pred
           ▼
     Flow Matching Loss (MSE)
           │
           ▼ 梯度回传
     更新: Connector + MQ Embeddings
```

**关键点**: Connector 被训练去输出 **`dim=2240`** 的特征，且这些特征是为 **Sana 1.6B** 的 cross-attention 优化的。

---

## 3. 关键错配：训练 vs 推理的目标 Diffusion 模型完全不同

这是**产生噪声输出的根本原因**。

### 3.1 架构错配对照表

| 维度 | 训练时 (Sana) | 推理时 (Wan Animate) | 匹配？ |
|------|-------------|---------------------|--------|
| Diffusion 模型 | Sana 1.6B (2D) | Wan Animate 14B (3D 视频) | ❌ **完全不同** |
| DiT 架构 | SanaTransformer2DModel | WanAnimateModel | ❌ |
| 条件维度 | 2240 (caption_channels) | 4096 (text_dim) | ❌ |
| 生成目标 | 2D 图像 (512×512) | 3D 视频 (77帧) | ❌ |
| VAE | AutoencoderDC (32× downsample) | CausalVAE (8× spatial) | ❌ |
| Loss 类型 | Flow Matching (2D latent) | Flow Matching (3D latent) | 部分匹配 |
| 条件拼接方式 | 直接作为 encoder_hidden_states | MQ 256 + T5 512 拼接 context | ❌ |

### 3.2 推理时的信号流 vs 训练时

```
推理时的路径 (demo_metaquery_animate.py):
═══════════════════════════════════════════════════════

输入: (caption + 参考图像)
   │
   ▼
┌─────────────────────────┐
│ Qwen3-VL 2B (backbone)  │ ← checkpoint 加载 ✅
└──────────┬──────────────┘
           │ logits → [B, 256, 1536]
           ▼
┌──────────────────────────────────────────────┐
│ Connector (checkpoint 加载)                   │ ← 训练过 ✅
│   1536 → 2240                                │
└──────────┬───────────────────────────────────┘
           │ [B, 256, 2240]
           ▼
┌──────────────────────────────────────────────┐
│ to_wan_proj (Xavier 随机初始化!!!)             │ ← ★未训练★ ❌
│   2240 → 4096                                │
└──────────┬───────────────────────────────────┘
           │ [B, 256, 4096]    (随机映射的无意义特征!)
           ▼
     torch.cat([MQ, T5], dim=0)
           │ [256+512, 4096]
           ▼
┌──────────────────────────────────────────────┐
│ Wan Animate 14B DiT                          │ ← ★没见过 MQ 特征★ ❌
│   text_embedding 期望: T5 文本特征            │
│   收到: 随机投影的 MQ + 正常 T5               │
│   cross-attention 完全被垃圾特征污染          │
└──────────┬───────────────────────────────────┘
           │ 预测的速度场完全错误
           ▼
     VAE 解码 → 噪声/雪花
```

---

## 4. Connector 输出有图/无图几乎一致的原因

### 回顾日志数据

| 指标 | 有图像 | 无图像 | 差异 |
|------|--------|--------|------|
| L2_norm | 1790.**2983** | 1790.**1710** | 0.07% |
| mean | 0.05**1712** | 0.05**1691** | 0.04% |
| std | 2.330**546** | 2.330**381** | 0.007% |

### 4.1 原因分析

差异如此微小，有以下几个叠加因素：

#### 因素 1: 训练数据量极少 (2400 条 / 240万)

```yaml
train_datasets:
  inst2m: 0.0024   # 0.24% 的训练数据
```

原始 MetaQuery-Instruct 数据集有 **~240万条** 图文对。你使用了 **千分之一 (2400条)**。

Connector 有 ~200M 参数（24层 Qwen2Encoder），而训练样本只有 2400 条。这意味着：
- 模型严重 **欠拟合**
- Connector 没有学会区分"有图 vs 无图"的语义差异
- MQ token embeddings 几乎停留在随机初始化附近

#### 因素 2: Connector 输出被 Sana 的条件空间约束

Connector 被训练去对齐 Sana 的 `caption_channels=2240`。训练 2400 步后，Connector 可能只学到了一个**近似恒等的平均映射**——无论输入什么，输出都被映射到一个狭窄的特征子空间。

#### 因素 3: MLLM 的 lm_head 被替换为 Identity

```python
self.mllm_backbone.lm_head = nn.Identity()
```

原来 `lm_head` 的输出是 vocab logits (vocab_size维)，替换为 Identity 后输出是 hidden_states (1536维)。这改变了 MLLM 输出的数值分布。但真正的 MQ 信息提取依赖 Connector 的训练质量。

#### 因素 4: freeze_hook 的梯度行为

```python
def freeze_hook(grad):
    grad[: self.num_embeddings].zero_()  # 原始 vocab 梯度清零
    return grad
```

只有 MQ 部分 (258个 token) 的 embedding 有梯度。在仅 2400 条数据上，这些 embedding 也难以充分学习"如何吸收图像信息"。

### 4.2 结论

**Connector 训练不充分** + **数据量不足** → MQ token 没有学会通过 attention 有效聚合图像信息 → 有图/无图输出几乎一样。

---

## 5. 为什么生成结果是固定长度的随机噪声雪花

### 5.1 根因链 (从最严重到最轻)

```
┌──────────────────────────────────────────────────────┐
│   ROOT CAUSE #1: to_wan_proj 完全随机 (Xavier 初始化)  │
│   ─────────────────────────────────────────────────── │
│   Connector 输出 dim=2240 的特征                      │
│   经过随机权重的 Linear(2240→4096)+GELU+Linear(4096→4096) │
│   映射为**无意义的 4096 维向量**                        │
│   这些向量与 T5 特征拼接后送入 DiT                     │
│   DiT 的 cross-attention 被完全错误的条件信号喂入       │
│                     ↓                                 │
│   去噪过程无法正确引导，输出纯噪声                      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│   ROOT CAUSE #2: DiT 从未见过 MQ 类型的条件             │
│   ─────────────────────────────────────────────────── │
│   Wan Animate 14B 训练时只见过 T5 文本特征              │
│   从未见过"前置 256 个 MQ token"的增强 context           │
│   text_len 从 512 被强制改为 768                       │
│   DiT 的 RoPE/位置编码对额外 256 个位置没有正确预期      │
│   cross-attention 完全不知道如何处理这些额外 token       │
│                     ↓                                 │
│   即使 MQ 特征是有意义的，DiT 也无法利用                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│   ROOT CAUSE #3: Connector 本身对齐的是 Sana 而非 Wan   │
│   ─────────────────────────────────────────────────── │
│   Connector 训练目标: Sana 的 caption_channels=2240    │
│   Wan 需要: text_dim=4096                            │
│   Connector 的"语义空间"与 Wan 的期望完全不在一个流形上   │
│   即使 to_wan_proj 训练好了，也需要重新对齐              │
│                     ↓                                 │
│   特征语义不匹配                                      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│   ROOT CAUSE #4: Connector 训练数据严重不足              │
│   ─────────────────────────────────────────────────── │
│   ~200M 参数 Connector 只用 2400 条数据训练              │
│   有图/无图输出几乎一致 → MQ 条件信号本身就是噪声级的      │
│                     ↓                                 │
│   即使其他一切正确，条件信号也无法有效引导生成              │
└──────────────────────────────────────────────────────┘
```

### 5.2 为什么是"固定长度"？

视频长度固定为 77 帧，这是因为生成参数中硬编码了：

```python
FRAME_NUM = 77       # 总帧数 (4n+1)
CLIP_LEN = 77        # 每 clip 帧数 (4n+1)
```

这不是 bug，而是正常的生成配置。长度为 77 帧是 Wan Animate 的标准设置。

### 5.3 雪花/噪声的具体表现

当 DiT cross-attention 收到的条件是**随机向量**时：

1. **去噪过程失败**：DiT 在每步预测的速度场 $v_\theta(x_t, t, c)$ 与真实速度场无关
2. **流式匹配退化**：$x_0 = x_1 - \int_0^1 v_\theta dt$ 的积分结果还是随机的
3. **VAE 解码随机 latent**：`vae.decode(random_latent)` → 高频随机纹理 (即雪花)
4. **每帧都是独立噪声**：因为条件完全错误，每个时间步的去噪不具备帧间一致性

---

## 6. 架构维度链完整对照表

### 6.1 MetaQuery 原始训练 (Sana 目标)

```
Qwen3-VL embedding: [vocab+258, 1536]
           │
     MLLM forward (冻结)
           │
     logits at MQ pos: [B, 256, 1536]
           │
     Connector:
       ├── Qwen2Encoder(24层, hidden=1536)  → [B, 256, 1536]
       ├── Linear(1536, 2240)               → [B, 256, 2240]  ★ Sana dim
       ├── GELU
       ├── Linear(2240, 2240)               → [B, 256, 2240]
       └── RMSNorm(2240)                    → [B, 256, 2240]
           │
     Sana 1.6B SanaTransformer2DModel:
       └── cross_attn(query=latent, kv=prompt_embeds)
           │    期望 dim=2240 ✅
           ▼
     model_pred → Loss
```

### 6.2 推理时嫁接到 Wan Animate

```
Qwen3-VL embedding: [vocab+258, 1536]
           │
     MLLM forward
           │
     logits at MQ pos: [B, 256, 1536]
           │
     Connector: → [B, 256, 2240]   (训练过, 但为 Sana 优化)
           │
     to_wan_proj (未训练!):
       ├── Linear(2240, 4096)  ← Xavier 随机
       ├── GELU
       └── Linear(4096, 4096)  ← Xavier 随机
           │
           │  [B, 256, 4096]  ← 随机映射!
           ▼
     torch.cat([MQ_4096, T5_4096], dim=0)
           │  [768, 4096]
           ▼
     Wan Animate 14B DiT:
       └── text_embedding(context)   期望: T5 语义特征
           │    收到: 随机向量 + T5
           ▼
     cross_attn → 完全错误的条件引导
           ▼
     噪声/雪花输出
```

---

## 7. 修复方案

### 方案 A: 端到端训练 Connector + to_wan_proj (推荐)

这就是 `train_connector_for_wan.py` 要做的事。但需要注意：

```
必须确保:
1. Connector + to_wan_proj 联合训练
2. 训练目标是 Wan Animate 14B DiT (冻结)
3. 使用视频数据集 (非 2D 图像)
4. 充足的训练数据量 (至少 10万+ 条)
5. 训练步数足够 (至少 5000~10000 步)
```

训练链路:
```
MLLM (冻结) → Connector (可训练) → to_wan_proj (可训练)
    → Wan Animate DiT (冻结) → Flow Matching Loss
                                   ↓
               反向传播: Connector + to_wan_proj + MQ Embeddings
```

### 方案 B: 先用更多数据重新训练 Sana 阶段

```
1. 将 inst2m: 0.0024 改为 inst2m: 1.0 (使用全部 240 万条)
2. 训练 10+ epochs，让 Connector 充分学习
3. 然后再通过 train_connector_for_wan.py 对齐到 Wan
```

### 方案 C: 跳过 MQ，直接使用 T5 + CLIP (作为 baseline)

如果只是想让 Animate 正常工作，可以暂时不拼接 MQ token：
```python
# bridge_animate.py
# 注释掉 MQ 拼接，仅使用原始 T5 context
context = t5_context  # 不拼接 MQ
```

这样至少能验证 Animate 管线本身是正常的。

### 关键修改清单

| 修改 | 优先级 | 说明 |
|------|--------|------|
| 运行 `train_connector_for_wan.py` 完成对齐训练 | P0 | 训练 to_wan_proj + 微调 Connector |
| 增加训练数据量 | P0 | inst2m: 0.0024 → 至少 0.1 (24万条) |
| 添加 T5-only baseline 验证 | P1 | 确认管线本身无 bug |
| 在 encoder.py 添加有图/无图差异验证 | P2 | 训练后确认 MQ 区分能力 |
| 在 bridge 添加 MQ 特征范数监控 | P2 | 确认 MQ 特征在 T5 特征合理范围内 |

---

## 总结

**你的生成输出是噪声的直接原因**：`to_wan_proj` 层（2240→4096 投影）是 **Xavier 随机初始化的**，**从未经过任何训练**。它将 Connector 输出映射为完全随机的 4096 维向量，污染了 Wan DiT 的 cross-attention，导致去噪完全失败。

**Connector 有图/无图输出几乎一致的原因**：训练数据量严重不足（2400 条 vs 原本 240万条），~200M 参数的 Connector 严重欠拟合，MQ token 没有学会通过 attention 聚合图像信息。

**解决路径**：必须运行 `train_connector_for_wan.py`，以 Wan Animate 14B 的 Flow Matching Loss 为目标，联合训练 Connector + to_wan_proj，且使用充足的训练数据。
