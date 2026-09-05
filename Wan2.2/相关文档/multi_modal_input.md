# MetaQuery 多模态输入机制深度分析

## 核心问题

> MetaQuery 是基于图文对训练的，那么：
> 1. 这对想要的 MetaQuery + Wan 视频生成有影响吗？
> 2. 基于视频输入时，同时给 Qwen3-VL 输入参考图片和文本，图片和文本作为不同模态是如何进入 Qwen3-VL 的？
> 3. 这和 MetaQuery 的原始设计有冲突吗？

---

## 一、项目整体架构概览

本项目由两个核心模块组成：

```
┌──────────────────────────────────────────────────────────────────────┐
│                     完整系统架构                                      │
│                                                                      │
│  ┌─────────────────────────────────────┐                             │
│  │  MetaQuery (metaquery-main/)        │                             │
│  │  ┌───────────┐  ┌────────────┐      │                             │
│  │  │ Qwen3-VL  │→│ Connector  │      │  训练阶段:                  │
│  │  │ (MLLM)    │  │ (Qwen2Enc │      │  图文对 → Sana 图像生成     │
│  │  │           │  │  + Linear) │      │                             │
│  │  └───────────┘  └────────────┘      │                             │
│  │       ↕              ↓              │                             │
│  │  [MetaQuery Tokens]  → Sana/UNet    │                             │
│  └─────────────────────────────────────┘                             │
│                    ↓ 推理阶段迁移                                     │
│  ┌─────────────────────────────────────┐                             │
│  │  Wan2.2 Bridge (wan/metaquery/)     │                             │
│  │  ┌───────────┐  ┌────────────┐      │                             │
│  │  │ Qwen3-VL  │→│ Connector  │      │  推理阶段:                  │
│  │  │ + MetaQ   │  │ + Proj     │      │  图片+文本 → Wan 视频生成   │
│  │  └───────────┘  └─────┬──────┘      │                             │
│  │                       ↓              │                             │
│  │              ┌────────────────┐      │                             │
│  │  T5 文本 ──→ │   Context      │      │                             │
│  │  MetaQ   ──→ │   Concat       │ → Wan DiT                        │
│  │              └────────────────┘      │                             │
│  └─────────────────────────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 MetaQuery 训练端 (`metaquery-main/`)

**核心文件说明：**

| 文件 | 作用 |
|------|------|
| `models/model.py` → `MLLMInContext` | 核心模型：MLLM backbone + Connector + Diffusion Transformer |
| `models/metaquery.py` → `MetaQuery` | 顶层封装：MLLMInContext + VAE + Noise Scheduler |
| `models/transformer_encoder.py` → `Qwen2Encoder` | Connector 的双向 Transformer 编码器 |
| `dataset.py` | 数据加载：支持 t2i / i2i / inst / editing 四种模式 |
| `train.py` | 训练入口 |

**支持的训练数据模式：**

| 模式 | 输入条件 | 输出目标 | 数据集 |
|------|----------|----------|--------|
| **t2i** (文本→图像) | 纯文本 caption | 目标图像 | CC12M |
| **i2i** (图像→图像) | 参考图像 (无文本) | 目标图像 | CC12M |
| **inst** (指令) | **参考图像 + 文本** | 目标图像 | MetaQuery-Instruct-2.4M |
| **editing** (编辑) | 源图像 + 编辑指令 | 编辑后图像 | OmniEdit-1.2M |

### 1.2 Wan2.2 推理端 (`wan/metaquery/`)

| 文件 | 作用 |
|------|------|
| `encoder.py` → `MetaQueryEncoder` | 加载 Qwen3-VL + Connector，投影到 Wan 维度 |
| `bridge.py` → `MetaQueryWanBridge` | T2V 桥接：MetaQuery context 拼接到 T5 context |
| `bridge_i2v.py` → `MetaQueryWanI2VBridge` | I2V 桥接：首帧 channel-concat + MetaQuery context-concat |
| `bridge_animate.py` → `MetaQueryWanAnimateBridge` | 动画桥接 |

---

## 二、MetaQuery 的核心设计：MetaQuery Token 机制

### 2.1 MetaQuery Token 是什么？

MetaQuery 的核心创新在于：**在 MLLM 的输出序列中"植入"一组特殊的可学习 token**，让这些 token 的隐藏状态作为图像生成的条件。

具体实现（见 `models/model.py` L210-230）：

```python
# 扩展词表，新增 num_metaqueries + 2 个特殊 token
self.mllm_backbone.resize_token_embeddings(
    num_embeddings + config.num_metaqueries + 2
)

# 注册特殊 token
tokenizer.add_special_tokens({
    "additional_special_tokens": 
        ["<begin_of_img>", "<end_of_img>"]
        + [f"<img{i}>" for i in range(num_metaqueries)]
})
```

在推理时，prompt 的末尾自动拼接这些 token：

```
[系统提示] [用户内容: 图片token + 文本] [助手开始]
<begin_of_img> <img0> <img1> ... <img255> <end_of_img>
```

### 2.2 MetaQuery Token 的提取

在 `encode_condition()` 方法中（`models/model.py` L454-471）：

```python
def encode_condition(self, input_ids, attention_mask, pixel_values, image_sizes):
    # Step 1: 整个序列过 Qwen3-VL，得到每个位置的 logits
    prompt_embeds = self.mllm_backbone(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_sizes,
        attention_mask=attention_mask,
    ).logits  # shape: [B, seq_len, hidden_size]

    # Step 2: 提取 <begin_of_img> 和 <end_of_img> 之间的 token 的表示
    boi_pos = torch.where(input_ids == self.boi_token_id)[1]
    eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]
    mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])
    prompt_embeds = prompt_embeds[mask].view(batch_size, -1, prompt_embeds.size(-1))
    # shape: [B, num_metaqueries, hidden_size]

    # Step 3: 通过 Connector 投影到 diffusion model 的维度
    return self.connector(prompt_embeds), attention_mask
```

**关键理解**：
- Qwen3-VL 的 `lm_head` 被替换为 `nn.Identity()`，所以 `.logits` 实际上就是最后一层的隐藏状态
- MetaQuery token 位于序列末尾，通过 Transformer 的自注意力机制，它们能 **attend 到前面所有的图片 token 和文本 token**
- 提取出来的 MetaQuery 向量包含了对输入条件（图片+文本）的"压缩理解"

---

## 三、图片和文本如何进入 Qwen3-VL（核心问题回答）

### 3.1 Qwen3-VL 的多模态输入架构

Qwen3-VL 是一个原生多模态模型，其输入处理流程如下：

```
┌──────────────────────────────────────────────────────────────────────┐
│              Qwen3-VL 多模态输入处理流程                              │
│                                                                      │
│  输入图片(PIL)            输入文本(str)                               │
│      ↓                       ↓                                       │
│  ┌──────────┐         ┌──────────────┐                               │
│  │ Vision   │         │ Tokenizer    │                               │
│  │ Encoder  │         │ (text→ids)   │                               │
│  │ (ViT)    │         └──────┬───────┘                               │
│  └────┬─────┘                │                                       │
│       ↓                      ↓                                       │
│  [视觉 token]          [文本 token ids]                              │
│  (patch embedding)     ↓                                             │
│       │           ┌──────────────┐                                   │
│       │           │ Text         │                                   │
│       │           │ Embedding    │                                   │
│       │           └──────┬───────┘                                   │
│       │                  ↓                                           │
│       │           [文本 embedding]                                   │
│       ↓                  ↓                                           │
│  ┌──────────────────────────────────┐                                │
│  │   统一的 Embedding 序列          │                                │
│  │   [...视觉tokens... ...文本tokens... ...MQ tokens...]            │
│  └──────────────┬───────────────────┘                                │
│                 ↓                                                     │
│  ┌──────────────────────────────────┐                                │
│  │   Transformer Decoder Layers     │                                │
│  │   (自注意力: 所有 token 互相看)   │                                │
│  └──────────────┬───────────────────┘                                │
│                 ↓                                                     │
│  [每个位置的隐藏状态表示]                                             │
│  → 只取 <img0>~<img255> 位置的表示 = MetaQuery 特征                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 图片的处理路径

当调用 `tokenize()` 方法时（`models/model.py` L310-420），图片输入如下处理：

```python
# 构建多模态对话格式
conversations = [
    prefix + [{
        "role": "user",
        "content": (
            [{"type": "image"} for _ in imgs]       # 图片占位符
            + [{"type": "text", "text": cap}]       # 文本
        ),
    }]
    for cap, imgs in zip(caption, image)
]

# 通过 Qwen3-VL 的 Processor 处理
# apply_chat_template: 将 <image> 占位符展开为视觉 placeholder token
# processor(..., images=images): 
#   → Vision Encoder 处理图片得到 pixel_values
#   → Tokenizer 处理文本得到 input_ids
text_inputs = tokenizer(
    text=prompts,       # 包含 <image> 占位符的文本
    images=images,      # PIL.Image 列表
    return_tensors="pt",
    padding=True,
)
```

**Qwen3-VL Processor 的具体行为**：

1. **图片处理**：
   - 将 PIL.Image 通过 **Vision Encoder (ViT)** 编码为一组 patch embedding
   - 根据 `min_pixels`/`max_pixels` 配置动态调整分辨率
   - 输出 `pixel_values`（归一化后的像素张量）和 `image_grid_thw`（时间-高度-宽度网格信息）

2. **文本处理**：
   - 将文本通过 Tokenizer 转为 token id 序列
   - 在图片占位符位置插入特殊的视觉 placeholder token id

3. **统一序列**：
   - `input_ids` 中包含 **文本 token id + 视觉 placeholder token id + MetaQuery 特殊 token id**
   - 在 Qwen3-VL 的 forward 中，视觉 placeholder 位置的 embedding 被 **替换为 Vision Encoder 输出的 patch embedding**

### 3.3 Qwen3-VL 内部的融合机制

在 `Qwen3VLForConditionalGeneration.forward()` 中（Transformers 库实现）：

```python
# 简化的内部流程：
def forward(self, input_ids, pixel_values, image_grid_thw, attention_mask):
    # 1. 文本 embedding
    text_embeds = self.embed_tokens(input_ids)  # [B, seq_len, hidden_dim]
    
    # 2. 视觉 embedding (如果有图片)
    if pixel_values is not None:
        vision_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # vision_embeds: [num_patches, hidden_dim]
        
        # 3. ★关键步骤★ 将视觉 embedding 填充到 input_ids 中视觉占位符的位置
        #    placeholder 位置的文本 embedding → 被替换为 → vision embedding
        text_embeds[vision_placeholder_positions] = vision_embeds
    
    # 4. 统一的 embedding 序列通过 Transformer decoder layers
    #    自注意力让所有 token (视觉 + 文本 + MetaQuery) 互相注意
    hidden_states = self.transformer_layers(text_embeds, attention_mask)
    
    return hidden_states
```

**关键点**：
- 图片和文本在 **embedding 层面被融合为统一序列**
- 所有 Transformer layer 都使用 **全序列自注意力 (full-context attention)**
- 代码中特意关闭了滑动窗口注意力：
  ```python
  self.mllm_backbone.model.config.use_sliding_window = False
  self.mllm_backbone.model.config.sliding_window = None
  ```
  目的就是让 MetaQuery token（位于序列最右端）能 attend 到序列最左端的视觉 token

### 3.4 完整的 Token 序列结构

当同时输入图片和文本时，完整的 `input_ids` 序列为：

```
[系统提示 tokens]
[<|im_start|>user\n]
[<|vision_start|> <视觉placeholder×N> <|vision_end|>]  ← 图片位置
[caption 文本 tokens]                                    ← 文本位置
[<|im_end|>]
[<|im_start|>assistant\n]
[<begin_of_img>]                                         ← MetaQuery 区域开始
[<img0> <img1> ... <img255>]                             ← 256 个 MetaQuery token
[<end_of_img>]                                           ← MetaQuery 区域结束
[<|im_end|>]
```

其中视觉 placeholder 的数量 N 取决于图片分辨率和 Qwen3-VL 的 ViT patch 大小。

---

## 四、MetaQuery 的训练数据分析

### 4.1 当前配置使用的训练数据

根据 `configs/qwen3vl4b_sana_inst_small.yaml`：

```yaml
train_datasets:
  inst2m: 0.024      # MetaQuery-Instruct-2.4M 的 1/100
```

**inst2m 数据集的格式** (来自 `xcpan/MetaQuery_Instruct_2.4M_512res`)：

| 字段 | 内容 |
|------|------|
| `source_images` | 输入参考图像列表 (可以有多张) |
| `caption` / `prompt` | 文本指令 |
| `target_image` | 目标生成图像 (512×512) |

**关键：inst2m 是图文对训练的，但不是"只有文本"的训练**：
- 它是 **参考图像 + 文本指令 → 目标图像** 的训练模式
- 输入 Qwen3-VL 的就是 **图片 + 文本的混合输入**
- `_inst_process_fn` 中有 10%-30% 的 null augmentation（随机丢弃图片或文本）

### 4.2 训练中的条件 Drop 策略

```python
# dataset.py L107-128
def _inst_process_fn(batch, target_transform):
    rand_probs = torch.rand((len(batch["target_image"]), 1))
    null_caption_mask = rand_probs < 0.2       # 20% 概率丢弃文本
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)  # 20% 概率丢弃图片
```

分布：
- **10% 同时丢弃文本和图片** → 学习无条件生成（CFG 的 null condition）
- **10% 仅丢弃图片** → 学习纯文本条件
- **10% 仅丢弃文本** → 学习纯图片条件
- **70% 图片+文本都保留** → 学习多模态联合条件

**这意味着 MetaQuery 在训练时就已经见过了"图片+文本同时输入"的场景。**

### 4.3 其他训练模式

| 模式 | 条件 Drop | 输入给 Qwen3-VL 的内容 |
|------|-----------|------------------------|
| **t2i** | 10% 丢弃文本 | 纯文本 (无 pixel_values) |
| **i2i** | 10% 丢弃图片 | 纯图片 (caption="" 空字符串) |
| **inst** | 20% 丢弃文本/图片 | **图片 + 文本** |
| **editing** | 10% 丢弃图片 | **源图片 + 编辑指令** |

---

## 五、对 MetaQuery + Wan 视频生成的影响分析

### 5.1 训练-推理一致性对比

```
┌──────────────────────────────────────────────────────────────────┐
│                    训练阶段 vs 推理阶段 对比                       │
├──────────────┬──────────────────────┬───────────────────────────┤
│              │ 训练 (inst2m)         │ 推理 (Wan I2V)            │
├──────────────┼──────────────────────┼───────────────────────────┤
│ MLLM         │ Qwen3-VL-4B          │ Qwen3-VL-2B/4B           │
│ 输入图片      │ source_images (参考图)│ 首帧/参考图               │
│ 输入文本      │ caption (文本指令)    │ prompt (视频描述)         │
│ 输出         │ MetaQuery → Sana → 图│ MetaQuery → Wan → 视频   │
│ 生成目标      │ 512×512 图像          │ 视频帧序列                │
│ Diffusion    │ Sana 1.6B            │ Wan 14B                  │
│ Connector维度 │ → 2240 (Sana)        │ → 2240 → 4096 (投影)     │
├──────────────┼──────────────────────┼───────────────────────────┤
│ 相同点        │ 都用 Qwen3-VL 处理图片+文本                      │
│              │ 都提取 <img0>~<img255> 的隐藏表示                 │
│              │ 都经过同一个 Connector                             │
├──────────────┼──────────────────────┼───────────────────────────┤
│ 不同点        │ 图→图                │ 图→视频 (多帧)            │
│              │ Sana 做 diffusion     │ Wan DiT 做 diffusion     │
│              │ 无额外投影层           │ 多一个 to_wan_proj 投影   │
│              │ 512px 目标            │ 720p+ 视频目标            │
└──────────────┴──────────────────────┴───────────────────────────┘
```

### 5.2 有影响吗？影响多大？

**结论：有影响，但影响可控，并且项目已经做了合理的设计来缓解。**

#### 影响分析 1: MLLM 特征提取层 — **无冲突**

MetaQuery 的 Qwen3-VL backbone 在训练时是 **冻结的**（见配置）：

```yaml
modules_to_freeze:
  - "vae"
  - "model.mllm_backbone"      # ★ MLLM 冻结！
modules_to_unfreeze:
  - "model.mllm_backbone.model.embed_tokens"  # 仅 MetaQuery 新增的 embedding 可训练
```

这意味着：
- Qwen3-VL 的 Vision Encoder、文本 Embedding、所有 Transformer layer 权重 **完全保持预训练状态**
- 只有新增的 `<begin_of_img>`, `<end_of_img>`, `<img0>`~`<img255>` 的 embedding 是可学习的
- 原始的 freeze_hook 确保旧 token 的 embedding 梯度为零

**所以 "图片如何进入 Qwen3-VL" 这个问题，答案是：和原版 Qwen3-VL 完全一致。** MetaQuery 训练没有改变 Qwen3-VL 处理多模态输入的方式。

#### 影响分析 2: Connector 层 — **存在域偏差**

Connector 是训练中唯一完全可学习的组件：

```python
self.connector = nn.Sequential(
    Qwen2Encoder(24层双向Transformer),   # 从 hidden_size=2560 (4B) 投影
    nn.Linear(2560, 2240),               # → 2240 (Sana 的 caption_channels)
    nn.GELU(),
    nn.Linear(2240, 2240),
    RMSNorm(2240),
)
```

- 训练时，Connector 学习将 MetaQuery 特征映射到 **Sana 的条件空间 (2240 维)**
- 推理时，Connector 输出再经过一个 **额外的投影层映射到 Wan 的空间 (4096 维)**：

```python
# encoder.py L235-244
self.to_wan_proj = nn.Sequential(
    nn.Linear(connector_out_dim, wan_text_dim),   # 2240 → 4096
    nn.GELU(),
    nn.Linear(wan_text_dim, wan_text_dim),         # 4096 → 4096
)
```

**潜在问题**：Connector 在训练时针对 Sana 优化了，其输出分布可能不完全适配 Wan 的 cross-attention。但 `to_wan_proj` 投影层可以学习适配这个差异（如果做微调的话）。

#### 影响分析 3: 图像→视频的语义迁移 — **可能是最大的影响**

| 方面 | 分析 |
|------|------|
| 静态语义 | MetaQuery 从图片中提取的风格、内容、构图等高层语义，在视频生成中依然有效 |
| 动态语义 | MetaQuery 训练时完全没见过"运动/时间连续性"概念，**无法提供运动引导** |
| 分辨率适配 | 训练用 384px 输入图片，推理可能用更高分辨率参考图，但 Qwen3-VL 的 ViT 支持动态分辨率 |
| 帧数概念 | MetaQuery 没有"帧"的概念，只提供单帧级别的语义条件 |

### 5.3 项目是如何解决这些影响的？

**双重条件架构 (I2V Bridge)**：

```
参考图 ──→ [Qwen3-VL + MetaQuery] ──→ Context Concat → 语义引导（风格/内容）
首帧图 ──→ [VAE Encode + Mask]     ──→ Channel Concat → 结构引导（像素级）
```

- **Channel Concat (首帧条件)**：从首帧 VAE latent 得到 20ch tensor，与噪声 latent (16ch) concat 为 36ch 输入。这提供 **像素级的帧间连续性**
- **Context Concat (MetaQuery 条件)**：MetaQuery 256 个 token 前置拼接到 T5 context，参与所有层的 cross-attention。这提供 **语义级的风格/内容引导**

两者互补：首帧条件管"长什么样"，MetaQuery 管"整体感觉是什么"。

---

## 六、图片+文本进入 Qwen3-VL 的详细流程

### 6.1 数据流追踪

以 I2V 推理为例，追踪完整的数据流：

```
用户输入:
  prompt = "Summer beach, a white cat on a surfboard..."
  first_frame = PIL.Image (720x1280)
  mq_reference = [PIL.Image (style_ref)]
      ↓
[Step 1] bridge_i2v.py generate()
      ↓
[Step 2] mq_encoder.encode(["prompt"], [[style_ref]])
      ↓
[Step 3] MLLMInContext.tokenize(tokenizer, ["prompt"], [[style_ref]])
      │
      ├─→ 构建对话:
      │   conversations = [{
      │       "role": "system",
      │       "content": [{"type": "text", "text": "You will be given an image..."}]
      │   }, {
      │       "role": "user",
      │       "content": [
      │           {"type": "image"},                      ← 图片占位
      │           {"type": "text", "text": "Summer..."}   ← 文本
      │       ]
      │   }]
      │
      ├─→ apply_chat_template → 展开为文本模板
      │   "<|im_start|>system\n..."
      │   "<|im_start|>user\n<|vision_start|><|image_pad|>×N<|vision_end|>"
      │   "Summer beach...<|im_end|>"
      │   "<|im_start|>assistant\n"
      │
      ├─→ 拼接 MetaQuery suffix:
      │   "...<begin_of_img><img0><img1>...<img255><end_of_img><|im_end|>"
      │
      └─→ tokenizer(text=prompts, images=[style_ref])
          │
          ├─ pixel_values: ViT 处理后的图像张量
          │  shape: [num_patches, 3, patch_h, patch_w] → Vision Encoder 输入
          │
          ├─ image_grid_thw: 图像网格尺寸 (temporal=1, height, width)
          │  用于 Vision Encoder 内部的位置编码
          │
          └─ input_ids: token id 序列
             [sys_tokens | user_start | vision_placeholders(N个) | text_tokens | MQ_tokens(258个)]
      ↓
[Step 4] encode_condition(input_ids, attention_mask, pixel_values, image_grid_thw)
      │
      │  Qwen3-VL 内部:
      │  ┌──────────────────────────────────────────────────────────┐
      │  │ 1. text_embeds = embed_tokens(input_ids)                 │
      │  │    → shape: [1, total_seq_len, 2560]                    │
      │  │                                                          │
      │  │ 2. vision_embeds = visual_encoder(pixel_values, grid_thw)│
      │  │    → shape: [num_vision_tokens, 2560]                   │
      │  │    (ViT 编码 → merge层 → 与文本同维度)                   │
      │  │                                                          │
      │  │ 3. text_embeds[vision_positions] = vision_embeds          │
      │  │    → 视觉 token 替换进统一序列                            │
      │  │                                                          │
      │  │ 4. 28层 Transformer (全序列自注意力):                     │
      │  │    所有位置互相 attend:                                   │
      │  │    [系统 | 图片patch | 文本 | MetaQuery] 全连接注意力     │
      │  │                                                          │
      │  │ 5. hidden_states → Identity (原 lm_head) → logits        │
      │  └──────────────────────────────────────────────────────────┘
      │
      └─→ 提取 logits[boi_pos+1 : eoi_pos] → MetaQuery 特征
          shape: [1, 256, 2560]
      ↓
[Step 5] Connector(MetaQuery特征) → [1, 256, 2240]
      ↓
[Step 6] to_wan_proj(connector_out) → [1, 256, 4096]
      ↓
[Step 7] 与 T5 context [1, L_t5, 4096] 拼接
      → augmented context [1, 256+L_t5, 4096]
      ↓
[Step 8] Wan DiT cross-attention 使用增强 context
```

### 6.2 视觉和文本 Token 在注意力中的交互

在 Qwen3-VL 的 Transformer decoder 中，每一层的自注意力 **不区分模态**：

```
注意力矩阵 (简化示意, 假设序列长 = 5+N+10+258):

            系统  图片patches  文本tokens  MetaQuery tokens
系统         ✓       ✓           ✓            ✓
图片patches  ✓       ✓           ✓            ✓
文本tokens   ✓       ✓           ✓            ✓
MetaQuery    ✓       ✓           ✓            ✓
     ↑
  因为关闭了 sliding_window，
  所以每个 MetaQuery token 都能直接看到所有图片 patch
```

**MetaQuery token 的信息来源**：
- MetaQuery token 自身的 embedding 是通过训练学到的"查询向量"
- 通过自注意力，它们从 **图片 patch token** 获取视觉特征
- 通过自注意力，它们从 **文本 token** 获取语义特征
- 两种信息在多层 Transformer 中 **逐层融合和提炼**
- 最终 MetaQuery token 的隐藏状态 = 对输入多模态信息的高层次"摘要"

---

## 七、与 MetaQuery 原始设计的冲突分析

### 7.1 MetaQuery 论文的原始设计意图

MetaQuery 的原始设计是：
- 用 **MLLM (Large Language Model + Vision)** 理解输入内容
- 将理解后的表示通过 MetaQuery token 传递给 diffusion model
- MLLM 冻结（保留预训练能力），只训练 Connector

### 7.2 冲突分析

| 方面 | 原始设计 | 当前用法 | 是否冲突 |
|------|----------|----------|----------|
| **MLLM 输入** | 图片或文本 (或两者) | 图片 + 文本 | **不冲突** ✅ Qwen3-VL 原生支持多模态 |
| **输入模态融合** | 模态在 MLLM 内部自然融合 | 同上 | **不冲突** ✅ |
| **输出维度** | → Sana (2240) | → Wan (4096, 需投影) | **轻微偏差** ⚠️ 额外投影层可适配 |
| **生成目标** | 静态图像 | 视频序列 | **有gap** ⚠️ 但 MetaQuery 只提供语义条件 |
| **Diffusion model** | Sana 1.6B | Wan 14B DiT | **需适配** ⚠️ cross-attention 维度不同 |
| **训练数据** | 图文对 + 指令数据 | 视频描述 + 参考图 | **有域偏差** ⚠️ 但语义层面相通 |

### 7.3 不冲突的核心原因

**图片+文本同时输入 Qwen3-VL 完全不与 MetaQuery 原始设计冲突，原因如下：**

1. **Qwen3-VL 本身就是多模态模型**：MetaQuery 选择 MLLM 而非纯 CLIP 作为条件编码器，就是为了利用其 **多模态理解能力**。输入图片+文本是 Qwen3-VL 最自然的用法。

2. **训练数据已覆盖此场景**：inst2m 和 editing 数据集的训练样本就是「参考图片 + 文本描述 → 目标图像」，和你在 I2V 推理中的「参考图片 + 文本 prompt → 视频」在输入格式上完全一致。

3. **MLLM 冻结保证泛化**：Qwen3-VL backbone 冻结意味着它的多模态理解能力完全保留——无论输入风景照、人像、插画，还是任何图片+文本组合，Qwen3-VL 都能生成有意义的隐藏表示。

4. **MetaQuery token 设计的灵活性**：MetaQuery token 通过自注意力机制 **自适应地** 从所有输入 token 中提取信息。它不关心前面有几张图、有多长的文本，只要这些内容在 Qwen3-VL 的上下文窗口内。

### 7.4 真正需要关注的潜在问题

| 问题 | 严重度 | 说明 | 缓解方案 |
|------|--------|------|----------|
| **Connector → Wan 的域迁移** | 中 | Connector 针对 Sana 优化，直接迁移到 Wan 可能不是最优 | `to_wan_proj` 投影层可微调适配 |
| **静态→动态的语义gap** | 中 | MetaQuery 没有运动概念 | I2V 首帧条件提供动态信息；MetaQuery 只负责语义 |
| **无 checkpoint 时的效果** | 高 | Connector 随机初始化，MetaQuery 输出基本是噪声 | 必须先训练 MetaQuery，或只用条件较弱的模式 |
| **分辨率/质量差异** | 低 | 训练用 512px，推理可能用更高分辨率 | Qwen3-VL ViT 支持动态分辨率，影响不大 |

---

## 八、总结

### 核心回答

**Q1: MetaQuery 基于图文对训练，对 Wan 视频生成有影响吗？**

有影响但影响可控。主要影响在于：
- Connector 针对 Sana 优化，迁移到 Wan 需要额外投影层（已实现）
- MetaQuery 没有"运动"概念，但视频的动态性由 Wan 的首帧条件和文本描述负责
- 整体架构设计（双重条件：首帧结构 + MetaQuery 语义）有效缓解了这一问题

**Q2: 图片和文本如何进入 Qwen3-VL？**

1. 图片通过 **Vision Encoder (ViT)** 编码为 patch embedding
2. 文本通过 **Tokenizer + Text Embedding** 编码为文本 embedding
3. 两种 embedding 在 **input embedding 层面合并为统一序列**
4. 统一序列通过 Qwen3-VL 的 **Transformer decoder** 做全序列自注意力
5. MetaQuery token 位于序列末尾，通过自注意力 **同时 attend 到图片和文本的所有 token**
6. 提取 MetaQuery token 的隐藏状态作为条件特征

**Q3: 这和 MetaQuery 原始设计有冲突吗？**

**没有冲突。** MetaQuery 的设计就是利用 MLLM 的多模态理解能力。图片+文本同时输入是 Qwen3-VL 最自然的使用方式，训练数据中也已经包含了这种「图片+文本」混合输入的样本（inst2m 数据集）。MetaQuery token 通过自注意力机制自适应地融合来自不同模态的信息，不依赖于输入必须是单一模态。

### 架构全景图

```
    ┌─────────────── 推理时完整数据流 ──────────────────┐
    │                                                    │
    │  参考图 + prompt                                    │
    │      │                                              │
    │      ↓                                              │
    │  ┌──────────────────────────────────────────┐       │
    │  │  Qwen3-VL (冻结)                          │       │
    │  │  ┌─────────┐  ┌─────────┐  ┌───────────┐ │       │
    │  │  │ ViT     │  │ TextEmb │  │ MQ Embed  │ │       │
    │  │  │(图片→   │  │(文本→   │  │(query→    │ │       │
    │  │  │ patch)  │  │ embed)  │  │ embed)    │ │       │
    │  │  └────┬────┘  └────┬────┘  └─────┬─────┘ │       │
    │  │       └──────┬─────┘             │       │       │
    │  │              ↓                   │       │       │
    │  │     [统一embedding序列]          │       │       │
    │  │     [图片patch + 文本 + MQ]──────┘       │       │
    │  │              ↓                           │       │
    │  │     28层 Full Attention Transformer       │       │
    │  │              ↓                           │       │
    │  │     提取 MQ token 隐藏状态                │       │
    │  └──────────────┬───────────────────────────┘       │
    │                 ↓                                    │
    │  ┌──────────────────────────────┐                    │
    │  │  Connector (可训练)           │                    │
    │  │  Qwen2Encoder(24L) + Linear  │                    │
    │  │  [B, 256, 2560] → [B, 256, 2240]                 │
    │  └──────────────┬───────────────┘                    │
    │                 ↓                                    │
    │  ┌──────────────────────────────┐                    │
    │  │  to_wan_proj (推理时额外投影)  │                    │
    │  │  2240 → 4096                  │                    │
    │  └──────────────┬───────────────┘                    │
    │                 ↓                                    │
    │  [MetaQuery context: 256 × 4096]                     │
    │           +                                          │
    │  [T5 text context: L × 4096]                         │
    │           ↓                                          │
    │  [增强 context: (256+L) × 4096] ──→ Wan DiT         │
    │                                    cross-attention   │
    │                                         ↕            │
    │  首帧图 → VAE → 20ch y tensor ──→ Wan DiT           │
    │                                    channel-concat    │
    │                                         ↓            │
    │                                    视频帧序列         │
    └────────────────────────────────────────────────────┘
```
