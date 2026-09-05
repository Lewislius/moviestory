# MetaQuery Animate 生成全流程深度追踪

> 日期: 2026-03-03
> 针对 `demo_metaquery_animate.py` 的完整输入→输出调用链分析

---

## 0. 关键结论：初始化 vs 推理编码不冲突

**你提到的两段代码不冲突，它们分属两个不同阶段：**

| 代码段 | 所处阶段 | 作用 | 执行时机 |
|--------|---------|------|---------|
| `MetaQueryWanAnimateBridge(**bridge_kwargs)` | **初始化阶段** | 加载模型权重到 GPU、构建网络结构 | 程序启动时执行 **1 次** |
| `self.mllm_model.encode_condition(input_ids, pixel_values, ...)` | **推理编码阶段** | 将具体的图片+文本输入送入 Qwen3-VL 做前向传播 | 每次生成时执行（至少 **2 次**：cond + uncond） |

**类比：**
- 初始化 = 买了一台洗衣机、装好并通电
- encode_condition = 每次洗衣服时，放入脏衣服和洗衣液，按下启动

两者是**串行的前后关系**，不存在冲突。初始化完成后，Qwen3-VL 的权重已加载到 `self.mq_encoder.mllm_model` 中，之后每次 `bridge.generate()` 调用时，用户的图片和文本才会真正通过 Qwen3-VL 做前向传播。

---

## 1. 整体流程概览

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                        程序生命周期                              │
                    │                                                                 │
                    │  ┌─ 阶段 A: 初始化 (执行 1 次) ──────────────────────────────┐  │
                    │  │  1. 加载 WanAnimate (T5, VAE, CLIP, DiT, FaceAdapter)    │  │
                    │  │  2. 加载 MetaQueryEncoder:                                │  │
                    │  │     - 从 checkpoint/预训练加载 Qwen3-VL + Connector       │  │
                    │  │     - 创建 to_wan_proj 投影层                             │  │
                    │  │  ★ 此时 Qwen3-VL 已加载到 GPU，但尚未处理任何图片/文本    │  │
                    │  └───────────────────────────────────────────────────────────┘  │
                    │                              │                                  │
                    │                              ▼                                  │
                    │  ┌─ 阶段 B: 推理生成 (每次生成执行) ─────────────────────────┐  │
                    │  │  3. T5 编码文本                                           │  │
                    │  │  4. ★ 图片+文本送入 Qwen3-VL 前向传播 → MetaQuery 特征   │  │
                    │  │  5. MQ + T5 context 拼接                                  │  │
                    │  │  6. CLIP 编码参考图                                        │  │
                    │  │  7. 逐 clip 去噪循环 (所有条件注入 Wan DiT)                │  │
                    │  │  8. VAE 解码 → 视频                                       │  │
                    │  └───────────────────────────────────────────────────────────┘  │
                    └─────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段 A：初始化（模型加载 — 不处理数据）

### 2.1 入口：`demo_metaquery_animate.py` main()

#### Step 1: 加载 WanAnimate 管线

```
文件: demo_metaquery_animate.py 第 120-131 行
```

```python
config = WAN_CONFIGS["animate-14B"]
wan_animate = WanAnimate(
    config=config,
    checkpoint_dir=WAN_CHECKPOINT_DIR,   # "/home/.../Wan2.2-Animate-14B"
    device_id=0,
    rank=0,
    t5_cpu=False,
    init_on_cpu=True,
)
```

**加载的组件（仅加载权重，不处理数据）：**
- `wan_animate.text_encoder` — T5-XXL 文本编码器
- `wan_animate.vae` — Wan2.2 视频 VAE
- `wan_animate.clip` — CLIP ViT-H/14 图像编码器
- `wan_animate.noise_model` — WanAnimateModel (DiT, 含 face_adapter, motion_encoder 等)

#### Step 2: 构建 MetaQueryWanAnimateBridge

```
文件: demo_metaquery_animate.py 第 136-153 行
```

```python
bridge_kwargs = dict(
    wan_animate_pipeline=wan_animate,
    metaquery_checkpoint=METAQUERY_CHECKPOINT,  # inst_small 路径
    num_metaqueries=NUM_METAQUERIES,            # 256
    mq_guidance_scale=MQ_GUIDANCE_SCALE,        # 1.0
    dtype=torch.bfloat16,
)
# 因为 METAQUERY_CHECKPOINT 不为空，所以 mllm_id 不传入

bridge = MetaQueryWanAnimateBridge(**bridge_kwargs)
```

**这一步的完整调用链：**

```
demo_metaquery_animate.py:153
  └── MetaQueryWanAnimateBridge.__init__()
        文件: wan/metaquery/bridge_animate.py 第 82-189 行
        │
        ├── self.wan = wan_animate                          # 保存 Wan 管线引用
        │
        ├── MetaQueryEncoder.__init__()                     # ★ 核心：加载 Qwen3-VL
        │     文件: wan/metaquery/encoder.py 第 78-130 行
        │     │
        │     ├── use_checkpoint = True                     # METAQUERY_CHECKPOINT 非空
        │     │
        │     ├── _init_from_checkpoint()                   # ← 走 checkpoint 加载路径
        │     │     文件: wan/metaquery/encoder.py 第 136-176 行
        │     │     │
        │     │     ├── find_newest_checkpoint(checkpoint_path)
        │     │     │     文件: metaquery-main/trainer_utils.py 第 59-68 行
        │     │     │     → 在 .../qwen3vl2b_inst_small/ 下找最新的 checkpoint-XXXX/
        │     │     │
        │     │     ├── MetaQuery.from_pretrained(ckpt, ...)
        │     │     │     文件: metaquery-main/models/metaquery.py
        │     │     │     │
        │     │     │     ├── MetaQuery.__init__(config)
        │     │     │     │     └── self.model = MLLMInContext(config)
        │     │     │     │           文件: metaquery-main/models/model.py 第 62-291 行
        │     │     │     │           │
        │     │     │     │           ├── 加载 Qwen3VLForConditionalGeneration  ← Qwen3-VL 权重
        │     │     │     │           ├── resize_token_embeddings(+258)         ← 扩展词表
        │     │     │     │           ├── lm_head = nn.Identity()               ← 替换输出头
        │     │     │     │           ├── 加载 SanaTransformer2DModel           ← 仅读 connector_out_dim
        │     │     │     │           ├── 构建 Connector (Qwen2Encoder+Linear)  ← 对齐模块
        │     │     │     │           └── 注册 BOI/EOI/MQ 特殊 token
        │     │     │     │
        │     │     │     └── from_pretrained() 用 checkpoint 覆盖所有权重
        │     │     │           → Qwen3-VL backbone: 微调后的权重 ✅
        │     │     │           → 258 个新 embedding: 训练过的 ✅
        │     │     │           → Connector: 训练过的对齐权重 ✅
        │     │     │
        │     │     ├── self.mllm_model = mq_model.model     # 保存 MLLMInContext
        │     │     ├── self.tokenizer = ...                   # 保存 tokenizer
        │     │     ├── self.tokenize = ...                    # 保存 tokenize 函数
        │     │     └── del mq_model (删除外壳，释放 VAE/Sana)
        │     │
        │     └── _init_projection_and_validate()
        │           文件: wan/metaquery/encoder.py 第 241-298 行
        │           │
        │           ├── 创建 to_wan_proj:
        │           │     nn.Linear(2240 → 4096) + GELU + nn.Linear(4096 → 4096)
        │           │     → Xavier 初始化
        │           │
        │           ├── 删除 MLLMInContext.transformer (Sana, 释放显存)
        │           │
        │           └── 完整性验证:
        │                 - mllm_backbone 类型 = Qwen3VL ✅
        │                 - BOI/EOI token 已注册 ✅
        │                 - Connector 参数量 > 0 ✅
        │                 - to_wan_proj 权重非零 ✅
        │                 - 维度链: 1536 → 2240 → 4096 ✅
        │
        ├── self._orig_text_len = wan.noise_model.text_len   # 记录原始值
        ├── self._aug_text_len = orig + 256                   # 计算扩展后值
        │
        └── 初始化验证:
              - text_embedding.in_features = 4096 ✅
              - in_dim = 36 (16+20) ✅
              - cross_attn 存在 ✅
              - face_adapter 存在 ✅
              - MetaQuery encoder 就绪 ✅
```

**初始化结束后的状态：**
```
bridge
├── .wan                     → WanAnimate 管线 (T5, VAE, CLIP, DiT)
├── .mq_encoder              → MetaQueryEncoder
│   ├── .mllm_model          → MLLMInContext (Qwen3-VL + Connector)
│   │   ├── .mllm_backbone   → Qwen3VLForConditionalGeneration [已加载权重, eval模式]
│   │   ├── .connector       → Qwen2Encoder + Linear 投影 [已加载训练权重]
│   │   ├── .boi_token_id    → <begin_of_img> 的 ID
│   │   └── .eoi_token_id    → <end_of_img> 的 ID
│   ├── .to_wan_proj         → Linear(2240→4096) + GELU + Linear(4096→4096) [Xavier 初始化]
│   └── .tokenizer           → AutoProcessor (含 tokenizer + image_processor)
├── .num_metaqueries = 256
├── ._orig_text_len          → DiT 原始 text_len
└── ._aug_text_len           → 原始 + 256
```

**★ 重点：此时所有模型权重已加载到 GPU，但没有任何数据（图片/文本）被处理过。**

---

## 3. 阶段 B：推理生成（数据流经全部模型）

### 3.0 入口

```
文件: demo_metaquery_animate.py 第 188-204 行
```

```python
video = bridge.generate(
    input_prompt=prompt,                # "Snoopy hit the tennis ball ..."
    ref_image=ref_image,                # PIL.Image: snoopy1.jpg
    face_source=face_source,            # None
    mq_reference_images=mq_ref_images,  # [snoopy1.jpg]
    frame_num=77, clip_len=77, refert_num=1,
    sampling_steps=20, guide_scale=1.0, seed=1234,
    ...
)
```

→ 调用 `MetaQueryWanAnimateBridge.generate()`

```
文件: wan/metaquery/bridge_animate.py 第 340-814 行
```

### 3.1 Step 1/7: 准备参考人物图

```
文件: bridge_animate.py 第 432-443 行
```

```python
ref_np = np.array(ref_image)                              # PIL → numpy
height = (ref_np.shape[0] // 8) * 8                       # 对齐到 8 的倍数
width = (ref_np.shape[1] // 8) * 8
ref_np = self._padding_resize(ref_np, height, width)      # 等比缩放 + 居中填充
```

```
输入: ref_image (PIL.Image, snoopy1.jpg)
输出: ref_np (numpy.ndarray, H×W×3, uint8)
```

### 3.2 Step 2/7: 准备面部视频帧

```
文件: bridge_animate.py 第 446-456 行
```

```python
# FACE_VIDEO_PATH = None 的情况:
face_frames = [np.zeros((512, 512, 3), dtype=np.uint8)] * frame_num
# 77 帧全零图像，归一化后为 -1.0，使 face_adapter 不起作用
```

```
输入: face_source = None
输出: face_frames = list[np.ndarray] (77 帧 512×512×3 全零)
```

### 3.3 Step 3/7: T5 文本编码

```
文件: bridge_animate.py 第 459-475 行
```

```python
context = wan.text_encoder([input_prompt], device)         # T5 编码正面 prompt
context_null = wan.text_encoder([n_prompt], device)        # T5 编码负面 prompt
```

```
输入: "Snoopy hit the tennis ball and danced with joy."
输出:
  context      = [Tensor shape [L_t5, 4096]]    # 正面文本编码
  context_null = [Tensor shape [L_t5, 4096]]    # 负面文本编码 (空字符串)
```

**注意：T5 编码只处理纯文本，不处理图像。**

---

### 3.4 Step 4/7: ★ MetaQuery (Qwen3-VL) 编码 — 图片+文本经过 Qwen3-VL

**这是你关心的核心步骤：图片和文本确实在此处经过了 Qwen3-VL 的完整前向传播。**

```
文件: bridge_animate.py 第 478-499 行
```

```python
# 确定 MetaQuery 输入图像
if not mq_reference_images:
    mq_images_for_encode = [ref_image]       # 默认用参考人物图
else:
    mq_images_for_encode = mq_reference_images  # 用户指定的 [snoopy1.jpg]

# ★ 有条件编码: 图片 + 正面文本 → Qwen3-VL
mq_context = self.mq_encoder.encode(
    [input_prompt],              # ["Snoopy hit the tennis ball ..."]
    [mq_images_for_encode]       # [[snoopy1.jpg]]
)

# ★ 无条件编码: 仅负面文本，无图片 → Qwen3-VL
mq_context_null = self.mq_encoder.encode(
    [n_prompt],                  # [""]
    None                         # 无图像
)
```

#### 3.4.1 有条件编码的完整调用链

```
bridge_animate.py:489  mq_context = self.mq_encoder.encode([prompt], [[snoopy1.jpg]])
  │
  └── MetaQueryEncoder.encode()
        文件: wan/metaquery/encoder.py 第 322-480 行
        │
        │  ┌─────────────────────────────────────────────────────────────────┐
        │  │  参数:                                                          │
        │  │    captions = ["Snoopy hit the tennis ball and danced with joy."]│
        │  │    input_images = [[PIL.Image(snoopy1.jpg)]]                    │
        │  └─────────────────────────────────────────────────────────────────┘
        │
        ├── [A] 分词 + 图像预处理 (tokenize)
        │     文件: encoder.py 第 340-345 行 → 调用 model.py 第 305-417 行
        │     │
        │     │  input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
        │     │      self.tokenizer, captions, input_images
        │     │  )
        │     │
        │     │  tokenize() 内部流程 (文件: metaquery-main/models/model.py 第 305-417 行):
        │     │  │
        │     │  ├── 1. 构建 conversation 对话格式:
        │     │  │     prefix = [{"role": "system", "content": system_prompt}]  (如果有)
        │     │  │     content = [
        │     │  │         {"type": "image"},                           ← snoopy1.jpg 的占位符
        │     │  │         {"type": "text", "text": "Snoopy hit ..."}  ← 文本 prompt
        │     │  │     ]
        │     │  │
        │     │  ├── 2. apply_chat_template → 生成 prompt 字符串:
        │     │  │     "<|im_start|>system\n...<|im_end|>\n"
        │     │  │     "<|im_start|>user\n<|vision_start|><|image_pad|>...<|vision_end|>"
        │     │  │     "Snoopy hit the tennis ball and danced with joy.<|im_end|>\n"
        │     │  │     "<|im_start|>assistant\n"
        │     │  │
        │     │  ├── 3. 追加 MetaQuery suffix:
        │     │  │     suffix = "\n<begin_of_img>"
        │     │  │             + "<img0><img1><img2>...<img255>"    ← 256 个 MQ token
        │     │  │             + "<end_of_img><|im_end|>"
        │     │  │     prompt = prompt + suffix
        │     │  │
        │     │  ├── 4. tokenizer 处理 (AutoProcessor.__call__):
        │     │  │     - 文本部分 → token IDs
        │     │  │     - 图像部分 → Qwen3-VL 内置 vision processor:
        │     │  │         snoopy1.jpg → resize → normalize → pixel_values tensor
        │     │  │         同时计算 image_grid_thw (图像 patch 网格)
        │     │  │     - MQ token 部分 → 对应的特殊 token IDs
        │     │  │
        │     │  └── 5. 返回: input_ids, attention_mask, pixel_values, image_grid_thw
        │     │
        │     │  encoder.py 中的后处理:
        │     │  ├── input_ids      = input_ids.to(cuda)
        │     │  ├── attention_mask = attention_mask.to(cuda)
        │     │  ├── pixel_values   = pixel_values.squeeze(0).to(cuda, bfloat16)
        │     │  └── image_sizes    = image_sizes.to(cuda)
        │     │
        │     └── 此时 input_ids 的结构:
        │           [...system tokens...|..user tokens..|<vision>图像patch tokens<vision_end>|
        │            文本tokens...|<begin_of_img>|<img0>|<img1>|...|<img255>|<end_of_img>|...]
        │
        ├── [B] 验证 BOI/EOI token 存在
        │     文件: encoder.py 第 380-403 行
        │     │
        │     ├── 找到 <begin_of_img> 位置: boi_pos_0
        │     ├── 找到 <end_of_img> 位置: eoi_pos_0
        │     └── 验证: eoi_pos_0 - boi_pos_0 - 1 == 256 (MQ token 数量正确)
        │
        ├── [C] ★★★ Qwen3-VL 完整前向传播 ★★★
        │     文件: encoder.py 第 411-416 行 → 调用 model.py 第 421-470 行
        │     │
        │     │  mq_features, mq_mask = self.mllm_model.encode_condition(
        │     │      input_ids=input_ids,           # 包含所有 token IDs
        │     │      attention_mask=attention_mask,
        │     │      pixel_values=pixel_values,     # ★ snoopy1.jpg 的像素数据
        │     │      image_sizes=image_sizes,       # 图像 patch 网格信息
        │     │  )
        │     │
        │     │  encode_condition() 内部 (model.py 第 421-470 行):
        │     │  │
        │     │  ├── [C1] Qwen3-VL 前向传播:
        │     │  │     prompt_embeds = self.mllm_backbone(    ← Qwen3VLForConditionalGeneration
        │     │  │         input_ids=input_ids,
        │     │  │         pixel_values=pixel_values,         ← ★ 图像在这里经过 ViT 编码
        │     │  │         image_grid_thw=image_sizes,
        │     │  │         attention_mask=attention_mask,
        │     │  │     ).logits                               ← lm_head=Identity, 实际返回 hidden_states
        │     │  │
        │     │  │     Qwen3-VL 内部执行:
        │     │  │     │
        │     │  │     ├── embed_tokens(input_ids) → 文本 + MQ token embedding
        │     │  │     ├── visual(pixel_values) → 图像 patch embedding
        │     │  │     ├── 将图像 patch embedding 替换到 input_ids 中 <image_pad> 的位置
        │     │  │     ├── 28 层 Transformer self-attention:
        │     │  │     │     每一层中，所有 token (文本 + 图像patch + MQ tokens)
        │     │  │     │     通过 attention 互相交互:
        │     │  │     │     - 文本 token 看到图像和 MQ tokens
        │     │  │     │     - 图像 patch token 看到文本和 MQ tokens
        │     │  │     │     - ★ MQ tokens 看到文本和图像所有 token
        │     │  │     │       → 通过 attention 从文本和图像中 "吸收" 语义信息
        │     │  │     │       → 经过 28 层累积，256 个 MQ token 的 hidden state
        │     │  │     │         已经凝聚了输入图片+文本的高层语义
        │     │  │     │
        │     │  │     └── output: [B, L_total, 1536] 所有 token 的最终隐藏状态
        │     │  │
        │     │  │     prompt_embeds shape: [1, L_total, 1536]
        │     │  │     其中 L_total = len(system) + len(user_text) + len(image_patches) + 258
        │     │  │
        │     │  ├── [C2] 提取 MetaQuery 隐藏状态:
        │     │  │     boi_pos = where(input_ids == boi_token_id)    # <begin_of_img> 位置
        │     │  │     eoi_pos = where(input_ids == eoi_token_id)    # <end_of_img> 位置
        │     │  │     mask = (indices > boi_pos) & (indices < eoi_pos)
        │     │  │     prompt_embeds = prompt_embeds[mask]            # 只取 BOI~EOI 之间的 256 个
        │     │  │     → shape: [1, 256, 1536]
        │     │  │
        │     │  └── [C3] Connector 处理:
        │     │        return self.connector(prompt_embeds), attention_mask
        │     │
        │     │        Connector 内部结构 (model.py 第 255-278 行):
        │     │        │
        │     │        ├── Qwen2Encoder (24 层双向 Transformer)
        │     │        │     hidden_size=1536, num_heads=24
        │     │        │     ★ 这 24 层双向 attention 让 256 个 MQ token 互相交互
        │     │        │       进一步整合和对齐各 token 携带的语义信息
        │     │        │     [1, 256, 1536] → [1, 256, 1536]
        │     │        │
        │     │        ├── nn.Linear(1536 → 2240)
        │     │        │     [1, 256, 1536] → [1, 256, 2240]
        │     │        │
        │     │        ├── nn.GELU(tanh)
        │     │        │
        │     │        ├── nn.Linear(2240 → 2240)
        │     │        │
        │     │        └── RMSNorm(2240)
        │     │              [1, 256, 2240] → [1, 256, 2240]
        │     │
        │     └── 返回: mq_features [1, 256, 2240], mq_mask
        │
        ├── [D] 验证 Connector 输出
        │     文件: encoder.py 第 418-448 行
        │     - shape == [1, 256, 2240] ✅
        │     - 非 NaN, 非 Inf, 非全零 ✅
        │
        ├── [E] to_wan_proj 投影到 Wan2.2 维度
        │     文件: encoder.py 第 451-470 行
        │     │
        │     │  wan_features = self.to_wan_proj(mq_features)
        │     │
        │     │  to_wan_proj 结构:
        │     │  ├── nn.Linear(2240 → 4096)
        │     │  ├── nn.GELU(tanh)
        │     │  └── nn.Linear(4096 → 4096)
        │     │
        │     └── wan_features shape: [1, 256, 4096]
        │
        └── [F] 返回
              return [wan_features[0]]   # → [Tensor shape [256, 4096]]
```

**所以回答你的核心问题：每次生成时，图片和文本确实经过了 Qwen3-VL 的完整前向传播。** 数据流路径：

```
snoopy1.jpg  ──► Qwen3-VL ViT ──► 图像 patch embedding ──┐
                                                           ├──► 28 层 Transformer
"Snoopy hit..." ──► embed_tokens ──► 文本 embedding ──────┤    (self-attention)
                                                           │         │
<img0>~<img255> ──► embed_tokens ──► MQ token embedding ──┘         │
                                                                     ▼
                                                    MQ tokens 的 hidden states [256, 1536]
                                                                     │
                                                                     ▼
                                                    Connector (24层 Qwen2Encoder) [256, 2240]
                                                                     │
                                                                     ▼
                                                    to_wan_proj [256, 4096]
```

#### 3.4.2 无条件编码的调用链

```python
mq_context_null = self.mq_encoder.encode([""], None)  # 无图像, 负面文本
```

流程相同，差异在于：
- `pixel_values = None` → Qwen3-VL ViT 不处理图像
- MQ tokens 只能从（空的）文本 token 中吸收信息
- 结果：无条件 MQ 特征，用于 CFG

---

### 3.5 Step 5/7: Context 拼接

```
文件: bridge_animate.py 第 510-518 行
```

```python
aug_context = self._augment_context(context, mq_context)
aug_context_null = self._augment_context(context_null, mq_context_null)
```

```
_augment_context() 内部 (bridge_animate.py 第 196-216 行):
  aug = torch.cat([mq_feat, t5_feat], dim=0)
  → [MQ:256 tokens | T5:L_t5 tokens]  shape=[256+L_t5, 4096]

  MetaQuery 特征在前，T5 在后
  → cross-attention 中 MQ 部分优先被关注
```

```
输入:
  context[0]    shape: [L_t5, 4096]   (T5 编码)
  mq_context[0] shape: [256, 4096]    (Qwen3-VL + Connector 编码)

输出:
  aug_context[0] shape: [256+L_t5, 4096]  (前 256 = MQ, 后 L_t5 = T5)
```

### 3.6 Step 6/7: CLIP 编码参考图

```
文件: bridge_animate.py 第 557-563 行
```

```python
ref_tensor = torch.tensor(ref_np / 127.5 - 1, dtype=bfloat16, device=device)
ref_tensor = rearrange(ref_tensor, "h w c -> c h w")
clip_context = wan.clip.visual([ref_tensor[:, None, :, :]])
# clip_context shape: [257, dim]  (CLS + 256 patches)
```

```
输入: ref_np (snoopy1.jpg 的 numpy 数组)
输出: clip_context [257, dim] — CLIP 全局视觉编码
```

### 3.7 Step 7/7: 逐 Clip 去噪循环

```
文件: bridge_animate.py 第 521-814 行
```

#### 准备阶段

```python
# 扩展 text_len
self._patch_wan_text_len(wan.noise_model, self._aug_text_len)
# text_len: 原始值 → 原始值 + 256
```

#### 每个 Clip 的循环

```
while True:
│
├── 构建面部张量:
│     face_clip = face_frames[start:end]    # 77 帧 [H=512, W=512, C=3]
│     face_pixel_values = tensor → [1, C=3, T=77, H=512, W=512]
│     (全零帧时 = -1.0, face_adapter 不起作用)
│
├── 参考图 VAE 编码:
│     ref_pv → wan.vae.encode → ref_latents [16, 1, H', W']
│     mask_ref [4, 1, H', W'] (前1帧=1, 其余=0)
│     y_ref = concat([mask_ref, ref_latents]) → [20, 1, H', W']
│
├── 时序引导帧 (首个 clip 时为零):
│     y_reft = vae.encode(zeros) / vae.encode(prev_clip_tail)
│     msk_reft = get_i2v_mask(...)
│     y_reft = concat([msk_reft, y_reft])
│
├── y = concat([y_ref, y_reft]) → [20, T'+1, H', W']
│     (参考图条件: mask 4ch + latent 16ch = 20ch)
│
├── 零 pose latent:
│     pose_latents = zeros(1, 16, T', H', W')  (骨架条件禁用)
│
├── 噪声初始化:
│     noise = randn(16, T'+1, H', W')
│
├── 去噪参数构建:
│     arg_c = {
│         "context": aug_context,              ← ★ MetaQuery + T5  (条件3)
│         "seq_len": max_seq_len,
│         "clip_fea": clip_context,            ← CLIP 视觉         (条件2)
│         "y": [y],                            ← 参考图 channel concat (条件1)
│         "pose_latents": pose_latents,        ← zeros (骨架禁用)
│         "face_pixel_values": face_pixel_values, ← 面部条件       (条件4)
│     }
│
├── 去噪循环 (20步):
│     for t in timesteps:
│     │
│     │   noise_pred_cond = wan.noise_model(latents, t, **arg_c)
│     │   │
│     │   │  WanAnimateModel.forward() 内部:
│     │   │  │
│     │   │  ├── Channel concat:
│     │   │  │     x_input = concat([noise(16ch), y(20ch)]) → 36ch
│     │   │  │     → patch_embedding → latent tokens
│     │   │  │
│     │   │  ├── CLIP 条件:
│     │   │  │     img_emb = self.img_emb(clip_fea)  → 投影到 DiT dim
│     │   │  │     → 在 cross-attention 中独立的 k_img, v_img
│     │   │  │
│     │   │  ├── 面部条件:
│     │   │  │     motion_encoder(face_pv) → face_encoder → face_adapter
│     │   │  │     → 每 5 层 block 通过交叉注意力注入
│     │   │  │
│     │   │  ├── ★ 文本 + MetaQuery context:
│     │   │  │     context = aug_context[0]  shape: [256+L_t5, 4096]
│     │   │  │     │
│     │   │  │     ├── text_embedding(context) → [256+L_t5, DiT_dim]
│     │   │  │     │     这个 Linear 层同时处理前 256 个 MQ token 和后面 T5 token
│     │   │  │     │     对 WanAnimateModel 来说它们都是 "context tokens"
│     │   │  │     │
│     │   │  │     ├── padding 到 text_len (= 原始 + 256)
│     │   │  │     │
│     │   │  │     └── 每个 WanAttentionBlock:
│     │   │  │           cross_attn(Q=latent, K/V=context_emb)
│     │   │  │           │
│     │   │  │           └── attention 权重分配:
│     │   │  │                 对 MQ 部分: 携带了 Qwen3-VL 提取的高层视觉语义
│     │   │  │                 对 T5 部分: 携带了 T5 编码的文本语义
│     │   │  │                 → DiT 自动学习如何从两类 token 中提取信息
│     │   │  │
│     │   │  └── 经过 N 个 block → output noise_pred
│     │   │
│     │   │  (如果 guide_scale > 1, 还会计算 noise_pred_uncond 做 CFG)
│     │   │
│     │   latents = scheduler.step(noise_pred, t, latents)
│     │
│     └── 20 步后 latents 收敛
│
├── VAE 解码:
│     out_frames = wan.vae.decode(latents[:, 1:])  → video frames
│     all_out_frames.append(out_frames)
│
└── 下一个 clip (如果有多个)
```

#### 循环结束后

```python
# 恢复 text_len
self._restore_wan_text_len(wan.noise_model)

# 拼接所有 clip
videos = torch.cat(all_out_frames, dim=2)[:, :, :frame_num]

# 返回
return videos[0]  # shape: [C=3, F=77, H, W]
```

### 3.8 保存视频

```
文件: demo_metaquery_animate.py 第 206-216 行 → save_video_as_mp4()
```

```python
video_np = (video * 0.5 + 0.5).clamp(0, 1) * 255  # [-1,1] → [0,255]
cv2.VideoWriter → .mp4 文件
```

---

## 4. 完整文件调用关系图

```
demo_metaquery_animate.py
│
├── [初始化阶段]
│   ├── wan/animate.py                          → WanAnimate.__init__()
│   ├── wan/metaquery/__init__.py               → import MetaQueryWanAnimateBridge
│   ├── wan/metaquery/bridge_animate.py         → MetaQueryWanAnimateBridge.__init__()
│   │   └── wan/metaquery/encoder.py            → MetaQueryEncoder.__init__()
│   │       ├── metaquery-main/trainer_utils.py → find_newest_checkpoint()
│   │       └── metaquery-main/models/
│   │           ├── metaquery.py                → MetaQuery.from_pretrained()
│   │           └── model.py                    → MLLMInContext.__init__()
│   │               ├── Qwen3VLForConditionalGeneration (transformers)
│   │               ├── SanaTransformer2DModel (diffusers)
│   │               └── Qwen2Encoder (transformers)
│   │
│   └── 初始化完成，所有模型权重已加载
│
├── [推理阶段] bridge.generate()
│   ├── wan/metaquery/bridge_animate.py         → generate()
│   │   ├── T5 编码
│   │   │   └── wan 内置 text_encoder
│   │   │
│   │   ├── ★ MetaQuery 编码
│   │   │   └── wan/metaquery/encoder.py        → encode()
│   │   │       └── metaquery-main/models/model.py → MLLMInContext.encode_condition()
│   │   │           ├── Qwen3VLForConditionalGeneration.forward()  ★ 完整前向传播
│   │   │           │   ├── visual() ← 图像经过 ViT
│   │   │           │   └── model() ← 28 层 Transformer (文本+图像+MQ token)
│   │   │           ├── 提取 BOI~EOI 间 256 个 MQ 隐藏状态
│   │   │           └── connector() ← 24 层 Qwen2Encoder + Linear
│   │   │
│   │   ├── Context 拼接 (MQ + T5)
│   │   ├── CLIP 编码
│   │   │   └── wan 内置 clip.visual()
│   │   │
│   │   └── 去噪循环
│   │       └── wan/modules/animate/           → WanAnimateModel.forward()
│   │           ├── channel concat (参考图条件)
│   │           ├── img_emb cross-attn (CLIP 条件)
│   │           ├── text cross-attn (MQ+T5 条件)    ★ MetaQuery 在此注入
│   │           └── face_adapter (面部条件)
│   │
│   └── VAE 解码 → video tensor
│
└── save_video_as_mp4() → .mp4 文件
```

---

## 5. 各条件在 DiT 中的注入点汇总

```
WanAnimateModel.forward() 内部各条件的注入位置:

    输入层:
    ┌────────────────────────────────────────────────┐
    │  noise_latent (16ch) + y_ref+y_reft (20ch)     │  条件1: 参考图
    │  = 36ch → patch_embedding                      │  (Channel Concat)
    └────────────────────────────┬───────────────────┘
                                 │
    Transformer Block 1~N:       │
    ┌────────────────────────────┼───────────────────┐
    │                            ▼                    │
    │  ┌─── self_attn ←── latent tokens              │
    │  │                                              │
    │  ├─── cross_attn ←── k_text/v_text             │  条件3: MetaQuery + T5
    │  │                    (MQ[256]+T5[L])           │  (Context Concat → Cross-Attn)
    │  │                                              │
    │  ├─── cross_attn ←── k_img/v_img               │  条件2: CLIP
    │  │                    (clip_context[257])        │  (Image Cross-Attn)
    │  │                                              │
    │  └─── (每5层) face_adapter ←── face_feat       │  条件4: 面部
    │                                                  │  (Face Adapter Cross-Attn)
    └──────────────────────────────────────────────────┘
```

---

## 6. 总结：初始化 vs 推理不冲突

```
时间线:
═══════════════════════════════════════════════════════════════════════

t=0  程序启动
     │
t=1  WanAnimate.__init__()          ← 加载 T5/VAE/CLIP/DiT 权重
     │
t=2  MetaQueryWanAnimateBridge()    ← 加载 Qwen3-VL + Connector 权重
     │                                 (★ 此时只是加载权重，不处理数据)
     │                                 (你看到的 bridge_kwargs 代码在这里)
     │
     │  ══ 初始化完成 ═══════════════════════════════════════════════
     │
t=3  bridge.generate(prompt, ref_image, ...)   ← 开始生成
     │
t=4    T5.encode("Snoopy ...")                  ← 文本经过 T5
     │
t=5    mq_encoder.encode(prompt, [snoopy.jpg])  ← ★ 图片+文本经过 Qwen3-VL
     │  │                                          (你看到的 encode_condition 在这里)
     │  ├── tokenize → input_ids + pixel_values
     │  ├── Qwen3-VL 完整前向传播 (28层)
     │  ├── 提取 256 个 MQ hidden states
     │  ├── Connector (24层)
     │  └── to_wan_proj → [256, 4096]
     │
t=6    concat(MQ, T5) → aug_context             ← 拼接
     │
t=7    CLIP.visual(snoopy.jpg)                   ← 图片经过 CLIP
     │
t=8    去噪循环 × 20 步                          ← 所有条件注入 DiT
     │  (每步: aug_context + clip + ref + face → WanAnimateModel)
     │
t=9    VAE.decode → video                        ← 解码
     │
t=10   save_video_as_mp4 → .mp4                  ← 保存
═══════════════════════════════════════════════════════════════════════
```

**核心结论：**

1. **`bridge_kwargs` 和 `MetaQueryWanAnimateBridge(**bridge_kwargs)`** — 这是 **t=2** 时刻的初始化代码，负责加载 Qwen3-VL 模型权重到 GPU。此时**没有任何用户数据（图片/文本）被处理**。

2. **`self.mllm_model.encode_condition(input_ids, pixel_values, ...)`** — 这是 **t=5** 时刻的推理代码，在 `bridge.generate()` 被调用时才执行。此时用户的 snoopy1.jpg 和 "Snoopy hit the tennis ball..." **确实经过了 Qwen3-VL 的完整 28 层 Transformer 前向传播**。

3. **两段代码不冲突**，它们是同一管线的两个阶段：先初始化模型，再用模型处理数据。
