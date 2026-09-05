# MetaQuery + Qwen3-VL + Wan2.2 Animate 完整管线详解

> 从原始输入到最终视频输出的完整流程拆解

---

## 一、管线总览

本管线将三套系统组合为一条视频生成管线：

| 系统 | 角色 | 核心能力 |
|------|------|---------|
| **Qwen3-VL** | 多模态语言模型 (MLLM) | 理解图像内容，提取高层语义特征 |
| **MetaQuery** | 视觉条件桥接器 | 通过可学习 query token 从 MLLM 提取条件特征 |
| **Wan2.2 Animate** | 视频扩散模型 | 基于扩散过程生成连续视频帧 |

**一句话概括**：用户给一张人物参考图 + 文本描述，Qwen3-VL 理解图像语义，MetaQuery 将语义编码为 256 个 token，与 T5 文本特征一起注入 Wan2.2 Animate 的去噪过程，同时参考图还通过 VAE 和 CLIP 直接提供像素级和全局视觉条件，最终生成人物动画视频。

---

## 二、完整输入一览

```
┌────────────────────────────────────────────────────────────────────┐
│                        用户输入                                     │
├───────────────────────┬─────────────┬──────────────────────────────┤
│        必需输入        │   可选输入    │          模型资源              │
├───────────────────────┼─────────────┼──────────────────────────────┤
│ ref_image             │ face_source │ Wan2.2 animate-14B 权重       │
│ (参考人物图, PIL)      │ (面部视频)   │ MetaQuery+Qwen3-VL 权重      │
│                       │             │ T5-XXL 文本编码器             │
│ input_prompt          │ mq_ref_imgs │ CLIP ViT-H/14                │
│ (文本提示词, str)      │ (额外语义图) │ Wan VAE                      │
└───────────────────────┴─────────────┴──────────────────────────────┘
```

### 2.1 必需输入

| 输入 | 类型 | 说明 |
|------|------|------|
| `ref_image` | PIL.Image | 参考人物图。定义了生成视频中的人物外观和输出分辨率 |
| `input_prompt` | str | 文本提示词。有默认值，通常不需要自定义 |

### 2.2 可选输入（参考视频已是可选项）

| 输入 | 类型 | 默认行为 | 说明 |
|------|------|---------|------|
| `face_source` | str / List / None | **None** → 传全零面部帧 | 面部驱动视频，控制表情动作 |
| `mq_reference_images` | List[PIL.Image] / None | **None** → 用 ref_image | 额外语义参考图（风格、场景等） |
| 骨架视频 (pose) | — | **永远传零** | 已被 Bridge 彻底移除，不需要提供 |

---

## 三、完整流程图

```
                        ┌─────────────────────┐
                        │     用户输入          │
                        │  ref_image + prompt  │
                        │  (+ face_source?)    │
                        │  (+ mq_ref_images?)  │
                        └──────┬──────────────┘
                               │
            ┌──────────────────┼────────────────────────────┐
            │                  │                            │
   ┌────────▼────────┐ ┌──────▼──────┐           ┌─────────▼─────────┐
   │  路径 A: Text    │ │ 路径 B: MQ  │           │   路径 C: Wan     │
   │  T5-XXL 文本编码 │ │ Qwen3-VL +  │           │   原始条件通路     │
   │                  │ │ MetaQuery   │           │                    │
   └────────┬────────┘ └──────┬──────┘           │  ┌──────┐ ┌─────┐ │
            │                 │                   │  │ VAE  │ │CLIP │ │
            │                 │                   │  └──┬───┘ └──┬──┘ │
            │                 │                   │     │        │    │
            │                 │                   │  ┌──▼───┐ ┌──▼──┐ │
            │                 │                   │  │y 条件│ │img_ │ │
            │                 │                   │  │20ch  │ │emb  │ │
            │                 │                   │  └──┬───┘ └──┬──┘ │
            │                 │                   │     │        │    │
            │                 │                   └─────┼────────┼────┘
            │                 │                         │        │
            ▼                 ▼                         ▼        ▼
   ┌──────────────────────────────────┐     ┌──────────────────────┐
   │      context concat              │     │  额外条件通道          │
   │  [MQ:256, 4096] ⊕ [T5:L, 4096]  │     │  + face_adapter      │
   │  = 增强 context                   │     │  + pose (全零)       │
   └───────────────┬──────────────────┘     └──────────┬───────────┘
                   │                                    │
                   └──────────────┬─────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  WanAnimateModel.forward()    │
                   │  42 层 Transformer Block      │
                   │  逐步去噪 (20~50步)            │
                   │  → latent [16, T, H/8, W/8]  │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Wan VAE Decoder              │
                   │  latent → 像素视频            │
                   │  [3, F, H, W]                 │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │       输出: MP4 视频          │
                   └──────────────────────────────┘
```

---

## 四、逐步详解

### Step 1: 参考图预处理

**代码位置**: `bridge_animate.py` `generate()` 第 420-432 行

```python
ref_np = np.array(ref_image)
height = (ref_np.shape[0] // 8) * 8   # 向下对齐到 8 的倍数
width  = (ref_np.shape[1] // 8) * 8
ref_np = self._padding_resize(ref_np, height=height, width=width)
```

| 操作 | 说明 |
|------|------|
| 加载 PIL.Image | 转为 numpy 数组 |
| 分辨率对齐 | H, W 向下取整到 8 的倍数 |
| padding_resize | 等比缩放 + 黑边填充到目标尺寸 |

**这一步决定了输出视频的分辨率** — 生成视频的 H×W 与参考图对齐后的尺寸一致。

---

### Step 2: 面部帧准备（可选）

**代码位置**: `bridge_animate.py` `generate()` 第 435-446 行

```python
if face_source is not None:
    face_frames = self._load_face_video(face_source, frame_num)
    # 每帧自动 resize 到 512×512
else:
    face_frames = [np.zeros((512, 512, 3), dtype=np.uint8)] * frame_num
    # 归一化后变 -1.0，与 CFG 无条件分支一致，使 face_adapter 不起作用
```

| face_source 值 | 行为 |
|:--|:--|
| `None` | 生成 frame_num 个全零 512×512 帧，面部条件不起作用 |
| 视频路径 str | 用 decord 读取，每帧 resize 到 512×512 |
| List[np.ndarray] | 直接使用，不足的帧数循环填充 |
| List[PIL.Image] | 转 ndarray，resize 到 512×512 |

---

### Step 3: T5-XXL 文本编码

**代码位置**: `bridge_animate.py` 第 449-462 行

```
input_prompt  ──→  T5-XXL Encoder  ──→  context [L_t5, 4096]
n_prompt      ──→  T5-XXL Encoder  ──→  context_null [L_t5, 4096]
```

T5-XXL 将文本编码为长度为 L_t5 (通常 ≤ 512) 的 4096 维特征序列。Animate 通常使用固定提示词（如「视频中的人在做动作」），用户一般不需要自定义。

---

### Step 4: MetaQuery (Qwen3-VL) 语义编码

**代码位置**: `bridge_animate.py` 第 465-493 行 + `encoder.py` `encode()`

这是 MetaQuery 的核心步骤，也是本管线相比原始 WanAnimate 的**主要增量**：

```
参考图(PIL.Image) ──→ Qwen3-VL AutoProcessor ──→ pixel_values + input_ids
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │  Qwen3-VL Forward   │
                                            │  (冻结的 MLLM)       │
                                            │                     │
                                            │  input_ids 中含有:  │
                                            │  ...<boi>           │
                                            │  <img>×256          │
                                            │  <eoi>...           │
                                            │                     │
                                            │  提取 BOI~EOI 间的  │
                                            │  hidden states      │
                                            └─────────┬───────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────────┐
                                            │  Connector           │
                                            │  (Qwen2Encoder +     │
                                            │   Linear + GELU +    │
                                            │   Linear + RMSNorm)  │
                                            │  → [256, 2240]       │
                                            └─────────┬───────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────────┐
                                            │  to_wan_proj         │
                                            │  Linear(2240→4096)   │
                                            │  GELU               │
                                            │  Linear(4096→4096)   │
                                            │  → [256, 4096]       │
                                            └─────────────────────┘
```

**输入**：`mq_reference_images`（若为 None，自动使用 `ref_image`）
**输出**：`mq_context [256, 4096]` — 256 个语义 token，维度与 T5 一致

**关键理解**：
- Qwen3-VL 的 `AutoProcessor` 支持**动态分辨率**，自动处理任意尺寸图片
- 256 个 MetaQuery token 是通过在 prompt 中插入 `<boi><img>×256<eoi>` 特殊标记实现的
- MLLM forward 后，提取 BOI 到 EOI 之间的 hidden states，这就是"可学习的视觉查询"
- Connector 是预训练好的映射网络，将 MLLM hidden states 投影到扩散模型可用的空间
- `to_wan_proj` 进一步将 Connector 输出投影到 Wan 的 4096 维 text embedding 空间

---

### Step 5: Context 拼接 (T5 + MetaQuery)

**代码位置**: `bridge_animate.py` 第 496-503 行

```
aug_context = cat([mq_context, t5_context], dim=0)
# [256, 4096] ⊕ [L_t5, 4096] = [256 + L_t5, 4096]
```

MetaQuery 特征 **前置拼接** 到 T5 特征上。拼接后的增强 context 长度为 `256 + L_t5`。

同时，`WanAnimateModel.text_len` 临时从原始值（如 512）扩展到扩展值（如 768），使模型的 text_embedding 层能容纳更长的 context。这个操作通过 `try/finally` 保护，确保无论是否异常都会恢复。

---

### Step 6: 扩展 text_len + CLIP 编码参考图

**代码位置**: `bridge_animate.py` 第 534-550 行

```python
# 在 try 块内:
self._patch_wan_text_len(wan.noise_model, self._aug_text_len)  # 512 → 768

# CLIP ViT-H/14 编码参考图
ref_tensor = torch.tensor(ref_np / 127.5 - 1, ...)  # 归一化到 [-1,1]
clip_context = wan.clip.visual([ref_tensor[:, None, :, :]])
# → clip_context [257, embed_dim] (256 patch + 1 cls token)
# CLIP 内部自动 resize 到 224×224
```

**CLIP 的作用**：提供全局视觉语义特征。在 WanAnimateModel 的每个 Transformer block 中，`WanAnimateCrossAttention` 计算 `k_img, v_img = img_emb(clip_context)` 并与文本注意力输出**相加**。

---

### Step 7: 逐 Clip 去噪循环

**代码位置**: `bridge_animate.py` 第 557-770 行

Animate 管线特有的**多 clip 生成机制**。当目标帧数 > clip_len 时，将视频分成多段（clip），逐段去噪：

```
Clip 1: frames [0, clip_len)
    ├─ 初始化: 随机噪声
    ├─ 面部帧: face_frames[0:clip_len]
    ├─ 参考图 VAE: ref_image → VAE encode → y_ref [20, 1, H/8, W/8]
    ├─ 时序引导: y_reft 为全零 (首个 clip 无前序帧)
    ├─ y = cat([y_ref, y_reft]) → [20, lat_t+1, H/8, W/8]
    ├─ pose_latents: 全零 [1, 16, lat_t, H/8, W/8]
    └─ 去噪 N 步 → VAE decode → out_frames

Clip 2: frames [clip_len-1, 2*clip_len-2)
    ├─ 初始化: 随机噪声
    ├─ 面部帧: face_frames[clip_len-1:2*clip_len-2]
    ├─ 参考图 VAE: 同上
    ├─ 时序引导: y_reft = VAE encode(前一 clip 最后 refert_num 帧)
    └─ ...

最终: torch.cat(all_clips)[:, :, :frame_num] → 完整视频
```

### Step 7.1: 每个 Clip 内的条件构建

```python
# y 条件 (channel concat)
y_ref  = cat([mask_ref, VAE_encode(ref_image)])   # [20, 1, H/8, W/8]
y_reft = cat([mask_reft, VAE_encode(前序帧/零)])   # [20, lat_t, H/8, W/8]
y = cat([y_ref, y_reft], dim=1)                     # [20, lat_t+1, H/8, W/8]

# pose 条件 (全零, 不使用骨架)
pose_latents = torch.zeros(1, 16, lat_t, H/8, W/8)

# face 条件
face_pixel_values = face_frames / 127.5 - 1  # [1, 3, clip_len, 512, 512]
```

### Step 7.2: 每个去噪步

```python
for t in timesteps:
    # 有条件前向
    noise_pred_cond = wan.noise_model(
        latents,
        t=t,
        context=aug_context,        # T5 + MetaQuery
        clip_fea=clip_context,      # CLIP 特征
        y=[y],                      # VAE 参考图 + 时序帧
        pose_latents=pose_latents,  # 全零 (无骨架)
        face_pixel_values=face_pv,  # 面部帧
    )

    # CFG (如果 guide_scale > 1)
    if guide_scale > 1:
        noise_pred_uncond = wan.noise_model(
            latents, t=t,
            context=aug_context_null,
            clip_fea=clip_context,
            y=[y],
            pose_latents=pose_latents,
            face_pixel_values=face_pv * 0 - 1,  # 面部置零
        )
        noise_pred = uncond + scale * (cond - uncond)

    latents = scheduler.step(noise_pred, t, latents)
```

---

### Step 8: WanAnimateModel 内部的条件注入点

WanAnimateModel 有 42 个 Transformer Block (`WanAttentionBlock`)。每个 block 内部的条件注入方式如下：

```
输入 x [B, S, D]  (S = spatial tokens, D = model_dim)
    │
    ├─ self_attention(x)
    │
    ├─ cross_attention(x, context)  ← 这里注入 T5 + MetaQuery 条件
    │   └─ WanAnimateCrossAttention:
    │       ├─ q = norm(x) → linear_q → q
    │       ├─ k_text, v_text = linear_kv(context)  ← T5 + MQ tokens
    │       ├─ k_img, v_img = img_emb(clip_fea)     ← CLIP 257 tokens
    │       ├─ attn_text = softmax(q @ k_text^T) @ v_text
    │       ├─ attn_img  = softmax(q @ k_img^T)  @ v_img
    │       └─ output = attn_text + attn_img          ← 文本+视觉注意力相加
    │
    ├─ (每 5 层) face_adapter(x, face_features)  ← 面部条件注入
    │
    ├─ feed_forward(x)
    │
    └─ x 通过 diag channel concat 接收 y 和 pose_latents
       (在 patch_embedding 阶段已与噪声 cat 成 36ch 输入)
```

**5 个条件在模型中的注入位置：**

| 条件 | 注入位置 | 注入方式 | 影响范围 |
|------|---------|---------|---------|
| T5 文本 | 每层 cross_attention | `k_text, v_text` 来自 context 的 T5 部分 | 全局语义 |
| MetaQuery | 每层 cross_attention | `k_text, v_text` 来自 context 的 MQ 部分 | 细粒度视觉语义 |
| CLIP | 每层 cross_attention | `k_img, v_img` → 与文本注意力输出相加 | 全局视觉一致性 |
| 参考图 VAE | patch_embedding 阶段 | 20ch (mask+latent) concat 到 16ch 噪声 = 36ch | 像素级结构 |
| 面部 | 每 5 层 face_adapter | 交叉注意力注入 motion features | 面部动作驱动 |

---

### Step 9: VAE 解码

```python
x0 = latents
out_frames = wan.vae.decode([x0[0][:, 1:]])  # 跳过参考帧位
# → [1, 3, F_clip, H, W]，值域 [-1, 1]
```

VAE 将 latent 空间 `[16, T_lat, H/8, W/8]` 上采样回像素空间 `[3, F, H, W]`。

---

### Step 10: 拼接所有 Clip + 保存

```python
videos = torch.cat(all_out_frames, dim=2)[:, :, :frame_num]
# → [1, 3, frame_num, H, W]

# 保存为 MP4
save_video_as_mp4(videos[0], output_path, fps=30)
```

---

## 五、参考视频是否为可选项？

### 回答：**参考视频已经是可选项，不需要修改代码。**

当前 `MetaQueryWanAnimateBridge.generate()` 的参数设计：

```python
def generate(
    self,
    input_prompt: str = "",              # ← 可选，有默认值
    ref_image: Image.Image = None,       # ← 必需（参考人物图）
    face_source = None,                  # ← 可选（面部视频）
    mq_reference_images = None,          # ← 可选（MetaQuery 额外参考图）
    ...
):
```

| 输入 | 是否可选？ | 不提供时的行为 |
|------|:---------:|-------------|
| `ref_image` | **必需** | assert 报错。这是人物外观的唯一来源，不可省略 |
| `input_prompt` | 可选 | 使用配置中的默认提示词 |
| `face_source` | **可选** | 生成全零 512×512 帧 → 面部条件不起作用 |
| `mq_reference_images` | **可选** | 自动使用 `ref_image` 作为 MetaQuery 输入 |
| 骨架视频 (pose) | **已移除** | 永远传零 `pose_latents`，接口中不暴露此参数 |

**对比原始 WanAnimate**：

| 输入 | 原始 WanAnimate | MetaQuery Bridge |
|------|:---:|:---:|
| 参考人物图 (`src_ref.png`) | 必需 | 必需 (`ref_image`) |
| 骨架视频 (`src_pose.mp4`) | **必需** | **已移除** |
| 面部视频 (`src_face.mp4`) | **必需** | **可选** (None → 全零) |
| MetaQuery 参考图 | — | 可选 (默认用 ref_image) |

### 最小输入示例（仅角色参考图 + 文本）

```python
video = bridge.generate(
    input_prompt="一个女孩在跳舞",
    ref_image=Image.open("person.jpg").convert("RGB"),
    # face_source=None,           # 不提供面部视频 → 面部条件不起作用
    # mq_reference_images=None,   # 不提供额外参考图 → 用 ref_image
)
```

这样只有三个条件通道起作用：
1. **VAE 参考图** — 提供像素级结构
2. **CLIP 参考图** — 提供全局视觉语义
3. **MetaQuery** — 提供 Qwen3-VL 理解的细粒度语义

面部和骨架条件均为零，不起作用。

---

## 六、输入分辨率要求

### 6.1 参考人物图 (`ref_image`)

| 约束 | 说明 |
|------|------|
| 分辨率 | **任意**。代码自动对齐到 8 的倍数 |
| 最小建议 | 短边 ≥ 512px |
| 最大建议 | 短边 ≤ 1280px（受 GPU 显存限制） |
| 宽高比 | **直接决定输出视频的宽高比** |
| 推荐 | 720×1280 或 1280×720（与 Wan 训练分布一致） |

**分辨率处理流程**：
```
原图 (如 1920×1080)
  → height = (1080 // 8) * 8 = 1080   ← 已是8的倍数，不变
  → width  = (1920 // 8) * 8 = 1920
  → padding_resize(ref, 1080, 1920)   ← 等比缩放 + 黑边填充
  → 输出视频分辨率: 1920×1080
```

### 6.2 面部视频 (`face_source`) — 可选

| 约束 | 说明 |
|------|------|
| 分辨率 | **任意**。代码自动 resize 到 512×512 |
| 推荐 | 直接提供 512×512 裁剪的面部区域（避免插值损失） |
| 帧数 | 不需要固定，代码自动循环填充到 frame_num |
| 格式 | MP4 视频路径 / numpy 数组列表 / PIL.Image 列表 |

### 6.3 MetaQuery 参考图 (`mq_reference_images`) — 可选

| 约束 | 说明 |
|------|------|
| 分辨率 | **任意**。Qwen3-VL AutoProcessor 动态处理 |
| 有效范围 | 总像素 20万~100万 (由 min/max_pixels 控制) |
| 推荐 | 短边 ≥ 448px，保持原始分辨率即可 |
| 数量 | 可以是 1 张或多张 |

### 6.4 帧数 (`frame_num` / `clip_len`)

| 约束 | 说明 |
|------|------|
| 格式 | **必须为 4n+1**（如 77, 81, 121, 161） |
| 原因 | VAE 时间压缩比为 4，要求 `(F-1) % 4 == 0` |
| 检查 | assert 硬检查，不满足则报错 |

---

## 七、各组件数据流维度追踪

以 `ref_image=720×1280, frame_num=77, num_metaqueries=256` 为例：

```
T5 文本编码:
  input_prompt → T5-XXL → context [L≤512, 4096]

MetaQuery 编码:
  ref_image (720×1280)
  → Qwen3-VL AutoProcessor → pixel_values [~N_patches, 1176]
  → Qwen3-VL forward → hidden_states [seq_len, 3584]
  → 提取 BOI-EOI 间 → [256, 3584]
  → Connector → [256, 2240]
  → to_wan_proj → [256, 4096]

Context 拼接:
  [256, 4096] ⊕ [L, 4096] → aug_context [256+L, 4096]

CLIP 编码:
  ref_image → resize 224×224 → CLIP ViT-H/14 → [257, 1280]
  → MLPProj → [257, model_dim]

VAE 参考图编码:
  ref_image 720×1280 → 归一化 [-1,1] → VAE encode
  → latent [16, 1, 90, 160]
  → + mask [4, 1, 90, 160]
  → y_ref [20, 1, 90, 160]

面部编码 (per clip):
  face_frames [77, 512, 512, 3] → 归一化 → [1, 3, 77, 512, 512]
  → motion_encoder + face_encoder → face features

噪声初始化:
  latent_shape = [16, (77-1)//4+1+1, 720//8, 1280//8]
               = [16, 20, 90, 160]

去噪循环 (20~50步):
  WanAnimateModel(
    x=[16, 20, 90, 160],     # 噪声 latent
    + y=[20, 20, 90, 160],   # channel concat → [36, 20, 90, 160]
    context=[256+L, 4096],    # T5 + MQ
    clip_fea=[257, dim],      # CLIP
    face_pv=[1,3,77,512,512], # 面部
    pose=[1,16,19,90,160],    # 全零
  )
  → noise_pred [16, 20, 90, 160]

VAE 解码:
  latent [16, 19, 90, 160] → VAE decode → [3, 77, 720, 1280]

输出: MP4 视频 [3, 77, 720, 1280]
```

---

## 八、与原始 WanAnimate 的差异总结

| 方面 | 原始 WanAnimate | MetaQuery Bridge |
|------|----------------|-----------------|
| 输入接口 | `src_root_path` (含 pose/face/ref 三个文件) | 分离参数：`ref_image`, `face_source`, `mq_ref_images` |
| 骨架条件 | **必需** — 从 `src_pose.mp4` 提取 | **移除** — 传零 pose_latents |
| 面部条件 | **必需** — 从 `src_face.mp4` 提取 | **可选** — None 时传零 |
| CLIP 条件 | 有 | 有（不变） |
| VAE 参考图 | 有 | 有（不变） |
| 文本条件 | T5 only | T5 + **MetaQuery 256 token** |
| text_len | 固定 512 | 动态 512 → 768（try/finally 保护） |
| CFG | 仅面部表情 | 同上 |
| 输出分辨率 | 来自骨架视频 | 来自参考图（对齐到 8 倍数） |

---

## 九、端到端示例代码

```python
from PIL import Image
from wan.configs import WAN_CONFIGS
from wan.animate import WanAnimate
from wan.metaquery import MetaQueryWanAnimateBridge

# 1. 加载 Wan Animate
wan = WanAnimate(
    config=WAN_CONFIGS["animate-14B"],
    checkpoint_dir="/path/to/wan-animate-14B",
    device_id=0, rank=0,
)

# 2. 初始化 MetaQuery Bridge
bridge = MetaQueryWanAnimateBridge(
    wan_animate_pipeline=wan,
    metaquery_checkpoint="/path/to/metaquery-ckpt",
    num_metaqueries=256,
)

# 3. 最小输入 — 仅参考图 + 文本
video = bridge.generate(
    input_prompt="一个女孩在微笑",
    ref_image=Image.open("person.jpg").convert("RGB"),
    frame_num=77,
    seed=42,
)

# 4. 完整输入 — 参考图 + 面部视频 + 额外语义参考图
video = bridge.generate(
    input_prompt="一个女孩在跳舞",
    ref_image=Image.open("person.jpg").convert("RGB"),
    face_source="face_512x512.mp4",
    mq_reference_images=[Image.open("dance_ref.jpg").convert("RGB")],
    frame_num=77,
    clip_len=77,
    sampling_steps=20,
    guide_scale=1.5,   # > 1 启用 CFG，增强面部表情
    seed=42,
)
```
