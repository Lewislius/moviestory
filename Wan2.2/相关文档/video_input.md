# Video Input 分析：WanAnimate 的视频输入机制全解

> 本文档回答：`EXAMPLE_PROMPT["animate-14B"]` 中的 `"video"` 是什么？它和面部视频有什么关系？WanAnimate 到底将什么作为条件输入？原始视频是否直接进入 DiT？

---

## 一、`EXAMPLE_PROMPT` 中 video/pose/mask 字段的真相

### 1.1 代码现状

```python
# generate.py 第 39-44 行
"animate-14B": {
    "prompt": "视频中的人在做动作",
    "video": "",
    "pose": "",
    "mask": "",
},
```

### 1.2 这些字段实际被使用了吗？

**没有。** 这三个字段 (`video`, `pose`, `mask`) 是占位符，在代码中**完全没有被引用**。

查看 `_validate_args()` 函数，它只从 `EXAMPLE_PROMPT` 中读取以下字段：

| 字段 | 读取代码 | 涉及的 task |
|------|---------|------------|
| `prompt` | `args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]` | 所有 task |
| `image` | `args.image = EXAMPLE_PROMPT[args.task]["image"]` | i2v, s2v |
| `audio` | `args.audio = EXAMPLE_PROMPT[args.task]["audio"]` | s2v |
| `tts_*` | `args.tts_* = EXAMPLE_PROMPT[args.task]["tts_*"]` | s2v (TTS模式) |

**`video`、`pose`、`mask` 从未被任何代码读取。** 它们仅仅是示例字典中的文档性占位键，可能是开发者留给未来扩展或给用户做参考用的。

### 1.3 那 animate 任务的视频输入在哪？

animate 任务使用的是 CLI 参数 `--src_root_path`，而不是 `EXAMPLE_PROMPT` 中的 `video` 字段：

```python
# generate.py 第 472 行
video = wan_animate.generate(
    src_root_path=args.src_root_path,   # ← 这里！不是 EXAMPLE_PROMPT["video"]
    ...
)
```

`--src_root_path` 指向一个**预处理后的文件夹**，内含从原始视频提取出的多种条件素材。

---

## 二、原始视频 → 条件素材：完整的预处理流程

### 2.1 核心结论

> **原始视频不会直接传入 WanAnimate 的 DiT 模型。** 它必须先经过预处理脚本，被分解为骨架视频、面部视频等条件素材，然后这些条件素材才被送入 DiT。

### 2.2 预处理流程总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户提供的原始输入                              │
│                                                                  │
│   原始驱动视频 (video.mp4)          参考角色图 (image.jpeg)        │
│   ├─ 包含一个真实人物的动作表演        └─ 目标角色的外观照片         │
│   ├─ 例：一个人在跳舞的视频                                       │
│   └─ 例：一个人在说话的视频                                       │
└────────────────────┬──────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              预处理脚本 (preprocess_data.py)                      │
│                                                                  │
│   使用的模型:                                                     │
│   ├─ YOLOv10m (人体检测)                                         │
│   ├─ ViTPose-H (全身关键点检测, 133点)                            │
│   ├─ SAM2 (人物分割, 仅 replacement 模式)                        │
│   └─ FLUX.1-Kontext (姿态重定向编辑, 可选)                       │
│                                                                  │
│   处理步骤:                                                       │
│   1. 按目标 FPS 抽帧                                              │
│   2. 按 resolution_area 缩放到目标分辨率                          │
│   3. 逐帧检测人体关键点                                           │
│   4. 从每帧裁剪 512×512 面部区域                                  │
│   5. 将关键点渲染为骨架图 (可选姿态重定向)                        │
│   6. (replacement 模式) 提取人物蒙版和背景                        │
└────────────────────┬──────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│             预处理输出文件夹 (src_root_path/)                     │
│                                                                  │
│   src_ref.png      ← 参考角色图 (直接复制)                       │
│   src_pose.mp4     ← 骨架视频 (白色骨架线条在黑色背景上)         │
│   src_face.mp4     ← 面部视频 (从每帧裁剪的 512×512 人脸)       │
│   src_bg.mp4       ← 背景视频 (仅 replacement 模式)             │
│   src_mask.mp4     ← 蒙版视频 (仅 replacement 模式)             │
│                                                                  │
│   ⚠️ 注意：原始视频 (video.mp4) 不会出现在这个文件夹中！          │
│   所有原始像素信息都已被转换为结构化条件                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 预处理脚本的关键代码

```python
# process_pipepline.py (animation 模式)

# 1. 人体关键点检测
tpl_pose_metas = self.pose2d(frames)

# 2. 面部裁剪 → src_face.mp4
for idx, meta in enumerate(tpl_pose_metas):
    face_bbox = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, ...)
    x1, x2, y1, y2 = face_bbox
    face_image = frames[idx][y1:y2, x1:x2]       # 从原帧裁剪面部区域
    face_image = cv2.resize(face_image, (512, 512)) # 统一缩放到 512×512
    face_images.append(face_image)

# 3. 骨架渲染 → src_pose.mp4
for idx, meta in enumerate(tpl_retarget_pose_metas):
    canvas = np.zeros_like(refer_img)              # 黑色画布
    conditioning_image = draw_aapose_by_meta_new(canvas, meta)  # 绘制骨架
    cond_images.append(conditioning_image)

# 4. 输出
mpy.ImageSequenceClip(face_images, fps=fps).write_videofile('src_face.mp4')
mpy.ImageSequenceClip(cond_images, fps=fps).write_videofile('src_pose.mp4')
```

---

## 三、WanAnimate 推理时到底输入了什么？

### 3.1 五大条件通道

WanAnimate DiT (`WanAnimateModel`) 接收的条件，**全部是从原始视频中提取的结构化信号，没有任何原始视频像素**：

```
WanAnimateModel.forward(
    x,                    # 噪声 latent [16, T, H, W]
    t,                    # 时间步
    clip_fea,             # ① CLIP 特征 (来自 src_ref.png)
    context,              # ② T5 文本特征 (来自 prompt)
    seq_len,              # 序列长度
    y,                    # ③ VAE 参考图条件 (来自 src_ref.png)
    pose_latents,         # ④ 骨架 VAE latent (来自 src_pose.mp4)
    face_pixel_values,    # ⑤ 面部原始像素 (来自 src_face.mp4)
)
```

### 3.2 每个条件的来源和注入方式

| # | 条件 | 提取自 | 编码方式 | 注入位置 |
|---|------|--------|---------|---------|
| ① | CLIP 特征 | `src_ref.png` (参考角色图) | CLIP ViT-H/14 → 257 tokens | Cross-Attention 图像分支 (独立 k_img/v_img) |
| ② | T5 文本特征 | 用户 prompt | T5-XXL → [L, 4096] | Cross-Attention 文本分支 |
| ③ | VAE 参考图 | `src_ref.png` (参考角色图) | Wan2.1-VAE → [16, 1, H/8, W/8] | Channel Concat (与噪声拼接, 36ch 输入) |
| ④ | 骨架 latent | `src_pose.mp4` (骨架视频) | Wan2.1-VAE → [16, T/4, H/8, W/8] | Additive (pose_patch_embedding 后加到 x) |
| ⑤ | 面部像素 | `src_face.mp4` (面部视频) | motion_encoder (StyleGAN2) → 512维 | Face Adapter (每5层 Cross-Attention 注入) |

### 3.3 为什么不直接传入原始视频？

原始驱动视频**不应该也不需要**直接传入 DiT，原因如下：

| 原因 | 解释 |
|------|------|
| **任务本质** | WanAnimate 的任务是"将参考角色图的人物，按照驱动视频中人物的动作/表情来生成新视频"。它需要的是**动作信息**，不是驱动视频的像素 |
| **避免外观泄漏** | 如果原始视频像素进入模型，模型可能复制驱动视频中人物的外观（衣服、发型、肤色），而非使用参考角色图的外观 |
| **条件解耦** | 将动作分解为骨架(身体) + 面部(表情)两个独立信号，可以分别控制，也可以独立替换 |
| **姿态重定向** | 骨架信号可以在预处理阶段做 retarget（身体比例映射），原始像素无法做到这一点 |
| **模型架构** | DiT 的 in_dim=36 (16 noise + 20 condition)，condition 部分是 VAE 编码后的 latent，不接受原始 RGB 视频 |

---

## 四、src_face.mp4 与 EXAMPLE_PROMPT["video"] 的关系

### 4.1 关系链

```
EXAMPLE_PROMPT["video"] (占位符, 未使用)
    │
    │  概念上指向
    ▼
原始驱动视频 (video.mp4)
    │
    │  预处理脚本提取
    ├──────────────────────────────┐
    ▼                              ▼
src_pose.mp4 (骨架)          src_face.mp4 (面部)
    │                              │
    │  VAE.encode()                │  直接像素输入
    ▼                              ▼
pose_latents                 face_pixel_values
(conditioning_pixel_values)   [B, 3, T, 512, 512]
[1, 16, T/4, H/8, W/8]
    │                              │
    │  pose_patch_embedding        │  motion_encoder → face_encoder
    │  → 加到 x[:,:,1:]           │  → face_adapter (每5层)
    ▼                              ▼
    ────────── DiT 内部融合 ──────────
```

### 4.2 关键区别

| 属性 | "video" (原始视频) | src_face.mp4 (面部视频) | src_pose.mp4 (骨架视频) |
|------|-------------------|----------------------|----------------------|
| **内容** | 完整的真实人物动作视频 | 仅人脸区域裁剪 | 仅骨架线条渲染 |
| **分辨率** | 任意 | 固定 512×512 | 与生成目标分辨率相同 |
| **信息** | 全部 (外观+动作+场景) | 仅面部表情/嘴型/眼神 | 仅身体骨架关键点 |
| **是否进入 DiT** | ❌ 绝对不进入 | ✅ 作为 face_pixel_values | ✅ 经 VAE 编码为 pose_latents |
| **编码器** | 无 | StyleGAN2 motion_encoder | Wan VAE |
| **使用方式** | 仅在预处理阶段读取 | 运行时每帧送入 DiT | 运行时 VAE 编码后送入 DiT |

---

## 五、完整数据流图

```
用户输入
═══════════════════════════════════════════════════════════════════

  原始视频 (video.mp4)                    参考角色图 (image.jpeg)
  ┌──────────────┐                       ┌──────────────┐
  │ 真实人物      │                       │ 目标角色      │
  │ 跳舞/说话/   │                       │ 的照片        │
  │ 走路 等      │                       │              │
  └──────┬───────┘                       └──────┬───────┘
         │                                       │
         ▼                                       │
  ┌─────────────────────┐                        │
  │   预处理 Pipeline     │                        │
  │   (preprocess_data)  │                        │
  │                      │                        │
  │  YOLOv10 → 检测人体  │                        │
  │  ViTPose → 关键点    │                        │
  │  裁剪面部 → 512×512  │                        │
  │  渲染骨架 → 黑底白线  │                        │
  │  (可选) 姿态重定向    │                        │
  └──┬──────────┬────────┘                        │
     │          │                                 │
     ▼          ▼                                 ▼
═══════════════════════════════════════════════════════════════════
预处理产物 (src_root_path/)

  src_pose.mp4     src_face.mp4              src_ref.png
  ┌──────────┐    ┌──────────┐              ┌──────────┐
  │ 骨架视频  │    │ 面部视频  │              │ 参考图    │
  │ 黑底白线  │    │ 512×512  │              │ (复制)    │
  │ 逐帧骨架  │    │ 人脸裁剪  │              │          │
  └─────┬────┘    └─────┬────┘              └──┬───┬───┘
        │               │                      │   │
═══════════════════════════════════════════════════════════════════
WanAnimate 推理阶段 (animate.py)

        │               │                      │   │
        ▼               ▼                      │   │
   VAE.encode()    直接像素归一化              │   │
        │            /127.5 - 1               │   │
        ▼               │                      ▼   ▼
  pose_latents    face_pixel_values      VAE.encode  CLIP.visual
  [16,T/4,H/8,W/8]  [1,3,T,512,512]     ↓          ↓
        │               │              y_ref    clip_context
        │               │           [20,1,H/8,W/8] [257,1280]
        │               │               │          │
═══════════════════════════════════════════════════════════════════
WanAnimateModel.forward() 内部

        │               │               │          │
        │               │               ▼          │
        │               │         channel concat   │
        │               │         with noise (16ch) │
        │               │           ↓ total 36ch   │
        │               │         patch_embedding   │
        │               │           ↓ → 5120 dim   │
        │               │               │          │
        ▼               │               │          │
  pose_patch_embed      │               │          │
    Conv3d(16→5120)     │               │          │
        │               │               │          │
        ▼               │               │          │
   x[:,:,1:] += pose   │               │          │
        ┃               │               │          │
        ┃               ▼               │          ▼
        ┃         motion_encoder         │     img_emb(CLIP)
        ┃         (StyleGAN2)           │     → 257 tokens
        ┃            ↓ 512维             │          │
        ┃         face_encoder           │     prepend to
        ┃            ↓ [B,L,4,1280]     │     context
        ┃               │               │          │
        ┃               │               │          │
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                        │               │          │
                        ▼               ▼          ▼
              ┌─────────────────────────────────────────┐
              │        Transformer Block × 40            │
              │                                          │
              │  self_attn(x)                            │
              │  cross_attn(x, [CLIP_257 | T5_text])    │
              │  face_adapter(x, motion_vec) [每5层]     │
              │  ffn(x)                                  │
              └─────────────────┬────────────────────────┘
                                │
                                ▼
                          生成的视频帧
                       (参考角色 + 驱动动作)
```

---

## 六、常见误解澄清

### 误解 1: "video 字段是面部视频的路径"
**错误。** `EXAMPLE_PROMPT["animate-14B"]["video"]` 概念上指的是原始驱动视频（即你录的那段人跳舞/说话的原始视频），但这个字段在代码中**完全没有被使用**。面部视频 (`src_face.mp4`) 是预处理脚本从原始视频中提取出来的产物。

### 误解 2: "原始视频会被编码后送入 DiT"
**错误。** 原始视频仅在预处理阶段被读取，用于提取骨架和面部。在推理阶段，`animate.py` 只读取预处理后的 `src_pose.mp4` 和 `src_face.mp4`，原始视频不参与推理。

### 误解 3: "面部视频就是原始视频"
**错误。** 面部视频是从原始视频中**逐帧裁剪**出来的 512×512 人脸特写序列。它丢弃了人脸以外的所有信息（身体、背景、其他人物等）。

### 误解 4: "WanAnimate 可以从原始视频生成"
**半对。** WanAnimate 需要原始视频作为起点，但它不能跳过预处理直接使用原始视频。必须先运行 `preprocess_data.py` 将原始视频分解为条件素材。

---

## 七、对 MetaQuery Bridge 的影响

在 MetaQuery Bridge (`bridge_animate.py`) 中：

| 条件 | 状态 | 说明 |
|------|------|------|
| 原始视频 | **不涉及** | Bridge 也不接受原始视频 |
| src_ref.png → `ref_image` | **必选** | 用户传入 PIL.Image，替代文件路径 |
| src_pose.mp4 → `pose_latents` | **禁用** | 始终传零，用 MQ 语义替代 |
| src_face.mp4 → `face_source` | **可选** | 用户可传面部视频路径/帧列表，不传则全零 |
| MetaQuery 语义 | **新增** | 来自 Qwen3-VL，替代骨架的高层语义条件 |

这意味着 Bridge 进一步简化了输入要求：用户只需提供**一张参考图 + 文本 prompt**（可选面部视频），无需运行复杂的预处理流程来提取骨架。

---

*文档基于 `generate.py`, `wan/animate.py`, `wan/modules/animate/preprocess/process_pipepline.py`, `wan/modules/animate/preprocess/UserGuider.md`, `wan/modules/animate/model_animate.py` 的完整代码分析*
