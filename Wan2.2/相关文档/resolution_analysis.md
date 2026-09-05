# MetaQuery + Wan2.2 数据集分辨率要求详解

> 本文档完整追踪参考图片/视频从输入到模型内部的每一步处理，明确回答"数据集应如何准备分辨率"。

---

## 一、核心结论（先看结论，后看推导）

**参考图片和视频在管线中被同时喂给两条路径：**
1. **Qwen3-VL (MetaQuery)** — 提取语义特征
2. **Wan 的原始条件通路** — 提取结构/外观特征（VAE、CLIP 等）

两条路径对分辨率的要求不同，但**代码中有自动 resize 机制**，所以最终结论是：

| 输入类型 | 是否需要固定分辨率？ | 实际约束 | 建议 |
|---------|-------------------|---------|------|
| **I2V 首帧图片** | **否** | 任意分辨率，代码自动 bicubic 缩放 | 短边 ≥ 480px，宽高比接近 16:9 或 9:16 最佳 |
| **Animate 参考人物图** | **否** | 代码自动对齐到 8 的倍数 | 短边 ≥ 512px |
| **Animate 面部视频** | **否（但有固定目标）** | 代码自动 resize 到 **512×512** | 建议直接提供 512×512 裁剪 |
| **MetaQuery 参考图** | **否** | Qwen3-VL AutoProcessor 动态处理 | 总像素 20万~100万 |
| **帧数** | **是** | 必须为 **4n+1**（如 81, 77, 121） | 严格遵守 |

---

## 二、参考图片在管线中的完整流转路径

### 2.1 I2V 管线（MetaQueryWanI2VBridge）

```
                          ┌────────────────────────────────────────────────┐
                          │                你输入的图片                      │
                          │           (任意分辨率, 如 1920×1080)            │
                          └──────────┬──────────────────┬─────────────────┘
                                     │                  │
                    ┌────────────────▼───────┐  ┌──────▼──────────────┐
                    │   路径 A: MetaQuery     │  │  路径 B: Wan I2V    │
                    │  (语义条件, context)     │  │ (结构条件, channel)  │
                    └────────────────────────┘  └─────────────────────┘
                              │                          │
                              ▼                          ▼
               ┌──────────────────────────┐  ┌─────────────────────────┐
               │ Qwen3-VL AutoProcessor   │  │ bicubic resize 到       │
               │ 自动缩放到 28 的倍数网格   │  │ max_area 计算的尺寸      │
               │ (如 ~896×504)            │  │ (如 1280×720)           │
               │ → ViT patch 化           │  │ → 归一化到 [-1,1]       │
               └──────────┬───────────────┘  └──────────┬──────────────┘
                          │                             │
                          ▼                             ▼
               ┌──────────────────────────┐  ┌─────────────────────────┐
               │ MLLM forward (Qwen3-VL)  │  │ Wan VAE encode          │
               │ → BOI-EOI 间提取         │  │ (首帧编码为 latent)      │
               │ → 256 个 MetaQuery token  │  │ → [16, T_lat, H_lat, W_lat] │
               │ → connector              │  │ + mask [4ch]             │
               │ → to_wan_proj            │  │ = y [20, T_lat, H_lat, W_lat] │
               └──────────┬───────────────┘  └──────────┬──────────────┘
                          │                             │
                          ▼                             ▼
               ┌──────────────────────────┐  ┌─────────────────────────┐
               │ [256, 4096] 特征         │  │ channel concat 到噪声     │
               │ 拼接到 T5 context 前面    │  │ 作为 y 条件               │
               │ → 增强 context            │  │                          │
               └──────────┬───────────────┘  └──────────┬──────────────┘
                          │                             │
                          └──────────┬──────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────────┐
                          │  WanModel.forward()           │
                          │  text_embedding(增强context)   │
                          │  + x(噪声+y)                  │
                          │  → 双重条件联合去噪             │
                          └──────────────────────────────┘
```

**关键代码（路径 B 的 resize）：**

```python
# bridge_i2v.py → _encode_first_frame()
# 与原始 WanI2V.generate() 完全一致的逻辑

img = TF.to_tensor(first_frame).sub_(0.5).div_(0.5)  # 原图直接加载
h, w = img.shape[1:]               # 原图的实际 HxW
aspect_ratio = h / w               # 保留宽高比

# 按 max_area 自适应计算 latent 尺寸
lat_h = round(sqrt(max_area * aspect_ratio) // vae_stride // patch_size * patch_size)
lat_w = round(sqrt(max_area / aspect_ratio) // vae_stride // patch_size * patch_size)
h = lat_h * vae_stride[1]  # 最终像素高度 (16 的倍数)
w = lat_w * vae_stride[2]  # 最终像素宽度 (16 的倍数)

# VAE 编码时做 bicubic resize
y = wan.vae.encode([
    torch.concat([
        F.interpolate(img[None].cpu(), size=(h, w), mode='bicubic'),  # ← 这里 resize
        torch.zeros(3, F - 1, h, w)
    ], dim=1).to(device)
])[0]
```

**结论：首帧图片的分辨率完全不受限制。** 代码会读取原图宽高比，计算出一个满足 `H*W ≈ max_area` 且 H/W 均为 16 倍数的目标尺寸，然后 bicubic resize。你传 4K 图也可以。

---

### 2.2 Animate 管线（MetaQueryWanAnimateBridge）

```
                          ┌────────────────────────────────────────────────┐
                          │               你输入的参考人物图                 │
                          │           (任意分辨率, 如 768×1024)             │
                          └──┬──────────────┬──────────────┬──────────────┘
                             │              │              │
               ┌─────────────▼─────┐  ┌────▼───────┐  ┌──▼─────────────────┐
               │  路径 A: MetaQuery │  │ 路径 B: VAE │  │  路径 C: CLIP      │
               │  (语义条件)        │  │ (结构条件)   │  │  (视觉全局条件)     │
               └───────────────────┘  └────────────┘  └────────────────────┘
                        │                   │                   │
                        ▼                   ▼                   ▼
               ┌────────────────┐  ┌───────────────┐  ┌────────────────────┐
               │ Qwen3-VL       │  │ 对齐到 8 倍数  │  │ bicubic resize 到  │
               │ AutoProcessor  │  │ h=(H//8)*8    │  │   224×224          │
               │ 自动处理分辨率  │  │ w=(W//8)*8    │  │  (CLIP.visual 内部) │
               │ → 256 token    │  │ padding_resize │  │  → 257 token       │
               └────────┬───────┘  │ 归一化 [-1,1]  │  └────────┬───────────┘
                        │          │ VAE encode     │           │
                        │          └───────┬───────┘           │
                        │                  │                    │
                        ▼                  ▼                    ▼
               ┌────────────────┐  ┌───────────────┐  ┌────────────────────┐
               │ context concat │  │ channel concat │  │ clip_fea → img_emb │
               │ T5 + MQ        │  │ mask + VAE lat │  │ → cross attention  │
               └────────┬───────┘  └───────┬───────┘  └────────┬───────────┘
                        │                  │                    │
                        └──────────┬───────┴────────────────────┘
                                   │
                          ┌────────▼──────────────────────────────┐
                          │ 还有一条路径 D: Face (可选)             │
                          │ face_source → resize 到 512×512       │
                          │ → motion_encoder → face_adapter       │
                          └───────────────────────────────────────┘
                                   │
                                   ▼
                          ┌──────────────────────────────┐
                          │  WanAnimateModel.forward()    │
                          │  四重条件联合去噪               │
                          └──────────────────────────────┘
```

**关键代码（路径 B 的分辨率对齐）：**

```python
# bridge_animate.py → generate()
ref_np = np.array(ref_image)
height = (ref_np.shape[0] // 8) * 8   # 向下取整到 8 的倍数
width  = (ref_np.shape[1] // 8) * 8   # 向下取整到 8 的倍数
if height == 0: height = 512
if width  == 0: width  = 512
ref_np = self._padding_resize(ref_np, height=height, width=width)
# _padding_resize: 等比缩放 + 黑边填充到 (height, width)
```

**关键代码（路径 C 的 CLIP 自动 resize）：**

```python
# wan/modules/animate/clip.py → CLIPModel.visual()
size = (self.model.image_size,) * 2  # (224, 224)
videos = torch.cat([
    F.interpolate(u.transpose(0,1), size=size, mode='bicubic', ...)
    for u in videos
])
# → 任何尺寸都会被自动 resize 到 224×224
```

**关键代码（路径 D 的面部 resize）：**

```python
# bridge_animate.py → _load_face_video()
for frame in face_frames:
    if frame.shape[0] != 512 or frame.shape[1] != 512:
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
# → 面部帧总是被 resize 到 512×512
```

**结论：Animate 的参考人物图分辨率也不受限制。** 代码会自动对齐到 8 倍数，然后 padding_resize；CLIP 和 Face encoder 各自有内置 resize。但注意：**参考图的宽高比直接决定了输出视频的宽高比**（不像 I2V 那样有 max_area 自适应）。

---

## 三、每条路径对分辨率的详细要求

### 3.1 路径 A: Qwen3-VL (MetaQuery) → 完全不限分辨率

| 项目 | 说明 |
|------|------|
| 输入接口 | `encoder.encode(captions, input_images)` 接收 `List[PIL.Image]` |
| 分辨率处理 | Qwen3-VL 的 `AutoProcessor` 自动完成：按 `min_pixels` / `max_pixels` 缩放到最近的有效网格 |
| 推理配置 | `min_pixels = 256 × 28 × 28 = 200,704`, `max_pixels = 1280 × 28 × 28 = 1,003,520` |
| 训练配置 | `min_pixels = 256 × 28 × 28`, `max_pixels = 768 × 28 × 28 = 602,112` |
| 输出 | 固定 `[256, 4096]`，**与输入分辨率完全无关** |
| 约束 | **无** — 任意分辨率的图片都可以 |

### 3.2 路径 B: Wan VAE encode → 有对齐约束，但代码自动处理

**I2V 管线：**

| 项目 | 说明 |
|------|------|
| 输入 | 原图直接加载为 tensor |
| resize 策略 | `F.interpolate(img, size=(h, w), mode='bicubic')` |
| 目标尺寸计算 | `h, w = f(aspect_ratio, max_area)`，自动对齐到 **16 的倍数** |
| 默认 max_area | `720 × 1280 = 921,600` |
| 输出 | `y [20, T_lat, H_lat, W_lat]` — 尺寸由计算公式决定 |
| 约束 | **无** — 代码自动计算并 resize |

**Animate 管线：**

| 项目 | 说明 |
|------|------|
| 输入 | `np.array(ref_image)` |
| resize 策略 | `_padding_resize()` 等比缩放 + 黑边填充 |
| 目标尺寸 | `(H//8*8, W//8*8)`，即原图对齐到 **8 的倍数** |
| 输出 | `y_ref [20, 1+1, lat_h, lat_w]` |
| 约束 | **自动对齐**，但注意参考图宽高比 = 输出视频宽高比 |

### 3.3 路径 C: CLIP ViT-H/14 → 固定 224×224，自动处理

| 项目 | 说明 |
|------|------|
| 仅用于 | Animate 管线 |
| 内部 resize | `F.interpolate(..., size=(224, 224), mode='bicubic')` |
| 输出 | `[257, embed_dim]`（256 patch + 1 cls） |
| 约束 | **无** — 自动 resize |

### 3.4 路径 D: Face encoder → 固定 512×512，自动处理

| 项目 | 说明 |
|------|------|
| 仅用于 | Animate 管线 |
| 内部 resize | `cv2.resize(frame, (512, 512))` |
| 输出 | motion 特征 |
| 约束 | **无** — 自动 resize。但如果原帧远小于 512，上采样会导致模糊 |

---

## 四、关于 `SUPPORTED_SIZES` 的误解澄清

### 什么是 `SUPPORTED_SIZES`？

它是 **官方 CLI 脚本 `generate.py` 的命令行参数校验**，定义在 [`wan/configs/__init__.py`](wan/configs/__init__.py#L43)：

```python
SUPPORTED_SIZES = {
    't2v-A14B':    ('720*1280', '1280*720', '480*832', '832*480'),
    'i2v-A14B':    ('720*1280', '1280*720', '480*832', '832*480'),
    'ti2v-5B':     ('704*1280', '1280*704'),
    's2v-14B':     (全部 8 种),
    'animate-14B': ('720*1280', '1280*720'),
}
```

校验代码在 [`generate.py` 第 100 行](generate.py#L100)：
```python
assert args.size in SUPPORTED_SIZES[task], "Unsupport size..."
```

### 它约束了什么？

**仅约束 `generate.py` CLI 的 `--size` 参数**（即用户指定的输出视频尺寸）。

### 它 **不** 约束什么？

1. ❌ 不约束输入参考图片的分辨率
2. ❌ 不约束 Bridge 代码中的任何操作
3. ❌ 不约束 Wan 模型本身的能力（模型能处理任何 16 倍数对齐的尺寸）

### "Bridge 绕过了它"的含义

Bridge 代码（`bridge_i2v.py`、`bridge_animate.py`）是我们自己写的调用入口，**直接调用 `wan.vae.encode()` 和 `wan.noise_model()`**，不经过 `generate.py`，所以 `SUPPORTED_SIZES` 的 assert 根本不会被执行。

但这**不意味着可以用任何尺寸**。Wan VAE 的 Conv3d stride=2 三层堆叠决定了 H/W **必须被 8 整除**，patch_size=(1,2,2) 进一步要求 latent H/W 必须偶数，综合约束为**像素 H/W 必须被 16 整除**。

Bridge 代码通过公式（I2V 的 `max_area` 计算）或简单取整（Animate 的 `//8*8`）自动满足这些约束。

---

## 五、你的图片到底被输入给了谁？

### ❌ 错误理解
> "参考图片只经过 Qwen3-VL 和 MetaQuery 处理，不直接输入给 Wan"

### ✅ 正确理解

**同一张图片会被 同时 送入两条甚至三条路径：**

| 管线 | 路径 A (MetaQuery) | 路径 B (Wan VAE) | 路径 C (CLIP) | 路径 D (Face) |
|------|:-:|:-:|:-:|:-:|
| **I2V** | ✅ 首帧 → Qwen3-VL | ✅ 首帧 → VAE encode → y | — | — |
| **Animate** | ✅ 参考图 → Qwen3-VL | ✅ 参考图 → VAE encode → y_ref | ✅ 参考图 → CLIP | ✅ 面部帧 → face_encoder |
| **T2V** | ✅ 参考图 → Qwen3-VL（可选） | — | — | — |

**每条路径各自独立处理分辨率：**
- 路径 A：Qwen3-VL `AutoProcessor` 自动缩放
- 路径 B：`F.interpolate` bicubic 或 `padding_resize` 对齐
- 路径 C：CLIP 内部 `F.interpolate` 到 224×224
- 路径 D：`cv2.resize` 到 512×512

所以你**不需要**预先将图片 resize 到某个特定分辨率——每条路径都会做自己需要的 resize。

---

## 六、数据集准备实操建议

### 6.1 I2V 训练/测试数据

```
推荐做法:
├── 首帧图片: 保持原始分辨率
│   ├── 建议: 短边 ≥ 480px (太小会导致 VAE encoder 输入过小)
│   ├── 建议: 宽高比接近 16:9、9:16、1:1 等常见比例
│   └── 不需要: resize 到固定尺寸
│
├── MetaQuery 参考图 (mq_reference_images):
│   ├── 默认: 如果不额外指定，代码自动用首帧图
│   ├── 如果额外指定: 任意分辨率，总像素 20万~100万
│   └── 不需要: resize 到固定尺寸
│
└── 视频标注: 帧数必须是 4n+1 (如 81, 121, 161)
```

### 6.2 Animate 训练/测试数据

```
推荐做法:
├── 参考人物图 (ref_image):
│   ├── 建议: 短边 ≥ 512px
│   ├── 注意: 宽高比 = 输出视频宽高比 (代码取原图尺寸对齐到 8 倍数)
│   ├── 建议: 使用 720×1280 或 1280×720 (与 SUPPORTED_SIZES 一致)
│   └── 不需要: 精确到像素级固定
│
├── 面部视频 (face_source):
│   ├── 最佳: 裁剪面部区域后 resize 到 512×512
│   ├── 如果不是 512×512: 代码自动 cv2.resize, 但可能有插值损失
│   └── 帧数: 不需要固定, 代码自动循环填充到 frame_num
│
├── MetaQuery 参考图:
│   ├── 默认: 自动使用 ref_image
│   └── 额外指定: 任意分辨率
│
└── 帧数: 必须是 4n+1
```

### 6.3 汇总：各路径的分辨率容忍度

| 路径 | 最小分辨率 | 最大分辨率 | 最佳分辨率 | 是否需要固定？ |
|------|-----------|-----------|-----------|:----------:|
| Qwen3-VL | ~450×450 (≥200K px) | ~1000×1000 (≤1M px) | 原图即可 | ❌ |
| Wan VAE (I2V) | ≥64×64 (理论) | 无上限 (显存限制) | 720×1280 区域 | ❌ |
| Wan VAE (Animate) | ≥128×128 | 无上限 | 720×1280 / 1280×720 | ❌ |
| CLIP ViT-H/14 | 任意 | 任意 | 224×224 (内部自动) | ❌ |
| Face encoder | 任意 | 任意 | **512×512** | ⚠️ 建议 |

---

## 七、一句话总结

> **你的数据集不需要固定分辨率。** 每个组件都有内置的 resize 机制。你只需确保：
> 1. 图片不要太小（短边 ≥ 480px）
> 2. 面部帧建议 512×512
> 3. 帧数严格 4n+1
> 4. 如果在意生成质量，参考图宽高比尽量接近 16:9 或 9:16（因为模型在这些比例上训练最充分）
