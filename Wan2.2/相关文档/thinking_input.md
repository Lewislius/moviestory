# Thinking Input: 面部视频、额外语义图与原始 WanAnimate 对比分析

> 本文档详细解释 WanAnimate Bridge 中 **面部视频 (face_pixel_values)** 和 **额外语义图 (mq_reference_images)** 的含义、作用、具体示例，并与原始 WanAnimate 实现进行逐项对比。

---

## 一、面部视频 (Face Video / face_pixel_values)

### 1.1 是什么？

面部视频是一段 **裁剪为 512×512 分辨率的人脸特写视频**，包含了目标人物的面部表情序列。在原始 WanAnimate 中，它来自 `src_root_path/src_face.mp4`。

**典型内容**: 紧密裁剪的人脸画面，保留完整的面部区域（额头到下颌、左耳到右耳），人脸居中占据画面主体。

### 1.2 作用是什么？

面部视频通过一套专门的**运动编码器 → 面部编码器 → 面部适配器**三级注入管线，将人脸表情动作迁移到生成的角色视频中。

#### 详细处理流程

```
face_pixel_values [B, 3, T, 512, 512]
    │
    ▼ rearrange → [(B*T), 3, 512, 512]
    │
    ▼ motion_encoder.get_motion()     ← StyleGAN2 Generator (size=512, style_dim=512, motion_dim=20)
    │                                    提取每帧的面部运动特征向量
    │
    ▼ motion_vec [(B*T), 512]
    │
    ▼ rearrange → [B, T, 512]
    │
    ▼ face_encoder (FaceEncoder)       ← in_dim=512, hidden_dim=5120, num_heads=4
    │                                    投影到 DiT 维度并做多头分组
    │
    ▼ motion_vec [B, L, H=4, C=1280]
    │
    ▼ 前补一帧零向量 (对齐参考帧位置)
    │  → [B, L+1, H, C]
    │
    ▼ face_adapter.fuser_blocks[i]     ← 每隔 5 层 DiT 注入一次 (block 0,5,10,15,20,25,30,35,40)
    │                                    共 9 个 fuser block
    │                                    通过 cross-attention 将 motion_vec 融合到 x
    │
    ▼ residual_out + x                 ← 残差连接
```

#### 关键细节
- **运动编码器 (motion_encoder)**: 基于 StyleGAN2 的 Generator 架构，输入 512×512 人脸图像，输出 512 维运动特征向量，编码了表情、嘴型、眼神等细粒度面部运动信息
- **面部编码器 (face_encoder)**: 将 512 维运动向量投影到 5120 维 (DiT 主维度)，并分成 4 个注意力头
- **面部适配器 (face_adapter)**: 每 5 层 DiT 块执行一次 cross-attention 注入，共 9 次 (40层 ÷ 5 + 第0层)。渐进式多尺度注入确保面部控制覆盖从粗到细的特征层

### 1.3 具体示例

| 场景 | 面部视频内容 | 生成效果 |
|------|-------------|---------|
| 说话驱动 | 录制一段人对着镜头说话的视频，裁剪出 512×512 人脸区域 | 生成的角色会模仿说话者的嘴型变化和面部肌肉运动 |
| 表情迁移 | 一段包含微笑→惊讶→皱眉的表情变化视频 | 参考角色图中的人物按照相同时序展现对应表情 |
| 眼神控制 | 面部视频中人物眼睛向左看→向右看→眨眼 | 生成角色的眼神方向和眨眼节奏跟随面部视频 |
| 不使用面部 | `face_source=None` (Bridge 中) | 面部特征全为 -1.0，面部适配器无效，角色表情由文本/语义条件控制 |

### 1.4 在原始 WanAnimate 中的实现

**原始实现** (`animate.py` > `prepare_source()` + `generate()`):
```python
# prepare_source() 加载面部视频 —— 必选，不可省略
face_video_reader = VideoReader(src_face_path)     # 读取 src_face.mp4
face_images = face_video_reader.get_batch(...).asnumpy()

# generate() 中构建面部张量
batch["face_pixel_values"] = rearrange(
    torch.tensor(np.stack(face_images[start:end]) / 127.5 - 1),
    "t h w c -> 1 c t h w",
)

# 在 arg_c 中传递给 noise_model
arg_c = {
    ...
    "face_pixel_values": face_pixel_values,   # 真实面部帧
}

# CFG 无条件分支：面部归零再减1
face_pixel_values_uncond = face_pixel_values * 0 - 1
```

**在原始代码中，面部视频是必须提供的，没有可选或跳过的机制。**

---

## 二、额外语义图 (MQ Reference Images / mq_reference_images)

### 2.1 是什么？

额外语义图是传入 **Qwen3-VL MetaQuery 编码器** 的参考图像列表。这是 **MetaQuery 框架新增的功能，原始 WanAnimate 中完全不存在**。

它利用 Qwen3-VL 视觉语言大模型的图像理解能力，从参考图中提取高层语义特征（256 个可学习的查询 token），作为额外的语义条件注入到扩散过程中。

### 2.2 作用是什么？

MQ 参考图的作用是提供 **超越 CLIP 257 token 的深层语义理解**：

| 特征维度 | CLIP (原始) | MetaQuery (新增) |
|----------|-----------|-----------------|
| 模型 | CLIP ViT-H/14 | Qwen3-VL (72B 级 MLLM) |
| Token 数 | 257 (1 cls + 256 patch) | 256 (可学习查询) |
| 理解层次 | 视觉相似性、风格、纹理 | 语义理解：物体关系、场景构图、角色身份、动作意图 |
| 注入位置 | Cross-Attention 专用分支 (context_img) | Cross-Attention 文本分支 (与 T5 共享) |
| 可跨模态 | 纯视觉 | 视觉 + 文本联合理解 (prompt 也参与编码) |

#### 详细处理流程

```
mq_reference_images (List[PIL.Image])    ← 用户指定，若未指定则用 ref_image
    │
    ▼ Qwen3-VL AutoProcessor              ← 动态分辨率处理 (min 200K ~ max 1M 像素)
    │   + prompt + <|vision_start|>BOI/EOI<|vision_end|>
    │
    ▼ Qwen3-VL forward (冻结)
    │   提取 hidden_states 中 BOI~EOI 区间的 token
    │
    ▼ MetaQuery Cross-Attention
    │   256 个可学习查询 token × hidden_states → 256 个语义向量
    │
    ▼ MQ Projector (Linear 3584→4096)     ← 对齐到 T5 text_dim=4096
    │
    ▼ mq_context [256, 4096]
    │
    ▼ _augment_context(): cat([MQ, T5], dim=0)
    │   → aug_context [256+L_t5, 4096]
    │
    ▼ WanAnimateModel.forward():
    │   text_embedding(pad(aug_context, 768))  ← text_len 从 512 扩展到 768
    │   → context_embedded [768, 5120]
    │
    ▼ cat([CLIP_257, context_embedded])
    │   → final_context [257+768, 5120] = [1025, 5120]
    │
    ▼ WanAnimateCrossAttention:
    │   context_img = final_context[:, :257]    → CLIP 专用分支 (k_img, v_img)
    │   context_text = final_context[:, 257:]   → 文本分支 (MQ_256 + T5_padded)
```

### 2.3 具体示例

| 场景 | mq_reference_images 内容 | 语义作用 |
|------|------------------------|---------|
| **默认** (未指定) | 自动使用 ref_image (参考人物图) | MQ 深度理解角色外观、服饰、发型、体型等 |
| **风格参考** | 一张特定风格的画作 (如油画风格) | MQ 捕获画面风格、色彩调性、笔触质感 |
| **场景参考** | 一张目标场景的照片 (如咖啡店内景) | MQ 理解空间布局、光照环境、物体摆设 |
| **多图参考** | [角色正面照, 角色侧面照, 场景照] | MQ 综合理解角色完整外观 + 目标环境 |
| **动作参考** | 一张展示目标动作的图片 (如跳舞姿态) | MQ 理解目标动作的语义意图 (替代骨架的高层方案) |

### 2.4 与 CLIP 的区别

CLIP 和 MQ 的注入路径**完全不同**，互不干扰：

```
WanAnimateCrossAttention.forward():
    │
    ├─ context_img = context[:, :257]         # CLIP tokens
    │   → k_img, v_img = k_img_proj(), v_img_proj()
    │   → img_x = flash_attention(q, k_img, v_img)
    │
    └─ context_text = context[:, 257:]        # MQ (256) + T5 (padded)
        → k, v = k_proj(), v_proj()
        → text_x = flash_attention(q, k, v)
    
    output = text_x + img_x                  # 两路相加
```

- **CLIP**: 捕获视觉特征 → 独立的 `k_img/v_img` 投影 → 独立注意力计算
- **MQ + T5**: 共享文本投影 `k/v` → 联合注意力计算。MQ token 和 T5 token 在注意力中可以互相交互

### 2.5 在原始 WanAnimate 中的实现

**原始 WanAnimate 中没有 mq_reference_images 相关的任何实现。** 这是 MetaQuery 桥接层的全新功能。原始模型只有 CLIP + T5 两种语义条件。

---

## 三、原始 WanAnimate vs Bridge 完整对比

### 3.1 条件通道逐项对比

| 条件通道 | 原始 WanAnimate | Bridge (MetaQuery) | 是否一致 |
|---------|----------------|-------------------|---------|
| **参考图 → VAE** | `src_ref.png` → VAE encode → `y_ref` (channel concat) | `ref_image` → VAE encode → `y_ref` (channel concat) | ✅ 完全一致 |
| **参考图 → CLIP** | `src_ref.png` → CLIP.visual → 257 tokens → `clip_context` | `ref_image` → CLIP.visual → 257 tokens → `clip_context` | ✅ 完全一致 |
| **文本 → T5** | prompt → T5 encoder → context [L, 4096] | prompt → T5 encoder → context [L, 4096] | ✅ 完全一致 |
| **骨架 → pose_latents** | `src_pose.mp4` → VAE.encode → 真实 pose latents | `torch.zeros(...)` (全零) | ⚡ 有意禁用 |
| **面部 → face_pixel_values** | `src_face.mp4` → 必选，始终加载 | `face_source` → 可选 (None→全零=-1) | ⚡ 有意放宽 |
| **MetaQuery 语义** | ❌ 不存在 | Qwen3-VL → 256 tokens → 拼接到 context | 🆕 全新增加 |
| **text_len** | 512 (固定) | 512 → 768 (运行时扩展，结束后恢复) | 🆕 适配 MQ |
| **CFG 面部无条件** | `face * 0 - 1` | `face * 0 - 1` | ✅ 完全一致 |
| **时序引导 (y_reft)** | 前一 clip 末帧 → VAE encode → concat | 前一 clip 末帧 → VAE encode → concat | ✅ 完全一致 |
| **噪声初始化** | `torch.randn(16, lat_t+1, lat_h, lat_w)` | `torch.randn(16, lat_t+1, lat_h, lat_w)` | ✅ 完全一致 |
| **多 clip 滑窗** | `start += clip_len - refert_num` | `start += clip_len - refert_num` | ✅ 完全一致 |

### 3.2 模型内部条件注入对比

```
WanAnimateModel.forward() 内部注入顺序:
┌─────────────────────────────────────────────────────────────────┐
│  1. Channel Concat:  x = cat([noise(16ch), y_ref+y_reft(20ch)])│  ← 两者一致
│     → patch_embedding(36→5120)                                   │
│                                                                  │
│  2. Pose Additive:   x[:,:,1:] += pose_patch_embedding(pose)   │  ← 原始: 真实 pose
│                                                                  │     Bridge: zeros (无效果)
│                                                                  │
│  3. Face Encode:     face → motion_encoder → face_encoder       │  ← 原始: 真实面部帧
│     → motion_vec [B, L+1, H, C]                                 │     Bridge: 可选 (None→zeros)
│                                                                  │
│  4. CLIP Prepend:    context = cat([CLIP_257, text_embedded])   │  ← 两者一致
│                                                                  │
│  5. Transformer Blocks ×40:                                      │
│     ├─ self_attn(x)                                              │  ← 两者一致
│     ├─ cross_attn(x, context)                                    │  ← 原始: CLIP+T5
│     │   ├─ context_img[:257] → CLIP 分支                         │     Bridge: CLIP+(MQ+T5)
│     │   └─ context[257:] → 文本分支                               │     MQ在文本分支中
│     ├─ face_adapter(x, motion_vec) [每5层]                       │  ← 两者一致 (Bridge可为零)
│     └─ ffn(x)                                                    │  ← 两者一致
│                                                                  │
│  6. Head:            unpatchify → output                         │  ← 两者一致
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Context 数据流对比

#### 原始 WanAnimate:
```
T5 context [L_t5, 4096]
    ↓ pad to [512, 4096]
    ↓ text_embedding (Linear 4096→5120, GELU, Linear 5120→5120)
    → context_embedded [512, 5120]
    ↓ cat with CLIP
    → [257 + 512, 5120] = [769, 5120]
    ↓ CrossAttention split
    → context_img [257, 5120]  +  context_text [512, 5120]
```

#### Bridge (MetaQuery):
```
MQ context [256, 4096]  +  T5 context [L_t5, 4096]
    ↓ cat → aug_context [256 + L_t5, 4096]
    ↓ pad to [768, 4096]              ← text_len 临时扩展到 768
    ↓ text_embedding (同一个 MLP)      ← MQ tokens 也经过 T5 的 text_embedding
    → context_embedded [768, 5120]
    ↓ cat with CLIP
    → [257 + 768, 5120] = [1025, 5120]
    ↓ CrossAttention split
    → context_img [257, 5120]  +  context_text [768, 5120]
                                     ↑ 包含 MQ(256) + T5(L_t5) + padding
```

**关键观察**: MQ token 经过了与 T5 相同的 `text_embedding` MLP。虽然 MQ token 来源于 Qwen3-VL (与 T5 不同的分布)，但维度相同 (4096)，且 MetaQuery 的投影层已将其对齐到 T5 的特征空间。这是 MetaQuery 框架的设计意图。

### 3.4 一致性总结

| 分类 | 结论 |
|------|------|
| **完全一致的部分** | VAE 参考图编码、CLIP 编码、T5 文本编码、CFG 构建、时序引导、噪声初始化、多 clip 滑窗、Solver、VAE 解码 |
| **有意修改的部分** | 骨架禁用 (zeros)、面部改为可选 (None→zeros)、新增 MQ 语义条件、text_len 扩展 |
| **需要优化的部分** | **无** — Bridge 在保留原始逻辑完整性的基础上进行了合理的功能扩展 |

---

## 四、为什么骨架被禁用而面部被保留？

### 4.1 设计理由

| 条件 | 原始用途 | MetaQuery 中的命运 | 原因 |
|------|---------|-------------------|------|
| **骨架 (pose)** | 逐帧骨架关键点视频驱动身体动作 | **禁用** (zeros) | MetaQuery 用高层语义 (文本+图像理解) 替代显式骨架，降低数据准备门槛 |
| **面部 (face)** | 面部表情/嘴型驱动 | **保留但可选** | 面部微表情/嘴型是细粒度控制，语义条件难以精确替代 |

### 4.2 zeros 的效果

- **pose_latents = zeros**: `after_patch_embedding()` 中 `x[:,:,1:] += pose_patch_embedding(zeros)`。由于 Conv3d 有 bias，输出不完全为零，但是一个恒定偏置，不携带任何帧间变化信息 → 模型退化为"无骨架引导"模式
- **face_pixel_values = zeros → 归一化后 = -1**: `motion_encoder.get_motion(all_minus_one)` 输出的 motion_vec 是一个恒定向量。这与 CFG 无条件分支 (`face*0-1=-1`) 完全一致 → `guide_scale` CFG 下，条件和无条件的面部贡献完全抵消

---

## 五、结论

### 5.1 Bridge 实现是否与原始一致？

**一致**，在所有共享的条件通道上，Bridge 忠实复现了原始 WanAnimate 的实现逻辑：
- VAE/CLIP/T5 编码路径 ✅
- 模型前向传播的参数传递 ✅
- CFG 构建 (条件/无条件) ✅
- 多 clip 滑窗去噪 ✅

### 5.2 是否需要优化？

**不需要代码修改**。当前的差异都是有意为之的设计决策：
1. 骨架禁用 → MetaQuery 的设计目标就是用语义替代显式控制信号
2. 面部可选 → 灵活性提升，不破坏原始逻辑
3. MQ 新增 → 核心创新点，通过 text_len 扩展和 context 拼接无缝接入

### 5.3 潜在注意事项

1. **text_embedding 共享**: MQ token 和 T5 token 共享同一个 `text_embedding` MLP。如果微调时只更新 MetaQuery 的投影层而冻结 `text_embedding`，需确保 MQ token 在 T5 特征空间中的分布合理
2. **pose_patch_embedding bias**: 即使输入全零，`Conv3d` 的 bias 会产生恒定偏置。如果完全不希望 pose 通道有任何影响，可以考虑将 `pose_patch_embedding.bias` 也置零，但当前行为已经不影响生成质量
3. **face_encoder 恒定输入**: 当面部为全 -1 时，`motion_encoder` 仍然会执行前向传播，产生计算开销。如果确定永远不使用面部条件，可以考虑短路跳过

---

*文档生成时间: 基于 wan/animate.py (649行), wan/modules/animate/model_animate.py (501行), wan/metaquery/bridge_animate.py (800行) 的完整代码分析*
