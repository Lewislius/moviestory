# MetaQuery × Wan2.2 联合视频生成

> 用 **Qwen3-VL MetaQuery** 视觉条件增强 **Wan2.2** 文本到视频生成的完整技术文档

---

## 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [注入策略：Context 拼接](#3-注入策略context-拼接)
4. [代码结构](#4-代码结构)
5. [核心模块详解](#5-核心模块详解)
6. [快速开始](#6-快速开始)
7. [运行时 Print 验证输出](#7-运行时-print-验证输出)
8. [常见问题](#8-常见问题)

---

## 1. 概述

本项目将 **MetaQuery**（基于 Qwen3-VL 的视觉-语言 MetaQuery 模型）与 **Wan2.2**（阿里巴巴的扩散视频生成模型）结合：

- **MetaQuery** 从参考图像 + 文本 prompt 中提取 256 个视觉语义 token
- **Wan2.2** 的 WanModel (DiT) 通过 cross-attention 读取这 256 个 token（与 T5 文本特征联合作为条件）
- 两者**无需修改 WanModel 内部结构**，仅在 context 拼接层介入

---

## 2. 架构设计

### 2.1 Wan2.2 标准流程

```
Prompt ──► T5 Encoder ──► context [L_t5, 4096]
                                │
                     WanModel.cross_attention (K, V from context)
                                │
                    noise ──► UNet-DiT ──► 去噪 ──► VAE.decode ──► 视频
```

### 2.2 MetaQuery 增强流程

```
 Prompt + 参考图像
        │
        ▼
 Qwen3-VL (视觉编码)
        │
        ▼
 Connector (双向 Qwen2Encoder × 24 层)
        │  [B, 256, connector_out_dim]
        ▼
 to_wan_proj (Linear → GELU → Linear)
        │  [B, 256, 4096]
        ▼
 mq_context: List[Tensor[256, 4096]]



 Prompt ──────────────────────────────────────────────────────────────
        │
        ▼
 T5 Encoder ──► t5_context: List[Tensor[L_t5, 4096]]
                         │
                         ▼
              cat([mq_context, t5_context], dim=0)
                         │
                         ▼
          aug_context: List[Tensor[256+L_t5, 4096]]
                         │
              WanModel.text_embedding (text_len扩展至768)
                         │
                WanModel.cross_attention (K, V)
                         │
                noise ──► DiT ──► VAE.decode ──► 视频
```

### 2.3 WanModel 跨注意力机制

Wan2.2 的 `WanAttentionBlock` 使用 `WanCrossAttention`，其中 Key/Value 来自 `context`，且 `k_lens=context_lens=None`（无长度遮蔽），因此可自由扩展 context 长度而无需修改注意力代码。

---

## 3. 注入策略：Context 拼接

### 3.1 为何选择 Context 拼接？

| 策略 | 优点 | 缺点 |
|------|------|------|
| **Context 拼接** (本方案) | 无需修改 WanModel 结构；MetaQuery 特征直接参与所有层的 cross-attn | text_len 需动态扩展 |
| ControlNet 旁路 | 精细控制每层影响 | 需大幅修改模型 |
| Embedding 加法融合 | 简单 | MetaQuery 信息被稀释 |

### 3.2 注入点

```python
# wan/text2video.py: WanT2V.generate()
# 原始代码:
context = self.text_encoder([input_prompt], self.device)
arg_c = {'context': context, 'seq_len': seq_len}

# 注入后 (bridge.py):
t5_context = self.text_encoder([input_prompt], self.device)        # [L_t5, 4096]
mq_context = self.mq_encoder.encode([input_prompt], input_images)  # [256, 4096]
aug_context = [torch.cat([mq, t5], dim=0) for mq, t5 in ...]      # [256+L_t5, 4096]
arg_c = {'context': aug_context, 'seq_len': seq_len}
```

### 3.3 text_len 动态扩展

WanModel 内部对 context 做 padding 到 `text_len` (默认 512)：

```python
# WanModel.forward:
x = torch.stack([
    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
    for u in context
])  # [B, text_len, 4096]
```

Bridge 在生成前临时将 `text_len` 扩展至 **768**（512 + 256），生成后恢复：

```python
# bridge.py
self._aug_text_len = 512 + 256  # = 768
wan.low_noise_model.text_len  = 768   # 生成前
wan.high_noise_model.text_len = 768
# ... 去噪循环 ...
wan.low_noise_model.text_len  = 512   # 生成后恢复
wan.high_noise_model.text_len = 512
```

---

## 4. 代码结构

```
Wan2.2/
├── generate.py                          # 原始 CLI（不含 MetaQuery）
├── generate_with_metaquery.py           # ✨ MetaQuery 增强 CLI 入口
└── wan/
    ├── text2video.py                    # 原始 WanT2V pipeline
    ├── modules/
    │   ├── model.py                     # WanModel (DiT)
    │   └── t5.py                        # T5EncoderModel
    └── metaquery/                       # ✨ 新增集成模块
        ├── __init__.py                  # 导出接口
        ├── encoder.py                   # MetaQueryEncoder
        └── bridge.py                    # MetaQueryWanBridge
```

### 模块职责

| 文件 | 类 | 职责 |
|------|-----|------|
| `encoder.py` | `MetaQueryEncoder` | 加载 MetaQuery+Qwen3-VL；提取 MQ 视觉特征；投影到 4096 维 |
| `bridge.py` | `MetaQueryWanBridge` | 封装 WanT2V；替换 generate() 注入 MQ 条件 |
| `generate_with_metaquery.py` | — | CLI 入口；参数解析；保存输出视频 |

---

## 5. 核心模块详解

### 5.1 MetaQueryEncoder

```python
from wan.metaquery import MetaQueryEncoder

encoder = MetaQueryEncoder(
    metaquery_checkpoint_path="/path/to/metaquery_qwen3vl",
    num_metaqueries=256,   # 与训练配置一致
    wan_text_dim=4096,     # Wan2.2 T5 特征维度
    dtype=torch.bfloat16,
    device="cuda",
)

# 提取视觉特征
mq_features = encoder.encode(
    captions=["一只猫在阳光下打盹"],
    input_images=[[Image.open("ref.jpg")]]   # List[List[Image]] 或 None
)
# mq_features: List[Tensor[256, 4096]]  每批次一项
```

**内部流程：**

```
input_images + captions
        │
        ▼ tokenize (特殊 MetaQuery 标记嵌入)
input_ids, attention_mask, pixel_values, image_sizes
        │
        ▼ Qwen3-VL (视觉与文本联合编码)
hidden_states [..., seq_len, hidden_dim]
        │
        ▼ 提取 <img0>...<img255> 位置的隐藏状态
mq_hidden [B, 256, qwen_hidden_dim]
        │
        ▼ Connector (24层双向 Qwen2Encoder + Linear + GELU + Linear + RMSNorm)
connector_out [B, 256, connector_out_dim (~2240)]
        │
        ▼ to_wan_proj (Linear + GELU + Linear，Xavier 初始化)
mq_features [B, 256, 4096]
        │
        ▼ split by batch
List[Tensor[256, 4096]]
```

### 5.2 MetaQueryWanBridge

```python
from wan.metaquery import MetaQueryWanBridge
import wan
from wan.configs import WAN_CONFIGS

# 初始化原始 pipeline
wan_pipeline = wan.WanT2V(
    config=WAN_CONFIGS["t2v-A14B"],
    checkpoint_dir="/path/to/wan2.2",
    device_id=0,
)

# 创建 Bridge（注入 MetaQuery）
bridge = MetaQueryWanBridge(
    wan_pipeline=wan_pipeline,
    metaquery_checkpoint="/path/to/metaquery_qwen3vl",
    num_metaqueries=256,
    dtype=torch.bfloat16,
)

# 生成视频（与原始 WanT2V.generate() 接口兼容，额外支持 input_images）
video = bridge.generate(
    input_prompt="一只橘猫慵懒地躺在阳光洒落的窗台上",
    input_images=[Image.open("ref_cat.jpg")],   # 可选参考图
    size=(1280, 720),
    frame_num=81,
    sampling_steps=50,
    guide_scale=5.0,
    seed=42,
    offload_model=True,
)
# video: Tensor [C=3, F=81, H=720, W=1280]
```

**generate() 执行阶段：**

| 阶段 | 功能 |
|------|------|
| Step 1/4 | T5 编码文本 → `context [L_t5, 4096]` |
| Step 2/4 | Qwen3-VL MetaQuery 编码 → `mq_context [256, 4096]` |
| Step 3/4 | 拼接 → `aug_context [256+L_t5, 4096]`；扩展 `text_len → 768` |
| Step 4/4 | 标准去噪循环（CFG）；VAE 解码；恢复 `text_len → 512` |

---

## 6. 快速开始

### 6.1 环境准备

```bash
# 确保已安装 Wan2.2 依赖
pip install -r Wan2.2/requirements.txt

# 确保已安装 MetaQuery 依赖
pip install transformers accelerate Pillow tqdm
```

### 6.2 CLI 生成命令

#### 仅使用文本条件（无参考图）

```bash
cd Wan2.2

python generate_with_metaquery.py \
    --task t2v-A14B \
    --ckpt_dir /data/checkpoints/wan2.2 \
    --metaquery_ckpt /data/checkpoints/metaquery_qwen3vl \
    --prompt "夕阳下，一匹白马在草原上奔驰，镜头环绕跟拍" \
    --size 1280*720 \
    --frame_num 81 \
    --sampling_steps 50 \
    --guide_scale 5.0 \
    --seed 42 \
    --output_dir ./outputs
```

#### 使用参考图像作为视觉条件

```bash
python generate_with_metaquery.py \
    --task t2v-A14B \
    --ckpt_dir /data/checkpoints/wan2.2 \
    --metaquery_ckpt /data/checkpoints/metaquery_qwen3vl \
    --prompt "这个场景动起来，微风吹过，树叶摇曳" \
    --input_images ref_scene.jpg style_ref.jpg \
    --num_metaqueries 256 \
    --size 1280*720 \
    --frame_num 81 \
    --sampling_steps 50 \
    --guide_scale 5.0 \
    --seed 42 \
    --offload_model \
    --output_dir ./outputs
```

#### Python API 调用

```python
import torch
from PIL import Image
import wan
from wan.configs import WAN_CONFIGS
from wan.metaquery import MetaQueryWanBridge

# 初始化
wan_pipeline = wan.WanT2V(config=WAN_CONFIGS["t2v-A14B"],
                           checkpoint_dir="/path/to/wan2.2", device_id=0)
bridge = MetaQueryWanBridge(wan_pipeline, "/path/to/metaquery_qwen3vl")

# 生成
video = bridge.generate(
    input_prompt="一只猫在窗台上打盹",
    input_images=[Image.open("cat_ref.jpg")],
    size=(1280, 720), frame_num=81, seed=42,
)

# 保存
from wan.utils.utils import save_video
import numpy as np
save_video(
    video=video.permute(1,2,3,0).cpu().float().numpy(),
    input_file_name="output.mp4",
    fps=16, nrow=1,
)
```

---

## 7. 运行时 Print 验证输出

运行时控制台会打印以下内容，证明两模型成功结合：

```
============================================================
[MetaQueryWanBridge] 初始化 MetaQuery + Wan2.2 联合管线
  Wan pipeline 类型: WanT2V
  MetaQuery ckpt   : /path/to/metaquery_qwen3vl
  num_metaqueries  : 256
  mq_guidance_scale: 1.0
============================================================

[MetaQueryEncoder] 加载 MetaQuery (Qwen3-VL) 模型...
  checkpoint: /path/to/metaquery_qwen3vl
  ✅ 加载 MetaQuery 原始 checkpoint
[MetaQueryEncoder] 保留 mllm_model，丢弃 VAE/Diffusion 节省显存
[MetaQueryEncoder] connector_out_dim = 2240
[MetaQueryEncoder] 构建 to_wan_proj: 2240 → 4096
[MetaQueryEncoder] ✅ MetaQueryEncoder 初始化完成！

[MetaQueryWanBridge] WanModel text_len: 512 → 768 (+256 MetaQuery tokens)
[MetaQueryWanBridge] ✅ Bridge 初始化完成！

============================================================
[MetaQueryWanBridge.generate] 开始 MetaQuery 增强视频生成
  prompt        : 一只橘猫慵懒地躺在阳光洒落的窗台上...
  input_images  : 1 张参考图
  size          : (1280, 720)
  frame_num     : 81
  sampling_steps: 50
  guide_scale   : 5.0
============================================================

[Step 1/4] T5 编码文本条件...
  ✅ T5 context shape : torch.Size([47, 4096])  (L=47, dim=4096)

[Step 2/4] Qwen3-VL MetaQuery 编码视觉条件...
[MetaQueryEncoder.encode] 输入: 1 条 caption, 1 批参考图
[MetaQueryEncoder.encode] pixel_values shape: torch.Size([1, 3, 448, 448])
[MetaQueryEncoder.encode] MetaQuery 原始特征 shape: torch.Size([1, 256, 2240])
[MetaQueryEncoder.encode] 投影后 shape (→ Wan text_dim): torch.Size([1, 256, 4096])
  ✅ MetaQuery context shape     : torch.Size([256, 4096]) (num_mq=256, dim=4096)
  ✅ MetaQuery null context shape: torch.Size([256, 4096])

[Step 3/4] 拼接 T5 + MetaQuery context...
  ✅ 增强后 context shape: torch.Size([303, 4096]) (T5:47 + MQ:256 = 303 tokens)
  ✅ WanModel.text_len 扩展: 512 → 768

MetaQuery+Wan 去噪: 100%|██████████| 50/50 [04:23<00:00]
  [去噪 step 1] t=999.0 | noise_pred_cond shape: torch.Size([16, 21, 90, 160]) |
    context tokens: 303 (T5:47 + MQ:256) | ✅ MetaQuery + T5 双重条件正常运行

[MetaQueryWanBridge] VAE 解码...
[MetaQueryWanBridge] WanModel.text_len 恢复: 512
[MetaQueryWanBridge] ✅ MetaQuery 增强视频生成完成！
```

---

## 8. 常见问题

### Q1: `connector_out_dim` 不匹配报错？

`to_wan_proj` 在 `MetaQueryEncoder.__init__` 中根据 checkpoint 的 `connector_out_dim` 自动推断。如果报维度不匹配，请确认 MetaQuery checkpoint 是否完整加载：

```python
print(encoder.mq_model.connector_out_dim)  # 应为 2240 (Sana 配置)
```

### Q2: 显存不足 (OOM)？

```bash
# 启用模型卸载（将不活跃模型挂载至 CPU）
python generate_with_metaquery.py ... --offload_model --t5_cpu
```

也可降低分辨率：`--size 720*480` 或减少帧数 `--frame_num 49`。

### Q3: MetaQuery 无参考图时效果与原始相同？

无参考图时（`input_images=None`），`MetaQueryEncoder.encode` 仅用文本 caption 提取特征，无视觉语义注入，效果接近纯文本条件。建议提供参考图以充分发挥 MetaQuery 优势。

### Q4: 如何扩展 MetaQuery token 数量？

修改初始化时的 `num_metaqueries` 参数（需与训练配置一致），`text_len` 会自动扩展为 `512 + num_metaqueries`。

---

## 附录：关键维度速查

| 变量 | 值 | 含义 |
|------|----|------|
| `num_metaqueries` | 256 | MetaQuery token 数量 |
| `wan_text_dim` | 4096 | Wan2.2 text_dim（T5 输出维度） |
| `connector_out_dim` | ~2240 | MetaQuery Connector 输出维度（Sana 配置） |
| `wan.dim` | 5120 | WanModel 内部维度（A14B 配置） |
| `text_len` (原始) | 512 | WanModel context padding 长度 |
| `text_len` (增强) | 768 | 512 + 256 MetaQuery tokens |
| `max_t5_len` | 512 | T5 最大序列长度 |
