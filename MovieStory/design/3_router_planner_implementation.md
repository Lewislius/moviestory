# 3-router planner 实现文档

> **2026-07-31 配置更新：** 本文后续保留了早期 144-token 方案的历史说明。
> 当前代码使用 256 token（96 role + 96 action + 64 global）、24 层 Connector、
> 三张独立 FP32 路由表与冻结的 Qwen 原始 embedding；首帧使用 timestep-0
> clean preserved reference slot，条件前缀不参与 loss。总损失只包含生成视频
> 与 ground-truth 视频的 velocity MSE。请以 `code/README.md`、启动脚本和测试
> 为当前事实来源。

## 1. 交付结构

```text
Project/MovieStory/
├── design/
│   ├── 3_router_planner_design.md
│   └── 3_router_planner_implementation.md
└── code/
    ├── three_router_planner/
    │   ├── config.py
    │   ├── planner.py
    │   ├── qwen_wan_adapter.py
    │   └── wan_first_frame.py
    ├── scripts/prepare_openvid100.py
    ├── tests/test_three_router_planner.py
    ├── train_3router_planner_wan.py
    ├── train_openvid100_3router.sh
    └── checkpoint/
```

## 2. 模块接口

### 2.1 `ThreeRouterConfig`

配置固定 route layout，并提供唯一的切片定义：

```python
config.route_slices == {
    "role": (0, 64),
    "action": (64, 112),
    "global": (112, 144),
}
```

训练入口默认 `hidden_size=2048`，因为本地
`Qwen3-VL-2B-Thinking/config.json` 的实际 text hidden size 为 2048。
纯模块类保留 1536 的设计默认值，便于在其他符合总设计的 backbone 上复用；
Qwen 适配器加载时会检查实际宽度，禁止静默错配。

### 2.2 `ThreeRouterPlanner`

输入：

```text
seed_tokens:    [B, 144, d_qwen]
guidance_scale: scalar | [B] | [B,1]，范围 [0,1]
```

计算：

```python
route_type = type_embedding[route_ids]           # [144,d]
gs_delta = GuidanceEncoder(gs)                   # [B,d]
planned = seed + route_type[None] + gs_delta[:,None]
```

输出 `ThreeRouterOutput`：

- `tokens [B,144,d]`；
- `role [B,64,d]`；
- `action [B,48,d]`；
- `global_route [B,32,d]`；
- `guidance_scale [B,1]`。

`diagnostics()` 计算三组 pooled cosine、route RMS 和 gs，调用方可写入日志，但
不会参与训练 loss。

### 2.3 `ThreeRouterMetaQueryEncoderForWan`

`build_three_router_encoder_class` 动态生成现有
`train_connector_for_wan.MetaQueryEncoderForWan` 的 drop-in 子类：

1. 复用基类加载 Qwen3-VL、processor 和 Qwen2 Connector；
2. 复用 `MLLMInContext.tokenize` 构造多模态输入；
3. 直接运行 Qwen backbone；
4. 按 BOI/EOI 逐样本提取恰好 144 个 raw hidden states；
5. 训练时采样 `gs~Beta(2,2)`，推理默认 `gs=1`；
6. 可选运行 planner；
7. 调用原有 Connector 得到 `[B,144,4096]`。

逐样本提取而不是用全 batch 的扁平 `view`，可以避免左 padding 或多模态序列长度
差异造成 route token 跨样本错位。

## 3. 与基础代码的接入方式

训练入口不会复制或修改 Wan/Qwen 源码。它按以下顺序接入：

```python
import train_connector_for_wan as connector_module
import train_metaquery_wan as base_train

PatchedEncoder = build_three_router_encoder_class(
    connector_module.MetaQueryEncoderForWan,
    config,
)
connector_module.MetaQueryEncoderForWan = PatchedEncoder
trainer = ThreeRouterWanTrainer(base_train.parse_args())
trainer.train()
```

现有 `MetaQueryWanTrainer._load_models` 在运行时从
`train_connector_for_wan` 取 Encoder，因此补丁会被正常使用。其余能力直接复用：

- `WanTI2V` checkpoint 加载；
- VAE 视频 latent 编码；
- MQ-only context 注入；
- flow-matching loss；
- optimizer、scheduler、显存日志；
- checkpoint-before-training / checkpoint-N / checkpoint-final。

checkpoint 的 `mq_encoder_full.pt` 和 safetensors 会自然包含：

```text
router_planner.route_type_embeddings
router_planner.route_ids
router_planner.guidance_encoder.net.*
```

训练器额外在每个 checkpoint 目录写入 `three_router_config.json`，避免推理时只能
从 tensor shape 猜 route layout。

### 3.1 Wan 端参考图首帧强绑定

训练入口默认启用 `--wan_first_frame_strong_bind`，只在 MovieStory 适配层中实现，
不修改共享的 Wan2.2 训练源码。每个 batch 的处理顺序如下：

1. 从不参与 MQ 图像随机置空的 `batch["ref_image"]` 取直接参考图；
2. 用 Wan VAE 单独编码得到 `z_ref [C,1,H,W]`；
3. 从目标视频 latent 删除原首槽，再在前面预留一个 reference 槽，保持总时间长度不变；
4. 在每次 `wan.model(...)` 前，把可能已被 flow noise 污染的首槽替换回干净的 `z_ref`；
5. 继承 `wan_animate_slot` 的 token timestep 规则，将首槽 timestep 固定为 0；
6. 继承其 loss mask，去噪 MSE 仅覆盖后续目标槽，不要求模型预测参考图本身。

因此参考图同时有两条互不替代的条件路径：

```text
ref_image ──Qwen role/global routes──> MQ context ──cross-attention──> Wan
         └─Wan VAE───────────────────> clean first latent slot ─────> Wan
```

其中第二条就是 Wan 端的像素级首帧强绑定，即使 MQ 图像条件被置空也不会消失。
checkpoint 的 `three_router_config.json` 会记录 `wan_first_frame_conditioning`
字段，启动时也会打印 `[WAN-FIRST-FRAME]` 解析结果。可显式传
`--disable_wan_first_frame_strong_bind` 做无首帧 latent 条件消融；该分支使用
标准 T2V flow matching，不再恢复监督不一致的旧 soft anchor。

## 4. OpenVid 前 100 条

`scripts/prepare_openvid100.py` 采用以下确定性规则：

1. 按 CSV 原始行顺序读取；
2. 自动识别 video/caption 列；
3. 跳过空字段和找不到本地文件的记录；
4. 选择前 100 条可解析记录；
5. 在 `code/tmp/openvid_first100/videos` 创建软链接；
6. 写出二列 CSV 与完整 `manifest.json`。

脚本不复制大视频，也不修改 NAS 数据。首次生成后默认复用现有 manifest；
只有显式传 `--overwrite` 才重建。

默认数据路径：

```text
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv
```

## 5. 运行方法

### 5.1 轻量检查

不加载 Qwen/Wan：

```bash
cd /home/liuzhirui/Project/MovieStory/code
python train_3router_planner_wan.py --router_check_only
```

单元测试：

```bash
PYTHONPATH=/home/liuzhirui/Project/MovieStory/code \
python -m unittest discover -s \
/home/liuzhirui/Project/MovieStory/code/tests -v
```

### 5.2 训练 planner

```bash
cd /home/liuzhirui/Project/MovieStory/code
bash train_openvid100_3router.sh
```

常用覆盖：

```bash
NUM_TRAIN_STEPS=1000 \
SAVE_STEPS=250 \
DIT_DEVICE=0 \
ENCODER_DEVICE=1 \
WAN_TRAIN_MODE=frozen \
bash train_openvid100_3router.sh
```

checkpoint 输出固定在：

```text
/home/liuzhirui/Project/MovieStory/code/checkpoint/
  three_router_openvid100_steps1000/
    checkpoint-before-training/
    checkpoint-250/
    ...
    checkpoint-final/
```

### 5.3 baseline

使用同样的 144 个 MetaQuery token、同一 Connector、同一 manifest，只旁路 planner：

```bash
ROUTER_ENABLED=0 bash train_openvid100_3router.sh
```

输出目录名为 `baseline_144mq_openvid100_steps*`。

## 6. 默认训练配置及取舍

| 项 | 默认值 | 原因 |
|---|---:|---|
| OpenVid 子集 | 100 | 用户要求，且适合增量过拟合验证 |
| MetaQuery | 144 | 64/48/32 固定布局 |
| Connector layers | 4 | 对应总设计的轻量 adapter，降低首次实验成本 |
| frame_num | 49 | 沿用参考 TI2V frame 脚本的 `4n+1` |
| max_area | 262144 | 控制显存 |
| batch | 1 | 视频尺寸可能不同 |
| grad accumulation | 2 | 保持低峰值显存 |
| Qwen backbone | frozen | 冻结优先 |
| Qwen input embedding | frozen | 防止 144 个唯一 token embedding 吞掉路由收益 |
| Wan | cond_only | 训练 Wan 条件相关层，并保留干净首帧输入 |
| Wan 首帧 | clean latent，timestep=0 | 强绑定 `ref_image`，首槽不加噪且不计 loss |
| loss | flow MSE | 保持与现有 Wan 训练链一致 |
| null image/caption | 0 | 小样本增量验证先避免条件缺失噪声 |

若使用新随机 Connector，100 条样本主要验证链路能否收敛；若要对“planner 是否有效”
做结论，应从同一个已有 **144-token** MetaQuery/Connector checkpoint 分叉运行
baseline/planner。历史 256-token checkpoint 的新增词表 embedding 形状不同，不能
直接当作 144-token 严格同起点；需要先做明确的 token 映射/裁剪实验。

## 7. 已知边界

- 当前增量不等同于总设计的完整 Stage 1；Oracle/Student、三路 ground、
  route distill、RSR/RAB 和 DIM 尚未加入。
- 现有 Wan 训练脚本把 144 tokens 作为单一 MQ context 消费，因此只能验证显式
  route 类型是否改善条件表示；Wan 尚未按三路使用独立 cross-attention。
- 当前 Wan 首帧是固定强绑定，可视作离散的 `gs=1`；连续强度与 Wan 侧可调 `gs`
  通路仍属于后续 executor 增量。
- 大模型训练未包含在 CPU 单元测试中；正式运行需要本地 checkpoint、OpenVid NAS
  挂载和足够 GPU 显存。

## 8. 故障定位

| 现象 | 优先检查 |
|---|---|
| hidden size mismatch | Qwen checkpoint 是否仍为本地 2B（应为 2048） |
| 未取到 144 route states | tokenizer special token、BOI/EOI 是否与 checkpoint 一致 |
| optimizer 报无可训参数 | Connector/planner 是否被误冻结 |
| checkpoint 不含 planner | 是否通过本训练入口启动，`ROUTER_ENABLED` 是否正确 |
| OpenVid 少于 100 | NAS 挂载、CSV 文件名与视频根目录 |
| planner/baseline 无差异 | 对比 checkpoint 起点、确认 on/off、检查 route tensor 差异 |
| route cosine 高 | 这是下一增量 RSR/ground 的信号，不在本层强加正交损失 |
