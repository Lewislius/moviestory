# MovieStory 原生 Wan2.2 I2V-A14B 三路 MetaQuery 训练代码全流程分析

> 分析日期：2026-08-04（UTC）  
> 分析对象：`/home/liuzhirui/Project/MovieStory/code` 中当前最新的原生双 DiT I2V 训练链  
> 核心入口：`train/train_metaquery_i2v_3router_4x48g.py`  
> 启动脚本：`train/train_openvid4000_3router_i2v_a14b_4x48g.sh`  
> Determined 配置：`train/train_openvid4000_3router_i2v_a14b_4x48g.yaml`

> **当前实现勘误（2026-08-04）：** 24 层 Connector 已由 DDP 完整复制改为逐层 FSDP `FULL_SHARD`；8 卡下其参数、梯度和 Adam 静态状态由约 12.2 GiB/卡降至理想约 1.53 GiB/卡。本文后续所有“Connector 仍由 DDP 复制”及由此推导的旧显存结论只代表改造前快照。当前实现、数值一致性处理、checkpoint 和新显存估算见 `CONNECTOR_FSDP_FULL_SHARD_2026-08-04.md`。

> **同日后续默认值更新：** 按后续训练要求，代码默认值已改为 `conditioning_mode=0`、`WAN_TRAIN_MODE=cond_only`、8 rank，并默认开启 Wan activation checkpointing。本文后续关于“修改前默认是 mode 1/frozen/4 rank/未开启 checkpoint”的文字保留为变更前快照；模型结构、条件注入、loss 与显存分析仍适用。由于本文估算认为 8 卡 `cond_only` 仍有较高 OOM 风险，正式长跑前必须先做少量 step 真机验收。

## 1. 先给结论

这段代码训练的是一套“冻结 Qwen3-VL 主干 + 三组可学习 MetaQuery + 一个很大的共享 Connector + 可选 Wan 参数”的 Wan2.2 I2V-A14B 条件系统。它没有修改 Wan transformer block、attention、patch embedding 或 `WanModel.forward` 的结构，而是只通过 Wan 已有的两个接口注入条件：

1. `context`：三路 Qwen/MetaQuery 产生的语义 token；mode 1 还会拼接冻结 UMT5 token。
2. `y`：Wan 原生 I2V 图像条件，即 4 通道时序 mask 与 16 通道首帧 VAE latent 拼成的 20 通道张量。

默认 shell 配置实际训练：

- 三组显式 MetaQuery embedding，合计 `256 × 2048 = 524,288` 个 FP32 参数；
- 24 层共享 Connector，约 **16.36 亿**个 BF16 参数；
- mode 1 时额外训练约 **839.8 万**个 FP32 MQ→T5 mapper 参数；
- 默认 `WAN_TRAIN_MODE=frozen`，所以两套 Wan 14B DiT 参数都不更新；
- Qwen3-VL、UMT5、VAE 均冻结；
- 但即使 Wan/Qwen 冻结，反向传播仍必须穿过它们，才能把视频去噪损失传回 MetaQuery 和 Connector，所以仍会产生大量激活显存。

损失只有原生 rectified-flow/flow-matching velocity MSE：

```text
x_t = (1 - t) x_0 + t ε
target = ε - x_0
loss = mean MSE(Wan(x_t, t, context, y), target)
```

没有 router 辅助 loss、T5 对齐 loss、图像重建 loss、感知 loss、SNR weighting 或 high/low 分支加权。

### 显存总判断

| 配置 | 判断 | 主要原因 |
|---|---|---|
| 4×48GB，`frozen`，默认不开 Wan activation checkpoint | **极高概率 OOM** | 仅双 DiT FP32 FSDP 常驻分片约 26.6 GiB/卡；Connector 训练状态约 12.2 GiB/卡；再加 Qwen、VAE、FSDP buffer 和激活，余量几乎为零 |
| 4×48GB，`cond_only` 或 `full` | **基本不可行** | Wan 梯度与 Adam 状态会再增加数十 GiB/卡 |
| 8×48GB，`frozen`，不开 Wan activation checkpoint | **仍有明显 OOM 风险** | 静态占用下降，但 40 层、约 1.3 万视频 token 的 Wan 反向激活很大 |
| 8×48GB，`frozen`，开启 Wan activation checkpoint，mode 0 | **最值得首跑，预计可行但仍需真机验证** | 每卡双 DiT 分片降到约 13.3 GiB，静态合计约 30 GiB，可给激活和通信 buffer 留出约 14 GiB |
| 8×48GB，`frozen`，开启 Wan activation checkpoint，mode 1 | **中高风险、可能可行** | 相比 mode 0 多 mapper、CPU T5 编码及最多 512 个 cross-attention context token |
| 8×48GB，`cond_only` | **仍很可能 OOM** | Wan 两分支合计约 88.03 亿可训练参数，分片后的梯度和两份 Adam 状态仍约增加 12.3 GiB/卡 |
| 8×48GB，`full` | **不可行** | 仅 Wan 的参数、梯度与 Adam 状态就已经超过单卡 48GB |

因此，如果硬件最多是 8×48GB，推荐第一选择是：

```text
8 卡 + WAN_TRAIN_MODE=frozen + mode 0 + Wan activation checkpointing
```

先做 5～10 个 optimizer step 的真实显存验收，再尝试 mode 1。不要直接在 4 卡上跑当前默认命令，也不要在 8 卡上直接启用 `cond_only/full`。

## 2. 代码范围与调用关系

本报告主要阅读了以下文件：

```text
code/
├── train/
│   ├── train_metaquery_i2v_3router_4x48g.py
│   ├── train_openvid4000_3router_i2v_a14b_4x48g.sh
│   └── train_openvid4000_3router_i2v_a14b_4x48g.yaml
├── native_i2v_3router/
│   ├── module.py
│   ├── encoder.py
│   ├── contracts.py
│   └── distributed.py
├── tests/test_wan_i2v_3router_contracts.py
└── README_METAQUERY_I2V_A14B_3ROUTER.md
```

训练还依赖仓库外的现有实现：

```text
/home/liuzhirui/model/Wan2.2/
├── wan/image2video.py
├── wan/modules/model.py
├── wan/configs/wan_i2v_A14B.py
└── scripts-metaquery-single/train/train_connector_for_wan.py

/home/liuzhirui/model/Qwen3-VL-main/metaquery-main/
└── models/transformer_encoder.py
```

总体调用链如下：

```text
Determined YAML / bash
        │
        ▼
torchrun（当前脚本固定 4 rank）
        │
        ▼
train_metaquery_i2v_3router_4x48g.py
        ├── 初始化 NCCL、检查 GPU
        ├── monkey-patch Wan 的 FSDP 包装函数
        ├── 构造三路 Qwen/MetaQuery encoder
        ├── 构造 mode 0/1 context composer
        ├── 加载 Wan2.2 I2V-A14B 双 DiT、VAE、UMT5
        ├── MQ encoder 用 DDP，两个 Wan DiT 用 FSDP FULL_SHARD
        ├── 构造 OpenVid dataset 与全局 sampler
        └── 500 个 optimizer step
                ├── 读视频前 49 帧
                ├── 三路 Qwen → 共享 Connector → context
                ├── 首帧 → 原生 I2V y
                ├── 全视频 → clean latent x0
                ├── 采样 t 和噪声 ε
                ├── 选择 low/high noise DiT
                ├── velocity MSE
                ├── backward、全局梯度裁剪、AdamW
                └── 每 100 step 保存 checkpoint
```

## 3. 启动时到底采用哪些默认值

### 3.1 三处默认值并不完全一致

需要特别注意以下差异：

- Python 参数 `--conditioning_mode` 默认是 `1`。
- bash 不传第一个位置参数时也默认是 `1`。
- 但 Determined YAML 设置 `WAN_I2V_CONDITIONING_MODE=0`，并把它传给 bash，所以**通过当前 YAML 启动时实际是 mode 0**。
- README 的推荐命令写的是 mode 1；这不等于 YAML 的实际默认值。
- bash 和 YAML 的 `WAN_TRAIN_MODE` 都默认是 `frozen`。

因此：

```text
直接 bash 脚本，不带参数       → mode 1 + frozen Wan
bash 脚本后显式写 0            → mode 0 + frozen Wan
按当前 Determined YAML 提交     → mode 0 + frozen Wan
```

### 3.2 当前 bash 展开的关键配置

| 项目 | 当前值 | 含义 |
|---|---:|---|
| Wan checkpoint | `model/Wan2.2/Wan2.2-I2V-A14B` | 原生 low/high 双 DiT、VAE、UMT5 |
| Qwen checkpoint | `Qwen3-VL-2B-Thinking` | 冻结多模态主干 |
| OpenVid 条数 | 4000 | `local_openvid_limit` |
| optimizer steps | 500 | 每步全局有效 batch 8 |
| global effective batch | 8 | 4 卡时每卡累积 2 次 |
| micro-batch | 1 | DataLoader 固定 `batch_size=1` |
| 帧数 | 49 | 必须满足 `4n+1` |
| 最大画面面积 | 262144 | 即 `512×512` 面积上限，保持宽高比 |
| MetaQuery 总数 | 256 | 必须等于 96+96+64 |
| role/action/global | 96/96/64 | 固定有序切片 |
| Qwen hidden size | 2048 | 与 Qwen3-VL-2B 一致 |
| Connector 层数 | 24 | 代码强制必须为 24 |
| MQ/Connector LR | `1e-5` | AdamW 第一组 |
| Wan LR | `1e-6` | `1e-5 × wan_lr_ratio(0.1)`；仅 Wan 可训练时存在 |
| weight decay | 0.1 | route MetaQuery embedding 单独为 0 |
| warmup | 25 step | 之后 cosine，最小倍率 0.01 |
| max grad norm | 1.0 | DDP+FSDP 混合全局 L2 裁剪 |
| caption dropout | 0.1 | caption 置空，但原生 I2V 首帧仍在 |
| MQ image dropout | 0.1 | Qwen 收到一张 224×224 黑图，原生 `y` 仍在 |
| Wan train mode | `frozen` | 默认不更新两个 DiT |
| Qwen/MQ gradient checkpoint | 开启 | `--disable_mq_gradient_checkpointing` 未传 |
| Wan activation checkpoint | **关闭** | 当前脚本未传 `--enable_wan_activation_checkpointing` |
| 保存频率 | 100 step | 100/200/300/400/500 |
| 日志频率 | 每 step | rank 0 打印并可写 W&B |
| 精度 | BF16 forward | MSE 和 flow pair 转 FP32 |

## 4. 完整训练流程

### 4.1 参数校验

训练开始前会检查：

1. `num_metaqueries == role + action + global`，当前为 `256 == 96+96+64`。
2. Connector 必须为 24 层。
3. `frame_num` 必须满足 `4n+1`，49 对应 VAE temporal stride 4 后的 13 个 latent frame。
4. `global_effective_batch` 必须能被 world size 整除。
5. dropout 概率必须位于 `[0,1]`。
6. Wan 与 Qwen checkpoint 路径必须存在。

### 4.2 初始化 4-rank 分布式环境

当前 shell 固定执行：

```bash
torchrun --standalone --nproc_per_node=4 ... --expected_world_size 4
```

Python 会：

- 读取 `WORLD_SIZE/RANK/LOCAL_RANK`；
- 强制 world size 与 `expected_world_size` 相等；
- 每个进程绑定一张 GPU；
- 使用 NCCL 初始化 process group；
- 做一次 `all_reduce` 探测，确保通信正常；
- 检查每卡总显存至少 44 GiB，训练前空闲显存至少 40 GiB。

这个检查只能排除卡型或已有进程占用问题，不能证明后续不会 OOM。

### 4.3 在加载 Wan 前安装 FSDP 包装策略

`install_native_i2v_fsdp()` 不改 `WanModel` 本身，而是替换 `wan.image2video` 模块局部引用的 `shard_model`。随后 `WanI2V` 创建 low/high 两个模型时，会分别使用：

- FSDP `FULL_SHARD`；
- transformer block 级 auto-wrap；
- `use_orig_params=True`；
- 参数前向 BF16；
- 梯度归约 FP32；
- buffer FP32；
- `limit_all_gathers=True`；
- `forward_prefetch=False`；
- 可选 non-reentrant block activation checkpoint。

当前启动脚本没有打开最后一项，这是 48GB 卡上的关键风险点。

### 4.4 构造三路 Qwen/MetaQuery encoder

基础类是外部的 `MetaQueryEncoderForWan`。它做以下工作：

1. 加载 Qwen3-VL-2B-Thinking，hidden size 为 2048。
2. 不加载 MetaQuery 工程原本的图像扩散 transformer。
3. 新建 24 层双向 Qwen2 Encoder 作为 Connector 主体。
4. Connector 尾部把 2048 维映射到 Wan 的 4096 维文本空间：

```text
24-layer bidirectional Qwen2Encoder(2048, FFN=8192)
  → Linear(2048, 4096)
  → GELU
  → Linear(4096, 4096)
  → RMSNorm(4096)
```

5. 冻结 Qwen backbone，训练共享 Connector。
6. 当前默认不训练 Qwen 原始输入 embedding 整表。

随后 `build_three_router_encoder_class()` 增加三路逻辑。

#### role 路由

```text
输入：参考图像 + 空 caption + role 的 96 个 MQ token
输出：[B, 96, 2048]
```

#### action 路由

```text
输入：caption + 无图像 + action 的 96 个 MQ token
输出：[B, 96, 2048]
```

#### global 路由

```text
输入：参考图像 + caption + global 的 64 个 MQ token
输出：[B, 64, 2048]
```

三路使用同一个冻结 Qwen 权重，但执行三次相互独立的 forward。每路只保留属于自己的 MQ token，其他两路的 MQ token 会从 prompt 中删除。

每路拥有一张独立、可训练的 FP32 MetaQuery embedding 表：

```text
role   [96, 2048]
action [96, 2048]
global [64, 2048]
```

它们从 Qwen 原始 `<img0>...<img255>` embedding 复制初始化。普通输入 token embedding 会 `detach()`，只有这三张显式表接收梯度。

三路输出按固定顺序拼接：

```text
[role, action, global] → [B, 256, 2048]
```

`ThreeRouterPlanner` 只是无参数的形状检查与 identity split，不做 gating、加权、top-k、MLP 或 route mixing。拼接结果只调用一次共享 Connector：

```text
[B, 256, 2048] → Connector → [B, 256, 4096]
```

因此，“路由隔离”发生在 Qwen 输入和三组 MQ 参数处；Connector 和 Wan 是共享后融合层。

### 4.5 构造 mode 0 或 mode 1 的 Wan `context`

#### mode 0：MQ 替代 T5

```text
context = MQ features
shape   = [B, 256, 4096]
Wan text_len = 256
```

UMT5 不参与每步文本编码。MQ token 直接进入 Wan 原生 `text_embedding` 和 cross-attention。

#### mode 1：mapped MQ + frozen T5

MQ 先经过近恒等残差 mapper：

```text
delta = Up(SiLU(Down(RMSNorm(MQ))))
mapped_MQ = MQ + tanh(residual_logit) × delta
```

当前参数：

- hidden size 4096；
- bottleneck 1024；
- 初始 residual scale 0.1；
- `Up` 权重以标准差 `1e-5` 初始化、bias 为 0，因此初始映射非常接近 identity；
- mapper 主权重为 FP32；
- 输出再转回 MQ 输入 dtype。

默认还会做一次无梯度 RMS 比例匹配：

```text
scale = clamp(RMS(T5) / RMS(mapped_MQ), 0.25, 4.0)
mapped_MQ = mapped_MQ × scale
```

最后：

```text
context = concat([mapped_MQ, frozen_T5], dim=tokens)
最大 shape = [B, 256+512, 4096]
Wan text_len = 768
```

顺序固定为 MQ 在前、T5 在后。UMT5 完全冻结并在 CPU 上编码，再把结果送到当前 GPU。

需要准确理解：4096 维相同只代表接口宽度相同，不能证明 MQ 与 UMT5 处于相同语义坐标系。mapper 与 RMS match 正是为减轻这种分布错位，但训练没有显式 T5 alignment loss。

### 4.6 构造数据集与全局样本流

数据集使用外部 `WanVideoDataset`，当前从 OpenVid 本地 CSV/视频目录取前 4000 条。

对一条成功样本：

1. caption token 超过 512 的样本被跳过重试。
2. 视频总帧少于 49、时长小于 0.5 秒或大于 20 秒的样本被跳过。
3. 从视频开头连续读取前 49 帧；**不是随机 clip，也不是全视频等距抽帧**。
4. 保持宽高比缩小到面积不超过 262144。
5. 高宽向下取为 32 的倍数，且至少为 32。
6. 像素归一化到 `[-1,1]`，得到 `[3,49,H,W]`。
7. 处理后的第 0 帧同时作为：
   - `ref_image`：永远存在，送入 Wan 原生 I2V `y`；
   - `mq_ref_image`：通常送入 role/global Qwen，但有 10% 概率设为 `None`。
8. caption 有独立 10% 概率置空。

这里没有旧版“全视频随机参考帧”逻辑，也没有 reference latent prefix。当前最新原生 I2V 路径的参考图就是目标 clip 的第一帧。

当 `mq_ref_image=None` 时，组合模块不会真正省略视觉输入，而是为 Qwen 构造一张 224×224 的黑色 RGB 图片；Wan 的原生首帧 `y` 完全不受该 dropout 影响。

`strict_dataset_size` 只保证 `len(dataset)==4000`。如果某个索引解码失败，Dataset 内部可能改取后续/随机样本，甚至返回 `_last_good_sample`，所以它不能严格证明 4000 个物理视频各使用一次。

`GlobalBatchSampler` 先生成一条确定性全局 `randperm`，再按 optimizer step 和 rank 切分。当前恰好需要：

```text
500 steps × global batch 8 = 4000 draws
```

所以在所有数据都能按索引正常读取时，只使用 seed=42 的第一轮 permutation，每个 dataset index 恰好出现一次。

### 4.7 一个 micro-step 内的目标视频 latent

VAE 冻结并使用 `torch.no_grad()`：

```text
video [3,49,H,W]
  → Wan VAE
  → x0 [16,13,H/8,W/8]
```

因为 VAE stride 是 `(4,8,8)`：

```text
latent_T = (49-1)/4 + 1 = 13
```

### 4.8 原生首帧 I2V 条件 `y`

对目标视频第 0 帧执行：

```text
first_frame [3,H,W]
  → resize 到目标 latent 对应的 H,W
  → 与后续 48 帧全零图拼接
  → conditioning_video [3,49,H,W]
  → 冻结 Wan VAE
  → image_latent [16,13,H/8,W/8]
```

同时构造 4 通道时序 mask：

```text
mask[:, latent time 0] = 1
mask[:, latent time 1:] = 0
shape = [4,13,H/8,W/8]
```

最后：

```text
y = concat([mask4, image_latent16], channel)
shape = [20,13,H/8,W/8]
```

Wan 原生 `forward` 内部再做：

```text
noisy video x_t: 16 channels
native I2V y:    20 channels
concat:          36 channels
  → 原生 patch_embedding(in_dim=36)
```

这就是参考图注入 Wan 的像素级路径。代码没有新增 ControlNet、adapter block、reference prefix、额外 cross-attention 或修改 timestep layout。

### 4.9 采样 timestep 与双 DiT 分支

每个 micro-step 只采样一个标量：

```text
t ~ Uniform[0,1)
wan_t = 1000 × t
```

rank 0 采样后广播给所有 rank，原因是两个 Wan 分支是两个独立 FSDP 模型；所有 rank 必须在同一轮调用相同分支，否则 FSDP collective 会失配或死锁。

分支规则：

```text
wan_t >= 900  → high_noise_model
wan_t <  900  → low_noise_model
```

因此理论频率为：

- low-noise 分支约 90%；
- high-noise 分支约 10%。

如果 Wan 可训练，两个分支不会在同一个 micro-step 同时更新；没有被选中的分支参数本轮 `grad=None`。这意味着 high-noise 分支只有约 10% 的更新机会。

### 4.10 flow-matching pair 与 Wan 前向

每个样本单独生成与 `x0` 同形状的标准高斯噪声：

```text
ε ~ N(0,I)
x_t = (1-t)x0 + tε
v_target = ε-x0
```

在最大 `512×512` 方形样本上：

```text
x0 shape          = [16,13,64,64]
y shape           = [20,13,64,64]
patch size        = [1,2,2]
Wan sequence len  = 13×64×64/(2×2) = 13,312 tokens
Wan hidden width  = 5,120
Wan blocks        = 40
```

随后调用未改签名的原生模型：

```python
predictions = model(
    noisy_inputs,
    t=model_t,
    context=context,
    seq_len=max(sequence_lengths),
    y=[condition.y for condition in conditions],
)
```

### 4.11 backward、梯度累积与优化器更新

4 卡当前配置：

```text
micro batch/rank = 1
rank 数 = 4
gradient accumulation = 8/4 = 2
global effective batch = 1×4×2 = 8
```

前一个 micro-step 对 DDP MQ encoder 使用 `no_sync()`，最后一个 micro-step 才同步 DDP 梯度。Wan FSDP 自身的通信不在这个 `no_sync()` 上下文里，因此这段优化只明确抑制 MQ DDP 的中间同步。

每个 micro loss 先除以 accumulation，然后 backward。DDP 最终对 4 rank 求平均，因此理想全局目标等价于 8 条样本 loss 的平均。

梯度裁剪把两类参数合并计算一个全局 L2 norm：

- DDP 复制参数：MQ/Connector/mapper 梯度已经在 rank 间相同，只计一次；
- FSDP Wan 参数：各 rank 只持有 shard，对平方和做 `all_reduce`；
- 两者合并后用同一个系数裁剪，阈值为 1.0。

随后：

```text
AdamW.step()
cosine scheduler.step()
zero_grad(set_to_none=True)
```

AdamW beta 为 `(0.9,0.95)`。scheduler 前 25 step 线性 warmup，之后 cosine 衰减，但倍率下限为 0.01，不会严格降到 0。

### 4.12 日志与 checkpoint

每个 step 记录：

- loss、grad norm、LR、step 时间和全局吞吐；
- 当前 high/low 分支及累计分支次数；
- route 间 cosine、各 route RMS；
- 三组 route embedding 的梯度 RMS；
- mode 1 的 MQ/T5 RMS 与匹配比例；
- 当前 rank 的 allocated/peak CUDA memory。

注意 `train/loss` 没有在 rank 间 all-reduce。rank 0 打印的是 rank 0 本地两个 micro-batch 的平均，不是严格的 8 样本全局平均；梯度本身仍通过 DDP/FSDP 正确聚合。

每 100 step 保存：

```text
checkpoint-N/
├── mq_qwen_connector_trainable.pt   # rank 0，MQ/Connector/mapper 可训练权重
├── architecture.json                # 结构与注入契约
├── training_args.json               # CLI 参数
└── training_rankXXXXX.pt            # Wan 本地 shard、optimizer、scheduler
```

在 `wan_train_mode=frozen` 时，rank 1～3 的 Wan trainable state 为空，因此只有 rank 0 会写 `training_rank00000.pt`。在 `cond_only/full` 时每个 rank 都会写本地 shard。

代码注释把 rank 文件描述为“same-world-size resume format”，但当前训练入口没有 `--resume_from_checkpoint` 或加载 optimizer/scheduler 的实现。因此这些文件包含恢复所需状态，不等于当前脚本已经支持一键续训。

## 5. 哪些部分真正训练

### 5.1 默认 `frozen` 模式

| 模块 | 参数精度/规模 | `requires_grad` | 是否经反向图 | 说明 |
|---|---:|---:|---:|---|
| role MQ embedding | `96×2048`, FP32 | 是 | 是 | weight decay=0 |
| action MQ embedding | `96×2048`, FP32 | 是 | 是 | weight decay=0 |
| global MQ embedding | `64×2048`, FP32 | 是 | 是 | weight decay=0 |
| 24 层共享 Connector | 约 1.636B, BF16 | 是 | 是 | 最大的默认训练模块 |
| MQ→T5 mapper | 约 8.398M, FP32 | mode 1 是 | mode 1 是 | mode 0 不创建 |
| Qwen3-VL backbone | 约 2B, BF16 | 否 | **是** | 为把梯度传回 MQ embedding，仍需 backward-through |
| Qwen 原 input embedding 整表 | 约 311M, BF16 | 默认否 | 普通 token 被 detach | 三组显式 MQ 表替代本路 MQ 位置 |
| ThreeRouterPlanner | 0 参数 | 不适用 | identity | 只有切片与诊断 |
| Wan UMT5 | checkpoint 约 10.6 GiB | 否 | 否 | mode 1 在 CPU no-grad 编码 |
| Wan VAE | checkpoint 约 0.47 GiB | 否 | 否 | GPU FP32 no-grad 编码两次 |
| low-noise Wan DiT | 14.289B, FP32 原始参数 | 否 | **是** | 约 90% micro-step 被调用 |
| high-noise Wan DiT | 14.289B, FP32 原始参数 | 否 | **是** | 约 10% micro-step 被调用 |

“冻结”不等于“没有反向显存”。Loss 到 Connector/MQ 的梯度必须穿过当前选中的 40 层 Wan DiT；route embedding 的梯度还必须穿过三次冻结 Qwen forward。

### 5.2 `cond_only` 模式

代码按参数名关键字选中两套 Wan 分支中的条件相关参数：

```text
cross_attn
cross-attn
crossattention
cross_attention
text_embedding
time_projection
modulation
cross_attn_norm
norm3
```

对当前实际 checkpoint 的 safetensors header 做精确统计，每个分支选中：

| 类别 | 每分支参数量 |
|---|---:|
| `cross_attn` | 4.195533B |
| `text_embedding` | 0.047196B |
| `time_projection` | 0.157317B |
| `modulation` | 0.001239B |
| `norm3` | 0.000410B |
| 合计 | **4.401695B/分支** |

两个分支合计约 **8.803389B** 可训练参数，占双 DiT 总参数的约 30.8%。这并不是小型 adapter 或 LoRA，所以 `cond_only` 仍然非常吃显存。

这些参数的 LR 是：

```text
wan_lr = learning_rate × wan_lr_ratio = 1e-5 × 0.1 = 1e-6
```

### 5.3 `full` 模式

low/high 两套 Wan 共约：

```text
2 × 14.288901B = 28.577802B parameters
```

全部设置为可训练。即使做 8 卡 FULL_SHARD，FP32 参数、梯度与 Adam 两份状态的总量仍远超 8×48GB 当前实现能承受的范围。

### 5.4 `train_mq_input_embeddings` 的实际注意点

如果传 `--train_mq_input_embeddings`，基础 encoder 会把 Qwen 原始 input embedding 整表设为可训练；但启用三路时 `_route_input_embeddings()` 对普通 embedding 结果立即调用 `.detach()`，再把三组显式 route 参数写入 MQ 位置。

因此在当前默认三路路径中，这个开关会把整表放进 optimizer 参数组，但正常 forward 很可能得不到该整表的梯度。它只在关闭三路、走 baseline `input_ids` 路径时更可能真正生效。默认不传这个 flag 是正确选择。

## 6. 条件究竟如何注入 Wan

当前存在两条完全不同的条件路径。

### 6.1 语义条件：原生 `context` cross-attention

```text
第一帧 ───────────────→ role Qwen ───┐
caption ──────────────→ action Qwen ─┼→ concat 256×2048
第一帧 + caption ─────→ global Qwen ─┘
                                           │
                                           ▼
                                24-layer shared Connector
                                           │
                                           ▼
                                    256×4096 MQ
                                      │          │
                         mode 0 ──────┘          └──── mode 1
                           │                           │
                           ▼                           ▼
                    Wan context            mapper + frozen T5
                                                   │
                                                   ▼
                                      concat [MQ, T5] as context
```

Wan 内部使用原有的：

```text
context → Wan.text_embedding → 每层 cross_attn
```

没有修改 cross-attention 的实现。

### 6.2 像素/时序条件：原生 I2V `y`

```text
目标视频第一帧
    ├── [第一帧, 48 个零帧] → VAE → 16ch image latent ─┐
    └── first-frame temporal mask → 4ch mask ─────────┤
                                                       ▼
                                                y = 20 channels
                                                       │
noisy target latent x_t = 16 channels ─────────────────┤
                                                       ▼
                              native concat → 36ch patch embedding
```

这条路径始终存在，不受 mode 0/1 影响，也不受 `null_caption_prob` 或 `null_image_prob` 影响。

### 6.3 同一张首帧通常被注入两次

正常无 image dropout 时，同一张目标第一帧同时：

1. 经 Qwen role/global 影响 `context`，提供语义/身份/场景信息；
2. 经 Wan VAE 和 mask 进入 `y`，提供原生 I2V 的强首帧条件。

MQ image dropout 只移除第一条路径中的真实图像，第二条原生 `y` 永远保留。因此这不是“完全无图条件”的 CFG dropout。

## 7. 损失函数详解

### 7.1 单样本定义

记：

- `x0`：ground-truth 49 帧视频的 VAE latent；
- `ε`：同形状标准高斯噪声；
- `t ∈ [0,1)`：归一化 flow 时间；
- `c`：mode 0/1 产生的 context；
- `y`：原生 I2V 20 通道首帧条件；
- `fθ`：按 `t` 选出的 low/high Wan DiT。

训练输入：

```math
x_t=(1-t)x_0+t\epsilon
```

velocity target：

```math
v^*=\epsilon-x_0
```

预测：

```math
\hat v=f_\theta(x_t,1000t,c,y)
```

单样本 loss：

```math
L=\operatorname{mean}_{C,T,H,W}\left(\hat v-v^*\right)^2
```

代码把 prediction 转为 FP32，target 本身也是 FP32，再使用默认 `reduction="mean"` 的 `F.mse_loss`。

### 7.2 batch 与分布式归约

模型内部先对 batch 中样本 loss 求平均；当前 micro-batch 固定为 1。训练循环再除以 accumulation。

4 卡时的理想等效目标为：

```math
L_{step}=\frac{1}{8}\sum_{r=0}^{3}\sum_{m=0}^{1}L_{r,m}
```

8 卡、全局 batch 仍为 8 时 accumulation 变成 1：

```math
L_{step}=\frac{1}{8}\sum_{r=0}^{7}L_r
```

### 7.3 没有的 loss

当前明确没有：

- role/action/global 各自监督；
- route orthogonality/cosine penalty；
- MQ-to-T5 feature alignment loss；
- T5 distillation；
- 首帧 reconstruction/perceptual loss；
- high/low branch balance loss；
- SNR 或 timestep reweighting；
- reference consistency loss；
- temporal smoothness loss。

日志里的 route cosine/RMS 只用于诊断，不会加入 loss。

## 8. 完整参数说明

### 8.1 路径和数据源参数

| 参数 | Python 默认值 | 当前 shell 有效值 | 作用 |
|---|---|---|---|
| `--wan_checkpoint_dir` | 本地 Wan2.2-I2V-A14B | 同默认绝对路径 | Wan 双 DiT/VAE/T5 |
| `--qwen3vl_model_id` | 本地 Qwen3-VL-2B | 同默认绝对路径 | Qwen 多模态 encoder |
| `--output_dir` | `./i2v_3router_output` | 含 mode、4000、4x48g、500 step | checkpoint 目录 |
| `--local_openvid_video_root` | `None` | NAS OpenVid `video/` | 视频根目录 |
| `--local_openvid_csv_path` | `None` | OpenVid CSV | caption/视频索引 |
| `--local_openvid_limit` | 4000 | 4000 | 最多读取记录数 |
| `--local_video_cache_dir` | `None` | `None` | Dataset 自行选择 `.hf_cache/video_cache` |
| `--caption_tokenizer_path` | `google/umt5-xxl` | Wan checkpoint 内 tokenizer | caption 长度检查 tokenizer |

### 8.2 拓扑和优化参数

| 参数 | 默认/当前值 | 作用 |
|---|---:|---|
| `--expected_world_size` | 4 | world size 硬校验 |
| `--global_effective_batch` | 8 | 决定每 rank accumulation |
| `--expected_train_samples` | 4000 | strict size 目标 |
| `--num_train_steps` | 500 | optimizer 更新次数 |
| `--learning_rate` | `1e-5` | MQ/Connector/mapper LR |
| `--wan_lr_ratio` | 0.1 | Wan 相对 LR |
| `--weight_decay` | 0.1 | route embedding 除外 |
| `--warmup_steps` | 25 | 线性 warmup |
| `--max_grad_norm` | 1.0 | 全局 L2 clipping |
| `--seed` | 42 | sampler 与每 rank RNG 基准 |
| `--save_steps` | 100 | checkpoint 周期 |
| `--log_steps` | 1 | rank 0 日志周期 |
| `--dataloader_num_workers` | 0 | 主进程同步解码 |

### 8.3 视频和 dropout 参数

| 参数 | 默认/当前值 | 作用 |
|---|---:|---|
| `--frame_num` | 49 | 取开头连续 49 帧 |
| `--max_area` | 262144 | resize 面积上限 |
| `--max_caption_tokens` | 512 | 超长样本重试 |
| `--min_duration_sec` | 0.5 | 最短时长 |
| `--max_duration_sec` | 20.0 | 最长时长 |
| `--null_caption_prob` | 0.1 | caption 置空概率 |
| `--null_image_prob` | 0.1 | MQ 真图替换为黑图的概率 |

两个 dropout 独立调用 Python `random.random()`。忽略解码重试造成的 RNG 消耗变化时，理论组合概率是：81% 图文都有、9% 只清 caption、9% 只清 MQ 图、1% 两者都清；原生 I2V `y` 在四种状态下都存在。

### 8.4 MQ、router 和 context 参数

| 参数 | 默认/当前值 | 作用 |
|---|---:|---|
| `--conditioning_mode` | Python/bash 1，YAML 0 | MQ-only 或 `[mapped MQ,T5]` |
| `--num_metaqueries` | 256 | MQ 总 token 数 |
| `--connector_num_hidden_layers` | 24 | 代码强制固定 24 |
| `--mapper_bottleneck_size` | 1024 | mode 1 mapper 中间宽度 |
| `--mapper_residual_scale` | 0.1 | mapper 初始残差强度 |
| `--disable_mapper_rms_match` | false | 默认开启 MQ/T5 RMS match |
| `--router_hidden_size` | 2048 | 必须匹配 Qwen hidden |
| `--router_role_tokens` | 96 | role token 数 |
| `--router_action_tokens` | 96 | action token 数 |
| `--router_global_tokens` | 64 | global token 数 |
| `--disable_3router` | false | 默认启用三次隔离 Qwen forward |
| `--disable_mq_gradient_checkpointing` | false | 默认开启 Qwen/Connector checkpoint |
| `--train_mq_input_embeddings` | false | 默认只训练显式 route 表 |
| `--connector_norm_init_scale` | 1.0 | Connector 尾部 RMSNorm 初值 |

### 8.5 Wan、FSDP 与安全参数

| 参数 | 默认/当前值 | 作用 |
|---|---:|---|
| `--wan_train_mode` | `frozen` | `frozen/cond_only/full` |
| `--wan_cond_name_pattern` | 空 | 空时使用内置关键字集合 |
| `--enable_wan_activation_checkpointing` | false | 当前最应在 48GB 上开启的选项 |
| `--minimum_gpu_memory_gib` | 44 | 卡容量下限 |
| `--minimum_free_gpu_memory_gib` | 40 | 启动空闲显存下限 |
| `--strict_dataset_size` | shell 已开启 | 强制 `len(dataset)==4000` |

### 8.6 W&B 与检查模式

| 参数 | 默认/当前值 | 作用 |
|---|---:|---|
| `--wandb_enabled` | shell 由 `WANDB_ENABLED` 决定 | YAML 当前设为 1 |
| `--wandb_project` | `wan-i2v-a14b-3router` | 项目名 |
| `--wandb_run_name` | 自动或 shell 拼接 | run 名称 |
| `--wandb_mode` | `online` | online/offline/disabled |
| `--check_only` | false | 不加载大模型的 contract 检查 |
| `--parse_only` | false | 打印解析参数；仍会先做路径/参数校验 |

### 8.7 影响 shell 的环境变量

除 CLI 外，启动脚本还支持：

```text
CONDA_SH, CONDA_ENV
WAN_CHECKPOINT, QWEN_CHECKPOINT
OPENVID_ROOT, OPENVID_VIDEO_ROOT, OPENVID_CSV_PATH
CAPTION_TOKENIZER
OPENVID_LIMIT, NUM_TRAIN_STEPS, GLOBAL_EFFECTIVE_BATCH
WAN_TRAIN_MODE, OUTPUT_DIR
LEARNING_RATE, WARMUP_STEPS, SAVE_STEPS
NULL_CAPTION_PROB, NULL_IMAGE_PROB
WANDB_ENABLED, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_MODE
PYTORCH_CUDA_ALLOC_CONF, NCCL_ASYNC_ERROR_HANDLING
TORCH_NCCL_ASYNC_ERROR_HANDLING, OMP_NUM_THREADS
WAN_DATA_PRECLEAN
```

当前默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`，可减轻碎片，但不能解决真实容量不足。

## 9. 4×48GB 与 8×48GB 的显存估算

### 9.1 可直接从 checkpoint 得到的硬数据

每个 Wan 分支的 safetensors index 声明：

```text
total_size = 57,155,604,736 bytes = 53.23 GiB
parameter count = 14,288,901,184
checkpoint dtype = FP32
```

low/high 两分支合计：

```text
28.578B parameters
106.46 GiB FP32 checkpoint payload
```

在理想 FULL_SHARD 静态均分下：

| GPU 数 | 双 DiT 原始参数 shard/卡 |
|---:|---:|
| 4 | 约 26.6 GiB |
| 8 | 约 13.3 GiB |

FSDP `MixedPrecision(param_dtype=BF16)`主要控制前向 all-gather/计算精度；`use_orig_params=True` 下原始 shard 通常仍按 checkpoint FP32 保存。因此不能简单把 106.46 GiB 再除以 2 当成静态权重。

### 9.2 默认可训练 Connector 的成本

按 24 层、hidden 2048、FFN 8192、Q/K/V/O、gate/up/down 及两层 4096 投影计算：

```text
Connector ≈ 1.636B parameters
BF16 parameter payload ≈ 3.05 GiB/卡
```

它由 DDP 完整复制，没有 FSDP/ZeRO 分片。以当前 AdamW 常见的同 dtype `exp_avg/exp_avg_sq` 估算：

```text
weights    ≈ 3.05 GiB
gradients  ≈ 3.05 GiB
Adam m     ≈ 3.05 GiB
Adam v     ≈ 3.05 GiB
合计       ≈ 12.2 GiB/卡
```

若具体 PyTorch 路径保留额外 master copy 或 foreach optimizer 临时 tensor，step 峰值还会更高。

### 9.3 `frozen` 的静态基线

粗略按每卡计算：

| 组件 | 4 卡 | 8 卡 | 是否分片 |
|---|---:|---:|---|
| 两套 Wan FP32 参数 shard | 26.6 GiB | 13.3 GiB | 是 |
| Connector 参数+梯度+Adam | 约 12.2 GiB | 约 12.2 GiB | 否，DDP 复制 |
| 冻结 Qwen3-VL-2B | 约 4.0 GiB | 约 4.0 GiB | 否 |
| VAE | 约 0.47 GiB 以上 | 同左 | 否 |
| route tables/mode 1 mapper | 约 0.01/0.13 GiB | 同左 | 否 |
| 静态小计 | **约 43.3 GiB** | **约 30.0 GiB** | 未含激活和 buffer |

48GB 显卡在 PyTorch 中通常只显示约 44.7 GiB。4 卡静态小计已经非常接近物理上限，还没有计算：

- 3 次 Qwen forward 的反向图；
- 40 层 Wan 对 context 的 backward 激活；
- FSDP all-gather/reshard buffer；
- DDP bucket、NCCL context、CUDA kernel workspace；
- noisy latent、`y`、prediction、target；
- AdamW step 的临时 tensor；
- allocator 碎片。

因此 4 卡 `frozen` 也应判断为极高概率 OOM，而不是“可能刚好卡住”。

### 9.4 为什么默认关闭 Wan checkpoint 尤其危险

最大面积样本约有 13,312 个 Wan token。单个 BF16 hidden state：

```text
13,312 × 5,120 × 2 bytes ≈ 130 MiB
```

Wan 有 40 层，每层还有 attention/FFN/norm 等中间量。即使参数冻结，为计算 `∂loss/∂context` 仍要保留反向所需激活。不开 block checkpoint 时，这部分很容易达到数十 GiB。

开启 checkpoint 后也不是零成本，但能把大量层内中间量换成 backward 重算，是 8×48GB 方案中必须优先打开的选项。

### 9.5 `cond_only` 为什么 8 卡也危险

两个分支合计约 8.803B 个 FP32 trainable Wan 参数。除已经计入的参数 shard 外，梯度与两份 Adam 状态大约额外需要：

| GPU 数 | Wan cond-only 额外 grad+Adam/卡 |
|---:|---:|
| 4 | 约 24.6 GiB |
| 8 | 约 12.3 GiB |

所以 8 卡静态基线会从约 30.0 GiB 上升到约 42.3 GiB，几乎不给激活和通信留空间。结论仍是很可能 OOM。

### 9.6 `full` 为什么不应尝试

8 卡时，双 DiT FP32 参数本身约 13.3 GiB/卡；再加 FP32 gradient 和两份 Adam 状态，Wan 单独就约：

```text
13.3 × (parameter + gradient + Adam m + Adam v)
≈ 53.2 GiB/卡
```

还没有计算 Connector/Qwen/激活，已经超过 48GB。

### 9.7 旧日志不能作为这条新代码的直接显存证明

`Project/MovieStory/log` 中已有旧 4 卡训练记录显示过约 30GB reserved，但那些 experiment 使用的是旧的 `train_3router_wan_4x48g.py` 训练链，不是本报告分析的原生 Wan2.2 I2V-A14B low/high 双 14B DiT 入口。

所以不能用旧日志得出“当前双 14B 代码 4×48GB 已实测安全”的结论。当前目录也没有这条新 native I2V 入口的真实峰值日志。

### 9.8 扩到 8 卡还要检查主机内存和启动 I/O

扩 rank 会降低 GPU 上的 Wan shard，却会增加未分片 CPU 组件的副本数：

- `WanI2V` 在每个 rank 都创建一套 CPU UMT5；其 checkpoint 约 10.6 GiB，所以 8 rank 仅 T5 权重就可能占约 85 GiB 主机内存。即使 mode 0 每步不调用 T5，初始化阶段仍会加载它。
- 每个 rank 都会执行 low/high `from_pretrained` 再交给 FSDP。实现并不是 rank 0 单独读取后直接散发空模型，因此启动时可能同时读取多个 53.23 GiB 分支 checkpoint，带来很高的临时 CPU RAM、page cache 和共享存储带宽压力。
- 每个 rank 还分别加载 Qwen checkpoint，虽然最终权重在各自 GPU 上，初始化时仍会造成额外主机内存与 I/O 峰值。

所以 8 卡节点除了 8×48GB GPU，建议至少确认有约 512GB 主机内存和足够快的 checkpoint 存储；更稳妥是 1TB RAM。若主机内存较小，任务可能在 GPU forward 前就因 CPU OOM 或加载超时失败。

## 10. 推荐的 8 卡启动策略

### 10.1 当前代码对 8 卡的支持边界

Python 主训练逻辑和 `GlobalBatchSampler` 本身支持 `expected_world_size=8`，因为全局 batch 8 能整除 8。但当前外围文件硬编码了 4 卡：

- bash：`--nproc_per_node=4`；
- bash：`--expected_world_size 4`；
- YAML：`slots_per_trial: 4`；
- 输出目录名字也固定包含 `4x48g`。

因此不能只申请 8 卡后直接执行原脚本；必须同时把以上拓扑值改为 8，并传：

```text
--enable_wan_activation_checkpointing
--wan_train_mode frozen
```

### 10.2 建议首跑配置

```text
world size                  = 8
global effective batch      = 8
micro batch/rank            = 1
gradient accumulation       = 1
conditioning mode           = 0
Wan train mode              = frozen
Wan activation checkpoint   = enabled
frame_num                   = 49
max_area                    = 262144
steps                       = 5～10（显存验收）
save_steps                  > 验收 steps（避免首次测试被保存干扰）
```

验收时必须观察所有 rank，而不只是 rank 0：

```text
torch.cuda.max_memory_allocated
nvidia-smi 每卡峰值
是否在 optimizer.step() 瞬间上升
low-noise 与 high-noise 分支是否都至少跑到一次
checkpoint 保存时 CPU RAM 与磁盘峰值
```

如果 5～10 step mode 0 峰值低于约 42 GiB，再测试 mode 1。建议保留至少 2～3 GiB 余量，不把 44.7 GiB 完全吃满。

### 10.3 8 卡与 4 卡并非逐随机变量完全等效

4 卡、global batch 8 时每个 optimizer step 有两个 micro-step，因此通常采两个共享 timestep；每个 timestep 覆盖 4 条样本。

8 卡、global batch 8 时 accumulation=1，每个 optimizer step 只有一个共享 timestep，覆盖全部 8 条样本。

两者的单样本 timestep 边缘分布仍是 Uniform，但 batch 内 timestep 相关性不同。这不会改变 loss 公式，却可能造成轻微优化统计差异。之所以全 rank 共享 timestep，是为了保证所有 rank 调用同一 FSDP low/high 分支。

### 10.4 如果 8 卡 frozen 仍 OOM

按优先级建议：

1. 确认 Wan activation checkpoint 确实启用；当前 metadata 尚未导出这个字段，应检查 FSDP wrapper 的 `_native_i2v_activation_checkpointing`，或先把该状态补进 `architecture.json`。
2. 先用 mode 0，避免 768-token context。
3. 保持 `WAN_TRAIN_MODE=frozen`。
4. 把 24 层 Connector 从整模块 DDP 改为 block-level FSDP/ZeRO，分片其约 12.2 GiB/卡训练状态；这是最有价值的代码级扩展。
5. 再考虑 8-bit optimizer 或 optimizer state CPU offload。
6. 让未被选择的另一套 Wan DiT CPU offload 可以节省常驻显存，但每个 micro-step 可能产生很重的 PCIe 迁移，且需要仔细处理 FSDP collective。
7. 最后才降低 `max_area`（如 196608）或 `frame_num`（仍须 `4n+1`，如 33）；这会改变训练数据规格。

当前代码不具备第 4～6 项，不能把它们当作已经生效的保护。

## 11. 代码行为中的重要风险与注意点

### 11.1 4 卡推荐文字与真实显存结构不匹配

参数 help 和 README 把 `frozen` 描述为推荐 4×48GB 模式，但代码同时：

- 常驻两套 14B DiT；
- 用 FP32 original FSDP shards；
- DDP 完整复制 1.636B Connector 和 Adam 状态；
- 默认关闭 Wan activation checkpoint。

从静态容量计算看，这个推荐过于乐观，必须以 8 卡真机短跑重新验收。

### 11.2 `null_image_prob` 不是原生 I2V 图像 dropout

它只把 Qwen 图像替换为黑图；`ref_image → y` 永远存在。若目标是训练真正的无图 CFG 分支，需要额外定义如何置空/替换原生 `y`，当前代码没有做。

### 11.3 训练样本不是随机参考帧

当前代码始终读取视频开头连续 49 帧，并使用第 0 帧。不要把旧 `four_gpu_training/random_reference.py` 的逻辑套用到这条 native I2V 路径。

### 11.4 high-noise 分支更新样本明显更少

Uniform `t` 加 0.9 boundary 意味着 high-noise 只有约 10% micro-step。500 step、4 卡 accumulation 2 时约有 1000 次 timestep 抽样，high 分支期望约 100 次；8 卡 accumulation 1 时只有 500 次抽样，期望约 50 次。

### 11.5 保存逻辑不等于完整续训逻辑

虽然保存 optimizer/scheduler/local shard，但没有加载入口。训练中断后不能仅传一个参数自动续跑。

### 11.6 显存日志是当前 rank 的累计峰值

代码没有在每 step 调 `torch.cuda.reset_peak_memory_stats()`，所以 `train/cuda_peak_gib` 是从进程启动以来的累计最高值。这适合安全验收，但不是每一步独立峰值。

### 11.7 数据失败可能改变“每条一次”的语义

Sampler index 不重复不代表 Dataset 最终返回的视频绝不重复。Dataset 的 retry 和 `_last_good_sample` fallback 可能替换坏样本。若训练需要严格 4000 个不同视频，应该在训练前预清洗并禁止 fallback。

## 12. 本次静态验证结果

已执行不加载大模型的 contract 检查：

```bash
python train/train_metaquery_i2v_3router_4x48g.py --check_only
```

结果确认：

- router 形状为 96/96/64，总计 256；
- mode 1 最大 context length 为 768；
- 4 rank 各得到 1000 draw，总计 4000；
- 原生图像条件契约为 `mask4 + image_latent16`；
- flow pair 与 velocity target 正确；
- dual-DiT boundary 为 900。

还直接执行了 `tests/test_wan_i2v_3router_contracts.py` 中 5 个 CPU 测试函数，全部通过：

1. 原生 I2V condition 与 Wan generate 操作一致；
2. flow target 为 `noise-clean`；
3. router 为固定顺序 identity split；
4. mode 1 顺序为 `[mapped MQ, frozen T5]`；
5. 全局 sampler 在 4 rank 间无重叠。

当前环境没有可用训练 GPU，因此没有把 4 卡或 8 卡 OOM 结论伪装成真机实测。本文显存结论来自实际 checkpoint dtype/参数量、FSDP/DDP/AdamW 代码路径和张量尺寸的保守估算；最终仍应以 8×48GB 节点上 5～10 step 的 `max_memory_allocated` 为准。

## 13. 最终建议

如果目标是先验证三路 MetaQuery 是否能有效控制原生 Wan2.2 I2V-A14B：

```text
第一阶段：8×48GB，mode 0，Wan frozen，开启 Wan activation checkpoint，5～10 step
第二阶段：同配置跑满 500 step，确认 loss、route grad、high/low 分支和 checkpoint
第三阶段：8×48GB 切 mode 1，先短跑再全量
```

不要在当前实现上尝试：

```text
4×48GB cond_only/full
8×48GB full
8×48GB cond_only（除非先分片 Connector/optimizer 或增加 offload）
```

如果最终确实需要更新 Wan 条件侧，建议把“cond-only 选中 44 亿参数/分支”改为 LoRA、低秩 adapter 或更窄的精确参数集合；当前按关键字选择的 `cond_only` 实际规模太大，不适合最多 8×48GB 的预算。
