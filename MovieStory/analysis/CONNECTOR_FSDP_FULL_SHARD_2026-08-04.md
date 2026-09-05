# 24 层 Connector 改为 FSDP FULL_SHARD 的实现说明

> 日期：2026-08-04（UTC）  
> 训练入口：`/home/liuzhirui/Project/MovieStory/code/train/train_metaquery_i2v_3router_4x48g.py`  
> 分布式实现：`/home/liuzhirui/Project/MovieStory/code/native_i2v_3router/distributed.py`

## 1. 结论

24 层共享 Connector 已从“MQ encoder 整体 DDP、Connector 每卡完整复制”改为 **Connector 自身的 FSDP `FULL_SHARD`**。冻结 Qwen 不进入这个 FSDP；三组 route MetaQuery table 和 mode 1 mapper 仍保持复制，但显式恢复 DDP 的初始化广播和平均梯度语义。

当前默认配置仍是：

```text
8 × 48GB
conditioning_mode = 0
WAN_TRAIN_MODE = cond_only
Wan activation checkpointing = enabled
global effective batch = 8
```

这次改造不改变模型结构、训练参数集合、forward、条件 token、loss、学习率或全局 batch。它改变的只是 Connector 参数、梯度和 Adam 状态在 8 张卡上的存储方式。

## 2. 分片边界

实际边界如下：

```text
冻结 Qwen3-VL                         每 rank 复制，不训练
三组 route MQ table                  每 rank 复制，训练
mode 1 MQ→T5 mapper（mode 0 不存在）  每 rank 复制，训练
24 层 Connector + 输出投影            FSDP FULL_SHARD，训练
Wan low/high DiT                     原有 FSDP FULL_SHARD
```

Connector 的 24 个 encoder layer 分别作为 FSDP wrap unit；外层 Connector 也是 FSDP root，因此 24 层之外的 projection、norm 等参数也会被分片。没有把整个 MQ encoder 包成 FSDP，避免冻结 Qwen 被反复 all-gather，也避免 Qwen/Connector 混合精度和 trainability 边界变得不清晰。

使用的关键策略为：

```text
ShardingStrategy.FULL_SHARD
use_orig_params = True
sync_module_states = True
limit_all_gathers = True
forward_prefetch = False
backward_prefetch = BACKWARD_POST
24-layer auto wrap
```

启动时会收集每个 rank 实际持有的 Connector 参数元素数并验证：所有 shard 的总和必须等于分片前完整参数量，而且多卡时任一 rank 都不能仍持有完整 Connector。审计结果会写入训练 metadata 和 checkpoint 的 `architecture.json`。

## 3. 显存变化

按约 16.36 亿 BF16 参数估算，原 DDP 每卡静态训练状态约为：

| 状态 | DDP 每卡 |
|---|---:|
| BF16 参数 | 约 3.05 GiB |
| BF16 梯度 | 约 3.05 GiB |
| Adam `exp_avg` | 约 3.05 GiB |
| Adam `exp_avg_sq` | 约 3.05 GiB |
| 合计 | 约 12.2 GiB |

8 卡 FULL_SHARD 的理想均分为：

```text
12.2 GiB ÷ 8 ≈ 1.53 GiB/卡
静态节省 ≈ 10.67 GiB/卡
```

实际峰值不会恒定停在 1.53 GiB，因为 FSDP forward/backward 会按 layer 临时 all-gather 当前单元的完整参数，并使用通信 buffer。24 层逐层 wrap 的目的就是避免一次性 all-gather 完整 16.36 亿参数。Connector 激活显存也不会被参数分片自动消除；现有 MQ gradient checkpointing 继续负责用重算换取激活显存。

结合原先对默认 8 卡 `cond_only` 的保守静态估算，Connector 分片可把每卡静态小计从约 42.3 GiB 降至约 31.6 GiB。当前 shell 还使用 41 帧、mode 0 和 Wan activation checkpointing，因此配置由“很可能 OOM”改善到“有现实可行性，但仍需真机短跑确认”。剩余主要风险来自：

- 两套 Wan DiT 的 FP32 FSDP shard；
- `cond_only` 选中的 Wan 梯度及两份 Adam 状态；
- Wan block all-gather、重算和视频 token 激活；
- 三次冻结 Qwen forward/backward-through 的激活；
- optimizer step、NCCL workspace 与 allocator 碎片；
- checkpoint 时 rank 0 在 CPU 汇集约 3.05 GiB 的完整 Connector state。

因此不能在没有 8×48GB 实测的情况下承诺绝不 OOM。建议先设置 `NUM_TRAIN_STEPS=5`、`SAVE_STEPS=5` 真跑，既覆盖 forward/backward/Adam，也覆盖完整 Connector checkpoint 导出；确认所有卡峰值最好低于约 42 GiB，再开始 500 step 长跑。

## 4. 为什么训练语义保持一致

### 4.1 参数、前向和 loss 不变

Connector 仍是原来的同一个 `nn.Sequential`、同一组 24 层和输出投影；FSDP 只在执行每个 layer 前临时聚合权重，执行后重新分片。仍然训练全部 Connector 参数，条件输出仍是 `[B, 256, 4096]`（默认 mode 0），loss 仍是：

```text
x_t = (1 - t) x0 + t * noise
target = noise - x0
loss = mean MSE(Wan(x_t, t, context, y), target)
```

### 4.2 初始参数同步与旧 DDP 一致

旧 DDP 构造时会把 rank 0 参数广播到其他 rank。Connector FSDP 用 `sync_module_states=True` 完成同样的 rank-0 初始化同步。

route MQ table 和 mode 1 mapper 没有进入 Connector FSDP，因此代码另外对它们执行一次 rank-0 broadcast。这个步骤尤其重要：mode 1 mapper 是随机初始化；若不广播，不同 rank 会从不同参数开始，之后仅平均梯度也无法恢复 DDP 语义。

### 4.3 梯度平均与全局 batch 不变

Connector shard 梯度由 FSDP reduce-scatter，语义为跨 rank 平均后的 shard。复制的 route MQ/mapper 梯度在每个 optimizer step 前显式执行：

```text
all_reduce(SUM)
gradient /= world_size
```

这与 DDP 的平均梯度一致。默认 8 rank、global batch 8 时每卡 micro batch 为 1、accumulation 为 1。如果将来 world size 减小、accumulation 增加，Connector 每个 micro-step 都执行 reduce-scatter；没有使用 `FSDP.no_sync()`，因为后者会在累积期间保留完整、未分片的 Connector gradient，抵消本次显存优化。每个 micro loss 先除以 accumulation，所以“逐 micro 平均后相加”与原全局平均目标一致。

### 4.4 梯度裁剪仍基于完整参数空间

复制参数的梯度平方和只计算一次；Connector 与 Wan 的本地 shard 梯度平方和在所有 rank 上求和；两部分相加得到完整模型的全局 L2 norm，再对全部梯度使用同一个裁剪系数。因此 `max_grad_norm=1.0` 的含义没有变成“每卡各裁各的”。

### 4.5 优化器方程与超参数不变

Connector 仍位于原学习率和 weight decay 组：

```text
learning_rate = 1e-5
weight_decay = 0.1
AdamW betas = (0.9, 0.95)
```

optimizer 在 Connector 完成 FSDP 包装后构造，所以 Adam 状态跟随本地 original-parameter shard，而不是先创建完整 Adam 状态再分片。`foreach=False` 避免 optimizer step 额外生成参数规模的 tensor list；逐参数 AdamW 方程和超参数不变。

FSDP/NCCL collective 的浮点归约顺序可能与 DDP bucket 不完全相同，所以不能承诺逐 bit 相同；这属于正常的分布式浮点舍入差异，不改变目标函数或预期优化行为。

## 5. Checkpoint 行为

每次保存时，所有 rank 参与 Connector full-state collective；只有 rank 0 得到 CPU 上的完整、无 FSDP wrapper 前缀的 Connector state，并继续生成：

```text
mq_qwen_connector_trainable.pt
```

因此推理侧需要的可移植 MQ/Connector 参数格式不因训练时分片而改变。每个 rank 仍各自保存 `training_rankXXXXX.pt`，其中 optimizer 包含该 rank 的 Connector/Wan shard。`architecture.json` 会记录 `FSDP_FULL_SHARD`、24 层 wrap、完整参数量和各 rank shard 数量。

当前训练入口仍没有实现 `--resume_from_checkpoint`，所以这些 rank state 是恢复所需数据，不等于已经支持一键断点续训。完整 Connector full-state 导出采用 PyTorch 2.8 官方 distributed checkpoint state-dict API；当前无可用 CUDA 节点，CPU FSDP 只完成了分片构造测试，最终仍需用 8 卡短跑验证 GPU full-state 保存路径。

## 6. 已完成的验证

已完成：

1. Python `py_compile`；
2. bash `bash -n`；
3. `--check_only`：默认 mode 0、8 rank、每 rank 500 draw、总计 4000；
4. 两 rank Gloo 测试：复制参数从 rank 0 广播成功；rank 梯度 1 和 2 平均为 1.5；
5. 两 rank CPU 微型 FSDP 测试：24 个 layer 被 wrap，Connector 参数在两 rank 间真实分片，shard 总和等于完整参数量；
6. 保存逻辑使用 full Connector state，拒绝带 `_fsdp_wrapped_module` 的不可移植 key。

尚未完成、必须在目标机器完成：

1. 8×48GB 的完整大模型加载；
2. 至少一个 low-noise 和一个 high-noise step；
3. AdamW 后的实际峰值显存；
4. step 5 的 GPU full-state checkpoint 保存和重新加载；
5. 500 step 长跑的吞吐、loss 和数值稳定性。

## 7. 推荐验收命令

```bash
cd /home/liuzhirui/Project/MovieStory/code
NUM_TRAIN_STEPS=5 SAVE_STEPS=5 NPROC_PER_NODE=8 \
  WAN_TRAIN_MODE=cond_only WAN_ACTIVATION_CHECKPOINTING=1 \
  bash train/train_openvid4000_3router_i2v_a14b_4x48g.sh 0
```

文件名中的 `4x48g` 只是历史兼容名；当前 shell、YAML 和 Python 默认均为 8 卡。
