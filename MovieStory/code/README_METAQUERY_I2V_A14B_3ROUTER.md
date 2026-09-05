# Wan2.2 I2V-A14B + MetaQuery/Qwen 3-router

这套代码把 Qwen3-VL、MetaQuery、共享 Connector 和 Wan2.2 I2V-A14B 组合为一个训练模块，同时把 Wan 生成侧保持为原生双 DiT I2V 路径。

## 不变量

- 不修改 `WanModel`、任一 transformer block、attention、patch embedding 或 forward 签名。
- 图像只经原生 `y` 注入：`4ch mask + 16ch VAE(first_frame + zero frames)`，再由原生 I2V forward 与 16ch noisy latent 做 channel concat。
- 视频 target 不添加 reference prefix、不删除第一 latent slot，不把局部 timestep 设为 0。
- 去噪目标固定为 `x_t=(1-t)x0+t*noise`、`target=noise-x0`、`loss=MSE(pred,target)`。
- `t >= 900` 调用原生 `high_noise_model`，否则调用 `low_noise_model`。分布式训练会广播同一个 `t`，保证八个 rank 同时进入同一分支。

## 3-router

- role：仅参考图像；96 个 MetaQuery token。
- action：仅 caption；96 个 MetaQuery token。
- global：参考图像和 caption；64 个 MetaQuery token。
- 三路分别运行冻结的 Qwen；所得 256 个 hidden states 按上述顺序拼接，仅运行一次共享 Connector。

`conditioning_mode=0` 使用 MQ context 替代 T5；`conditioning_mode=1` 使用近恒等 mapper 后的 MQ，再拼接冻结的原生 UMT5 context。

## 文件

- `native_i2v_3router/module.py`：完整组合模块 `MetaQueryQwenWanI2VA14B`。
- `native_i2v_3router/contracts.py`：原生 I2V 图像条件与 flow-matching contract。
- `native_i2v_3router/encoder.py`：3-router Qwen/MetaQuery 和共享 Connector。
- `native_i2v_3router/distributed.py`：Connector/双 DiT FSDP、复制 MQ 同步、全局 batch sampler、混合分片梯度裁剪。
- `train/train_metaquery_i2v_3router_4x48g.py`：历史文件名保留，当前默认是 8-rank 训练入口。
- `train/train_openvid4000_3router_i2v_a14b_4x48g.sh`：OpenVid4000 启动脚本。
- `tests/test_wan_i2v_3router_contracts.py`：原生条件、loss、router 与 sampler 的 CPU contract 测试。

## 使用

先做不加载大模型的配置检查：

```bash
python train/train_metaquery_i2v_3router_4x48g.py --check_only
```

默认八卡训练（mode 0、Wan `cond_only`、开启 Wan activation checkpointing）：

```bash
bash train/train_openvid4000_3router_i2v_a14b_4x48g.sh
```

默认值为 `conditioning_mode=0`、`WAN_TRAIN_MODE=cond_only`、`NPROC_PER_NODE=8`。训练三路 MQ embeddings、共享 Connector，以及两个原生 Wan DiT 中参数名命中条件侧规则的参数。可用第一个位置参数切换 mode 1，也可通过环境变量覆盖卡数、Wan 训练范围和 activation checkpoint：

```bash
# mode 1，其他默认值不变
bash train/train_openvid4000_3router_i2v_a14b_4x48g.sh 1

# 只训练 MQ/Connector，并显式关闭 Wan activation checkpointing
WAN_TRAIN_MODE=frozen WAN_ACTIVATION_CHECKPOINTING=0 \
  bash train/train_openvid4000_3router_i2v_a14b_4x48g.sh 0
```

文件名中的 `4x48g` 为兼容旧提交命令而保留，不代表当前默认 world size。`cond_only` 仍有较高显存开销，建议先用少量 step 做 8×48GB 真机峰值验收。

## Connector FSDP

24 层共享 Connector 当前使用 block-level FSDP `FULL_SHARD`，外层 projection 也由 Connector FSDP root 分片；冻结 Qwen 保持复制。8 卡下 Connector 参数、梯度和 Adam 静态状态由约 12.2 GiB/卡降至理想约 1.53 GiB/卡，预计节省约 10.7 GiB/卡。

route MQ table 和 mode 1 mapper 保持复制：初始化时显式从 rank 0 广播，backward 后显式做平均 all-reduce，以保持旧 DDP 的参数初始化和全局平均梯度语义。checkpoint 时只汇集 Connector，不汇集冻结 Qwen，并继续输出可移植的 `mq_qwen_connector_trainable.pt`。正式 500 step 前建议先用 8 卡执行 5 step 且在第 5 step 保存，验证 AdamW 峰值和 full-state checkpoint。

## 数据读取保护

OpenVid 默认先由 rank 0 预清洗候选池，其他 rank 读取同一缓存。NAS 视频探测和解码使用可终止的 `ffprobe/ffmpeg` 子进程，默认 probe 超时 8 秒、41 帧解码超时 30 秒、每次取样最多尝试 4 个视频，避免无超时的 OpenCV 读取让单个 rank 永久掉队。分布式 process-group timeout 默认为 300 秒。

训练默认不生成或保留 `checkpoint-before-training`；若输出目录中存在旧的同名目录，rank 0 会在 step 1 前清理。每 `save_steps` 的常规 checkpoint 和最后一步 checkpoint 不受影响。
