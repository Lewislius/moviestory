# MovieStory 4×48GB 双模式训练与随机参考帧方案（2026-08-03）

## 1. 最终实现

本次新增了一套独立训练实现，没有覆盖原来的 `train_3router_planner_wan.py`、`train_openvid4000_3router.sh` 和 2×96GB 配置。

新增入口：

- `train_3router_wan_4x48g.py`：4 卡分布式训练主程序。
- `train_openvid4000_3router_4x48g.sh`：接受序号 `0` 或 `1` 的启动脚本。
- `train_openvid4000_3router_4x48g.yaml`：Determined 4×48GB 资源配置。
- `four_gpu_training/conditioning.py`：模式 0/1 条件合成及 MQ→T5 映射模块。
- `four_gpu_training/data.py`：从完整视频随机读取参考帧。
- `four_gpu_training/random_reference.py`：随机参考帧的 Wan clean-prefix 注入。
- `four_gpu_training/distributed.py`：Wan FSDP、等效全局 batch 采样器。
- `scripts/prepare_openvid_subset.py`：在建子集时预先检查帧数、时长、首帧解码和 caption token 数，确保进入 Dataset 后仍为 4000 条。
- `tests/test_four_gpu_training.py`：CPU 单元测试。

启动方式：

```bash
# 模式 0：处理后的 MQ 完全替代 T5 条件
bash /home/liuzhirui/Project/MovieStory/code/train_openvid4000_3router_4x48g.sh 0

# 模式 1：映射后的 MQ 与提示词 T5 token 拼接
bash /home/liuzhirui/Project/MovieStory/code/train_openvid4000_3router_4x48g.sh 1
```

Determined 配置默认读取 `MOVIESTORY_CONDITIONING_MODE=0`。运行模式 1 时将其改成 `1` 即可。

## 2. 模式 0 和模式 1 的精确定义

### 模式 0：MQ 替代 T5

条件链为：

```text
caption + 随机参考图
        ↓
三路 Qwen/MQ（role、action、global）
        ↓
共享 24 层 Connector
        ↓
256 × 4096 MQ tokens
        ↓
直接作为 Wan context（不拼接 T5）
```

Wan 的 `text_len` 为 256。参考图片同时还会通过 VAE 编码，以独立 clean reference prefix 的形式注入 Wan；因此“MQ 替代 T5”不代表移除参考图的 Wan 侧条件。

此模式不创建 MQ→T5 mapper，避免对现有 MQ-only 语义做额外变换。

### 模式 1：mapped MQ + T5

条件链为：

```text
三路 MQ → Connector → MQToT5Mapper ─┐
                                      ├→ [mapped MQ, raw frozen-T5 tokens] → Wan
caption → frozen UMT5 ────────────────┘
```

拼接顺序固定为 `[mapped MQ, T5]`。最大条件长度是 `256 + 512 = 768`，每个样本实际使用的 T5 token 可以短于 512，其余位置由 Wan 原始逻辑补零。

T5 始终冻结并放在 CPU。模式 1 只对 mapped MQ 做 RMS 尺度匹配，原始 T5 token 不会被缩放或修改。

## 3. MQ 和 T5 是否处于同一语义空间

结论：**不是同一个语义空间。**

二者最后一维都是 4096，只说明 Wan cross-attention 的接口尺寸相同：

- T5 token 来自 UMT5 文本编码器，其坐标基、方向关系和统计流形是 Wan 预训练时看到的文本条件。
- MQ token 来自 Qwen3-VL 隐状态，再经过独立训练的 24 层 Connector。即使输出也是 4096 维，其基底和语义方向也不自动等于 UMT5。
- 仅做 RMS 范数匹配只能修正整体尺度，不能修正语义方向、协方差结构和 token 关系。

因此模式 1 增加了可训练的 `MQToT5Mapper`：

```text
RMSNorm(4096)
  → Linear(4096, 1024)
  → SiLU
  → Linear(1024, 4096)
  → residual connection
```

该模块使用 FP32 主权重，接近恒等映射初始化。这样不会在训练开始时突然破坏已有效的 MQ 特征，但能通过最终的视频去噪损失学习到 Wan/T5 条件流形所需的语义校正。Mapper 已作为 MQ encoder 的子模块保存到 `mq_encoder_full.pt`、`mq_encoder_trainable.pt` 和 safetensors 文件中。

没有增加额外的 T5 对齐 loss；总 loss 仍严格只有 ground-truth video velocity denoising MSE。这避免了硬件迁移时改变原训练目标。Mapper 的梯度直接来自 Wan 视频去噪目标，是功能空间上的对齐。

## 4. 2×96GB 到 4×48GB 的训练等效关系

原训练配置：

```text
单训练进程
micro batch = 1
gradient accumulation = 8
effective batch = 8
500 optimizer steps × 8 = 4000 sample draws
```

新训练配置：

```text
4 个 torchrun rank
每 rank micro batch = 1
每 rank gradient accumulation = 2
global effective batch = 4 × 1 × 2 = 8
500 optimizer steps × 8 = 4000 sample draws
```

`GlobalBatchEquivalentSampler` 先产生唯一的全局样本序列，再把每一步的 8 条样本按 2 条/rank 分发。每个 sampler 元素还携带拓扑无关的 `global_draw_id=0..3999`。子集准备阶段先使用与训练一致的 49 帧、0.5～20 秒、512 caption token 条件过滤，并检查首帧可解码；DataLoader 构造时再次强制 `len(dataset)==4000`。因此每条数据恰好使用一次，少一条都会在训练开始前报错，不会重复其他样本补足。

准备脚本用 8 个进程并行做视频探测；如果旧的 `openvid_first4000/manifest.json` 存在，会先复用其中的候选记录，再按原 CSV 顺序向后补足被过滤的条目。v2 manifest 生成后，启动脚本设置 `WAN_DATA_PRECLEAN=0`，避免 4 个 rank 各自串行重复解码 4000 个视频；最终长度仍由 DataLoader 的 4000 条硬检查兜底。

下列训练量保持不变：

- 256 个 MQ token，三路布局仍为 96/96/64。
- 24 层共享 Connector。
- learning rate `1e-5`、warmup 50、optimizer step 500。
- 每个 optimizer step 的全局有效 batch 为 8。
- 视频帧数 49、`max_area=262144`。
- Wan 仍为 `cond_only` 训练。
- 主目标仍只有视频 velocity MSE，不加入 T5 alignment、图像保持或 Wan distillation loss。
- Wan 使用 BF16 前向和 FSDP FP32 梯度归约；MQ/Connector 的 DDP 梯度按其参数 dtype 归约，与其原始参数精度一致。

为减少硬件拓扑导致的随机差异，参考帧、caption/MQ dropout、Qwen dropout、flow timestep 和 diffusion noise 都由 `global_draw_id + video_path + seed` 派生。某条数据分到哪一张 GPU 不会改变它的随机条件。梯度裁剪也不是在各 rank 上分别裁剪：代码只计一次 DDP 已复制的 MQ 梯度，对 FSDP Wan shard 做全局平方和归约，然后按完整模型的同一个 L2 norm 系数裁剪。

需要准确理解“训练效果无差异”的边界：本实现保证的是**相同训练目标、全局 batch、样本与样本随机条件、梯度裁剪和超参数语义**。分布式 all-reduce 的浮点加法顺序与单进程不同，CUDA kernel 也可能存在非确定性，所以不能科学地声称 checkpoint 逐 bit 相同。随机参考帧和模式 1 mapper 本身也是本需求指定的功能变化，它们会有意改变旧版首帧训练的结果；4×48GB 的硬件拆分不会额外改变优化目标。

## 5. 48GB 显存保护方案

### 5.1 Wan DiT 四路 FULL_SHARD

Wan DiT 的参数、梯度和优化器相关状态通过 4-rank FSDP `FULL_SHARD` 分片，不再让每张卡保存完整的 Wan 训练状态。

使用 `use_orig_params=True` 很重要：原 trainer 在 Wan 包装之后按参数名选择 `cross_attn`、`text_embedding`、`time_projection`、`modulation` 等 `cond_only` 参数。如果使用默认 flattened parameter，参数名和独立 `requires_grad` 状态会丢失，可能导致 cond-only 选择为空或训练错误。

### 5.2 激活与静态模型内存

- Wan transformer block 使用 non-reentrant activation checkpointing。
- Qwen/MQ Connector 使用原代码已有的 gradient checkpointing。
- UMT5 冻结并常驻 CPU，不占 GPU 权重显存。
- Qwen 输入 embedding 冻结；只有三组独立 FP32 MQ table、Connector、模式 1 mapper 及 Wan cond-only 参数训练。
- 使用 BF16 前向与 FSDP mixed precision。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` 减少碎片。
- 每步主动清理无用 CUDA cache，并记录峰值显存。
- 启动时验证恰好有 4 个 rank、每张卡至少 44 GiB 硬件容量且模型加载前至少 42 GiB 空闲，避免与其他进程争抢显存。
- 普通局部梯度裁剪已替换为 DDP+FSDP 完整参数空间上的全局裁剪，避免 rank 间更新不一致。
- 任意 OOM、解码错误或非有限 loss 都会立即中止整个 torchrun；不会跳过 micro-batch 后继续生成一个表面成功但不等效的 checkpoint。
- 不通过降低分辨率、减少帧数、减小 Connector 或改变全局 batch 来“躲避”OOM。

配置对当前 49 帧、262144 像素面积、Wan TI2V-5B、Qwen3-VL-2B 和 24 层 Connector 留出了分片余量。若实际节点的软件版本或其他进程占用导致 OOM，任务会立即失败，不会跳样继续。当前代码检查环境没有可用 GPU，因此最终物理峰值仍必须在目标 4×48GB 节点用真实 step 验收；在未运行目标 GPU 的情况下，无法诚实地给出“任何驱动/任何占用都绝对不 OOM”的承诺。

## 6. 随机参考帧的实现

旧数据集只读取视频开头连续 49 帧，并把处理后的第 0 帧同时作为 `ref_image` 和 `mq_ref_image`。

新数据集保持目标视频 tensor 完全不变，另外打开原始完整视频：

1. 读取完整视频的总帧数。
2. 使用 `seed + video_path + global_draw_id` 的稳定哈希，在 `[0, total_frames-1]` 中抽取索引。
3. 对该位置做随机 seek；个别 codec seek 失败时尝试多个稳定候选位置。
4. 将参考帧 resize 到目标训练视频的 H×W。
5. 该图片作为始终存在的 `ref_image` 注入 Wan，并根据 image dropout 决定是否也作为 `mq_ref_image` 注入 Qwen/MQ。

训练日志和 batch 中记录：

- `reference_frame_index`
- `reference_total_frames`
- `reference_frame_ratio`
- `moviestory_reference_is_target_first_frame`

caption dropout、MQ image dropout 和 joint-null dropout 也改成稳定样本哈希；同一 seed 下，Qwen dropout、flow timestep 和 diffusion noise 也绑定同一个 sample seed，避免仅因数据分配到不同 rank 而改变随机训练条件。joint-null 时 caption 和 MQ 图片同时置空，但 Wan 的随机参考图仍保留，与现有 CFG 条件契约一致。

## 7. 随机参考图是软锚定还是强绑定

应分成两个层面回答，不能简单只叫“软”或“强”：

### 语义/时间关系：软锚定

随机帧不再被定义成目标视频的时刻 0。它可能来自中段或尾段，因此模型学习的是“这个角色/外观/场景与目标视频有关”，而不是“输出第 0 帧必须重建这张图”。从参考图和生成视频时间轴的关系看，它是**软语义锚定、弱时间绑定**，确实有利于降低对首帧位置的过拟合并增强参考图位置泛化。

### Wan 注入机制：仍是强 clean-prefix 条件

参考 latent 在 Wan 输入中仍然：

- 以单独 prefix slot 注入；
- timestep 固定为 0；
- 在每次 Wan forward 前恢复为无噪 clean latent；
- prefix 不参与 denoising loss。

因此从数值注入机制看，它仍是**硬 clean-prefix 条件**，而不是旧 `animate_like` 那种按 alpha 混合到 noisy target 首帧的软像素锚定。

最准确的描述是：**硬参考条件 + 软时间锚定**。

### 为什么不能再删除目标首 latent

旧强首帧代码会删除目标视频的第一个 latent slot，因为 reference 正是目标第 0 帧，保留两份会重复。

现在 reference 是任意随机帧，它通常不等于目标第 0 帧。如果仍删除目标首 latent，就会无监督地丢掉真实视频开头，并错误改变目标时长。因此新 mixin 明确保留完整 target latent，仅在前面额外加入一个 reference prefix，loss 只计算完整 target 部分。

## 8. 训练流程

一次训练 micro-step 的流程如下：

1. 等效全局 sampler 为当前 rank 提供本 optimizer step 对应的 2 条本地样本及其全局 draw id。
2. 数据集保持原 49 帧目标 clip，同时从完整视频随机读取一张 reference。
3. role 路由读取 reference image，action 路由读取 caption，global 路由读取 image+caption。
4. 三路 Qwen hidden state 按 96/96/64 拼接，经过同一个 24 层 Connector。
5. 模式 0 直接得到 256 个 Wan context token；模式 1 先经过 MQ→T5 mapper，再与 frozen T5 prompt token 拼接。
6. 目标视频完整 VAE 编码；随机 reference 另行 VAE 编码成 1 个 clean prefix slot。
7. reference prefix 使用 timestep 0，target slots 使用随机 flow-matching timestep。
8. FSDP Wan 前向；loss 只计算 target video velocity MSE，reference prefix 不计入 loss。
9. 每 rank 累积 2 次，DDP/FSDP 对 4 rank 梯度平均；在完整 DDP+FSDP 参数空间计算一次全局 L2 norm 后执行 optimizer step。
10. 保存 MQ/Connector/mapper、可移植的 Wan cond-only 权重以及 `four_gpu_training_config.json`；上游 bundle 也会记录 optimizer/scheduler，但它不作为跨拓扑完整 FSDP resume 的承诺。

## 9. 验证命令和已有结果

CPU 自检：

```bash
cd /home/liuzhirui/Project/MovieStory/code
python train_3router_wan_4x48g.py \
  --four_gpu_check_only \
  --expected_world_size 4 \
  --global_effective_batch 8 \
  --expected_train_samples 4000

python -m unittest discover -s tests -p 'test_four_gpu_training.py' -v
```

当前已验证：

- 新增 Python 文件全部通过 `py_compile`。
- 7 个 4×48GB 专项单元测试全部通过。
- 模式 0 输出纯 MQ。
- 模式 1 输出 `[mapped MQ, T5]`，mapper 反向梯度有效。
- 4 个 rank 分别得到 1000 条，总 draw=4000、unique=4000。
- 重建后的 `global_draw_id` 严格连续为 0～3999，样本随机数不随 rank 拆分变化。
- DDP MQ + FSDP Wan 混合参数的全局梯度范数和裁剪系数计算正确。
- 随机 reference 只替换参考条件，不修改 target video tensor。
- clean reference prefix 注入正确，target slot 没有被删除。
- 已在真实 OpenVid 数据上生成 v2 子集：旧 4000 候选中替换了 14 条时长超限、36 条 caption 超 512 token、3 条不足 49 帧的视频；最终 manifest、CSV、软链接和实际 `WanVideoDataset` 长度均为 4000。
- 已真实读取一条样本：target 保持 `(3,49,512,512)`，随机参考来自完整视频第 49/64 帧并 resize 为 `(512,512)`，不是默认 target 第 0 帧。

目标环境必须再验证：

- 4 张卡的 `max_memory_allocated` 峰值。
- FSDP full-state checkpoint 在目标 PyTorch/CUDA 版本上的导出耗时。
- 模式 0/1 各自至少 2～5 个真实 optimizer step 的 loss、route gradient 和 mapper gradient 日志。

## 10. Checkpoint 与推理注意事项

每个 checkpoint 会额外包含 `four_gpu_training_config.json`（格式版本 `moviestory_4x48g_random_reference_v2`）。推理端应先读取其中的 `conditioning.mode`：

- 模式 0：按原 MQ-only 方式注入 256 tokens。
- 模式 1：必须实例化并加载 checkpoint 内的 `mq_to_t5_mapper`，重新编码 prompt T5，并按 `[mapped MQ, T5]` 顺序拼接。若仍用旧 MQ-only 推理脚本，mapper 和 T5 分支不会生效，训练/推理条件将不一致。

随机参考训练不再承诺生成视频第 0 帧逐像素等于参考图。如果任务需要严格的首帧续写，应继续使用旧的真实首帧 strong-binding 数据契约；如果任务目标是角色/外观参考和跨位置泛化，则应使用本次随机参考方案。

当前上游 trainer 的 `--resume_mq_encoder_path` 是“加载 MQ/Connector/mapper 权重重新开始优化”，不是恢复 optimizer step 的完整分布式断点续训。checkpoint 中的 MQ/mapper 和 Wan cond-only 权重可用于推理；不要把 rank 0 保存的混合 optimizer state 当成可跨拓扑恢复的完整 FSDP optimizer checkpoint。
