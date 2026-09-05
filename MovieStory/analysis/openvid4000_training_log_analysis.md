# OpenVid4000 三路 Router 训练日志与续训机制分析

## 1. 分析范围

本报告基于以下文件与当前实际代码路径：

- 日志：`log/experiment_53696_trial_53978_logs.txt`
- 逐步指标：`code/checkpoint/three_router_mq256_conn24_legacysoft_openvid4000_steps500/logs/train_metrics.jsonl`
- 训练配置：`code/train_openvid4000_3router.yaml`
- 启动脚本：`code/train_openvid4000_3router.sh`
- 包装训练器：`code/train_3router_planner_wan.py`
- 上游实际训练循环：
  `/home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_metaquery_wan.py`
- step 240 与最终 checkpoint 中的 Safetensors 权重

这里所说的 `loss` 是损失值，不是百分比意义上的“损失率”。

## 2. 先给结论

### 2.1 训练不是完全没有下降

单步 loss 一直波动，但平滑趋势在前半程有明显下降：

| optimizer step | 50 步 loss 均值 | 标准差 | 区间最小/最大 | 区间末 EMA |
|---|---:|---:|---:|---:|
| 1-50 | 0.5017 | 0.1039 | 0.3396 / 0.6794 | 0.4831 |
| 51-100 | 0.3891 | 0.0694 | 0.2819 / 0.5762 | 0.3995 |
| 101-150 | 0.3616 | 0.0755 | 0.2429 / 0.6353 | 0.3728 |
| 151-200 | 0.3707 | 0.0641 | 0.2526 / 0.5954 | 0.3734 |
| 201-250 | 0.3592 | 0.0775 | 0.2117 / 0.5672 | 0.3572 |
| 251-300 | 0.3487 | 0.0568 | 0.2571 / 0.5344 | 0.3475 |
| 301-350 | 0.3508 | 0.0821 | 0.2396 / 0.6188 | 0.3377 |
| 351-400 | 0.3430 | 0.0685 | 0.2232 / 0.5366 | 0.3368 |
| 401-450 | 0.3374 | 0.0666 | 0.2048 / 0.5026 | 0.3398 |
| 451-500 | 0.3493 | 0.0709 | 0.2517 / 0.5283 | 0.3526 |

全部 500 步的原始 loss 线性斜率约为 `-2.32e-4/step`。EMA 从
`0.6664` 降到 `0.3526`，最低到过 `0.3244`。因此准确描述应是：

> 前约 100-150 步快速下降，此后在约 0.34-0.37 附近高方差平台化，
> 最后 50 步略有回升；仅凭训练 loss 不能判断泛化是否改善。

最后一步 `loss=0.5034` 只是一个高于近期均值的随机 batch，不能代表整个训练
退化。相反，只看最低值 `0.2048` 也不能证明训练已经很好。

### 2.2 三路参数确实更新了

证据是充分的：

1. step 1 因 warmup 初始学习率为 0，显示一次 `WAIT_LR`。
2. step 2-500 共 499 次均为 `[3-ROUTER][UPDATE] status=PASS`。
3. 三路每步都有非零累计梯度，绝大多数参数元素发生变化，`stale_steps` 和
   `no_grad_steps` 全程为 0。
4. 最终相对本次进程初始值的参数 RMS 位移分别为：

| 路由 | `initial_delta_rms` at step 500 |
|---|---:|
| role | `1.4809e-4` |
| action | `1.2647e-4` |
| global | `1.3672e-4` |

5. 直接比较 checkpoint-240 与 checkpoint-final：

| 参数 | 240 -> 500 的 delta RMS | 最大绝对变化 | 变化元素比例 |
|---|---:|---:|---:|
| role MetaQuery table | `3.0937e-5` | `2.3436e-4` | 99.9990% |
| action MetaQuery table | `3.2654e-5` | `1.8083e-4` | 100% |
| global MetaQuery table | `3.2049e-5` | `2.0377e-4` | 100% |

`cond_only` Wan 参数也在更新。例如 step 240 到 500：

| Wan 参数样本 | delta RMS | 变化元素比例 |
|---|---:|---:|
| `blocks.0.modulation` | `8.1502e-5` | 100% |
| `text_embedding.0.bias` | `3.4027e-5` | 100% |
| `time_projection.1.bias` | `5.9815e-5` | 100% |

因此“没有发生 optimizer update”不是 loss 平台化的原因。

### 2.3 同一次运行会连续使用更新后的三路参数

会。每个 optimizer step 的流程是：

1. 8 个 micro-batch 累积梯度；
2. 对当前参数执行一次 AdamW update；
3. 下一轮 forward 直接读取已更新的
   `route_metaquery_embeddings.role/action/global`；
4. 不会在每个 step 重新初始化这三张参数表。

日志中的 `initial_delta_rms` 从 0 持续增加也证明了这一点。

### 2.4 重新启动一个新训练默认不会继承上一次结果

当前 `train_openvid4000_3router.sh` 没有传
`--resume_mq_encoder_path`。历史 checkpoint 的 `training_args.json` 也记录：

```text
resume_mq_encoder_path: null
```

新进程会重新从 Qwen 对应 MetaQuery token rows 初始化三路参数表，再创建新的
Connector/Wan 训练状态。日志中也没有“已加载初始权重”记录。因此，单独重新
运行脚本属于从头训练，不是接着上次训练。

### 2.5 当前的 `resume` 也不等于完整续训

即使以后手工传入：

```bash
--resume_mq_encoder_path <checkpoint-final>
```

当前代码也只会加载 MQ encoder state，其中包括 Connector 和三路 MetaQuery
tables。它不会恢复：

- `wan_dit_trainable.pt` 中本次训练过的 12.12 亿个 Wan `cond_only` 参数；
- `optimizer.pt` 中 AdamW 的一阶/二阶动量；
- `scheduler.pt` 中学习率阶段；
- global step；
- RNG、DataLoader shuffle 和数据消费位置。

所以它准确地说是 MQ encoder 权重热启动，不是训练状态 resume。新任务的
optimizer step 仍从 0 开始，warmup 也从头开始。

## 3. 本次训练实际执行了什么

### 3.1 step、batch 和数据量

实际参数为：

- Dataset：OpenVid 前 4000 条可解析记录；
- DataLoader batch size：代码硬编码为 1；
- gradient accumulation：8；
- optimizer steps：500；
- 理论 micro-batch 数：`500 × 8 = 4000`；
- skip：OOM、普通错误、总跳过数均为 0。

因此这是大致一遍数据的一次训练。README 中旧的“2000 steps、累积 2”描述与
本次真实运行不符，现已更正文档和脚本注释。

### 3.2 实际优化目标

每个样本先经 VAE 得到 `x0`，随机采样：

```text
t ~ Uniform(0, 1)
noise ~ Normal(0, I)
x_t = (1 - t) * x0 + t * noise
target_velocity = noise - x0
loss = MSE(Wan(x_t, t, MQ_context), target_velocity)
```

一个 optimizer step 的 `train/loss_step` 是 8 个 micro-batch loss 的平均。

虽然配置包含 `--enable_t5_alignment`，当前实际 `_compute_loss` 明确把：

```text
total_loss = denoise_loss
```

并把全部 alignment / image preserve / function distillation loss 设为 0。
所以这次训练没有 T5 对齐辅助损失。启用的是 MQ/T5 RMS 探测和动态缩放，不是
T5 alignment loss。配置名称在这里容易误导。

### 3.3 哪些参数在训练

日志报告：

| 参数组 | 可训练参数量 |
|---|---:|
| Connector + MQ embeddings | 16.367 亿 |
| Wan DiT `cond_only` | 12.124 亿 |
| 合计 | 28.491 亿 |

三路 MetaQuery table 使用独立 FP32 master parameter，并被移入
`weight_decay=0` 的 optimizer group。其他 MQ/Connector 与 Wan 组使用 AdamW，
默认 `weight_decay=0.1`。

“cond_only”并不代表只训练几个标量；本次仍训练了约 12.12 亿 Wan 参数。相对
4000 条样本，这是非常大的可训练容量。

## 4. 为什么单步 loss 波动明显

### 4.1 每一步不是在测同一道题

DataLoader 使用 `shuffle=True`，每个 optimizer step 是 8 个不同视频。视频
内容、运动强度、构图、压缩质量和 caption 难度都不同。训练集只走约一遍，
同一个 step 的 loss 不能与上一个 step 做严格的同样本比较。

这也是必须同时看 `loss_ema`、窗口均值和固定验证集的原因。

### 4.2 flow-matching timestep 和噪声每次随机

每个 micro-batch 都重新采样 `t` 和高斯噪声。同一视频在不同 `t/noise` 下的
MSE 难度也不同。当前日志没有记录 8 个 micro-batch 的 timestep 分布，无法
把 loss 峰值准确归因到具体噪声区间。

### 4.3 条件 dropout 额外增加方差

配置中：

```text
null_caption_prob = 0.1
null_image_prob = 0.1
```

所以训练会随机出现有文无图、有图无文、图文均有等不同条件强度。日志共有
4000 次 MQ/T5 probe，其中 399 次 T5 RMS 降到约 `0.025684`，比例
`9.975%`，与 10% caption dropout 高度吻合。这些样本的动态缩放还触发了
`clip_min=0.03`，与普通 caption 的约 `0.078-0.091` 缩放不同。

### 4.4 首帧软锚定随随机 timestep 变化

本次使用 `legacy_t2v + animate_like`，且：

```text
alpha0 = 1.0
warmup_ratio = 1.0
```

首 latent 的锚定系数因此会随随机 `t` 在接近 0 到接近 1 之间变化。日志中的
`ref_anchor_alpha_mean` 最小约 `7.75e-7`、最大约 `0.99996`。

更重要的是，代码在生成标准 flow-matching `x_t` 后又修改首 latent，使其靠近
参考 latent，但 target 仍是原始 `noise - x0`。这是一种条件注入近似，不再是
严格处于原始线性 flow 轨迹上的输入，可能增加首帧部分的目标方差。

### 4.5 梯度裁剪频繁

`max_grad_norm=1.0`，而日志的 `grad_norm` 是裁剪前总范数。500 步中有 258 步
超过 1，早期最高 `15.50`。这说明裁剪确实经常参与更新，避免了大梯度直接
进入 optimizer；同时也说明不同 batch 的有效更新幅度被强烈压平，可能造成
早期学习不稳定、后期改善缓慢。

原始 loss 与 grad norm 的相关系数约为 `0.527`，高 loss batch 往往也产生较大
梯度，但相关不代表因果。

### 4.6 学习率后期已经很小

调度为 50 步 warmup + cosine decay，峰值 `1e-5`，最低比例 0.01，即
`1e-7`。后约 28 步已处于 `1e-7` floor。此时三路每步 update RMS 约
`2e-8` 到 `3e-8`，相对更新只有约 `2e-6` 到 `4e-6`。

所以最后阶段 loss 继续随机波动，但参数已经几乎不再大幅移动。最后 50 步均值
从此前 `0.3374` 回到 `0.3493`，并不能靠这么小的学习率快速纠正。

### 4.7 MQ/T5 原始范数长期错配

4000 个 micro-batch 全部出现：

```text
MQ RMS ≈ 1.0
T5 RMS ≈ 0.08（caption dropout 时约 0.0257）
raw ratio ≈ 11-39
```

因此日志打印了 4000 次 `[MQ-NORM][WARN]`，也做了 4000 次动态缩放。普通
caption 的 post ratio 被拉回约 1，caption dropout 样本受下限限制，只能到
约 1.168。

这不表示传入 Wan 的最终 MQ 一直大 12 倍，因为 `[ADJUST]` 已经缩放；但它表示
Connector 的原始输出分布与 Wan 熟悉的 T5 分布并不自然匹配，并且每个样本都
依赖动态归一化。这个机制可能掩盖尺度问题，并让空 caption 与普通 caption
走不同的缩放分支。

### 4.8 三路输出仍然高度相似

前 50 步到最后 50 步的平均 cosine：

| cosine | 前 50 步 | 后 50 步 | 全程范围 |
|---|---:|---:|---:|
| role-action | 0.9935 | 0.9831 | 0.9757-0.9956 |
| role-global | 0.9852 | 0.9562 | 0.9418-0.9966 |
| action-global | 0.9929 | 0.9823 | 0.9720-0.9962 |

role/global 出现了一些分化，但总体仍接近共线。当前 Router planner 本身只是
identity split，三路最终进入同一个共享 Connector；没有显式任务监督要求三路
分别表达人物、动作、全局信息。因此“输入模态隔离”不自动保证学到语义专门化。

### 4.9 可训练规模与数据规模不匹配

28.49 亿可训练参数只看约 4000 个样本，且没有独立验证集。初期 loss 快速下降
后平台化，可能同时包含：

- Connector/Wan 条件路径快速适应；
- 数据和随机 timestep 的不可约方差；
- 小数据下对不同 batch 的更新方向冲突；
- Wan 大规模条件层与新 Connector 同学习率联训带来的相互漂移；
- 对训练数据的拟合不等价于泛化改善。

### 4.10 没有验证 loss，无法回答“是否真正训练成功”

YAML 的 searcher 声明：

```yaml
metric: validation_loss
```

但训练代码没有 validation loop，也没有向 Determined 上报
`validation_loss`。日志只有训练 loss。因此目前能确认的是：

- optimizer 正常运行；
- 参数真实变化；
- 训练 loss 的平滑趋势下降后平台化；

不能确认的是：

- 未见视频上的 loss 是否下降；
- 生成质量、人物一致性和动作一致性是否提升；
- 三路语义是否真正解耦；
- 是否过拟合。

## 5. 日志中各指标的准确含义

### 5.1 `[Step N/500]` 和进度条指标

| 日志名 | JSONL/W&B 名 | 含义 | 本次如何看 |
|---|---|---|---|
| `loss` | `train/loss_step` | 8 个成功 micro-batch 的总 loss 平均 | 本次等于 denoise |
| `denoise` | `train/loss_denoise` | flow-matching velocity MSE | 真正优化的主目标 |
| `align` | `train/loss_align_total` | T5 等辅助对齐 loss 总和 | 本次始终 0，未参与优化 |
| `func` | `train/loss_align_wan_func` | Wan 函数蒸馏 loss | 本次始终 0 |
| `avg` | `train/loss_ema` | `0.95*旧EMA + 0.05*当前loss` | 比单步 loss 更适合看趋势 |
| `lr` | `train/lr` | scheduler step 后 MQ 主组学习率 | 50 步升到 1e-5，再衰减到 1e-7 |
| `grad_norm` | `train/grad_norm` | 裁剪前全部可训练参数总梯度范数 | 大于 1 时实际会被裁剪 |
| `dt` | `train/step_time_sec` | 一个 optimizer step 的耗时 | 均值约 42.37 秒 |
| `samp/s` | `train/samples_per_sec` | 成功样本数 / step 耗时 | 均值约 0.192 |
| `param_delta` | `train/param_sample_abs_delta_mean` | 固定抽样参数相对进程初值的平均绝对位移 | 最终约 `1.122e-4` |
| `skip(oom/err/total)` | 三种 skip counter | OOM、其他错误、总跳过 step | 本次全为 0 |

`loss_step` 是 step 内 8 个 micro-batch 的平均，但下面不少 probe 只保留第 8 个
micro-batch 的最后值。两类指标不能直接当成同一统计口径。

### 5.2 其他训练状态指标

| 指标 | 含义 |
|---|---|
| `train/step` | 已完成的 optimizer update 次数 |
| `train/backward_ok_microbatches` | 本 step 成功 backward 的 micro-batch 数，本次固定 8 |
| `train/effective_batch_samples` | 成功 micro-batch × batch size，本次固定 8 |
| `train/skipped_step_count` | 因异常未执行 optimizer update 的 step 累计数 |
| `train/oom_skip_count` | 其中 OOM 跳过数 |
| `train/error_skip_count` | 其中非 OOM 错误跳过数 |
| `train/trainable_param_count` | optimizer 可训练参数总数，本次 2,849,085,440 |
| `train/param_sample_norm` | 固定抽样参数当前 L2 norm，不是全模型 norm |
| `train/param_sample_norm_delta_ratio` | 抽样参数 norm 相对初始 norm 的变化比例 |
| `train/param_sample_l2_delta` | 抽样参数当前值与初始值之差的 L2 norm |

step 1 的 `param_delta=0` 是正常的：optimizer pre-hook 捕获到的初始学习率为 0，
所以第一次 optimizer step 没改变参数。随后 scheduler 先推进并在通用日志中
显示 `train/lr=2e-7`。因此 step 1 会同时看到：

```text
[3-ROUTER][UPDATE] WAIT_LR ... lr=0
[Step 1] ... lr=2e-7
```

前者是本次 update 实际使用的 pre-step LR，后者是 scheduler 更新后供下一步
使用的 LR，两者相差一个调度位置。

### 5.3 首帧与 conditioning 指标

| 指标 | 含义 | 本次值 |
|---|---|---|
| `train/ref_anchor_alpha_mean` | 最后一个 micro-batch 中实际应用锚定的平均 alpha | 随 t 在 0-1 波动 |
| `train/ref_anchor_applied` | 最后一个 micro-batch 中应用锚定的样本数 | B=1 时通常为 1 |
| `train/ref_anchor_mode_cfg` | 配置的锚定模式 | `animate_like` |
| `train/ref_anchor_mode_effective` | 本次最后 micro-batch 实际模式 | `animate_like` |
| `train/ref_anchor_effective_is_animate` | effective mode 是否 animate_like 的 0/1 标记 | 1 |
| `train/video_conditioning_mode_effective` | 实际视频 conditioning 路径 | `legacy_t2v` |
| `train/prefix_latent_slots` | Wan animate prefix slots 数 | 本次 0 |
| `train/target_latent_slots` | 49 帧经 VAE 后的 target latent slots | 本次 13 |
| `train/prefix_loss_dropped` | 是否/多少样本丢弃 prefix loss | 本次 0 |

`train/video_conditioning_mode_cfg` 当前错误地读取了 `dit_condition_mode`，所以
日志显示 `mq_only`；它不是实际 `train_video_conditioning_mode`。
`train/video_conditioning_mode_effective=legacy_t2v` 才是本次真实路径。

### 5.4 MQ/T5 范数指标

| 指标/日志 | 含义 |
|---|---|
| `mq_rms` / `train/mq_rms` | Connector 输出在动态匹配前的 token RMS |
| `t5_rms` / `train/t5_rms_probe` | 同 caption 的 T5 condition RMS |
| `ratio` / `train/mq_t5_rms_ratio` | 匹配前 `MQ RMS / T5 RMS` |
| `train/mq_norm_warn` | ratio 是否超出 `[0.25, 4.0]` |
| `applied_scale` / `train/mq_norm_match_scale` | 实际乘到 MQ feature 上的缩放 |
| `post_ratio` | 缩放后的估计 MQ/T5 ratio |
| `raw_target_scale` | 理论上的 `T5 RMS / MQ RMS` |
| `clip=[0.03,4.0]` | 动态缩放允许的上下限 |
| `loss_call` | `_compute_loss` 被调用次数，即 micro-batch 计数 |

JSONL 中这些值是该 optimizer step 最后一个 micro-batch 的 probe，不是 8 个
micro-batch 的均值。控制台则打印全部 4000 次调用。

### 5.5 `[3-ROUTER][DIAG]`

| 指标 | 含义 |
|---|---|
| `cos(role,action)` | role/action 路由 token 先沿 token 维平均、L2 normalize 后的 cosine |
| `cos(role,global)` | role/global 的 pooled cosine |
| `cos(action,global)` | action/global 的 pooled cosine |
| `rms=(...)` | 三路 Qwen 输出 seed 在 batch/token/hidden 上的 RMS |
| `mq_emb_grad=(...)` | 三张 MetaQuery table 最近一次 backward hook 的梯度 RMS |

这里的三路 RMS 约 2.9 是 Connector 之前的 Qwen hidden state RMS，不是
`train/mq_rms≈1.0` 的 Connector 输出 RMS。

由于 gradient accumulation 会让 hook 执行 8 次，`mq_emb_grad` 最终只保留
第 8 个 micro-batch 的梯度 RMS。要看本 optimizer step 真正用于更新的累计
梯度，应看下一节的 `g/step_grad_rms`。

### 5.6 `[3-ROUTER][UPDATE]`

每一路都输出：

| 简写 | W&B/JSONL 后缀 | 含义 |
|---|---|---|
| `p` | `param_rms` | update 后该参数表本身的 RMS |
| `g` | `step_grad_rms` | optimizer step 前、8 个 micro-batch 累积后的梯度 RMS |
| `d` | `step_update_rms` | 本次 `parameter_after - parameter_before` 的 RMS |
| `rel` | `step_update_relative` | `step_update_rms / before_param_rms` |
| `changed` | `step_changed_fraction` | 本次精确不相等的参数元素比例 |
| `d_init` | `initial_delta_rms` | 当前参数与本进程初始化参数之差的 RMS |
| `lr` | `optimizer_lr` | 本次 optimizer update 实际读取到的该组 LR |

其他只在 JSONL/W&B 中出现的字段：

| 指标 | 含义 |
|---|---|
| `update_expected` | `lr>0` 且累计梯度非零时为 1 |
| `update_applied` | 至少一个参数元素真实变化时为 1 |
| `stale_steps` | 应更新但没变化的连续 step 数 |
| `no_grad_steps` | LR 非零但无梯度的连续 step 数 |
| `router_optimizer_step` | tracker 观察到的 optimizer.step 次数 |
| `router_all_updates_applied` | 所有正 LR 路由都有梯度且真实变化时为 1 |

状态含义：

- `WAIT_LR`：所有路由学习率为 0，本次只有 step 1；
- `PASS`：所有应更新路由都有梯度且参数真实变化；
- `FAIL_NO_GRAD`：至少一路正 LR 但无梯度；
- `FAIL_STALE`：有梯度和正 LR但参数没有变化。

当前 patience 是 5；连续 5 步 no-grad 或 stale 会直接中止，而不是继续生成
无效 checkpoint。

### 5.7 checkpoint 与完成日志

- `Checkpoint 已保存`：MQ encoder、Wan 可训练子集、optimizer、scheduler 和
  指标文件被写入目录；
- `resources exited successfully with a zero exit code`：进程正常退出；
- 这只说明作业运行成功，不说明模型质量达标。

日志称保存过 `checkpoint-before-training`，但当前文件系统中该目录已经不存在。
现有 manifest 仍指向它。这使得当前无法再做精确的初始权重与最终权重逐 tensor
比较；不过 tracker 的过程证据及 step 240/500 的直接权重比较已经足以证明更新。

## 6. 建议的解决顺序（本报告未实现）

以下只是待审核方案，本次没有修改训练算法或续训实现。

### P0：先把“是否学会”变成可测问题

1. 建立固定 validation subset。
   固定视频、caption、参考图、timestep 和 noise seed，定期计算 validation
   denoise loss。另保留随机 validation loss，分别衡量模型变化和期望风险。
2. 真正上报 `validation_loss`。
   让 Determined YAML 的 searcher metric 与代码一致，同时上传 W&B。
3. 记录每个 step 内 8 个 micro-batch 的统计。
   至少包括 timestep mean/min/max、anchor alpha mean、text/image dropout
   数量、MQ/T5 RMS mean/min/max，而不是只保留最后一个 micro-batch。
4. 做生成质量验证。
   固定 prompt/reference/seed，定期生成短视频，评价人物一致性、动作可控性、
   首帧保持和时间一致性。仅靠 velocity MSE 无法判断三路语义质量。
5. 使用独立输出目录。
   当前重复执行同一脚本会复用同一 output dir，并以 append 模式继续写
   `train_metrics.jsonl`，但模型却从头初始化，容易把两个实验误当成一次。

### P1：实现真正的完整 resume

建议新增明确区分的两种模式：

- `warm_start_mq`：只载入 MQ encoder，global step 和 optimizer 重置；
- `resume_training`：严格恢复 MQ、Wan trainable state、optimizer、scheduler、
  global step、RNG 和数据位置。

完整 resume 还应：

1. 校验三路 token 数、hidden size、Connector 层数和 Wan train mode；
2. 对 missing/unexpected keys 采用白名单，不能只打印数量后继续；
3. 恢复后记录 checkpoint 来源、权重 hash、起始 step；
4. 用一个小测试证明 resume 前后的下一次 update 与不中断训练一致；
5. 明确 W&B 是创建关联的新 run，还是用固定 run ID 恢复同一个 run。

### P2：缩小联训范围并做受控对照

建议至少跑以下可比较实验，每组使用相同固定 validation：

1. Wan frozen，只训练 Connector + 三路 table；
2. Wan LoRA；
3. 当前 `cond_only`，但 `wan_lr_ratio` 显著低于 1；
4. 当前配置 baseline，即 `ROUTER_ENABLED=0`；
5. 三路配置与 baseline 的相同数据、seed、step 和 LR 对照。

优先验证“训练 12.12 亿 Wan 条件参数是否真的必要”。4000 样本下同时大规模
更新 Wan 与新 Connector，可能比只适配小模块更不稳定，也更难判断收益来源。

### P3：降低随机目标方差

可审核的候选方案：

1. 对 timestep 做分层/分桶采样，保证每个 optimizer step 覆盖低、中、高噪声；
2. 增大有效 batch，或在显存不变时增加 accumulation；
3. 记录并按 timestep 比较 loss，而不是只看混合总均值；
4. 评估 timestep loss weighting；
5. 统计裁剪比例，若长期过高，再评估降低 LR、分参数组 LR 或调整 clip norm。

不能直接根据一次训练就断言应调高学习率。后期 LR 很低是事实，但早期有 258
步超过 clip threshold，说明提高全局 LR 也可能放大不稳定。

### P4：处理 MQ/T5 范数机制

建议依次做：

1. 单独统计非空 caption 与空 caption 的 T5 RMS；
2. 将 raw MQ RMS、post-match MQ RMS 都写入指标；
3. 评估把 Connector norm 初始尺度设到更接近约 0.08，减少每样本动态缩放；
4. 对 null caption 使用明确的稳定目标尺度，而不是让 `clip_min` 决定；
5. 对比“固定缩放、动态缩放、关闭缩放”三组 validation 与生成质量。

不要只因为 4000 次 WARN 就直接关闭 match。当前 match 确实阻止了约 12-39 倍
的条件尺度直接进入 Wan，需要先有对照实验。

### P5：使首帧训练目标与推理模式一致

建议对照：

1. `legacy_t2v + animate_like` 当前模式；
2. `legacy_t2v + none`；
3. clean preserved reference slot 强绑定模式。

重点分别测首帧误差与后续帧误差。若继续使用软锚定，需要明确验证“修改过的
首 latent 输入 + 原始 velocity target”是否符合期望的条件 flow 目标。

### P6：验证三路是否真正专门化

高 cosine 本身不是绝对错误，但当前没有证据表明三路学到预期语义。建议：

1. role-only、action-only、global-only 条件消融；
2. 替换参考人物但保持动作文本，观察 role 路径敏感性；
3. 替换动作文本但保持参考图，观察 action 路径敏感性；
4. 记录 Connector 后的三路 cosine，而不只看 Connector 前 Qwen states；
5. 在确认任务指标后，再考虑轻量 decorrelation 或路由专门化约束。

不建议仅为了让 cosine 变小而添加正交 loss。数值分离不等价于语义分工。

## 7. W&B 中应该重点查看的面板

当前接入会逐 optimizer step 上传本地 `metrics` 字典。建议在 W&B workspace
重点组合以下面板：

1. `train/loss_step` 与 `train/loss_ema`；
2. `train/lr` 与 `train/grad_norm`；
3. 三路 `step_grad_rms` 与 `step_update_rms`；
4. 三路 `initial_delta_rms`；
5. `router_all_updates_applied`、`stale_steps`、`no_grad_steps`；
6. 三个 route cosine 与三路 RMS；
7. `mq_t5_rms_ratio` 与 `mq_norm_match_scale`；
8. `step_time_sec`、`samples_per_sec` 和 skip counters。

API key 应使用 `WANDB_API_KEY` 环境变量或平台 secret 注入，不能写进代码、
YAML 或命令行。W&B 官方文档也明确建议使用环境变量并避免把 key 提交到版本
控制或作为命令行参数传递：

- <https://docs.wandb.ai/models/track/environment-variables>
- <https://docs.wandb.ai/platform/app/settings-page/user-settings>

## 8. 建议的验收标准

下一轮实验不能只以“脚本正常退出”或“单步 loss 低于某阈值”为成功标准。
建议同时满足：

1. 所有路由 update 证据持续为 PASS；
2. 固定 validation loss 的 EMA 明显下降；
3. 随机 validation loss 没有持续恶化；
4. 固定生成样例在人、动作、全局语义上有可重复改善；
5. baseline 与三路实验使用完全相同的数据、seed 和训练预算；
6. 完整 resume 测试与不中断训练在下一步更新上等价；
7. W&B run、checkpoint、JSONL 和配置能一一对应，不混合不同进程的历史。

