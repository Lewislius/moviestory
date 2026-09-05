# MovieStory 纯噪声视频：训练到推理全流程诊断

> 分析日期：2026-07-31  
> 分析对象：`three_router_mq256_conn24_legacysoft_openvid4000_steps500/checkpoint-final`  
> 实际输出：`code/inference_outputs/openvid4000_3router/seed42_20260731-072618.mp4`

## 1. 结论摘要

这次“纯雪花”不是一个单一问题，而是至少两层独立故障叠加：

| 层级 | 结论 | 置信度 |
|---|---|---:|
| 视频落盘 | `_save_video` 少传了 batch 维，`[C,T,H,W]` 被要求 `[B,C,T,H,W]` 的 Wan 保存器错误解释 | 已证实 |
| 底层生成 | 将错误 MP4 可逆重排回正确的 49 帧后，底层画面仍是彩色噪声，不能只归因于保存维度 | 已证实 |
| 首帧条件 | 训练先构造标准 flow `x_t`，再改写首 latent，但仍使用原来的 velocity target；输入与监督不在同一 flow path 上 | 已证实为逻辑缺陷 |
| CFG | 推理用“长负面提示词 + 无图”的 MQ 作为无条件分支并设 `CFG=5`，训练却主要见自然 caption，真正“空文+空图”只约占 1% | 高概率主因 |
| 条件替换 | T5 被完全移除，DiT 只接收 MQ；RMS 匹配只对齐一个标量，不能保证 MQ 落入 Wan 已学会的 T5 条件流形 | 高概率主因 |
| 未接线功能 | CLI/配置显示 T5 alignment 已启用，但实际 loss 固定为纯 denoise MSE；T5 对齐、图像保持、函数蒸馏函数均未接入活动 loss | 已证实 |
| 训练充分性 | 仅 4000 个视频、约一遍数据，却联合训练约 28.49 亿参数，没有 validation 和训练中生成样例 | 高风险设计 |
| 审计可靠性 | 验证报告把 `Infinity/NaN` 当作 pass，图像对 Wan 输出零影响也被降级成 warning；没有逐步 latent/pred 数值监控 | 已证实 |

因此，当前 checkpoint “文件完整、参数确实更新、条件张量确实接入”并不等于“学到了可采样的 velocity field”。现有证据更符合：

1. MQ/DiT 条件路径在训练 loss 上有下降；
2. 但高噪声采样阶段的 velocity 预测没有达到可迭代生成的精度；
3. `CFG=5` 和错误首帧锚定进一步放大偏差；
4. 最终 VAE 解码的是仍处于噪声分布附近的 latent；
5. 保存器又把这个噪声视频错误重排成了 512 帧窄条，使视觉上更像雪花。

---

## 2. 直接取证

### 2.1 报告声称的输出与真实 MP4 不一致

推理验证报告声称：

```text
output_size = 512 x 512
frame_num   = 49
latent      = [48, 13, 32, 32]
```

但实际 `ffprobe` 结果是：

```text
codec_name=h264
width=512
height=64
nb_frames=512
duration=21.333333
```

脚本要生成的是 `49 帧 × 512×512`，实际却保存成 `512 帧 × 512×64`。这不是编码器轻微缩放，而是维度被解释错了。

当前代码直接调用：

```python
# code/infer_3router_planner_wan.py:1472-1476
save_video(video, save_file=str(output_path), fps=fps)
```

其中 `video` 是 VAE 输出的 `[C,T,H,W]`。

Wan 保存器在
`/home/liuzhirui/model/Wan2.2/wan/utils/utils.py:90-110`
中沿 `tensor.unbind(2)` 拆帧，它的输入契约实际是
`[B,C,T,H,W]`。Wan 官方入口也明确传入：

```python
# /home/liuzhirui/model/Wan2.2/generate.py:551-557
save_video(
    tensor=video[None],
    ...
    nrow=1,
)
```

当前少掉的正是 `[None]`/`unsqueeze(0)`。

错误解释过程如下：

```text
实际 VAE 输出              [C=3, T=49, H=512, W=512]
保存器期望                  [B,   C,    T,     H,     W]
当前 unbind(dim=2) 实际拆到  H=512
结果                        512 个“帧”
每个错误帧高度              T=49
H.264 宏块补齐              49 -> 64
最终 MP4                    512 帧，512x64
```

### 2.2 修正维度后，底层 49 帧仍然是噪声

错误 MP4 的重排基本可逆：

```text
错误视频解码       [Y=512, T=49, W=512, C=3]
裁掉编码补齐行     每个错误帧只取前 49 行
交换 Y/T           [T=49, Y=512, W=512, C=3]
```

重排后得到正确容器几何：

```text
width=512
height=512
nb_frames=49
duration=2.041667
```

但重排后的第 0 帧和第 24 帧仍是无语义的彩色噪声。这证明：

> 保存 bug 是确定问题，但不是底层生成失败的唯一原因。

对重排结果的辅助统计如下。这些值经过一次有损 H.264，不能作为训练指标，但足以确认输出异常：

| 指标 | 数值 |
|---|---:|
| 像素均值 | 160.90 |
| 像素标准差 | 78.14 |
| 接近 0/255 的饱和像素比例 | 16.97% |
| 相邻帧平均绝对差 | 26.47 / 255 |
| 第 0 帧与参考图像素相关 | 0.313 |
| 第 0 帧与参考图平均绝对差 | 69.23 / 255 |

输出不是完全独立同分布的白噪声；VAE 和时空模型仍带来低频、时间相关结构。但它没有形成可辨识人物、背景或参考首帧，工程上仍应判定为去噪失败。

### 2.3 基础 Wan/VAE 不是普遍损坏

基础目录
`/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/outputs`
中 2026-07-29 的多段官方输出是正常的 121 帧视频，抽取首帧也能看到清晰人物画面。

这至少排除了以下宽泛解释：

- 基础 Wan checkpoint 整体损坏；
- VAE 文件整体损坏；
- 当前机器上的视频编码器只能写噪声；
- 所有 Wan scheduler 调用都必然失败。

它不能单独证明本次 MQ 采样的每个设置正确，但把问题范围收窄到了 MovieStory 的训练后条件路径、首帧注入、CFG 和自定义落盘代码。

### 2.4 checkpoint 不是空壳，加载也不是当前首要嫌疑

最终 checkpoint 的静态和运行时证据包括：

- 三路 MQ 参数形状正确：
  - role `[96,2048]`
  - action `[96,2048]`
  - global `[64,2048]`
- Connector 有 341 个 tensor；
- Wan `cond_only` 有 397 个 tensor；
- 推理抽样 tensor 与 checkpoint 精确相等；
- 三路 MQ 相对初始化都有非零累计更新；
- step 2-500 的路由 update 均为 PASS；
- 运行时 MQ context 置零会改变 Wan 首步输出。

对全部 397 个 Wan 训练 tensor 与基础 Wan 权重做逐 tensor 比较：

| 指标 | 数值 |
|---|---:|
| 参数量 | 1,212,426,240 |
| 基础权重总体 RMS | 0.027787 |
| final 相对 base 的 delta RMS | 0.0002057 |
| 总体相对 RMS 位移 | 0.740% |
| 最大绝对位移 | 0.001850 |
| `text_embedding.0.bias` 相对 RMS 位移 | 20.13% |
| `text_embedding.2.bias` 相对 RMS 位移 | 18.48% |

结论不是“权重没加载”，而是：

> 这些权重确实被训练和加载了，但“被更新”不能证明更新后的向量场可用于长链采样。

---

## 3. 完整训练流程

```text
OpenVid CSV + 视频目录
        |
        v
prepare_openvid_subset.py
按 CSV 顺序挑选前 4000 个可解析文件，创建子集
        |
        v
WanVideoDataset
读取每段开头连续 49 帧 -> RGB -> 面积不超过 262144 -> H/W 对齐 32
像素归一化到 [-1,1]
frame[0] 同时作为 ref_image 与 mq_ref_image
caption dropout=0.1，MQ image dropout=0.1
        |
        +-------------------------+
        |                         |
        v                         v
Qwen 三路条件                    Wan VAE
role:   空文本 + 参考图           video [3,49,H,W]
action: caption + 无图            -> x0 [48,13,H/16,W/16]
global: caption + 参考图
        |
        v
三组独立 FP32 MetaQuery 参数
Qwen 输出 [96/96/64, 2048]
        |
        v
按 role/action/global 拼接 [B,256,2048]
        |
        v
共享 24 层 Qwen2Encoder Connector
-> MQ [B,256,4096]
        |
        v
按同 caption 的 T5 RMS 做单标量缩放
T5 token 本身不送入 DiT
        |
        +-------------------------+
                                  |
                                  v
                       t~Uniform(0,1), eps~N(0,I)
                       x_t=(1-t)x0+t*eps
                                  |
                                  v
                       再改写第一个 latent slot
                       x_t[:,0] <- (1-alpha)x_t[:,0]+alpha*ref
                       但 target 仍是 eps-x0
                                  |
                                  v
                       Wan DiT(text_len=256, MQ-only)
                       -> predicted velocity
                                  |
                                  v
                       loss=MSE(pred, eps-x0)
                                  |
                                  v
                       AdamW + grad accumulation 8
                       500 optimizer steps
```

### 3.1 数据阶段

活动 Dataset 位于：

`/home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_connector_for_wan.py`

关键行为：

1. 本地 OpenVid 文件与 CSV caption 配对；
2. 每段视频读取开头连续 49 帧，而不是全视频等距抽帧；
3. 分辨率保持长宽比缩小，随后向下对齐 32；
4. 转为 `[3,T,H,W]` 并归一化到 `[-1,1]`；
5. 第 0 帧转为 PIL 参考图；
6. 10% 概率清空 caption，10% 概率把 `mq_ref_image` 设为 `None`。

本次 batch size 实际为 1，因此缺图样本会走真正的无图 Qwen 路径。代码中的
DataLoader 把 `batch_size=1` 写死；CLI 的 `--batch_size` 在当前数据加载处没有生效，但本实验默认值正好也是 1，所以不构成本次噪声的直接原因。

### 3.2 三路 Router/Connector

`code/three_router_planner/qwen_wan_adapter.py` 的实际流程是：

- role：只给参考图，caption 替换为空；
- action：只给 caption，不给图片；
- global：同时给图片和 caption；
- 每路只保留属于该路由的 MetaQuery token；
- 三路分别做一次冻结 Qwen forward；
- 拼成 256 token 后只做一次共享 Connector forward；
- `ThreeRouterPlanner` 本身是 parameter-free identity split，没有额外路由网络。

因此当前所谓“Router”实现的是输入模态隔离与 token 分段，不是一个会学习选择、门控或专家分配的 router。

这段代码的形状和梯度连接基本成立，测试和运行时审计也证明三张 MQ 表在更新。但日志中三路 pooled cosine 长期约 `0.95-0.99`，说明仅靠模态隔离没有自然形成强语义分工。

### 3.3 MQ/T5 RMS 匹配

训练中 Connector 输出 RMS 约为 `1.0`，T5 condition RMS 约为 `0.08`。代码每个 micro-batch 都计算：

```text
scale = clip(T5_RMS / MQ_RMS, 0.03, 4.0)
MQ_effective = MQ * scale
```

推理也复现了同样的缩放，正向 MQ 从约 `1.0` 缩到 `0.0764`，负向 MQ 缩到 `0.0820`。

这只能说明二阶幅值大致相等，不能保证：

- token 方向相同；
- cross-attention key/value 分布相同；
- 条件语义相同；
- 不同 timestep 下 Wan 响应相同；
- CFG 的正负条件差具有正确含义。

RMS 对齐不能代替 feature alignment 或 function distillation。

### 3.4 Flow Matching 主目标

活动训练代码位于：

`/home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_metaquery_wan.py:10218-10485`

标准部分是：

```text
t ~ Uniform(0,1)
epsilon ~ N(0,I)
x_t = (1-t)x0 + t*epsilon
v_target = epsilon - x0
```

`v_target` 的符号与 Wan flow scheduler 是一致的，不是这里写反。

### 3.5 首帧软锚定破坏了训练监督的一致性

问题发生在：

`train_metaquery_wan.py:10368-10385`

先构造：

```text
x_t = (1-t)x0 + t*epsilon
```

然后改写：

```text
x'_t[first] = (1-alpha(t))*x_t[first] + alpha(t)*ref_latent
```

但 target 仍然是：

```text
v_target = epsilon - x0
```

当 `alpha≈1`、`ref_latent≈x0[first]` 时，DiT 看到的是近似干净首 latent，却被告知 timestep 接近 1000，并被要求预测与当前输入不再对应的随机 `epsilon-x0`。

本实验设置：

```text
alpha0=1.0
warmup_ratio=1.0
```

意味着所有非零 timestep 都会有一定锚定，高噪声区几乎完全替换首 slot。这个监督对第一个 latent slot 是不适定的。更严重的是 Wan 有时空 self-attention，首 slot 的错误状态可能污染其他 12 个 latent 时间槽。

这不是单纯“软条件可能不够强”，而是训练输入、timestep 和 velocity target 的数学契约没有同步修改。

### 3.6 实际 loss 没有启用配置中声称的辅助功能

CLI 和 checkpoint 记录：

```text
enable_t5_alignment=true
lambda_t5_align_l2=0.2
lambda_t5_align_cos=0.1
lambda_t5_align_stats=0.02
```

代码也定义了：

- `_compute_mq_aux_losses`
- `_compute_wan_func_distill_loss`

但活动 `_compute_loss` 没有调用它们，而是明确执行：

```python
# train_metaquery_wan.py:10469-10480
total_loss = denoise_loss
# 所有 align/image/function 指标强制置 0
```

全文件搜索显示，活动版本的两个函数只有定义，没有活动调用点。日志也确认 500 步中：

```text
align=0.0000
func=0.0000
```

所以以下表述需要纠正：

| 配置/功能 | 实际状态 |
|---|---|
| T5 RMS probe/match | 已实现并执行 |
| T5 feature alignment loss | 函数存在，但活动训练未接线 |
| image preserve loss | 函数存在，但活动训练未接线 |
| Wan function distillation | 函数存在，但活动训练未接线，且本次配置也关闭 |
| 训练主目标 | 只有 denoise velocity MSE |

### 3.7 参数规模与训练量

本次实际可训练参数：

| 参数组 | 参数量 |
|---|---:|
| Connector + MQ | 约 16.37 亿 |
| Wan `cond_only` | 约 12.12 亿 |
| 合计 | 约 28.49 亿 |

训练量：

```text
4000 个视频
batch_size=1
gradient_accumulation=8
500 optimizer steps
约等于只看一遍数据
```

虽然 loss EMA 从约 `0.666` 降到 `0.353`，但没有固定 validation、没有 timestep 分桶、没有训练中采样，因此无法判断高噪声端是否学会。扩散/flow 模型在单步 MSE 上“有下降”但多步采样仍完全失败并不矛盾：每一步的小偏差会在 50 步中累积。

`cond_only` 也不是很小的适配层。默认关键词选中了所有 block 的：

- cross attention；
- `norm3`；
- `modulation`；
- text embedding；
- time projection。

其中 time projection 和 modulation 会影响整个去噪动态，不只是“读取 MQ 的小支路”。用与 Connector 相同的 `1e-5` 学习率，在 4000 个样本上更新 12.12 亿参数，存在破坏基础 velocity field 的风险。

---

## 4. 完整推理流程

```text
checkpoint-final
  |-- mq_encoder_trainable.safetensors
  |-- model.safetensors 中冻结 Qwen 特殊 token 行
  `-- wan_dit_trainable.safetensors
        |
        v
加载基础 Qwen + Connector + 三路 MQ + Wan 条件权重
        |
        +-----------------------------+
        |                             |
        v                             v
正条件 MQ                           “无条件”MQ
prompt + ref_image                  默认长 negative prompt + 无图
三路 Qwen -> Connector             三路 Qwen -> Connector
RMS 对齐到正 T5 RMS                 RMS 对齐到负 T5 RMS
        |                             |
        +-------------+---------------+
                      |
                      v
参考图缩放到 512x512 -> VAE -> ref_latent [48,1,32,32]
随机噪声 latent [48,13,32,32]
UniPC, shift=5, 50 steps
                      |
                      v
每一步采样前改写第一 latent slot
latent[:,0] <- (1-alpha)*latent[:,0] + alpha*ref_latent
                      |
                      v
pred_cond   = Wan(latent,t,positive_MQ)
pred_uncond = Wan(latent,t,negative_MQ)
pred = pred_uncond + 5*(pred_cond-pred_uncond)
                      |
                      v
UniPC scheduler.step
                      |
                      v
最终 latent -> VAE decode -> [3,49,512,512]
                      |
                      v
错误地直接传入要求 5D 的 Wan save_video
-> 实际保存为 512 帧、512x64
```

### 4.1 checkpoint 加载

当前严格加载逻辑做得相对完整：

- 校验 token 数和 Connector 层数；
- 恢复三张 FP32 route table；
- 恢复共享 Connector；
- 从完整 MQ checkpoint 恢复冻结的 Qwen 特殊 token 行；
- 恢复 397 个 Wan 训练 tensor；
- 对抽样 tensor 做 checkpoint 精确相等检查。

`strict=False` 产生的 missing keys 是因为加载的是训练子集，不代表随机漏加载。当前证据不支持“checkpoint 根本没生效”这一解释。

### 4.2 正条件路径

正条件与训练主路径大体同构：

```text
role=image
action=prompt
global=image+prompt
joint Connector
256 MQ tokens
RMS match
Wan text_len=256
```

### 4.3 CFG 无条件分支不与训练分布严格对齐

推理在 `code/infer_3router_planner_wan.py:1321-1326` 中，当用户未传负面提示时自动使用 Wan 的标准长负面 prompt；随后在 `:1080-1101` 将它送入 MQ encoder。

但训练的 dropout 是：

```text
caption 10% 独立清空
image   10% 独立清空
```

真正“空 caption + 空 image”的概率约为：

```text
0.1 * 0.1 = 1%
```

4000 个 micro-batch 中理论上只有约 40 个完整无条件样本。训练从未专门把 Wan 的那段长负面 prompt 定义成 CFG unconditional condition。

推理却执行：

```text
pred = pred_uncond + 5*(pred_cond-pred_uncond)
     = 5*pred_cond - 4*pred_uncond
```

如果 `pred_uncond` 只是略有偏差，系数 `-4` 也会显著放大；如果它完全是训练外分布，第一步就可能把 UniPC 带离合理 latent 范围。旧版 MetaQuery 推理代码自身也留有注释：MQ-only 未充分训练 unconditional 分支时，低 guide scale 或关闭 CFG 更稳定。

这项是当前最值得先做的无重训消融：

```text
guide_scale=1.0
uncond context 改成 encode_mq("", None)
```

### 4.4 推理首帧锚定也不满足标准 I2V 契约

当前推理每步在 scheduler 前把首 slot 向干净参考 latent 混合，但：

- 首 slot 的 timestep 仍与全视频相同；
- 在首步 `t≈999` 时，`alpha≈1`，即干净 latent 被标成最高噪声 timestep；
- scheduler 的多步历史建立在被外部投影过的 sample 上；
- scheduler step 后没有立即重锁；
- 最终没有把首 latent 硬锁回 reference。

报告中的 trace：

```text
step 0:  t=999, alpha≈0.999998
step 25: t=833, alpha≈0.932751
step 49: t=92,  alpha≈0.020739
```

标准 Wan I2V 的做法是：

1. 首 slot 由 reference latent 固定；
2. 对应 token timestep 设为 0；
3. 每次 scheduler step 后重新覆盖首 slot；
4. 训练时 preserved prefix 不参与 denoise loss。

MovieStory 已经有 `StrongFirstFrameTrainingMixin` 实现这套契约，但本次启动脚本显式用了：

```text
--disable_wan_first_frame_strong_bind
```

所以正确功能已经有一部分实现，却没有用于这个 checkpoint。

### 4.5 当前运行时审计会漏掉数值爆炸

本次 `.verify.json` 中：

```text
dit_image_context_ablation.diff_rms     = Infinity
dit_image_context_ablation.relative_diff = NaN
status                                  = pass
```

原因是代码只判断：

```python
image_context_diff > audit_epsilon
```

`Infinity > epsilon` 为真，所以被判定为 pass；没有要求数值 finite。Python
`json.dumps` 又默认允许 `NaN/Infinity`，生成的文件也不是严格 JSON。

另外：

```text
image_condition_diff_rms = 0.0
image_condition_status   = warning
```

也就是说移除 role/global 的图片上下文后，Wan 首步预测逐元素没有可观测变化。代码和测试特意把这个结果视为 warning，而不是 failure。考虑到最终首帧确实没有保持参考人物，这个告警不能继续被忽略。

当前缺失的关键审计：

- `pred_cond` RMS/absmax/finite；
- `pred_uncond` RMS/absmax/finite；
- CFG delta RMS；
- guided prediction RMS；
- 每一步 latent RMS/absmax/finite；
- scheduler 预测的 `x0` RMS；
- VAE 输出 finite、范围、饱和率；
- 输出视频真实 width/height/frame count；
- 第 0 帧与 reference 的相似度。

### 4.6 当前测试全部通过，但没有覆盖真正失败点

现有 30 个单元测试全部通过。它们覆盖：

- 参数 shape；
- Router split；
- 梯度和 optimizer update；
- RMS match；
- soft anchor 系数；
- checkpoint/ablation 的部分逻辑。

没有覆盖：

- `_save_video` 的 4D/5D 契约；
- MP4 帧数和分辨率；
- 一次真实 scheduler step 的数值范围；
- CFG 正/负分支尺度；
- VAE roundtrip；
- 小分辨率端到端生成；
- `Infinity/NaN` 必须失败；
- 配置开启的辅助 loss 必须非零、必须被调用。

因此“测试全绿”和“生成纯噪声”可以同时发生。

---

## 5. 训练与推理契约对照

| 项目 | 训练 | 推理 | 判断 |
|---|---|---|---|
| 正 MQ 路由 | role 图、action 文、global 图文 | 相同 | 基本一致 |
| MQ token 数 | 256 | 256 | 一致 |
| Connector | 24 层共享，一次 joint forward | 相同 | 一致 |
| DiT context | MQ-only | MQ-only | 一致，但风险高 |
| MQ RMS | 按同 caption T5 RMS 缩放 | 正负各按 T5 RMS 缩放 | 形式一致 |
| flow target | `epsilon-x0` | scheduler 按 flow prediction 使用 | 符号一致 |
| timestep | `Uniform(0,1)` | shift=5 的 50 个离散点 | 可接受但缺高噪声验证 |
| 首帧 | 改写 `x_t` 首 slot，t 不变，target 不变 | 每步改写首 slot，t 不变 | 形式相似，但两边共同违反严格 flow 契约 |
| unconditional | 独立 caption/image dropout，完整 null 约 1% | 长负面 prompt + 无图 | 明显错位 |
| CFG | 不存在多分支组合监督 | scale=5 | 高风险 |
| 辅助对齐 | 配置开、实际未接线 | 只做 RMS | 预训练条件流形缺少保护 |
| 数值监控 | 只看总 loss/grad | 几乎无 step 数值监控 | 不足 |
| 输出保存 | 无 | 4D 传给 5D writer | 确定错误 |

---

## 6. 根因优先级

### P0-A：视频保存缺 batch 维

**状态：确定。**

影响：

- 正确的 49 帧被沿高度切成 512 个错误帧；
- 输出分辨率变为 `512x64`；
- 即使底层画面稍有结构，播放时也会呈现快速变化的横向雪花。

它解释了容器和视觉表现的一部分，但不能解释重排后仍然是噪声。

### P0-B：CFG 无条件分支与训练分布不一致

**状态：高概率，需要 `guide_scale=1` 消融确认。**

影响链：

```text
训练外 negative MQ
   -> pred_uncond 偏差
   -> CFG=5 放大 cond/uncond 差
   -> 第一批高噪声 step 偏离
   -> UniPC 多步历史继续累积错误
   -> 最终 latent 仍接近噪声分布
```

当前报告甚至没有记录 `pred_uncond` 和 guided prediction 的 RMS，无法排除数值放大。

### P0-C：首帧软锚定的 flow target 不一致

**状态：确定是数学/代码逻辑问题；是否为唯一主因需消融。**

训练给模型的第一个 latent slot 同时具有：

```text
输入：接近干净 reference
timestep：接近最高噪声
target：原始随机噪声路径的 epsilon-x0
```

这是互相冲突的监督。推理又把同样的错位送入时空 DiT 和多步 solver。

### P1-A：MQ-only 直接替换 T5，实际对齐功能未接线

**状态：确定的结构风险。**

基础 Wan 的可用生成能力建立在 T5 condition 上。当前一次性把 512 个 T5 token 完全替换为 256 个新 MQ token，只靠 denoise MSE 和 RMS 标量匹配重新适配。

同时：

- T5 alignment loss 没接线；
- image preserve loss 没接线；
- function distillation 没接线；
- 没有保留 T5 residual/gate；
- 没有固定 pretrained Wan teacher。

在这种设置下，500 步后仍输出噪声并不反常。

### P1-B：用 4000 个样本联合更新 28.49 亿参数

**状态：确定的训练风险。**

特别是 `cond_only` 更新了 time projection 和 block modulation，不只是 cross-attention。总体权重位移不算巨大，但若干小 bias 相对基础值移动 18%-20%，足以显著改变归一化和条件响应。

建议先证明“冻结 Wan，只训练 MQ/Connector”可以生成，再逐层解冻。

### P1-C：没有 validation、timestep 分桶和训练中采样

**状态：确定缺失。**

YAML 的 searcher 写了 `validation_loss`，训练代码却没有 validation loop，也没有上报该指标。训练成功的现有判据只有：

- loss 有下降；
- 参数有梯度；
- 参数发生变化。

这些只能证明 optimizer 在运行，不能证明 sampling 可用。

### P1-D：审计把异常结果当成功

**状态：确定。**

- `Infinity/NaN` context diff 被判 pass；
- 图像对 Wan prediction 的影响为 0 被判 warning；
- 最终输出只检查文件存在和字节数；
- 报告自称 generation pass，但真实帧数/尺寸不符；
- 没有 snow/noise 判定。

### P2：其他工程问题

1. `--batch_size` 在 DataLoader 处被硬编码的 1 覆盖；
2. `train/video_conditioning_mode_cfg` 日志错误读取 `dit_condition_mode`；
3. `checkpoint-before-training` 已缺失，难以完整比较 base/initial/final；
4. 当前 resume 只热启动 MQ，不恢复 Wan、optimizer、scheduler、step、RNG；
5. 训练脚本硬编码了 W&B API key，应该立即撤销/轮换并改为 secret/env 注入；
6. 上游训练文件累积了大量注释掉的历史版本，增加“修改了非活动代码”的风险。

这些不一定直接制造雪花，但会降低实验可复现性和诊断可信度。

---

## 7. 改进方案

### 7.1 P0：不重训，先隔离问题

#### 7.1.1 修复视频落盘

最小修复：

```python
def _save_video(video, output_path, fps):
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"expected [3,T,H,W], got {tuple(video.shape)}")
    if not torch.isfinite(video).all():
        raise FloatingPointError("decoded video contains NaN/Inf")

    save_video(
        video.unsqueeze(0),  # [B,C,T,H,W]
        save_file=str(output_path),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
```

保存后必须读取容器 metadata 并断言：

```text
nb_frames == frame_num
width  == processed_ref.width
height == processed_ref.height
fps    == requested_fps
```

也可以直接使用项目旧版的 OpenCV writer，把
`[C,T,H,W] -> [T,H,W,C]` 后逐帧写出，避免复用含 batch/grid 语义的 Wan helper。

#### 7.1.2 立即增加采样数值轨迹

至少在 step `0/1/5/10/25/49` 写入：

```text
timestep
alpha
latent rms/std/absmax/finite
ref_latent rms/std/absmax/finite
pred_cond rms/absmax/finite
pred_uncond rms/absmax/finite
cond_minus_uncond rms
guided_pred rms/absmax/finite
predicted_x0 rms/absmax/finite
```

失败条件：

- 任意 NaN/Inf；
- RMS 或 absmax 相对前一步突增超过预设倍数；
- CFG 后 prediction 比 cond/uncond 大一个数量级；
- 最终 latent RMS 明显偏离真实 VAE latent 的训练分布。

所有报告写 JSON 时使用：

```python
json.dumps(..., allow_nan=False)
```

#### 7.1.3 第一轮消融矩阵

所有实验固定同一 prompt、reference、seed、分辨率、solver 和 50 steps：

| 编号 | Wan 权重 | MQ 权重 | 首帧 | uncond | CFG | 目的 |
|---|---|---|---|---|---:|---|
| A | 官方 base | T5 官方路径 | 官方 I2V | 官方 | 5 | 基线，应正常 |
| B | final | final | 当前 soft anchor | 不计算 | 1 | 判断 CFG 是否主因 |
| C | final | final | 当前 soft anchor | `MQ("",None)` | 2 | 测训练一致 null |
| D | final | final | 当前 soft anchor | negative MQ | 5 | 当前失败复现 |
| E | base | final | 当前 soft anchor | 不计算 | 1 | 判断 Wan 更新是否破坏 |
| F | final | final | 关闭 anchor | 不计算 | 1 | 判断 anchor 是否主因 |
| G | final | final | 官方 hard reference slot | 不计算 | 1 | 判断正确首帧契约 |
| H | checkpoint-240 | 对应 MQ | 与 G 相同 | 不计算 | 1 | 判断后半程训练是否退化 |

判断顺序：

1. A 失败：先修基础环境，不讨论训练；
2. A 正常、B 正常、D 失败：CFG/uncond 是主因；
3. B 失败、F 正常：soft anchor 是主因；
4. B/F 失败、G 正常：必须改用标准 I2V preserved slot；
5. E 正常、B 失败：Wan `cond_only` 更新破坏基础向量场；
6. E/B 都失败：MQ/Connector 本身未学到可用 condition；
7. 240 正常、500 失败：后半程训练退化；
8. 全部 MQ 方案失败：当前训练目标/训练量不足，必须重训。

### 7.2 P1：修正推理契约

#### 7.2.1 CFG 默认先设为 1

在真正训练好 unconditional 分支前：

```text
GUIDE_SCALE=1.0
```

`guide_scale<=1` 时不要额外计算 unconditioned forward。

#### 7.2.2 无条件 MQ 必须使用训练一致的 null

默认改成：

```python
uncond_mq = encode_mq("", None)
```

不要把 Wan 的标准 T5 negative prompt 自动假定成 MQ unconditional prompt。若要支持 negative prompt，应把同一种 negative MQ 分布纳入训练。

建议参数：

```text
--cfg_uncond_mode empty_mq|negative_mq|zero_mq
```

默认 `empty_mq`，并在报告中记录 cond/uncond context 的 RMS、cosine、diff RMS 及两支 Wan prediction 的数值。

#### 7.2.3 首帧使用标准 preserved slot

不要继续使用“干净 latent + 高噪声 timestep + 原 velocity target”的组合。

优先复用已有 `StrongFirstFrameTrainingMixin` 的契约：

```text
reference prefix latent: clean
reference token timestep: 0
target video: 去掉重复的第一 latent slot
loss: 不计算 preserved prefix
sampling: 每步后重锁 reference slot
final: 确认首 slot 与 ref latent 一致
```

若必须保留软锚定，需要重新推导随 `alpha(t)` 改变后的 path、timestep 和
`dx/dt` target，不能只改 `x_t` 而不改监督。

#### 7.2.4 降低多步 solver 对外部投影的敏感性

UniPC 是多步方法，会保存历史模型输出。若每步对 sample 做额外非 ODE 投影，历史项可能不再一致。

建议：

1. 先用官方 Wan I2V mask/preserved-slot 路径；
2. 诊断阶段增加 Flow Euler 单步 solver 对照；
3. 不要在未知数值稳定性下同时叠加 soft projection、CFG=5 和 UniPC。

### 7.3 P1：修正训练目标

#### 7.3.1 使用强绑定训练契约重训

启动参数改为：

```text
--wan_first_frame_strong_bind
```

并核对 metadata：

```text
mode=wan_animate_slot
ref_slots=1
timestep_zero=1
drop_prefix_loss=1
original_first_target_slot_removed=1
```

#### 7.3.2 专门训练 CFG null 分支

不要只用两个独立 10% dropout。增加联合 dropout：

```text
p_full_condition = 0.80~0.90
p_joint_null      = 0.10~0.20  # caption="" 且 image=None
其余概率再做单模态 dropout
```

推理的 null 表示必须与训练完全一致。

#### 7.3.3 真正接入辅助 loss

短期至少选择一种，不要只在配置中显示 enabled：

1. **固定 Wan teacher 的 function distillation**  
   对同一 `x_t,t`，约束 MQ 条件下的 velocity 接近原始 Wan+T5 的 velocity。
2. **T5/MQ feature distribution alignment**  
   作为辅助项，但不能仅做 token 一一 L2。
3. **图像条件敏感性约束**  
   有图和无图的 MQ/Wan prediction 应有可测差异。

注意：当前 `_compute_wan_func_distill_loss` 即使接线，也调用的是当前正在训练的
Wan 实例，只是用了 `no_grad`。如果 Wan 参数同时更新，它不是固定的 pretrained
teacher。正确做法是：

- 单独保留 frozen base Wan teacher；或
- 预缓存固定 `(x_t,t,T5)` teacher velocity；或
- 至少冻结 Wan，只训练 MQ/Connector 时再用同一个 Wan 作为 teacher。

#### 7.3.4 不要第一阶段就大规模改 Wan

推荐阶段化：

```text
Stage 1: 冻结 Wan，只训练 MQ route tables + Connector
Stage 2: 只解冻 cross-attention 的 K/V/O 或小型 adapter/LoRA
Stage 3: 有稳定 validation/sample 后，再谨慎解冻更多条件层
```

避免第一阶段训练：

- time projection；
- 全部 block modulation；
- 大量 norm bias。

Wan 学习率应显著低于 Connector，例如先测试：

```text
connector_lr = 1e-5
wan_lr       = 5e-7 ~ 1e-6
```

### 7.4 P1：保留基础 T5 能力

当前 `mq_only` 是最激进的方案。更稳妥的过渡方案：

```text
context = T5 + gated_MQ
```

或：

```text
MQ 先预测对 T5 context 的 residual/adapter
gate 从 0 开始训练
```

这样 checkpoint-0 仍能调用基础 Wan 的可用生成能力，训练只需学增量，而不是用 4000 个视频从头教会一个全新的 256-token 条件空间。

如果最终产品必须严格 MQ-only，也建议先通过 T5 residual/teacher 蒸馏得到稳定模型，再逐步退火移除 T5。

### 7.5 P1：建立真正的 validation

固定一组 validation 样本，并为每个样本固定：

- 视频；
- caption；
- reference；
- timestep；
- noise seed。

每 N 步记录：

1. 固定 validation denoise loss；
2. 按 timestep 分桶的 loss：
   - `[0,0.2)`
   - `[0.2,0.5)`
   - `[0.5,0.8)`
   - `[0.8,1.0]`
3. prediction 与 target 的 RMS、cosine；
4. 由 `x_t - t*v_pred` 得到的 `x0_pred` 误差；
5. CFG null 分支 loss；
6. image ablation 的 prediction diff；
7. 固定 seed 的短视频 sample。

只有训练 loss，没有高噪声分桶时，无法发现“平均 loss 下降但采样第一步完全错误”。

### 7.6 P2：补测试和工程约束

必须新增：

- `test_save_video_preserves_t_h_w`
- `test_saved_mp4_metadata_matches_request`
- `test_nonfinite_context_fails_audit`
- `test_nonfinite_prediction_or_latent_aborts`
- `test_cfg_scale_one_skips_uncond`
- `test_empty_mq_uncond_matches_training_contract`
- `test_strong_reference_slot_has_timestep_zero`
- `test_reference_prefix_excluded_from_loss`
- `test_enabled_aux_loss_is_actually_called`
- GPU 可用时的小分辨率端到端 smoke generation

同时：

- 删除/撤销脚本中硬编码的 W&B key；
- 保留 `checkpoint-before-training`；
- 实现完整 resume；
- 清理上游文件中的多份注释历史版本；
- 把活动训练实现拆成单一文件，避免编辑到无效副本。

---

## 8. 建议执行顺序

### 第一批：无需重训

1. 修复 4D/5D 保存 bug；
2. 增加 MP4 metadata 断言；
3. 增加逐步 finite/RMS/absmax 审计；
4. 同 seed 跑 `CFG=1`；
5. 将 uncond 改成 `MQ("",None)` 后测试 `CFG=2`；
6. 对照关闭 soft anchor；
7. 对照标准 hard reference slot；
8. 对照不加载 397 个 Wan 更新；
9. 对照 checkpoint-240。

这批实验能在不重新训练的情况下定位“当前 checkpoint 还有没有可救的采样设置”。

### 第二批：需要小规模重训

1. 强绑定 reference slot；
2. joint null dropout；
3. 冻结 Wan；
4. 实际接入固定 teacher function distillation；
5. 先用 100-500 个视频做过拟合验证；
6. 每 50 step 固定 seed 生成，确认能从噪声变成可辨识画面。

### 第三批：正式训练

1. 扩大到 4000+ 数据；
2. 分阶段解冻 Wan；
3. 完整 validation；
4. 选择 checkpoint 依据 validation/sample，而不是最终 step；
5. 再逐步恢复 CFG，并用 1/2/3/5 做质量曲线。

---

## 9. 验收标准

修复不应只以“程序退出码为 0”或“MP4 文件存在”为标准。

最低验收：

```text
[ ] MP4 帧数 = 49
[ ] MP4 尺寸 = 512x512
[ ] 全部 context/prediction/latent/video finite
[ ] guide=1 能形成可辨识画面
[ ] 第 0 帧与 reference 有明显结构一致性
[ ] image ablation 能改变 Wan prediction
[ ] 固定 validation 高噪声桶 loss 不发散
[ ] checkpoint-0/中间/final 有固定 seed sample 对照
[ ] 配置为 enabled 的 loss 在日志中非零且有梯度
[ ] 基础 Wan 官方路径始终保持正常
```

在这组标准通过之前，不建议直接把问题归结为“只训练了 500 步所以质量差”，也不建议盲目追加训练。当前首先需要修复的是可证明的保存维度错误、首帧 flow 契约和 CFG/null 条件错位。

