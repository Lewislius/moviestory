# MovieStory 3-router：P0-A / P0-B / P0-C / P1-D 修复说明

> 实施日期：2026-07-31  
> 范围：全部 MovieStory 3-router 训练入口、推理入口、首帧共享模块、测试与运行文档  
> 损失约束：只优化生成视频与 ground-truth 视频对应的 latent velocity MSE

## 1. 修复后的端到端契约

### 1.1 训练

```text
ground-truth video [3,49,H,W]
  │
  ├─ frame 0 ───────────────────────────────> ref_image
  │                                             │
  │                                             └─ Wan VAE
  │                                                  ↓
  │                                          clean ref latent [C,1,h,w]
  │
  └─ Wan VAE
       ↓
     GT video latent [C,13,h,w]
       ↓ 删除重复的第一个 target slot
     target latent [C,12,h,w]
       ↓ 前面拼接 reference condition slot
     full sequence [C,13,h,w]
       │
       ├─ reference slot:
       │    input = clean ref latent
       │    token timestep = 0
       │    loss mask = 0
       │
       └─ target slots:
            input = (1-t) * GT + t * noise
            token timestep = sampled t
            target velocity = noise - GT
            loss mask = 1
```

总损失：

```text
total_loss
  = MSE(
      predicted_velocity[target video slots],
      ground_truth_velocity[target video slots]
    )
```

没有接入以下辅助损失：

```text
T5 alignment loss              = 0
MQ image preserve loss         = 0
Wan function distillation loss = 0
reference condition prefix loss= 0
```

MQ/T5 RMS probe 和 RMS match 仍然保留，但它们只是条件张量归一化，不是 loss。

### 1.2 推理

```text
initial random latent [C,13,h,w]
  │
  ├─ slot 0 = clean reference latent
  └─ slots 1: = random noise

for every scheduler step:
  1. step 前重锁 slot 0
  2. slot 0 的 DiT token timestep = 0
  3. 其他 token 使用当前采样 timestep
  4. Wan 预测 velocity
  5. scheduler 更新 latent
  6. step 后再次重锁 slot 0

VAE decode -> [3,49,H,W]
Wan writer input -> [1,3,49,H,W]
ffprobe -> 必须为 H×W、49 帧、指定 FPS
```

---

## 2. P0-A：视频维度与容器元数据

修改文件：

- `code/infer_3router_planner_wan.py`
- `code/tests/test_three_router_inference.py`

### 原问题

Wan writer 要求：

```text
[B,C,T,H,W]
```

旧代码直接传入：

```text
[C,T,H,W]
```

导致 writer 沿错误维度拆帧，实际输出成为 `512×64、512 帧`。

### 当前实现

保存前严格断言：

```text
video.ndim == 4
video.shape == [3, expected_frame_num, expected_height, expected_width]
all values are finite
```

传给 Wan writer：

```python
writer_input = video.unsqueeze(0)
```

即：

```text
[3,T,H,W] -> [1,3,T,H,W]
```

另外加入：

1. `nrow=1`；
2. 写入临时 MP4；
3. 使用 `ffprobe -count_frames` 读取真实容器；
4. 强制核验 width、height、frame count、FPS；
5. 只有全部通过后才原子替换正式输出；
6. writer 静默失败、旧文件残留或错误容器都不能被报告为成功。

测试使用真实 H.264 临时容器验证：

```text
writer input = [1,3,5,32,48]
ffprobe width = 48
ffprobe height = 32
ffprobe frames = 5
```

---

## 3. P0-B：CFG 与训练 null 分布

修改文件：

- `code/train_3router_planner_wan.py`
- `code/train_openvid100_3router.sh`
- `code/train_openvid4000_3router.sh`
- `code/infer_3router_planner_wan.py`
- `code/infer_openvid4000_3router.sh`
- `code/infer_openvid4000_3router.yaml`

### 3.1 安全推理默认值

默认值从：

```text
guide_scale = 5
Wan long negative prompt -> MQ negative context
```

改为：

```text
guide_scale = 1
不执行 CFG unconditional sampling branch
```

`guide_scale=1` 时采样预测严格为：

```text
prediction = pred_conditioned
```

不会再用训练外 long-negative MQ 以 `-4` 系数放大误差。

### 3.2 CFG 可选模式

当显式设置 `guide_scale > 1` 时，新增：

```text
--cfg_uncond_mode empty_mq|negative_mq|zero_mq
```

含义：

| 模式 | unconditional context | 用途 |
|---|---|---|
| `empty_mq` | `MQ("", None)` | 默认；与新训练 joint-null 一致 |
| `negative_mq` | `MQ(negative_prompt, None)` | 显式实验，不再偷偷自动使用 |
| `zero_mq` | 全零 context | 消融实验 |

默认是 `empty_mq`。

### 3.3 训练联合空条件

新增：

```text
--joint_null_prob 0.1
```

命中时强制：

```text
caption = ""
mq_ref_image = None
```

但保留：

```text
ref_image = ground-truth video frame 0
```

因此 null 分支只去掉 MQ 图文条件，Wan preserved reference slot 仍然存在，与推理
CFG 两支共享同一首帧 latent 条件的行为一致。

联合空条件不会新增 loss，仍然使用同一个 ground-truth 视频 denoising loss。

---

## 4. P0-C：移除监督不一致的首帧软锚定

修改文件：

- `code/three_router_planner/wan_first_frame.py`
- `code/train_3router_planner_wan.py`
- 两个 3-router 训练 shell
- `code/infer_3router_planner_wan.py`

### 4.1 训练默认改为 strong binding

Python 参数默认：

```text
wan_first_frame_strong_bind = true
```

两个训练脚本显式传入：

```text
--wan_first_frame_strong_bind
--train_video_conditioning_mode wan_animate_slot
--train_ref_anchor_mode none
```

强绑定设置：

```text
ref slots                 = 1
preserve timestep zero    = true
drop prefix loss          = true
soft anchor mode          = none
soft anchor alpha         = 0
soft anchor warmup        = 0
```

### 4.2 条件与 target 分离

reference slot 是已知条件，不是要生成的 ground-truth target：

```text
reference slot:
  clean input + timestep 0 + no loss

target video slots:
  standard flow x_t + sampled timestep + velocity MSE
```

这样不再出现：

```text
input 接近干净 reference
timestep 却接近最高噪声
target 仍是原随机 flow velocity
```

### 4.3 推理与训练对称

推理默认：

```text
--first_frame_mode preserved
```

行为：

1. 初始噪声的第一 latent slot 替换为 VAE reference latent；
2. 对应 DiT token timestep 固定为 0；
3. 每个 scheduler step 前重锁；
4. 每个 scheduler step 后再次重锁；
5. 最终解码前第一 slot 与 reference latent 保持精确一致。

可用：

```text
--first_frame_mode none
```

做无首帧 latent 条件消融。代码不再提供 soft-anchor 推理路径。

### 4.4 关闭 strong binding 的含义

训练参数 `--disable_wan_first_frame_strong_bind` 仍保留用于消融，但现在会设置：

```text
enable_ti2v_first_frame_condition = false
train_video_conditioning_mode     = legacy_t2v
train_ref_anchor_mode             = none
alpha                             = 0
```

即标准 T2V flow matching，不会恢复旧的监督不一致 soft anchor。

---

## 5. P1-D：审计不能再把异常判为成功

修改文件：

- `code/infer_3router_planner_wan.py`
- `code/tests/test_three_router_inference.py`

### 5.1 NaN / Inf

以下内容遇到非有限值会立即失败：

- MQ/T5 context；
- 每一步输入 latent；
- conditional prediction；
- unconditional prediction；
- guided prediction；
- estimated predicted x0；
- scheduler 更新后的 latent；
- reference latent；
- VAE decoded video；
- verification JSON 中任何实数。

JSON 使用：

```python
allow_nan=False
```

读取 metadata JSON 时也拒绝非标准 `NaN`、`Infinity`、`-Infinity`。

### 5.2 每步数值轨迹

每个采样 step 都执行 finite 检查，并在以下 step 写入完整统计：

```text
0 / 1 / 5 / 10 / middle / final
```

统计包括：

```text
latent before/after: shape, rms, std, absmax, finite
pred conditioned:    shape, rms, std, absmax, finite
pred unconditioned:  shape, rms, std, absmax, finite
guided prediction:  shape, rms, std, absmax, finite
predicted x0:        shape, rms, std, absmax, finite
CFG amplification
latent step growth
reference prefix max error
```

默认增长熔断：

```text
--audit_growth_limit 20
```

CFG prediction 或相邻 latent RMS 增长超过限制会终止，不会写 generation pass。

### 5.3 图片条件零影响

旧行为：

```text
image_condition_diff_rms == 0 -> pass_with_warning
```

新行为：

```text
image_condition_diff_rms <= audit_epsilon -> fail + RuntimeError
```

因此之前 checkpoint 上已经观测到的“图片 MQ context 变化，但 Wan prediction
完全不变”会在默认 `runtime_audit=full` 下阻止输出成功报告。

### 5.4 generation 状态

状态顺序：

```text
decoded_pending_container_validation
  -> MP4 写入
  -> ffprobe metadata validation
  -> generation pass
```

任何异常都会记录：

```json
{
  "status": "fail",
  "failure": {
    "type": "...",
    "message": "..."
  }
}
```

不会再因“MP4 文件存在且非空”直接报告 pass。

### 5.5 纯雪花质量审计

VAE decode 后、写 MP4 前新增内容质量检查：

```text
第一帧与 processed reference 的 MAE
第一帧与 processed reference 的相关系数
除第一帧外生成帧的空间高频 RMS
空间高频 RMS / 生成内容 RMS
相邻生成帧相关系数
```

默认失败阈值：

```text
first-frame/reference MAE > 0.4
first-frame/reference correlation < 0.5
generated spatial high-frequency ratio > 0.9
```

这样 preserved reference 已损坏或其余帧呈雪花状高频噪声时，不会继续写出
generation pass。三个阈值都可由推理 CLI 调整，实际 GPU validation 后应依据
正常 Wan 样例的统计分布进一步标定。

---

## 6. 唯一损失契约

3-router wrapper 现在会主动覆盖所有辅助 loss 配置：

```text
enable_t5_alignment       = false
lambda_t5_align_l2        = 0
lambda_t5_align_cos       = 0
lambda_t5_align_stats     = 0
enable_mq_image_preserve  = false
lambda_mq_image_preserve  = 0
enable_wan_func_distill   = false
lambda_wan_func_distill   = 0
```

每次 `_compute_loss` 后还会运行断言：

```text
loss 是 finite scalar
loss == _last_loss_denoise
所有 _last_loss_aux_* == 0
```

checkpoint metadata 记录：

```text
loss_contract.name = video_ground_truth_velocity_mse_only
optimized_terms = [video_ground_truth_velocity_mse]
```

---

## 7. checkpoint 兼容性

已有的 step-500 checkpoint 可以继续完成文件、张量和权重加载检查，但 metadata
显示它是在：

```text
legacy_t2v_soft_anchor
```

下训练的。检查状态会显示：

```text
pass_with_warning
```

并明确说明：

1. 新推理会使用 corrected preserved slot；
2. 这不能反向修复旧 checkpoint 已经接受过的不一致训练监督；
3. 默认 full audit 还可能因已有 checkpoint 的图片条件零影响而中止。

因此代码修复后可以先对旧 checkpoint 做 `guide=1 + preserved slot` 消融，但正式
评估应使用新 strong-binding 配置重新训练的 checkpoint。

---

## 8. 验证结果

已完成：

```text
Python py_compile                    PASS
bash -n 三个训练/推理 shell          PASS
3-router inference --parse_only      PASS
3-router training --router_parse_only PASS
旧 checkpoint --check_only           PASS_WITH_WARNING
unit tests                            37/37 PASS
真实临时 H.264 容器维度测试           PASS
合成纯雪花质量审计测试                 PASS
```

真实临时容器测试确认：

```text
input video       [3,5,32,48]
writer input      [1,3,5,32,48]
container width   48
container height  32
container frames  5
container fps     24
```

当前执行环境无法初始化 CUDA driver，因此未在本机完成 5B Wan 的完整生成。完整
GPU 验收仍需确认：

```text
MP4 = 512×512
frames = 49
guide=1 可形成可辨识画面
reference prefix error = 0
full audit 不出现图片条件零影响
```

---

## 9. 其他同步修正

- 新训练输出目录使用 `strongbind` 标识，避免覆盖或误认旧 `legacysoft`
  checkpoint。
- `code/README.md` 和两份 3-router 设计说明已同步为当前运行契约。
