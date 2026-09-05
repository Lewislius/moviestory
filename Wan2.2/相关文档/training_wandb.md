# WandB 在 MetaQuery 训练中的作用分析

## 一、WandB 总体定位

WandB (Weights & Biases) 是本项目的**主要实验追踪后端**，负责记录训练过程中的所有关键指标和可视化内容。整个集成链路分布在 4 个层级：

```
环境变量 / Shell 脚本 (初始化配置)
        ↓
train.py (项目级设置 + GT 图像上传)
        ↓
HuggingFace Trainer 内部 (自动指标记录)
        ↓
trainer.py MetaQueryTrainer / MetaQueryCallback (自定义记录)
```

---

## 二、各文件中 WandB 的具体作用

### 2.1 `train.py` — 项目级配置与初始上传

#### 2.1.1 环境变量设置 (L26-27)

```python
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "MetaQuery"
```

| 环境变量 | 值 | 作用 |
|----------|-----|------|
| `WANDB__SERVICE_WAIT` | `300` | WandB 服务进程启动的超时等待时间（秒）。分布式训练时多进程同时启动，WandB 内部的 service process 初始化可能较慢，设为 300 秒避免超时崩溃 |
| `WANDB_PROJECT` | `"MetaQuery"` | 所有训练 run 归属到 WandB 上的 `MetaQuery` 项目中。WandB 网页端会显示一个名为 MetaQuery 的 project，包含所有 run |

#### 2.1.2 `report_to` 配置 (L114)

```python
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = "wandb"
```

这是 HuggingFace Trainer 的核心开关。设为 `"wandb"` 后，Trainer 内部会：
- 在训练开始时自动调用 `wandb.init()` 创建一个新 run
- 每次 `trainer.log()` 调用时自动把指标推送给 WandB
- 训练结束时自动调用 `wandb.finish()`

如果设为 `"tensorboard"` 或 `"none"`，则完全切换后端，不与 WandB 交互。

#### 2.1.3 `run_name` 配置 (L115)

```python
    run_name: str = "test"
```

在 WandB 网页端显示的 run 名称。实际训练时通过 shell 脚本覆盖，例如：
- Stage 1: `qwen3vl4b_t2i_small`
- Stage 2: `qwen3vl4b_inst_small`

每个 run_name 对应 WandB 上的一条独立训练记录，方便对比不同阶段/配置的实验。

#### 2.1.4 Ground Truth 图像上传 (L180)

```python
trainer.log_images({"gt_images": [wandb.Image(image) for image in gt_images]})
```

**作用**：训练开始前，将评估集的 Ground Truth 图像上传到 WandB。

**流程**：
1. `gt_images` 是从 eval_dataset 中取出的真实目标图像
2. 每张图用 `wandb.Image()` 封装为 WandB 可识别的图像对象
3. 通过 `log_images()` → `callback_handler.on_log()` → WandB callback 上传

**在 WandB 网页端的效果**：在 run 的 Media 面板中显示一组名为 `gt_images` 的图片，作为训练目标的视觉参考基准线。在后续评估时生成的图像可与这些 GT 图直观对比。

---

### 2.2 `trainer.py` — 自定义指标与图像记录

#### 2.2.1 `log_images()` 方法 (L165-168)

```python
def log_images(self, logs: Dict[str, float]) -> None:
    logs["step"] = self.state.global_step
    self.control = self.callback_handler.on_log(
        self.args, self.state, self.control, logs
    )
```

**作用**：这是一个桥接方法，用于将 **图像类型** 的数据发送给 WandB。

**为什么需要单独的方法**：HuggingFace Trainer 的标准 `self.log()` 方法只处理标量指标（loss、lr 等），而 `wandb.Image` 对象需要通过 callback 系统中的 `on_log` 钩子才能正确传递给 WandB Integration Callback。这个方法做了两件事：
1. 给 logs dict 注入当前 `global_step` 作为 x 轴坐标
2. 触发 `on_log` callback 链，最终 WandB 的 integration callback 拦截到含有 `wandb.Image` 的 logs 并上传

#### 2.2.2 `prediction_step()` 中的评估图像上传 (L193)

```python
def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
    # ... 模型采样生成图像 ...
    with torch.no_grad():
        samples = model.sample_images(**inputs, **sample_kwargs)
    samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
    samples = numpy_to_pil(samples)
    self.log_images({"images": [wandb.Image(image) for image in samples]})
```

**作用**：在每次 **评估 (evaluation)** 时，模型会实际生成图像（通过 diffusion sampling），将生成结果上传到 WandB。

**触发时机**：由 `TrainingArguments` 中的这些参数控制：
```python
eval_strategy: str = "steps"
eval_steps: int = 1000        # 每 1000 步评估一次
eval_on_start: bool = True    # 训练开始前先评估一次
```

**在 WandB 网页端的效果**：
- Media 面板中出现名为 `images` 的图像序列
- 每次评估对应一个 step，可以滑动查看不同训练阶段的生成效果
- 与之前上传的 `gt_images` 对比，直观判断模型是否在学习

#### 2.2.3 `_maybe_log_save_evaluate()` 中的训练指标记录 (L244-273)

```python
def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ...):
    # Loss Spike 检测
    if tr_loss.item() > 20 * self.running_loss:
        self.control.should_training_stop = True

    # Grad Norm Spike 检测
    if grad_norm > 25 * self.running_grad_norm and grad_norm > 1:
        self.control.should_training_stop = True

    if self.control.should_log:
        logs = {}
        tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
        logs["loss"] = round(tr_loss_scalar / (...), 4)
        logs["grad_norm"] = grad_norm
        logs["learning_rate"] = self._get_learning_rate()
        self.log(logs, start_time)       # ← 推送到 WandB
```

**记录的标量指标**：

| 指标 | 含义 | 在 WandB 中的图表 |
|------|------|-------------------|
| `loss` | 当前 step 的平均训练损失（跨所有 GPU gather 后计算） | 主看板的 loss 曲线 |
| `grad_norm` | 当前 step 的梯度范数 | grad_norm 曲线，用于监控训练稳定性 |
| `learning_rate` | 当前学习率 | lr 曲线，可看到 warmup + cosine decay |

**异常检测功能**（不直接推送给 WandB，但影响训练流程）：
- Loss Spike：当前 loss > 20 × 滑动平均 loss → 停止训练
- Grad Norm Spike：当前 grad_norm > 25 × 滑动平均 grad_norm → 停止训练
- NaN Grad：grad_norm 为 NaN 或 > 1e6 → 停止训练

训练被异常停止后，WandB 上该 run 会显示为提前终止状态，在 loss 曲线上对应一个突然断裂。

#### 2.2.4 评估指标记录 (L107)

```python
# evaluate() 方法中
self.log(output.metrics)
```

评估完成后记录的指标（由 HuggingFace Trainer 自动生成）：

| 指标 | 含义 |
|------|------|
| `eval_loss` | 评估集上的平均损失 |
| `eval_runtime` | 评估耗时（秒） |
| `eval_samples_per_second` | 评估吞吐量 |
| `eval_steps_per_second` | 评估速度 |

---

### 2.3 `MetaQueryCallback` — 训练开始时的参数统计

```python
class MetaQueryCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model, **kwargs):
        # 打印所有参数的 name / shape / dtype / trainable
        # 打印 connector 维度信息
```

这个 callback **不直接调用 WandB API**，但它的输出会出现在训练日志中，如果 WandB 配置了 console log 捕获（默认行为），这些信息也会被 WandB 记录到 run 的 Logs 面板中。

#### `on_log()` 进度显示 (L335-352)

```python
def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_world_process_zero and logs is not None:
        print(f"📊 训练进度: {progress:.2f}% | Step {state.global_step}/{state.max_steps}")
        print(f"📉 当前损失: {logs['loss']:.4f}")
        print(f"📈 学习率: {logs['learning_rate']:.2e}")
```

**作用**：在 stdout 打印人类可读的进度信息。这与 WandB 是互补的——WandB 提供网页端可视化，这里提供终端实时输出。

---

### 2.4 Shell 脚本 — 环境级 WandB 配置

#### `train_metaquery_full.sh`

```bash
# 超时设置
export WANDB__SERVICE_WAIT=300

# 可选禁用 WandB
DISABLE_WANDB="no"
# 命令行参数 --no-wandb
if [[ "${DISABLE_WANDB}" == "yes" ]]; then
    export WANDB_MODE=disabled     # ← 完全禁用 WandB
fi
```

| 环境变量 | 作用 |
|----------|------|
| `WANDB__SERVICE_WAIT=300` | 与 train.py 中相同，shell 层再设一次确保优先级 |
| `WANDB_MODE=disabled` | 设置后 WandB 进入离线模式，不上传任何数据。适用于没有网络或不需要追踪的调试场景 |

#### `train_qwen3vl_full_init.sh`

```bash
export WANDB__SERVICE_WAIT=300
```

同样的超时配置，确保在 Determined AI 集群环境下 WandB 服务有足够时间初始化。

---

## 三、WandB 数据流全景图

```
训练循环每一步
│
├── 前向传播 → loss
├── 反向传播 → grad_norm
│
└── _maybe_log_save_evaluate()
    │
    ├── 标量指标 ──→ self.log({"loss", "grad_norm", "learning_rate"})
    │                      ↓
    │               HuggingFace Trainer.log()
    │                      ↓
    │               WandbCallback.on_log()  ← HF Trainer 自动注册
    │                      ↓
    │               wandb.log({"loss": ..., "grad_norm": ..., ...})
    │                      ↓
    │               ┌─────────────────────────────────┐
    │               │  WandB 云端 / 本地               │
    │               │  Charts: loss, grad_norm, lr     │
    │               └─────────────────────────────────┘
    │
    ├── [每 eval_steps 步] evaluate()
    │   ├── prediction_step()
    │   │   ├── model.sample_images()  → 生成图像
    │   │   └── self.log_images({"images": [wandb.Image(...)]})
    │   │              ↓
    │   │       callback_handler.on_log()
    │   │              ↓
    │   │       WandbCallback → wandb.log({"images": [...]})
    │   │              ↓
    │   │       ┌────────────────────────────────────┐
    │   │       │  WandB 云端 / 本地                  │
    │   │       │  Media: 生成图像可视化               │
    │   │       └────────────────────────────────────┘
    │   │
    │   └── self.log(output.metrics)  → eval_loss, eval_runtime 等
    │
    └── [异常检测] Loss/Grad Spike
        → self.control.should_training_stop = True
        → WandB 上该 run 提前终止
```

---

## 四、WandB 网页端实际可见内容

在 `https://wandb.ai/<your-team>/MetaQuery` 项目下，每个训练 run 包含：

### 4.1 Charts (图表面板)

| 图表 | 数据来源 | 更新频率 |
|------|----------|----------|
| **train/loss** | `_maybe_log_save_evaluate()` → `self.log()` | 每 `logging_steps=1` 步 |
| **train/grad_norm** | 同上 | 同上 |
| **train/learning_rate** | 同上 | 同上 |
| **eval/loss** | `evaluate()` → `self.log()` | 每 `eval_steps` 步 |
| **eval/runtime** | HF Trainer 自动生成 | 同上 |
| **train/global_step** | HF Trainer 自动生成 | 每步 |
| **train/epoch** | HF Trainer 自动生成 | 每步 |

### 4.2 Media (媒体面板)

| 媒体 | 数据来源 | 记录时机 |
|------|----------|----------|
| **gt_images** | `train.py` L180 | 训练开始时上传一次 |
| **images** | `prediction_step()` L193 | 每次 evaluation 时上传 |

### 4.3 System (系统监控)

WandB 自动采集（无需项目代码配置）：

| 指标 | 说明 |
|------|------|
| GPU 利用率 | 各 GPU 的利用率百分比 |
| GPU 显存 | 各 GPU 的显存使用量 |
| GPU 温度 | GPU 温度 |
| CPU 利用率 | CPU 使用百分比 |
| 内存使用 | 系统 RAM 使用量 |
| 磁盘 I/O | 读写速度 |
| 网络 I/O | 上下行带宽 |

### 4.4 Config (超参数)

HuggingFace Trainer 自动上传完整的 `TrainingArguments` 到 WandB run config，包括：
- 所有训练超参数 (lr, batch_size, optimizer, ...)
- 模型配置信息
- 数据集配置

### 4.5 Logs (日志面板)

捕获 stdout/stderr 输出，包括：
- `MetaQueryCallback.on_train_begin()` 打印的参数统计表
- `MetaQueryCallback.on_log()` 打印的进度信息
- Loss/Grad Spike 报警信息

---

## 五、关键设计特点总结

### 5.1 双层日志系统

项目采用 **WandB (云端) + stdout (终端)** 双层日志：
- WandB 负责持久化、可视化、跨实验对比
- `MetaQueryCallback.on_log()` 负责实时终端输出

两者使用相同的数据源（`self.log()` / 标量指标），但展示形式不同。

### 5.2 图像记录的两个时机

| 时机 | 调用位置 | 图像类型 | 目的 |
|------|----------|----------|------|
| 训练开始前 | `train.py` L180 | Ground Truth | 建立视觉基准线 |
| 每次评估时 | `trainer.py` L193 | 模型生成图 | 追踪生成质量演变 |

### 5.3 自动异常检测 → 训练中止

```python
# Loss Spike: 20倍阈值
if tr_loss.item() > 20 * self.running_loss:
    self.control.should_training_stop = True

# Grad Norm Spike: 25倍阈值
if grad_norm > 25 * self.running_grad_norm and grad_norm > 1:
    self.control.should_training_stop = True
```

这不是 WandB 的功能，但 WandB 记录的 loss/grad_norm 曲线可以帮助事后分析 spike 原因。

### 5.4 分布式训练兼容

- `WANDB__SERVICE_WAIT=300` 确保多 GPU 启动时 WandB service 不超时
- HF Trainer 内部只在 `world_process_zero` (rank 0) 上初始化 WandB run
- `MetaQueryCallback.on_log()` 中 `state.is_world_process_zero` 检查确保只在主进程打印
- `_maybe_log_save_evaluate()` 中 `self._nested_gather(tr_loss).mean()` 保证 loss 是所有 GPU 的平均值

---

## 六、配置速查

### 启用 WandB（默认）

```bash
# 确保已登录
wandb login

# 直接运行训练脚本即可
bash scripts/train_metaquery_full.sh --base-dir /data/metaquery
```

### 禁用 WandB

```bash
# 方法 1: 命令行参数
bash scripts/train_metaquery_full.sh --base-dir /data/metaquery --no-wandb

# 方法 2: 环境变量
export WANDB_MODE=disabled

# 方法 3: 改代码中的 report_to
# train.py 中: report_to: str = "none"        # 完全不用
#              report_to: str = "tensorboard"  # 改用 TensorBoard
```

### 离线模式（先本地记录，后上传）

```bash
export WANDB_MODE=offline
# 训练完成后手动同步
wandb sync ./wandb/latest-run
```
