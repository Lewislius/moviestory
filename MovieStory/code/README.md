# MovieStory 3-router planner

本目录实现 `pipeline_design.md` 的第一个独立增量：将 Qwen3-VL MetaQuery
显式拆成 `Q_role(96) / Q_action(96) / Q_global(64)`，总计 256 token，经
24 层 Qwen2Encoder Connector 送入 Wan。

三路 MetaQuery 使用三张相互独立的 FP32 参数表；Qwen3-VL 原始输入 embedding
整表冻结。普通 token embedding 在进入 Qwen 前会断开梯度，只有选中的三张路由
参数表被写入对应 MetaQuery 位置，因此去噪 loss 可以穿过冻结的 Qwen 主干持续
更新 role/action/global 参数，而不会修改 Qwen 原始词表语义。

Wan 首帧默认使用 flow-consistent preserved slot：参考图经 VAE 编码为干净首
latent，该 slot 的 token timestep 固定为 0，并在每次 Wan forward 前重锁；
重复的目标视频首 latent 被移除，reference 条件前缀不参与 loss。总损失严格只由
后续生成视频 latent 与 ground-truth 视频 latent 的 velocity MSE 构成，不启用
T5 对齐、图像保持或函数蒸馏 loss。MQ/T5 RMS 探针与自动匹配仍作为条件归一化
保留；文本、图片独立 dropout 和联合 `MQ("", None)` dropout 默认均为 0.1。

轻量自检：

```bash
python train_3router_planner_wan.py --router_check_only
PYTHONPATH=. python -m unittest discover -s tests -v
```

`--router_check_only` 不加载 Qwen/Wan，但会对三组 FP32 MetaQuery 参数执行一次
真实的 AdamW 更新，并报告各路由的 `grad_rms`、`update_rms`、
`changed_fraction` 和 `updated`。

正式训练默认每个 optimizer step 输出两类证据：

```text
[3-ROUTER][DIAG]   ... 三路输出 RMS、路由间 cosine、反向梯度
[3-ROUTER][UPDATE] ... 参数 RMS、step update、相对更新、变化比例、累计变化、LR
```

`status=PASS` 表示当前 step 中所有“学习率和梯度均非零”的路由参数都真实变化；
warmup 初始零学习率显示 `WAIT_LR`。若任一路由连续 5 步没有梯度，或有非零梯度
和学习率却没有参数变化，训练会以 `[3-ROUTER][NO-GRAD]` 或
`[3-ROUTER][STALE]` 中止，避免静默产生无效 checkpoint。
完整逐步指标同时写入 `logs/train_metrics.jsonl`，启用 W&B 时也会同步记录。
可用 `--router_log_steps` 调整控制台频率，用
`--router_stale_update_patience 0` 关闭熔断（不建议）。

W&B SDK 已安装在 `moviestory` 环境中。API key 只通过环境变量传入，不要写入
脚本、YAML 或命令行：

```bash
export WANDB_API_KEY="<your-api-key>"
export WANDB_PROJECT="moviestory-3router"
bash train_openvid4000_3router.sh
```

只要检测到 `WANDB_API_KEY`，启动脚本就会自动导入并初始化 `wandb`，逐
optimizer step 上传 `train/loss_step`、`train/loss_ema`、学习率、梯度、
吞吐量、MQ/T5 范数和三路参数更新证据。`train_openvid4000_3router.yaml`
显式设置了 `WANDB_ENABLED=1`；在 Determined 中运行时，应通过平台的 secret
或 trial 环境注入 `WANDB_API_KEY`，也可以提前在共享 HOME 执行
`wandb login`。可选环境变量：

```text
WANDB_ENABLED=auto|1|0
WANDB_PROJECT=moviestory-3router
WANDB_ENTITY=<user-or-team>
WANDB_RUN_NAME=<run-name>
WANDB_TAGS=tag1,tag2
WANDB_MODE=online|offline|disabled
WANDB_LOG_EVERY_STEP=1|0
WANDB_LOG_CHECKPOINT=1|0
```

`WANDB_LOG_CHECKPOINT=1` 当前只记录 checkpoint 路径和 step，不上传数十 GB
的 checkpoint 文件。即使 W&B 开启，本地 JSONL 仍会保留。

OpenVid 前 4000 条可解析视频训练：

```bash
bash train_openvid4000_3router.sh
```

启动脚本按 CSV 顺序构建仅含前 4000 条可解析视频的软链接子集。训练使用
`batch_size=1`、`gradient_accumulation_steps=8` 和 500 个 optimizer steps，
对应 4000 个 micro-batch；不可解码或不满足上游过滤条件的视频仍会由 Dataset
按既有规则剔除。

OpenVid 没有独立参考图片时，上游 Dataset 会把处理后视频的第 0 帧同时作为
`ref_image` 和 `mq_ref_image`。图片 dropout 只影响 MQ 路由；Wan preserved
slot 始终使用独立的 `ref_image`，因此不会随 MQ 图片条件一起丢失。

启动脚本和 Python 入口都默认启用 `--wan_first_frame_strong_bind`。如需无首帧
latent 条件的消融，可显式传 `--disable_wan_first_frame_strong_bind`；该选项会
使用普通、数学一致的 T2V flow path，不会恢复已经废弃的 soft anchor。

同配置 baseline：

```bash
ROUTER_ENABLED=0 bash train_openvid4000_3router.sh
```

3-router 与 baseline 会分别写入带有 `mq256_conn24_strongbind` 标识的
`checkpoint/three_router_*` 和 `checkpoint/baseline_*`，不会互相覆盖，也
不会覆盖目录中已有的旧 144-token checkpoint。旧 checkpoint 与新 token 数量
及 Connector 深度不兼容，不应作为新训练的 resume 输入。

## 推理

先做不加载模型的 checkpoint 检查：

```bash
CHECK_ONLY=1 bash infer_openvid4000_3router_strongbind.sh
```

正式推理可通过环境变量指定图文输入：

```bash
REF_IMAGE=/path/to/reference.jpg \
PROMPT="A woman turns her head and smiles naturally." \
bash infer_openvid4000_3router_strongbind.sh
```

新的专用入口默认读取
`checkpoint/three_router_mq256_conn24_strongbind_openvid4000_steps500/latest`，
并在加载大模型前逐项核对 `clean_preserved_latent_slot`、单 reference slot、
timestep-0、目标首 latent 已移除、prefix 不参与 loss、无 soft anchor，以及 joint-null
仍保留 Wan reference 等训练契约。它不会接受旧 `legacysoft` checkpoint，也不允许
`first_frame_mode=none` 或训练外的 CFG null 模式。默认 `FRAME_NUM=49` 与训练一致；
如需长视频可显式设置其他 `4n+1` 帧数，例如 `FRAME_NUM=145`。

推理会严格恢复 `mq_encoder_trainable.*` 中更新后的三张 FP32 MQ 参数表和共享
Connector；还会从完整 MQ checkpoint 精确恢复冻结的 Qwen BOI/EOI/MetaQuery
特殊 token 行，同时恢复 `wan_dit_trainable.*` 中的 397 个 Wan 条件分支 tensor。
正向条件固定为 role=图片、action=文本、global=图片+文本，三路分别经过 Qwen
后再进行一次 256-token 共享 Connector forward。DiT 只接收 256 个 MQ token；
T5 仅复现训练期 RMS 匹配，不会再拼接成 768-token context。推理默认
`GUIDE_SCALE=1`，因此不计算训练外 unconditional forward；启用 CFG 时默认使用
与联合空条件训练一致的 `empty_mq`。首 latent 使用与训练相同的 timestep-0
preserved slot，并在每个 solver step 后重锁。默认 `full` 审计会执行去图/去文
消融、逐步 finite/RMS/absmax 检查，并把 Wan 对图片条件零响应判为失败。保存前
还会检查首帧/reference 相似度与生成帧空间高频比例，纯雪花不会报告成功。保存
前严格验证 `[C,T,H,W]`，写入 Wan writer 时补成 `[1,C,T,H,W]`；落盘后再
用 `ffprobe` 核验尺寸、帧数和 FPS。所有证据写入与视频同名的 `.verify.json`。

上传 Determined：

```bash
det experiment create infer_openvid4000_3router_strongbind.yaml .
```

目录中的 `.detignore` 会排除本地 checkpoint、临时数据和推理输出，避免把约
125 GB 的共享文件作为 experiment context 重复上传。

YAML 默认申请两张卡，Qwen/Connector 使用逻辑卡 0，Wan/VAE 使用逻辑卡 1。
单卡运行时可设置 `ENCODER_DEVICE=0 DIT_DEVICE=0`。

## 4×48GB strongbind 双模式训练与 mode0 推理

新的独立版本使用 4-rank Wan FSDP、全局有效 batch 8，两个模式都使用
target video 的第 0 帧作为 Wan strongbind 参考：

```bash
# 0: MQ 替代 T5
bash train/train_openvid4000_3router_4x48g.sh 0

# 1: MQ 经可训练 mapper 映射后与 frozen T5 token 拼接
bash train/train_openvid4000_3router_4x48g.sh 1
```

mode0 中，三路 Qwen 输出经单次共享 Connector 得到 256 个 MQ token，它们在
frozen T5 RMS 归一化后完全替代 T5 context。Wan 侧会移除重复的 target
首 latent，将干净参考 latent 作为第 0 slot，该 slot 的 token timestep 恒为 0、
不参与 loss，并在每次 Wan forward/采样步后重新锁定。这一路径不使用
random reference 或 soft anchor。

本次成功的 mode0 checkpoint 已经被下面的专用推理入口设为默认值：

```text
checkpoint/three_router_mq-replaces-t5_strongbind_openvid4000_4x48g_steps150/checkpoint-final
```

先执行不加载大模型的 checkpoint 和权重布局检查：

```bash
CHECK_ONLY=1 bash inference/infer_openvid4000_3router_4x48g_mode0.sh
```

再使用参考图和 prompt 进行正式推理：

```bash
REF_IMAGE=/path/to/reference.jpg \
PROMPT="A woman turns her head and smiles naturally." \
bash inference/infer_openvid4000_3router_4x48g_mode0.sh
```

该入口会严格拒绝 mode1、randomref 和 legacy-soft checkpoint；它会恢复三张路由
MQ 参数表、24 层 Connector、冻结 Qwen 的新增 special-token 行和 397 个 Wan
`cond_only` tensor。正式推理需要 CUDA；默认使用两张卡，也可通过
`ENCODER_DEVICE=0 DIT_DEVICE=0` 改为同卡。

详细机制、路径和限制见：

- `../design/3_router_planner_design.md`
- `../design/3_router_planner_implementation.md`
- `../analysis/openvid4000_training_log_analysis.md`

## 原生 Wan2.2 I2V-A14B 3-router

原生 I2V-A14B 版本单独组织在 `native_i2v_3router/`，训练入口和集群配置位于
`train/`，不会复用上面的 TI2V clean-prefix / `wan_animate_slot` 条件路径：

```text
native_i2v_3router/                         # 组合模块、原生 I2V contract、分布式工具
train/train_metaquery_i2v_3router_4x48g.py  # 历史文件名；当前默认 8-rank 双 DiT 训练入口
train/train_openvid4000_3router_i2v_a14b_4x48g.sh
train/train_openvid4000_3router_i2v_a14b_4x48g.yaml
tests/test_wan_i2v_3router_contracts.py
README_METAQUERY_I2V_A14B_3ROUTER.md
```

该版本严格使用原生 I2V `y=concat(mask[4], VAE(first_frame+zeros)[16])`
图像注入、high/low-noise 双模型边界以及 velocity MSE，不添加 latent prefix，
也不更改 Wan 内部网络结构。快速检查与正式启动：

```bash
python train/train_metaquery_i2v_3router_4x48g.py --check_only
bash train/train_openvid4000_3router_i2v_a14b_4x48g.sh
```

当前默认是 `conditioning_mode=0`、Wan `cond_only`、8 卡，并开启 Wan block
activation checkpointing；mode 1 仍可通过在 shell 后传 `1` 启动。

完整契约、两种 context mode 和 checkpoint 说明见
`README_METAQUERY_I2V_A14B_3ROUTER.md`。
