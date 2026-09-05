# 3-router planner 独立增量设计

> **2026-07-31 当前实现基线：** 本文后续的 144-token、64/48/32、route-type
> embedding 与强首帧绑定描述属于早期方案，已被当前代码取代。当前有效配置为
> 256 token（role/action/global = 96/96/64）、24 层 Qwen2Encoder Connector、
> 三张独立 FP32 路由表、Qwen 原始 embedding 整表冻结，以及 timestep-0
> clean preserved reference slot（条件前缀不参与 loss）。总损失只包含生成
> 视频与 ground-truth 视频的 velocity MSE。运行行为以
> `code/README.md` 和训练入口为准。

> 文档定位：只定义 `pipeline_design.md` 四层能力架构中的第一层
> **3-router planner**。本增量不实现 RSR/RAB、DIM、Oracle/Student、DSN
> 或 Wan 侧增强，目的是先验证“把 MetaQuery 显式分为角色/动作/全局三路”本身是否有效。

## 1. 结论

第一个可验证增量采用固定 144 个 Qwen3-VL MetaQuery token：

| 路由 | token 范围 | 数量 | 唯一职责 |
|---|---:|---:|---|
| `Q_role` | `[0, 64)` | 64 | 身份、外观、形体、长期稳定信息 |
| `Q_action` | `[64, 112)` | 48 | 动作意图、节奏、力度与可执行性 |
| `Q_global` | `[112, 144)` | 32 | 场景摘要、冲突协调、全局生成意图 |

三路仍在同一段 Qwen 序列中联合编码，但通过固定区间和可学习 route-type
embedding 得到显式类型。连续引导强度 `gs ∈ [0,1]` 经两层 MLP 编码后作为
共享增量写入全部 route token。训练时 `gs ~ Beta(2,2)`，推理时由调用方指定。

这一增量的输出仍交给现有 MetaQuery Connector，再作为 Wan 的 context 使用：

```text
I_ref + T
  -> Qwen3-VL 原始 MetaQuery hidden states [B,144,2048]
  -> 3-router planner
       Q_role   [B,64,2048]
       Q_action [B,48,2048]
       Q_global [B,32,2048]
  -> 现有 Qwen2 Connector
  -> Wan context [B,144,4096]
```

## 2. 为什么本次边界只到 planner

总设计将系统分为四层：

1. 3-route planner；
2. route interpreter（RSR/RAB/DIM）；
3. route adapter（Connector/DSN）；
4. Wan condition executor。

若第一次实验同时加入 RSR、DIM、双通道 cross-attention 等模块，即使指标变好，
也无法判断收益来自“显式三路”还是后续解释/注入机制。因此本增量只改变三件事：

- MetaQuery 总数从历史脚本默认的 256 固定为 144；
- 144 个 token 固定分为 64/48/32；
- 在 Connector 前加入 route type embedding 和 `gs` embedding。

Connector、Wan 主干、视频 flow-matching loss、数据解码与 checkpoint 结构全部复用
现有实现。这样 A/B 实验只需切换 `ROUTER_ENABLED=1/0`，两组都使用 144 tokens，
避免把 token 数量变化误当成路由收益。

## 3. 与总设计原则的对应关系

### 3.1 数据驱动复杂度

新增参数只有：

- 3 个 route type embedding；
- `1 -> 64 -> d_qwen` 的 GuidanceEncoder。

以本地 Qwen3-VL-2B 的 `d_qwen=2048` 计算，新增参数约 0.14M，远低于总设计
120M 的预算。第一次实验不加入新的 attention block。

### 3.2 冻结优先

默认冻结：

- Qwen3-VL backbone；
- Qwen 原始词表 embedding（启动脚本使用 `--freeze_mq_input_embeddings`）；
- Wan DiT（`WAN_TRAIN_MODE=frozen`）；
- Wan T5 与 VAE。

训练现有 Connector 与 3-router planner。若 Connector 已有可复用 checkpoint，
可用 `--resume_mq_encoder_path` 继续训练，减少从随机 Connector 开始的混杂因素。

### 3.3 连续模式统一

`gs` 不是离散模式标签：

- `gs=1`：更偏参考图身份保持；
- `gs=0`：更偏动作自由；
- 中间值：学习连续折中。

GuidanceEncoder 最后一层零初始化，使新分支初始为严格 no-op；训练开始时不会突然
扰动 Qwen hidden states。route type embedding 使用小高斯初始化。

### 3.4 非冗余

本层只声明“每个 token 属于哪种控制语义”，不做以下工作：

- 不重新读取外部证据（RSR/RAB 的职责）；
- 不让角色/动作做定向交互（DIM 的职责）；
- 不改变到 Wan 的分布（Connector/DSN 的职责）；
- 不改变 Wan 如何消费条件（Wan executor 的职责）。

## 4. 基础代码分析与设计修正

### 4.1 Qwen3-VL / MetaQuery

`model/Qwen3-VL-main/metaquery-main/models/model.py` 的有效实现已经提供关键基础：

- 为 backbone 扩展 `<begin_of_img>`、144 个 `<img{i}>` 和 `<end_of_img>`；
- 用 `MLLMInContext.tokenize` 把 MetaQuery 区放在输入序列尾部；
- `encode_condition` 通过 BOI/EOI 位置取出 MetaQuery hidden states；
- Qwen3-VL 的 `lm_head` 被替换为 `Identity`，因此 forward 的 `logits` 实际就是
  最后一层语言 hidden states；
- Connector 使用 Qwen2Encoder 与 MLP 把 Qwen hidden width 投影到目标条件宽度。

本地 `Qwen3-VL-2B-Thinking/config.json` 的 text hidden size 是 **2048**。总设计稿
中的 1536 应视为抽象宽度/早期估算，不可直接用于此 checkpoint。工程实现因此使用
2048，并在加载后强校验；否则在 Connector 的第一层就会产生维度错误。

### 4.2 Wan / MetaQuery 联训

`train_connector_for_wan.py` 的有效版本：

- 构造 `MetaQueryEncoderForWan`；
- 将 Connector 输出固定为 Wan text dimension 4096；
- 冻结 Qwen backbone，并控制 Connector/embedding 的可训练性；
- `WanVideoDataset` 返回 `caption/video/ref_image/mq_ref_image/video_path`。

`train_metaquery_wan.py` 的有效版本：

- 将 MetaQuery 特征组织为 `List[Tensor]` context；
- 调用 Wan VAE 编码视频；
- 以 flow matching 的 `noise - x0` 为目标；
- 支持冻结/条件层/全量 Wan 训练模式；
- 保存训练前、周期性与最终 checkpoint bundle。

本增量保持这些接口不变，planner 位于“Qwen raw route states”和“现有 Connector”
之间，因此 Wan 无需感知 planner 的存在。

### 4.3 参考启动脚本

`train_stage1_openvid_local_metaquery_overfit20_ti2v_frame.sh` 提供了可复用约定：

- OpenVid 本地视频与 CSV 配对；
- `frame_num=49`、单样本 batch、梯度累积；
- Qwen 与 Wan 可放在不同 GPU；
- 输出训练前与最终 checkpoint；
- 可通过 first-frame/MQ 条件驱动 TI2V。

原脚本中 `local_openvid_limit` 会在配对后随机打乱再截断，不能严格表示“CSV
前 100 条”。本增量另建确定性子集清单：按 CSV 顺序取前 100 条可解析记录，
以软链接构造小数据目录，并保存 `manifest.json` 以便复现实验。

## 5. 训练目标

第一增量只使用现有 Wan flow-matching loss：

```text
L_increment_1 = L_flow
```

原因：

- 这是最直接的下游可执行性检验；
- 不引入 Oracle/Student，保持增量边界；
- 不加入正交损失，避免 type embedding 通过几何捷径“伪造分工”。

route 间余弦与各路 RMS 只作为诊断，不作为损失。若三路在多轮训练后仍高度相似，
应进入第二增量（RSR/RAB 或 Oracle ground），而不是在本层偷偷加入解释器。

## 6. 实验与门禁

### 6.1 必跑 A/B

| 实验 | 3-router | MetaQuery 数 | 其余设置 |
|---|---:|---:|---|
| baseline-144 | 关闭 | 144 | 相同 |
| planner-144 | 开启 | 144 | 相同 |

主判断不是单次训练 loss，而是：

- 验证集 flow loss；
- 生成视频 CLIP-I / CLIP-T；
- route on/off 时 Wan 输出差异；
- 三路 pooled cosine 与 RMS 是否稳定；
- 参数增量与 wall time。

### 6.2 工程门禁

- 输入/输出严格为 `[B,144,2048]`；
- 分路形状严格为 64/48/32；
- `gs` 超出 `[0,1]` 必须失败；
- GuidanceEncoder 初始化时对 `gs=0/1` 输出一致；
- 梯度能到 route embedding、GuidanceEncoder 与 Connector；
- checkpoint 含 `router_planner.*` 和 `three_router_config.json`；
- baseline 和 planner 使用同一份 OpenVid100 manifest。

### 6.3 进入下一增量的条件

只有满足以下任一条件，才进入 RSR/RAB/DIM：

- planner-144 相对 baseline-144 有稳定下游收益，但三路余弦仍偏高；
- 三路存在可区分统计，但 Wan 对 route on/off 不敏感；
- 训练 loss 下降、生成指标不提升，说明需要更强语义监督。

若 planner-144 完全无收益，先检查 token 提取、checkpoint 加载与 route
on/off 等价性，不直接叠加更多模块。
