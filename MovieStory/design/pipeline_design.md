# Pipeline设计方案 v5 最终确定版

> **文档定位**：本文件为pipeline设计的**唯一最终执行口径**。综合 `pipeline_design_v5_unified.md`（实现详尽版）与 `pipeline_design_v5_unified_codex.md`（收敛精炼版）的优点，经系统性交叉审查后产出的确定版方案。与任何历史文档冲突时以本文件为准。
>
> **文档风格**：结论优先，机制解释次之；保留必要代码实现，不保留长篇审查过程。正文只保留唯一默认执行方案，候选增强独立列于§14。
>
> **核心目标**：给定任意动画角色参考图 `I_ref`、动作文本 `T`、可选参考动作视频 `V_ref`、连续引导强度 `gs ∈ [0,1]`，生成该角色执行对应动作的高质量视频。要求身份稳定、动作可读、非人泛化、长时序稳定。

---

## 0. 目标、约束与成功标准

### 0.1 目标

> 在不显式使用骨架的前提下，实现"任意动画角色进行任意动作"的高质量视频生成。

同时覆盖：人类与非人角色、标准与非标准骨架、强首图到纯Animate的连续模式、有无 `V_ref` 两种条件形态。

### 0.2 输入与输出

| 输入 | 说明 |
|------|------|
| `I_ref` | 角色参考图 |
| `T` | 动作文本描述 |
| `V_ref`（可选） | 参考动作视频 |
| `gs ∈ [0,1]` | 连续引导强度（1.0=传统I2V，0.0=纯Animate） |

**输出**：视频 `Y_hat`

### 0.3 设计约束

| 约束 | 含义 |
|------|------|
| 不用显式骨架 | 不依赖人体姿态估计或标准骨架工具 |
| 数据规模受限 | 高质量动画样本~3000，泛化视频样本~200K |
| 冻结优先 | 尽量复用 Qwen3-VL、T5、CLIP、Wan 预训练能力 |
| 非冗余优先 | 每个主线模块解决独立问题，不重复注入同类信息 |
| 单主线优先 | 正文只保留唯一默认执行方案，候选增强独立列出 |
| 参数预算 | 全链路新增可训练参数控制在 ~120M |

### 0.4 成功标准

| 维度 | 标准 |
|------|------|
| 身份稳定 | `gs=1.0` 时角色身份显著优于无route条件基线 |
| 动作表达 | `gs=0.5` 与 `gs=0.0` 时动作语义可读，CLIP-T优于纯文本基线 |
| 非人泛化 | 非人、夸张比例、动物角色不明显人形化 |
| 长时序 | 中后段帧无明显身份漂移和结构塌陷 |
| 条件利用 | 关闭route条件后质量有显著下降 |
| 主线简洁 | 无"前文保留后文删除"的口径冲突 |

---

## 1. 设计原则

1. **数据驱动复杂度**：3000动画样本约束下，每个模块必须有明确的梯度来源和收敛理由。
2. **冻结优先**：复用预训练模型（CLIP/T5/Wan主干），仅通过轻量可训练模块注入新能力。
3. **连续模式统一**：用单一 `gs` 参数控制I2V→Animate的连续过渡，不维护多套独立逻辑。
4. **优雅降级**：route tokens是增强而非必需——CSG + T5应能驱动基本生成质量。
5. **动画泛化**：不依赖人体特化工具（DWPose等），不依赖GradCAM等脆弱外部模型。
6. **非冗余**：每个模块解决一个不可由其他模块替代的问题。每条信息通路不与其他通路冗余。

---

## 2. 架构总览

### 2.1 四层能力架构

从系统实现视角，主线按4个能力层理解：

| 能力层 | 组成模块 | 职责 |
|--------|---------|------|
| **3-route planner** | `Q_角色 / Q_动作 / Q_全局` | 将角色、动作、全局三类控制显式分开 |
| **route interpreter** | `RSR + 轻量RAB + 双向DIM` | 高价值取证 + 角色-动作联合解释 |
| **route adapter** | `Connector + DSN` | MLLM空间 → Wan可消费空间 |
| **Wan condition executor** | 统一e0增强 + 解耦双通道CA + TCC + 身份持续保持包 | 让Wan全程正确消费route条件 |

其中**身份持续保持包** = `gs首帧 + CSG + CRLA + L_struct`，它们不是独立的平行主线，而是同一个身份保持目标的四个层级：初始构图、语义锚定、像素记忆、训练约束。

### 2.2 关键接口口径

```text
输入:
  I_ref + T + optional V_ref + gs

MLLM 输出:
  Q_角色(64) / Q_动作(48) / Q_全局(32)

Connector 输出:
  C_角色粗(64) / C_角色细(64) / C_动作(48) / C_全局(32)

Wan 条件接口:
  text_context  = T5 text tokens
  route_context = C_* + CSG（分层组合）
  e0_enhanced   = e0_base + global_delta + temporal_delta[per_frame]
```

### 2.3 系统架构图

```
┌─────────────── 输入 ─────────────────┐
│  I_ref（参考角色图）                   │
│  T（动作文本描述）                     │
│  V_ref（可选参考动作视频）              │
│  gs ∈ [0,1]                          │
└──────────────────────────────────────┘
        │
        ▼
┌── 证据预编码（冻结，一次性） ──────────┐
│  CLIP ViT → I_patch [~49,1536]       │
│           → I_global [8,1536]        │
│  T5-XXL  → T_text [L_t5,4096]       │
│  CLIP ViT → V_static [8,1536]       │
│           → V_motion [7,1536]        │
│  E_all = [I_patch‖I_global‖T_text    │
│           ‖V_static?‖V_motion?]      │
│  Oracle: E_oracle = [E_all           │
│           ‖Y_static‖Y_motion]        │
└──────────────────────────────────────┘
        │
        ▼
┌── MLLM路由规划（Qwen3-VL-2B） ──────┐
│  输入: [SYS][gs_emb][I_ref][T]       │
│        [V_ref?][BOI]                 │
│        [Q_角色×64][Q_动作×48]         │
│        [Q_全局×32][EOI]              │
│  RSR: Layer 13/27后各1次（含轻量RAB） │
│  Oracle/Student 双分支蒸馏            │
└──────────────────────────────────────┘
        │
        ▼
┌── DIM（双向，零初始化，无门控） ────────┐
│  Q_动作' = interact(Q_动作, Q_角色)    │
│  Q_角色' = interact(Q_角色, Q_动作')   │
│  Q_全局' = interact(Q_全局, [Q_角色'‖Q_动作']) │
└──────────────────────────────────────┘
        │
        ▼
┌── Route Adapter（Connector + DSN） ──┐
│  4层Transformer(144tok共享)          │
│  → 双头投影: C_角色粗/C_角色细        │
│  → C_动作, C_全局                    │
│  → DSN: 统计分布软对齐到Wan空间       │
└──────────────────────────────────────┘
        │
        ▼
┌── Wan 2.2 I2V 5B ────────────────────┐
│  连续gs首帧条件                       │
│  统一e0条件增强（全局+帧级时序）       │
│  Text/Route解耦双通道cross-attention  │
│  TCC时步条件调度                      │
│  CSG角色语义锚定                      │
│  CRLA角色像素记忆（blocks 5-25）      │
│  TA-LoRA(rank=32) + SA-LoRA(rank=8)  │
│  → VAE decode → 视频                 │
└──────────────────────────────────────┘
```

### 2.4 模块非冗余职责表

| 模块 | 解决什么独立问题 | 不与谁重复 | 不加会损失什么 |
|------|----------------|-----------|--------------|
| 3路route | 角色/动作/全局三类控制显式分开 | 不等于text，不等于首帧 | 条件表达混成一团 |
| RSR + 轻量RAB | 从证据池做学习性取证 | 不等于DIM（DIM读彼此，RSR读证据） | route学不到稳定分工 |
| 双向DIM | 角色与动作联合解释 | 不等于RSR | 动作不适配角色形体 |
| Route Adapter | MLLM空间→Wan空间接口 | 不等于CA LoRA（LoRA做Wan端适配） | route条件被Wan误读 |
| 统一e0增强 | route语义进入AdaLN全局调制通路 | 不等于cross-attn（通道级 vs token级） | Wan自注意力和FFN无条件感知 |
| 解耦双通道CA | text与route独立消费路径 | 不等于TCC（TCC调强度，CA给路径） | route被text系统性压制 |
| TCC | 不同去噪时步使用不同条件重点 | 不等于e0增强（时步级 vs 帧级） | 粗结构与细节条件互相打架 |
| CSG | 低gs时角色语义锚定 | 不等于CRLA（语义级 vs 像素级） | Animate模式下角色锚定不足 |
| CRLA | 中后段像素级角色记忆 | 不等于CSG | 中后段身份漂移 |
| L_struct | 训练侧角色区域一致性约束 | 不等于CSG/CRLA（训练约束 vs 推理路径） | 有注入但缺持续保持监督 |

### 2.5 模块结论总表

| 模块 | 决策 | 参数量 | 训练阶段 |
|------|------|--------|---------|
| 语义路由分解（3路） | **保留** | — | Stage 1 |
| 证据编码（统一池） | **保留** | ~12M | Stage 1 |
| RSR（2次，含轻量RAB） | **保留** | ~4M | Stage 1 |
| DIM（双向，零初始化） | **保留** | ~3M | Stage 1 |
| Oracle/Student蒸馏 | **保留** | — | Stage 1 |
| Connector + DSN | **保留** | ~30M | Stage 2 |
| 连续模式谱 | **保留** | — | Stage 2 |
| 统一e0条件增强 | **保留** | ~4M | Stage 2 warmup |
| 解耦双通道交叉注意力 | **保留** | ~7M | Stage 2 main |
| CSG（固定token） | **保留** | ~4M | Stage 2 warmup |
| TCC（基函数+残差） | **保留** | ~0.003M | Stage 2 main |
| CRLA（固定48 token） | **保留** | ~4M | Stage 2 main |
| TA-LoRA (rank=32) | **保留** | ~40M | Stage 2 main |
| SA-LoRA (rank=8) | **保留** | ~5M | Stage 2 main |
| L_struct（角色区域版） | **保留** | — | Stage 2 main |
| 对比损失 + 正交正则 | **默认启用** | — | Stage 1 |
| text/route独立Dropout | **保留** | — | Stage 2 |
| **总可训练参数** | | **~113M** | |

**不纳入主线的模块**：

| 模块 | 不纳入原因 |
|------|-----------|
| ~~注意力缩放偏置~~ | 解耦CA从根本上解决T5/route竞争 |
| ~~条件化LoRA缩放~~ | 候选增强，主线稳定后消融验证（详见§14） |
| ~~动作时序注意力偏置~~ | 候选增强，与TCC+TA-LoRA目标部分重叠（详见§14） |
| ~~ChRA-V多层参考注意力~~ | CSG+CRLA已覆盖 |
| ~~T_a/T_s文本分流~~ | 线性投影无法分离语义 |
| ~~5路route~~ | 监督稀释，分工容易重复 |

---

## 3. 关键设计决策

### D1：为什么保持3路而不恢复5路？

5路（外观/结构/动作/动态/场景）的问题不在概念上，而在数据与监督上：
- 外观身份 / 身体结构的证据来源高度重叠（都从参考图读取），RSR难以发展出显著不同的读取策略
- 动作语义 / 动态属性同理（都从文本和视频读取）
- 路由变多后每路有效监督更稀，3000动画样本下更难判断"分工失败来自哪一路"

3路的足够性：
- Q_角色编码所有角色相关视觉信息 → Connector双头投影分为粗结构和细节
- Q_动作编码所有动作相关语义
- Q_全局作为协调器汇总两路
- 粗/细分离放在Connector层做，不需要MLLM层面独立路由

### D2：为什么DIM升级为双向版且不用门控？

原单向DIM只有Q_动作←Q_角色，Q_角色得不到动作反馈，Q_全局完全不参与。

双向DIM：Q_动作先知道角色约束 → Q_角色再知道动作对外观的要求 → Q_全局最后做整合。共享参数的单次交互在144 tokens规模下已有充分信息提取能力。

门控移除理由：零初始化输出投影已保证训练初期 `proj(x)=0`，gate再乘上去只是 `0×0=0`，完全冗余。DIM的交互强度应在训练中固化，推理时无需动态调节。

### D3：为什么用连续模式谱？

离散3模式（ti2v/animate/animate_soft）边界人为、每种模式只占部分数据、推理无法精细控制。连续 `gs ∈ [0,1]` 训练时从 Beta(2,2) 采样，见过更多工况，推理时可精确控制。

### D4：为什么用解耦双通道CA而非注意力偏置？

T5 tokens (~512个) 在softmax中天然压制route tokens (~144个)。注意力偏置只调logit不改K/V计算，是局部补偿；解耦CA让text和route走独立通路，从根本上消除零和竞争。

### D5：为什么统一e0条件增强包含帧级时序？

Wan原始e0 = timestep + T5_pool，所有帧在同一去噪步使用完全相同的e0。但"跳跃"动作中起跳帧和腾空帧应有不同的AdaLN调制。帧级时序增量通过 `C_动作 + 帧位置嵌入` 生成每帧独立的e0偏移。零初始化，初期无影响，训练中逐步学习。

### D6：为什么L_contrast作用在DIM之前？

DIM之后的 `Q_动作'` 已经包含角色形体适配（猫的跳跃和蛇的跳跃应该不同），如果在post-DIM tokens上强求动作不变性，会和"任意角色做任意动作"的目标冲突。因此对比损失施加在DIM之前的route seed上：约束 `Q_角色^0` 在不同动作下保持稳定、`Q_动作^0` 在不同角色下保持语义主轴稳定。

---

## 4. MLLM侧设计

### 4.1 MLLM职责边界

MLLM端定位为**角色-动作联合解释器**，只做三件事：
- 把参考角色、动作文本、可选参考视频联合解释为高价值控制信息
- 判断"这个角色执行这个动作时，哪些信息必须稳定、哪些应可变"
- 把结果压缩为Wan可消费的route条件

MLLM **不**做：不输出显式骨架、不输出逐帧pose计划、不承担物理仿真、不替代Wan做时间演化。

### 4.2 路由定义与token分配

| 路由 | token数 | 职责 | 不负责什么 |
|------|---------|------|-----------|
| `Q_角色` | 64 | 身份、外观、形体、长期稳定信息 | 不负责开放域动作语义主表达 |
| `Q_动作` | 48 | 动作意图、节奏、力度、可执行性 | 不负责长期身份锚定 |
| `Q_全局` | 32 | 汇总协调、场景摘要、冲突调解 | 不做单独外部取证 |
| **总计** | **144** | | |

每路携带可学习的路由类型嵌入（3种 × 1536维，小值高斯初始化）。

固定3路 / 144 tokens的原因：
- 优先提高单token信息密度，而非靠堆更多token获得表面容量
- 只有在证明3路已明确出现容量瓶颈时，才允许上调token数

### 4.3 gs编码

```python
class GuidanceEncoder(nn.Module):
    """将连续gs标量映射为MLLM前缀嵌入"""
    def __init__(self, d=1536):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, d))
    def forward(self, gs):  # gs: [B, 1]
        return self.mlp(gs)  # [B, 1536]
```

这里的 `gs` 不是普通超参数，而是**整条生成链路的工作模式坐标**：`gs=1.0` 更接近传统I2V，强调参考图锁定；`gs=0.0` 更接近纯Animate，强调动作与生成自由度；中间值表示连续过渡。把它只留到Wan侧再使用是不够的，因为MLLM在做route规划时就已经需要知道：当前到底应该更偏向"守住角色"还是更偏向"放开动作"。

因此，`4.3` 的目标不是把一个标量简单升维，而是把 `gs` 变成一个能被Qwen3-VL所有后续层消费的**全局模式条件**。实现上采用一个极小的MLP，把 `gs ∈ [0,1]` 映射为 `gs_emb ∈ R^1536`，并作为前缀token插入主序列。

**完整流程**：

1. 训练时从 Beta(2,2) 采样 `gs`，推理时由用户直接指定。
2. `GuidanceEncoder` 将单标量 `gs` 编码成 `gs_emb`。
3. `gs_emb` 放在 `[SYS]` 之后、所有视觉/文本/route token 之前。
4. 由于它位于最前缀位置，后续所有token在因果注意力下都天然可见该条件。
5. route token在后续Qwen层、两次RSR、以及最终DIM前形成的隐藏状态，都会隐式带上 `gs` 的模式信息。

可概括为：

```python
gs_emb = GuidanceEncoder(gs)          # [B, 1536]
seq = [SYS, gs_emb, I_ref_tokens, T_tokens, V_ref_tokens?, BOI,
       Q_role, Q_action, Q_global, EOI]
```

**为什么一定要作为token前缀，而不是后面拼一个标量**：

- Qwen3-VL本来就擅长消费token化条件；把 `gs` 变成token，最符合主干的预训练接口形式；
- 作为前缀token，它能在所有层持续影响route规划，而不是只在末端做一次硬开关；
- `gs` 进入MLLM后，影响的不只是强度大小，还会影响"该去读哪些证据、该怎样解释角色和动作的张力"。

**它对三路route的语义影响**可以这样理解：

- 当 `gs` 较高时，`Q_角色` 更应坚持参考图中的身份、外观与局部结构；`Q_动作` 需要在不破坏角色稳定性的前提下表达动作。
- 当 `gs` 较低时，`Q_动作` 可以更开放地吸收 `T_text / V_motion` 的动态先验；`Q_角色` 仍负责身份锚定，但不应把首图构图和静态姿态强行锁死。
- 当 `gs` 处于中间值时，三路route要学习一种折中：既保留角色可识别性，又给动作和镜头更多生成自由。

所以，`gs` 调的不是某一个模块的单独权重，而是**整个系统对"身份保持 vs 动作自由"这条连续谱的操作点**。

**作用边界**也要固定：

- `gs` 是全局模式条件，不提供具体动作语义，不能替代文本 `T`；
- `gs` 不提供角色身份内容，不能替代参考图 `I_ref`；
- `gs` 不进入 `E_all` 作为"证据"，因为它不是外部观测，而是生成偏好的全局控制信号。

参数量：~0.1M。

### 4.4 证据编码

所有证据基于冻结模型提取，通过可学习投影层统一到1536维：

| 证据 | 形状 | 说明 |
|------|------|------|
| `I_patch` | [~49, 1536] | 参考图CLIP patch tokens |
| `I_global` | [8, 1536] | 参考图CLIP [CLS] → MLP展开 |
| `T_text` | [L_t5, 1536] | T5 last-hidden → 投影 |
| `V_static` | [8, 1536] | 参考视频CLIP关键帧（若有） |
| `V_motion` | [7, 1536] | CLIP语义帧差（若有） |
| `Y_static` | [8, 1536] | GT视频均匀采样关键帧，经与 `V_static` 共享的投影得到（Oracle only） |
| `Y_motion` | [7, 1536] | 相邻GT关键帧CLIP语义差分，经与 `V_motion` 共享的投影得到（Oracle only） |

统一证据池：`E_all = [I_patch ‖ I_global ‖ T_text ‖ V_static? ‖ V_motion?]`

**Oracle特权证据 `Y_static / Y_motion` 的获得方式**：

1. 将GT视频 `Y_gt` 解码为RGB帧序列 `F_1 ... F_T`。
2. 沿时间轴做**等间隔采样**，固定取 `K_y = 8` 帧。采样索引定义为  
   `τ_k = round((k-1) * (T-1) / (K_y-1)) , k=1...8`。  
   若 `T < 8`，允许最近邻重复采样，仍保持固定8帧输出形状。
3. 对采样帧做与 `V_ref` 完全一致的CLIP预处理（resize / crop / normalize），送入同一个**冻结CLIP图像编码器**，得到全局特征  
   `g_k = CLIP_cls(F_(τ_k))`。
4. 构造静态特权证据：  
   `Y_static = Proj_Vs([g_1, ..., g_8]) ∈ R^(8×1536)`。  
   它保留的是跨帧相对稳定的角色身份、外观、轮廓与局部结构线索。
5. 构造运动特权证据：先在CLIP语义空间做相邻帧差分  
   `Δg_k = g_(k+1) - g_k , k=1...7`，再投影  
   `Y_motion = Proj_Vm([Δg_1, ..., Δg_7]) ∈ R^(7×1536)`。  
   它表达的是动作方向、节奏变化和语义运动趋势，而不是像素级光流。
6. 为保持RAB仍然只有5个evidence-group，`Y_static` 归入 `V_static` 组，`Y_motion` 归入 `V_motion` 组，不额外新增偏置参数。

可写成：

```python
def extract_oracle_evidence(Y_gt):
    frames = decode_video(Y_gt)                               # [T]
    idx = round(linspace(0, len(frames) - 1, 8))             # [8]
    key_frames = [frames[t] for t in idx]

    g = [CLIP_cls(preprocess(f)) for f in key_frames]        # 8 x [768]
    Y_static = Proj_Vs(stack(g))                             # [8, 1536]

    dg = [g[k + 1] - g[k] for k in range(7)]                 # 7 x [768]
    Y_motion = Proj_Vm(stack(dg))                            # [7, 1536]
    return Y_static, Y_motion
```

这样做有两个直接好处：

- `Y_static / Y_motion` 与 `V_static / V_motion` 使用**同一套模态提取与投影口径**，Oracle和Student更容易在同一特征流形上对齐；
- `Y_motion` 采用CLIP语义帧差，而不是像素级光流，表达的是"动作语义和节奏如何沿时间变化"，这更适合当前route级监督粒度。

**V_ref使用边界**（必须固定）：
- `V_ref` 默认只提供动作先验和节奏线索，不提供角色身份锚定
- 若 `V_ref` 与 `I_ref` 的身份或形体先验冲突，以 `I_ref/Q_角色` 为主
- 没有 `V_ref` 时，`Q_动作` 应更多依赖 `T_text + Q_角色` 做角色适配，不伪造稠密运动结构

### 4.5 RSR（路由专化证据读取）

在Qwen3-VL-2B的28层中，Layer 13和Layer 27后各插入1个RSR模块，两次RSR共享参数。

**只给 `Q_角色` 和 `Q_动作` 使用**，`Q_全局` 不做RSR——它不应复制"证据读取"职责，而应在MLLM自注意力和DIM中做汇总协调。

每路一套RouteCA（标准cross-attention + FFN），从 `E_all` 中学习性选择证据。Oracle和Student共享RSR参数。

RSR的本质不是再造一层"大融合"，而是给route token一个显式的、可学习的"出去找证据"动作。没有RSR时，`Q_角色 / Q_动作` 主要依赖Qwen3-VL内部自注意力被动吸收上下文，很容易两路都混到相似信息；加入RSR后，两路route在中途各自对证据池发起一次外部读取，分工会清晰得多。

**单次RSR内部流程**：

1. 从当前Qwen层输出中取出该层的 `Q_角色^(l)` 和 `Q_动作^(l)`。
2. 将统一证据池 `E_all` 作为Key/Value；route token自身作为Query。
3. `Q_角色` 进入 `RouteCA_角色`，`Q_动作` 进入 `RouteCA_动作`；两路参数彼此独立，因此可以学到不同读取策略。
4. 在cross-attention中，对 `E_all` 的所有证据token做软选择，读取最有用的信息写回route状态。
5. 经过残差连接和FFN，再把更新后的route token送回主干，继续参与后续Qwen层的自注意力计算。

可写成：

```python
def rsr_once(Q_role, Q_action, E_all):
    Q_role = Q_role + CrossAttn(LN(Q_role), LN(E_all), LN(E_all))
    Q_role = Q_role + FFN(LN(Q_role))

    Q_action = Q_action + CrossAttn(LN(Q_action), LN(E_all), LN(E_all))
    Q_action = Q_action + FFN(LN(Q_action))
    return Q_role, Q_action
```

这里的关键点是：**RSR只更新route token，不改写证据池本身**。因此它更像"route去读证据"，而不是让所有token再次彼此混合。

**两次RSR的时序作用不同**：

- **Layer 13后的第一次RSR**：偏"粗取证"。这时route token还比较接近初始规划状态，第一次读取主要建立方向感。`Q_角色` 先把身份、外观、形体等线索抓住；`Q_动作` 先把文本动作语义、节奏先验、视频运动趋势抓住。
- **Layer 27后的第二次RSR**：偏"精取证/纠偏"。经过中间十几层Qwen自注意力后，route已经带有更丰富的上下文理解，第二次读取会更有针对性，比如补细节、修正第一次读错的重心、把动作语义和角色形体约束对齐。

两次RSR**共享参数**的含义，不是两次做一模一样的事，而是"用同一个读取算子，在不同上下文状态下执行两次"。第一次解决"先找到哪类证据"，第二次解决"结合上下文后再精修一次"。这样做有三点好处：

- 降低参数量，避免3000样本规模下过拟合；
- 强迫两次读取遵循同一种route语义，不会前后层各学一套不一致规则；
- 让Oracle和Student在同一个读证据机制上对齐，蒸馏时更稳定。

**为什么 `Q_全局` 不做RSR**：

- `Q_全局` 的职责是协调，不是抢证据；
- 如果它也直接去读 `E_all`，很容易和 `Q_角色 / Q_动作` 发生职责重叠，削弱route分工；
- 让 `Q_全局` 留在Qwen自注意力与DIM里做汇总，等于先让两条专路线完成取证，再由全局路做整合和冲突调解。

所以这里的角色边界是明确的：

- `Q_角色`：读"这个角色是谁、长什么样、身体结构如何"
- `Q_动作`：读"要做什么动作、动作节奏如何、怎样落到该角色身上"
- `Q_全局`：不单独外出取证，只负责最后的整体协调

从信息流角度看，RSR与DIM也不是一回事：

- **RSR**：`route -> evidence`，解决"各路分别去哪里取证"
- **DIM**：`route -> route`，解决"取完证之后两路如何彼此解释、再由全局汇总"

参数量：2套RouteCA ≈ ~4M。

### 4.6 轻量RAB（路由注意力偏置）

RAB仅在RSR内部使用，提供route到证据源的归纳偏置。

设计：
- `route × evidence-group` 的共享learnable bias
- evidence-group = {I_patch, I_global, T_text, V_static, V_motion}
- `attn_logits(q_r, k) += b[r, group(k)]`，其中 `r ∈ {角色, 动作}`
- 两次RSR共享同一组偏置
- 粒度为组级（非token级），控制参数量避免3000样本下过拟合

推荐初始化：Q_角色对 `I_patch/I_global` 轻微正偏置（+0.1），Q_动作对 `T_text/V_motion` 轻微正偏置（+0.1），其余为0。

RAB可以理解为：在"内容相似度打分"之外，再额外告诉模型一个很弱的先验提示，即"哪一路通常应该先看哪类证据"。它不是硬规则，也不是mask，更不是手工路由；它只是给softmax前的attention logits轻轻推一把。

**加入位置**就在RSR的cross-attention内部：

```python
logits[i, j] = (q_i @ k_j) / sqrt(d_head) + b[route_id, group(j)]
alpha = softmax(logits, dim=-1)
out = alpha @ V
```

其中：

- `route_id ∈ {角色, 动作}`
- `group(j)` 表示第 `j` 个证据token属于哪一组
- `b[route_id, group(j)]` 是一个标量，并广播到该组内所有token

因此，RAB做的事情很朴素：**在内容匹配分数之外，额外给某些证据组一个小的先验加分或减分**。例如：

- `Q_角色` 对 `I_patch / I_global` 有 `+0.1`，意味着在别的条件接近时，它更倾向先看图像外观与整体结构；
- `Q_动作` 对 `T_text / V_motion` 有 `+0.1`，意味着它更倾向先看动作语义和时间变化；
- 其余项初始化为0，表示不预设强烈偏好，允许训练自己决定是否需要跨组读取。

**它的作用机制**有三个层面：

1. **加快早期收敛**：模型不必完全从零开始摸索"角色更该看外观、动作更该看文本/运动"这类常识性分工。
2. **减少路由塌缩**：当训练样本少、证据池又异质时，两路route容易都盯着最显眼的一类证据；RAB会给它们一个轻量分流倾向。
3. **保留内容驱动灵活性**：因为它只是加到logits上的小偏置，真正是否读取仍由 `q·k` 相似度决定，所以不会阻止 `Q_动作` 在需要时去看 `I_global` 做形体适配，也不会阻止 `Q_角色` 在需要时读取 `T_text` 补身份描述。

**为什么只做到group级，而不做到token级/位置级**：

- token级偏置参数太多，3000样本下很容易把偏置学成记忆表；
- group级偏置只表达"这一路通常先看哪类来源"，正好匹配这里需要的归纳偏置粒度；
- 10个标量几乎不增加容量，却能稳定地把route分工往正确方向推。

**为什么两次RSR共享同一组RAB**：

- 这样早期粗取证和后期精取证遵循同一套来源偏好，不会前后层自相矛盾；
- RAB表达的是"route类型对证据来源的先验"，它应当是跨层稳定的，而不是每次读取都重新定义一遍。

**需要特别强调**：RAB只在RSR里生效，不进入Qwen普通自注意力，也不进入DIM/Wan。也就是说，它只负责帮助route在"外出取证"时更快找到重点，不负责后续route间的语义融合。

当某组证据缺失时（如没有 `V_ref`，因此没有 `V_static / V_motion`），该组对应token本来就不在 `E_all` 中，RAB不会制造虚假信息；`Q_动作` 会自然把注意力重新分配给 `T_text` 以及必要的视觉形体线索。

参数量：10个标量（2路由 × 5组），可忽略。

### 4.7 注意力策略

1. 非路由token行：causal mask（保持Qwen3-VL原始行为）
2. 路由token行：可读取全部前缀 + 路由区域内双向注意力
3. gs_emb token作为前缀，所有token可见

这里采用的是**混合注意力策略**，目标很明确：既要尽量保留Qwen3-VL原有的序列建模习惯，又要让route token具备"集中读前缀证据并彼此协商"的能力。

先把序列分成两块来理解：

- **前缀区 `P`**：`[SYS][gs_emb][I_ref][T][V_ref?][BOI]`
- **路由区 `R`**：`[Q_角色×64][Q_动作×48][Q_全局×32][EOI]`

对应的mask规则是：

```python
if query_i in P:
    key_j visible iff j <= i          # 标准causal
elif query_i in R:
    key_j visible iff j in P or j in R  # 读全部前缀 + 路由区双向
```

也就是说：

- 前缀区token仍按Qwen原生的因果方式编码，不因为route存在而被反向改写；
- 路由区token位于序列尾部，因此天然可以读到整个前缀；
- 额外放开路由区内部的双向注意力，让 `Q_角色 / Q_动作 / Q_全局` 在每层里都能做轻量协商。

**这套策略的具体信息流**如下：

1. 图像token、文本token、可选视频token先在前缀区按原生causal方式形成上下文表示。
2. route token到达自身位置后，读取此前全部前缀上下文，因此可以直接感知 `I_ref / T / V_ref / gs_emb`。
3. route token之间再在局部双向交流，形成一轮粗层面的任务分配与上下文共享。
4. 在Layer 13/27插入的RSR中，route再显式对 `E_all` 做外部取证；最后在DIM中完成更强的跨route解释。

所以 `4.7` 不是在替代 `RSR` 或 `DIM`，而是在给它们提供一个合理的底座：

- 混合mask负责**不破坏Qwen原有编码方式**；
- RSR负责**route对证据池的显式外部读取**；
- DIM负责**route之间最终的定向信息调制**。

**为什么前缀区不直接改成全双向**：

- 那会显著改变Qwen3-VL的预训练行为，训练初期更容易不稳定；
- 前缀token如果能被后面的route反向改写，证据编码会和route规划纠缠在一起，不利于职责清晰；
- 在当前样本规模下，保留原生causal是更稳妥的改法。

**为什么路由区内部要双向，而不是继续严格causal**：

- 如果route区也完全causal，后面的route能看到前面的route，前面的route却看不到后面的route，三路分工会受到位置顺序的人为影响；
- 双向后，`Q_角色 / Q_动作 / Q_全局` 在同一层里拥有更对称的信息交换条件，便于形成稳定分工；
- 这种双向只发生在很小的route子空间中，计算和干扰都可控。

**`gs_emb` 作为前缀、所有token可见** 的含义也要明确：

- 它不需要额外广播到每一路route；
- 只要放在足够靠前的位置，所有后续token在causal/route mask下都会自然看到它；
- 因而 `gs` 的模式信息会从最早层开始持续影响整段route规划，而不是到末端才被动注入。

从系统作用上看，`4.7` 解决的是一个接口问题：**让route能充分读取前缀信息，但不让它反向污染前缀证据编码**。这正好匹配本方案"前缀负责编码证据，route负责汇总和解释证据"的职责划分。

### 4.8 Oracle/Student双分支

**Oracle**（训练时存在，推理移除）：序列含 `Y_gt` 关键帧token，证据池扩展为 `E_oracle`。

**Student**（训练+推理）：仅用 `I/T/V` 证据。

共享：Qwen3-VL backbone、RSR（RouteCA）、路由类型嵌入。
独立：DIM参数（Student / Oracle 各一套）。

这一节的核心思想是**特权信息蒸馏**：在训练期，让一个能看到 `Y_gt` 关键信息的Oracle分支先学会"正确的route应该长什么样"，再把这种route语义蒸馏给真正部署时可用的Student分支。这样做的目的，是在仅有3000动画样本的条件下，尽快把 `Q_角色 / Q_动作 / Q_全局` 训成有意义的控制表示，而不是把希望都压在Wan端慢慢自己摸索。

先区分两条分支各自看到什么：

- **Student**：只看推理时真实可得的信息，即 `I_ref + T + V_ref? + gs`
- **Oracle**：在Student同样输入的基础上，再额外看到从 `Y_gt` 提取的特权证据，如 `Y_static / Y_motion`

实现上，Oracle新增的 `Y_gt` 关键帧token进入同一条MLLM前缀，并同时被纳入扩展证据池 `E_oracle`。因此Oracle不仅能在普通自注意力里接触这些GT线索，也能在RSR取证时直接读取到它们。

**训练时的完整流程**：

1. 对同一个样本构建基础证据池  
   `E_all = [I_patch ‖ I_global ‖ T_text ‖ V_static? ‖ V_motion?]`
2. 再为Oracle构建扩展证据池  
   `E_oracle = [E_all ‖ Y_static ‖ Y_motion]`
3. Student与Oracle共享 `gs_emb`、路由类型嵌入、Qwen3-VL backbone 与 RSR。
4. Student用 `E_all` 完成两次RSR取证；Oracle用 `E_oracle` 完成同构的两次RSR取证。
5. MLLM主干结束后，两条分支分别进入各自的DIM：
   - Student DIM：学习如何仅凭可部署证据做角色-动作联合解释
   - Oracle DIM：学习在GT特权信息辅助下形成更理想的联合解释
6. Oracle分支通过**拆分式** `L_oracle_ground` 直接锚定三路route：
   - `Q_角色^O ↔ Y_static`
   - `Q_动作^O ↔ Y_motion`
   - `Q_全局^O ↔ 多帧CLIP(Y_gt)`
7. Student分支通过**三路** `L_route_distill` 去逼近Oracle的route输出。
8. 若batch中存在有效pair，再用低权重 `L_contrast` 稳定Student在pre-DIM阶段的route分工。

可以概括成：

```python
# shared
gs_emb = GuidanceEncoder(gs)

# student
Q_s = Qwen_RSR(seq=[SYS, gs_emb, I_ref, T, V_ref?, BOI, routes, EOI], evidence=E_all)
Q_s = DIM_student(Q_s)

# oracle
Q_o = Qwen_RSR(seq=[SYS, gs_emb, I_ref, T, V_ref?, Y_gt_tokens, BOI, routes, EOI],
               evidence=E_oracle)
Q_o = DIM_oracle(Q_o)

L = L_oracle_ground(Q_o, Y_static, Y_motion, Y_gt) \
  + L_route_distill(stopgrad(Q_o), Q_s) \
  + λ_pair * L_contrast(Q_s_pre_DIM, valid_pairs)
```

**各损失在这里的职责**：

- `L_oracle_ground`：先把Oracle三路route各自锚到对应GT目标上，让教师本身真正"看懂答案"。
- `L_route_distill`：让Student的 `Q_角色 / Q_动作 / Q_全局` 分别逼近Oracle的三路语义。
- `L_contrast`：只作为低权重辅助项，稳定Student在pre-DIM阶段的角色/动作分工。

当前默认主线**不再单列** `L_global_distill` 和 `L_orthogonal`：

- `L_global_distill` 被并入 `L_route_distill` 的三路平均项中，单列已无必要；
- `L_orthogonal` 在旧口径里主要用来防止 `Q_角色 / Q_动作` 塌缩；但在新口径下，这两路已经分别被 `Y_static / Y_motion` 直接ground，再加上route-wise distill，主因子分工已经足够明确，继续强加几何正交先验反而可能惩罚合理共享语义。

只有当后续实验仍观察到明显route塌缩时，`L_orthogonal` 才作为**回滚辅助项**恢复，默认权重不超过 `0.05`。

**为什么共享Qwen/RSR，却不共享DIM**：

- Qwen backbone 与 RSR承担的是"从可见证据里抽route基础表示"这件事，这应该是两条分支共享的核心能力；
- 但DIM承担的是"角色与动作如何彼此解释"。Oracle因为能看到 `Y_gt`，其跨route交互统计会更强、更理想，也更接近教师态；
- 如果DIM也强行共享，Student和Oracle会在同一套交互参数上拉扯，容易产生梯度冲突，并把教师分支的特权耦合方式硬灌给Student。

因此，本方案选择：**共享取证机制，分开解释机制**。这样Student能学到Oracle的目标方向，但不必完全复制其依赖GT特权信息的内部交互细节。

**推理阶段的行为**非常简单：

- 直接删除Oracle分支；
- 不再构建 `Y_gt` token，也不再需要 `E_oracle`；
- 仅保留Student：`I_ref + T + V_ref? + gs -> route tokens -> DIM -> Connector -> Wan`

这意味着Oracle不是第二条推理路径，而只是训练期教师。它的作用是帮助Student更快、更稳地学会"什么叫好的route表示"；真正落地部署时，系统仍然只有一条Student主线。

从作用上讲，`4.8` 解决的是**弱监督下route语义难以成形**的问题。没有Oracle时，Student只能通过间接损失慢慢猜；有了Oracle，Student就能对着一个"看过答案的人"学，从而更快把角色路由、动作路由和全局路由分开学清楚。

Stage 1损失（2项主损失 + 1项可选辅助）：

```
L_stage1 = 1.0 * L_oracle_ground    # Oracle三路direct ground
         + 1.0 * L_route_distill    # Student三路蒸馏
         + λ_pair * L_contrast      # 仅pre-DIM；有有效跨样本pair时 λ_pair=0.2，否则 λ_pair=0
```

**统一记号**：

- `Q_r^S / Q_r^O`：Student / Oracle 在 **DIM之后** 的 route token，`r ∈ {角色, 动作, 全局}`
- `Q_角色^{S,0} / Q_动作^{S,0}`：Student 在 **DIM之前** 的 route seed
- `pool(Q)`：先沿token维做mean pooling，再做L2归一化  
  `pool(Q) = normalize((1/N) Σ_i Q_i)`
- `cos(a, b)`：余弦相似度  
  `cos(a, b) = (a^T b) / (||a|| ||b||)`
- `sg(·)`：stop-gradient，只把右侧张量当作目标，不回传梯度

默认训练中，先得到：

```python
# Student / Oracle post-DIM
u_char_s = pool(Q_角色^S)
u_act_s  = pool(Q_动作^S)
u_glb_s  = pool(Q_全局^S)

u_char_o = pool(Q_角色^O)
u_act_o  = pool(Q_动作^O)
u_glb_o  = pool(Q_全局^O)

# Student pre-DIM
u_char0 = pool(Q_角色^{S,0})
u_act0  = pool(Q_动作^{S,0})
```

**① `L_oracle_ground(Q_o, Y_static, Y_motion, Y_gt)`：Oracle三路direct ground**

旧口径里，`L_oracle_ground` 只直接约束 `Q_全局^O`，这会导致 `Q_角色^O / Q_动作^O` 主要靠间接梯度学习。新口径改为**三路分别ground**，让Oracle的角色路、动作路、全局路都直接看到自己对应的GT目标。

先定义Oracle侧GT目标摘要：

```python
u_y_static = pool(Y_static)                                          # [B, 1536]
u_y_motion = pool(Y_motion)                                          # [B, 1536]
y_gt_glb   = normalize((1/K) * Σ_k CLIP_cls(Y_gt^(τ_k)))             # [B, 768]
```

其中 `τ_k` 是从 `Y_gt` 中均匀采样的 `K = 4` 个关键帧索引。

三路ground分别为：

```python
L_char_ground^O =
    (1/B) * Σ_b [1 - cos(u_char_o[b], sg(u_y_static[b]))]

L_action_ground^O =
    (1/B) * Σ_b [1 - cos(u_act_o[b], sg(u_y_motion[b]))]

v_glb_o = mean(Q_全局^O, dim=token)                                  # [B, 1536]
z_glb_o = CLIPAlignHead(v_glb_o)                                     # [B, 768]

L_global_ground^O =
    (1/B) * Σ_b [1 - cos(z_glb_o[b], sg(y_gt_glb[b]))]
```

最终：

```python
L_oracle_ground =
    (L_char_ground^O + L_action_ground^O + L_global_ground^O) / 3
```

`CLIPAlignHead` 仍是一个小MLP投影头，默认可写为：

```python
CLIPAlignHead(v) = normalize(
    W2(LayerNorm(GELU(W1(v))))
)
```

**计算流程**：

1. 从GT视频提取 `Y_static / Y_motion`，并额外均匀采样 `K=4` 帧用于全局CLIP目标。
2. Oracle前向得到 post-DIM 的 `Q_角色^O / Q_动作^O / Q_全局^O`。
3. `Q_角色^O` 与 `pool(Y_static)` 做余弦对齐。
4. `Q_动作^O` 与 `pool(Y_motion)` 做余弦对齐。
5. `Q_全局^O` 先过 `CLIPAlignHead`，再与多帧 `CLIP_cls(Y_gt)` 的平均做余弦对齐。
6. 三项取平均，得到 `L_oracle_ground`。

**计算原理**：

- `Q_角色` 应直接锚定GT视频里**跨帧保持稳定**的身份/外观信息，因此对齐 `Y_static`；
- `Q_动作` 应直接锚定GT视频里**沿时间变化**的动作语义主轴，因此对齐 `Y_motion`；
- `Q_全局` 负责汇总协调，因此继续对齐整段视频的全局语义摘要；
- 这样Oracle三路都成为**有明确监督语义的教师**，而不再是"只有全局直接看答案，其余两路靠顺带学到"。

**② `L_route_distill(stopgrad(Q_o), Q_s)`：Student三路蒸馏**

在新口径里，`L_route_distill` 直接覆盖三条route，因此旧的 `L_global_distill` 可以删除。公式写为：

```python
L_route_distill =
    (1/3) * (
        [1 - cos(u_char_s, sg(u_char_o))] +
        [1 - cos(u_act_s,  sg(u_act_o ))] +
        [1 - cos(u_glb_s,  sg(u_glb_o ))]
    )
```

若写成batch均值形式：

```python
L_route_distill = (1/B) * Σ_b (1/3) * (
    1 - cos(u_char_s[b], sg(u_char_o[b])) +
    1 - cos(u_act_s[b],  sg(u_act_o[b])) +
    1 - cos(u_glb_s[b],  sg(u_glb_o[b]))
)
```

**计算流程**：

1. 让Oracle和Student对同一样本各自前向，得到 post-DIM 的三路route。
2. 对三路token分别做 `pool`，得到 `u_char_* / u_act_* / u_glb_*`。
3. Oracle侧执行 `stop-gradient`，只作为教师目标。
4. 分别计算角色路、动作路、全局路的蒸馏项，再取平均。

**计算原理**：

- 现在Oracle的三路都已经被直接ground，因此Student最自然的学习目标，就是逐路逼近Oracle；
- 把 `Q_全局` 并入 `L_route_distill` 后，蒸馏逻辑统一为"三路都做 route-wise teacher-student cosine distill"，不再需要额外单列一项全局蒸馏；
- Oracle侧必须 `stop-gradient`，否则教师目标会随着Student反向移动，蒸馏会失去稳定参照。

**③ `L_contrast(Q_s_pre_DIM, valid_pairs)`：pre-DIM解耦对比损失**

这里的"contrast"不是经典InfoNCE那种大规模正负样本分类，而是一个**基于配对不变性的轻量约束**。它只拉近我们想保持稳定的因素，不强行把所有别的样本推远。

这里最容易误解的一点是：**单条样本并不会产生多个 `Q_角色` 或多个 `Q_动作`**。对一条训练样本 `(I_ref, T, V_ref?, Y_gt, gs)` 来说，经过Student前向后只会得到**一组** `Q_角色^{S,0}` 和 **一组** `Q_动作^{S,0}`。  
`L_contrast` 里的"同角色不同动作"与"同动作不同角色"，说的都是**跨样本配对**，不是单样本内部有多个route。

先定义两类有效pair集合：

- `P_char`：同角色不同动作，如样本 `i = (A, act1)` 与样本 `j = (A, act2)`
- `P_motion`：同动作不同角色，如样本 `i = (A, act)` 与样本 `j = (B, act)`

也就是说：

- **同角色不同动作**：两条不同样本里，角色身份标签相同，但动作标签不同。我们希望它们的 `Q_角色^{S,0}` 彼此接近。
- **同动作不同角色**：两条不同样本里，动作标签相同，但角色身份标签不同。我们希望它们的 `Q_动作^{S,0}` 彼此接近。

一个具体例子：

```text
样本 i:  角色=皮卡丘, 动作=跑步
样本 j:  角色=皮卡丘, 动作=挥手      -> (i, j) ∈ P_char

样本 m:  角色=皮卡丘, 动作=跑步
样本 n:  角色=哆啦A梦, 动作=跑步    -> (m, n) ∈ P_motion
```

所以这里的"多个情况"，不是一条数据里有多个 `q-action / q-role`，而是**训练集里有多条样本，某些样本之间可以组成pair**。

**pair从哪里来**：

1. 若数据集有可靠的 `char_id`（角色ID）与 `action_id`（动作ID/动作簇）标注，则直接在batch内或sampler中按标签组pair。
2. 若没有严格标签，但有可用的弱标注，可以用：
   - 文件夹/剧集名 + 角色名 作为 `char_id`
   - 文本动作模板归一化、关键词规则或离线动作聚类 作为 `action_id`
3. 若角色或动作标签不可靠，**不要硬构pair**；该batch直接令 `λ_pair = 0`，跳过 `L_contrast`。

公式写为：

```python
L_char_inv =
    (1 / |P_char|) * Σ_(i,j in P_char) [1 - cos(u_char0[i], u_char0[j])]

L_motion_inv =
    (1 / |P_motion|) * Σ_(i,j in P_motion) [1 - cos(u_act0[i], u_act0[j])]

L_contrast = 0.5 * (L_char_inv + L_motion_inv)
```

更直观地看，就是：

```python
# 同角色不同动作：角色路应稳定
L_char_inv = 1 - cos(pool(Q_角色^{S,0}(A, act1)),
                     pool(Q_角色^{S,0}(A, act2)))

# 同动作不同角色：动作路应稳定
L_motion_inv = 1 - cos(pool(Q_动作^{S,0}(A, act)),
                       pool(Q_动作^{S,0}(B, act)))
```

**计算流程**：

1. 每条样本各自前向，只得到自己的一组 `Q_角色^{S,0}` 和 `Q_动作^{S,0}`。
2. 在batch内或配对采样器中，用 `char_id / action_id` 在**不同样本之间**构造 `P_char / P_motion`。
3. 对pair中的两条样本分别做 `pool`。
4. 对角色不变性pair算 `L_char_inv`，对动作不变性pair算 `L_motion_inv`。
5. 两项取平均得到 `L_contrast`。若某一batch缺少某类有效pair，或该批次标签不可靠，则令 `λ_pair = 0`，该项不参与总损失。

**计算原理**：

- 同角色不同动作时，`Q_角色` 不该因为动作变化而大幅漂移；
- 同动作不同角色时，`Q_动作` 不该因为角色不同而失去动作主轴；
- 这等于在Student内部人为制造一个"因子分解"信号：角色路对动作扰动不敏感，动作路对角色扰动不敏感。

**为什么必须作用在pre-DIM，而不是post-DIM**：

- DIM之后的 `Q_动作'` 本来就应该吸收角色形体约束，例如"猫的跳跃"和"蛇的跳跃"不应完全相同；
- 如果在post-DIM上继续强拉动作表示一致，会直接和"角色适配动作"目标冲突；
- 所以 `L_contrast` 只约束**route seed的初始分工**，不约束最终联合解释结果。

**为什么从5项损失简化为3项**：

- `L_oracle_ground`：先把Oracle训成可信教师
- `L_route_distill`：让Student学会Oracle的三路语义
- `L_contrast`：让Student在pre-DIM阶段形成可分解的route seed

具体地：

1. 删掉 `L_global_distill`：因为它已经被纳入三路 `L_route_distill`，单列只是重复记账。
2. 删掉 `L_orthogonal`：因为三路Oracle direct ground已经给了角色/动作不同的监督靶标，再叠加route-wise distill后，主线分工已经足够明确；继续施加几何正交先验常常只会额外约束合法共享信息。
3. 保留 `L_contrast`：因为它提供的是**跨样本不变性信号**，这不是单样本的ground/distill能完全替代的，所以仍值得保留；但它必须依赖可靠pair，因此只作为**可选低权重辅助项**，而不是每个batch都强制启用。

因此，Stage 1的核心不是直接生成视频，而是把MLLM内部的route表示先训成**可解释、可分工、可蒸馏**的控制空间。

### 4.9 DIM（双向信息调制）

MLLM所有层完成后，提取各路隐藏状态，执行双向信息调制。

```python
class DIM(nn.Module):
    """
    双向动态信息调制。共享参数的跨路由交叉注意力。
    零初始化输出投影 → 训练初始为恒等映射，无需门控。
    """
    def __init__(self, d=1536, n_heads=12):
        super().__init__()
        self.norm_q = LayerNorm(d)
        self.norm_kv = LayerNorm(d)
        self.cross_attn = MultiheadAttention(d, n_heads)
        self.norm_ff = LayerNorm(d)
        self.ffn = SwiGLU(d, d * 4)

        # 零初始化所有输出投影
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _interact(self, q, kv):
        h = self.cross_attn(
            query=self.norm_q(q), key=self.norm_kv(kv), value=self.norm_kv(kv)
        )
        q = q + h  # 残差（h初始全零）
        q = q + self.ffn(self.norm_ff(q))
        return q

    def forward(self, q_char, q_motion, q_global):
        # ① Q_动作读Q_角色 → "角色体型如何影响动作实现"
        q_motion = self._interact(q_motion, q_char)
        # ② Q_角色读已更新的Q_动作 → "动作会让角色外观如何变化"
        q_char = self._interact(q_char, q_motion)
        # ③ Q_全局聚合两路 → 综合协调
        q_both = torch.cat([q_char, q_motion], dim=1)
        q_global = self._interact(q_global, q_both)
        return q_char, q_motion, q_global
```

| 调用 | Q输入 | KV输入 | 学到什么 |
|------|-------|--------|---------|
| ①Q_动作←Q_角色 | 48tok动作 | 64tok角色 | 角色体型/结构对动作的约束 |
| ②Q_角色←Q_动作' | 64tok角色 | 48tok更新后的动作 | 动作对角色外观变化的要求 |
| ③Q_全局←[Q_角色'‖Q_动作'] | 32tok全局 | 112tok两路拼接 | 整体场景如何协调 |

参数量：~1.5M × 2分支(Student+Oracle) = ~3M。

---

## 5. 接口层：Route Adapter

系统级口径上，Connector + DSN视作一个route adapter能力块。

### 5.1 Connector

DIM输出的3路token通过共享的4层双向Transformer + 分路投影到4096维：

```
所有路由token拼接(144tok) → 4层Transformer（双向，共享）
→ 分路投影：
  Q_角色 → Linear_粗(1536, 4096) → C_角色粗 [64, 4096]
  Q_角色 → Linear_细(1536, 4096) → C_角色细 [64, 4096]
  Q_动作 → Linear_动(1536, 4096) → C_动作   [48, 4096]
  Q_全局 → Linear_全(1536, 4096) → C_全局   [32, 4096]
```

双头投影：C_角色粗侧重结构信息（浅层Wan），C_角色细侧重细节（深层Wan）。二者都来自DIM后的 `Q_角色'`，已包含角色-动作联合解释。

参数量：~30M。

### 5.2 DSN（分布统计归一化）

DSN的目标不是把route token变成text内容，而是把route条件的范数、均值/方差与注意力可读性**软对齐**到Wan conditioner manifold，降低Wan误读概率，同时保留route自己的新增信息轴。

```python
class DistribStatNorm(nn.Module):
    def __init__(self, d=4096, blend_init=0.2):
        super().__init__()
        self.register_buffer('t5_mu', torch.zeros(d))
        self.register_buffer('t5_std', torch.ones(d))
        self.blend_logit = nn.Parameter(torch.tensor(blend_init))
        self.affine_scale = nn.Parameter(torch.ones(d))
        self.affine_bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        x_mu = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        blend = torch.sigmoid(self.blend_logit)
        t_mu = blend * self.t5_mu + (1 - blend) * x_mu
        t_std = blend * self.t5_std + (1 - blend) * x_std
        x_norm = (x - x_mu) / x_std * t_std + t_mu
        return x_norm * self.affine_scale + self.affine_bias
```

各路C_*分别过DSN（共享参数）。参数量：~0.1M。

---

## 6. Wan侧设计

**本章统一直观例子**：

- `I_ref`：一只蓝色猫形动画角色，圆脸、短腿、大耳朵、红围巾
- `T`："先助跑，再起跳，空中挥手，最后落地站稳"
- `V_ref`：可选，若存在，则只提供动作节奏与运动趋势
- `gs`：
  - 高 `gs`（如 `0.9`）表示更接近I2V，参考图构图和身份约束很强
  - 低 `gs`（如 `0.2`）表示更接近Animate，动作和镜头自由度更高

下面所有Wan侧模块，都默认围绕这个例子解释："同一个蓝色猫角色，在不同 `gs` 下如何既保持身份、又把跳跃动作做自然"。

### 6.1 连续gs首帧条件

统一模式谱，不维护分裂的多套模式：

```python
def prepare_first_frame(I_ref, vae, gs):
    latent = vae.encode(I_ref)         # [B, 16, H, W]
    noise = torch.randn_like(latent)
    blended = gs * latent + (1 - gs) * noise
    mask = torch.full((latent.shape[0], 1, *latent.shape[2:]), gs, device=latent.device)
    return torch.cat([blended, mask], dim=1)  # [B, 17, H, W]
```

训练时 `gs` 从 Beta(2,2) 分布采样。

**直观例子**：

- 当 `gs = 1.0` 时，Wan拿到的首帧latent几乎就是参考图本身的VAE编码。对上面的蓝色猫例子，这意味着第一帧很大概率就是那只蓝色猫，围巾、脸型、耳朵位置都会被强约束。
- 当 `gs = 0.5` 时，首帧latent是"一半参考图、一半噪声"。这时角色身份仍有明显牵引，但镜头和姿态已经可以更自由，比如不再死守参考图中的站姿。
- 当 `gs = 0.0` 时，首帧完全退化为纯噪声，系统进入真正的Animate模式。此时身份保持主要依赖后面的 `CSG + CRLA + L_struct`，而不是首帧像素锁定。

**实际流程**：

1. 先把 `I_ref` 编到VAE latent空间，得到参考图的紧凑像素表示。
2. 再采样一份同形状高斯噪声。
3. 用 `gs` 做线性混合：`gs` 越高，越像参考图；越低，越像纯噪声。
4. 额外拼一个全图同值的 `mask` 通道，把当前参考强度显式告诉Wan。
5. Wan后续所有block都能看到这个17通道首帧输入，因此它从一开始就知道："这次该更像I2V，还是更像Animate。"

**为什么要带 `mask` 通道**：

- 仅靠混合后的latent，模型不一定能稳定区分"这是参考图很强导致的清晰"还是"只是碰巧噪声较弱"；
- `mask = gs` 等于给Wan一个明确标记：当前首帧约束强度到底是多少；
- 这能减少训练时对 `gs` 的歧义理解。

**人类可读的理解**：

这一节做的事情很像给画师一个起稿自由度滑杆：

- 高 `gs`：照着参考图临摹起稿
- 中 `gs`：只保留角色大意，允许重构动作和构图
- 低 `gs`：只记住"是谁"，但不要照着参考姿态画

### 6.2 统一e0条件增强

**解决什么**：Wan原始 `e0 = TimestepEmbed(t) + TextPoolEmbed`，仅含timestep和T5全局pool。AdaLN是DiT中最强的条件注入路径（通道级全局强制调制），但当前不含任何route条件信息。且所有帧在同一去噪步使用完全相同的e0。

**为什么cross-attention不能替代**：cross-attention是token级选择性读取，e0调制是通道级全局强制注入。两者作用层级不同，不可互替。

```python
class UnifiedE0Conditioning(nn.Module):
    """
    统一e0条件增强：全局语义调制 + 帧级时序差异化。
    一次性输出帧级e0增量。所有输出层零初始化。
    """
    def __init__(self, d_route=4096, d_wan=2048, max_frames=21, bottleneck=256):
        super().__init__()
        # 全局语义增量：C_全局+C_动作 → e0偏移
        self.global_proj = nn.Sequential(
            nn.Linear(d_route * 2, bottleneck), nn.SiLU(),
            nn.Linear(bottleneck, d_wan)
        )
        # 帧级时序增量：C_动作 + 帧位置 → 每帧额外偏移
        self.motion_to_seed = nn.Linear(d_route, bottleneck)
        self.frame_pos = nn.Parameter(torch.randn(max_frames, bottleneck) * 0.01)
        self.seed_to_delta = nn.Linear(bottleneck, d_wan)

        # 零初始化所有输出
        nn.init.zeros_(self.global_proj[-1].weight)
        nn.init.zeros_(self.global_proj[-1].bias)
        nn.init.zeros_(self.seed_to_delta.weight)
        nn.init.zeros_(self.seed_to_delta.bias)

    def forward(self, e0_base, c_global_pooled, c_motion_pooled, n_frames):
        """
        e0_base: [B, D_wan]   原始e0
        返回: [B, T, D_wan]   帧级增强e0
        """
        # 全局：C_全局+C_动作联合 → 所有帧共享的e0偏移
        global_delta = self.global_proj(
            torch.cat([c_global_pooled, c_motion_pooled], dim=-1)
        )  # [B, D_wan]

        # 帧级：C_动作+帧位置 → 每帧独立的e0偏移
        seed = self.motion_to_seed(c_motion_pooled)          # [B, bottleneck]
        frame_seeds = seed.unsqueeze(1) + self.frame_pos[:n_frames]
        temporal_delta = self.seed_to_delta(frame_seeds)      # [B, T, D_wan]

        # 合成帧级e0
        return e0_base.unsqueeze(1) + global_delta.unsqueeze(1) + temporal_delta
```

语义含义：
- `global_delta` 告诉每一层"现在在生成一只猫做跳跃动作"（全局语义）
- `temporal_delta` 告诉每一层"当前帧是起跳/腾空/落地阶段"（帧级动态）
- 两者共同作用于AdaLN的scale/shift/gate

**为什么不注入C_角色？** C_角色已通过CRLA和交叉注意力注入。e0调制是全局的（无空间选择性），角色信息需要空间选择性，适合token级通路。

参数量：~4M。训练阶段：Stage 2 warmup即引入（零初始化，初期无影响）。

**直观例子**：

继续用蓝色猫角色举例。假设文本是"先助跑，再起跳，空中挥手，最后落地站稳"。

- 仅靠原始Wan的 `e0`，所有帧在同一去噪步看到的都是同一个"时间步 + 文本池化"向量。对模型来说，"起跳帧"和"落地帧"在全局调制层面没有本质区别。
- 加上统一e0条件增强后：
  - `global_delta` 会告诉所有层："现在在生成的是蓝色猫角色的跳跃视频，而不是别的角色或别的动作"
  - `temporal_delta` 会告诉不同帧："这一帧更像助跑阶段、腾空阶段还是落地阶段"

**实际流程**：

1. 先保留Wan原始的 `e0_base = timestep_embed + text_pool_embed`，不破坏预训练主干。
2. 从 `C_全局` 和 `C_动作` 池化出两个摘要向量：
   - `C_全局` 代表整段视频的大局协调
   - `C_动作` 代表动作语义和节奏主轴
3. 把二者拼起来，生成一个所有帧共享的 `global_delta`。
4. 再只用 `C_动作` 生成一个低维动作种子 `seed`。
5. 把 `seed` 与每一帧的 `frame_pos` 相加，得到每帧不同的 `frame_seeds`。
6. 通过 `seed_to_delta` 把这些 `frame_seeds` 映射到Wan的 `d_wan` 维度，得到每帧独立的 `temporal_delta`。
7. 最终得到 `e0_per_frame = e0_base + global_delta + temporal_delta`，供每一帧的AdaLN使用。

**为什么这比只靠cross-attention更重要**：

- cross-attention像"当前token自己去查外部条件"
- e0/AdaLN像"整个block一开始就知道自己现在在处理什么类型的视频"

对蓝色猫跳跃这个例子来说，cross-attention可以告诉某个局部token："这里可能要看红围巾信息"；但e0调制会让整层网络都进入"生成猫形跳跃动作"的处理模式。这两种作用层级不同。

**为什么不把 `C_角色` 也塞进e0**：

- `C_角色` 更像"空间上哪里长什么样"，例如耳朵尖不尖、围巾在脖子哪里；
- e0是全局通道调制，没有空间选择性；
- 如果把细角色信息直接塞进e0，容易让全图都带上不该有的角色偏执。角色细节更适合走 `CRLA + route cross-attention` 这种token级路径。

**人类可读的理解**：

这一节做的事像给整个视频生成器一个"当前任务状态面板"：

- 全局面板：现在画的是谁、在做什么
- 帧级面板：这一帧处于动作的哪个阶段

这样Wan不是被动等token来提醒它，而是从block一开始就带着状态工作。

### 6.3 解耦双通道交叉注意力

**解决什么**：T5 tokens在softmax竞争中系统性压制route tokens。拼接注入的根本问题是softmax归一化导致的零和竞争。

```python
class DecoupledCrossAttention(nn.Module):
    """
    Text通道复用Wan预训练cross-attention权重。
    Route通道使用独立K/V投影（瓶颈256），零初始化输出。
    32层共享同一套route K/V投影。
    """
    def __init__(self, d_wan=2048, d_context=4096):
        super().__init__()
        self.route_k_proj = nn.Sequential(
            nn.Linear(d_context, 256, bias=False),
            nn.Linear(256, d_wan, bias=False)
        )
        self.route_v_proj = nn.Sequential(
            nn.Linear(d_context, 256, bias=False),
            nn.Linear(256, d_wan, bias=False)
        )
        self.route_out_proj = nn.Linear(d_wan, d_wan, bias=False)
        nn.init.zeros_(self.route_out_proj.weight)

    def forward(self, x, q_proj, text_context, route_context,
                original_cross_attn_fn):
        # Text通道：走原始Wan cross-attention（含SA-LoRA）
        text_out = original_cross_attn_fn(x, text_context)

        # Route通道：独立K/V投影
        q = q_proj(x)
        k_route = self.route_k_proj(route_context)
        v_route = self.route_v_proj(route_context)
        route_attn = F.scaled_dot_product_attention(q, k_route, v_route)
        route_out = self.route_out_proj(route_attn)

        return text_out + route_out
```

与SA-LoRA的关系：SA-LoRA继续作用在text通道K/V上，角色变为"仅适配text通道"。route通道K/V从头训练。

参数量：~7M（32层共享）。训练阶段：Stage 2 main。

**直观例子**：

文本 `T` 说的是："先助跑，再起跳，空中挥手，最后落地站稳"。  
route条件说的是："这是蓝色猫角色，短腿、大耳朵、红围巾；跳跃幅度要适配这种身体结构。"

如果把 text token 和 route token 直接拼在一个context里，Wan会更偏向自己熟悉的T5 token，因为：

- text token 数量更多
- text通道是预训练就会用的
- route token 是新加的，起初更弱

结果就是：模型知道"要跳跃"，却不一定知道"这只短腿蓝色猫应该怎么跳"。

**实际流程**：

1. 对同一份视频latent查询 `x`，先走一条**text通道**：
   - 复用Wan原始cross-attention
   - 继续使用SA-LoRA去微调它如何读取T5文本
2. 再走一条**route通道**：
   - 用同一份 `q = q_proj(x)` 作为查询
   - 但 `k/v` 不再来自T5，而来自独立投影后的 `route_context`
3. 两条通道分别产出 `text_out` 和 `route_out`
4. 最后把两者相加送回主干

**蓝色猫例子里，两条通道分别在做什么**：

- text通道负责理解"先助跑再起跳再挥手"这种开放域动作语义
- route通道负责补充"动作要落在蓝色猫这种体型和外观上，该怎么做才像它自己"

**为什么说它消除了零和竞争**：

原来拼接context时，softmax会让text和route在同一个注意力分布里抢概率。  
现在两者分开以后：

- text通道内部只和text竞争
- route通道内部只和route竞争

这意味着route不需要先"打赢"文本，才能被模型看见。

**一个更直观的对比**：

- 拼接方案：像一个会议里 500 个老员工和 100 个新员工一起抢发言，老员工天然更占优势
- 解耦方案：开两个并行会场，一个专门讨论文本语义，一个专门讨论角色动作控制，最后再合并结论

**为什么 route 通道不用SA-LoRA**：

- SA-LoRA是在"如何读文本"这条旧通路上做低秩适配
- route通道不是旧通路微调，而是新信号源的专用入口
- 所以route通道最稳的做法是直接从头训练自己的K/V投影，而不是硬复用text适配器

### 6.4 分层route context

| 层组 | route context | 作用 | token数 |
|------|--------------|------|---------|
| 浅层 0-9 | `C_全局 + C_角色粗 + CSG_app` | 角色整体构图和主要外观 | 32+64+8 = 104 |
| 中层 10-21 | `C_全局 + C_动作 + C_角色粗 + CSG_all` | 动作轨迹与角色适配 | 32+48+64+16 = 160 |
| 深层 22-31 | `C_全局 + C_角色细 + CSG_all` | 纹理、表情、局部细节 | 32+64+16 = 112 |

`text_context` 单独保留T5 text tokens，最大长度512。

**直观例子**：

把Wan 32层想成三段画画流程：

- 浅层像在起草图：先决定"镜头里是谁、整体长什么样"
- 中层像在摆动作：决定"这一帧怎么动、身体怎么配合动作"
- 深层像在修细节：补表情、纹理、局部边缘和小结构

对蓝色猫例子来说：

- 浅层最重要的是"这是一只蓝色猫，不是别的角色"
- 中层最重要的是"它要完成助跑、起跳、挥手、落地这些动作阶段"
- 深层最重要的是"耳朵边缘、围巾纹理、眼睛高光要像那只猫"

**实际流程**：

1. 先把Connector输出拆成 `C_角色粗 / C_角色细 / C_动作 / C_全局`。
2. 再把CSG拆成 `CSG_app` 和 `CSG_all`。
3. 按层组分配route context：
   - 0-9层：放全局 + 角色粗 + 外观锚点
   - 10-21层：放全局 + 动作 + 角色粗 + 全量CSG
   - 22-31层：放全局 + 角色细 + 全量CSG
4. 每一层仍然保留独立的 `text_context`，只是不与route混在同一softmax里。

**为什么中层要加入 `C_动作`，深层反而拿掉**：

- 中层是"把动作搭起来"的关键阶段，最需要动作条件介入
- 深层更像收尾抛光阶段，如果动作条件在这里过强，容易把已经画好的身份细节再冲乱
- 所以深层主要看 `C_角色细 + CSG_all`，把角色判别特征收稳

**人类可读的理解**：

这一节不是简单地"把所有条件喂给所有层"，而是把条件按用途分发：

- 早期先看谁
- 中期先看怎么动
- 后期先看长得像不像

### 6.5 CSG（角色语义锚点）

CSG弥补低gs时首图约束变弱的问题，提供语义级角色锚定。

两条流：
- 外观流：CLIP [CLS] → 投影 → `CSG_app`（8 token），仅浅层使用
- 结构流：CLIP patch汇聚 → 投影 → `CSG_str`（8 token）
- `CSG_all = [CSG_app ‖ CSG_str]`（16 token），中深层使用

Token数固定，不做gs自适应（TCC已提供gs相关的权重调度）。

CSG的负边界：
- 只锚定角色判别性外观，不负责把每帧拉回参考图姿态
- 在中层动作建立阶段，`C_动作` 的可执行性优先于CSG的静态外观执念

参数量：~4M。

**直观例子**：

当 `gs` 很低时，首帧对角色的像素锁定很弱。蓝色猫例子里，模型可能还知道"要跳跃"，但容易把角色逐渐画成人形、把耳朵变短、把围巾忘掉。  
CSG就是在这种时候提供一个**语义级角色提醒**：别忘了它是谁。

两条流可以这样理解：

- `CSG_app`：告诉模型"这个角色整体看起来是什么气质和配色"，比如蓝色、红围巾、圆脸
- `CSG_str`：告诉模型"这个角色的身体结构和轮廓是什么样"，比如短腿、大耳朵、头身比特殊

**实际流程**：

1. 先对参考图 `I_ref` 做与CLIP一致的预处理（resize / crop / normalize），送入冻结CLIP图像编码器。
2. 从CLIP取出两类视觉摘要：
   - 全局特征 `g_cls = CLIP_cls(I_ref)`，形状 `[B, 768]`
   - patch特征 `G_patch = CLIP_patch(I_ref)`，形状 `[B, N_p, 768]`
3. 用 `g_cls` 构造外观流 `CSG_app`
4. 用 `G_patch` 的空间聚合结果构造结构流 `CSG_str`
5. 浅层只用 `CSG_app`，因为浅层更关心大外观和整体辨识度
6. 中深层使用 `CSG_all = [CSG_app ‖ CSG_str]`，因为后面开始需要结构与细节一起约束

更细地说，两个流的处理过程如下。

**A. 从 CLIP 全局特征构造 `CSG_app`**

目标：把一张参考图的**整体视觉气质**变成8个可供Wan读取的外观token。

处理流程：

1. 取CLIP全局特征  
   `g_cls = CLIP_cls(I_ref) ∈ R^768`
2. 对 `g_cls` 做轻量归一化/线性投影，得到一个外观语义种子  
   `s_app = Proj_app_seed(LN(g_cls)) ∈ R^d_mid`
3. 把同一个种子复制到8个slot，并给每个slot加上**可学习的外观token原型**  
   `h_app[j] = s_app + E_app[j],  j=1...8`
4. 对每个slot独立通过共享的外观投影头，映射到Wan cross-attention维度  
   `CSG_app[j] = Proj_app_out(h_app[j]) ∈ R^4096`
5. 得到  
   `CSG_app ∈ R^(8×4096)`

可写成：

```python
g_cls = CLIP_cls(I_ref)                          # [B, 768]
s_app = Proj_app_seed(LN(g_cls))                 # [B, d_mid]
h_app = s_app.unsqueeze(1) + E_app.unsqueeze(0) # [B, 8, d_mid]
CSG_app = Proj_app_out(h_app)                    # [B, 8, 4096]
```

这里最关键的一点是：**8个外观token不是简单复制8份同一个向量**。  
虽然它们共享同一个全局外观种子 `s_app`，但每个slot都有不同的可学习原型 `E_app[j]`，所以训练后会自然分工成不同的外观子方向，例如：

- 某些slot更偏颜色与材质（蓝色、围巾红色）
- 某些slot更偏脸部辨识度（圆脸、眼睛区域）
- 某些slot更偏整体风格（卡通感、线条粗细、艺术风格）

**为什么外观流用 `CLIP_cls` 而不是 patch**：

- `CLIP_cls` 最擅长概括整张图的整体辨识特征
- 它对颜色、风格、角色大轮廓的摘要更稳定
- 这正适合浅层Wan先建立"这是谁、整体看起来像什么"

**B. 从 CLIP patch聚合特征构造 `CSG_str`**

目标：把参考图的**身体结构和空间轮廓**变成8个结构token。

处理流程：

1. 取CLIP patch特征  
   `G_patch = CLIP_patch(I_ref) ∈ R^(N_p×768)`
2. 沿空间维做聚合，默认使用简单稳定的平均池化  
   `g_patch = mean_i G_patch[i] ∈ R^768`
3. 对 `g_patch` 做轻量归一化/线性投影，得到结构语义种子  
   `s_str = Proj_str_seed(LN(g_patch)) ∈ R^d_mid`
4. 和外观流一样，把该种子复制到8个slot，并加上**结构token原型**  
   `h_str[j] = s_str + E_str[j],  j=1...8`
5. 通过结构投影头映射到Wan context维度  
   `CSG_str[j] = Proj_str_out(h_str[j]) ∈ R^4096`
6. 得到  
   `CSG_str ∈ R^(8×4096)`

可写成：

```python
G_patch = CLIP_patch(I_ref)                      # [B, N_p, 768]
g_patch = G_patch.mean(dim=1)                    # [B, 768]
s_str = Proj_str_seed(LN(g_patch))               # [B, d_mid]
h_str = s_str.unsqueeze(1) + E_str.unsqueeze(0) # [B, 8, d_mid]
CSG_str = Proj_str_out(h_str)                    # [B, 8, 4096]
```

`CSG_str` 里的8个token训练后通常会偏向不同的结构子方向，例如：

- 头身比和整体轮廓
- 耳朵/尾巴等非人角色突出结构
- 四肢长短和关节感
- 围巾、帽子等会影响轮廓读感的附属物

**为什么结构流不用整张图 `CLIP_cls`，而要走 patch聚合**：

- 结构信息本质上来自空间分布，而不是纯全局语义标签
- patch特征虽然最终被平均池化，但它的来源仍然是局部空间编码，天然更带结构感
- 这比直接复用 `CLIP_cls` 更适合表达"头大身小、短腿、大耳朵"这类形体信息

**为什么只做 patch平均，而不直接保留所有patch token**：

- CSG的职责是"轻量语义锚点"，不是再造一套高成本视觉记忆
- 保留所有patch会让token数和计算量迅速上升，并与CRLA职责重叠
- CRLA已经负责更细的像素级记忆，所以CSG结构流只保留**压缩后的结构摘要**即可

**最终拼接方式**：

```python
CSG_all = torch.cat([CSG_app, CSG_str], dim=1)   # [B, 16, 4096]
```

也就是说：

- `CSG_app`：8个外观token，给浅层建立角色整体辨识度
- `CSG_all`：16个token，给中深层同时提供外观 + 结构提醒

**一个完整例子**：

对蓝色猫角色来说：

- `CSG_app` 更像在反复提醒Wan："这是蓝色、红围巾、圆脸、卡通感很强的角色"
- `CSG_str` 更像在提醒："它不是标准人形，腿短、耳朵大、头身比特殊，动作不要按普通人去画"

所以当模型在中层建立跳跃动作时，`C_动作` 负责"跳起来"，`CSG_str` 负责"按蓝色猫的身体结构去跳"，二者配合起来，动作才既成立又不人形化。

**为什么CSG不是把每帧拽回参考图姿态**：

蓝色猫参考图也许是站着的，但文本要求它助跑、起跳、挥手、落地。  
CSG只负责提醒"还是那只蓝色猫"，不负责逼所有帧都长得像参考图同一个姿势。  
否则模型会为了保身份，把跳跃动作做僵。

**和CRLA的区别**：

- CSG是**语义锚点**，像一句反复提醒的话："别忘了是这只角色"
- CRLA是**像素记忆**，像把参考图里更细的局部视觉信息存起来随时查

**人类可读的理解**：

CSG更像角色设定卡，不是角色照片本身。它提醒模型角色的辨识特征，但不强迫每一帧都复刻参考图。

### 6.6 TCC（时步条件调度）

TCC只调route条件，不调text_context。

默认规律：
- C_角色粗：高噪声阶段更强
- C_动作：中间时步最强
- C_角色细：低噪声阶段更强
- CSG：Animate模式比I2V模式更强
- C_全局 / text：恒定

实现：手工基函数 + 零初始化小残差MLP做轻微修正。

```python
class TCC(nn.Module):
    def __init__(self, n_routes=4):
        super().__init__()
        self.residual_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, n_routes)
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def get_scales(self, t_n, gs):
        """t_n ∈ [0,1]: 0=干净, 1=纯噪声"""
        base_i2v = {
            'C_角色粗': 0.70 + 0.30 * t_n,
            'C_角色细': 0.35 + 0.65 * (1 - t_n),
            'C_动作':   0.55 + 0.45 * math.sin(t_n * math.pi),
            'CSG':      0.50 + 0.30 * t_n,
        }
        base_animate = {
            'C_角色粗': 0.85 + 0.15 * t_n,
            'C_角色细': 0.50 + 0.50 * (1 - t_n),
            'C_动作':   0.55 + 0.45 * math.sin(t_n * math.pi),
            'CSG':      0.75 + 0.25 * t_n,
        }
        base = {k: gs * base_i2v[k] + (1 - gs) * base_animate[k]
                for k in base_i2v}

        residuals = self.residual_mlp(
            torch.tensor([t_n, gs], device=self.residual_mlp[0].weight.device)
        ).squeeze()
        for i, k in enumerate(['C_角色粗', 'C_角色细', 'C_动作', 'CSG']):
            base[k] = max(0.1, min(1.2, base[k] + 0.1 * residuals[i].item()))

        base['C_全局'] = 1.00
        base['T5'] = 1.00
        return base
```

参数量：~0.003M。

**直观例子**：

同样是蓝色猫跳跃视频，Wan在不同去噪阶段需要的信息重点不一样：

- 高噪声阶段：画面还很乱，先把"大轮廓、角色整体"定住最重要
- 中间阶段：开始把动作搭起来，助跑和起跳轨迹最关键
- 低噪声阶段：大结构已定，更需要补脸部、围巾、边缘纹理

TCC做的就是把这种"不同去噪阶段该更信什么条件"显式写出来。

**实际流程**：

1. 读入当前归一化时步 `t_n` 和模式强度 `gs`
2. 先按人工先验给出基础权重曲线：
   - `C_角色粗`：高噪声更强
   - `C_动作`：中段最强
   - `C_角色细`：低噪声更强
   - `CSG`：在Animate模式里更强
3. 再用一个极小残差MLP做轻微修正，但修正初值为0，不会一开始乱改
4. 对每一层当前使用的route context乘上这些scale，再送入解耦CA

**几个具体数值感受**（以默认基函数为例）：

- `t_n = 0.9`、I2V模式时，`C_角色粗 ≈ 0.97`，说明高噪声时先把角色大轮廓抓稳
- `t_n = 0.5` 时，`C_动作 = 0.55 + 0.45*sin(π/2) = 1.0`，说明中段动作最强
- `t_n = 0.1` 时，`C_角色细 ≈ 0.94`，说明后段更该补细节

**为什么 `text_context` 不跟着TCC一起调**：

- 文本负责开放域语义底座，整条去噪过程都需要它稳定存在
- route条件更像可执行控制，适合按阶段动态强调
- 因此TCC只调route，不调text

**人类可读的理解**：

TCC像一个分镜导演在不同阶段给不同部门不同优先级：

- 起稿时先看角色大形
- 中段先看动作安排
- 收尾时先看角色细节

### 6.7 CRLA（角色像素记忆）

CRLA解决中后段身份漂移和非人角色局部结构走样。

```python
class CharRegionMemory(nn.Module):
    def __init__(self, c_vae=16, d_wan=2048, n_tokens=48):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj_in = nn.Linear(c_vae, d_wan)
        self.importance_head = nn.Linear(d_wan, 1)
        self.proj_out = nn.Linear(d_wan, d_wan)
        self.crla_scale = nn.Parameter(torch.zeros(1))  # 唯一保留的门控

    def extract_ref_tokens(self, ref_latent):
        flat = ref_latent.flatten(2).permute(0, 2, 1)  # [B, HW, C_vae]
        proj = self.proj_in(flat)                        # [B, HW, D_wan]
        weights = torch.softmax(self.importance_head(proj).squeeze(-1), dim=-1)
        proj = proj * weights.unsqueeze(-1)
        pooled = F.adaptive_avg_pool1d(
            proj.permute(0, 2, 1), self.n_tokens
        ).permute(0, 2, 1)                               # [B, 48, D_wan]
        return pooled

    def forward(self, x, char_ref_tokens):
        char_attn = F.scaled_dot_product_attention(
            q=x.unsqueeze(1), k=char_ref_tokens.unsqueeze(1),
            v=char_ref_tokens.unsqueeze(1)
        ).squeeze(1)
        return x + self.crla_scale * self.proj_out(char_attn)
```

注入策略：blocks 5-25激活（21个block共享同一套参数）。固定48 token，不做gs自适应。

CRLA的负边界：
- 优先修复中后段身份漂移与局部结构走样
- 不应把大幅运动或夸张表情的帧强行拽回静态参考图
- 若CRLA与中层动作分支冲突，优先下调CRLA的中层增益

参数量：~4M。

**直观例子**：

蓝色猫跳到第15帧以后，可能开始出现这些问题：

- 耳朵形状慢慢变钝
- 红围巾位置和花纹飘掉
- 脸越来越像泛化的人形卡通脸

CRLA就是专门处理这类"中后段身份漂移和局部结构走样"的模块。

**实际流程**：

1. 先把参考图 `I_ref` 编进VAE latent空间。
2. 把二维latent摊平成很多空间位置token。
3. 用 `importance_head` 给每个位置打重要性分数，让更像角色关键部位的位置权重更高。
4. 再做自适应池化，压成固定48个 `char_ref_tokens`。
5. 在Wan的blocks 5-25中，让当前视频token对这些 `char_ref_tokens` 做一次角色记忆读取。
6. 通过 `proj_out` 和 `crla_scale` 把读到的参考信息残差写回当前特征。

**为什么它有效**：

- CSG只告诉模型"这还是那只蓝色猫"
- CRLA进一步告诉模型"那只蓝色猫的耳朵、围巾、脸部局部细节到底长什么样"

**为什么只在 blocks 5-25 激活**：

- 太早注入：高噪声时局部像素记忆容易噪音大于收益
- 太晚注入：很多身份漂移已经形成，修复成本更高
- 中段到后中段最适合一边生成、一边纠偏

**为什么要保留 `crla_scale` 这个唯一门控**：

- CRLA直接写入的是像素细节类记忆，注入过强时容易把动作帧硬拉回静态参考图
- 例如蓝色猫正在空中挥手时，CRLA若过强，可能试图把手臂和围巾重新拽回参考图的站姿
- 所以需要一个运行时强度阀门

**人类可读的理解**：

CRLA像给模型一份"角色细节备忘录"。动作在继续，但模型可以不时翻一下备忘录，确认耳朵、脸型、围巾这些局部特征别画丢。

### 6.8 LoRA策略

| LoRA | 目标 | rank | 作用 |
|------|------|------|------|
| TA-LoRA | WanSelfAttention q/k/v/o | 32 | 适配动画运动模式 |
| SA-LoRA | Text通道 WanCrossAttention k/v | 8 | 适配text通道读取 |

SA-LoRA降至rank=8的原因：解耦CA后SA-LoRA仅适配text通道，职责变窄。route通道有自己从头训练的K/V投影。

**直观例子**：

还是蓝色猫跳跃这个例子。Wan原始预训练更多见的是通用视频分布，不一定熟悉动画角色的夸张运动和卡通外观。

- `TA-LoRA` 负责让**自注意力**更会处理动画式时序变化，比如夸张起跳、腾空停顿、落地回弹
- `SA-LoRA` 负责让**text通道交叉注意力**更会理解动画场景里的文本描述，比如"夸张地挥手""轻快地落地"

**实际流程**：

1. 在自注意力的 `q/k/v/o` 上挂 `TA-LoRA`，让Wan的时空token交互更适配动画视频
2. 在text通道的cross-attention `k/v` 上挂 `SA-LoRA`，只微调它如何读取T5文本
3. route通道不使用SA-LoRA，因为route本来就是新建的独立K/V投影，不走老的文本读取路径

**为什么 `TA-LoRA` 和 `SA-LoRA` 要分开**：

- 一个解决"视频token之间怎么动"
- 一个解决"怎么读文字"

这两个问题虽然都和条件注入有关，但落点完全不同，混成一个LoRA会让职责变脏。

**人类可读的理解**：

可以把它看成给Wan补两门短训课：

- `TA-LoRA`：动画运动课
- `SA-LoRA`：动画文本理解课

### 6.9 结构一致性损失

```
L_struct = (1/K) Σ_k [1 - cosine(sg(φ_char(I_ref)), φ_char(frame_k))]
```

其中 `φ_char(·)` 表示参考引导的**角色区域特征**（不是整帧）：
- 用参考图和生成帧的CLIP patch attention生成粗角色区域权重
- 只在角色相关区域上池化特征
- 不强约束背景、镜头与整帧构图

K=4非首帧均匀采样，`λ_struct = 0.05 → 0.1`，保持为弱约束。

**直观例子**：

如果蓝色猫跳着跳着，脸还是蓝色猫的脸，但身体比例、耳朵位置、围巾区域越来越不对，就会出现"看着像同一角色，但结构有点走样"的问题。  
`L_struct` 就是用来轻轻拉住这种角色区域结构的。

**实际流程**：

1. 从参考图 `I_ref` 中提取角色区域特征 `φ_char(I_ref)`。
2. 从生成视频里均匀采样 `K=4` 个非首帧。
3. 对每一帧也提取同样定义的角色区域特征 `φ_char(frame_k)`。
4. 逐帧计算与参考图角色区域特征的余弦距离。
5. 对4帧求平均，得到 `L_struct`。

**为什么不是拿整帧比**：

- 我们关心的是角色有没有走样，不是背景、镜头、光照是否完全一样
- 如果拿整帧比，模型会被迫去保背景和构图，反而压制Animate自由度

**为什么它是弱约束**：

- 动作视频中，角色姿态和局部形变本来就会变化
- 所以这里只要求"角色区域的身份结构别飘太远"，不是要求每帧都像参考图定格

**人类可读的理解**：

`L_struct` 更像一个温和的巡检员：不要求每帧复刻参考图，只在角色明显长歪时提醒模型收回来。

### 6.10 Wan Block完整信息通路

```
┌─────────────────────────────────────────────────────────┐
│ Wan DiT Block（最终版）                                   │
│                                                         │
│ ① e0 AdaLN调制（通道级全局调制）                          │
│   └─ 原始: timestep + T5_pool                           │
│   └─ +统一e0条件增强: C_全局+C_动作→全局+帧级e0增量      │
│   └─ 作用于自注意力和FFN的scale/shift/gate               │
│                                                         │
│ ② 自注意力（token级时空交互）                              │
│   └─ TA-LoRA(rank=32): 时序注意力投影适配                 │
│                                                         │
│ ③ CRLA（角色像素级记忆注入，blocks 5-25）                  │
│   └─ VAE latent→角色参考token→self-attn KV               │
│   └─ crla_scale（唯一保留门控）控制注入强度                │
│                                                         │
│ ④ 解耦交叉注意力（外部条件注入）                           │
│   └─ Text通道: T5 tokens→Wan预训练权重+SA-LoRA(rank=8)   │
│   └─ Route通道: C_角色/C_动作/C_全局/CSG→独立K/V投影     │
│                                                         │
│ ⑤ FFN（e0调制的非线性变换）                               │
│   └─ AdaLN调制来自①的增强e0                              │
│                                                         │
│  唯一门控: crla_scale；其余全部零初始化输出投影             │
└─────────────────────────────────────────────────────────┘
```

增强后的block伪代码：

```python
# === 去噪前一次性预计算 ===
e0_base = timestep_embed(t) + text_pool_embed
e0_per_frame = unified_e0_cond(
    e0_base, pool(C_全局), pool(C_动作), n_frames
)  # [B, T, D_wan]
char_ref = crla.extract_ref_tokens(vae_encode(I_ref))  # [B, 48, D_wan]

# 分层context + text/route拆分
context_shallow = {text: T_t5, route: [C_全局, C_角色粗, CSG_app]}
context_middle  = {text: T_t5, route: [C_全局, C_动作, C_角色粗, CSG_all]}
context_deep    = {text: T_t5, route: [C_全局, C_角色细, CSG_all]}

# === 去噪循环 ===
for each denoising step t:
    scales = TCC.get_scales(t_norm, gs)
    for i, block in enumerate(wan_blocks):
        layer_type = get_layer_type(i)  # shallow/middle/deep
        cur_text = context[layer_type].text
        cur_route = apply_tcc_scales(context[layer_type].route, scales)

        # ① 增强e0的自注意力（含TA-LoRA）
        x = block.self_attn(x, e0_per_frame)

        # ② CRLA角色记忆注入（blocks 5-25）
        if 5 <= i <= 25:
            x = crla(x, char_ref)

        # ③ 解耦双通道交叉注意力
        x = decoupled_cross_attn(
            x, block.q_proj, cur_text, cur_route, block.cross_attn
        )

        # ④ 增强e0的FFN
        x = block.ffn(x, e0_per_frame)
```

**把一个中层block走一遍的直观例子**：

假设当前是：

- 第 `t_n = 0.5` 个去噪阶段（动作最关键）
- 第 `i = 14` 个Wan block（属于中层）
- 当前处理的是蓝色猫"腾空挥手"附近的帧

这时一个block里的事情可以按人类语言理解成：

1. **先看全局状态**  
   `e0_per_frame` 告诉block："这是蓝色猫在跳跃，而且当前帧更接近腾空阶段。"
2. **做自注意力**  
   视频token之间先互相交流，形成这一帧和相邻帧的整体时空结构；`TA-LoRA` 帮它更会处理动画式运动。
3. **查角色像素记忆**  
   因为 `i = 14` 落在 `5-25` 之间，CRLA会提醒它：耳朵别画错、围巾别忘、脸型别漂。
4. **分开读文本和route**  
   - text通道读到"空中挥手"
   - route通道读到"这是短腿蓝色猫，挥手幅度和身体重心要适配它"
5. **做FFN收束**  
   在增强后的 `e0` 调制下，把当前block的综合结果写回主干表示

**为什么这个顺序合理**：

- 先让视频token自己做一轮时空交互，搭出当前帧与相邻帧的大关系
- 再补角色记忆，防止时空建模把身份带跑
- 再读文本和route外部条件，把该帧该怎么动、该长得像谁进一步落实
- 最后在FFN里做非线性整合

**人类可读的理解**：

一个Wan block不是"读一次条件就结束"，而是像一次完整的小工作流：

- 先进入当前任务模式
- 再看视频内部上下文
- 再查角色备忘录
- 再分别读文本和route外部说明
- 最后综合整理后交给下一层

### 6.11 门控审计

| 模块 | 原门控 | 决策 | 理由 |
|------|--------|------|------|
| DIM | gate_m, gate_c, gate_g | **移除** | 输出投影已零初始化，gate冗余 |
| CRLA | crla_scale | **保留** | proj_out非零初始化，需运行时控制注入强度 |
| 统一e0条件增强 | — | **无需** | 输出层零初始化 |
| 解耦交叉注意力 | — | **无需** | route_out_proj零初始化 |

审计结果：管线中唯一保留的门控为 **crla_scale**。

**直观例子**：

这里的核心问题不是"门控越多越安全"，而是"哪些模块真的需要一个额外开关"。

比如：

- `UnifiedE0Conditioning` 的输出层已经零初始化，所以训练刚开始它天然等于旁路，不会突然破坏Wan原行为
- 解耦route通道的 `route_out_proj` 也是零初始化，初期同样等价于"没开这个模块"
- 但CRLA的 `proj_out` 不是零初始化，而且它注入的是强角色记忆，如果不设强度阀门，最容易在运行时过度拉回参考图

**实际流程**：

1. 检查每个新增模块在初始化时，是否已经天然等价于恒等映射/零残差
2. 若是，则不再叠加门控，避免参数和调度复杂度上升
3. 若不是，且该模块可能对生成行为产生强硬写入，则保留门控

**为什么这有助于稳定训练**：

- 零初始化输出层已经提供了"从无到有"的平滑激活路径
- 如果再叠一层门控，很多时候只是把"何时开始学"这个问题又多包了一层，没有本质收益
- 少门控也更容易排查问题：行为异常时，不会分不清到底是模块本身有问题，还是门控没学对

**人类可读的理解**：

本节的原则很简单：  
能靠零初始化自然旁路的模块，就别再多装一个开关；只有像CRLA这种确实可能"写得太猛"的模块，才值得保留显式强度控制。

### 6.12 身份保持主线边界

身份保持相关机制最终只保留4项：

| 机制 | 层级 | 作用 |
|------|------|------|
| 连续gs首帧 | 初始构图约束 | 首帧像素级参考 |
| CSG | 语义锚定 | 低gs时角色判别性外观锚点 |
| CRLA | 像素记忆 | 中后段像素级角色记忆 |
| L_struct | 训练约束 | 弱角色区域一致性约束 |

执行原则（必须严格遵守）：
- 身份模块只负责"这个角色还是这个角色"
- 动作模块负责"这个角色这帧该怎么动"
- **当二者在中层建立阶段冲突时，优先保动作可执行性**，再用深层细节与弱L_struct把角色判别特征拉回来
- 后续若身份仍不足，先检查这4项的调度/层位/增益；只有证明瓶颈不是调度问题时，才考虑额外身份模块

**直观例子**：

蓝色猫参考图可能是站立正视图，但文本要求它完成一个夸张起跳和空中挥手。  
这时身份保持和动作执行天然会有冲突：

- 身份侧会倾向于守住参考图里的脸型、耳朵、围巾和整体体态
- 动作侧会要求身体倾斜、四肢伸展、围巾飘动

如果身份模块过强，就会把动作做僵；如果动作模块过强，角色又会越跳越不像原角色。

**实际流程**：

1. `gs首帧` 在最前面给一个像素级起点
2. `CSG` 在低gs时持续提醒"还是这个角色"
3. `CRLA` 在中后段修复局部身份漂移
4. `L_struct` 在训练期做弱一致性约束

这4项是**同一条身份主线的不同层级**，不是四个彼此抢活的独立身份模块。

**为什么中层冲突时优先保动作**：

- 视频生成首先要成立为"这个角色真的完成了这个动作"
- 若中层动作没建立起来，后面再多身份修复也只会得到一个"像原角色但动作不自然"的结果
- 所以中层先保动作可执行性，深层再把角色判别细节拉回来

**一个具体冲突例子**：

- 文本要求蓝色猫空中大幅挥手
- 参考图里它的手臂是下垂静止的

正确处理应该是：

- 中层允许手臂明显抬起，否则动作不成立
- 深层再确保抬起后的手臂仍然像那只蓝色猫的手臂，而不是变成人类写实手臂

**人类可读的理解**：

这一节其实是在给整个Wan侧定纪律：  
身份保持不是把视频每一帧都拽回参考图，而是在**不毁掉动作可执行性**的前提下，尽量让观众一直认得出这是同一个角色。

---

## 7. learned token与text的关系

### 7.1 为什么是hybrid而非token-only

route token不能完全替代text：
- text携带大量长尾动作、场景、风格、镜头修饰信息
- route token适合承载高价值控制，不适合承载全部开放域语义
- 有限动画监督下强行把所有语义压到少量route token上，语义熵不足

hybrid的分工：
- `text` 负责开放域语义覆盖（"说清楚是什么动作"）
- `route` 负责角色-动作联合控制（"说清楚是谁在做、怎么做才适配"）

### 7.2 如何避免text压制route

不靠单一偏置补丁，而靠机制拆分：

1. **解耦通道**：text和route走独立cross-attention，从根本上消除softmax零和竞争
2. **e0全局调制**：route通过e0/AdaLN获得第二条通路，text不进入此通路
3. **TCC调度**：TCC只调route条件强度，text恒定
4. **独立Dropout**：训练时 `p_text=0.1, p_route=0.1` 独立采样，两类条件都被迫学会在缺失另一方时仍提供增益

### 7.3 最终结论

> 最终主线不是"route替代text"，而是"text提供广义语义，route提供高价值可执行控制；两者通过解耦通道共同驱动Wan"。

Wan侧不再把route位置偏置作为主线。RAB固定在MLLM的RSR取证阶段；进入Wan后，route的竞争力由独立CA_route与route-conditioned e0保证。

---

## 8. 训练策略

### 8.1 Stage 1：路由语义学习

| 维度 | 说明 |
|------|------|
| 目标 | 验证route信息是否真实有效 |
| 数据 | 3000动画 + N万通用视频混合 |
| 可训模块 | Qwen3-VL末4-8层LoRA/轻解冻、RSR、RAB、DIM、证据投影、GuidanceEncoder、CLIPAlignHead |
| 冻结 | Qwen3-VL下层(0-13)、CLIP、T5、Connector、Wan全部 |
| 总可训参数 | ~20M |

损失函数：

```
L_stage1 = 1.0 * L_oracle_ground
         + 1.0 * L_route_distill
         + λ_pair * L_contrast     （DIM之前的route seed；有可靠跨样本pair时 λ_pair=0.2，否则 λ_pair=0）
```

### 8.2 Stage 2：生成能力适配

渐进解冻：

| Phase | 占比 | 可训模块 | 损失 |
|-------|------|---------|------|
| warmup | 15% | Connector + DSN + CSG + 首帧mask + 统一e0条件增强 | L_flow |
| main | 55% | + TA-LoRA + SA-LoRA + CRLA + TCC残差 + 解耦CA | L_flow + λ_struct * L_struct |
| refine | 30% | + MLLM末4层(LR=3e-6) | L_flow + λ_struct * L_struct |

关键设置：
- `gs` 从 Beta(2,2) 分布采样
- `V_ref` 有无混合采样
- route dropout 与 text dropout 独立采样（各10%）
- `λ_struct` 从 0.05 起步，主线稳定后上调到 0.1
- 统一e0条件增强在warmup即引入（零初始化，初期无影响）

### 8.3 Stage 3（可选）：端到端精调

Stage 2稳定收敛后，高质量子集3000-5000 steps。

```
L_stage3 = L_flow + 0.10 * L_struct + 0.001 * L_align
```

梯度路径：L → Wan → Connector+DSN → MLLM末4层(LR=1e-6)。RSR/DIM冻结，保护route分工。

---

## 9. 推理流程

```
输入: I_ref, T, V_ref(可选), gs

1. 证据预编码
   CLIP编I_ref/V_ref, T5编T, 构建E_all

2. MLLM路由推理（Student分支）
   输入gs前缀嵌入, 运行两次RSR+RAB
   → Q_角色(64), Q_动作(48), Q_全局(32)

3. 双向DIM
   → Q_角色', Q_动作', Q_全局'

4. Route Adapter（Connector + DSN）
   → C_角色粗, C_角色细, C_动作, C_全局

5. 预计算Wan条件
   first_frame_latent = gs * VAE(I_ref) + (1-gs) * noise
   CSG_app(8tok), CSG_all(16tok)
   CRLA ref_tokens(48tok)
   e0_per_frame = UnifiedE0Cond(e0_base, pool(C_全局), pool(C_动作))

6. 构建text_context和三组route_context

7. Wan去噪循环
   每步: TCC调度 → 遍历32 blocks:
     增强e0 → self-attn+TA-LoRA → CRLA(5-25) →
     解耦双通道CA → FFN

8. VAE decode → 输出视频
```

---

## 10. 参数预算

### 10.1 总览

| 组件 | 参数量 | 训练阶段 |
|------|--------|---------|
| 证据投影层（5个Proj） | ~12M | Stage 1 |
| GuidanceEncoder + 路由类型嵌入 | ~0.1M | Stage 1 |
| RSR（2路共享RouteCA） | ~4M | Stage 1 |
| RAB | <0.001M | Stage 1 |
| DIM双向版（2分支） | ~3M | Stage 1 |
| CLIPAlignHead | ~1M | Stage 1 |
| **Stage 1 合计** | **~20M** | |
| Connector（4层Transformer+投影） | ~30M | Stage 2 |
| DSN | ~0.1M | Stage 2 |
| CSG StyleProjector（2套） | ~4M | Stage 2 |
| 统一e0条件增强 | ~4M | Stage 2 warmup |
| 解耦双通道交叉注意力 | ~7M | Stage 2 main |
| TA-LoRA (rank=32) | ~40M | Stage 2 main |
| SA-LoRA (rank=8) | ~5M | Stage 2 main |
| CRLA | ~4M | Stage 2 main |
| TCC残差MLP | ~0.003M | Stage 2 main |
| 首帧mask通道 | ~0.01M | Stage 2 warmup |
| **Stage 2 合计** | **~94M** | |
| **总可训练参数** | **~114M** | |

### 10.2 预算原则

- 若要超过120M上限，必须先通过主消融证明信息增益不是由"单纯堆容量"带来的
- 若预算吃紧，先缩token数和LoRA rank，不先新增并行模块
- 优先分配顺序：route adapter（中） > Wan条件执行层（中到高） > route interpreter（低到中） > 身份保持包（低到中）

### 10.3 冻结模型

| 模型 | 参数量 |
|------|--------|
| Qwen3-VL-2B | ~2B |
| CLIP ViT-B/32 | ~0.1B |
| T5-XXL | ~11B |
| Wan 2.2 I2V 5B | ~5B |

---

## 11. 风险分析与评测

### 11.1 典型失败链

- 如果MLLM只做语义复述而不输出可执行控制信息 → Wan依赖text常识生成"通用动作" → 非人角色被人形化
- 如果route与text在同一通道竞争 → Wan优先依赖更熟悉的text → route条件即使存在也被弱化
- 如果身份保持只靠首帧或浅层条件 → 中后段逐渐丢失角色局部结构与纹理
- 如果训练约束把"动作稳定"与"角色适配动作"写成同一目标 → 模型学出折中表示

### 11.2 风险矩阵与回滚策略

| 风险 | 级别 | 门禁 | 回滚策略 |
|------|------|------|---------|
| forward等价性 | 最高 | 未插RSR的forward误差<5e-4 | 修正实现 |
| 路由特化失败 | 高 | Q_角色/Q_动作 cosine<0.5 | 先检查三路Oracle ground与route distill；再增大L_contrast，必要时临时恢复L_orthogonal(0.05) |
| route进入Wan后被忽略 | 高 | route on/off差异显著 | 检查DSN、解耦CA、e0增强 |
| 3000样本过拟合 | 高 | 验证集loss不反转 | 减参/加正则/加通用数据 |
| Oracle/Student梯度冲突 | 高 | 梯度余弦EMA不长期<-0.2 | 降Oracle权重→交替更新 |
| Animate身份不足 | 中-高 | CLIP-I ≥ I2V基线的65% | 提高CSG/CRLA权重，不先加新模块 |
| 统一e0增强破坏预训练 | 中 | 输出权重范数<0.3 | 加权重衰减/临时旁路 |
| 非人角色被人形化 | 中 | 人工评测 | 检查Q_动作←Q_角色是否有效 |
| context显存溢出 | 中 | 1卡稳定性压测 | 降batch/降CSG |

回滚原则：
- route条件本身无效 → 回滚到MLLM端，不在Wan端打补丁
- route有效但Wan不消费 → 回滚到接口层和解耦注入层
- 身份保持不足 → 优先调CSG/CRLA/TCC，不先引入新模块

### 11.3 评测矩阵

所有主评测覆盖 **3类样本 × 3类gs**：

| 子集 | gs=1.0(I2V) | gs=0.5(软) | gs=0.0(Animate) |
|------|-------------|------------|-----------------|
| 人类角色 | ✓ | ✓ | ✓ |
| 非人角色 | ✓ | ✓ | ✓ |
| 长时段连续 | ✓ | ✓ | ✓ |

核心指标：
- CLIP-I：身份与外观一致性
- CLIP-T：动作语义对齐
- FVD/FID：整体视频质量
- 帧间一致性方差：长时序稳定性
- 人工评测：非人角色动作自然度、动作节奏可读性

---

## 12. 消融实验计划

按优先级分步验证：

| 优先级 | 实验 | 预期 | 失败判断 |
|--------|------|------|---------|
| **1** | route条件整体 on/off | route on显著优于off | 若无差异→route无效，回到MLLM |
| **2** | hybrid vs token-only | hybrid优于token-only | 若token-only更好→简化主线 |
| **3** | 解耦双通道CA vs 简单拼接 | 解耦版route消融差更大 | 若差异不大→保持拼接方案 |
| **4** | 统一e0条件增强 on/off | FVD和CLIP-T同步改善 | 若无改善→e0信息冗余→删除 |
| **5** | 轻量RAB on/off | route分工指标改善 | 若无差异→冻结RAB权重为0 |
| **6** | 双向DIM vs 单向DIM | Q_角色特化度+CLIP-I改善 | 若无差异→单向已足够 |
| **7** | TCC on/off | 不同时步条件利用率差异 | 若恒定权重同样好→删除 |
| **8** | CRLA on/off | 中后段身份漂移下降 | 若无差异→CRLA路径有问题 |
| **9** | Stage 2 refine on/off | CLIP-I/T同步改善 | 若route分工被破坏→不做refine |
| **10** | L_struct on/off | 长时序CLIP-I方差改善 | 若无差异→可删除 |

---

## 13. 增量式工程实现计划

### Phase 0：基础设施

- 搭建数据五元组 `(I_ref, T, V_ref?, Y_gt, gs)`；若要启用 `L_contrast`，额外准备可选 `char_id / action_id`
- 完成CLIP/T5特征离线缓存
- 跑通Qwen3-VL与Wan的基础加载和forward验证

**关键看**：特征缓存版本一致性；改写forward后与原模型输出对齐。

### Phase 1：验证route信息是否真有用

- 实现 `Q_角色/Q_动作/Q_全局`
- 加入RSR与轻量RAB
- 建立Oracle/Student训练
- 加入双向DIM

**关键看**：`L_oracle_ground/L_route_distill` 稳定下降；若启用pair训练，再观察 `L_contrast` 下降；同时检查Q_角色与Q_动作余弦相似度是否下降。

**失败信号**：route表示塌成同类向量；关掉RAB/RSR后几乎不变。

### Phase 2：验证接口是否被Wan正确消费

- 实现Route Adapter（Connector + DSN）
- 不改Wan主注入逻辑前先做离线探针
- 检查 `C_*` 统计量是否稳定接近T5分布

**关键看**：DSN前后统计是否变稳；Wan对 `C_*` 的attention是否非随机。

**失败信号**：`C_*` 分布波动大；route条件进入Wan后无使用痕迹。

### Phase 3：验证注入机制主线

- 实现连续gs首帧通道
- 实现统一e0条件增强
- 实现解耦双通道cross-attention
- 实现CSG

**关键看**：hybrid优于token-only；解耦通道优于简单拼接；route on/off产生清晰差异。

**失败信号**：route通道被学成近零；e0增强权重长期不增长。

### Phase 4：验证身份持续保持

- 加入TCC
- 加入CRLA
- 加入L_struct
- 加入基础LoRA

**关键看**：中后段帧身份漂移下降；非人角色长时序更稳。

**失败信号**：CRLA on/off差异极小；gs=0时身份仍明显崩坏。

### Phase 5：联合训练与端到端对齐

- 开启Stage 2 refine
- 必要时开启Stage 3
- 冻结RSR/DIM，仅微调MLLM末4层

**关键看**：CLIP-I与CLIP-T同步改善。

**失败信号**：端到端微调后route特化明显退化。

### Phase 6：消融与回滚

- 按§12消融顺序执行
- 任何新增增强只有在主线稳定后才有资格进入二轮实验
- 候选增强（§14）必须先证明主线瓶颈确实落在对应问题上

回滚原则：
- route本身无效 → 回MLLM端
- route有效但Wan不消费 → 回接口层
- 身份不足 → 先调CSG/CRLA/TCC

---

## 14. 候选增强（暂不入默认主线）

以下模块具备合理的机制动机，但在当前数据规模与工程复杂度下不进入默认主线。后续可作为受控实验，但必须先证明主线瓶颈确实落在对应问题上。

### 14.1 条件化LoRA缩放

**动机**：TA-LoRA权重对所有输入相同，"行走"和"跳跃"使用完全相同的时序适配。条件化缩放根据C_动作动态调节每层LoRA输出强度，使高动态动作获得更强适配。

**与主线的互补关系**：e0增强是通道级全局调制，LoRA缩放是权重级条件化。作用在block的不同层面。

**不入主线原因**：
- 收益未验证，与TCC已有的时步调度存在目标重叠
- ~0.5M参数成本虽小，但增加调参维度
- 主线应先验证基础LoRA是否已足够

**允许考虑的前提**：基础LoRA已稳定，且消融证明不同类型动作的CLIP-T差异显著。

### 14.2 动作时序注意力偏置

**动机**：从C_动作生成帧间注意力偏置矩阵 `[T,T]`，引导不同动作类型的帧间交互模式（周期性行走 vs 多阶段跳跃）。

**不入主线原因**：
- 与TCC、TA-LoRA的目标有部分重叠
- 偏置矩阵可能收敛到近零，说明数据驱动的注意力已足够
- 3000样本下引导信号弱

**允许考虑的前提**：多阶段动作节奏明显失败，且可视化确认帧间注意力模式异常。

### 14.3 其他候选

| 候选 | 不入主线原因 | 考虑前提 |
|------|------------|---------|
| 5路route | 监督稀释，分工容易重复 | 3路已明确出现容量瓶颈 |
| token-only替代text | 损失开放域语义覆盖 | 仅作消融对照 |
| Wan侧route位置偏置 | 解耦CA后已非主矛盾 | route on/off有效但CA_route消费不足 |
| Wan主干recurrent depth | 数据规模下收益与调试成本不匹配 | 明确计算预算支持 |
| 深层VAE细节token | 信息源与CRLA完全重复 | CRLA已调优仍不能保细节 |
| 额外帧间一致性自注意力 | 与CRLA+L_struct目标重叠 | 长时序失败主要来自帧间信息传播不足 |
| 结果级有限轮外循环修复 | 可作HQ增强，但不属默认主线 | 必须复用Q→DIM→Connector→Wan链路 |

---

## 15. 核心设计点与创新点

### 15.1 核心设计点

1. 用 **3-route planner** 把角色、动作、全局三类控制显式分开，只保留最必要的三类，不继续扩路。
2. 用 **route interpreter（RSR + 轻量RAB + 双向DIM）** 先完成高价值取证和角色-动作联合解释，再送往Wan。
3. 用 **route adapter（Connector + DSN）** 做接口对齐，但不把route token伪装成text token。
4. 用 **Wan condition executor** 明确保留text的开放域语义，同时给route独立消费路径（解耦CA）和全局调制路径（统一e0增强）。
5. 用 **身份持续保持包（gs + CSG + CRLA + L_struct）** 统一承载身份保持，坚持"中层动作优先、深层细节回拉"的执行边界。

### 15.2 核心创新点

以下属于"合理借鉴"，不单独视为创新：结构化条件token、分层条件注入、LoRA适配、时步条件调度。

本方案真正具备研究表述价值的创新点：

1. **MLLM作为角色-动作联合解释器**：输出的是可执行控制信息，不是语义摘要。route token在DIM前保持分工约束、DIM后获得联合适配。
2. **text与route的互补机制设计**：不是route替代text，而是通过"route adapter + 解耦双通道注入 + e0全局调制"让两者建成互补关系，从机制上解决softmax零和竞争。
3. **连续模式谱下的统一架构**：gs ∈ [0,1] 连续控制I2V到Animate的过渡，所有模块在同一架构下工作，不维护多套逻辑。
4. **帧级e0条件增强**：让AdaLN通路感知路由条件的全局语义和帧级动作阶段，解决DiT中"所有帧同e0"的固有限制。
5. **无骨架的非人角色动作适配**：通过"Q_动作先读Q_角色→角色形体约束动作实现"的DIM交互顺序，让非人角色动作适配在语义层完成而非骨架映射。

### 15.3 核心关注点

- 如何让MLLM输出的条件不是"描述"而是Wan真能消费的"可执行控制信息"
- 如何同时保留text的开放域覆盖又不让text压制route
- 如何让角色身份全程持续生效而不只在首帧或浅层
- 如何在不依赖骨架前提下让非人角色执行动作且不过度人形化
- 如何在有限动画数据下保持主线足够简洁避免监督稀释

---

## 16. 术语表

| 术语 | 全称 | 含义 |
|------|------|------|
| `Q_角色 / Q_动作 / Q_全局` | Route Query | MLLM输出的3路route token |
| `C_角色粗 / C_角色细 / C_动作 / C_全局` | Route Condition | Connector输出到Wan空间的条件token |
| `RSR` | Route-Specific Reading | 路由专化证据读取 |
| `RAB` | Route Attention Bias | 路由取证偏置（仅RSR内） |
| `DIM` | Dynamic Information Modulation | 路由间双向联合解释 |
| `DSN` | Distribution Statistics Normalization | 分布统计归一化 |
| `CSG` | Character Style Guide | 角色语义锚定token |
| `CRLA` | Character Region Latent Attention | 角色像素记忆注入 |
| `TCC` | Timestep Cascaded Conditioning | 时步条件调度 |
| `gs` | guidance_strength | 连续首图约束强度 |
| `E_all` | Evidence Pool | 统一证据池 |

---

> 本文件为 `pipeline_design_v5_unified_final.md` 的唯一权威口径。后续增强应在不破坏主线清晰性的前提下，以"先消融验证→再升级默认方案"的原则推进。候选增强（§14）只能在主线稳定后作为受控实验引入，不能绕过消融直接进入默认路径。
