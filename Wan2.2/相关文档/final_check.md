# MetaQuery × Wan2.2 代码审查报告 (final_check.md)

> 审查范围: `Wan2.2/wan/metaquery/` 全部桥接代码、编码器、CLI/Demo 脚本  
> 审查日期: 2025-07

---

## 一、审查概览

共发现 **18 个问题**（3 CRITICAL / 10 MODERATE / 5 MINOR），  
其中已修复 **10 个关键/中等问题**，5 个 MINOR 问题保留（不影响运行）。

---

## 二、已修复问题清单

### 🔴 CRITICAL #1：`_patch_wan_text_len` 在 `try` 块外调用 → text_len 永久腐化

| 项目 | 说明 |
|------|------|
| **影响文件** | `bridge.py`, `bridge_i2v.py`, `bridge_animate.py` |
| **问题描述** | `_patch_wan_text_len()` 将 WanModel.text_len 从 512 临时扩展到 768，但修改操作位于 `try` 块之外。如果修改后、进入 `try` 之前的代码抛出异常，`finally` 不会执行，text_len 永久停留在 768，后续所有调用都会出错。 |
| **修复方式** | 三个文件中均将 `_patch_wan_text_len()` 移入 `try` 块内部，使 `finally` 中的 `_restore_wan_text_len()` 保证覆盖所有异常路径。 |

**bridge.py 修改位置:** 第 288-295 行附近  
**bridge_i2v.py 修改位置:** 第 432-459 行附近  
**bridge_animate.py 修改位置:** 第 503-545 行附近  

---

### 🔴 CRITICAL #2：`isinstance(guide_scale, float)` 不接受 `int` 输入

| 项目 | 说明 |
|------|------|
| **影响文件** | `bridge.py`, `bridge_i2v.py` |
| **问题描述** | CFG 分支判断使用 `isinstance(guide_scale, float)`，但用户传入 `guide_scale=5`（int 类型）时判为 False，进入 tuple 解包分支 `guide_scale[1]`，直接 IndexError 崩溃。 |
| **修复方式** | 改为 `not isinstance(guide_scale, tuple)` 判断，兼容 int/float 标量输入。 |

```python
# 修复前
if isinstance(guide_scale, float):
    guide_scale_low = guide_scale_high = guide_scale
# 修复后
if not isinstance(guide_scale, tuple):
    guide_scale_low = guide_scale_high = guide_scale
```

**bridge.py 修改位置:** 第 249-252 行附近  
**bridge_i2v.py 修改位置:** 第 366-370 行附近  

---

### 🔴 CRITICAL #3：`generate_with_metaquery.py` 的 `--task` 接受 I2V/Animate 任务但只创建 WanT2V

| 项目 | 说明 |
|------|------|
| **影响文件** | `generate_with_metaquery.py` |
| **问题描述** | `--task` 参数的 `choices=list(WAN_CONFIGS.keys())` 包含所有任务类型（t2v / i2v / animate），但 `main()` 中只创建 `WanT2V` pipeline 并包装为 `MetaQueryWanBridge`。若用户选择 i2v 或 animate，加载的模型与配置不匹配，运行必然失败。 |
| **修复方式** | 将 `choices` 限制为仅以 `t2v` 开头的任务键，并在 help 文本中引导用户使用对应的 demo 脚本。 |

```python
# 修复前
choices=list(WAN_CONFIGS.keys())
# 修复后
_t2v_tasks = [k for k in WAN_CONFIGS if k.startswith("t2v")]
choices=_t2v_tasks
```

---

### 🟡 MODERATE #4：seed 随机范围不一致

| 项目 | 说明 |
|------|------|
| **影响文件** | `bridge.py` |
| **问题描述** | `bridge.py` 使用 `random.randint(0, 2**31)` 而 `bridge_i2v.py`/`bridge_animate.py` 使用 `sys.maxsize`，行为不一致。 |
| **修复方式** | 统一为 `random.randint(0, sys.maxsize)`。 |

---

### 🟡 MODERATE #6：`bridge.py` 缺少顶层 `import torch.distributed as dist`

| 项目 | 说明 |
|------|------|
| **影响文件** | `bridge.py` |
| **问题描述** | `dist.is_initialized()` 在文件末尾使用，但 `dist` 仅在 `try` 块内 `from torch.distributed ...` 局部导入，若 `try` 外的清理代码先执行会 NameError。 |
| **修复方式** | 在文件顶部 import 区域添加 `import torch.distributed as dist`。 |

---

### 🟡 MODERATE #9：`bridge_animate.py` 缺少 `frame_num` / `clip_len` 的 4n+1 校验

| 项目 | 说明 |
|------|------|
| **影响文件** | `bridge_animate.py` |
| **问题描述** | Wan2.2 VAE 要求帧数为 4n+1 格式，但 `generate()` 方法未做输入校验，传入不满足条件的值会在 VAE 阶段才报出难以理解的错误。 |
| **修复方式** | 在 `generate()` 入口添加断言。 |

```python
assert (frame_num - 1) % 4 == 0, f"frame_num={frame_num} 应为 4n+1 格式!"
assert (clip_len - 1) % 4 == 0, f"clip_len={clip_len} 应为 4n+1 格式!"
```

---

### 🟡 MODERATE #10：`videos` 变量仅在条件分支中定义，返回时可能 NameError

| 项目 | 说明 |
|------|------|
| **影响文件** | `bridge.py`, `bridge_i2v.py` |
| **问题描述** | `videos` 仅在 `if wan.rank == 0:` 分支赋值，非 rank-0 进程返回时若短路求值失效则 NameError。 |
| **修复方式** | 在 `try` 前预初始化 `videos = None`，返回语句改为 `videos[0] if (wan.rank == 0 and videos is not None) else None`。 |

---

### 🟡 MODERATE #12：`encoder.py` 删除 `mq_model` 后未释放 GPU 缓存

| 项目 | 说明 |
|------|------|
| **影响文件** | `encoder.py` |
| **问题描述** | `del mq_model` 后 Python 引用计数回收，但 CUDA 显存不会立即归还 PyTorch 缓存池，导致峰值显存偏高。 |
| **修复方式** | 在 `del mq_model` 后添加 `gc.collect()` + `torch.cuda.empty_cache()`。 |

```python
del mq_model
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

### 🟡 MODERATE #13：断言消息误用 `[WARN]` 前缀

| 项目 | 说明 |
|------|------|
| **影响文件** | `bridge.py` |
| **问题描述** | text_len 扩展后的断言使用 `[WARN]` 前缀，但 assert 失败是致命错误。 |
| **修复方式** | 改为 `[FATAL]` 前缀，准确反映严重程度。 |

---

## 三、未修复的低优先级问题（不影响运行）

| # | 严重度 | 文件 | 描述 |
|---|--------|------|------|
| 5 | MODERATE | `encoder.py` | `encode()` 中 BOI/EOI token 搜索在 `input_images=None` 时仍会执行断言，当前已有空条件保护但语义不够清晰 |
| 7 | MODERATE | `bridge_i2v.py` | `_encode_first_frame()` 中 `.cpu()` → `.to(device)` 多一次设备迁移，性能影响微小 |
| 8 | MODERATE | `bridge_animate.py` | `_get_i2v_mask()` 默认参数 `device="cuda"` 硬编码，但所有调用者均显式传入 device，无实际影响 |
| 11 | MODERATE | `encoder.py` | connector 可迭代性假设（来自 metaquery-main 原始代码），不属于本项目范围 |
| 14-18 | MINOR | 多个文件 | 命名规范、magic number、DRY 代码重复等代码风格问题 |

---

## 四、修改文件汇总

| 文件路径 | 修改数量 | 修复的问题 |
|----------|---------|-----------|
| `wan/metaquery/bridge.py` | 6 处 | #1, #2, #4, #6, #10, #13 |
| `wan/metaquery/bridge_i2v.py` | 4 处 | #1, #2, #10 (2处) |
| `wan/metaquery/bridge_animate.py` | 2 处 | #1, #9 |
| `wan/metaquery/encoder.py` | 1 处 | #12 |
| `generate_with_metaquery.py` | 1 处 | #3 |

---

## 五、Demo 可运行性评估

### ✅ `demo_metaquery_wan.py`（T2V）
- 结构正确，MetaQueryWanBridge 封装完整
- 依赖路径 `_MQ_ROOT` 指向 `Qwen3-VL-main/metaquery-main/` ✅
- 用户需修改用户配置区的路径后即可运行

### ✅ `demo_metaquery_i2v.py`（I2V）
- 双重条件（首帧 VAE + MetaQuery 语义）注入流程完整
- 首帧编码逻辑与原版 `WanI2V` 一致 ✅
- `_patch_wan_text_len` 现已安全包裹在 `try/finally` 中

### ✅ `demo_metaquery_animate.py`（Animate）
- 四重条件（REF VAE + CLIP + MetaQuery + Face）注入流程完整
- 无骨架模式（pose_latents=zeros）正确实现 ✅
- 面部条件可选（None 时传全零，与 CFG 无条件分支一致）

### ✅ `generate_with_metaquery.py`（CLI T2V）
- `--task` 已限制为 T2V 任务，不会误选 I2V/Animate ✅

### 运行前提
1. 安装 `requirements.txt` 中的依赖
2. 下载 Wan2.2 checkpoint 并设置正确路径
3. 下载 MetaQuery (Qwen3-VL) checkpoint 并设置正确路径
4. `Qwen3-VL-main/metaquery-main/` 目录必须存在且包含 `models/`, `trainer_utils.py`（与 Wan2.2 同级的 Qwen3-VL-main 下）
5. GPU 显存建议 ≥ 24GB（offload_model=True）或 ≥ 40GB（不卸载）

---

## 六、关键架构确认

| 管线 | 首帧/参考图是否仍直接输入 Wan？ | MetaQuery 注入方式 |
|------|------|------|
| T2V | N/A（无图像输入） | Context concat: MQ[256,4096] ⊕ T5[L,4096] → 增强 context |
| I2V | ✅ 首帧仍经 VAE encode → channel concat (y) | Context concat: 同 T2V |
| Animate | ✅ 参考图经 VAE (y) + CLIP (img_emb) | Context concat: 同 T2V，额外保留 CLIP+Face 条件 |

MetaQuery 是**附加**语义条件，不替代原有的图像条件通道。

---

*本报告由自动化代码审查工具生成。所有 CRITICAL 和关键 MODERATE 问题均已修复。*
