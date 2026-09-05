"""
inference_metaquery_wan.py
==========================
MetaQuery + Wan2.2 TI2V 推理脚本。

使用训练好的 MetaQuery Connector, 结合 Wan TI2V 5B 生成视频。

★ 推理流程:
    1. MetaQuery Encoder: (参考图 + 文本描述) → MQ features [256, 4096]
    2. T5: 文本描述 → T5 features [512, 4096]
    3. 拼接: context = [MQ_feat + T5_feat] → [768, 4096]
    4. (可选) 参考图 VAE 编码 → first frame latent (i2v 模式)
    5. 扩展 text_len: 512 → 768
    6. 去噪循环: DiT(noise, t, context) → velocity estimate → 迭代采样
    7. VAE 解码 → 视频帧

★ 生成模式:
    - t2v: 文本+MetaQuery → 视频 (纯生成)
    - i2v: 文本+MetaQuery+参考图第一帧 → 视频 (参考图作首帧)

用法:
    python inference_metaquery_wan.py \
        --checkpoint_path /path/to/checkpoint-final/mq_encoder_full.pt \
        --prompt "Tom chases Jerry across the kitchen" \
        --ref_image ./reference.png \
        --mode i2v \
        --output_path output.mp4
"""

import os
import sys
import gc
import json
import math
import random
import argparse
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ── 路径设置 ─────────────────────────────────────────────────────────────────
WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))
METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
sys.path.insert(0, METAQUERY_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Inference: MetaQuery + Wan TI2V")

    # ── 模型路径 ──────────────────────────────────────────────────────────
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="checkpoint 文件或目录路径（支持 mq_encoder_full.pt / checkpoint-final/）")
    p.add_argument("--wan_checkpoint_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
                   help="Wan2.2 TI2V checkpoint 目录")
    p.add_argument("--qwen3vl_model_id", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking",
                   help="Qwen3-VL 模型 ID 或本地路径")

    # ── 输入 ──────────────────────────────────────────────────────────────
    p.add_argument("--prompt", type=str, required=True,
                   help="文本描述")
    p.add_argument("--ref_image", type=str, default=None,
                   help="参考图路径 (用于 MetaQuery 编码, 也用于 i2v 的第一帧)")
    p.add_argument("--negative_prompt", type=str, default="",
                   help="负面提示词")

    # ── 生成参数 ──────────────────────────────────────────────────────────
    p.add_argument("--mode", type=str, default="i2v", choices=["t2v", "i2v"],
                   help="生成模式: t2v 或 i2v")
    p.add_argument(
        "--i2v_method",
        type=str,
        default="legacy_ref_lock",
        choices=["legacy_ref_lock", "animate_ref_slot"],
        help=(
            "i2v 具体实现: "
            "legacy_ref_lock=原有首帧锁定/软锚定方案；"
            "animate_ref_slot=Wan-Animate 风格 reference slot 方案"
        ),
    )
    p.add_argument("--frame_num", type=int, default=81,
                   help="生成帧数 (4n+1)")
    p.add_argument("--size", type=int, nargs=2, default=[832, 480],
                   help="视频尺寸 (宽 高)")
    p.add_argument(
        "--i2v_force_size",
        action="store_true",
        help="i2v 模式下强制使用 --size，不再按参考图比例+max_area自动计算",
    )
    p.add_argument(
        "--i2v_ref_strategy",
        type=str,
        default="animate_like",
        choices=["animate_like", "hard_lock"],
        help=(
            "i2v 首帧注入策略: "
            "animate_like=前期强锚定后期释放（更接近 wan-animate/SCAIL 的非硬锁范式）；"
            "hard_lock=每步硬锁首帧（与旧实现一致）"
        ),
    )
    p.add_argument("--max_area", type=int, default=480 * 832,
                   help="最大面积")
    p.add_argument(
        "--animate_refslot_segment_frames",
        type=int,
        default=78,
        help="animate_ref_slot: 单个推理 segment 的像素帧数（默认 78）",
    )
    p.add_argument(
        "--animate_refslot_ref_frames",
        type=int,
        default=1,
        help="animate_ref_slot: 角色参考静态帧数（默认 1）",
    )
    p.add_argument(
        "--animate_refslot_temporal_frames",
        type=int,
        default=1,
        help="animate_ref_slot: 非首段 temporal reference 帧数（推荐 1 或 5）",
    )
    p.add_argument(
        "--animate_refslot_conditional_frames",
        type=int,
        default=0,
        help="animate_ref_slot: 额外 conditional 帧数；若无条件输入可设 0（会注入全零 latent）",
    )
    p.add_argument(
        "--animate_refslot_preserve_reinject",
        action="store_true",
        default=True,
        help="animate_ref_slot: 每步采样后重注入 preserved prefix（默认开启）",
    )
    p.add_argument(
        "--animate_refslot_no_preserve_reinject",
        action="store_false",
        dest="animate_refslot_preserve_reinject",
        help="animate_ref_slot: 关闭每步 preserved prefix 重注入",
    )
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample_solver", type=str, default="unipc",
                   choices=["unipc", "dpm++"])
    p.add_argument("--seed", type=int, default=42)

    # ── 输出 ──────────────────────────────────────────────────────────────
    p.add_argument("--output_path", type=str, default="output_metaquery.mp4")

    # ── MetaQuery ─────────────────────────────────────────────────────────
    p.add_argument("--num_metaqueries", type=int, default=256)
    p.add_argument("--connector_num_hidden_layers", type=int, default=24)
    p.add_argument(
        "--verify_level",
        type=str,
        default="none",
        choices=["none", "basic", "full"],
        help="验证级别: none/basic/full",
    )
    p.add_argument(
        "--verify_fail_on_warning",
        action="store_true",
        help="开启后，验证 warning 直接视作失败",
    )
    p.add_argument(
        "--verify_report_path",
        type=str,
        default="",
        help="验证报告 JSON 输出路径（留空则自动命名为 output_path.verify.json）",
    )
    p.add_argument(
        "--verify_train_before_checkpoint",
        type=str,
        default="",
        help="训练前基线 checkpoint（用于对比当前 checkpoint 参数是否更新）",
    )

    # ── 设备 ──────────────────────────────────────────────────────────────
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--offload_model", action="store_true",
                   help="DiT/T5 用完后 offload 到 CPU")

    return p.parse_args()


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# =============================================================================
# MetaQuery Encoder (推理模式)
# =============================================================================
class MetaQueryEncoderForWanInference(nn.Module):
    """推理用 MetaQuery Encoder，加载训练好的 checkpoint"""

    WAN_TEXT_DIM = 4096

    def __init__(
        self,
        qwen3vl_model_id: str,
        checkpoint_path: str,
        num_metaqueries: int = 256,
        connector_num_hidden_layers: int = 24,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        verify_level: str = "none",
        fail_on_warning: bool = False,
        train_before_checkpoint_path: str = "",
    ):
        super().__init__()
        self.num_metaqueries = num_metaqueries
        self.wan_text_dim = self.WAN_TEXT_DIM
        self.dtype = dtype
        self.device = torch.device(device)
        self.verify_level = verify_level
        self.fail_on_warning = fail_on_warning
        self.verify_enabled = verify_level != "none"
        self.verify_report: Dict[str, Any] = {
            "verify_level": verify_level,
            "checkpoint_path_input": checkpoint_path,
            "resolved_checkpoint_path": "",
            "resolved_checkpoint_dir": "",
            "training_artifacts": {},
            "state_dict_stats": {},
            "connector_stats": {},
            "checkpoint_update_vs_before": {},
            "warnings": [],
        }

        print("=" * 60)
        print("[MetaQuery Inference] 初始化")
        print(f"  Checkpoint: {checkpoint_path}")
        print("=" * 60)

        # ── 使用训练脚本中定义的同一个类初始化 ─────────────────────────
        from train_connector_for_wan import MetaQueryEncoderForWan
        try:
            from train_metaquery_wan_animate_like import load_mq_encoder_state
            _mq_state_source = "train_metaquery_wan_animate_like"
        except Exception:
            from train_metaquery_wan import load_mq_encoder_state
            _mq_state_source = "train_metaquery_wan"
        self._load_mq_encoder_state_fn = load_mq_encoder_state
        encoder = MetaQueryEncoderForWan(
            qwen3vl_model_id=qwen3vl_model_id,
            num_metaqueries=num_metaqueries,
            connector_num_hidden_layers=connector_num_hidden_layers,
            dtype=dtype,
            device=device,
        )
        print(f"[MetaQuery Inference] load_mq_encoder_state 来源: {_mq_state_source}")

        # ── 加载训练好的权重 ─────────────────────────────────────────────
        state_dict, resolved_path = load_mq_encoder_state(
            checkpoint_path,
            map_location=self.device,
        )
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        print(f"  Resolved ckpt: {resolved_path}")
        print(f"  Missing keys : {len(missing)}")
        print(f"  Unexpected   : {len(unexpected)}")

        self._verify_checkpoint_artifacts(Path(resolved_path))
        self._verify_state_loading(state_dict, missing, unexpected)
        self._verify_checkpoint_updated_vs_before(
            after_state_dict=state_dict,
            train_before_checkpoint_path=train_before_checkpoint_path,
        )
        self._verify_connector_weights(encoder)

        self.encoder = encoder
        self.encoder.eval()
        print("[MetaQuery Inference] ✅ 加载完成")

    def _warn(self, msg: str) -> None:
        self.verify_report["warnings"].append(msg)
        print(f"  [VERIFY][WARN] {msg}")
        if self.fail_on_warning:
            raise RuntimeError(f"[VERIFY] {msg}")

    def _verify_checkpoint_artifacts(self, resolved_path: Path) -> None:
        if not self.verify_enabled:
            return
        resolved_path = resolved_path.resolve()
        ckpt_dir = resolved_path if resolved_path.is_dir() else resolved_path.parent
        self.verify_report["resolved_checkpoint_path"] = str(resolved_path)
        self.verify_report["resolved_checkpoint_dir"] = str(ckpt_dir)

        artifacts = {
            "config.json": (ckpt_dir / "config.json").exists(),
            "trainer_state.json": (ckpt_dir / "trainer_state.json").exists(),
            "optimizer.pt": (ckpt_dir / "optimizer.pt").exists(),
            "scheduler.pt": (ckpt_dir / "scheduler.pt").exists(),
            "training_args.bin": (ckpt_dir / "training_args.bin").exists(),
            "training_args.json": (ckpt_dir / "training_args.json").exists(),
            "metrics_summary.json": (ckpt_dir / "metrics_summary.json").exists(),
            "metrics_tail.json": (ckpt_dir / "metrics_tail.json").exists(),
            "latest": (ckpt_dir.parent / "latest").exists(),
            "mq_encoder_trainable.pt": (ckpt_dir / "mq_encoder_trainable.pt").exists(),
            "mq_encoder_trainable.safetensors": (ckpt_dir / "mq_encoder_trainable.safetensors").exists(),
            "model.safetensors": (ckpt_dir / "model.safetensors").exists(),
            "mq_encoder_full.pt": (ckpt_dir / "mq_encoder_full.pt").exists(),
        }
        self.verify_report["training_artifacts"] = artifacts

        print("  [VERIFY] checkpoint 文件布局检查:")
        for name, exists in artifacts.items():
            print(f"    - {name}: {'OK' if exists else 'MISSING'}")

        required_any = artifacts["model.safetensors"] or artifacts["mq_encoder_full.pt"]
        if not required_any:
            raise RuntimeError(
                "[VERIFY] checkpoint 缺少 model.safetensors 或 mq_encoder_full.pt，"
                "无法证明是完整训练输出"
            )
        if not artifacts["config.json"]:
            self._warn("config.json 缺失，无法核对 num_metaqueries/wan_text_dim 等训练配置")
        if not artifacts["trainer_state.json"]:
            self._warn("trainer_state.json 缺失，无法核对训练步数与格式信息")
        if not artifacts["training_args.json"]:
            self._warn("training_args.json 缺失，无法核对完整训练超参数")
        if not (artifacts["mq_encoder_trainable.pt"] or artifacts["mq_encoder_trainable.safetensors"]):
            self._warn("mq_encoder_trainable.* 缺失，无法复核 trainable 子模块产物")

        trainer_state_path = ckpt_dir / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
                global_step = int(trainer_state.get("global_step", 0))
                ckpt_format = str(trainer_state.get("checkpoint_format", ""))
                self.verify_report["training_artifacts"]["trainer_global_step"] = global_step
                self.verify_report["training_artifacts"]["trainer_checkpoint_format"] = ckpt_format
                if global_step <= 0:
                    self._warn(f"trainer_state.global_step={global_step}，看起来不像已训练完成的 checkpoint")
            except Exception as e:
                self._warn(f"读取 trainer_state.json 失败: {e}")

        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                cfg_num_mq = int(cfg.get("num_metaqueries", self.num_metaqueries))
                cfg_wan_text_dim = int(cfg.get("wan_text_dim", self.wan_text_dim))
                self.verify_report["training_artifacts"]["config_num_metaqueries"] = cfg_num_mq
                self.verify_report["training_artifacts"]["config_wan_text_dim"] = cfg_wan_text_dim
                if cfg_num_mq != self.num_metaqueries:
                    self._warn(
                        f"config.num_metaqueries={cfg_num_mq} 与推理参数 --num_metaqueries={self.num_metaqueries} 不一致"
                    )
                if cfg_wan_text_dim != self.wan_text_dim:
                    raise RuntimeError(
                        f"[VERIFY] config.wan_text_dim={cfg_wan_text_dim} 与 Wan 期望 {self.wan_text_dim} 不一致"
                    )
            except Exception as e:
                self._warn(f"读取 config.json 失败: {e}")

        training_args_json_path = ckpt_dir / "training_args.json"
        if training_args_json_path.exists():
            try:
                targs = json.loads(training_args_json_path.read_text(encoding="utf-8"))
                keys = [
                    "num_train_steps",
                    "num_metaqueries",
                    "frame_num",
                    "max_area",
                    "learning_rate",
                    "warmup_steps",
                    "gradient_accumulation_steps",
                    "null_caption_prob",
                    "null_image_prob",
                ]
                self.verify_report["training_artifacts"]["training_args_excerpt"] = {
                    k: targs.get(k, None) for k in keys
                }
                if "num_metaqueries" in targs:
                    cfg_mq = int(targs.get("num_metaqueries"))
                    if cfg_mq != self.num_metaqueries:
                        self._warn(
                            f"training_args.num_metaqueries={cfg_mq} 与推理参数 --num_metaqueries={self.num_metaqueries} 不一致"
                        )
            except Exception as e:
                self._warn(f"读取 training_args.json 失败: {e}")

        metrics_summary_path = ckpt_dir / "metrics_summary.json"
        if metrics_summary_path.exists():
            try:
                ms = json.loads(metrics_summary_path.read_text(encoding="utf-8"))
                self.verify_report["training_artifacts"]["metrics_summary"] = ms
                if int(ms.get("logged_steps", 0)) <= 0:
                    self._warn("metrics_summary.logged_steps<=0，训练指标记录可能不完整")
            except Exception as e:
                self._warn(f"读取 metrics_summary.json 失败: {e}")

        optimizer_path = ckpt_dir / "optimizer.pt"
        if optimizer_path.exists() and self.verify_level == "full":
            try:
                opt_state = _safe_torch_load(optimizer_path, map_location="cpu")
                state_len = len(opt_state.get("state", {})) if isinstance(opt_state, dict) else 0
                self.verify_report["training_artifacts"]["optimizer_state_entries"] = state_len
                if state_len == 0:
                    self._warn("optimizer.pt 存在但 state 为空")
            except Exception as e:
                self._warn(f"读取 optimizer.pt 失败: {e}")

    def _verify_state_loading(self, state_dict, missing, unexpected) -> None:
        if not self.verify_enabled:
            return

        state_key_count = len(state_dict)
        connector_keys = [k for k in state_dict.keys() if "connector" in k]
        embed_keys = [k for k in state_dict.keys() if "embed_tokens" in k and "weight" in k]
        self.verify_report["state_dict_stats"] = {
            "state_key_count": state_key_count,
            "connector_key_count": len(connector_keys),
            "embed_key_count": len(embed_keys),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
        }
        print(
            f"  [VERIFY] state_dict keys={state_key_count}, connector_keys={len(connector_keys)}, "
            f"embed_keys={len(embed_keys)}"
        )

        if len(connector_keys) == 0:
            raise RuntimeError("[VERIFY] state_dict 中未发现 connector 相关权重，checkpoint 可能错误")

        critical_missing = [
            k for k in missing
            if "connector" in k or "embed_tokens" in k
        ]
        if critical_missing:
            raise RuntimeError(
                f"[VERIFY] load_state_dict 缺失关键参数: {critical_missing[:8]}"
            )
        if unexpected:
            self._warn(f"load_state_dict 存在 unexpected keys (前8个): {unexpected[:8]}")
        if missing:
            self._warn(f"load_state_dict 存在 missing keys (前8个): {missing[:8]}")

    def _verify_connector_weights(self, encoder) -> None:
        if not self.verify_enabled:
            return
        connector = encoder.mllm_model.connector
        total_params = 0
        nonzero_params = 0
        finite_ok = True
        l2_sum = 0.0
        for p in connector.parameters():
            total_params += p.numel()
            nonzero_params += int((p.detach().abs() > 0).sum().item())
            finite_ok = finite_ok and bool(torch.isfinite(p).all())
            l2_sum += float(p.detach().float().norm().item())

        self.verify_report["connector_stats"] = {
            "total_params": int(total_params),
            "nonzero_params": int(nonzero_params),
            "nonzero_ratio": float(nonzero_params / max(total_params, 1)),
            "finite_ok": bool(finite_ok),
            "l2_sum": float(l2_sum),
        }
        print(
            f"  [VERIFY] connector params={total_params:,}, nonzero_ratio={nonzero_params / max(total_params, 1):.6f}, "
            f"finite={finite_ok}, l2_sum={l2_sum:.4f}"
        )
        if total_params == 0:
            raise RuntimeError("[VERIFY] connector 参数量为 0")
        if not finite_ok:
            raise RuntimeError("[VERIFY] connector 参数存在 NaN/Inf")
        if l2_sum <= 0:
            raise RuntimeError("[VERIFY] connector 参数范数为 0，疑似未正常训练/加载")

    def _accumulate_delta_stats(self, before: torch.Tensor, after: torch.Tensor, eps: float = 1e-7) -> Dict[str, float]:
        b = before.detach().to("cpu").reshape(-1)
        a = after.detach().to("cpu").reshape(-1)
        n = b.numel()
        chunk = 1_000_000
        changed = 0
        max_abs = 0.0
        sum_abs = 0.0
        for i in range(0, n, chunk):
            bb = b[i:i + chunk].to(torch.float32)
            aa = a[i:i + chunk].to(torch.float32)
            d = (aa - bb).abs()
            changed += int((d > eps).sum().item())
            local_max = float(d.max().item()) if d.numel() > 0 else 0.0
            if local_max > max_abs:
                max_abs = local_max
            sum_abs += float(d.sum().item())
        mean_abs = sum_abs / max(n, 1)
        return {
            "numel": int(n),
            "changed_elems": int(changed),
            "changed_ratio": float(changed / max(n, 1)),
            "max_abs_delta": float(max_abs),
            "mean_abs_delta": float(mean_abs),
        }

    def _verify_checkpoint_updated_vs_before(
        self,
        after_state_dict: Dict[str, torch.Tensor],
        train_before_checkpoint_path: str,
    ) -> None:
        if not self.verify_enabled:
            return
        if not train_before_checkpoint_path:
            return

        try:
            before_state_dict, before_resolved = self._load_mq_encoder_state_fn(
                train_before_checkpoint_path,
                map_location="cpu",
            )
        except Exception as e:
            self._warn(f"读取训练前 checkpoint 失败: {e}")
            return

        shared_keys = [k for k in after_state_dict.keys() if k in before_state_dict]
        if len(shared_keys) == 0:
            raise RuntimeError("[VERIFY] before/after checkpoint 没有共享参数键，无法比较训练更新")

        connector_keys = [k for k in shared_keys if "mllm_model.connector" in k]
        embed_keys = [k for k in shared_keys if "embed_tokens.weight" in k]

        def summarize(keys: list[str]) -> Dict[str, Any]:
            out = {
                "tensor_count": 0,
                "numel": 0,
                "changed_elems": 0,
                "changed_ratio": 0.0,
                "max_abs_delta": 0.0,
                "mean_abs_delta_weighted": 0.0,
            }
            weighted_mean_sum = 0.0
            valid_tensors = 0
            for key in keys:
                b = before_state_dict[key]
                a = after_state_dict[key]
                if b.shape != a.shape:
                    self._warn(f"before/after shape 不一致: {key} {tuple(b.shape)} vs {tuple(a.shape)}")
                    continue
                stats = self._accumulate_delta_stats(b, a)
                out["numel"] += stats["numel"]
                out["changed_elems"] += stats["changed_elems"]
                if stats["max_abs_delta"] > out["max_abs_delta"]:
                    out["max_abs_delta"] = stats["max_abs_delta"]
                weighted_mean_sum += stats["mean_abs_delta"] * stats["numel"]
                valid_tensors += 1

            out["tensor_count"] = valid_tensors
            out["changed_ratio"] = float(out["changed_elems"] / max(out["numel"], 1))
            out["mean_abs_delta_weighted"] = float(weighted_mean_sum / max(out["numel"], 1))
            return out

        summary_all = summarize(shared_keys)
        summary_connector = summarize(connector_keys)
        summary_embed = summarize(embed_keys)

        self.verify_report["checkpoint_update_vs_before"] = {
            "before_checkpoint_path_input": train_before_checkpoint_path,
            "before_checkpoint_resolved": str(before_resolved),
            "shared_tensor_count": len(shared_keys),
            "all": summary_all,
            "connector": summary_connector,
            "embed_tokens": summary_embed,
        }

        print("[VERIFY] checkpoint 前后参数对比:")
        print(
            f"  all      : tensors={summary_all['tensor_count']} changed_ratio={summary_all['changed_ratio']:.6e} "
            f"max_abs={summary_all['max_abs_delta']:.6e}"
        )
        print(
            f"  connector: tensors={summary_connector['tensor_count']} changed_ratio={summary_connector['changed_ratio']:.6e} "
            f"max_abs={summary_connector['max_abs_delta']:.6e}"
        )
        print(
            f"  embed    : tensors={summary_embed['tensor_count']} changed_ratio={summary_embed['changed_ratio']:.6e} "
            f"max_abs={summary_embed['max_abs_delta']:.6e}"
        )

        if summary_connector["tensor_count"] == 0:
            raise RuntimeError("[VERIFY] 无法在 checkpoint 中定位 connector 参数，无法证明训练更新")
        if summary_connector["changed_elems"] == 0:
            raise RuntimeError(
                "[VERIFY] 对比训练前后 checkpoint，connector 参数未发生变化，训练链路可能异常"
            )
        if summary_all["changed_elems"] == 0:
            raise RuntimeError("[VERIFY] 训练前后 checkpoint 完全一致，训练未生效或路径配置错误")

    @torch.no_grad()
    def encode(self, caption, ref_image=None):
        """
        编码 (文本 + 参考图) → MQ features

        Args:
            caption: str
            ref_image: PIL Image or None

        Returns:
            Tensor [1, 256, 4096]
        """
        captions = [caption]
        images = [[ref_image]] if ref_image is not None else None
        mq_feat = self.encoder(captions, images)
        return mq_feat  # [1, 256, 4096]

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return self


# =============================================================================
# MetaQuery + Wan TI2V 推理管线
# =============================================================================
class MetaQueryWanPipeline:
    """
    使用 MetaQuery 增强的 Wan TI2V 推理管线。

    核心: 将 MQ features 与 T5 features 拼接作为 context,
    扩展 DiT 的 text_len 以容纳额外的 256 个 MQ tokens。
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}")
        self.verify_level = getattr(args, "verify_level", "none")
        self.verify_fail_on_warning = bool(getattr(args, "verify_fail_on_warning", False))
        self.verify_enabled = self.verify_level != "none"
        self.verify_report: Dict[str, Any] = {
            "verify_level": self.verify_level,
            "mode": getattr(args, "mode", "unknown"),
            "checkpoint": {},
            "runtime": {},
            "warnings": [],
        }

        self._load_pipeline()
        self._load_mq_encoder()

    def _load_pipeline(self):
        """加载 Wan TI2V Pipeline"""
        from wan import WanTI2V
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['ti2v-5B']
        self.wan = WanTI2V(
            config=config,
            checkpoint_dir=self.args.wan_checkpoint_dir,
            device_id=self.args.device,
            rank=0,
            t5_cpu=False,
            init_on_cpu=True,
        )
        self.wan_config = config
        self._orig_text_len = self.wan.model.text_len  # 512
        print(f"[Pipeline] Wan TI2V 已加载, text_len={self._orig_text_len}")

    def _load_mq_encoder(self):
        """加载 MetaQuery Encoder"""
        self.mq_encoder = MetaQueryEncoderForWanInference(
            qwen3vl_model_id=self.args.qwen3vl_model_id,
            checkpoint_path=self.args.checkpoint_path,
            num_metaqueries=self.args.num_metaqueries,
            connector_num_hidden_layers=self.args.connector_num_hidden_layers,
            dtype=torch.bfloat16,
            device=f"cuda:{self.args.device}",
            verify_level=self.verify_level,
            fail_on_warning=self.verify_fail_on_warning,
            train_before_checkpoint_path=getattr(self.args, "verify_train_before_checkpoint", ""),
        )
        self.verify_report["checkpoint"] = dict(self.mq_encoder.verify_report)

    def _warn(self, msg: str) -> None:
        self.verify_report["warnings"].append(msg)
        print(f"[VERIFY][WARN] {msg}")
        if self.verify_fail_on_warning:
            raise RuntimeError(f"[VERIFY] {msg}")

    def _record_runtime_metric(self, key: str, value: Any) -> None:
        self.verify_report["runtime"][key] = value

    @staticmethod
    def _compute_i2v_ref_blend_alpha(
        strategy: str,
        step_idx: int,
        total_steps: int,
    ) -> float:
        if strategy == "hard_lock":
            return 1.0
        # animate_like:
        # 在前 35% 步数内从 0.95 余弦衰减到 0，后续不再重注入。
        # 这比 hard_lock 更接近“强参考条件但不锁死 latent”。
        if strategy != "animate_like":
            return 1.0
        if total_steps <= 1:
            return 0.95
        warmup_steps = max(1, int(round(total_steps * 0.35)))
        if step_idx >= warmup_steps:
            return 0.0
        if warmup_steps == 1:
            return 0.95
        p = float(step_idx) / float(warmup_steps - 1)
        alpha = 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))
        return float(max(0.0, min(1.0, alpha)))

    @staticmethod
    def _frames_to_latent_slots(frame_count: int, stride_t: int) -> int:
        f = max(0, int(frame_count))
        if f <= 0:
            return 0
        return int((f - 1) // max(int(stride_t), 1) + 1)

    def _verify_context_plugged(
        self,
        mq_feat: torch.Tensor,
        t5_feat: torch.Tensor,
        aug_feat: torch.Tensor,
        tag: str,
    ) -> None:
        if not self.verify_enabled:
            return
        mq_len = mq_feat.shape[0]
        t5_len = t5_feat.shape[0]
        if aug_feat.shape[0] != mq_len + t5_len:
            raise RuntimeError(
                f"[VERIFY] {tag} 拼接长度异常: aug={aug_feat.shape[0]}, mq+t5={mq_len + t5_len}"
            )
        prefix_ok = torch.allclose(
            aug_feat[:mq_len].float(),
            mq_feat.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        suffix_ok = torch.allclose(
            aug_feat[mq_len:].float(),
            t5_feat.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        self._record_runtime_metric(f"{tag}_mq_tokens", int(mq_len))
        self._record_runtime_metric(f"{tag}_t5_tokens", int(t5_len))
        self._record_runtime_metric(f"{tag}_aug_tokens", int(aug_feat.shape[0]))
        if not prefix_ok:
            raise RuntimeError(f"[VERIFY] {tag} MQ 前缀未正确拼接进 context")
        if not suffix_ok:
            raise RuntimeError(f"[VERIFY] {tag} T5 后缀未正确拼接进 context")

    def _verify_mq_feature_sensitivity(
        self,
        prompt: str,
        negative_prompt: str,
        ref_image: Image.Image | None,
        mq_feat: torch.Tensor,
        mq_feat_null: torch.Tensor,
        mq_feat_noimg: torch.Tensor | None,
        tag: str,
    ) -> None:
        if not self.verify_enabled:
            return
        cond_norm = float(mq_feat.float().norm().item())
        uncond_norm = float(mq_feat_null.float().norm().item())
        diff_norm = float((mq_feat - mq_feat_null).float().norm().item())
        self._record_runtime_metric(f"{tag}_mq_cond_norm", cond_norm)
        self._record_runtime_metric(f"{tag}_mq_uncond_norm", uncond_norm)
        self._record_runtime_metric(f"{tag}_mq_cond_uncond_diff_norm", diff_norm)
        self._record_runtime_metric(f"{tag}_mq_ref_image_provided", int(ref_image is not None))
        if cond_norm <= 0 or uncond_norm <= 0:
            raise RuntimeError("[VERIFY] MQ 特征范数为 0，编码器可能未正常工作")
        if diff_norm <= 1e-6:
            self._warn("MQ 条件/无条件特征几乎无差异，CFG 的 MQ 分支可能无效")

        image_diff = None
        if ref_image is not None:
            if mq_feat_noimg is None:
                mq_feat_noimg = self.mq_encoder.encode(prompt, None)[0].to(
                    self.device, dtype=torch.bfloat16
                )
            image_diff = float((mq_feat - mq_feat_noimg).float().norm().item())
            image_ratio = image_diff / (cond_norm + 1e-8)
            cond_vec = mq_feat.float().reshape(-1)
            noimg_vec = mq_feat_noimg.float().reshape(-1)
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    cond_vec.unsqueeze(0), noimg_vec.unsqueeze(0), dim=1
                ).item()
            )
            self._record_runtime_metric(f"{tag}_mq_image_sensitivity_diff_norm", image_diff)
            self._record_runtime_metric(f"{tag}_mq_image_sensitivity_ratio", image_ratio)
            self._record_runtime_metric(f"{tag}_mq_image_sensitivity_cosine", cosine)
            if image_diff <= 1e-6:
                self._warn("MQ 对参考图不敏感")
        else:
            self._record_runtime_metric(f"{tag}_mq_image_sensitivity_diff_norm", 0.0)
            self._record_runtime_metric(f"{tag}_mq_image_sensitivity_ratio", 0.0)
            self._record_runtime_metric(f"{tag}_mq_image_sensitivity_cosine", 1.0)

        if self.verify_level == "full":
            alt_prompt = prompt + " [mq_probe_variant]"
            mq_feat_alt = self.mq_encoder.encode(alt_prompt, ref_image)[0].to(
                self.device, dtype=torch.bfloat16
            )
            prompt_diff = float((mq_feat - mq_feat_alt).float().norm().item())
            self._record_runtime_metric(f"{tag}_mq_prompt_sensitivity_diff_norm", prompt_diff)
            if prompt_diff <= 1e-6:
                self._warn("MQ 对 prompt 的敏感性过低（full 验证）")

            if image_diff is not None:
                if image_diff <= 1e-6:
                    self._warn("MQ 对参考图不敏感（full 验证）")

    def _verify_wan_image_influence_on_wan(
        self,
        pred_cond: torch.Tensor,
        latent_input: list[torch.Tensor],
        timestep_masked: torch.Tensor,
        seq_len: int,
        mq_feat_noimg: torch.Tensor | None,
        t5_feat: torch.Tensor,
        tag: str,
    ) -> None:
        if not self.verify_enabled:
            return

        if mq_feat_noimg is None:
            self._record_runtime_metric(f"{tag}_wan_image_influence_ref_available", 0)
            self._record_runtime_metric(f"{tag}_wan_image_influence_diff_norm", 0.0)
            self._record_runtime_metric(f"{tag}_wan_image_influence_ratio", 0.0)
            self._record_runtime_metric(f"{tag}_wan_image_influence_cosine", 1.0)
            return

        noimg_context = [torch.cat([mq_feat_noimg, t5_feat], dim=0)]
        pred_noimg = self.wan.model(
            latent_input,
            t=timestep_masked,
            context=noimg_context,
            seq_len=seq_len,
        )[0]
        diff_norm = float((pred_cond - pred_noimg).float().norm().item())
        cond_norm = float(pred_cond.float().norm().item())
        ratio = diff_norm / (cond_norm + 1e-8)
        cond_vec = pred_cond.float().reshape(-1)
        noimg_vec = pred_noimg.float().reshape(-1)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                cond_vec.unsqueeze(0), noimg_vec.unsqueeze(0), dim=1
            ).item()
        )
        self._record_runtime_metric(f"{tag}_wan_image_influence_ref_available", 1)
        self._record_runtime_metric(f"{tag}_wan_image_influence_diff_norm", diff_norm)
        self._record_runtime_metric(f"{tag}_wan_image_influence_ratio", ratio)
        self._record_runtime_metric(f"{tag}_wan_image_influence_cosine", cosine)
        print(
            f"[VERIFY] {tag} 首步图像分支对照: "
            f"||pred(mq_ref+t5)-pred(mq_noimg+t5)||={diff_norm:.6f}, ratio={ratio:.6e}, cosine={cosine:.6f}"
        )
        if diff_norm <= 1e-6:
            self._warn("Wan 侧图像分支影响几乎为 0（mq_ref 与 mq_noimg 预测近乎一致）")

    def _verify_mq_influence_on_wan(
        self,
        pred_cond: torch.Tensor,
        latent_input: list[torch.Tensor],
        timestep_masked: torch.Tensor,
        seq_len: int,
        mq_feat: torch.Tensor,
        t5_feat: torch.Tensor,
        tag: str,
    ) -> None:
        if not self.verify_enabled:
            return
        zero_mq_context = [torch.cat([torch.zeros_like(mq_feat), t5_feat], dim=0)]
        pred_t5_only = self.wan.model(
            latent_input,
            t=timestep_masked,
            context=zero_mq_context,
            seq_len=seq_len,
        )[0]
        diff_norm = float((pred_cond - pred_t5_only).float().norm().item())
        cond_norm = float(pred_cond.float().norm().item())
        ratio = diff_norm / (cond_norm + 1e-8)
        self._record_runtime_metric(f"{tag}_mq_influence_diff_norm", diff_norm)
        self._record_runtime_metric(f"{tag}_mq_influence_ratio", ratio)
        print(
            f"[VERIFY] {tag} 首步对照: ||pred(mq+t5)-pred(zero_mq+t5)||={diff_norm:.6f}, "
            f"ratio={ratio:.6e}"
        )
        if diff_norm <= 1e-6:
            raise RuntimeError(
                "[VERIFY] MQ 置零前后 Wan 预测几乎无差异，说明 checkpoint 可能未真实参与去噪"
            )

    def dump_verify_report(self, report_path: str) -> None:
        if not self.verify_enabled:
            return
        if not report_path:
            return
        warnings = self.verify_report.get("warnings", [])
        self.verify_report["summary"] = {
            "status": "pass_with_warnings" if warnings else "pass",
            "warning_count": len(warnings),
        }
        _write_json(Path(report_path), self.verify_report)
        print(f"[VERIFY] 报告已写入: {report_path}")

    def generate_i2v(
        self,
        prompt: str,
        ref_image: Image.Image,
        negative_prompt: str = "",
        max_area: int = 480 * 832,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        seed: int = 42,
    ):
        """
        MetaQuery 增强的 i2v 生成。

        参考图同时用于:
        1. MetaQuery 编码 (角色/场景理解)
        2. VAE 编码作为第一帧 (i2v 条件)
        """
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from wan.utils.utils import best_output_size, masks_like

        device = self.device
        num_mq = self.args.num_metaqueries

        if not negative_prompt:
            negative_prompt = self.wan.sample_neg_prompt

        # ── 1. MetaQuery 编码 ────────────────────────────────────────────
        print("[Generate] MetaQuery 编码...")
        mq_feat = self.mq_encoder.encode(prompt, ref_image)  # [1, 256, 4096]
        mq_feat = mq_feat[0].to(device, dtype=torch.bfloat16)  # [256, 4096]
        mq_feat_noimg = None
        if ref_image is not None:
            mq_feat_noimg = self.mq_encoder.encode(prompt, None)[0].to(
                device, dtype=torch.bfloat16
            )
        null_prompt = negative_prompt
        null_ref_image = Image.new("RGB", ref_image.size) if ref_image is not None else None
        mq_feat_null = self.mq_encoder.encode(null_prompt, null_ref_image)[0].to(
            device, dtype=torch.bfloat16
        )
        self._verify_mq_feature_sensitivity(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ref_image=ref_image,
            mq_feat=mq_feat,
            mq_feat_null=mq_feat_null,
            mq_feat_noimg=mq_feat_noimg,
            tag="i2v",
        )

        # ── 2. T5 编码 ──────────────────────────────────────────────────
        print("[Generate] T5 编码...")
        self.wan.text_encoder.model.to(device)
        t5_context = self.wan.text_encoder([prompt], device)      # List[Tensor [512, 4096]]
        t5_null = self.wan.text_encoder([negative_prompt], device)
        if self.args.offload_model:
            self.wan.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # ── 3. 拼接 context = [MQ + T5] ─────────────────────────────────
        aug_context = [torch.cat([mq_feat, t5_context[0]], dim=0)]      # [768, 4096]
        aug_null    = [torch.cat([mq_feat_null, t5_null[0]], dim=0)]    # [768, 4096]

        print(f"  MQ features  : {mq_feat.shape}")
        print(f"  T5 features  : {t5_context[0].shape}")
        print(f"  Aug context  : {aug_context[0].shape}")
        self._verify_context_plugged(mq_feat, t5_context[0], aug_context[0], tag="i2v")

        # ── 4. 图像预处理 (和 WanTI2V.i2v 一样) ─────────────────────────
        ih, iw = ref_image.height, ref_image.width
        dh = self.wan_config.patch_size[1] * self.wan_config.vae_stride[1]
        dw = self.wan_config.patch_size[2] * self.wan_config.vae_stride[2]
        if self.args.i2v_force_size:
            req_w, req_h = int(self.args.size[0]), int(self.args.size[1])
            if req_w < dw or req_h < dh:
                raise ValueError(
                    f"i2v_force_size 要求的尺寸过小: {req_w}x{req_h}, 最小应 >= {dw}x{dh}"
                )
            # 保证满足 patch/vae 对齐（32 对齐）
            ow = (req_w // dw) * dw
            oh = (req_h // dh) * dh
            if ow != req_w or oh != req_h:
                print(
                    f"[Generate][i2v] i2v_force_size 对齐修正: {req_w}x{req_h} -> {ow}x{oh} "
                    f"(align {dw}x{dh})"
                )
        else:
            ow, oh = best_output_size(iw, ih, dw, dh, max_area)
        print(
            f"[Generate][i2v] output_size={ow}x{oh} (input_ref={iw}x{ih}, "
            f"force_size={self.args.i2v_force_size}, max_area={max_area})"
        )

        scale = max(ow / iw, oh / ih)
        img_resized = ref_image.resize(
            (round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (img_resized.width - ow) // 2
        y1 = (img_resized.height - oh) // 2
        img_cropped = img_resized.crop((x1, y1, x1 + ow, y1 + oh))

        img_tensor = TF.to_tensor(img_cropped).sub_(0.5).div_(0.5)
        img_tensor = img_tensor.to(device).unsqueeze(1)  # [3, 1, H, W]

        # ── 5. 计算 seq_len 和噪声 ──────────────────────────────────────
        F = frame_num
        vae_stride = self.wan_config.vae_stride
        patch_size = self.wan_config.patch_size

        latent_T = (F - 1) // vae_stride[0] + 1
        latent_H = oh // vae_stride[1]
        latent_W = ow // vae_stride[2]
        z_dim = self.wan.vae.model.z_dim

        seq_len = latent_T * latent_H * latent_W // (patch_size[1] * patch_size[2])
        seq_len = int(math.ceil(seq_len))

        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed if seed >= 0 else random.randint(0, sys.maxsize))

        noise = torch.randn(
            z_dim, latent_T, latent_H, latent_W,
            dtype=torch.float32, device=device, generator=seed_g)

        # ── 6. VAE 编码参考图作为第一帧 ──────────────────────────────────
        z = self.wan.vae.encode([img_tensor])  # z[0]: [C_z, 1, H', W']

        # ── 7. 掩码 (第一帧固定) ─────────────────────────────────────────
        mask1, mask2 = masks_like([noise], zero=True)
        # mask2[0][:, 0] = 0, mask2[0][:, 1:] = 1
        latent = (1.0 - mask2[0]) * z[0] + mask2[0] * noise

        # ── 8. 扩展 text_len → 去噪循环 ─────────────────────────────────
        aug_text_len = self._orig_text_len + num_mq
        self.wan.model.text_len = aug_text_len
        ref_strategy = str(getattr(self.args, "i2v_ref_strategy", "animate_like")).strip().lower()
        if ref_strategy not in ("animate_like", "hard_lock"):
            raise ValueError(f"Unknown --i2v_ref_strategy: {ref_strategy}")
        self._record_runtime_metric("i2v_ref_strategy", ref_strategy)
        self._record_runtime_metric("i2v_ref_strategy_alpha0_nominal", 0.95)
        self._record_runtime_metric("i2v_ref_strategy_warmup_ratio", 0.35)
        print(f"  text_len: {self._orig_text_len} → {aug_text_len}")
        print(f"  i2v_ref_strategy={ref_strategy}")
        self._record_runtime_metric("i2v_text_len_before", int(self._orig_text_len))
        self._record_runtime_metric("i2v_text_len_after", int(self.wan.model.text_len))
        if self.wan.model.text_len != aug_text_len:
            raise RuntimeError(
                f"[VERIFY] i2v text_len 设置失败: current={self.wan.model.text_len}, expected={aug_text_len}"
            )

        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.wan.model, 'no_sync', noop_no_sync)

        with (
            torch.amp.autocast('cuda', dtype=self.wan.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            # 调度器
            if sample_solver == 'unipc':
                scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.wan.num_train_timesteps,
                    shift=1, use_dynamic_shifting=False)
                scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
                timesteps = scheduler.timesteps
            elif sample_solver == 'dpm++':
                scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.wan.num_train_timesteps,
                    shift=1, use_dynamic_shifting=False)
                sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    scheduler, device=device, sigmas=sigmas)
            else:
                raise NotImplementedError(f"Unknown solver: {sample_solver}")

            self.wan.model.to(device)
            torch.cuda.empty_cache()

            print(f"[Generate] 开始去噪 ({len(timesteps)} steps)...")
            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_input = [latent.to(device)]
                timestep = torch.stack([t]).to(device)

                # 时间步掩码 (第一帧 t=0)
                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep_masked = temp_ts.unsqueeze(0)

                # CFG: 条件预测
                pred_cond = self.wan.model(
                    latent_input, t=timestep_masked,
                    context=aug_context, seq_len=seq_len)[0]

                # CFG: 无条件预测
                pred_uncond = self.wan.model(
                    latent_input, t=timestep_masked,
                    context=aug_null, seq_len=seq_len)[0]

                if step_idx == 0:
                    self._verify_mq_influence_on_wan(
                        pred_cond=pred_cond,
                        latent_input=latent_input,
                        timestep_masked=timestep_masked,
                        seq_len=seq_len,
                        mq_feat=mq_feat,
                        t5_feat=t5_context[0],
                        tag="i2v",
                    )
                    self._verify_wan_image_influence_on_wan(
                        pred_cond=pred_cond,
                        latent_input=latent_input,
                        timestep_masked=timestep_masked,
                        seq_len=seq_len,
                        mq_feat_noimg=mq_feat_noimg,
                        t5_feat=t5_context[0],
                        tag="i2v",
                    )

                # 引导
                pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)

                # 调度器步进
                temp_x0 = scheduler.step(
                    pred.unsqueeze(0), t,
                    latent.unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                # 重新应用首帧参考（animate_like 或 hard_lock）
                alpha = self._compute_i2v_ref_blend_alpha(
                    strategy=ref_strategy,
                    step_idx=step_idx,
                    total_steps=len(timesteps),
                )
                if step_idx == 0:
                    self._record_runtime_metric("i2v_ref_alpha_step0", float(alpha))
                if step_idx == len(timesteps) - 1:
                    self._record_runtime_metric("i2v_ref_alpha_last", float(alpha))
                if alpha > 0.0:
                    ref_anchor = alpha * z[0] + (1.0 - alpha) * latent
                    latent = (1.0 - mask2[0]) * ref_anchor + mask2[0] * latent

            x0 = [latent]

            if self.args.offload_model:
                self.wan.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # VAE 解码
            print("[Generate] VAE 解码...")
            videos = self.wan.vae.decode(x0)

        # 恢复 text_len
        self.wan.model.text_len = self._orig_text_len
        self._record_runtime_metric("i2v_text_len_restored", int(self.wan.model.text_len))
        if self.wan.model.text_len != self._orig_text_len:
            raise RuntimeError(
                f"[VERIFY] i2v text_len 未恢复: current={self.wan.model.text_len}, expected={self._orig_text_len}"
            )

        del noise, latent, x0
        gc.collect()
        torch.cuda.empty_cache()

        return videos[0]

    def generate_i2v_animate_ref_slot(
        self,
        prompt: str,
        ref_image: Image.Image,
        negative_prompt: str = "",
        max_area: int = 480 * 832,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        seed: int = 42,
    ):
        """
        Wan-Animate 风格 reference-slot i2v:
        1) 参考图编码为独立 reference latent
        2) reference/temporal/conditional prefix 与 target latent 在时间维拼接
        3) preserved prefix 的 token timestep 置 0（可选每步重注入）
        4) prefix 对应输出丢弃，仅保留 target frames 组成最终视频
        """
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from wan.utils.utils import best_output_size

        device = self.device
        num_mq = self.args.num_metaqueries
        self._record_runtime_metric("i2v_method", "animate_ref_slot")

        if not negative_prompt:
            negative_prompt = self.wan.sample_neg_prompt

        # ── 1. MetaQuery 编码 ────────────────────────────────────────────
        print("[Generate][animate_ref_slot] MetaQuery 编码...")
        mq_feat = self.mq_encoder.encode(prompt, ref_image)[0].to(device, dtype=torch.bfloat16)
        mq_feat_noimg = self.mq_encoder.encode(prompt, None)[0].to(device, dtype=torch.bfloat16)
        null_ref_image = Image.new("RGB", ref_image.size)
        mq_feat_null = self.mq_encoder.encode(negative_prompt, null_ref_image)[0].to(
            device, dtype=torch.bfloat16
        )
        self._verify_mq_feature_sensitivity(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ref_image=ref_image,
            mq_feat=mq_feat,
            mq_feat_null=mq_feat_null,
            mq_feat_noimg=mq_feat_noimg,
            tag="i2v_animate_ref_slot",
        )

        # ── 2. T5 编码 ──────────────────────────────────────────────────
        print("[Generate][animate_ref_slot] T5 编码...")
        self.wan.text_encoder.model.to(device)
        t5_context = self.wan.text_encoder([prompt], device)
        t5_null = self.wan.text_encoder([negative_prompt], device)
        if self.args.offload_model:
            self.wan.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # ── 3. 拼接 context = [MQ + T5] ─────────────────────────────────
        aug_context = [torch.cat([mq_feat, t5_context[0]], dim=0)]
        aug_null = [torch.cat([mq_feat_null, t5_null[0]], dim=0)]
        self._verify_context_plugged(mq_feat, t5_context[0], aug_context[0], tag="i2v_animate_ref_slot")

        # ── 4. 图像预处理 + reference latent ─────────────────────────────
        ih, iw = ref_image.height, ref_image.width
        patch_size = self.wan_config.patch_size
        vae_stride = self.wan_config.vae_stride
        dh = patch_size[1] * vae_stride[1]
        dw = patch_size[2] * vae_stride[2]
        if self.args.i2v_force_size:
            req_w, req_h = int(self.args.size[0]), int(self.args.size[1])
            if req_w < dw or req_h < dh:
                raise ValueError(
                    f"i2v_force_size 要求的尺寸过小: {req_w}x{req_h}, 最小应 >= {dw}x{dh}"
                )
            ow = (req_w // dw) * dw
            oh = (req_h // dh) * dh
        else:
            ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img_resized = ref_image.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (img_resized.width - ow) // 2
        y1 = (img_resized.height - oh) // 2
        img_cropped = img_resized.crop((x1, y1, x1 + ow, y1 + oh))
        img_tensor = TF.to_tensor(img_cropped).sub_(0.5).div_(0.5).to(device).unsqueeze(1)

        ref_lat = self.wan.vae.encode([img_tensor])[0].float()  # [C, 1, H', W']
        C, _, latent_H, latent_W = ref_lat.shape

        # ── 5. slot 配置（像素帧数 -> latent slots）───────────────────────
        stride_t = int(vae_stride[0])
        target_lat_T = int((frame_num - 1) // stride_t + 1)
        segment_frames = int(getattr(self.args, "animate_refslot_segment_frames", 78))
        segment_lat_T = self._frames_to_latent_slots(segment_frames, stride_t)
        ref_slots_lat = max(1, self._frames_to_latent_slots(
            int(getattr(self.args, "animate_refslot_ref_frames", 1)), stride_t
        ))
        temporal_slots_lat_cfg = self._frames_to_latent_slots(
            int(getattr(self.args, "animate_refslot_temporal_frames", 1)), stride_t
        )
        cond_slots_lat = self._frames_to_latent_slots(
            int(getattr(self.args, "animate_refslot_conditional_frames", 0)), stride_t
        )
        reinject_preserved = bool(getattr(self.args, "animate_refslot_preserve_reinject", True))

        if segment_lat_T <= ref_slots_lat + cond_slots_lat:
            raise ValueError(
                f"animate_ref_slot 参数非法: segment_lat_T={segment_lat_T}, "
                f"ref_slots={ref_slots_lat}, cond_slots={cond_slots_lat}"
            )

        self._record_runtime_metric("i2v_animate_refslot_segment_frames", int(segment_frames))
        self._record_runtime_metric("i2v_animate_refslot_segment_latent_slots", int(segment_lat_T))
        self._record_runtime_metric("i2v_animate_refslot_ref_slots_latent", int(ref_slots_lat))
        self._record_runtime_metric("i2v_animate_refslot_temporal_slots_latent_cfg", int(temporal_slots_lat_cfg))
        self._record_runtime_metric("i2v_animate_refslot_conditional_slots_latent", int(cond_slots_lat))
        self._record_runtime_metric("i2v_animate_refslot_reinject_preserved", int(reinject_preserved))
        print(
            f"[Generate][animate_ref_slot] target_lat={target_lat_T}, segment_lat={segment_lat_T}, "
            f"ref_slots={ref_slots_lat}, temporal_slots_cfg={temporal_slots_lat_cfg}, cond_slots={cond_slots_lat}"
        )

        # ── 6. 扩展 text_len + 去噪 ──────────────────────────────────────
        aug_text_len = self._orig_text_len + num_mq
        self.wan.model.text_len = aug_text_len
        self._record_runtime_metric("i2v_animate_refslot_text_len_before", int(self._orig_text_len))
        self._record_runtime_metric("i2v_animate_refslot_text_len_after", int(self.wan.model.text_len))

        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed if seed >= 0 else random.randint(0, sys.maxsize))

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.wan.model, 'no_sync', noop_no_sync)

        try:
            with (
                torch.amp.autocast('cuda', dtype=self.wan.param_dtype),
                torch.no_grad(),
                no_sync(),
            ):
                if sample_solver == 'unipc':
                    scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.wan.num_train_timesteps,
                        shift=1, use_dynamic_shifting=False)
                    scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
                    timesteps = scheduler.timesteps
                elif sample_solver == 'dpm++':
                    scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.wan.num_train_timesteps,
                        shift=1, use_dynamic_shifting=False)
                    sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas)
                else:
                    raise NotImplementedError(f"Unknown solver: {sample_solver}")

                self.wan.model.to(device)
                torch.cuda.empty_cache()

                target_latents_collected = []
                remaining_lat = int(target_lat_T)
                prev_temporal_lat = None
                seg_idx = 0
                did_first_verify = False

                while remaining_lat > 0:
                    temporal_slots_use = int(temporal_slots_lat_cfg if seg_idx > 0 else 0)
                    ref_prefix = ref_lat.repeat(1, ref_slots_lat, 1, 1)

                    if temporal_slots_use > 0:
                        if prev_temporal_lat is None or prev_temporal_lat.numel() == 0:
                            temporal_prefix = torch.zeros(
                                C, temporal_slots_use, latent_H, latent_W,
                                device=device, dtype=torch.float32
                            )
                        else:
                            temporal_prefix = prev_temporal_lat[:, -temporal_slots_use:, :, :].float()
                            if temporal_prefix.shape[1] < temporal_slots_use:
                                pad_t = temporal_slots_use - temporal_prefix.shape[1]
                                pad = torch.zeros(
                                    C, pad_t, latent_H, latent_W,
                                    device=device, dtype=torch.float32
                                )
                                temporal_prefix = torch.cat([pad, temporal_prefix], dim=1)
                    else:
                        temporal_prefix = torch.zeros(
                            C, 0, latent_H, latent_W, device=device, dtype=torch.float32
                        )

                    cond_prefix = torch.zeros(
                        C, cond_slots_lat, latent_H, latent_W,
                        device=device, dtype=torch.float32
                    )
                    preserved_prefix = torch.cat([ref_prefix, temporal_prefix, cond_prefix], dim=1)
                    prefix_slots = int(preserved_prefix.shape[1])
                    target_slots_per_seg = int(segment_lat_T - prefix_slots)
                    if target_slots_per_seg <= 0:
                        raise RuntimeError(
                            f"segment_lat_T={segment_lat_T} <= prefix_slots={prefix_slots}, "
                            "无法产生 target latent"
                        )
                    take_slots = min(remaining_lat, target_slots_per_seg)

                    noise = torch.randn(
                        C, segment_lat_T, latent_H, latent_W,
                        dtype=torch.float32, device=device, generator=seed_g
                    )
                    latent = noise.clone()
                    latent[:, :prefix_slots] = preserved_prefix

                    tokens_per_frame = int(math.ceil((latent_H * latent_W) / (patch_size[1] * patch_size[2])))
                    seq_len = int(tokens_per_frame * segment_lat_T)
                    prefix_token_count = int(min(seq_len, prefix_slots * tokens_per_frame))

                    print(
                        f"[Generate][animate_ref_slot] seg={seg_idx} prefix_slots={prefix_slots} "
                        f"(ref={ref_slots_lat}, temporal={temporal_slots_use}, cond={cond_slots_lat}) "
                        f"target_slots={take_slots}/{target_slots_per_seg} remaining_lat={remaining_lat}"
                    )

                    for step_idx, t in enumerate(tqdm(timesteps, desc=f"seg{seg_idx} denoise", leave=False)):
                        latent_input = [latent]
                        timestep = torch.stack([t]).to(device)
                        t_scalar = float(timestep.item())
                        t_row = torch.full((seq_len,), t_scalar, device=device, dtype=torch.float32)
                        if prefix_token_count > 0:
                            t_row[:prefix_token_count] = 0.0
                        timestep_masked = t_row.unsqueeze(0)

                        pred_cond = self.wan.model(
                            latent_input, t=timestep_masked,
                            context=aug_context, seq_len=seq_len)[0]
                        pred_uncond = self.wan.model(
                            latent_input, t=timestep_masked,
                            context=aug_null, seq_len=seq_len)[0]

                        if not did_first_verify and step_idx == 0:
                            self._verify_mq_influence_on_wan(
                                pred_cond=pred_cond,
                                latent_input=latent_input,
                                timestep_masked=timestep_masked,
                                seq_len=seq_len,
                                mq_feat=mq_feat,
                                t5_feat=t5_context[0],
                                tag="i2v_animate_ref_slot",
                            )
                            self._verify_wan_image_influence_on_wan(
                                pred_cond=pred_cond,
                                latent_input=latent_input,
                                timestep_masked=timestep_masked,
                                seq_len=seq_len,
                                mq_feat_noimg=mq_feat_noimg,
                                t5_feat=t5_context[0],
                                tag="i2v_animate_ref_slot",
                            )
                            did_first_verify = True

                        pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)
                        temp_x0 = scheduler.step(
                            pred.unsqueeze(0), t,
                            latent.unsqueeze(0),
                            return_dict=False, generator=seed_g)[0]
                        latent = temp_x0.squeeze(0)

                        if reinject_preserved and prefix_slots > 0:
                            latent[:, :prefix_slots] = preserved_prefix

                    segment_target = latent[:, prefix_slots:prefix_slots + target_slots_per_seg]
                    target_latents_collected.append(segment_target[:, :take_slots].clone())

                    if temporal_slots_lat_cfg > 0:
                        prev_temporal_lat = segment_target[:, -min(segment_target.shape[1], temporal_slots_lat_cfg):].detach()
                    else:
                        prev_temporal_lat = None

                    remaining_lat -= int(take_slots)
                    seg_idx += 1

                full_latent = torch.cat(target_latents_collected, dim=1)[:, :target_lat_T]
                self._record_runtime_metric("i2v_animate_refslot_segments", int(seg_idx))
                self._record_runtime_metric("i2v_animate_refslot_output_latent_slots", int(full_latent.shape[1]))

                print("[Generate][animate_ref_slot] VAE 解码...")
                videos = self.wan.vae.decode([full_latent])

                if self.args.offload_model:
                    self.wan.model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        finally:
            self.wan.model.text_len = self._orig_text_len
            self._record_runtime_metric("i2v_animate_refslot_text_len_restored", int(self.wan.model.text_len))

        gc.collect()
        torch.cuda.empty_cache()
        return videos[0]

    def generate_t2v(
        self,
        prompt: str,
        ref_image: Image.Image = None,
        negative_prompt: str = "",
        size: tuple = (832, 480),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        seed: int = 42,
    ):
        """
        MetaQuery 增强的 t2v 生成。

        参考图仅用于 MetaQuery 编码 (角色理解)，
        视频完全从噪声生成 (无第一帧约束)。
        """
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from wan.utils.utils import masks_like

        device = self.device
        num_mq = self.args.num_metaqueries

        if not negative_prompt:
            negative_prompt = self.wan.sample_neg_prompt

        # ── 1. MetaQuery 编码 ────────────────────────────────────────────
        print("[Generate] MetaQuery 编码...")
        mq_feat = self.mq_encoder.encode(prompt, ref_image)  # [1, 256, 4096]
        mq_feat = mq_feat[0].to(device, dtype=torch.bfloat16)  # [256, 4096]
        mq_feat_noimg = None
        if ref_image is not None:
            mq_feat_noimg = self.mq_encoder.encode(prompt, None)[0].to(
                device, dtype=torch.bfloat16
            )
        null_prompt = negative_prompt
        null_ref_image = Image.new("RGB", ref_image.size) if ref_image is not None else None
        mq_feat_null = self.mq_encoder.encode(null_prompt, null_ref_image)[0].to(
            device, dtype=torch.bfloat16
        )
        self._verify_mq_feature_sensitivity(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ref_image=ref_image,
            mq_feat=mq_feat,
            mq_feat_null=mq_feat_null,
            mq_feat_noimg=mq_feat_noimg,
            tag="t2v",
        )

        # ── 2. T5 编码 ──────────────────────────────────────────────────
        print("[Generate] T5 编码...")
        self.wan.text_encoder.model.to(device)
        t5_context = self.wan.text_encoder([prompt], device)
        t5_null = self.wan.text_encoder([negative_prompt], device)
        if self.args.offload_model:
            self.wan.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # ── 3. 拼接 context ─────────────────────────────────────────────
        aug_context = [torch.cat([mq_feat, t5_context[0]], dim=0)]
        aug_null    = [torch.cat([mq_feat_null, t5_null[0]], dim=0)]
        self._verify_context_plugged(mq_feat, t5_context[0], aug_context[0], tag="t2v")

        # ── 4. 计算尺寸 ─────────────────────────────────────────────────
        W, H = size
        vae_stride = self.wan_config.vae_stride
        patch_size = self.wan_config.patch_size

        F = frame_num
        z_dim = self.wan.vae.model.z_dim
        target_shape = (
            z_dim,
            (F - 1) // vae_stride[0] + 1,
            H // vae_stride[1],
            W // vae_stride[2],
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (patch_size[1] * patch_size[2]) * target_shape[1])

        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed if seed >= 0 else random.randint(0, sys.maxsize))
        noise = torch.randn(*target_shape, dtype=torch.float32,
                            device=device, generator=seed_g)

        # ── 5. 去噪循环 ─────────────────────────────────────────────────
        aug_text_len = self._orig_text_len + num_mq
        self.wan.model.text_len = aug_text_len
        self._record_runtime_metric("t2v_text_len_before", int(self._orig_text_len))
        self._record_runtime_metric("t2v_text_len_after", int(self.wan.model.text_len))
        if self.wan.model.text_len != aug_text_len:
            raise RuntimeError(
                f"[VERIFY] t2v text_len 设置失败: current={self.wan.model.text_len}, expected={aug_text_len}"
            )

        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.wan.model, 'no_sync', noop_no_sync)

        with (
            torch.amp.autocast('cuda', dtype=self.wan.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            if sample_solver == 'unipc':
                scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.wan.num_train_timesteps,
                    shift=1, use_dynamic_shifting=False)
                scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
                timesteps = scheduler.timesteps
            elif sample_solver == 'dpm++':
                scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.wan.num_train_timesteps,
                    shift=1, use_dynamic_shifting=False)
                sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    scheduler, device=device, sigmas=sigmas)
            else:
                raise NotImplementedError(f"Unknown solver: {sample_solver}")

            self.wan.model.to(device)
            torch.cuda.empty_cache()

            latents = [noise]
            mask1, mask2 = masks_like(latents, zero=False)

            print(f"[Generate] 开始 t2v 去噪 ({len(timesteps)} steps)...")
            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_input = latents
                timestep = torch.stack([t])

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep_masked = temp_ts.unsqueeze(0)

                pred_cond = self.wan.model(
                    latent_input, t=timestep_masked,
                    context=aug_context, seq_len=seq_len)[0]

                pred_uncond = self.wan.model(
                    latent_input, t=timestep_masked,
                    context=aug_null, seq_len=seq_len)[0]

                if step_idx == 0:
                    self._verify_mq_influence_on_wan(
                        pred_cond=pred_cond,
                        latent_input=latent_input,
                        timestep_masked=timestep_masked,
                        seq_len=seq_len,
                        mq_feat=mq_feat,
                        t5_feat=t5_context[0],
                        tag="t2v",
                    )
                    self._verify_wan_image_influence_on_wan(
                        pred_cond=pred_cond,
                        latent_input=latent_input,
                        timestep_masked=timestep_masked,
                        seq_len=seq_len,
                        mq_feat_noimg=mq_feat_noimg,
                        t5_feat=t5_context[0],
                        tag="t2v",
                    )

                pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)

                temp_x0 = scheduler.step(
                    pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents

            if self.args.offload_model:
                self.wan.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            print("[Generate] VAE 解码...")
            videos = self.wan.vae.decode(x0)

        self.wan.model.text_len = self._orig_text_len
        self._record_runtime_metric("t2v_text_len_restored", int(self.wan.model.text_len))
        if self.wan.model.text_len != self._orig_text_len:
            raise RuntimeError(
                f"[VERIFY] t2v text_len 未恢复: current={self.wan.model.text_len}, expected={self._orig_text_len}"
            )

        del noise, latents, x0
        gc.collect()
        torch.cuda.empty_cache()

        return videos[0]


# =============================================================================
# 视频保存
# =============================================================================
def save_video(video_tensor, output_path, fps=24):
    """
    保存视频 tensor 为 mp4。

    Args:
        video_tensor: [3, T, H, W], 值域 [0, 1] 或 [-1, 1]
        output_path: 输出路径
        fps: 帧率
    """
    import cv2

    # 归一化到 [0, 255]
    video = video_tensor.cpu().float()
    if video.min() < 0:
        video = (video + 1.0) / 2.0
    video = video.clamp(0, 1) * 255
    video = video.byte()

    # [3, T, H, W] → [T, H, W, 3]
    video = video.permute(1, 2, 3, 0).numpy()

    T, H, W, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in video:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"✅ 视频已保存: {output_path} ({T} frames, {W}x{H}, {fps}fps)")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    args = parse_args()

    pipeline = MetaQueryWanPipeline(args)
    verify_report_path = args.verify_report_path.strip()
    if args.verify_level != "none" and not verify_report_path:
        verify_report_path = f"{args.output_path}.verify.json"

    # 加载参考图
    ref_image = None
    if args.ref_image and os.path.exists(args.ref_image):
        ref_image = Image.open(args.ref_image).convert("RGB")
        print(f"[Main] 参考图: {args.ref_image} ({ref_image.size})")

    try:
        # 生成
        if args.mode == "i2v":
            if ref_image is None:
                raise ValueError("i2v 模式需要 --ref_image!")
            if args.i2v_method == "animate_ref_slot":
                video = pipeline.generate_i2v_animate_ref_slot(
                    prompt=args.prompt,
                    ref_image=ref_image,
                    negative_prompt=args.negative_prompt,
                    max_area=args.max_area,
                    frame_num=args.frame_num,
                    shift=args.shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    seed=args.seed,
                )
            else:
                video = pipeline.generate_i2v(
                    prompt=args.prompt,
                    ref_image=ref_image,
                    negative_prompt=args.negative_prompt,
                    max_area=args.max_area,
                    frame_num=args.frame_num,
                    shift=args.shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    seed=args.seed,
                )
        else:
            video = pipeline.generate_t2v(
                prompt=args.prompt,
                ref_image=ref_image,
                negative_prompt=args.negative_prompt,
                size=tuple(args.size),
                frame_num=args.frame_num,
                shift=args.shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sampling_steps,
                guide_scale=args.guide_scale,
                seed=args.seed,
            )

        # 保存
        save_video(video, args.output_path)
    finally:
        pipeline.dump_verify_report(verify_report_path)



# 这个也是，上面是MQ + T5的，这个是只MQ的
"""
inference_metaquery_wan.py
==========================
MetaQuery + Wan2.2 TI2V 推理脚本。

使用训练好的 MetaQuery Connector, 结合 Wan TI2V 5B 生成视频。

★ 推理流程:
    1. MetaQuery Encoder: (参考图 + 文本描述) → MQ features [256, 4096]
    2. MQ-only: context = MQ_feat（不再显式注入 T5/ref latent）
    3. text_len 设为 num_metaqueries
    4. 去噪循环: DiT(noise, t, context) → velocity estimate → 迭代采样
    5. VAE 解码 → 视频帧

★ 生成模式:
    - t2v: MQ-only 条件生成
    - i2v: 仅用于尺寸/接口兼容，底层走 MQ-only 路径

用法:
    python inference_metaquery_wan.py \
        --checkpoint_path /path/to/checkpoint-final/mq_encoder_full.pt \
        --prompt "Tom chases Jerry across the kitchen" \
        --ref_image ./reference.png \
        --mode i2v \
        --output_path output.mp4
"""

# import os
# import sys
# import gc
# import json
# import math
# import random
# import argparse
# from pathlib import Path
# from contextlib import contextmanager
# from typing import Any, Dict

# import torch
# import torch.nn as nn
# import torchvision.transforms.functional as TF
# from PIL import Image
# from tqdm import tqdm

# # ── 路径设置 ─────────────────────────────────────────────────────────────────
# WAN_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(WAN_ROOT))
# METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
# sys.path.insert(0, METAQUERY_ROOT)


# def parse_args():
#     p = argparse.ArgumentParser(description="Inference: MetaQuery + Wan TI2V")

#     # ── 模型路径 ──────────────────────────────────────────────────────────
#     p.add_argument("--checkpoint_path", type=str, required=True,
#                    help="checkpoint 文件或目录路径（支持 mq_encoder_full.pt / checkpoint-final/）")
#     p.add_argument("--wan_checkpoint_dir", type=str,
#                    default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
#                    help="Wan2.2 TI2V checkpoint 目录")
#     p.add_argument("--qwen3vl_model_id", type=str,
#                    default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking",
#                    help="Qwen3-VL 模型 ID 或本地路径")

#     # ── 输入 ──────────────────────────────────────────────────────────────
#     p.add_argument("--prompt", type=str, required=True,
#                    help="文本描述")
#     p.add_argument("--ref_image", type=str, default=None,
#                    help="参考图路径（用于 MetaQuery 编码）")
#     p.add_argument("--negative_prompt", type=str, default="",
#                    help="负面提示词")

#     # ── 生成参数 ──────────────────────────────────────────────────────────
#     p.add_argument("--mode", type=str, default="i2v", choices=["t2v", "i2v"],
#                    help="生成模式: t2v 或 i2v")
#     p.add_argument(
#         "--i2v_method",
#         type=str,
#         default="legacy_ref_lock",
#         choices=["legacy_ref_lock", "animate_ref_slot"],
#         help=(
#             "兼容参数（MQ-only 下不做显式 ref 注入）: "
#             "legacy_ref_lock / animate_ref_slot 都会回退到 MQ-only i2v 路径"
#         ),
#     )
#     p.add_argument("--frame_num", type=int, default=81,
#                    help="生成帧数 (4n+1)")
#     p.add_argument("--size", type=int, nargs=2, default=[832, 480],
#                    help="视频尺寸 (宽 高)")
#     p.add_argument(
#         "--i2v_force_size",
#         action="store_true",
#         help="i2v 模式下强制使用 --size，不再按参考图比例+max_area自动计算",
#     )
#     p.add_argument(
#         "--i2v_ref_strategy",
#         type=str,
#         default="animate_like",
#         choices=["animate_like", "hard_lock"],
#         help=(
#             "i2v 首帧注入策略: "
#             "animate_like=前期强锚定后期释放（更接近 wan-animate/SCAIL 的非硬锁范式）；"
#             "hard_lock=每步硬锁首帧（与旧实现一致）"
#         ),
#     )
#     p.add_argument("--max_area", type=int, default=480 * 832,
#                    help="最大面积")
#     p.add_argument(
#         "--animate_refslot_segment_frames",
#         type=int,
#         default=78,
#         help="animate_ref_slot: 单个推理 segment 的像素帧数（默认 78）",
#     )
#     p.add_argument(
#         "--animate_refslot_ref_frames",
#         type=int,
#         default=1,
#         help="animate_ref_slot: 角色参考静态帧数（默认 1）",
#     )
#     p.add_argument(
#         "--animate_refslot_temporal_frames",
#         type=int,
#         default=1,
#         help="animate_ref_slot: 非首段 temporal reference 帧数（推荐 1 或 5）",
#     )
#     p.add_argument(
#         "--animate_refslot_conditional_frames",
#         type=int,
#         default=0,
#         help="animate_ref_slot: 额外 conditional 帧数；若无条件输入可设 0（会注入全零 latent）",
#     )
#     p.add_argument(
#         "--animate_refslot_preserve_reinject",
#         action="store_true",
#         default=True,
#         help="animate_ref_slot: 每步采样后重注入 preserved prefix（默认开启）",
#     )
#     p.add_argument(
#         "--animate_refslot_no_preserve_reinject",
#         action="store_false",
#         dest="animate_refslot_preserve_reinject",
#         help="animate_ref_slot: 关闭每步 preserved prefix 重注入",
#     )
#     p.add_argument("--sampling_steps", type=int, default=50)
#     p.add_argument("--guide_scale", type=float, default=5.0)
#     p.add_argument("--shift", type=float, default=5.0)
#     p.add_argument("--sample_solver", type=str, default="unipc",
#                    choices=["unipc", "dpm++"])
#     p.add_argument("--seed", type=int, default=42)

#     # ── 输出 ──────────────────────────────────────────────────────────────
#     p.add_argument("--output_path", type=str, default="output_metaquery.mp4")

#     # ── MetaQuery ─────────────────────────────────────────────────────────
#     p.add_argument("--num_metaqueries", type=int, default=256)
#     p.add_argument("--connector_num_hidden_layers", type=int, default=24)
#     p.add_argument(
#         "--dit_condition_mode",
#         type=str,
#         default="mq_only",
#         choices=["mq_only"],
#         help="DiT 显式条件注入模式。当前仅支持 mq_only（仅注入 MetaQuery tokens）。",
#     )
#     p.add_argument(
#         "--verify_level",
#         type=str,
#         default="none",
#         choices=["none", "basic", "full"],
#         help="验证级别: none/basic/full",
#     )
#     p.add_argument(
#         "--verify_fail_on_warning",
#         action="store_true",
#         help="开启后，验证 warning 直接视作失败",
#     )
#     p.add_argument(
#         "--verify_report_path",
#         type=str,
#         default="",
#         help="验证报告 JSON 输出路径（留空则自动命名为 output_path.verify.json）",
#     )
#     p.add_argument(
#         "--verify_train_before_checkpoint",
#         type=str,
#         default="",
#         help="训练前基线 checkpoint（用于对比当前 checkpoint 参数是否更新）",
#     )

#     # ── 设备 ──────────────────────────────────────────────────────────────
#     p.add_argument("--device", type=int, default=0)
#     p.add_argument("--offload_model", action="store_true",
#                    help="DiT/T5 用完后 offload 到 CPU")

#     return p.parse_args()


# def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
#     try:
#         return torch.load(path, map_location=map_location, weights_only=True)
#     except TypeError:
#         return torch.load(path, map_location=map_location)


# def _write_json(path: Path, payload: Dict[str, Any]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)


# # =============================================================================
# # MetaQuery Encoder (推理模式)
# # =============================================================================
# class MetaQueryEncoderForWanInference(nn.Module):
#     """推理用 MetaQuery Encoder，加载训练好的 checkpoint"""

#     WAN_TEXT_DIM = 4096

#     def __init__(
#         self,
#         qwen3vl_model_id: str,
#         checkpoint_path: str,
#         num_metaqueries: int = 256,
#         connector_num_hidden_layers: int = 24,
#         dtype: torch.dtype = torch.bfloat16,
#         device: str = "cuda",
#         verify_level: str = "none",
#         fail_on_warning: bool = False,
#         train_before_checkpoint_path: str = "",
#     ):
#         super().__init__()
#         self.num_metaqueries = num_metaqueries
#         self.wan_text_dim = self.WAN_TEXT_DIM
#         self.dtype = dtype
#         self.device = torch.device(device)
#         self.verify_level = verify_level
#         self.fail_on_warning = fail_on_warning
#         self.verify_enabled = verify_level != "none"
#         self.verify_report: Dict[str, Any] = {
#             "verify_level": verify_level,
#             "checkpoint_path_input": checkpoint_path,
#             "resolved_checkpoint_path": "",
#             "resolved_checkpoint_dir": "",
#             "training_artifacts": {},
#             "state_dict_stats": {},
#             "connector_stats": {},
#             "checkpoint_update_vs_before": {},
#             "warnings": [],
#         }

#         print("=" * 60)
#         print("[MetaQuery Inference] 初始化")
#         print(f"  Checkpoint: {checkpoint_path}")
#         print("=" * 60)

#         # ── 使用训练脚本中定义的同一个类初始化 ─────────────────────────
#         from train_connector_for_wan import MetaQueryEncoderForWan
#         try:
#             from train_metaquery_wan_animate_like import load_mq_encoder_state
#             _mq_state_source = "train_metaquery_wan_animate_like"
#         except Exception:
#             from train_metaquery_wan import load_mq_encoder_state
#             _mq_state_source = "train_metaquery_wan"
#         self._load_mq_encoder_state_fn = load_mq_encoder_state
#         encoder = MetaQueryEncoderForWan(
#             qwen3vl_model_id=qwen3vl_model_id,
#             num_metaqueries=num_metaqueries,
#             connector_num_hidden_layers=connector_num_hidden_layers,
#             dtype=dtype,
#             device=device,
#         )
#         print(f"[MetaQuery Inference] load_mq_encoder_state 来源: {_mq_state_source}")

#         # ── 加载训练好的权重 ─────────────────────────────────────────────
#         state_dict, resolved_path = load_mq_encoder_state(
#             checkpoint_path,
#             map_location=self.device,
#         )
#         missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
#         print(f"  Resolved ckpt: {resolved_path}")
#         print(f"  Missing keys : {len(missing)}")
#         print(f"  Unexpected   : {len(unexpected)}")

#         self._verify_checkpoint_artifacts(Path(resolved_path))
#         self._verify_state_loading(state_dict, missing, unexpected)
#         self._verify_checkpoint_updated_vs_before(
#             after_state_dict=state_dict,
#             train_before_checkpoint_path=train_before_checkpoint_path,
#         )
#         self._verify_connector_weights(encoder)

#         self.encoder = encoder
#         self.encoder.eval()
#         print("[MetaQuery Inference] ✅ 加载完成")

#     def _warn(self, msg: str) -> None:
#         self.verify_report["warnings"].append(msg)
#         print(f"  [VERIFY][WARN] {msg}")
#         if self.fail_on_warning:
#             raise RuntimeError(f"[VERIFY] {msg}")

#     def _verify_checkpoint_artifacts(self, resolved_path: Path) -> None:
#         if not self.verify_enabled:
#             return
#         resolved_path = resolved_path.resolve()
#         ckpt_dir = resolved_path if resolved_path.is_dir() else resolved_path.parent
#         self.verify_report["resolved_checkpoint_path"] = str(resolved_path)
#         self.verify_report["resolved_checkpoint_dir"] = str(ckpt_dir)

#         artifacts = {
#             "config.json": (ckpt_dir / "config.json").exists(),
#             "trainer_state.json": (ckpt_dir / "trainer_state.json").exists(),
#             "optimizer.pt": (ckpt_dir / "optimizer.pt").exists(),
#             "scheduler.pt": (ckpt_dir / "scheduler.pt").exists(),
#             "training_args.bin": (ckpt_dir / "training_args.bin").exists(),
#             "training_args.json": (ckpt_dir / "training_args.json").exists(),
#             "metrics_summary.json": (ckpt_dir / "metrics_summary.json").exists(),
#             "metrics_tail.json": (ckpt_dir / "metrics_tail.json").exists(),
#             "latest": (ckpt_dir.parent / "latest").exists(),
#             "mq_encoder_trainable.pt": (ckpt_dir / "mq_encoder_trainable.pt").exists(),
#             "mq_encoder_trainable.safetensors": (ckpt_dir / "mq_encoder_trainable.safetensors").exists(),
#             "model.safetensors": (ckpt_dir / "model.safetensors").exists(),
#             "mq_encoder_full.pt": (ckpt_dir / "mq_encoder_full.pt").exists(),
#         }
#         self.verify_report["training_artifacts"] = artifacts

#         print("  [VERIFY] checkpoint 文件布局检查:")
#         for name, exists in artifacts.items():
#             print(f"    - {name}: {'OK' if exists else 'MISSING'}")

#         required_any = artifacts["model.safetensors"] or artifacts["mq_encoder_full.pt"]
#         if not required_any:
#             raise RuntimeError(
#                 "[VERIFY] checkpoint 缺少 model.safetensors 或 mq_encoder_full.pt，"
#                 "无法证明是完整训练输出"
#             )
#         if not artifacts["config.json"]:
#             self._warn("config.json 缺失，无法核对 num_metaqueries/wan_text_dim 等训练配置")
#         if not artifacts["trainer_state.json"]:
#             self._warn("trainer_state.json 缺失，无法核对训练步数与格式信息")
#         if not artifacts["training_args.json"]:
#             self._warn("training_args.json 缺失，无法核对完整训练超参数")
#         if not (artifacts["mq_encoder_trainable.pt"] or artifacts["mq_encoder_trainable.safetensors"]):
#             self._warn("mq_encoder_trainable.* 缺失，无法复核 trainable 子模块产物")

#         trainer_state_path = ckpt_dir / "trainer_state.json"
#         if trainer_state_path.exists():
#             try:
#                 trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
#                 global_step = int(trainer_state.get("global_step", 0))
#                 ckpt_format = str(trainer_state.get("checkpoint_format", ""))
#                 self.verify_report["training_artifacts"]["trainer_global_step"] = global_step
#                 self.verify_report["training_artifacts"]["trainer_checkpoint_format"] = ckpt_format
#                 if global_step <= 0:
#                     self._warn(f"trainer_state.global_step={global_step}，看起来不像已训练完成的 checkpoint")
#             except Exception as e:
#                 self._warn(f"读取 trainer_state.json 失败: {e}")

#         config_path = ckpt_dir / "config.json"
#         if config_path.exists():
#             try:
#                 cfg = json.loads(config_path.read_text(encoding="utf-8"))
#                 cfg_num_mq = int(cfg.get("num_metaqueries", self.num_metaqueries))
#                 cfg_wan_text_dim = int(cfg.get("wan_text_dim", self.wan_text_dim))
#                 self.verify_report["training_artifacts"]["config_num_metaqueries"] = cfg_num_mq
#                 self.verify_report["training_artifacts"]["config_wan_text_dim"] = cfg_wan_text_dim
#                 if cfg_num_mq != self.num_metaqueries:
#                     self._warn(
#                         f"config.num_metaqueries={cfg_num_mq} 与推理参数 --num_metaqueries={self.num_metaqueries} 不一致"
#                     )
#                 if cfg_wan_text_dim != self.wan_text_dim:
#                     raise RuntimeError(
#                         f"[VERIFY] config.wan_text_dim={cfg_wan_text_dim} 与 Wan 期望 {self.wan_text_dim} 不一致"
#                     )
#             except Exception as e:
#                 self._warn(f"读取 config.json 失败: {e}")

#         training_args_json_path = ckpt_dir / "training_args.json"
#         if training_args_json_path.exists():
#             try:
#                 targs = json.loads(training_args_json_path.read_text(encoding="utf-8"))
#                 keys = [
#                     "num_train_steps",
#                     "num_metaqueries",
#                     "frame_num",
#                     "max_area",
#                     "learning_rate",
#                     "warmup_steps",
#                     "gradient_accumulation_steps",
#                     "null_caption_prob",
#                     "null_image_prob",
#                 ]
#                 self.verify_report["training_artifacts"]["training_args_excerpt"] = {
#                     k: targs.get(k, None) for k in keys
#                 }
#                 if "num_metaqueries" in targs:
#                     cfg_mq = int(targs.get("num_metaqueries"))
#                     if cfg_mq != self.num_metaqueries:
#                         self._warn(
#                             f"training_args.num_metaqueries={cfg_mq} 与推理参数 --num_metaqueries={self.num_metaqueries} 不一致"
#                         )
#             except Exception as e:
#                 self._warn(f"读取 training_args.json 失败: {e}")

#         metrics_summary_path = ckpt_dir / "metrics_summary.json"
#         if metrics_summary_path.exists():
#             try:
#                 ms = json.loads(metrics_summary_path.read_text(encoding="utf-8"))
#                 self.verify_report["training_artifacts"]["metrics_summary"] = ms
#                 if int(ms.get("logged_steps", 0)) <= 0:
#                     self._warn("metrics_summary.logged_steps<=0，训练指标记录可能不完整")
#             except Exception as e:
#                 self._warn(f"读取 metrics_summary.json 失败: {e}")

#         optimizer_path = ckpt_dir / "optimizer.pt"
#         if optimizer_path.exists() and self.verify_level == "full":
#             try:
#                 opt_state = _safe_torch_load(optimizer_path, map_location="cpu")
#                 state_len = len(opt_state.get("state", {})) if isinstance(opt_state, dict) else 0
#                 self.verify_report["training_artifacts"]["optimizer_state_entries"] = state_len
#                 if state_len == 0:
#                     self._warn("optimizer.pt 存在但 state 为空")
#             except Exception as e:
#                 self._warn(f"读取 optimizer.pt 失败: {e}")

#     def _verify_state_loading(self, state_dict, missing, unexpected) -> None:
#         if not self.verify_enabled:
#             return

#         state_key_count = len(state_dict)
#         connector_keys = [k for k in state_dict.keys() if "connector" in k]
#         embed_keys = [k for k in state_dict.keys() if "embed_tokens" in k and "weight" in k]
#         self.verify_report["state_dict_stats"] = {
#             "state_key_count": state_key_count,
#             "connector_key_count": len(connector_keys),
#             "embed_key_count": len(embed_keys),
#             "missing_count": len(missing),
#             "unexpected_count": len(unexpected),
#         }
#         print(
#             f"  [VERIFY] state_dict keys={state_key_count}, connector_keys={len(connector_keys)}, "
#             f"embed_keys={len(embed_keys)}"
#         )

#         if len(connector_keys) == 0:
#             raise RuntimeError("[VERIFY] state_dict 中未发现 connector 相关权重，checkpoint 可能错误")

#         critical_missing = [
#             k for k in missing
#             if "connector" in k or "embed_tokens" in k
#         ]
#         if critical_missing:
#             raise RuntimeError(
#                 f"[VERIFY] load_state_dict 缺失关键参数: {critical_missing[:8]}"
#             )
#         if unexpected:
#             self._warn(f"load_state_dict 存在 unexpected keys (前8个): {unexpected[:8]}")
#         if missing:
#             self._warn(f"load_state_dict 存在 missing keys (前8个): {missing[:8]}")

#     def _verify_connector_weights(self, encoder) -> None:
#         if not self.verify_enabled:
#             return
#         connector = encoder.mllm_model.connector
#         total_params = 0
#         nonzero_params = 0
#         finite_ok = True
#         l2_sum = 0.0
#         for p in connector.parameters():
#             total_params += p.numel()
#             nonzero_params += int((p.detach().abs() > 0).sum().item())
#             finite_ok = finite_ok and bool(torch.isfinite(p).all())
#             l2_sum += float(p.detach().float().norm().item())

#         self.verify_report["connector_stats"] = {
#             "total_params": int(total_params),
#             "nonzero_params": int(nonzero_params),
#             "nonzero_ratio": float(nonzero_params / max(total_params, 1)),
#             "finite_ok": bool(finite_ok),
#             "l2_sum": float(l2_sum),
#         }
#         print(
#             f"  [VERIFY] connector params={total_params:,}, nonzero_ratio={nonzero_params / max(total_params, 1):.6f}, "
#             f"finite={finite_ok}, l2_sum={l2_sum:.4f}"
#         )
#         if total_params == 0:
#             raise RuntimeError("[VERIFY] connector 参数量为 0")
#         if not finite_ok:
#             raise RuntimeError("[VERIFY] connector 参数存在 NaN/Inf")
#         if l2_sum <= 0:
#             raise RuntimeError("[VERIFY] connector 参数范数为 0，疑似未正常训练/加载")

#     def _accumulate_delta_stats(self, before: torch.Tensor, after: torch.Tensor, eps: float = 1e-7) -> Dict[str, float]:
#         b = before.detach().to("cpu").reshape(-1)
#         a = after.detach().to("cpu").reshape(-1)
#         n = b.numel()
#         chunk = 1_000_000
#         changed = 0
#         max_abs = 0.0
#         sum_abs = 0.0
#         for i in range(0, n, chunk):
#             bb = b[i:i + chunk].to(torch.float32)
#             aa = a[i:i + chunk].to(torch.float32)
#             d = (aa - bb).abs()
#             changed += int((d > eps).sum().item())
#             local_max = float(d.max().item()) if d.numel() > 0 else 0.0
#             if local_max > max_abs:
#                 max_abs = local_max
#             sum_abs += float(d.sum().item())
#         mean_abs = sum_abs / max(n, 1)
#         return {
#             "numel": int(n),
#             "changed_elems": int(changed),
#             "changed_ratio": float(changed / max(n, 1)),
#             "max_abs_delta": float(max_abs),
#             "mean_abs_delta": float(mean_abs),
#         }

#     def _verify_checkpoint_updated_vs_before(
#         self,
#         after_state_dict: Dict[str, torch.Tensor],
#         train_before_checkpoint_path: str,
#     ) -> None:
#         if not self.verify_enabled:
#             return
#         if not train_before_checkpoint_path:
#             return

#         try:
#             before_state_dict, before_resolved = self._load_mq_encoder_state_fn(
#                 train_before_checkpoint_path,
#                 map_location="cpu",
#             )
#         except Exception as e:
#             self._warn(f"读取训练前 checkpoint 失败: {e}")
#             return

#         shared_keys = [k for k in after_state_dict.keys() if k in before_state_dict]
#         if len(shared_keys) == 0:
#             raise RuntimeError("[VERIFY] before/after checkpoint 没有共享参数键，无法比较训练更新")

#         connector_keys = [k for k in shared_keys if "mllm_model.connector" in k]
#         embed_keys = [k for k in shared_keys if "embed_tokens.weight" in k]

#         def summarize(keys: list[str]) -> Dict[str, Any]:
#             out = {
#                 "tensor_count": 0,
#                 "numel": 0,
#                 "changed_elems": 0,
#                 "changed_ratio": 0.0,
#                 "max_abs_delta": 0.0,
#                 "mean_abs_delta_weighted": 0.0,
#             }
#             weighted_mean_sum = 0.0
#             valid_tensors = 0
#             for key in keys:
#                 b = before_state_dict[key]
#                 a = after_state_dict[key]
#                 if b.shape != a.shape:
#                     self._warn(f"before/after shape 不一致: {key} {tuple(b.shape)} vs {tuple(a.shape)}")
#                     continue
#                 stats = self._accumulate_delta_stats(b, a)
#                 out["numel"] += stats["numel"]
#                 out["changed_elems"] += stats["changed_elems"]
#                 if stats["max_abs_delta"] > out["max_abs_delta"]:
#                     out["max_abs_delta"] = stats["max_abs_delta"]
#                 weighted_mean_sum += stats["mean_abs_delta"] * stats["numel"]
#                 valid_tensors += 1

#             out["tensor_count"] = valid_tensors
#             out["changed_ratio"] = float(out["changed_elems"] / max(out["numel"], 1))
#             out["mean_abs_delta_weighted"] = float(weighted_mean_sum / max(out["numel"], 1))
#             return out

#         summary_all = summarize(shared_keys)
#         summary_connector = summarize(connector_keys)
#         summary_embed = summarize(embed_keys)

#         self.verify_report["checkpoint_update_vs_before"] = {
#             "before_checkpoint_path_input": train_before_checkpoint_path,
#             "before_checkpoint_resolved": str(before_resolved),
#             "shared_tensor_count": len(shared_keys),
#             "all": summary_all,
#             "connector": summary_connector,
#             "embed_tokens": summary_embed,
#         }

#         print("[VERIFY] checkpoint 前后参数对比:")
#         print(
#             f"  all      : tensors={summary_all['tensor_count']} changed_ratio={summary_all['changed_ratio']:.6e} "
#             f"max_abs={summary_all['max_abs_delta']:.6e}"
#         )
#         print(
#             f"  connector: tensors={summary_connector['tensor_count']} changed_ratio={summary_connector['changed_ratio']:.6e} "
#             f"max_abs={summary_connector['max_abs_delta']:.6e}"
#         )
#         print(
#             f"  embed    : tensors={summary_embed['tensor_count']} changed_ratio={summary_embed['changed_ratio']:.6e} "
#             f"max_abs={summary_embed['max_abs_delta']:.6e}"
#         )

#         if summary_connector["tensor_count"] == 0:
#             raise RuntimeError("[VERIFY] 无法在 checkpoint 中定位 connector 参数，无法证明训练更新")
#         if summary_connector["changed_elems"] == 0:
#             raise RuntimeError(
#                 "[VERIFY] 对比训练前后 checkpoint，connector 参数未发生变化，训练链路可能异常"
#             )
#         if summary_all["changed_elems"] == 0:
#             raise RuntimeError("[VERIFY] 训练前后 checkpoint 完全一致，训练未生效或路径配置错误")

#     @torch.no_grad()
#     def encode(self, caption, ref_image=None):
#         """
#         编码 (文本 + 参考图) → MQ features

#         Args:
#             caption: str
#             ref_image: PIL Image or None

#         Returns:
#             Tensor [1, 256, 4096]
#         """
#         captions = [caption]
#         images = [[ref_image]] if ref_image is not None else None
#         mq_feat = self.encoder(captions, images)
#         return mq_feat  # [1, 256, 4096]

#     def to(self, *args, **kwargs):
#         self.encoder = self.encoder.to(*args, **kwargs)
#         return self


# # =============================================================================
# # MetaQuery + Wan TI2V 推理管线
# # =============================================================================
# class MetaQueryWanPipeline:
#     """
#     使用 MetaQuery 增强的 Wan TI2V 推理管线。

#     核心: 仅将 MQ features 作为 DiT context（MQ-only），
#     text_len 仅按 MQ tokens 设置。
#     """

#     def __init__(self, args):
#         self.args = args
#         if str(getattr(args, "dit_condition_mode", "mq_only")).strip().lower() != "mq_only":
#             raise ValueError("当前仅支持 --dit_condition_mode mq_only")
#         self.device = torch.device(f"cuda:{args.device}")
#         self.verify_level = getattr(args, "verify_level", "none")
#         self.verify_fail_on_warning = bool(getattr(args, "verify_fail_on_warning", False))
#         self.verify_enabled = self.verify_level != "none"
#         self.verify_report: Dict[str, Any] = {
#             "verify_level": self.verify_level,
#             "mode": getattr(args, "mode", "unknown"),
#             "checkpoint": {},
#             "runtime": {},
#             "warnings": [],
#         }

#         self._load_pipeline()
#         self._load_mq_encoder()

#     def _load_pipeline(self):
#         """加载 Wan TI2V Pipeline"""
#         from wan import WanTI2V
#         from wan.configs import WAN_CONFIGS

#         config = WAN_CONFIGS['ti2v-5B']
#         self.wan = WanTI2V(
#             config=config,
#             checkpoint_dir=self.args.wan_checkpoint_dir,
#             device_id=self.args.device,
#             rank=0,
#             t5_cpu=False,
#             init_on_cpu=True,
#         )
#         self.wan_config = config
#         self._orig_text_len = self.wan.model.text_len  # 512
#         self._aug_text_len = int(self.args.num_metaqueries)
#         print(f"[Pipeline] Wan TI2V 已加载, text_len={self._orig_text_len}")

#     def _load_mq_encoder(self):
#         """加载 MetaQuery Encoder"""
#         self.mq_encoder = MetaQueryEncoderForWanInference(
#             qwen3vl_model_id=self.args.qwen3vl_model_id,
#             checkpoint_path=self.args.checkpoint_path,
#             num_metaqueries=self.args.num_metaqueries,
#             connector_num_hidden_layers=self.args.connector_num_hidden_layers,
#             dtype=torch.bfloat16,
#             device=f"cuda:{self.args.device}",
#             verify_level=self.verify_level,
#             fail_on_warning=self.verify_fail_on_warning,
#             train_before_checkpoint_path=getattr(self.args, "verify_train_before_checkpoint", ""),
#         )
#         self.verify_report["checkpoint"] = dict(self.mq_encoder.verify_report)

#     def _warn(self, msg: str) -> None:
#         self.verify_report["warnings"].append(msg)
#         print(f"[VERIFY][WARN] {msg}")
#         if self.verify_fail_on_warning:
#             raise RuntimeError(f"[VERIFY] {msg}")

#     def _record_runtime_metric(self, key: str, value: Any) -> None:
#         self.verify_report["runtime"][key] = value

#     @staticmethod
#     def _compute_i2v_ref_blend_alpha(
#         strategy: str,
#         step_idx: int,
#         total_steps: int,
#     ) -> float:
#         if strategy == "hard_lock":
#             return 1.0
#         # animate_like:
#         # 在前 35% 步数内从 0.95 余弦衰减到 0，后续不再重注入。
#         # 这比 hard_lock 更接近“强参考条件但不锁死 latent”。
#         if strategy != "animate_like":
#             return 1.0
#         if total_steps <= 1:
#             return 0.95
#         warmup_steps = max(1, int(round(total_steps * 0.35)))
#         if step_idx >= warmup_steps:
#             return 0.0
#         if warmup_steps == 1:
#             return 0.95
#         p = float(step_idx) / float(warmup_steps - 1)
#         alpha = 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))
#         return float(max(0.0, min(1.0, alpha)))

#     @staticmethod
#     def _frames_to_latent_slots(frame_count: int, stride_t: int) -> int:
#         f = max(0, int(frame_count))
#         if f <= 0:
#             return 0
#         return int((f - 1) // max(int(stride_t), 1) + 1)

#     def _verify_context_plugged(
#         self,
#         mq_feat: torch.Tensor,
#         aug_feat: torch.Tensor,
#         tag: str,
#     ) -> None:
#         if not self.verify_enabled:
#             return
#         mq_len = mq_feat.shape[0]
#         if aug_feat.shape[0] != mq_len:
#             raise RuntimeError(
#                 f"[VERIFY] {tag} MQ-only context 长度异常: aug={aug_feat.shape[0]}, mq={mq_len}"
#             )
#         mq_ok = torch.allclose(
#             aug_feat.float(),
#             mq_feat.float(),
#             atol=1e-3,
#             rtol=1e-3,
#         )
#         self._record_runtime_metric(f"{tag}_mq_tokens", int(mq_len))
#         self._record_runtime_metric(f"{tag}_t5_tokens", 0)
#         self._record_runtime_metric(f"{tag}_aug_tokens", int(aug_feat.shape[0]))
#         if not mq_ok:
#             raise RuntimeError(f"[VERIFY] {tag} MQ-only context 未正确注入")

#     def _verify_mq_feature_sensitivity(
#         self,
#         prompt: str,
#         negative_prompt: str,
#         ref_image: Image.Image | None,
#         mq_feat: torch.Tensor,
#         mq_feat_null: torch.Tensor,
#         mq_feat_noimg: torch.Tensor | None,
#         tag: str,
#     ) -> None:
#         if not self.verify_enabled:
#             return
#         cond_norm = float(mq_feat.float().norm().item())
#         uncond_norm = float(mq_feat_null.float().norm().item())
#         diff_norm = float((mq_feat - mq_feat_null).float().norm().item())
#         self._record_runtime_metric(f"{tag}_mq_cond_norm", cond_norm)
#         self._record_runtime_metric(f"{tag}_mq_uncond_norm", uncond_norm)
#         self._record_runtime_metric(f"{tag}_mq_cond_uncond_diff_norm", diff_norm)
#         self._record_runtime_metric(f"{tag}_mq_ref_image_provided", int(ref_image is not None))
#         if cond_norm <= 0 or uncond_norm <= 0:
#             raise RuntimeError("[VERIFY] MQ 特征范数为 0，编码器可能未正常工作")
#         if diff_norm <= 1e-6:
#             self._warn("MQ 条件/无条件特征几乎无差异，CFG 的 MQ 分支可能无效")

#         image_diff = None
#         if ref_image is not None:
#             if mq_feat_noimg is None:
#                 mq_feat_noimg = self.mq_encoder.encode(prompt, None)[0].to(
#                     self.device, dtype=torch.bfloat16
#                 )
#             image_diff = float((mq_feat - mq_feat_noimg).float().norm().item())
#             image_ratio = image_diff / (cond_norm + 1e-8)
#             cond_vec = mq_feat.float().reshape(-1)
#             noimg_vec = mq_feat_noimg.float().reshape(-1)
#             cosine = float(
#                 torch.nn.functional.cosine_similarity(
#                     cond_vec.unsqueeze(0), noimg_vec.unsqueeze(0), dim=1
#                 ).item()
#             )
#             self._record_runtime_metric(f"{tag}_mq_image_sensitivity_diff_norm", image_diff)
#             self._record_runtime_metric(f"{tag}_mq_image_sensitivity_ratio", image_ratio)
#             self._record_runtime_metric(f"{tag}_mq_image_sensitivity_cosine", cosine)
#             if image_diff <= 1e-6:
#                 self._warn("MQ 对参考图不敏感")
#         else:
#             self._record_runtime_metric(f"{tag}_mq_image_sensitivity_diff_norm", 0.0)
#             self._record_runtime_metric(f"{tag}_mq_image_sensitivity_ratio", 0.0)
#             self._record_runtime_metric(f"{tag}_mq_image_sensitivity_cosine", 1.0)

#         if self.verify_level == "full":
#             alt_prompt = prompt + " [mq_probe_variant]"
#             mq_feat_alt = self.mq_encoder.encode(alt_prompt, ref_image)[0].to(
#                 self.device, dtype=torch.bfloat16
#             )
#             prompt_diff = float((mq_feat - mq_feat_alt).float().norm().item())
#             self._record_runtime_metric(f"{tag}_mq_prompt_sensitivity_diff_norm", prompt_diff)
#             if prompt_diff <= 1e-6:
#                 self._warn("MQ 对 prompt 的敏感性过低（full 验证）")

#             if image_diff is not None:
#                 if image_diff <= 1e-6:
#                     self._warn("MQ 对参考图不敏感（full 验证）")

#     def _verify_wan_image_influence_on_wan(
#         self,
#         pred_cond: torch.Tensor,
#         latent_input: list[torch.Tensor],
#         timestep_masked: torch.Tensor,
#         seq_len: int,
#         mq_feat_noimg: torch.Tensor | None,
#         tag: str,
#     ) -> None:
#         if not self.verify_enabled:
#             return

#         if mq_feat_noimg is None:
#             self._record_runtime_metric(f"{tag}_wan_image_influence_ref_available", 0)
#             self._record_runtime_metric(f"{tag}_wan_image_influence_diff_norm", 0.0)
#             self._record_runtime_metric(f"{tag}_wan_image_influence_ratio", 0.0)
#             self._record_runtime_metric(f"{tag}_wan_image_influence_cosine", 1.0)
#             return

#         noimg_context = [mq_feat_noimg]
#         pred_noimg = self.wan.model(
#             latent_input,
#             t=timestep_masked,
#             context=noimg_context,
#             seq_len=seq_len,
#         )[0]
#         diff_norm = float((pred_cond - pred_noimg).float().norm().item())
#         cond_norm = float(pred_cond.float().norm().item())
#         ratio = diff_norm / (cond_norm + 1e-8)
#         cond_vec = pred_cond.float().reshape(-1)
#         noimg_vec = pred_noimg.float().reshape(-1)
#         cosine = float(
#             torch.nn.functional.cosine_similarity(
#                 cond_vec.unsqueeze(0), noimg_vec.unsqueeze(0), dim=1
#             ).item()
#         )
#         self._record_runtime_metric(f"{tag}_wan_image_influence_ref_available", 1)
#         self._record_runtime_metric(f"{tag}_wan_image_influence_diff_norm", diff_norm)
#         self._record_runtime_metric(f"{tag}_wan_image_influence_ratio", ratio)
#         self._record_runtime_metric(f"{tag}_wan_image_influence_cosine", cosine)
#         print(
#             f"[VERIFY] {tag} 首步图像分支对照: "
#             f"||pred(mq_ref)-pred(mq_noimg)||={diff_norm:.6f}, ratio={ratio:.6e}, cosine={cosine:.6f}"
#         )
#         if diff_norm <= 1e-6:
#             self._warn("Wan 侧图像分支影响几乎为 0（mq_ref 与 mq_noimg 预测近乎一致）")

#     def _verify_mq_influence_on_wan(
#         self,
#         pred_cond: torch.Tensor,
#         latent_input: list[torch.Tensor],
#         timestep_masked: torch.Tensor,
#         seq_len: int,
#         mq_feat: torch.Tensor,
#         tag: str,
#     ) -> None:
#         if not self.verify_enabled:
#             return
#         zero_mq_context = [torch.zeros_like(mq_feat)]
#         pred_t5_only = self.wan.model(
#             latent_input,
#             t=timestep_masked,
#             context=zero_mq_context,
#             seq_len=seq_len,
#         )[0]
#         diff_norm = float((pred_cond - pred_t5_only).float().norm().item())
#         cond_norm = float(pred_cond.float().norm().item())
#         ratio = diff_norm / (cond_norm + 1e-8)
#         self._record_runtime_metric(f"{tag}_mq_influence_diff_norm", diff_norm)
#         self._record_runtime_metric(f"{tag}_mq_influence_ratio", ratio)
#         print(
#             f"[VERIFY] {tag} 首步对照: ||pred(mq)-pred(zero_mq)||={diff_norm:.6f}, "
#             f"ratio={ratio:.6e}"
#         )
#         if diff_norm <= 1e-6:
#             raise RuntimeError(
#                 "[VERIFY] MQ 置零前后 Wan 预测几乎无差异，说明 checkpoint 可能未真实参与去噪"
#             )

#     def dump_verify_report(self, report_path: str) -> None:
#         if not self.verify_enabled:
#             return
#         if not report_path:
#             return
#         warnings = self.verify_report.get("warnings", [])
#         self.verify_report["summary"] = {
#             "status": "pass_with_warnings" if warnings else "pass",
#             "warning_count": len(warnings),
#         }
#         _write_json(Path(report_path), self.verify_report)
#         print(f"[VERIFY] 报告已写入: {report_path}")

#     def generate_i2v(
#         self,
#         prompt: str,
#         ref_image: Image.Image,
#         negative_prompt: str = "",
#         max_area: int = 480 * 832,
#         frame_num: int = 81,
#         shift: float = 5.0,
#         sample_solver: str = "unipc",
#         sampling_steps: int = 50,
#         guide_scale: float = 5.0,
#         seed: int = 42,
#     ):
#         """
#         MQ-only i2v：仅将参考图/文本编码到 MetaQuery，
#         不再把 T5 或参考图 latent 作为 DiT 显式条件注入。
#         """
#         from wan.utils.utils import best_output_size

#         ih, iw = ref_image.height, ref_image.width
#         dh = self.wan_config.patch_size[1] * self.wan_config.vae_stride[1]
#         dw = self.wan_config.patch_size[2] * self.wan_config.vae_stride[2]
#         if self.args.i2v_force_size:
#             req_w, req_h = int(self.args.size[0]), int(self.args.size[1])
#             if req_w < dw or req_h < dh:
#                 raise ValueError(
#                     f"i2v_force_size 要求的尺寸过小: {req_w}x{req_h}, 最小应 >= {dw}x{dh}"
#                 )
#             # 保证满足 patch/vae 对齐（32 对齐）
#             ow = (req_w // dw) * dw
#             oh = (req_h // dh) * dh
#             if ow != req_w or oh != req_h:
#                 print(
#                     f"[Generate][i2v] i2v_force_size 对齐修正: {req_w}x{req_h} -> {ow}x{oh} "
#                     f"(align {dw}x{dh})"
#                 )
#         else:
#             ow, oh = best_output_size(iw, ih, dw, dh, max_area)
#         self._record_runtime_metric("i2v_mode_effective", "mq_only_t2v_path")
#         self._record_runtime_metric("i2v_text_explicit_in_dit", 0)
#         self._record_runtime_metric("i2v_ref_explicit_in_dit", 0)
#         print(
#             f"[Generate][i2v][MQ-only] output_size={ow}x{oh} "
#             f"(input_ref={iw}x{ih}, force_size={self.args.i2v_force_size}, max_area={max_area})"
#         )

#         return self.generate_t2v(
#             prompt=prompt,
#             ref_image=ref_image,
#             negative_prompt=negative_prompt,
#             size=(ow, oh),
#             frame_num=frame_num,
#             shift=shift,
#             sample_solver=sample_solver,
#             sampling_steps=sampling_steps,
#             guide_scale=guide_scale,
#             seed=seed,
#         )

#     def generate_i2v_animate_ref_slot(
#         self,
#         prompt: str,
#         ref_image: Image.Image,
#         negative_prompt: str = "",
#         max_area: int = 480 * 832,
#         frame_num: int = 81,
#         shift: float = 5.0,
#         sample_solver: str = "unipc",
#         sampling_steps: int = 50,
#         guide_scale: float = 5.0,
#         seed: int = 42,
#     ):
#         """
#         兼容入口：MQ-only 模式下不再执行 reference-slot 显式注入，
#         统一回退到 generate_i2v（MQ-only）路径。
#         """
#         # MQ-only 模式下，不再使用 reference-slot 显式注入；统一回退到 MQ-only i2v 路径。
#         self._record_runtime_metric("i2v_method", "animate_ref_slot")
#         self._record_runtime_metric("i2v_animate_refslot_disabled_reason", "mq_only_no_explicit_ref_in_dit")
#         return self.generate_i2v(
#             prompt=prompt,
#             ref_image=ref_image,
#             negative_prompt=negative_prompt,
#             max_area=max_area,
#             frame_num=frame_num,
#             shift=shift,
#             sample_solver=sample_solver,
#             sampling_steps=sampling_steps,
#             guide_scale=guide_scale,
#             seed=seed,
#         )

#         from wan.utils.fm_solvers import (
#             FlowDPMSolverMultistepScheduler,
#             get_sampling_sigmas,
#             retrieve_timesteps,
#         )
#         from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
#         from wan.utils.utils import best_output_size

#         device = self.device
#         self._record_runtime_metric("i2v_method", "animate_ref_slot")

#         if not negative_prompt:
#             negative_prompt = self.wan.sample_neg_prompt

#         # ── 1. MetaQuery 编码 ────────────────────────────────────────────
#         print("[Generate][animate_ref_slot] MetaQuery 编码...")
#         mq_feat = self.mq_encoder.encode(prompt, ref_image)[0].to(device, dtype=torch.bfloat16)
#         mq_feat_noimg = self.mq_encoder.encode(prompt, None)[0].to(device, dtype=torch.bfloat16)
#         null_ref_image = Image.new("RGB", ref_image.size)
#         mq_feat_null = self.mq_encoder.encode(negative_prompt, null_ref_image)[0].to(
#             device, dtype=torch.bfloat16
#         )
#         self._verify_mq_feature_sensitivity(
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             ref_image=ref_image,
#             mq_feat=mq_feat,
#             mq_feat_null=mq_feat_null,
#             mq_feat_noimg=mq_feat_noimg,
#             tag="i2v_animate_ref_slot",
#         )

#         # ── 2. T5 编码 ──────────────────────────────────────────────────
#         print("[Generate][animate_ref_slot] T5 编码...")
#         self.wan.text_encoder.model.to(device)
#         t5_context = self.wan.text_encoder([prompt], device)
#         t5_null = self.wan.text_encoder([negative_prompt], device)
#         if self.args.offload_model:
#             self.wan.text_encoder.model.cpu()
#             torch.cuda.empty_cache()

#         # ── 3. 拼接 context = [MQ + T5] ─────────────────────────────────
#         aug_context = [torch.cat([mq_feat, t5_context[0]], dim=0)]
#         aug_null = [torch.cat([mq_feat_null, t5_null[0]], dim=0)]
#         self._verify_context_plugged(mq_feat, aug_context[0], tag="i2v_animate_ref_slot")

#         # ── 4. 图像预处理 + reference latent ─────────────────────────────
#         ih, iw = ref_image.height, ref_image.width
#         patch_size = self.wan_config.patch_size
#         vae_stride = self.wan_config.vae_stride
#         dh = patch_size[1] * vae_stride[1]
#         dw = patch_size[2] * vae_stride[2]
#         if self.args.i2v_force_size:
#             req_w, req_h = int(self.args.size[0]), int(self.args.size[1])
#             if req_w < dw or req_h < dh:
#                 raise ValueError(
#                     f"i2v_force_size 要求的尺寸过小: {req_w}x{req_h}, 最小应 >= {dw}x{dh}"
#                 )
#             ow = (req_w // dw) * dw
#             oh = (req_h // dh) * dh
#         else:
#             ow, oh = best_output_size(iw, ih, dw, dh, max_area)

#         scale = max(ow / iw, oh / ih)
#         img_resized = ref_image.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
#         x1 = (img_resized.width - ow) // 2
#         y1 = (img_resized.height - oh) // 2
#         img_cropped = img_resized.crop((x1, y1, x1 + ow, y1 + oh))
#         img_tensor = TF.to_tensor(img_cropped).sub_(0.5).div_(0.5).to(device).unsqueeze(1)

#         ref_lat = self.wan.vae.encode([img_tensor])[0].float()  # [C, 1, H', W']
#         C, _, latent_H, latent_W = ref_lat.shape

#         # ── 5. slot 配置（像素帧数 -> latent slots）───────────────────────
#         stride_t = int(vae_stride[0])
#         target_lat_T = int((frame_num - 1) // stride_t + 1)
#         segment_frames = int(getattr(self.args, "animate_refslot_segment_frames", 78))
#         segment_lat_T = self._frames_to_latent_slots(segment_frames, stride_t)
#         ref_slots_lat = max(1, self._frames_to_latent_slots(
#             int(getattr(self.args, "animate_refslot_ref_frames", 1)), stride_t
#         ))
#         temporal_slots_lat_cfg = self._frames_to_latent_slots(
#             int(getattr(self.args, "animate_refslot_temporal_frames", 1)), stride_t
#         )
#         cond_slots_lat = self._frames_to_latent_slots(
#             int(getattr(self.args, "animate_refslot_conditional_frames", 0)), stride_t
#         )
#         reinject_preserved = bool(getattr(self.args, "animate_refslot_preserve_reinject", True))

#         if segment_lat_T <= ref_slots_lat + cond_slots_lat:
#             raise ValueError(
#                 f"animate_ref_slot 参数非法: segment_lat_T={segment_lat_T}, "
#                 f"ref_slots={ref_slots_lat}, cond_slots={cond_slots_lat}"
#             )

#         self._record_runtime_metric("i2v_animate_refslot_segment_frames", int(segment_frames))
#         self._record_runtime_metric("i2v_animate_refslot_segment_latent_slots", int(segment_lat_T))
#         self._record_runtime_metric("i2v_animate_refslot_ref_slots_latent", int(ref_slots_lat))
#         self._record_runtime_metric("i2v_animate_refslot_temporal_slots_latent_cfg", int(temporal_slots_lat_cfg))
#         self._record_runtime_metric("i2v_animate_refslot_conditional_slots_latent", int(cond_slots_lat))
#         self._record_runtime_metric("i2v_animate_refslot_reinject_preserved", int(reinject_preserved))
#         print(
#             f"[Generate][animate_ref_slot] target_lat={target_lat_T}, segment_lat={segment_lat_T}, "
#             f"ref_slots={ref_slots_lat}, temporal_slots_cfg={temporal_slots_lat_cfg}, cond_slots={cond_slots_lat}"
#         )

#         # ── 6. 扩展 text_len + 去噪 ──────────────────────────────────────
#         aug_text_len = self._orig_text_len + num_mq
#         self.wan.model.text_len = aug_text_len
#         self._record_runtime_metric("i2v_animate_refslot_text_len_before", int(self._orig_text_len))
#         self._record_runtime_metric("i2v_animate_refslot_text_len_after", int(self.wan.model.text_len))

#         seed_g = torch.Generator(device=device)
#         seed_g.manual_seed(seed if seed >= 0 else random.randint(0, sys.maxsize))

#         @contextmanager
#         def noop_no_sync():
#             yield

#         no_sync = getattr(self.wan.model, 'no_sync', noop_no_sync)

#         try:
#             with (
#                 torch.amp.autocast('cuda', dtype=self.wan.param_dtype),
#                 torch.no_grad(),
#                 no_sync(),
#             ):
#                 if sample_solver == 'unipc':
#                     scheduler = FlowUniPCMultistepScheduler(
#                         num_train_timesteps=self.wan.num_train_timesteps,
#                         shift=1, use_dynamic_shifting=False)
#                     scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
#                     timesteps = scheduler.timesteps
#                 elif sample_solver == 'dpm++':
#                     scheduler = FlowDPMSolverMultistepScheduler(
#                         num_train_timesteps=self.wan.num_train_timesteps,
#                         shift=1, use_dynamic_shifting=False)
#                     sigmas = get_sampling_sigmas(sampling_steps, shift)
#                     timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas)
#                 else:
#                     raise NotImplementedError(f"Unknown solver: {sample_solver}")

#                 self.wan.model.to(device)
#                 torch.cuda.empty_cache()

#                 target_latents_collected = []
#                 remaining_lat = int(target_lat_T)
#                 prev_temporal_lat = None
#                 seg_idx = 0
#                 did_first_verify = False

#                 while remaining_lat > 0:
#                     temporal_slots_use = int(temporal_slots_lat_cfg if seg_idx > 0 else 0)
#                     ref_prefix = ref_lat.repeat(1, ref_slots_lat, 1, 1)

#                     if temporal_slots_use > 0:
#                         if prev_temporal_lat is None or prev_temporal_lat.numel() == 0:
#                             temporal_prefix = torch.zeros(
#                                 C, temporal_slots_use, latent_H, latent_W,
#                                 device=device, dtype=torch.float32
#                             )
#                         else:
#                             temporal_prefix = prev_temporal_lat[:, -temporal_slots_use:, :, :].float()
#                             if temporal_prefix.shape[1] < temporal_slots_use:
#                                 pad_t = temporal_slots_use - temporal_prefix.shape[1]
#                                 pad = torch.zeros(
#                                     C, pad_t, latent_H, latent_W,
#                                     device=device, dtype=torch.float32
#                                 )
#                                 temporal_prefix = torch.cat([pad, temporal_prefix], dim=1)
#                     else:
#                         temporal_prefix = torch.zeros(
#                             C, 0, latent_H, latent_W, device=device, dtype=torch.float32
#                         )

#                     cond_prefix = torch.zeros(
#                         C, cond_slots_lat, latent_H, latent_W,
#                         device=device, dtype=torch.float32
#                     )
#                     preserved_prefix = torch.cat([ref_prefix, temporal_prefix, cond_prefix], dim=1)
#                     prefix_slots = int(preserved_prefix.shape[1])
#                     target_slots_per_seg = int(segment_lat_T - prefix_slots)
#                     if target_slots_per_seg <= 0:
#                         raise RuntimeError(
#                             f"segment_lat_T={segment_lat_T} <= prefix_slots={prefix_slots}, "
#                             "无法产生 target latent"
#                         )
#                     take_slots = min(remaining_lat, target_slots_per_seg)

#                     noise = torch.randn(
#                         C, segment_lat_T, latent_H, latent_W,
#                         dtype=torch.float32, device=device, generator=seed_g
#                     )
#                     latent = noise.clone()
#                     latent[:, :prefix_slots] = preserved_prefix

#                     tokens_per_frame = int(math.ceil((latent_H * latent_W) / (patch_size[1] * patch_size[2])))
#                     seq_len = int(tokens_per_frame * segment_lat_T)
#                     prefix_token_count = int(min(seq_len, prefix_slots * tokens_per_frame))

#                     print(
#                         f"[Generate][animate_ref_slot] seg={seg_idx} prefix_slots={prefix_slots} "
#                         f"(ref={ref_slots_lat}, temporal={temporal_slots_use}, cond={cond_slots_lat}) "
#                         f"target_slots={take_slots}/{target_slots_per_seg} remaining_lat={remaining_lat}"
#                     )

#                     for step_idx, t in enumerate(tqdm(timesteps, desc=f"seg{seg_idx} denoise", leave=False)):
#                         latent_input = [latent]
#                         timestep = torch.stack([t]).to(device)
#                         t_scalar = float(timestep.item())
#                         t_row = torch.full((seq_len,), t_scalar, device=device, dtype=torch.float32)
#                         if prefix_token_count > 0:
#                             t_row[:prefix_token_count] = 0.0
#                         timestep_masked = t_row.unsqueeze(0)

#                         pred_cond = self.wan.model(
#                             latent_input, t=timestep_masked,
#                             context=aug_context, seq_len=seq_len)[0]
#                         pred_uncond = self.wan.model(
#                             latent_input, t=timestep_masked,
#                             context=aug_null, seq_len=seq_len)[0]

#                         if not did_first_verify and step_idx == 0:
#                             self._verify_mq_influence_on_wan(
#                                 pred_cond=pred_cond,
#                                 latent_input=latent_input,
#                                 timestep_masked=timestep_masked,
#                                 seq_len=seq_len,
#                                 mq_feat=mq_feat,
#                                 tag="i2v_animate_ref_slot",
#                             )
#                             self._verify_wan_image_influence_on_wan(
#                                 pred_cond=pred_cond,
#                                 latent_input=latent_input,
#                                 timestep_masked=timestep_masked,
#                                 seq_len=seq_len,
#                                 mq_feat_noimg=mq_feat_noimg,
#                                 tag="i2v_animate_ref_slot",
#                             )
#                             did_first_verify = True

#                         pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)
#                         temp_x0 = scheduler.step(
#                             pred.unsqueeze(0), t,
#                             latent.unsqueeze(0),
#                             return_dict=False, generator=seed_g)[0]
#                         latent = temp_x0.squeeze(0)

#                         if reinject_preserved and prefix_slots > 0:
#                             latent[:, :prefix_slots] = preserved_prefix

#                     segment_target = latent[:, prefix_slots:prefix_slots + target_slots_per_seg]
#                     target_latents_collected.append(segment_target[:, :take_slots].clone())

#                     if temporal_slots_lat_cfg > 0:
#                         prev_temporal_lat = segment_target[:, -min(segment_target.shape[1], temporal_slots_lat_cfg):].detach()
#                     else:
#                         prev_temporal_lat = None

#                     remaining_lat -= int(take_slots)
#                     seg_idx += 1

#                 full_latent = torch.cat(target_latents_collected, dim=1)[:, :target_lat_T]
#                 self._record_runtime_metric("i2v_animate_refslot_segments", int(seg_idx))
#                 self._record_runtime_metric("i2v_animate_refslot_output_latent_slots", int(full_latent.shape[1]))

#                 print("[Generate][animate_ref_slot] VAE 解码...")
#                 videos = self.wan.vae.decode([full_latent])

#                 if self.args.offload_model:
#                     self.wan.model.cpu()
#                     torch.cuda.synchronize()
#                     torch.cuda.empty_cache()

#         finally:
#             self.wan.model.text_len = self._orig_text_len
#             self._record_runtime_metric("i2v_animate_refslot_text_len_restored", int(self.wan.model.text_len))

#         gc.collect()
#         torch.cuda.empty_cache()
#         return videos[0]

#     def generate_t2v(
#         self,
#         prompt: str,
#         ref_image: Image.Image = None,
#         negative_prompt: str = "",
#         size: tuple = (832, 480),
#         frame_num: int = 81,
#         shift: float = 5.0,
#         sample_solver: str = "unipc",
#         sampling_steps: int = 50,
#         guide_scale: float = 5.0,
#         seed: int = 42,
#     ):
#         """
#         MetaQuery 增强的 t2v 生成。

#         参考图仅用于 MetaQuery 编码 (角色理解)，
#         视频完全从噪声生成 (无第一帧约束)。
#         """
#         from wan.utils.fm_solvers import (
#             FlowDPMSolverMultistepScheduler,
#             get_sampling_sigmas,
#             retrieve_timesteps,
#         )
#         from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
#         from wan.utils.utils import masks_like

#         device = self.device
#         num_mq = self.args.num_metaqueries

#         if not negative_prompt:
#             negative_prompt = self.wan.sample_neg_prompt

#         # ── 1. MetaQuery 编码 ────────────────────────────────────────────
#         print("[Generate] MetaQuery 编码...")
#         mq_feat = self.mq_encoder.encode(prompt, ref_image)  # [1, 256, 4096]
#         mq_feat = mq_feat[0].to(device, dtype=torch.bfloat16)  # [256, 4096]
#         mq_feat_noimg = None
#         if ref_image is not None:
#             mq_feat_noimg = self.mq_encoder.encode(prompt, None)[0].to(
#                 device, dtype=torch.bfloat16
#             )
#         null_prompt = negative_prompt
#         null_ref_image = Image.new("RGB", ref_image.size) if ref_image is not None else None
#         mq_feat_null = self.mq_encoder.encode(null_prompt, null_ref_image)[0].to(
#             device, dtype=torch.bfloat16
#         )
#         self._verify_mq_feature_sensitivity(
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             ref_image=ref_image,
#             mq_feat=mq_feat,
#             mq_feat_null=mq_feat_null,
#             mq_feat_noimg=mq_feat_noimg,
#             tag="t2v",
#         )

#         # ── 2. MQ-only context ──────────────────────────────────────────
#         aug_context = [mq_feat]
#         aug_null = [mq_feat_null]
#         self._verify_context_plugged(mq_feat, aug_context[0], tag="t2v")

#         # ── 4. 计算尺寸 ─────────────────────────────────────────────────
#         W, H = size
#         vae_stride = self.wan_config.vae_stride
#         patch_size = self.wan_config.patch_size

#         F = frame_num
#         z_dim = self.wan.vae.model.z_dim
#         target_shape = (
#             z_dim,
#             (F - 1) // vae_stride[0] + 1,
#             H // vae_stride[1],
#             W // vae_stride[2],
#         )

#         seq_len = math.ceil(
#             (target_shape[2] * target_shape[3]) /
#             (patch_size[1] * patch_size[2]) * target_shape[1])

#         seed_g = torch.Generator(device=device)
#         seed_g.manual_seed(seed if seed >= 0 else random.randint(0, sys.maxsize))
#         noise = torch.randn(*target_shape, dtype=torch.float32,
#                             device=device, generator=seed_g)

#         # ── 5. 去噪循环 ─────────────────────────────────────────────────
#         aug_text_len = self._aug_text_len
#         self.wan.model.text_len = aug_text_len
#         self._record_runtime_metric("t2v_text_len_before", int(self._orig_text_len))
#         self._record_runtime_metric("t2v_text_len_after", int(self.wan.model.text_len))
#         if self.wan.model.text_len != aug_text_len:
#             raise RuntimeError(
#                 f"[VERIFY] t2v text_len 设置失败: current={self.wan.model.text_len}, expected={aug_text_len}"
#             )

#         @contextmanager
#         def noop_no_sync():
#             yield
#         no_sync = getattr(self.wan.model, 'no_sync', noop_no_sync)

#         with (
#             torch.amp.autocast('cuda', dtype=self.wan.param_dtype),
#             torch.no_grad(),
#             no_sync(),
#         ):
#             if sample_solver == 'unipc':
#                 scheduler = FlowUniPCMultistepScheduler(
#                     num_train_timesteps=self.wan.num_train_timesteps,
#                     shift=1, use_dynamic_shifting=False)
#                 scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
#                 timesteps = scheduler.timesteps
#             elif sample_solver == 'dpm++':
#                 scheduler = FlowDPMSolverMultistepScheduler(
#                     num_train_timesteps=self.wan.num_train_timesteps,
#                     shift=1, use_dynamic_shifting=False)
#                 sigmas = get_sampling_sigmas(sampling_steps, shift)
#                 timesteps, _ = retrieve_timesteps(
#                     scheduler, device=device, sigmas=sigmas)
#             else:
#                 raise NotImplementedError(f"Unknown solver: {sample_solver}")

#             self.wan.model.to(device)
#             torch.cuda.empty_cache()

#             latents = [noise]
#             mask1, mask2 = masks_like(latents, zero=False)

#             print(f"[Generate] 开始 t2v 去噪 ({len(timesteps)} steps)...")
#             for step_idx, t in enumerate(tqdm(timesteps)):
#                 latent_input = latents
#                 timestep = torch.stack([t])

#                 temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
#                 temp_ts = torch.cat([
#                     temp_ts,
#                     temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
#                 ])
#                 timestep_masked = temp_ts.unsqueeze(0)

#                 pred_cond = self.wan.model(
#                     latent_input, t=timestep_masked,
#                     context=aug_context, seq_len=seq_len)[0]

#                 pred_uncond = self.wan.model(
#                     latent_input, t=timestep_masked,
#                     context=aug_null, seq_len=seq_len)[0]

#                 if step_idx == 0:
#                     self._verify_mq_influence_on_wan(
#                         pred_cond=pred_cond,
#                         latent_input=latent_input,
#                         timestep_masked=timestep_masked,
#                         seq_len=seq_len,
#                         mq_feat=mq_feat,
#                         tag="t2v",
#                     )
#                     self._verify_wan_image_influence_on_wan(
#                         pred_cond=pred_cond,
#                         latent_input=latent_input,
#                         timestep_masked=timestep_masked,
#                         seq_len=seq_len,
#                         mq_feat_noimg=mq_feat_noimg,
#                         tag="t2v",
#                     )

#                 pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)

#                 temp_x0 = scheduler.step(
#                     pred.unsqueeze(0), t, latents[0].unsqueeze(0),
#                     return_dict=False, generator=seed_g)[0]
#                 latents = [temp_x0.squeeze(0)]

#             x0 = latents

#             if self.args.offload_model:
#                 self.wan.model.cpu()
#                 torch.cuda.synchronize()
#                 torch.cuda.empty_cache()

#             print("[Generate] VAE 解码...")
#             videos = self.wan.vae.decode(x0)

#         self.wan.model.text_len = self._orig_text_len
#         self._record_runtime_metric("t2v_text_len_restored", int(self.wan.model.text_len))
#         if self.wan.model.text_len != self._orig_text_len:
#             raise RuntimeError(
#                 f"[VERIFY] t2v text_len 未恢复: current={self.wan.model.text_len}, expected={self._orig_text_len}"
#             )

#         del noise, latents, x0
#         gc.collect()
#         torch.cuda.empty_cache()

#         return videos[0]


# # =============================================================================
# # 视频保存
# # =============================================================================
# def save_video(video_tensor, output_path, fps=24):
#     """
#     保存视频 tensor 为 mp4。

#     Args:
#         video_tensor: [3, T, H, W], 值域 [0, 1] 或 [-1, 1]
#         output_path: 输出路径
#         fps: 帧率
#     """
#     import cv2

#     # 归一化到 [0, 255]
#     video = video_tensor.cpu().float()
#     if video.min() < 0:
#         video = (video + 1.0) / 2.0
#     video = video.clamp(0, 1) * 255
#     video = video.byte()

#     # [3, T, H, W] → [T, H, W, 3]
#     video = video.permute(1, 2, 3, 0).numpy()

#     T, H, W, _ = video.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
#     for frame in video:
#         writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#     writer.release()
#     print(f"✅ 视频已保存: {output_path} ({T} frames, {W}x{H}, {fps}fps)")


# # =============================================================================
# # Main
# # =============================================================================
# if __name__ == "__main__":
#     args = parse_args()

#     pipeline = MetaQueryWanPipeline(args)
#     verify_report_path = args.verify_report_path.strip()
#     if args.verify_level != "none" and not verify_report_path:
#         verify_report_path = f"{args.output_path}.verify.json"

#     # 加载参考图
#     ref_image = None
#     if args.ref_image and os.path.exists(args.ref_image):
#         ref_image = Image.open(args.ref_image).convert("RGB")
#         print(f"[Main] 参考图: {args.ref_image} ({ref_image.size})")

#     try:
#         # 生成
#         if args.mode == "i2v":
#             if ref_image is None:
#                 raise ValueError("i2v 模式需要 --ref_image!")
#             if args.i2v_method == "animate_ref_slot":
#                 video = pipeline.generate_i2v_animate_ref_slot(
#                     prompt=args.prompt,
#                     ref_image=ref_image,
#                     negative_prompt=args.negative_prompt,
#                     max_area=args.max_area,
#                     frame_num=args.frame_num,
#                     shift=args.shift,
#                     sample_solver=args.sample_solver,
#                     sampling_steps=args.sampling_steps,
#                     guide_scale=args.guide_scale,
#                     seed=args.seed,
#                 )
#             else:
#                 video = pipeline.generate_i2v(
#                     prompt=args.prompt,
#                     ref_image=ref_image,
#                     negative_prompt=args.negative_prompt,
#                     max_area=args.max_area,
#                     frame_num=args.frame_num,
#                     shift=args.shift,
#                     sample_solver=args.sample_solver,
#                     sampling_steps=args.sampling_steps,
#                     guide_scale=args.guide_scale,
#                     seed=args.seed,
#                 )
#         else:
#             video = pipeline.generate_t2v(
#                 prompt=args.prompt,
#                 ref_image=ref_image,
#                 negative_prompt=args.negative_prompt,
#                 size=tuple(args.size),
#                 frame_num=args.frame_num,
#                 shift=args.shift,
#                 sample_solver=args.sample_solver,
#                 sampling_steps=args.sampling_steps,
#                 guide_scale=args.guide_scale,
#                 seed=args.seed,
#             )

#         # 保存
#         save_video(video, args.output_path)
#     finally:
#         pipeline.dump_verify_report(verify_report_path)
