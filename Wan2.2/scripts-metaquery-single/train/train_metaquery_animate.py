"""
train_metaquery_animate.py
===========================
MetaQuery + Wan2.2 Animate (不使用骨架/面部条件) 联合训练脚本。

★ 核心架构分析 (WanAnimateModel):
    - in_dim=36: x[16] concat y[20] = [mask(4ch) + VAE_latent(16ch)]
    - dim=5120, text_dim=4096, text_len=512
    - CLIP visual → img_emb(MLPProj) → [B, 257, 5120] 拼接到 context 前面
    - pose_latents → pose_patch_embedding → 加到 x 的时序维度上
    - face_pixel_values → motion_encoder → face_encoder → face_adapter → 逐块注入
    - VAE: Wan2_1_VAE, stride=(4,8,8)

★ 本训练方案:
    - MetaQuery 替代 CLIP visual 的角色: 提供更丰富的视觉语义
    - 不使用 pose_latents (传入全零)
    - face_pixel_values 可选 (传入全零或真实面部视频)
    - Connector 直接输出 4096 (Wan text_dim), 经过 DiT 的 text_embedding 处理
    - MQ context 拼接到 T5 context 前面, 替换原始 CLIP 位置
    - text_len 扩展: 512 → 512 + 256 (MQ tokens)
    - 注意: img_emb (CLIP→dim) 的 257 tokens 不再使用, 因为 MQ 提供了更好的视觉特征
    - y 参数仍然保留 (参考图 VAE latent + mask, 这是 Animate 核心条件)

用法:
    python train_metaquery_animate.py \
        --wan_checkpoint_dir /path/to/Wan2.2-Animate-14B \
        --qwen3vl_model_id /path/to/Qwen3-VL-2B-Thinking
"""

import os
import sys
import gc
import json
import math
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from einops import rearrange

# ── 路径设置 ─────────────────────────────────────────────────────────────────
WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))
METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
sys.path.insert(0, METAQUERY_ROOT)

# 复用 TI2V 的 MetaQuery Encoder (输出 4096) + checkpoint 工具
from train_metaquery_wan import (
    MetaQueryEncoderForWan,
    TomAndJerryVideoDataset,
    load_mq_encoder_state,
    save_mq_checkpoint_bundle,
)


# =============================================================================
# 配置
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train MetaQuery for Wan Animate (no skeleton)")

    p.add_argument("--wan_checkpoint_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B")
    p.add_argument("--qwen3vl_model_id", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking")
    p.add_argument("--output_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/metaquery_animate_training")
    p.add_argument("--caption_json_root", type=str, default=None,
                   help="旧版Tom&Jerry标注JSON根目录（仅旧数据格式需要；使用本地OpenVid训练可不传）")
    p.add_argument("--manifest_path", type=str, default=None)
    p.add_argument("--local_openvid_video_root", type=str, default=None,
                   help="本地 OpenVid 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video")
    p.add_argument("--local_openvid_csv_path", type=str, default=None,
                   help="本地 OpenVid CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv")
    p.add_argument("--local_openvid_limit", type=int, default=None,
                   help="仅使用前N条本地匹配样本，默认使用全部已匹配样本")
    p.add_argument("--local_openvid_hd_video_root", type=str, default=None,
                   help="本地 OpenVid HD 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video_HD")
    p.add_argument("--local_openvid_hd_csv_path", type=str, default=None,
                   help="本地 OpenVid HD CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv")
    p.add_argument("--local_openvid_hd_limit", type=int, default=None,
                   help="仅使用前N条本地HD匹配样本，默认使用全部已匹配样本")
    p.add_argument("--local_video_cache_dir", type=str, default=None,
                   help="本地视频缓存目录（用于URL/字节解码缓存，可选）")

    # 训练
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--log_every_step", action="store_true",
                   help="每个优化 step 都打印详细训练日志")
    p.add_argument("--wandb_log_every_step", action="store_true",
                   help="每个优化 step 都写入 W&B（默认按 log_steps 写入）")
    p.add_argument("--metrics_jsonl_path", type=str, default="",
                   help="可选：将每步指标追加写入 JSONL 文件")
    p.add_argument("--log_cuda_memory", action="store_true",
                   help="记录并输出 CUDA 显存指标")
    p.add_argument("--frame_num", type=int, default=77,
                   help="Animate 默认 frame_num=77, 必须是 4n+1")
    p.add_argument("--max_area", type=int, default=512 * 512,
                   help="Animate 使用 512x512 分辨率")
    p.add_argument("--max_caption_tokens", type=int, default=512)
    p.add_argument("--caption_tokenizer_path", type=str, default="google/umt5-xxl")
    p.add_argument("--min_duration_sec", type=float, default=0.5)
    p.add_argument("--max_duration_sec", type=float, default=20.0)
    p.add_argument("--probe_missing_meta", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    # MetaQuery
    p.add_argument("--num_metaqueries", type=int, default=256)
    p.add_argument("--connector_num_hidden_layers", type=int, default=24)
    p.add_argument("--mq_gradient_checkpointing", action="store_true",
                   help="启用 MetaQuery 编码器梯度检查点，降低显存占用")
    p.add_argument("--null_caption_prob", type=float, default=0.1)
    p.add_argument("--null_image_prob", type=float, default=0.1)

    # 设备
    p.add_argument("--dit_device", type=int, default=0)
    p.add_argument("--encoder_device", type=int, default=1)
    p.add_argument("--resume_mq_encoder_path", type=str, default=None)
    p.add_argument("--t5_cpu", action="store_true",
                   help="将Wan的T5文本编码器保留在CPU，显著降低GPU显存占用（速度会变慢）")
    p.add_argument("--dit_fsdp", action="store_true",
                   help="启用 Wan DiT 的 FSDP 参数分片，降低单卡模型权重占用")
    p.add_argument("--t5_fsdp", action="store_true",
                   help="启用 T5 编码器的 FSDP 参数分片")
    p.add_argument("--use_sp", action="store_true",
                   help="启用 sequence parallel（xDiT/USP 路径）")
    p.add_argument("--no_init_on_cpu", action="store_true",
                   help="关闭 init_on_cpu；默认开启以减小加载瞬时显存峰值")
    p.add_argument("--convert_model_dtype", action="store_true",
                   help="将 Wan DiT 参数显式转换到 config.param_dtype（仅非FSDP时生效）")
    p.add_argument("--aggressive_empty_cache", action="store_true",
                   help="每步训练后执行 torch.cuda.empty_cache()，缓解显存碎片")

    # face video (可选)
    p.add_argument("--use_face", action="store_true",
                   help="启用面部视频条件 (需要 face video 数据)")
    p.add_argument("--wandb_enabled", action="store_true",
                   help="启用 Weights & Biases 训练日志")
    p.add_argument("--wandb_project", type=str, default="wan-metaquery",
                   help="W&B project 名称")
    p.add_argument("--wandb_entity", type=str, default="",
                   help="W&B entity/team 名称")
    p.add_argument("--wandb_run_name", type=str, default="",
                   help="W&B run 名称, 留空自动生成")
    p.add_argument("--wandb_tags", type=str, default="",
                   help="W&B tags, 逗号分隔")
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="W&B 模式")
    p.add_argument("--wandb_api_key", type=str, default="",
                   help="W&B API Key, 传入后会写入 WANDB_API_KEY 环境变量")
    p.add_argument("--wandb_log_checkpoint", action="store_true",
                   help="在 W&B 中记录 checkpoint 路径")

    return p.parse_args()


# =============================================================================
# Trainer
# =============================================================================
class MetaQueryAnimateTrainer:
    """
    MetaQuery + Wan Animate 联合训练 (无骨架, 面部可选)。

    ★ WanAnimateModel.forward 签名:
        forward(x, t, clip_fea, context, seq_len, y=None,
                pose_latents=None, face_pixel_values=None)

    ★ 我们的修改:
        - clip_fea: 传入零向量 (MQ 替代了 CLIP visual 的角色)
          但 img_emb 仍会将其投影到 dim, 拼接到 context 前面
          这 257 个 "零 CLIP" token 不影响训练, DiT 会学到忽略它们
        - context: [MQ_tokens(256, 4096) + T5_tokens(512, 4096)]
          text_len 扩展到 512+256=768
        - pose_latents: 全零 tensor
        - face_pixel_values: 全零 tensor (或真实面部视频, 如果启用)
        - y: 参考图 VAE latent + mask (保留, 这是 Animate 核心条件)

    ★ 替代方案 (更激进):
        也可以让 MQ 占据 img_emb 的位置而不是放在 text context 前面,
        但这需要修改 WanAnimateModel 内部代码, 不如 context 拼接简单。
    """

    def __init__(self, args):
        self.args = args
        self.dev_dit = torch.device(f"cuda:{args.dit_device}")
        self.dev_enc = torch.device(f"cuda:{args.encoder_device}")
        self.wandb = None
        self.wandb_run = None
        self.is_main_process = self._is_main_process()
        self._skipped_step_count = 0
        self._oom_skip_count = 0
        self._error_skip_count = 0
        self._param_monitor = []
        self._trainable_param_count = 0
        self._init_trainable_norm = 0.0
        self._init_param_sample_norm = 0.0
        _metrics_jsonl = (args.metrics_jsonl_path or "").strip()
        self._metrics_jsonl_path = str(Path(_metrics_jsonl).expanduser().resolve()) if _metrics_jsonl else ""
        self._metrics_history = []
        self._train_before_checkpoint_path = ""
        self._train_wall_start = 0.0

        print("\n" + "=" * 60)
        print("  MetaQuery + Wan Animate 联合训练 (无骨架)")
        print("=" * 60)
        self._load_models()
        self._setup_optimizer()
        self._init_trainability_monitor()
        self._init_wandb()

    def _is_main_process(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        rank_env = os.environ.get("RANK")
        if rank_env is None:
            return True
        return int(rank_env) == 0

    def _mq_encoder_module(self):
        return self.mq_encoder.module if hasattr(self.mq_encoder, "module") else self.mq_encoder

    def _mq_trainable_params(self):
        module = self._mq_encoder_module()
        if hasattr(module, "get_trainable_params"):
            return module.get_trainable_params()
        return [p for p in module.parameters() if p.requires_grad]

    def _init_trainability_monitor(self):
        module = self._mq_encoder_module()
        self._param_monitor = []
        total_sq = 0.0
        sample_sq = 0.0
        total_params = 0
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            data = p.detach().float().view(-1)
            numel = int(data.numel())
            if numel <= 0:
                continue
            sample_k = min(8, numel)
            if sample_k == 1:
                idx = torch.zeros(1, dtype=torch.long)
            else:
                idx = torch.linspace(0, numel - 1, steps=sample_k, dtype=torch.long)
            init_vals = data.index_select(0, idx.to(data.device)).cpu()
            self._param_monitor.append((name, p, idx.cpu(), init_vals))
            total_sq += float(torch.sum(data * data).item())
            sample_sq += float(torch.sum(init_vals * init_vals).item())
            total_params += numel
        self._trainable_param_count = total_params
        self._init_trainable_norm = math.sqrt(max(total_sq, 0.0))
        self._init_param_sample_norm = math.sqrt(max(sample_sq, 0.0))
        if self.is_main_process:
            print(
                "[VERIFY][TRAIN-INIT] "
                f"trainable_params={self._trainable_param_count:,} "
                f"init_param_norm={self._init_trainable_norm:.6f} "
                f"monitor_tensors={len(self._param_monitor)}"
            )

    def _collect_trainability_metrics(self):
        sample_abs_sum = 0.0
        sample_l2_sum = 0.0
        sample_cur_sq_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for _, p, idx_cpu, init_vals_cpu in self._param_monitor:
                data = p.detach().float().view(-1)
                idx = idx_cpu.to(data.device)
                now_vals = data.index_select(0, idx).cpu()
                diff = now_vals - init_vals_cpu
                sample_abs_sum += float(diff.abs().sum().item())
                sample_l2_sum += float(torch.sum(diff * diff).item())
                sample_cur_sq_sum += float(torch.sum(now_vals * now_vals).item())
                sample_count += int(diff.numel())
        cur_sample_norm = math.sqrt(max(sample_cur_sq_sum, 0.0))
        init_sample_norm = max(self._init_param_sample_norm, 1e-12)
        return {
            "train/param_sample_norm": float(cur_sample_norm),
            "train/param_sample_norm_delta_ratio": float(abs(cur_sample_norm - self._init_param_sample_norm) / init_sample_norm),
            "train/param_sample_abs_delta_mean": float(sample_abs_sum / max(sample_count, 1)),
            "train/param_sample_l2_delta": float(math.sqrt(max(sample_l2_sum, 0.0))),
            "train/trainable_param_count": int(self._trainable_param_count),
        }

    def _collect_cuda_memory_metrics(self):
        if not (torch.cuda.is_available() and self.args.log_cuda_memory):
            return {}
        dit_idx = self.dev_dit.index if self.dev_dit.type == "cuda" else None
        enc_idx = self.dev_enc.index if self.dev_enc.type == "cuda" else None

        def _mem(prefix, dev_idx):
            if dev_idx is None:
                return {}
            return {
                f"train/cuda_{prefix}_alloc_mb": float(torch.cuda.memory_allocated(dev_idx) / 1024 / 1024),
                f"train/cuda_{prefix}_reserved_mb": float(torch.cuda.memory_reserved(dev_idx) / 1024 / 1024),
                f"train/cuda_{prefix}_max_alloc_mb": float(torch.cuda.max_memory_allocated(dev_idx) / 1024 / 1024),
            }

        metrics = {}
        metrics.update(_mem("dit", dit_idx))
        metrics.update(_mem("enc", enc_idx))
        return metrics

    def _append_metrics_jsonl(self, metrics):
        if not self.is_main_process:
            return
        if not self._metrics_jsonl_path:
            return
        try:
            path = Path(self._metrics_jsonl_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] 写入 metrics_jsonl 失败: {e}")

    def _record_metrics(self, metrics):
        keep_keys = [
            "train/step",
            "train/loss_step",
            "train/loss_ema",
            "train/lr",
            "train/grad_norm",
            "train/step_time_sec",
            "train/samples_per_sec",
            "train/param_sample_abs_delta_mean",
            "train/param_sample_l2_delta",
            "train/param_sample_norm_delta_ratio",
            "train/skipped_step_count",
            "train/oom_skip_count",
            "train/error_skip_count",
        ]
        self._metrics_history.append({k: metrics[k] for k in keep_keys if k in metrics})

    def _build_metrics_summary(self, step: int):
        summary = {
            "current_step": int(step),
            "logged_steps": int(len(self._metrics_history)),
            "metrics_jsonl_path": self._metrics_jsonl_path,
            "skipped_step_count": int(self._skipped_step_count),
            "oom_skip_count": int(self._oom_skip_count),
            "error_skip_count": int(self._error_skip_count),
        }
        if self._metrics_history:
            last = self._metrics_history[-1]
            loss_vals = [float(m.get("train/loss_step", 0.0)) for m in self._metrics_history if "train/loss_step" in m]
            step_time_vals = [float(m.get("train/step_time_sec", 0.0)) for m in self._metrics_history if "train/step_time_sec" in m]
            sps_vals = [float(m.get("train/samples_per_sec", 0.0)) for m in self._metrics_history if "train/samples_per_sec" in m]
            summary.update(
                {
                    "step_first": int(self._metrics_history[0].get("train/step", 0)),
                    "step_last": int(last.get("train/step", 0)),
                    "loss_last": float(last.get("train/loss_step", 0.0)),
                    "loss_ema_last": float(last.get("train/loss_ema", 0.0)),
                    "lr_last": float(last.get("train/lr", 0.0)),
                    "grad_norm_last": float(last.get("train/grad_norm", 0.0)),
                    "loss_min": float(min(loss_vals) if loss_vals else 0.0),
                    "loss_max": float(max(loss_vals) if loss_vals else 0.0),
                    "step_time_sec_avg": float(sum(step_time_vals) / max(len(step_time_vals), 1)),
                    "samples_per_sec_avg": float(sum(sps_vals) / max(len(sps_vals), 1)),
                }
            )
        if self._train_wall_start > 0:
            summary["wall_time_sec"] = float(max(time.perf_counter() - self._train_wall_start, 0.0))
        return summary

    def _write_training_chain_manifest(self, output_dir: Path, final_checkpoint_path: str, final_step: int):
        if not self.is_main_process:
            return
        output_dir = output_dir.expanduser().resolve()
        payload = {
            "before_checkpoint_path": self._train_before_checkpoint_path,
            "final_checkpoint_path": str(Path(final_checkpoint_path).expanduser().resolve()),
            "metrics_jsonl_path": self._metrics_jsonl_path,
            "args": {str(k): v for k, v in vars(self.args).items()},
            "metrics_summary": self._build_metrics_summary(step=final_step),
        }
        with open(output_dir / "training_chain_manifest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _wandb_config(self):
        args = self.args
        return {
            "task": "wan_animate",
            "learning_rate": args.learning_rate,
            "num_train_steps": args.num_train_steps,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "frame_num": args.frame_num,
            "max_area": args.max_area,
            "num_metaqueries": args.num_metaqueries,
            "connector_num_hidden_layers": args.connector_num_hidden_layers,
            "mq_gradient_checkpointing": args.mq_gradient_checkpointing,
            "null_caption_prob": args.null_caption_prob,
            "null_image_prob": args.null_image_prob,
            "t5_cpu": args.t5_cpu,
            "dit_fsdp": args.dit_fsdp,
            "t5_fsdp": args.t5_fsdp,
            "use_sp": args.use_sp,
            "aggressive_empty_cache": args.aggressive_empty_cache,
            "seed": args.seed,
            "save_steps": args.save_steps,
            "log_steps": args.log_steps,
            "log_every_step": args.log_every_step,
            "wandb_log_every_step": args.wandb_log_every_step,
            "metrics_jsonl_path": args.metrics_jsonl_path,
            "log_cuda_memory": args.log_cuda_memory,
            "output_dir": args.output_dir,
            "manifest_path": args.manifest_path,
            "local_openvid_video_root": args.local_openvid_video_root,
            "local_openvid_csv_path": args.local_openvid_csv_path,
            "local_openvid_limit": args.local_openvid_limit,
            "local_openvid_hd_video_root": args.local_openvid_hd_video_root,
            "local_openvid_hd_csv_path": args.local_openvid_hd_csv_path,
            "local_openvid_hd_limit": args.local_openvid_hd_limit,
            "wan_checkpoint_dir": args.wan_checkpoint_dir,
            "qwen3vl_model_id": args.qwen3vl_model_id,
            "use_face": args.use_face,
        }

    def _init_wandb(self):
        args = self.args
        if not getattr(args, "wandb_enabled", False):
            return
        if not self.is_main_process:
            return
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        try:
            import wandb
        except ImportError:
            print("[W&B] 未安装 wandb, 已跳过日志记录")
            return
        run_name = args.wandb_run_name.strip() or f"wan-animate-metaquery-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        self.wandb = wandb
        self.wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            mode=args.wandb_mode,
            config=self._wandb_config(),
            tags=tags or None,
        )
        print(f"[W&B] 已初始化: project={args.wandb_project}, run={run_name}")

    def _load_models(self):
        args = self.args

        # ── 1. Wan Animate Pipeline ──────────────────────────────────────
        print("\n[1/3] 加载 Wan Animate Pipeline...")
        from wan import WanAnimate
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['animate-14B']
        runtime_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        self.wan = WanAnimate(
            config=config,
            checkpoint_dir=args.wan_checkpoint_dir,
            device_id=args.dit_device,
            rank=runtime_rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=args.use_sp,
            t5_cpu=args.t5_cpu,
            init_on_cpu=not args.no_init_on_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        if not (args.dit_fsdp or args.use_sp):
            self.wan.noise_model.to(self.dev_dit)
        self.wan.noise_model.eval().requires_grad_(False)
        self.wan_config = config
        self.text_len = config.text_len  # 512
        print(f"  ✅ Wan Animate 14B 已加载")
        print(f"    in_dim={self.wan.noise_model.in_dim}")
        print(f"    text_dim={self.wan.noise_model.text_dim}")
        print(f"    dim={self.wan.noise_model.dim}")
        print(f"    text_len={self.text_len}")

        # ── 2. MetaQuery Encoder → 4096 ─────────────────────────────────
        print("\n[2/3] 加载 MetaQuery Encoder (→4096)...")
        self.mq_encoder = MetaQueryEncoderForWan(
            qwen3vl_model_id=args.qwen3vl_model_id,
            num_metaqueries=args.num_metaqueries,
            connector_num_hidden_layers=args.connector_num_hidden_layers,
            gradient_checkpointing=args.mq_gradient_checkpointing,
            dtype=torch.bfloat16,
            device=f"cuda:{args.encoder_device}",
        )
        self.mq_encoder.train()
        if args.resume_mq_encoder_path:
            state, resolved_path = load_mq_encoder_state(
                args.resume_mq_encoder_path,
                map_location="cpu",
            )
            missing, unexpected = self.mq_encoder.load_state_dict(state, strict=False)
            print(f"  ✅ 已加载初始权重: {resolved_path}")
            print(f"     missing={len(missing)}, unexpected={len(unexpected)}")

        # ── 3. 验证 ─────────────────────────────────────────────────────
        print("\n[3/3] 验证维度对齐...")
        assert self.wan.noise_model.text_dim == 4096
        assert self.mq_encoder.wan_text_dim == 4096

        self._orig_text_len = self.wan.noise_model.text_len
        self._aug_text_len = self._orig_text_len + args.num_metaqueries
        print(f"  ✅ text_len: {self._orig_text_len} → {self._aug_text_len}")

    def _setup_optimizer(self):
        args = self.args
        trainable_params = self._mq_trainable_params()
        total = sum(p.numel() for p in trainable_params)
        print(f"\n[Optimizer] 可训练参数: {total / 1e6:.1f}M")

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate,
            betas=(0.9, 0.95), weight_decay=0.1,
        )

        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, args.num_train_steps - args.warmup_steps)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, device="cuda"):
        """构建 Animate 的 i2v mask (参考 WanAnimate.get_i2v_mask)"""
        msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        msk[:, :mask_len] = 1
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, lat_t, lat_h, lat_w]
        return msk

    def _compute_loss(self, batch):
        """计算 Flow Matching 损失。"""
        args = self.args
        captions = batch["caption"]
        videos = batch["video"]
        mq_refs = batch["mq_ref_image"]
        ref_images = batch["ref_image"]
        B = len(captions)

        # ── 1. MetaQuery 编码 ────────────────────────────────────────────
        mq_images = []
        for ref in mq_refs:
            mq_images.append([ref] if ref is not None else [Image.new("RGB", (224, 224))])
        mq_features = self.mq_encoder(captions, mq_images)
        # [B, 256, 4096]

        # ── 2. T5 编码 ──────────────────────────────────────────────────
        with torch.no_grad():
            if not self.args.t5_cpu and not self.args.t5_fsdp:
                self.wan.text_encoder.model.to(self.dev_dit)
                t5_context = self.wan.text_encoder(captions, self.dev_dit)
            else:
                t5_context = self.wan.text_encoder(captions, torch.device("cpu"))
                t5_context = [t.to(self.dev_dit, dtype=torch.bfloat16) for t in t5_context]

        # ── 3. 拼接 MQ + T5 → augmented context ─────────────────────────
        augmented_context = []
        for i in range(B):
            mq_feat = mq_features[i].to(self.dev_dit, dtype=torch.bfloat16)
            t5_feat = t5_context[i].to(self.dev_dit, dtype=torch.bfloat16)
            aug = torch.cat([mq_feat, t5_feat], dim=0)  # [768, 4096]
            augmented_context.append(aug)

        # ── 4. VAE 编码视频 ─────────────────────────────────────────────
        with torch.no_grad():
            latents = []
            for v in videos:
                z = self.wan.vae.encode([v.to(self.dev_dit, dtype=torch.bfloat16)])
                latents.append(z[0])

        # ── 5. 构建 y (参考图 VAE latent + mask) ────────────────────────
        with torch.no_grad():
            y_list = []
            model_latents = []
            for i in range(B):
                lat = latents[i]
                C, T, H, W = lat.shape

                # 参考图 → VAE 编码
                ref_img = ref_images[i]
                ref_np = np.array(ref_img.resize((W * 8, H * 8)))  # 匹配 latent 尺寸
                ref_tensor = torch.from_numpy(ref_np).float().permute(2, 0, 1) / 127.5 - 1.0
                ref_tensor = ref_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
                ref_z = self.wan.vae.encode([ref_tensor])[0]  # [16, 1, H', W']

                # y_ref: mask(4ch) + ref_latent(16ch) = 20ch, 1 frame
                mask_ref = self._get_i2v_mask(1, H, W, mask_len=1, device=self.dev_dit)
                y_ref = torch.cat([mask_ref, ref_z], dim=0).to(torch.bfloat16)

                # y_reft: 全零 (不使用时序参考帧)
                # mask(4ch) + zeros(16ch) = 20ch, T frames
                msk_reft = self._get_i2v_mask(T, H, W, mask_len=0, device=self.dev_dit)
                zero_reft = torch.zeros(16, T, H, W, device=self.dev_dit, dtype=torch.bfloat16)
                y_reft = torch.cat([msk_reft, zero_reft], dim=0).to(torch.bfloat16)

                # y = concat(y_ref, y_reft) → [20, 1+T, H', W']
                y = torch.cat([y_ref, y_reft], dim=1)
                y_list.append(y)

                # Animate 模型内部会先拼接 x 与 y (按 channel 维),
                # 因此 x 的时间长度也必须是 1+T（首帧参考 + T 帧视频潜变量）。
                x_lat = torch.cat([ref_z, lat], dim=1).to(torch.bfloat16)
                model_latents.append(x_lat)

        # ── 6. 构建其他 Animate 条件 ────────────────────────────────────
        # CLIP: 零向量 (MQ 替代 CLIP)
        clip_fea = torch.zeros(B, 257, 1280, device=self.dev_dit, dtype=torch.bfloat16)

        # Pose: 全零
        pose_latents_list = []
        for lat in latents:
            C, T, H, W = lat.shape
            pose_z = torch.zeros(16, T, H, W, device=self.dev_dit, dtype=torch.bfloat16)
            pose_latents_list.append(pose_z)
        pose_latents = torch.stack(pose_latents_list)

        # Face: 默认关闭。关闭后会跳过 noise_model 内的 motion/face 分支，显著降低显存。
        if args.use_face:
            face_pixel_values = torch.zeros(
                B, 3, args.frame_num, 512, 512,
                device=self.dev_dit, dtype=torch.bfloat16
            )
        else:
            face_pixel_values = None

        # ── 7. Flow Matching ────────────────────────────────────────────
        patch_size = self.wan_config.patch_size
        x_inputs = []
        timestep_list = []
        target_list = []
        max_seq_len = 0

        for lat in model_latents:
            C, T, H, W = lat.shape
            seq_len_i = math.ceil(np.prod([T, H, W]) // (patch_size[1] * patch_size[2]))
            max_seq_len = max(max_seq_len, seq_len_i)

            t_val = torch.rand(1, device=self.dev_dit, dtype=torch.float32)
            noise = torch.randn_like(lat, dtype=torch.float32)

            sigma = t_val.view(-1, 1, 1, 1)
            noisy_lat = (1.0 - sigma) * lat.float() + sigma * noise
            velocity = noise - lat.float()

            x_inputs.append(noisy_lat)
            timestep_list.append(t_val * self.wan.num_train_timesteps)
            target_list.append(velocity)

        timesteps_wan = torch.cat(timestep_list).to(self.dev_dit)

        # ── 8. 扩展 text_len → DiT forward ──────────────────────────────
        orig_text_len = self.wan.noise_model.text_len
        self.wan.noise_model.text_len = self._aug_text_len

        try:
            from wan.modules.animate.animate_utils import TensorList

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model_output = self.wan.noise_model(
                    TensorList(x_inputs),
                    t=timesteps_wan,
                    clip_fea=clip_fea,
                    context=augmented_context,
                    seq_len=max_seq_len,
                    y=[y_list[i] for i in range(B)],
                    pose_latents=pose_latents,
                    face_pixel_values=face_pixel_values,
                )

            total_loss = 0.0
            for i in range(B):
                pred = model_output[i].float()
                target = target_list[i]
                loss = F.mse_loss(pred, target)
                total_loss += loss
            total_loss /= B

        finally:
            self.wan.noise_model.text_len = orig_text_len

        return total_loss

    def train(self):
        args = self.args
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        dataset = TomAndJerryVideoDataset(
            caption_json_root=args.caption_json_root,
            manifest_path=args.manifest_path,
            seed=args.seed,
            frame_num=args.frame_num,
            max_area=args.max_area,
            null_caption_prob=args.null_caption_prob,
            null_image_prob=args.null_image_prob,
            max_caption_tokens=args.max_caption_tokens,
            caption_tokenizer_path=args.caption_tokenizer_path,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            probe_missing_meta=args.probe_missing_meta,
            local_openvid_video_root=args.local_openvid_video_root,
            local_openvid_csv_path=args.local_openvid_csv_path,
            local_openvid_limit=args.local_openvid_limit,
            local_openvid_hd_video_root=args.local_openvid_hd_video_root,
            local_openvid_hd_csv_path=args.local_openvid_hd_csv_path,
            local_openvid_hd_limit=args.local_openvid_hd_limit,
            local_video_cache_dir=args.local_video_cache_dir,
        )
        if len(dataset) == 0:
            raise RuntimeError("数据集为空！")

        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=args.dataloader_num_workers,
            pin_memory=True, collate_fn=self._collate_fn,
        )

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self._metrics_jsonl_path:
            self._metrics_jsonl_path = str((output_dir / "logs" / "train_metrics.jsonl").expanduser().resolve())
        args.output_dir = str(output_dir)
        args.metrics_jsonl_path = self._metrics_jsonl_path
        args.output_dir = str(output_dir)
        args.metrics_jsonl_path = self._metrics_jsonl_path
        self._train_wall_start = time.perf_counter()

        self._train_before_checkpoint_path = str(output_dir / "checkpoint-before-training")
        self._save_checkpoint(
            self._train_before_checkpoint_path,
            step=0,
            extra_info={
                "is_before_training": True,
                "resume_mq_encoder_path": getattr(args, "resume_mq_encoder_path", None),
                "note": "trainable params snapshot before optimizer updates",
            },
        )
        if self.is_main_process:
            print(f"[VERIFY] 已保存训练前快照: {self._train_before_checkpoint_path}")

        self.mq_encoder.train()
        step = 0
        running_loss = 0.0
        data_iter = iter(dataloader)
        pbar = tqdm(total=args.num_train_steps, desc="Animate Training")
        self.optimizer.zero_grad(set_to_none=True)

        while step < args.num_train_steps:
            step_wall_start = time.perf_counter()
            accum_loss = 0.0
            skip_optimizer_step = False
            had_fatal_cuda_error = False
            backward_ok = 0
            skip_reason = ""
            for accum_step in range(args.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                try:
                    loss = self._compute_loss(batch)
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    accum_loss += loss.item()
                    backward_ok += 1
                except Exception as e:
                    err = str(e)
                    bad_video = None
                    try:
                        bad_video = batch.get("video_path", None)
                    except Exception:
                        bad_video = None
                    print(f"[WARN] step {step} accum_step {accum_step} 训练异常: {err}")
                    if bad_video is not None:
                        print(f"[WARN] step {step} accum_step {accum_step} bad_video={bad_video}")
                    err_l = err.lower()
                    is_illegal_access = "illegal memory access" in err_l
                    is_device_assert = "device-side assert" in err_l
                    if isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in err_l):
                        skip_optimizer_step = True
                        skip_reason = "oom"
                        self.optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        break
                    if is_illegal_access or is_device_assert:
                        had_fatal_cuda_error = True
                        skip_optimizer_step = True
                        skip_reason = "fatal_cuda"
                        self.optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        break
                    skip_optimizer_step = True
                    skip_reason = "error"
                    self.optimizer.zero_grad(set_to_none=True)
                    break

            if had_fatal_cuda_error:
                raise RuntimeError(
                    f"Fatal CUDA kernel error at step={step}. "
                    "检测到 illegal memory access/device-side assert，已中止训练。"
                )

            if backward_ok == 0:
                self._skipped_step_count += 1
                if skip_reason == "oom":
                    self._oom_skip_count += 1
                elif skip_reason and skip_reason != "fatal_cuda":
                    self._error_skip_count += 1
                continue

            if skip_optimizer_step:
                self._skipped_step_count += 1
                if skip_reason == "oom":
                    self._oom_skip_count += 1
                else:
                    self._error_skip_count += 1
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._mq_trainable_params(), args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if args.aggressive_empty_cache:
                torch.cuda.empty_cache()

            step += 1
            step_time = max(time.perf_counter() - step_wall_start, 1e-6)
            running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss
            lr = self.scheduler.get_last_lr()[0]
            grad_norm_value = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
            effective_samples = int(max(backward_ok, 0))
            samples_per_sec = float(effective_samples / step_time)

            metrics = {
                "train/loss_step": float(accum_loss),
                "train/loss_ema": float(running_loss),
                "train/lr": float(lr),
                "train/grad_norm": float(grad_norm_value),
                "train/step": int(step),
                "train/step_time_sec": float(step_time),
                "train/samples_per_sec": float(samples_per_sec),
                "train/backward_ok_microbatches": int(backward_ok),
                "train/effective_batch_samples": int(effective_samples),
                "train/skipped_step_count": int(self._skipped_step_count),
                "train/oom_skip_count": int(self._oom_skip_count),
                "train/error_skip_count": int(self._error_skip_count),
            }
            metrics.update(self._collect_trainability_metrics())
            metrics.update(self._collect_cuda_memory_metrics())

            should_log = bool(args.log_every_step or (step % args.log_steps == 0))
            should_wandb_log = bool(
                self.wandb_run is not None and (args.wandb_log_every_step or should_log)
            )

            if should_log:
                pbar.set_postfix(loss=f"{accum_loss:.4f}", avg=f"{running_loss:.4f}", lr=f"{lr:.2e}")
                print(
                    f"\n[Step {step}] loss={accum_loss:.4f} avg={running_loss:.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm_value:.2f} dt={step_time:.2f}s "
                    f"samp/s={samples_per_sec:.2f} param_delta={metrics['train/param_sample_abs_delta_mean']:.3e} "
                    f"skip(oom/err/total)={self._oom_skip_count}/{self._error_skip_count}/{self._skipped_step_count}"
                )
            if should_wandb_log:
                self.wandb.log(metrics, step=step)
            self._append_metrics_jsonl(metrics)
            self._record_metrics(metrics)

            if step % args.save_steps == 0:
                self._save_checkpoint(output_dir / f"checkpoint-{step}", step)
            pbar.update(1)

        pbar.close()
        final_ckpt_path = str(output_dir / "checkpoint-final")
        self._save_checkpoint(final_ckpt_path, step)
        self._write_training_chain_manifest(output_dir, final_checkpoint_path=final_ckpt_path, final_step=step)
        if self.wandb_run is not None:
            self.wandb.finish()

    def _save_checkpoint(self, path, step, extra_info=None):
        if not self.is_main_process:
            return
        path = Path(path).expanduser().resolve()
        module = self._mq_encoder_module()
        ckpt_info = save_mq_checkpoint_bundle(
            path=path,
            module=module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            args=self.args,
            metrics_tail=self._metrics_history[-200:],
            metrics_summary=self._build_metrics_summary(step=step),
            extra_info={
                "before_checkpoint_path": self._train_before_checkpoint_path,
                "metrics_jsonl_path": self._metrics_jsonl_path,
                **(extra_info or {}),
            },
        )
        print(f"  💾 Saved: {ckpt_info['path']}")
        if self.wandb_run is not None and self.args.wandb_log_checkpoint:
            self.wandb.log(
                {
                    "checkpoint/step": int(step),
                    "checkpoint/path": str(ckpt_info["path"]),
                },
                step=step,
            )

    @staticmethod
    def _collate_fn(batch):
        return {k: [item[k] for item in batch] for k in batch[0]}


if __name__ == "__main__":
    args = parse_args()
    trainer = MetaQueryAnimateTrainer(args)
    trainer.train()
