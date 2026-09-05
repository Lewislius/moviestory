# 下面这个是，对DiT进行部分/全部训练，同时不添加什么额外损失函数的情况，目前的这个可以设置早停,即在超过某个step之后，若当前的loss小于一个阈值，则停止
# """
# train_metaquery_wan.py
# =======================
# MetaQuery + Wan2.2 TI2V (Text+Image → Video) 联合训练脚本。

# ★ 核心思路:
#     复刻原始 MetaQuery 训练范式 —— 冻结 DiT，训练 Connector：
#     1. Qwen3-VL (冻结, 仅 MQ embeddings 可训练)
#     2. Connector: Qwen2Encoder(24L) + Linear + GELU + Linear + RMSNorm → dim=4096 (直接对齐 Wan)
#     3. to_wan_proj: 不再需要! Connector 直接输出 Wan text_dim=4096
#     4. Wan TI2V DiT (冻结): 接收 [MQ_tokens + T5_tokens] 作为 context
#     5. 计算 Flow Matching Loss → 反向传播更新 Connector + MQ Embeddings

# ★ 为什么选 WanTI2V (而非 I2V 或 Animate):
#     - TI2V 5B 是 Wan2.2 最新的 Text+Image→Video 统一模型
#     - 使用相同 DiT architecture 处理 t2v 和 i2v (model_type='ti2v')
#     - 不需要 CLIP encoder (I2V 需要 CLIP, Animate 需要 CLIP+Face+Pose)
#     - 参考图通过 VAE 编码后的 latent mask 注入 (最优雅的方式)
#     - 5B 参数量适中, 显存友好

# ★ 不需要 to_wan_proj:
#     直接让 Connector 输出 dim=4096 (Wan text_dim)
#     → 训练时 DiT 的 text_embedding 层直接消费 MQ 特征
#     → 无中间随机投影层, 梯度直接流过

# 用法:
#     # 单卡
#     python train_metaquery_wan.py --wan_checkpoint_dir /path/to/Wan2.2-TI2V-5B

#     # 多卡
#     torchrun --nproc_per_node=2 train_metaquery_wan.py
# """

# import os
# import sys
# import gc
# import json
# import math
# import time
# import argparse
# import random
# from pathlib import Path
# from datetime import datetime
# from contextlib import contextmanager
# from typing import Dict, Tuple, Any, List

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from PIL import Image
# from tqdm import tqdm

# # ── 路径设置 ─────────────────────────────────────────────────────────────────
# WAN_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(WAN_ROOT))

# METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
# sys.path.insert(0, METAQUERY_ROOT)


# # =============================================================================
# # 配置
# # =============================================================================
# def parse_args():
#     p = argparse.ArgumentParser(description="Train MetaQuery Connector for Wan TI2V")

#     # ── 模型路径 ──────────────────────────────────────────────────────────
#     p.add_argument("--wan_checkpoint_dir", type=str,
#                    default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
#                    help="Wan2.2 TI2V checkpoint 目录")
#     p.add_argument("--qwen3vl_model_id", type=str,
#                    default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking",
#                    help="Qwen3-VL 模型 ID 或本地路径")
#     p.add_argument("--output_dir", type=str,
#                    default="/home/liuzhirui/model/Wan2.2/metaquery_wan_ti2v_training",
#                    help="训练输出目录")

#     # ── 数据(OpenVid/WanVideoDataset) ───────────────────────────────────────
#     p.add_argument("--local_openvid_video_root", type=str, default=None,
#                    help="本地 OpenVid 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video")
#     p.add_argument("--local_openvid_csv_path", type=str, default=None,
#                    help="本地 OpenVid CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv")
#     p.add_argument("--local_openvid_limit", type=int, default=None,
#                    help="仅使用前N条本地匹配样本，默认使用全部已匹配样本")
#     p.add_argument("--local_openvid_hd_video_root", type=str, default=None,
#                    help="本地 OpenVid HD 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video_HD")
#     p.add_argument("--local_openvid_hd_csv_path", type=str, default=None,
#                    help="本地 OpenVid HD CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv")
#     p.add_argument("--local_openvid_hd_limit", type=int, default=None,
#                    help="仅使用前N条本地HD匹配样本，默认使用全部已匹配样本")
#     p.add_argument("--local_video_cache_dir", type=str, default=None,
#                    help="本地视频缓存目录（用于URL/字节解码缓存，可选）")
#     p.add_argument("--frame_num", type=int, default=41,
#                    help="每个视频片段采样帧数 (4n+1)")
#     p.add_argument("--max_area", type=int, default=480 * 832,
#                    help="视频最大面积 (宽×高)")
#     p.add_argument("--max_caption_tokens", type=int, default=512,
#                    help="超过该token长度的caption会被过滤")
#     p.add_argument("--caption_tokenizer_path", type=str, default="google/umt5-xxl",
#                    help="用于caption长度统计的tokenizer")
#     p.add_argument("--min_duration_sec", type=float, default=0.5,
#                    help="最短时长过滤阈值")
#     p.add_argument("--max_duration_sec", type=float, default=20.0,
#                    help="最长时长过滤阈值")

#     # ── 训练参数 ──────────────────────────────────────────────────────────
#     p.add_argument("--learning_rate", type=float, default=5e-5)
#     p.add_argument("--num_train_steps", type=int, default=5000)
#     p.add_argument("--warmup_steps", type=int, default=200)
#     p.add_argument(
#         "--lr_scheduler_type",
#         type=str,
#         default="cosine_with_warmup",
#         choices=["cosine_with_warmup", "constant_with_warmup", "warmup_hold_cooldown"],
#         help=(
#             "学习率调度器类型。"
#             "constant_with_warmup=warmup后恒定；"
#             "cosine_with_warmup=warmup后余弦衰减；"
#             "warmup_hold_cooldown=warmup线性升+中段恒定+末段线性降。"
#         ),
#     )
#     p.add_argument(
#         "--cooldown_steps",
#         type=int,
#         default=-1,
#         help="warmup_hold_cooldown 模式下末段降学习率步数。<0 表示使用 warmup_steps。",
#     )
#     p.add_argument(
#         "--lr_min_ratio",
#         type=float,
#         default=0.01,
#         help="cosine_with_warmup 模式下的最小学习率比例。",
#     )
#     p.add_argument("--batch_size", type=int, default=1)
#     p.add_argument("--gradient_accumulation_steps", type=int, default=4)
#     p.add_argument("--max_grad_norm", type=float, default=1.0)
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--save_steps", type=int, default=500)
#     p.add_argument("--log_steps", type=int, default=10)
#     p.add_argument("--enable_loss_early_stop", action="store_true", default=False,
#                    help="启用可选早停：当 step>=loss_early_stop_min_step 且 loss_step<loss_early_stop_threshold 时提前结束训练并保存 checkpoint。")
#     p.add_argument("--disable_loss_early_stop", action="store_false", dest="enable_loss_early_stop",
#                    help="关闭 loss 早停（默认）。")
#     p.add_argument("--loss_early_stop_min_step", type=int, default=800,
#                    help="loss 早停触发的最小 step（含）。")
#     p.add_argument("--loss_early_stop_threshold", type=float, default=0.25,
#                    help="loss 早停阈值：当 train/loss_step 小于该值时触发。")
#     p.add_argument("--log_every_step", action="store_true",
#                    help="每个优化 step 都打印详细训练日志")
#     p.add_argument("--wandb_log_every_step", action="store_true",
#                    help="每个优化 step 都写入 W&B（默认按 log_steps 写入）")
#     p.add_argument("--metrics_jsonl_path", type=str, default="",
#                    help="可选：将每步指标追加写入 JSONL 文件")
#     p.add_argument("--log_cuda_memory", action="store_true",
#                    help="记录并输出 CUDA 显存指标")
#     p.add_argument("--dataloader_num_workers", type=int, default=0)

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
#     p.add_argument("--mq_gradient_checkpointing", action="store_true",
#                    help="启用 MetaQuery 编码器梯度检查点，降低显存占用")
#     p.add_argument("--train_mq_input_embeddings", action="store_true", default=True,
#                    help="训练 Qwen 输入 embedding（默认开启）")
#     p.add_argument("--freeze_mq_input_embeddings", action="store_false", dest="train_mq_input_embeddings",
#                    help="冻结 Qwen 输入 embedding，仅训练 connector")
#     p.add_argument("--null_caption_prob", type=float, default=0.1)
#     p.add_argument("--null_image_prob", type=float, default=0.1)
#     p.add_argument("--enable_t5_alignment", action="store_true", default=True,
#                    help="启用 T5 对齐辅助损失（默认开启）：让 MQ 条件分布更接近 Wan 已适配的 T5 条件流形。")
#     p.add_argument("--disable_t5_alignment", action="store_false", dest="enable_t5_alignment",
#                    help="关闭 T5 对齐辅助损失，仅使用去噪主损失。")
#     p.add_argument(
#         "--t5_align_mode",
#         type=str,
#         default="gram_cka",
#         choices=["anchor", "gram_cka", "sinkhorn_ot"],
#         help=(
#             "T5 对齐方式。anchor=前K token 一一对齐；"
#             "gram_cka=基于 token 关系矩阵(Gram+CKA)的排列无关对齐；"
#             "sinkhorn_ot=基于 OT/Sinkhorn 的软匹配对齐。"
#         ),
#     )
#     p.add_argument("--t5_align_anchor_tokens", type=int, default=64,
#                    help="用于 T5 对齐的 anchor token 数（从 256 个 MQ token 前缀取）。")
#     p.add_argument("--lambda_t5_align_l2", type=float, default=0.2,
#                    help="T5 对齐主项权重：anchor 模式对应 token-L2；gram_cka 模式对应 Gram-L2；sinkhorn_ot 模式对应 OT 代价。")
#     p.add_argument("--lambda_t5_align_cos", type=float, default=0.1,
#                    help="T5 对齐次项权重：anchor 模式对应 token-cos；gram_cka 模式对应 CKA；sinkhorn_ot 模式默认忽略。")
#     p.add_argument("--lambda_t5_align_stats", type=float, default=0.02,
#                    help="T5 对齐的均值/方差统计损失权重。")
#     p.add_argument("--t5_align_ot_epsilon", type=float, default=0.05,
#                    help="Sinkhorn OT 熵正则温度 epsilon（越小越接近硬匹配）。")
#     p.add_argument("--t5_align_ot_iters", type=int, default=25,
#                    help="Sinkhorn OT 迭代次数。")
#     p.add_argument("--enable_mq_image_preserve", action="store_true", default=False,
#                    help="启用图像保持约束：有参考图时，MQ(cond) 与 MQ(text-only) 保持最小间隔。")
#     p.add_argument("--lambda_mq_image_preserve", type=float, default=0.02,
#                    help="图像保持约束权重。")
#     p.add_argument("--mq_image_preserve_margin", type=float, default=0.10,
#                    help="图像保持约束的最小间隔阈值（L2 均方根距离）。")
#     p.add_argument("--enable_wan_func_distill", action="store_true", default=False,
#                    help="启用 Wan 函数级蒸馏：约束 pred_mq(x_t,t) 贴近 pred_t5(x_t,t)。")
#     p.add_argument("--disable_wan_func_distill", action="store_false", dest="enable_wan_func_distill",
#                    help="关闭 Wan 函数级蒸馏。")
#     p.add_argument("--lambda_wan_func_distill", type=float, default=0.0,
#                    help="Wan 函数级蒸馏损失权重。")
#     p.add_argument(
#         "--wan_func_teacher_mode",
#         type=str,
#         default="t5_only",
#         choices=["t5_only", "t5_plus_mq"],
#         help="函数级蒸馏 teacher 条件。t5_only=仅 T5；t5_plus_mq=T5 与 MQ 拼接。",
#     )
#     p.add_argument("--train_video_conditioning_mode", type=str, default="legacy_t2v",
#                    choices=["legacy_t2v", "wan_animate_slot"],
#                    help=(
#                        "训练期视频条件注入方式: "
#                        "legacy_t2v=现有 TI2V 训练（可选首帧软锚定）；"
#                        "wan_animate_slot=参考图作为 preserved reference slot 注入，前缀 slot 不计入主损失"
#                    ))
#     p.add_argument("--train_animate_ref_frames", type=int, default=1,
#                    help="wan_animate_slot 模式下参考图保留帧数（像素帧数，内部按 VAE stride 映射到 latent slots）")
#     p.add_argument("--train_animate_temporal_frames", type=int, default=0,
#                    help="wan_animate_slot 模式下 temporal guidance 帧数（像素帧数；若无外部时序条件可保持 0）")
#     p.add_argument("--train_animate_conditional_frames", type=int, default=0,
#                    help="wan_animate_slot 模式下额外 conditional 帧数（像素帧数；无条件时保持 0，将注入全零 latent）")
#     p.add_argument("--train_animate_preserve_timestep_zero", action="store_true", default=True,
#                    help="wan_animate_slot: preserved prefix 对应 token 的 timestep 置 0（默认开启）")
#     p.add_argument("--train_animate_no_preserve_timestep_zero", action="store_false",
#                    dest="train_animate_preserve_timestep_zero",
#                    help="wan_animate_slot: 关闭 preserved prefix timestep=0")
#     p.add_argument("--train_animate_drop_prefix_loss", action="store_true", default=True,
#                    help="wan_animate_slot: 仅在 target frames 上计算损失，丢弃 reference/temporal/conditional prefix（默认开启）")
#     p.add_argument("--train_animate_no_drop_prefix_loss", action="store_false",
#                    dest="train_animate_drop_prefix_loss",
#                    help="wan_animate_slot: 不丢弃 prefix，整段都计入损失")
#     p.add_argument("--train_ref_anchor_mode", type=str, default="none",
#                    choices=["none", "animate_like", "mixed50"],
#                    help="训练时是否对 x_t 的首帧加入软参考锚定。none=保持原始 t2v；animate_like=全程启用软锚定；mixed50=约50%批次启用软锚定")
#     p.add_argument("--train_ref_anchor_alpha0", type=float, default=0.95,
#                    help="animate_like 模式的最大锚定强度 alpha0")
#     p.add_argument("--train_ref_anchor_warmup_ratio", type=float, default=0.35,
#                    help="animate_like 模式在高噪声区间启用锚定的占比（0~1）")

#     # ── 设备 ──────────────────────────────────────────────────────────────
#     p.add_argument("--dit_device", type=int, default=0,
#                    help="DiT + VAE + T5 所在 GPU")
#     p.add_argument("--encoder_device", type=int, default=1,
#                    help="Qwen3-VL + Connector 所在 GPU")
#     p.add_argument("--resume_mq_encoder_path", type=str, default=None,
#                    help="从已有mq_encoder权重继续训练")
#     p.add_argument("--t5_cpu", action="store_true",
#                    help="将Wan的T5文本编码器保留在CPU，显著降低GPU显存占用（速度会变慢）")
#     p.add_argument("--dit_fsdp", action="store_true",
#                    help="启用 Wan DiT 的 FSDP 参数分片，降低单卡模型权重占用")
#     p.add_argument("--t5_fsdp", action="store_true",
#                    help="启用 T5 编码器的 FSDP 参数分片")
#     p.add_argument("--use_sp", action="store_true",
#                    help="启用 sequence parallel（xDiT/USP 路径）")
#     p.add_argument("--no_init_on_cpu", action="store_true",
#                    help="关闭 init_on_cpu；默认开启以减小加载瞬时显存峰值")
#     p.add_argument("--convert_model_dtype", action="store_true",
#                    help="将 Wan DiT 参数显式转换到 config.param_dtype（仅非FSDP时生效）")
#     p.add_argument("--aggressive_empty_cache", action="store_true",
#                    help="每步训练后执行 torch.cuda.empty_cache()，缓解显存碎片")
#     p.add_argument("--wandb_enabled", action="store_true",
#                    help="启用 Weights & Biases 训练日志")
#     p.add_argument("--wandb_project", type=str, default="wan-metaquery",
#                    help="W&B project 名称")
#     p.add_argument("--wandb_entity", type=str, default="",
#                    help="W&B entity/team 名称")
#     p.add_argument("--wandb_run_name", type=str, default="",
#                    help="W&B run 名称, 留空自动生成")
#     p.add_argument("--wandb_tags", type=str, default="",
#                    help="W&B tags, 逗号分隔")
#     p.add_argument("--wandb_mode", type=str, default="online",
#                    choices=["online", "offline", "disabled"],
#                    help="W&B 模式")
#     p.add_argument("--wandb_api_key", type=str, default="",
#                    help="W&B API Key, 传入后会写入 WANDB_API_KEY 环境变量")
#     p.add_argument("--wandb_log_checkpoint", action="store_true",
#                    help="在 W&B 中记录 checkpoint 路径")
#     p.add_argument("--strict_freeze_check", action="store_true", default=True,
#                    help="启用严格冻结校验：若发现 Wan/T5/VAE 可训练或 optimizer 混入非 MQ 参数则中止")
#     p.add_argument("--no_strict_freeze_check", action="store_false", dest="strict_freeze_check",
#                    help="关闭严格冻结校验，仅打印告警")
#     p.add_argument(
#         "--wan_train_mode",
#         type=str,
#         default="auto",
#         choices=["auto", "frozen", "full", "cond_only"],
#         help=(
#             "Wan DiT 训练模式。auto=按显存策略自动在 full/cond_only 之间选择；"
#             "frozen=冻结；full=全量训练；cond_only=仅训 cross-attn + conditioning projection/AdaLN/modulation。"
#         ),
#     )
#     p.add_argument(
#         "--wan_auto_full_mem_gb",
#         type=float,
#         default=120.0,
#         help="auto 模式下，当 DiT 卡总显存 >= 该阈值时选择 full，否则选择 cond_only。",
#     )
#     p.add_argument(
#         "--wan_lr_ratio",
#         type=float,
#         default=1.0,
#         help="Wan 可训练参数学习率倍率（实际 lr = learning_rate * wan_lr_ratio）。",
#     )
#     p.add_argument(
#         "--wan_cond_name_pattern",
#         type=str,
#         default="",
#         help=(
#             "可选：自定义 cond_only 的参数名匹配关键字，逗号分隔。"
#             "为空时使用内置规则(cross_attn,text_embedding,time_projection,modulation,norm3,cross_attn_norm)。"
#         ),
#     )

#     return p.parse_args()


# def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
#     """兼容不同 torch 版本的安全加载。"""
#     try:
#         return torch.load(path, map_location=map_location, weights_only=True)
#     except TypeError:
#         return torch.load(path, map_location=map_location)


# def _extract_model_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
#     """从不同 checkpoint 负载中提取模型权重字典。"""
#     if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
#         return payload["model_state_dict"]
#     if isinstance(payload, dict):
#         tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
#         non_tensor_values = [v for v in payload.values() if not torch.is_tensor(v)]
#         if tensor_values and not non_tensor_values:
#             return payload
#     raise ValueError("无法从 checkpoint 提取 model_state_dict")


# def _to_cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     out = {}
#     for k, v in state_dict.items():
#         if torch.is_tensor(v):
#             out[k] = v.detach().cpu().contiguous()
#     return out


# def load_mq_encoder_state(path_or_dir: str, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, torch.Tensor], str]:
#     """
#     加载 MetaQuery encoder 权重:
#     - 支持传入单个文件: mq_encoder_full.pt / training_state.pt / model.safetensors
#     - 支持传入目录: 自动按优先级查找文件
#     """
#     path = Path(path_or_dir)
#     if not path.exists():
#         raise FileNotFoundError(f"checkpoint 路径不存在: {path}")

#     if path.is_dir():
#         candidates = [
#             path / "mq_encoder_full.pt",
#             path / "mq_encoder_full.safetensors",
#             path / "model.safetensors",
#             path / "pytorch_model.bin",
#             path / "training_state.pt",
#         ]
#         picked = next((p for p in candidates if p.exists()), None)
#         if picked is None:
#             raise FileNotFoundError(
#                 f"checkpoint 目录中未找到可加载权重文件: {path} "
#                 f"(expect one of {[c.name for c in candidates]})"
#             )
#         path = picked

#     suffix = path.suffix.lower()
#     if suffix == ".safetensors":
#         try:
#             from safetensors.torch import load_file
#         except Exception as e:
#             raise RuntimeError(
#                 f"检测到 safetensors 权重但未能导入 safetensors: {path}"
#             ) from e
#         state = load_file(str(path), device="cpu")
#     else:
#         payload = _safe_torch_load(path, map_location=map_location)
#         state = _extract_model_state_dict(payload)

#     return state, str(path.expanduser().resolve())


# def _write_json(path: Path, payload: Dict[str, Any]) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)


# def _to_jsonable(value: Any) -> Any:
#     if value is None or isinstance(value, (str, int, float, bool)):
#         return value
#     if isinstance(value, (list, tuple)):
#         return [_to_jsonable(v) for v in value]
#     if isinstance(value, dict):
#         return {str(k): _to_jsonable(v) for k, v in value.items()}
#     if isinstance(value, Path):
#         return str(value)
#     return str(value)


# def save_mq_checkpoint_bundle(
#     path: Path,
#     module: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LRScheduler,
#     step: int,
#     args: argparse.Namespace,
#     wan_module: nn.Module | None = None,
#     wan_train_mode: str = "frozen",
#     metrics_tail: List[Dict[str, Any]] | None = None,
#     metrics_summary: Dict[str, Any] | None = None,
#     extra_info: Dict[str, Any] | None = None,
# ) -> Dict[str, Any]:
#     """
#     保存“最小可用 + 兼容增强”的 checkpoint bundle。
#     兼容你当前推理脚本（mq_encoder_full.pt）并补充常见训练文件。
#     """
#     path = path.expanduser().resolve()
#     path.mkdir(parents=True, exist_ok=True)

#     full_state_cpu = _to_cpu_state_dict(module.state_dict())
#     name_to_param = dict(module.named_parameters())
#     trainable_state_cpu = {
#         name: tensor
#         for name, tensor in full_state_cpu.items()
#         if name_to_param.get(name, None) is not None
#         and name_to_param[name].requires_grad
#     }

#     torch.save(
#         {
#             "step": step,
#             "model_state_dict": trainable_state_cpu,
#             "optimizer_state_dict": optimizer.state_dict(),
#             "scheduler_state_dict": scheduler.state_dict(),
#         },
#         path / "training_state.pt",
#     )
#     torch.save(full_state_cpu, path / "mq_encoder_full.pt")
#     torch.save(trainable_state_cpu, path / "mq_encoder_trainable.pt")

#     wan_trainable_state_cpu: Dict[str, torch.Tensor] = {}
#     wan_trainable_param_count = 0
#     if wan_module is not None and isinstance(wan_module, nn.Module):
#         for name, p in wan_module.named_parameters():
#             if not p.requires_grad:
#                 continue
#             wan_trainable_state_cpu[name] = p.detach().cpu().contiguous()
#             wan_trainable_param_count += int(p.numel())
#     if wan_trainable_state_cpu:
#         torch.save(wan_trainable_state_cpu, path / "wan_dit_trainable.pt")

#     torch.save(vars(args), path / "training_args.bin")
#     _write_json(
#         path / "training_args.json",
#         {str(k): _to_jsonable(v) for k, v in vars(args).items()},
#     )
#     torch.save(optimizer.state_dict(), path / "optimizer.pt")
#     torch.save(scheduler.state_dict(), path / "scheduler.pt")

#     trainer_state = {
#         "global_step": int(step),
#         "checkpoint_format": "wan_metaquery_v2",
#         "has_full_pt": True,
#         "has_training_state": True,
#         "has_trainable_pt": True,
#         "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
#         "wan_train_mode": str(wan_train_mode),
#         "wan_trainable_param_count": int(wan_trainable_param_count),
#         "has_metrics_summary": bool(metrics_summary),
#         "metrics_tail_count": int(len(metrics_tail) if metrics_tail is not None else 0),
#     }
#     if extra_info:
#         trainer_state["extra_info"] = _to_jsonable(extra_info)
#     _write_json(path / "trainer_state.json", trainer_state)

#     config_payload = {
#         "format": "wan_metaquery_encoder",
#         "num_metaqueries": int(getattr(args, "num_metaqueries", 256)),
#         "connector_num_hidden_layers": int(getattr(args, "connector_num_hidden_layers", 24)),
#         "wan_text_dim": int(getattr(module, "wan_text_dim", 4096)),
#         "qwen3vl_model_id": str(getattr(args, "qwen3vl_model_id", "")),
#         "train_mq_input_embeddings": bool(getattr(args, "train_mq_input_embeddings", True)),
#         "wan_train_mode": str(wan_train_mode),
#         "wan_trainable_param_count": int(wan_trainable_param_count),
#         "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
#         "checkpoint_step": int(step),
#         "num_train_steps": int(getattr(args, "num_train_steps", 0)),
#         "save_steps": int(getattr(args, "save_steps", 0)),
#         "log_steps": int(getattr(args, "log_steps", 0)),
#         "enable_loss_early_stop": bool(getattr(args, "enable_loss_early_stop", False)),
#         "loss_early_stop_min_step": int(getattr(args, "loss_early_stop_min_step", 800)),
#         "loss_early_stop_threshold": float(getattr(args, "loss_early_stop_threshold", 0.25)),
#         "frame_num": int(getattr(args, "frame_num", 0)),
#         "max_area": int(getattr(args, "max_area", 0)),
#         "learning_rate": float(getattr(args, "learning_rate", 0.0)),
#         "warmup_steps": int(getattr(args, "warmup_steps", 0)),
#         "lr_scheduler_type": str(getattr(args, "lr_scheduler_type", "cosine_with_warmup")),
#         "cooldown_steps": int(getattr(args, "cooldown_steps", -1)),
#         "lr_min_ratio": float(getattr(args, "lr_min_ratio", 0.01)),
#         "enable_t5_alignment": bool(getattr(args, "enable_t5_alignment", True)),
#         "t5_align_mode": str(getattr(args, "t5_align_mode", "gram_cka")),
#         "t5_align_anchor_tokens": int(getattr(args, "t5_align_anchor_tokens", 64)),
#         "lambda_t5_align_l2": float(getattr(args, "lambda_t5_align_l2", 0.0)),
#         "lambda_t5_align_cos": float(getattr(args, "lambda_t5_align_cos", 0.0)),
#         "lambda_t5_align_stats": float(getattr(args, "lambda_t5_align_stats", 0.0)),
#         "t5_align_ot_epsilon": float(getattr(args, "t5_align_ot_epsilon", 0.05)),
#         "t5_align_ot_iters": int(getattr(args, "t5_align_ot_iters", 25)),
#         "enable_mq_image_preserve": bool(getattr(args, "enable_mq_image_preserve", False)),
#         "lambda_mq_image_preserve": float(getattr(args, "lambda_mq_image_preserve", 0.0)),
#         "mq_image_preserve_margin": float(getattr(args, "mq_image_preserve_margin", 0.0)),
#         "enable_wan_func_distill": bool(getattr(args, "enable_wan_func_distill", False)),
#         "lambda_wan_func_distill": float(getattr(args, "lambda_wan_func_distill", 0.0)),
#         "wan_func_teacher_mode": str(getattr(args, "wan_func_teacher_mode", "t5_only")),
#         "batch_size": int(getattr(args, "batch_size", 1)),
#         "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1)),
#         "null_caption_prob": float(getattr(args, "null_caption_prob", 0.0)),
#         "null_image_prob": float(getattr(args, "null_image_prob", 0.0)),
#         "wan_train_mode": str(getattr(args, "wan_train_mode", "auto")),
#         "wan_auto_full_mem_gb": float(getattr(args, "wan_auto_full_mem_gb", 120.0)),
#         "wan_lr_ratio": float(getattr(args, "wan_lr_ratio", 1.0)),
#         "wan_cond_name_pattern": str(getattr(args, "wan_cond_name_pattern", "")),
#     }
#     # 记录 MLLM embedding 行信息，便于推理期验证“新增 MQ token embedding 是否被保存/加载”。
#     try:
#         emb = module.mllm_model.mllm_backbone.get_input_embeddings()
#         if emb is not None and getattr(emb, "weight", None) is not None:
#             rows_total = int(emb.weight.shape[0])
#             rows_base = int(getattr(module.mllm_model, "num_embeddings", 0))
#             config_payload["mllm_embed_rows_total"] = rows_total
#             config_payload["mllm_embed_rows_base"] = rows_base
#             config_payload["mllm_embed_rows_added"] = max(rows_total - rows_base, 0)
#     except Exception:
#         pass
#     if extra_info:
#         config_payload["extra_info"] = _to_jsonable(extra_info)
#     _write_json(path / "config.json", config_payload)
#     if metrics_summary:
#         _write_json(path / "metrics_summary.json", {str(k): _to_jsonable(v) for k, v in metrics_summary.items()})
#     if metrics_tail is not None:
#         _write_json(
#             path / "metrics_tail.json",
#             {"records": [{str(k): _to_jsonable(v) for k, v in row.items()} for row in metrics_tail]},
#         )

#     try:
#         from safetensors.torch import save_file

#         save_file(full_state_cpu, str(path / "model.safetensors"))
#         save_file(trainable_state_cpu, str(path / "mq_encoder_trainable.safetensors"))
#         if wan_trainable_state_cpu:
#             save_file(wan_trainable_state_cpu, str(path / "wan_dit_trainable.safetensors"))
#     except Exception:
#         # safetensors 为增强项，不可用时保持兼容主流程
#         pass

#     # 兼容“latest”指针
#     try:
#         with open(path.parent / "latest", "w", encoding="utf-8") as f:
#             f.write(f"{path.name}\n")
#     except Exception:
#         pass

#     return {
#         "step": int(step),
#         "path": str(path),
#     }
# # =============================================================================
# # 数据集
# # =============================================================================
# try:
#     from train_connector_for_wan import WanVideoDataset as _DefaultWanVideoDataset
# except Exception:
#     _DefaultWanVideoDataset = None

# # 单一数据集入口：仅使用 WanVideoDataset。
# # 在 train_metaquery_wan_new.py 中可通过设置 base_ti2v.WanDatasetClass 进行覆写。
# WanDatasetClass = _DefaultWanVideoDataset


# # =============================================================================
# # Trainer
# # =============================================================================
# class MetaQueryWanTrainer:
#     """
#     MetaQuery + Wan TI2V 联合训练。

#     训练流程:
#         1. MetaQuery (Connector 可训练) → [B, 256, 4096]
#         2. T5 编码文本 → [B, text_len, 4096]
#         3. 拼接: [MQ + T5] → [B, 256+text_len, 4096]
#         4. VAE 编码视频帧 → latent
#         5. 采样噪声+时间步 → noisy_latent
#         6. 参考图 VAE 编码 → first frame mask
#         7. DiT (冻结) forward: 预测速度
#         8. Flow Matching Loss → 反向传播 Connector + MQ Embeddings
#     """

#     def __init__(self, args):
#         self.args = args
#         self.dev_dit = torch.device(f"cuda:{args.dit_device}")
#         self.dev_enc = torch.device(f"cuda:{args.encoder_device}")
#         self.wandb = None
#         self.wandb_run = None
#         self.is_main_process = self._is_main_process()
#         self._printed_grad_health = False
#         self._skipped_step_count = 0
#         self._oom_skip_count = 0
#         self._error_skip_count = 0
#         self._printed_context_inject_check = False
#         self._param_monitor = []
#         self._trainable_param_count = 0
#         self._init_trainable_norm = 0.0
#         self._init_param_sample_norm = 0.0
#         _metrics_jsonl = (args.metrics_jsonl_path or "").strip()
#         self._metrics_jsonl_path = str(Path(_metrics_jsonl).expanduser().resolve()) if _metrics_jsonl else ""
#         self._metrics_history: List[Dict[str, Any]] = []
#         self._train_before_checkpoint_path = ""
#         self._train_wall_start = 0.0
#         self._last_train_ref_anchor_alpha_mean = 0.0
#         self._last_train_ref_anchor_applied = 0
#         self._last_train_ref_anchor_effective_mode = "none"
#         self._train_ref_anchor_mixed_counter = 0
#         self._current_train_ref_anchor_mode = "none"
#         self._last_train_video_conditioning_mode = "mq_only"
#         self._last_train_prefix_latent_slots = 0
#         self._last_train_target_latent_slots = 0
#         self._last_train_prefix_loss_dropped = 0
#         self._last_loss_denoise = 0.0
#         self._last_loss_aux_align_total = 0.0
#         self._last_loss_aux_t5_l2 = 0.0
#         self._last_loss_aux_t5_cos = 0.0
#         self._last_loss_aux_t5_stats = 0.0
#         self._last_loss_aux_t5_gram = 0.0
#         self._last_loss_aux_t5_cka = 0.0
#         self._last_loss_aux_t5_ot = 0.0
#         self._last_loss_aux_image_preserve = 0.0
#         self._last_loss_aux_wan_func = 0.0
#         self._effective_wan_train_mode = "frozen"
#         self._wan_trainable_names: List[str] = []
#         self._wan_trainable_params_cache: List[torch.nn.Parameter] = []

#         print("\n" + "=" * 60)
#         print("  MetaQuery + Wan TI2V 联合训练")
#         print("=" * 60)
#         print(f"  DiT 设备       : {self.dev_dit}")
#         print(f"  Encoder 设备   : {self.dev_enc}")
#         print(f"  学习率         : {args.learning_rate}")
#         print(f"  LR 调度器      : {args.lr_scheduler_type}")
#         print(f"  Cooldown 步数  : {args.cooldown_steps} (-1 表示使用 warmup_steps)")
#         print(f"  训练步数       : {args.num_train_steps}")
#         print(f"  有效 batch     : {args.batch_size * args.gradient_accumulation_steps}")
#         print(
#             f"  Loss 早停       : enabled={int(bool(args.enable_loss_early_stop))} "
#             f"min_step={args.loss_early_stop_min_step} threshold={args.loss_early_stop_threshold}"
#         )
#         print(
#             f"  Wan 训练模式    : req={args.wan_train_mode} auto_full_mem_gb={args.wan_auto_full_mem_gb} "
#             f"wan_lr_ratio={args.wan_lr_ratio}"
#         )
#         print(
#             f"  T5 对齐(已禁用) : cfg_enabled={int(bool(args.enable_t5_alignment))} "
#             f"mode={args.t5_align_mode} "
#             f"anchor={args.t5_align_anchor_tokens} "
#             f"l2={args.lambda_t5_align_l2} cos={args.lambda_t5_align_cos} stats={args.lambda_t5_align_stats} "
#             f"ot_eps={args.t5_align_ot_epsilon} ot_iters={args.t5_align_ot_iters}"
#         )
#         print(
#             f"  图像保持(已禁用): cfg_enabled={int(bool(args.enable_mq_image_preserve))} "
#             f"lambda={args.lambda_mq_image_preserve} margin={args.mq_image_preserve_margin}"
#         )
#         print(
#             f"  函数蒸馏(已禁用): cfg_enabled={int(bool(args.enable_wan_func_distill))} "
#             f"lambda={args.lambda_wan_func_distill} teacher={args.wan_func_teacher_mode}"
#         )
#         print("  额外损失开关   : 当前版本固定仅使用 denoise MSE（其余辅助损失已禁用）")
#         print("=" * 60)

#         self._load_models()
#         self._log_runtime_topology()
#         self._setup_optimizer()
#         self._audit_runtime_trainability(stage="init")
#         self._init_trainability_monitor()
#         self._init_wandb()

#     def _is_main_process(self):
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             return torch.distributed.get_rank() == 0
#         rank_env = os.environ.get("RANK")
#         if rank_env is None:
#             return True
#         return int(rank_env) == 0

#     def _mq_encoder_module(self):
#         return self.mq_encoder.module if hasattr(self.mq_encoder, "module") else self.mq_encoder

#     def _mq_trainable_params(self):
#         module = self._mq_encoder_module()
#         if hasattr(module, "get_trainable_params"):
#             return module.get_trainable_params()
#         return [p for p in module.parameters() if p.requires_grad]

#     def _resolve_wan_train_mode(self) -> str:
#         mode = str(getattr(self.args, "wan_train_mode", "auto")).strip().lower()
#         if mode != "auto":
#             return mode
#         total_gb = 0.0
#         if self.dev_dit.type == "cuda" and torch.cuda.is_available():
#             try:
#                 props = torch.cuda.get_device_properties(self.dev_dit)
#                 total_gb = float(props.total_memory) / float(1024 ** 3)
#             except Exception:
#                 total_gb = 0.0
#         threshold = float(getattr(self.args, "wan_auto_full_mem_gb", 120.0))
#         return "full" if total_gb >= threshold else "cond_only"

#     def _wan_cond_keywords(self) -> List[str]:
#         custom = str(getattr(self.args, "wan_cond_name_pattern", "")).strip()
#         if custom:
#             return [k.strip().lower() for k in custom.split(",") if k.strip()]
#         return [
#             "cross_attn",
#             "cross-attn",
#             "crossattention",
#             "cross_attention",
#             "text_embedding",
#             "time_projection",
#             "modulation",
#             "cross_attn_norm",
#             "norm3",
#         ]

#     def _configure_wan_trainable_params(self) -> None:
#         wan_model = getattr(self.wan, "model", None)
#         if wan_model is None:
#             self._effective_wan_train_mode = "frozen"
#             self._wan_trainable_names = []
#             self._wan_trainable_params_cache = []
#             return

#         # 先全冻结，再按模式打开。
#         self._force_freeze(wan_model)
#         mode = self._resolve_wan_train_mode()
#         self._effective_wan_train_mode = mode
#         selected_names: List[str] = []
#         selected_params: List[torch.nn.Parameter] = []

#         if mode == "full":
#             for name, p in wan_model.named_parameters():
#                 p.requires_grad_(True)
#                 selected_names.append(name)
#                 selected_params.append(p)
#         elif mode == "cond_only":
#             kws = self._wan_cond_keywords()
#             for name, p in wan_model.named_parameters():
#                 lname = name.lower()
#                 if any(kw in lname for kw in kws):
#                     p.requires_grad_(True)
#                     selected_names.append(name)
#                     selected_params.append(p)
#         elif mode == "frozen":
#             pass
#         else:
#             raise ValueError(f"Unknown --wan_train_mode: {mode}")

#         self._wan_trainable_names = selected_names
#         self._wan_trainable_params_cache = selected_params
#         if selected_params:
#             wan_model.train()
#         else:
#             wan_model.eval()

#         if self.is_main_process:
#             total = sum(int(p.numel()) for p in selected_params)
#             print(
#                 f"[WAN-TRAIN] requested={self.args.wan_train_mode} effective={mode} "
#                 f"trainable_tensors={len(selected_params)} trainable_params={total:,}"
#             )
#             if mode == "cond_only":
#                 kws = self._wan_cond_keywords()
#                 preview = ", ".join(kws[:10])
#                 print(f"[WAN-TRAIN] cond_only keywords={preview}")
#             if selected_names:
#                 preview = ", ".join(selected_names[:8])
#                 more = "" if len(selected_names) <= 8 else f" ... +{len(selected_names)-8}"
#                 print(f"[WAN-TRAIN] selected preview: {preview}{more}")

#     def _wan_trainable_params(self) -> List[torch.nn.Parameter]:
#         return list(self._wan_trainable_params_cache)

#     def _all_trainable_params(self) -> List[torch.nn.Parameter]:
#         out: List[torch.nn.Parameter] = []
#         seen = set()
#         for p in self._mq_trainable_params():
#             if id(p) not in seen:
#                 out.append(p)
#                 seen.add(id(p))
#         for p in self._wan_trainable_params():
#             if id(p) not in seen:
#                 out.append(p)
#                 seen.add(id(p))
#         return out

#     @staticmethod
#     def _module_param_stats(module: nn.Module | None) -> Dict[str, int]:
#         total = 0
#         trainable = 0
#         if module is None or not isinstance(module, nn.Module):
#             return {"total": 0, "trainable": 0}
#         for p in module.parameters():
#             n = int(p.numel())
#             total += n
#             if p.requires_grad:
#                 trainable += n
#         return {"total": total, "trainable": trainable}

#     @staticmethod
#     def _named_param_id_map(module: nn.Module | None, prefix: str) -> Dict[int, str]:
#         out: Dict[int, str] = {}
#         if module is None or not isinstance(module, nn.Module):
#             return out
#         for name, p in module.named_parameters():
#             out[id(p)] = f"{prefix}.{name}"
#         return out

#     @staticmethod
#     def _force_freeze(module: nn.Module | None) -> None:
#         if module is None or not isinstance(module, nn.Module):
#             return
#         try:
#             module.eval()
#         except Exception:
#             pass
#         try:
#             module.requires_grad_(False)
#         except Exception:
#             for p in module.parameters():
#                 p.requires_grad_(False)

#     def _log_runtime_topology(self) -> None:
#         if not self.is_main_process:
#             return
#         args = self.args
#         same_gpu = (self.dev_dit == self.dev_enc)
#         print(
#             "[AUDIT][TOPO] "
#             f"dit_device={self.dev_dit} encoder_device={self.dev_enc} same_gpu={same_gpu} "
#             f"t5_cpu={args.t5_cpu} t5_fsdp={args.t5_fsdp} dit_fsdp={args.dit_fsdp} use_sp={args.use_sp} "
#             f"num_metaqueries={args.num_metaqueries} aug_text_len={getattr(self, '_aug_text_len', -1)} "
#             f"wan_mode_effective={getattr(self, '_effective_wan_train_mode', 'frozen')}"
#         )
#         if same_gpu:
#             print("[AUDIT][TOPO][WARN] DiT 与 Qwen/Connector 在同一 GPU，显存峰值风险较高。")
#         if (not args.t5_cpu) and (not args.t5_fsdp):
#             print("[AUDIT][TOPO] T5 文本编码器会在 DiT 卡上参与前向（no_grad）。")
#         try:
#             from wan.modules import attention as _attn
#             fa2 = bool(getattr(_attn, "FLASH_ATTN_2_AVAILABLE", False))
#             fa3 = bool(getattr(_attn, "FLASH_ATTN_3_AVAILABLE", False))
#             force_sdpa = bool(getattr(_attn, "_FORCE_SDPA", False))
#             print(
#                 "[AUDIT][ATTN] "
#                 f"flash_attn2={fa2} flash_attn3={fa3} force_sdpa={force_sdpa}"
#             )
#         except Exception as e:
#             print(f"[AUDIT][ATTN][WARN] 无法读取 attention backend 信息: {e}")

#     def _audit_runtime_trainability(self, stage: str = "runtime", strict: bool | None = None) -> None:
#         args = self.args
#         if strict is None:
#             strict = bool(getattr(args, "strict_freeze_check", True))

#         wan_mode = str(getattr(self, "_effective_wan_train_mode", "frozen"))
#         # Wan 是否冻结由 wan_train_mode 决定；T5/VAE 始终冻结。
#         t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
#         self._force_freeze(t5_model)
#         vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
#         if vae_model is None:
#             vae_model = getattr(self.wan, "vae", None)
#         self._force_freeze(vae_model)

#         stats_wan = self._module_param_stats(getattr(self.wan, "model", None))
#         stats_t5 = self._module_param_stats(t5_model)
#         stats_vae = self._module_param_stats(vae_model)

#         mq_module = self._mq_encoder_module()
#         stats_mq = self._module_param_stats(mq_module)
#         mq_trainable_params = self._mq_trainable_params()
#         wan_trainable_params = self._wan_trainable_params()
#         mq_trainable_ids = {id(p) for p in mq_trainable_params}
#         wan_trainable_ids = {id(p) for p in wan_trainable_params}
#         allowed_trainable_ids = mq_trainable_ids | wan_trainable_ids
#         emb_trainable = 0
#         emb_rows_total = 0
#         emb_rows_base = 0
#         emb_rows_added = 0
#         emb_hidden = 0
#         try:
#             backbone = mq_module.mllm_model.mllm_backbone
#             emb = backbone.get_input_embeddings()
#             if emb is not None and getattr(emb, "weight", None) is not None:
#                 w = emb.weight
#                 emb_rows_total = int(w.shape[0])
#                 emb_hidden = int(w.shape[1]) if w.ndim >= 2 else 0
#                 emb_rows_base = int(getattr(mq_module.mllm_model, "num_embeddings", 0))
#                 emb_rows_added = max(emb_rows_total - emb_rows_base, 0)
#                 if bool(w.requires_grad):
#                     emb_trainable = int(w.numel())
#         except Exception:
#             pass

#         opt_params = []
#         for g in self.optimizer.param_groups:
#             opt_params.extend(g.get("params", []))
#         opt_ids = [id(p) for p in opt_params]
#         opt_id_set = set(opt_ids)

#         outside_ids = [pid for pid in opt_ids if pid not in allowed_trainable_ids]
#         missing_mq_ids = [pid for pid in mq_trainable_ids if pid not in opt_id_set]
#         missing_wan_ids = [pid for pid in wan_trainable_ids if pid not in opt_id_set]
#         duplicate_count = max(len(opt_ids) - len(opt_id_set), 0)

#         name_map: Dict[int, str] = {}
#         name_map.update(self._named_param_id_map(getattr(self.wan, "model", None), "wan.model"))
#         name_map.update(self._named_param_id_map(t5_model, "wan.text_encoder.model"))
#         name_map.update(self._named_param_id_map(vae_model, "wan.vae.model"))
#         name_map.update(self._named_param_id_map(mq_module, "mq_encoder"))

#         unexpected_mq_names = []
#         for name, p in mq_module.named_parameters():
#             if not p.requires_grad:
#                 continue
#             lower = name.lower()
#             if ("connector" in lower) or ("embed" in lower):
#                 continue
#             unexpected_mq_names.append(name)

#         if self.is_main_process:
#             print(
#                 f"[AUDIT][FREEZE][{stage}] "
#                 f"wan_trainable={stats_wan['trainable']:,}/{stats_wan['total']:,} "
#                 f"t5_trainable={stats_t5['trainable']:,}/{stats_t5['total']:,} "
#                 f"vae_trainable={stats_vae['trainable']:,}/{stats_vae['total']:,} "
#                 f"mq_trainable={stats_mq['trainable']:,}/{stats_mq['total']:,} "
#                 f"mq_trainable_tensors={len(mq_trainable_params)} "
#                 f"wan_mode={wan_mode} wan_trainable_tensors={len(wan_trainable_params)}"
#             )
#             print(
#                 f"[AUDIT][OPT][{stage}] "
#                 f"optimizer_params={len(opt_ids)} "
#                 f"outside_allowed={len(outside_ids)} missing_mq={len(missing_mq_ids)} "
#                 f"missing_wan={len(missing_wan_ids)} duplicates={duplicate_count}"
#             )
#             print(
#                 f"[AUDIT][MQ-EMB][{stage}] "
#                 f"enabled={int(bool(args.train_mq_input_embeddings))} "
#                 f"embed_trainable={emb_trainable:,} "
#                 f"rows_total={emb_rows_total} rows_base={emb_rows_base} rows_added={emb_rows_added} "
#                 f"hidden={emb_hidden} expected_added≈num_metaqueries+2={int(args.num_metaqueries) + 2}"
#             )
#             if unexpected_mq_names:
#                 preview = ", ".join(unexpected_mq_names[:6])
#                 more = "" if len(unexpected_mq_names) <= 6 else f" ... +{len(unexpected_mq_names)-6}"
#                 print(
#                     "[AUDIT][MQ][WARN] 检测到非 connector/embed 命名的可训练参数: "
#                     f"{preview}{more}"
#                 )

#         errors = []
#         if wan_mode == "frozen" and stats_wan["trainable"] > 0:
#             errors.append(f"Wan DiT 期望冻结但仍有可训练参数: {stats_wan['trainable']}")
#         if wan_mode != "frozen" and len(wan_trainable_ids) == 0:
#             errors.append(f"Wan DiT 训练模式={wan_mode} 但未选中可训练参数")
#         if stats_t5["trainable"] > 0:
#             errors.append(f"Wan T5 仍有可训练参数: {stats_t5['trainable']}")
#         if stats_vae["trainable"] > 0:
#             errors.append(f"Wan VAE 仍有可训练参数: {stats_vae['trainable']}")
#         if len(mq_trainable_ids) == 0:
#             errors.append("MQ encoder 无可训练参数")
#         if bool(args.train_mq_input_embeddings) and emb_trainable <= 0:
#             errors.append("设置了 train_mq_input_embeddings，但输入 embedding 未开启训练")
#         if (not bool(args.train_mq_input_embeddings)) and emb_trainable > 0:
#             errors.append("设置了 freeze_mq_input_embeddings，但输入 embedding 仍可训练")
#         if outside_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in outside_ids[:8]]
#             errors.append(f"optimizer 含非允许参数(MQ+Wan): {names}")
#         if missing_mq_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_mq_ids[:8]]
#             errors.append(f"部分 MQ 可训练参数未进 optimizer: {names}")
#         if missing_wan_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_wan_ids[:8]]
#             errors.append(f"部分 Wan 可训练参数未进 optimizer: {names}")
#         if duplicate_count > 0:
#             errors.append(f"optimizer 参数重复引用: {duplicate_count}")
#         if wan_mode != "frozen" and torch.distributed.is_available() and torch.distributed.is_initialized():
#             ws = int(torch.distributed.get_world_size())
#             if ws > 1:
#                 errors.append("当前仅实现 MQ-encoder 的 DDP 包装；Wan 可训练模式请使用单进程训练（WORLD_SIZE=1）")

#         if errors:
#             msg = " | ".join(errors)
#             if strict:
#                 raise RuntimeError(f"[AUDIT][FAIL][{stage}] {msg}")
#             if self.is_main_process:
#                 print(f"[AUDIT][WARN][{stage}] {msg}")

#     def post_wrap_ddp_audit(self) -> None:
#         # DDP 包装后再做一次 optimizer 与 trainable 参数一致性检查
#         if not hasattr(self.mq_encoder, "module"):
#             return
#         self._audit_runtime_trainability(stage="post_ddp")

#     def _log_grad_health_once(self):
#         if self._printed_grad_health:
#             return
#         module = self._mq_encoder_module()
#         connector_has_grad = False
#         mq_embed_has_grad = False
#         wan_has_grad = False
#         connector_grad_norm = 0.0
#         mq_embed_grad_norm = 0.0
#         wan_grad_norm = 0.0
#         mq_embed_added_grad_norm = 0.0
#         mq_embed_base_grad_norm = 0.0
#         mq_embed_boundary_grad_norm = 0.0
#         mq_embed_query_grad_norm = 0.0
#         try:
#             for _, p in module.mllm_model.connector.named_parameters():
#                 if p.grad is not None:
#                     connector_has_grad = True
#                     connector_grad_norm = float(p.grad.detach().float().norm().item())
#                     break
#             emb = module.mllm_model.mllm_backbone.get_input_embeddings()
#             if emb is not None and getattr(emb, "weight", None) is not None and emb.weight.grad is not None:
#                 mq_embed_has_grad = True
#                 g = emb.weight.grad.detach().float()
#                 mq_embed_grad_norm = float(g.norm().item())
#                 base_rows = int(getattr(module.mllm_model, "num_embeddings", 0))
#                 if g.ndim >= 2 and 0 < base_rows < int(g.shape[0]):
#                     mq_embed_base_grad_norm = float(g[:base_rows].norm().item())
#                     mq_embed_added_grad_norm = float(g[base_rows:].norm().item())
#                     boundary_end = min(base_rows + 2, int(g.shape[0]))
#                     query_end = min(boundary_end + int(self.args.num_metaqueries), int(g.shape[0]))
#                     if boundary_end > base_rows:
#                         mq_embed_boundary_grad_norm = float(g[base_rows:boundary_end].norm().item())
#                     if query_end > boundary_end:
#                         mq_embed_query_grad_norm = float(g[boundary_end:query_end].norm().item())
#         except Exception:
#             pass
#         try:
#             for p in self._wan_trainable_params():
#                 if p.grad is not None:
#                     wan_has_grad = True
#                     wan_grad_norm = float(p.grad.detach().float().norm().item())
#                     break
#         except Exception:
#             pass
#         print(
#             "[GRAD-CHECK] "
#             f"connector_has_grad={connector_has_grad} connector_grad_norm={connector_grad_norm:.4e} "
#             f"mq_embed_has_grad={mq_embed_has_grad} mq_embed_grad_norm={mq_embed_grad_norm:.4e} "
#             f"wan_has_grad={wan_has_grad} wan_grad_norm={wan_grad_norm:.4e} "
#             f"mq_embed_added_grad_norm={mq_embed_added_grad_norm:.4e} "
#             f"mq_embed_base_grad_norm={mq_embed_base_grad_norm:.4e} "
#             f"mq_embed_boundary_grad_norm={mq_embed_boundary_grad_norm:.4e} "
#             f"mq_embed_query_grad_norm={mq_embed_query_grad_norm:.4e}"
#         )
#         self._printed_grad_health = True

#     def _verify_train_context_injection_once(
#         self,
#         mq_feat: torch.Tensor,
#         aug_feat: torch.Tensor,
#     ) -> None:
#         if self._printed_context_inject_check:
#             return
#         mq_len = int(mq_feat.shape[0])
#         aug_len = int(aug_feat.shape[0])
#         if aug_len != mq_len:
#             raise RuntimeError(
#                 f"[VERIFY][TRAIN] MQ-only context 长度异常: aug={aug_len}, mq={mq_len}"
#             )
#         mq_ok = torch.allclose(
#             aug_feat.float(),
#             mq_feat.float(),
#             atol=1e-3,
#             rtol=1e-3,
#         )
#         if not mq_ok:
#             raise RuntimeError("[VERIFY][TRAIN] MQ-only context 未正确注入 Wan context")
#         if aug_len > self._aug_text_len:
#             raise RuntimeError(
#                 f"[VERIFY][TRAIN] aug_len 超出 text_len: aug={aug_len}, text_len={self._aug_text_len}"
#             )
#         print(
#             "[VERIFY][TRAIN] context 注入检查通过: "
#             f"mq_tokens={mq_len} aug_tokens={aug_len} model_text_len={self._aug_text_len}"
#         )
#         self._printed_context_inject_check = True

#     def _init_trainability_monitor(self):
#         self._param_monitor = []
#         total_sq = 0.0
#         sample_sq = 0.0
#         total_params = 0
#         named_params: List[Tuple[str, torch.nn.Parameter]] = []
#         mq_module = self._mq_encoder_module()
#         named_params.extend((f"mq_encoder.{n}", p) for n, p in mq_module.named_parameters() if p.requires_grad)
#         wan_model = getattr(self.wan, "model", None)
#         if isinstance(wan_model, nn.Module):
#             named_params.extend((f"wan.model.{n}", p) for n, p in wan_model.named_parameters() if p.requires_grad)
#         for name, p in named_params:
#             data = p.detach().float().view(-1)
#             numel = int(data.numel())
#             if numel <= 0:
#                 continue
#             sample_k = min(8, numel)
#             if sample_k == 1:
#                 idx = torch.zeros(1, dtype=torch.long)
#             else:
#                 idx = torch.linspace(0, numel - 1, steps=sample_k, dtype=torch.long)
#             init_vals = data.index_select(0, idx.to(data.device)).cpu()
#             self._param_monitor.append((name, p, idx.cpu(), init_vals))
#             total_sq += float(torch.sum(data * data).item())
#             sample_sq += float(torch.sum(init_vals * init_vals).item())
#             total_params += numel
#         self._trainable_param_count = total_params
#         self._init_trainable_norm = math.sqrt(max(total_sq, 0.0))
#         self._init_param_sample_norm = math.sqrt(max(sample_sq, 0.0))
#         if self.is_main_process:
#             print(
#                 "[VERIFY][TRAIN-INIT] "
#                 f"trainable_params={self._trainable_param_count:,} "
#                 f"init_param_norm={self._init_trainable_norm:.6f} "
#                 f"monitor_tensors={len(self._param_monitor)}"
#             )

#     def _collect_trainability_metrics(self):
#         sample_abs_sum = 0.0
#         sample_l2_sum = 0.0
#         sample_cur_sq_sum = 0.0
#         sample_count = 0
#         with torch.no_grad():
#             for _, p, idx_cpu, init_vals_cpu in self._param_monitor:
#                 data = p.detach().float().view(-1)
#                 idx = idx_cpu.to(data.device)
#                 now_vals = data.index_select(0, idx).cpu()
#                 diff = now_vals - init_vals_cpu
#                 sample_abs_sum += float(diff.abs().sum().item())
#                 sample_l2_sum += float(torch.sum(diff * diff).item())
#                 sample_cur_sq_sum += float(torch.sum(now_vals * now_vals).item())
#                 sample_count += int(diff.numel())
#         cur_sample_norm = math.sqrt(max(sample_cur_sq_sum, 0.0))
#         init_sample_norm = max(self._init_param_sample_norm, 1e-12)
#         return {
#             "train/param_sample_norm": float(cur_sample_norm),
#             "train/param_sample_norm_delta_ratio": float(abs(cur_sample_norm - self._init_param_sample_norm) / init_sample_norm),
#             "train/param_sample_abs_delta_mean": float(sample_abs_sum / max(sample_count, 1)),
#             "train/param_sample_l2_delta": float(math.sqrt(max(sample_l2_sum, 0.0))),
#             "train/trainable_param_count": int(self._trainable_param_count),
#         }

#     def _collect_cuda_memory_metrics(self):
#         if not (torch.cuda.is_available() and self.args.log_cuda_memory):
#             return {}
#         dit_idx = self.dev_dit.index if self.dev_dit.type == "cuda" else None
#         enc_idx = self.dev_enc.index if self.dev_enc.type == "cuda" else None

#         def _mem(prefix, dev_idx):
#             if dev_idx is None:
#                 return {}
#             return {
#                 f"train/cuda_{prefix}_alloc_mb": float(torch.cuda.memory_allocated(dev_idx) / 1024 / 1024),
#                 f"train/cuda_{prefix}_reserved_mb": float(torch.cuda.memory_reserved(dev_idx) / 1024 / 1024),
#                 f"train/cuda_{prefix}_max_alloc_mb": float(torch.cuda.max_memory_allocated(dev_idx) / 1024 / 1024),
#             }

#         metrics = {}
#         metrics.update(_mem("dit", dit_idx))
#         metrics.update(_mem("enc", enc_idx))
#         return metrics

#     def _append_metrics_jsonl(self, metrics):
#         if not self.is_main_process:
#             return
#         if not self._metrics_jsonl_path:
#             return
#         try:
#             path = Path(self._metrics_jsonl_path).expanduser().resolve()
#             path.parent.mkdir(parents=True, exist_ok=True)
#             with path.open("a", encoding="utf-8") as f:
#                 f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
#         except Exception as e:
#             print(f"[WARN] 写入 metrics_jsonl 失败: {e}")

#     def _record_metrics(self, metrics: Dict[str, Any]) -> None:
#         keep_keys = [
#             "train/step",
#             "train/loss_step",
#             "train/loss_denoise",
#             "train/loss_align_total",
#             "train/loss_align_t5_l2",
#             "train/loss_align_t5_cos",
#             "train/loss_align_t5_stats",
#             "train/loss_align_t5_gram",
#             "train/loss_align_t5_cka",
#             "train/loss_align_t5_ot",
#             "train/loss_align_img_preserve",
#             "train/loss_align_wan_func",
#             "train/loss_ema",
#             "train/lr",
#             "train/grad_norm",
#             "train/step_time_sec",
#             "train/samples_per_sec",
#             "train/param_sample_abs_delta_mean",
#             "train/param_sample_l2_delta",
#             "train/param_sample_norm_delta_ratio",
#             "train/skipped_step_count",
#             "train/oom_skip_count",
#             "train/error_skip_count",
#         ]
#         row = {k: metrics[k] for k in keep_keys if k in metrics}
#         self._metrics_history.append(row)

#     def _build_metrics_summary(self, step: int) -> Dict[str, Any]:
#         summary: Dict[str, Any] = {
#             "current_step": int(step),
#             "logged_steps": int(len(self._metrics_history)),
#             "metrics_jsonl_path": self._metrics_jsonl_path,
#             "skipped_step_count": int(self._skipped_step_count),
#             "oom_skip_count": int(self._oom_skip_count),
#             "error_skip_count": int(self._error_skip_count),
#         }
#         if self._metrics_history:
#             last = self._metrics_history[-1]
#             loss_vals = [float(m.get("train/loss_step", 0.0)) for m in self._metrics_history if "train/loss_step" in m]
#             step_time_vals = [float(m.get("train/step_time_sec", 0.0)) for m in self._metrics_history if "train/step_time_sec" in m]
#             sps_vals = [float(m.get("train/samples_per_sec", 0.0)) for m in self._metrics_history if "train/samples_per_sec" in m]
#             summary.update(
#                 {
#                     "step_first": int(self._metrics_history[0].get("train/step", 0)),
#                     "step_last": int(last.get("train/step", 0)),
#                     "loss_last": float(last.get("train/loss_step", 0.0)),
#                     "loss_ema_last": float(last.get("train/loss_ema", 0.0)),
#                     "lr_last": float(last.get("train/lr", 0.0)),
#                     "grad_norm_last": float(last.get("train/grad_norm", 0.0)),
#                     "loss_min": float(min(loss_vals) if loss_vals else 0.0),
#                     "loss_max": float(max(loss_vals) if loss_vals else 0.0),
#                     "step_time_sec_avg": float(sum(step_time_vals) / max(len(step_time_vals), 1)),
#                     "samples_per_sec_avg": float(sum(sps_vals) / max(len(sps_vals), 1)),
#                 }
#             )
#         if self._train_wall_start > 0:
#             summary["wall_time_sec"] = float(max(time.perf_counter() - self._train_wall_start, 0.0))
#         return summary

#     def _write_training_chain_manifest(self, output_dir: Path, final_checkpoint_path: str, final_step: int) -> None:
#         if not self.is_main_process:
#             return
#         output_dir = output_dir.expanduser().resolve()
#         payload = {
#             "before_checkpoint_path": self._train_before_checkpoint_path,
#             "final_checkpoint_path": str(Path(final_checkpoint_path).expanduser().resolve()),
#             "metrics_jsonl_path": self._metrics_jsonl_path,
#             "args": {str(k): _to_jsonable(v) for k, v in vars(self.args).items()},
#             "metrics_summary": self._build_metrics_summary(step=final_step),
#         }
#         _write_json(output_dir / "training_chain_manifest.json", payload)

#     def _wandb_config(self):
#         args = self.args
#         return {
#             "task": "wan_ti2v",
#             "learning_rate": args.learning_rate,
#             "num_train_steps": args.num_train_steps,
#             "warmup_steps": args.warmup_steps,
#             "lr_scheduler_type": args.lr_scheduler_type,
#             "cooldown_steps": args.cooldown_steps,
#             "lr_min_ratio": args.lr_min_ratio,
#             "enable_t5_alignment": args.enable_t5_alignment,
#             "t5_align_mode": args.t5_align_mode,
#             "t5_align_anchor_tokens": args.t5_align_anchor_tokens,
#             "lambda_t5_align_l2": args.lambda_t5_align_l2,
#             "lambda_t5_align_cos": args.lambda_t5_align_cos,
#             "lambda_t5_align_stats": args.lambda_t5_align_stats,
#             "t5_align_ot_epsilon": args.t5_align_ot_epsilon,
#             "t5_align_ot_iters": args.t5_align_ot_iters,
#             "enable_mq_image_preserve": args.enable_mq_image_preserve,
#             "lambda_mq_image_preserve": args.lambda_mq_image_preserve,
#             "mq_image_preserve_margin": args.mq_image_preserve_margin,
#             "enable_wan_func_distill": args.enable_wan_func_distill,
#             "lambda_wan_func_distill": args.lambda_wan_func_distill,
#             "wan_func_teacher_mode": args.wan_func_teacher_mode,
#             "batch_size": args.batch_size,
#             "gradient_accumulation_steps": args.gradient_accumulation_steps,
#             "max_grad_norm": args.max_grad_norm,
#             "frame_num": args.frame_num,
#             "max_area": args.max_area,
#             "num_metaqueries": args.num_metaqueries,
#             "connector_num_hidden_layers": args.connector_num_hidden_layers,
#             "dit_condition_mode": args.dit_condition_mode,
#             "mq_gradient_checkpointing": args.mq_gradient_checkpointing,
#             "train_mq_input_embeddings": args.train_mq_input_embeddings,
#             "null_caption_prob": args.null_caption_prob,
#             "null_image_prob": args.null_image_prob,
#             "wan_train_mode": args.wan_train_mode,
#             "wan_auto_full_mem_gb": args.wan_auto_full_mem_gb,
#             "wan_lr_ratio": args.wan_lr_ratio,
#             "wan_cond_name_pattern": args.wan_cond_name_pattern,
#             "t5_cpu": args.t5_cpu,
#             "dit_fsdp": args.dit_fsdp,
#             "t5_fsdp": args.t5_fsdp,
#             "use_sp": args.use_sp,
#             "aggressive_empty_cache": args.aggressive_empty_cache,
#             "seed": args.seed,
#             "save_steps": args.save_steps,
#             "log_steps": args.log_steps,
#             "enable_loss_early_stop": args.enable_loss_early_stop,
#             "loss_early_stop_min_step": args.loss_early_stop_min_step,
#             "loss_early_stop_threshold": args.loss_early_stop_threshold,
#             "log_every_step": args.log_every_step,
#             "wandb_log_every_step": args.wandb_log_every_step,
#             "metrics_jsonl_path": args.metrics_jsonl_path,
#             "log_cuda_memory": args.log_cuda_memory,
#             "output_dir": args.output_dir,
#             "local_openvid_video_root": args.local_openvid_video_root,
#             "local_openvid_csv_path": args.local_openvid_csv_path,
#             "local_openvid_limit": args.local_openvid_limit,
#             "local_openvid_hd_video_root": args.local_openvid_hd_video_root,
#             "local_openvid_hd_csv_path": args.local_openvid_hd_csv_path,
#             "local_openvid_hd_limit": args.local_openvid_hd_limit,
#             "wan_checkpoint_dir": args.wan_checkpoint_dir,
#             "qwen3vl_model_id": args.qwen3vl_model_id,
#         }

#     def _init_wandb(self):
#         args = self.args
#         if not getattr(args, "wandb_enabled", False):
#             return
#         if not self.is_main_process:
#             return
#         if args.wandb_api_key:
#             os.environ["WANDB_API_KEY"] = args.wandb_api_key
#         try:
#             import wandb
#         except ImportError:
#             print("[W&B] 未安装 wandb, 已跳过日志记录")
#             return
#         run_name = args.wandb_run_name.strip() or f"wan-ti2v-metaquery-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
#         tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
#         self.wandb = wandb
#         self.wandb_run = wandb.init(
#             project=args.wandb_project,
#             entity=args.wandb_entity or None,
#             name=run_name,
#             mode=args.wandb_mode,
#             config=self._wandb_config(),
#             tags=tags or None,
#         )
#         print(f"[W&B] 已初始化: project={args.wandb_project}, run={run_name}")

#     def _load_models(self):
#         """加载所有模型。"""
#         args = self.args

#         # ── 1. Wan TI2V Pipeline ─────────────────────────────────────────
#         print("\n[1/3] 加载 Wan TI2V Pipeline...")
#         from wan import WanTI2V
#         from wan.configs import WAN_CONFIGS

#         config = WAN_CONFIGS['ti2v-5B']
#         runtime_rank = (
#             torch.distributed.get_rank()
#             if torch.distributed.is_available() and torch.distributed.is_initialized()
#             else 0
#         )
#         self.wan = WanTI2V(
#             config=config,
#             checkpoint_dir=args.wan_checkpoint_dir,
#             device_id=args.dit_device,
#             rank=runtime_rank,
#             t5_fsdp=args.t5_fsdp,
#             dit_fsdp=args.dit_fsdp,
#             use_sp=args.use_sp,
#             t5_cpu=args.t5_cpu,
#             init_on_cpu=not args.no_init_on_cpu,
#             convert_model_dtype=args.convert_model_dtype,
#         )

#         # DiT 冻结；FSDP/SP 路径不再显式 .to，避免破坏分片包装
#         if not (args.dit_fsdp or args.use_sp):
#             self.wan.model.to(self.dev_dit)
#         self.wan.model.eval().requires_grad_(False)
#         t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
#         vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
#         if vae_model is None:
#             vae_model = getattr(self.wan, "vae", None)
#         self._force_freeze(t5_model)
#         self._force_freeze(vae_model)

#         self.wan_config = config
#         self.text_len = config.text_len  # 512
#         print(f"  ✅ Wan TI2V 5B 已加载, text_len={self.text_len}")

#         # ── 2. MetaQuery Encoder (直接输出 4096) ─────────────────────────
#         print("\n[2/3] 加载 MetaQuery Encoder (→4096)...")
#         # 统一使用 train_connector_for_wan.py 中的实现，避免同名类双份定义导致“改了不生效”。
#         from train_connector_for_wan import MetaQueryEncoderForWan as SharedMetaQueryEncoderForWan
#         self.mq_encoder = SharedMetaQueryEncoderForWan(
#             qwen3vl_model_id=args.qwen3vl_model_id,
#             num_metaqueries=args.num_metaqueries,
#             connector_num_hidden_layers=args.connector_num_hidden_layers,
#             gradient_checkpointing=args.mq_gradient_checkpointing,
#             train_input_embeddings=args.train_mq_input_embeddings,
#             dtype=torch.bfloat16,
#             device=f"cuda:{args.encoder_device}",
#         )
#         print(f"  ✅ Encoder实现来源: {self.mq_encoder.__class__.__module__}.{self.mq_encoder.__class__.__name__}")
#         self.mq_encoder.train()
#         if args.resume_mq_encoder_path:
#             state, resolved_path = load_mq_encoder_state(
#                 args.resume_mq_encoder_path,
#                 map_location="cpu",
#             )
#             missing, unexpected = self.mq_encoder.load_state_dict(state, strict=False)
#             print(f"  ✅ 已加载初始权重: {resolved_path}")
#             print(f"     missing={len(missing)}, unexpected={len(unexpected)}")
#         print(f"  ✅ MetaQuery Encoder 已加载")

#         # ── 3. 验证维度 ──────────────────────────────────────────────────
#         print("\n[3/3] 验证维度对齐...")
#         wan_text_dim = self.wan.model.text_dim  # 4096
#         mq_out_dim = self.mq_encoder.wan_text_dim  # 4096
#         assert wan_text_dim == mq_out_dim, (
#             f"维度不匹配! Wan text_dim={wan_text_dim}, MQ out={mq_out_dim}"
#         )
#         print(f"  ✅ MQ output dim = Wan text_dim = {wan_text_dim}")

#         # MQ-only: DiT text_len 仅容纳 MQ tokens
#         self._orig_text_len = self.wan.model.text_len
#         self._aug_text_len = args.num_metaqueries
#         print(f"  ✅ text_len(MQ-only): {self._orig_text_len} → {self._aug_text_len}")
#         self._configure_wan_trainable_params()

#     def _setup_optimizer(self):
#         """设置优化器和学习率调度。"""
#         args = self.args

#         mq_params = self._mq_trainable_params()
#         wan_params = self._wan_trainable_params()
#         trainable_params = self._all_trainable_params()
#         print(f"\n[Optimizer] 可训练参数组:")
#         print(f"  Connector + MQ Embeddings: {sum(p.numel() for p in mq_params) / 1e6:.1f}M")
#         print(f"  Wan DiT (mode={self._effective_wan_train_mode}): {sum(p.numel() for p in wan_params) / 1e6:.1f}M")
#         print(f"  Total trainable: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")
#         if len(trainable_params) <= 0:
#             raise RuntimeError("无可训练参数：请检查 MQ/Wan 训练配置。")

#         param_groups: List[Dict[str, Any]] = []
#         if mq_params:
#             param_groups.append(
#                 {
#                     "name": "mq",
#                     "params": mq_params,
#                     "lr": float(args.learning_rate),
#                 }
#             )
#         if wan_params:
#             param_groups.append(
#                 {
#                     "name": "wan",
#                     "params": wan_params,
#                     "lr": float(args.learning_rate) * float(getattr(args, "wan_lr_ratio", 1.0)),
#                 }
#             )

#         self.optimizer = torch.optim.AdamW(
#             param_groups,
#             betas=(0.9, 0.95),
#             weight_decay=0.1,
#             eps=1e-8,
#         )

#         def lr_lambda(step):
#             warmup = max(int(args.warmup_steps), 0)
#             total = max(int(args.num_train_steps), 1)
#             cooldown = int(getattr(args, "cooldown_steps", -1))
#             if cooldown < 0:
#                 cooldown = warmup
#             cooldown = max(cooldown, 0)
#             warmup = min(warmup, total)
#             cooldown = min(cooldown, max(total - warmup, 0))

#             if step < warmup:
#                 return step / max(1, warmup)
#             if args.lr_scheduler_type == "constant_with_warmup":
#                 return 1.0
#             if args.lr_scheduler_type == "warmup_hold_cooldown":
#                 cooldown_start = total - cooldown
#                 if cooldown <= 0 or step < cooldown_start:
#                     return 1.0
#                 progress = (step - cooldown_start) / max(1, cooldown)
#                 progress = min(max(progress, 0.0), 1.0)
#                 return 1.0 - (1.0 - float(args.lr_min_ratio)) * progress
#             progress = (step - warmup) / max(1, total - warmup)
#             return max(float(args.lr_min_ratio), 0.5 * (1.0 + math.cos(math.pi * progress)))

#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

#     def _encode_text(self, prompts):
#         """T5 编码文本"""
#         with torch.no_grad():
#             if not self.args.t5_cpu and not self.args.t5_fsdp:
#                 self.wan.text_encoder.model.to(self.dev_dit)
#                 context = self.wan.text_encoder(prompts, self.dev_dit)
#             else:
#                 context = self.wan.text_encoder(prompts, torch.device("cpu"))
#                 context = [t.to(self.dev_dit, dtype=torch.bfloat16) for t in context]
#         return context  # List[Tensor], each [text_len, 4096]

#     @staticmethod
#     def _resize_token_sequence(seq: torch.Tensor, out_tokens: int) -> torch.Tensor:
#         """
#         将 [L, D] token 序列重采样到 [out_tokens, D]。
#         使用线性插值仅做 teacher 侧长度对齐，不引入额外可训练参数。
#         """
#         if seq.dim() != 2:
#             raise ValueError(f"expect [L, D], got shape={tuple(seq.shape)}")
#         out_tokens = max(1, int(out_tokens))
#         if int(seq.shape[0]) == out_tokens:
#             return seq
#         # F.interpolate 期望 [N, C, L]
#         x = seq.transpose(0, 1).unsqueeze(0).float()
#         x = F.interpolate(x, size=out_tokens, mode="linear", align_corners=False)
#         return x.squeeze(0).transpose(0, 1)

#     @staticmethod
#     def _token_gram_matrix(tokens: torch.Tensor) -> torch.Tensor:
#         """
#         计算 token 关系矩阵（Gram）。
#         输入: [B, T, D]，输出: [B, T, T]
#         """
#         if tokens.dim() != 3:
#             raise ValueError(f"expect [B, T, D], got shape={tuple(tokens.shape)}")
#         x = tokens - tokens.mean(dim=1, keepdim=True)
#         x = F.normalize(x, p=2, dim=-1, eps=1e-6)
#         return torch.matmul(x, x.transpose(1, 2))

#     @staticmethod
#     def _linear_cka_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         线性 CKA 损失，返回 1-CKA（越小越好）。
#         输入: x/y [B, T, D]
#         """
#         if x.shape != y.shape:
#             raise ValueError(f"CKA shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")
#         x_c = x - x.mean(dim=1, keepdim=True)
#         y_c = y - y.mean(dim=1, keepdim=True)
#         kx = torch.matmul(x_c, x_c.transpose(1, 2))
#         ky = torch.matmul(y_c, y_c.transpose(1, 2))
#         hsic = (kx * ky).sum(dim=(1, 2))
#         denom = torch.sqrt(
#             kx.square().sum(dim=(1, 2)).clamp_min(1e-12)
#             * ky.square().sum(dim=(1, 2)).clamp_min(1e-12)
#         )
#         cka = hsic / denom.clamp_min(1e-12)
#         return (1.0 - cka.clamp(-1.0, 1.0)).mean()

#     @staticmethod
#     def _sinkhorn_ot_cost(
#         src_tokens: torch.Tensor,
#         tgt_tokens: torch.Tensor,
#         epsilon: float = 0.05,
#         iters: int = 25,
#     ) -> torch.Tensor:
#         """
#         Sinkhorn OT 软匹配代价（排列无关）。
#         输入: src/tgt [B, T, D]
#         输出: 标量（batch 平均 OT cost）
#         """
#         if src_tokens.dim() != 3 or tgt_tokens.dim() != 3:
#             raise ValueError(
#                 f"Sinkhorn expect [B,T,D], got src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
#             )
#         if int(src_tokens.shape[0]) != int(tgt_tokens.shape[0]) or int(src_tokens.shape[2]) != int(tgt_tokens.shape[2]):
#             raise ValueError(
#                 f"Sinkhorn shape mismatch: src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
#             )
#         bsz, n_tok, _ = src_tokens.shape
#         m_tok = int(tgt_tokens.shape[1])
#         if n_tok <= 0 or m_tok <= 0:
#             return src_tokens.new_zeros(())

#         cost = torch.cdist(src_tokens, tgt_tokens, p=2).pow(2)  # [B, N, M]
#         eps = max(float(epsilon), 1e-6)
#         kernel = torch.exp(-cost / eps).clamp_min(1e-12)
#         a = src_tokens.new_full((bsz, n_tok), 1.0 / float(n_tok))
#         b = src_tokens.new_full((bsz, m_tok), 1.0 / float(m_tok))
#         u = torch.ones_like(a)
#         v = torch.ones_like(b)
#         kernel_t = kernel.transpose(1, 2)

#         n_iter = max(int(iters), 1)
#         for _ in range(n_iter):
#             kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
#             u = a / kv
#             ktu = torch.bmm(kernel_t, u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
#             v = b / ktu

#         plan = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)  # [B, N, M]
#         return (plan * cost).sum(dim=(1, 2)).mean()

#     def _compute_mq_aux_losses(
#         self,
#         captions: List[str],
#         mq_refs: List[Any],
#         mq_features: torch.Tensor,
#         t5_context: List[torch.Tensor] | None = None,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         计算 MQ 辅助约束：
#         1) T5 对齐（支持 anchor/Gram+CKA/Sinkhorn）
#         2) T5 统计对齐（均值/方差）
#         3) 图像保持（可选）：有图条件需与 text-only MQ 保持最小间隔
#         """
#         device = self.dev_dit
#         zero = mq_features.new_zeros(())
#         out = {
#             "t5_l2": zero,
#             "t5_cos": zero,
#             "t5_stats": zero,
#             "t5_gram": zero,
#             "t5_cka": zero,
#             "t5_ot": zero,
#             "image_preserve": zero,
#             "total": zero,
#         }
#         args = self.args
#         need_t5 = bool(args.enable_t5_alignment) and (
#             float(args.lambda_t5_align_l2) > 0.0
#             or float(args.lambda_t5_align_cos) > 0.0
#             or float(args.lambda_t5_align_stats) > 0.0
#         )
#         need_img = bool(args.enable_mq_image_preserve) and float(args.lambda_mq_image_preserve) > 0.0
#         if not (need_t5 or need_img):
#             return out

#         mq_float = mq_features.to(device=device, dtype=torch.float32)
#         tokens = int(mq_float.shape[1])
#         hidden = int(mq_float.shape[2])
#         anchor_tokens = max(1, min(int(args.t5_align_anchor_tokens), tokens))

#         if need_t5:
#             with torch.no_grad():
#                 teacher_ctx = t5_context if t5_context is not None else self._encode_text(captions)
#                 pooled_t5 = []
#                 for t5_seq in teacher_ctx:
#                     # t5_seq: [L_t5, 4096]
#                     t5_seq_f = t5_seq.to(device=device, dtype=torch.float32)
#                     if int(t5_seq_f.shape[-1]) != hidden:
#                         raise RuntimeError(
#                             f"T5 hidden={int(t5_seq_f.shape[-1])} 与 MQ hidden={hidden} 不一致"
#                         )
#                     pooled_t5.append(self._resize_token_sequence(t5_seq_f, tokens))
#                 t5_teacher = torch.stack(pooled_t5, dim=0)  # [B, tokens, 4096]

#             align_mode = str(getattr(args, "t5_align_mode", "gram_cka")).strip().lower()
#             if align_mode == "anchor":
#                 mq_anchor = mq_float[:, :anchor_tokens, :]
#                 t5_anchor = t5_teacher[:, :anchor_tokens, :]
#                 out["t5_l2"] = F.mse_loss(mq_anchor, t5_anchor)

#                 mq_anchor_flat = mq_anchor.reshape(-1, hidden)
#                 t5_anchor_flat = t5_anchor.reshape(-1, hidden)
#                 cos_sim = F.cosine_similarity(mq_anchor_flat, t5_anchor_flat, dim=-1).mean()
#                 out["t5_cos"] = (1.0 - cos_sim)
#             elif align_mode == "gram_cka":
#                 mq_gram = self._token_gram_matrix(mq_float)
#                 t5_gram = self._token_gram_matrix(t5_teacher)
#                 out["t5_gram"] = F.mse_loss(mq_gram, t5_gram)
#                 out["t5_cka"] = self._linear_cka_loss(mq_float, t5_teacher)
#                 # 复用旧命名，保持日志/脚本兼容
#                 out["t5_l2"] = out["t5_gram"]
#                 out["t5_cos"] = out["t5_cka"]
#             elif align_mode == "sinkhorn_ot":
#                 out["t5_ot"] = self._sinkhorn_ot_cost(
#                     mq_float,
#                     t5_teacher,
#                     epsilon=float(getattr(args, "t5_align_ot_epsilon", 0.05)),
#                     iters=int(getattr(args, "t5_align_ot_iters", 25)),
#                 )
#                 out["t5_l2"] = out["t5_ot"]
#             else:
#                 raise ValueError(f"Unknown --t5_align_mode: {align_mode}")

#             mq_mean = mq_float.mean(dim=1)
#             mq_std = mq_float.std(dim=1, unbiased=False)
#             t5_mean = t5_teacher.mean(dim=1)
#             t5_std = t5_teacher.std(dim=1, unbiased=False)
#             out["t5_stats"] = F.mse_loss(mq_mean, t5_mean) + F.mse_loss(mq_std, t5_std)

#         if need_img:
#             has_ref = torch.tensor(
#                 [1 if ref is not None else 0 for ref in mq_refs],
#                 device=device,
#                 dtype=torch.bool,
#             )
#             if bool(torch.any(has_ref).item()):
#                 with torch.no_grad():
#                     mq_text_only = self.mq_encoder(captions, None).to(device=device, dtype=torch.float32)
#                 diff = mq_float[has_ref] - mq_text_only[has_ref]
#                 # 每样本的 token+channel RMS 距离
#                 rms = torch.sqrt(torch.mean(diff * diff, dim=(1, 2)) + 1e-8)
#                 margin = float(args.mq_image_preserve_margin)
#                 out["image_preserve"] = F.relu(margin - rms).mean()

#         out["total"] = (
#             float(args.lambda_t5_align_l2) * out["t5_l2"]
#             + float(args.lambda_t5_align_cos) * out["t5_cos"]
#             + float(args.lambda_t5_align_stats) * out["t5_stats"]
#             + float(args.lambda_mq_image_preserve) * out["image_preserve"]
#         )
#         return out

#     def _encode_video(self, video_tensors):
#         """VAE 编码视频 → latent"""
#         with torch.no_grad():
#             # video_tensors: [B, 3, T, H, W] or list of [3, T, H, W]
#             latents = []
#             for v in video_tensors:
#                 # v: [3, T, H, W] → VAE expects this format
#                 z = self.wan.vae.encode([v.to(self.dev_dit, dtype=torch.bfloat16)])
#                 latents.append(z[0])  # z[0]: [C_z, T', H', W']
#         return latents

#     def _encode_first_frame(self, first_frame_tensor):
#         """VAE 编码参考图第一帧 → i2v condition latent"""
#         with torch.no_grad():
#             # first_frame: [3, H, W] → [3, 1, H, W]
#             ff = first_frame_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
#             z = self.wan.vae.encode([ff])
#         return z[0]  # [C_z, 1, H', W']

#     def _resolve_train_ref_anchor_mode(self) -> str:
#         """
#         返回当前 batch 实际使用的锚定模式。
#         - none / animate_like: 直接使用
#         - mixed50: 按 optimizer step 交替 none / animate_like，保证长期约 50/50
#         """
#         mode = str(getattr(self.args, "train_ref_anchor_mode", "none")).strip().lower()
#         if mode in ("none", "animate_like"):
#             return mode
#         if mode == "mixed50":
#             use_animate = (self._train_ref_anchor_mixed_counter % 2 == 1)
#             self._train_ref_anchor_mixed_counter += 1
#             return "animate_like" if use_animate else "none"
#         raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

#     def _train_ref_anchor_alpha(self, t_norm: torch.Tensor, mode: str | None = None) -> torch.Tensor:
#         """
#         训练期首帧软锚定系数（0~1）。
#         说明：
#         - none: 始终 0，不改动训练行为
#         - animate_like: 高噪声(早期)强锚定，随后余弦衰减到 0
#         """
#         if mode is None:
#             mode = self._resolve_train_ref_anchor_mode()
#         if mode == "none":
#             return torch.zeros_like(t_norm, dtype=torch.float32)
#         if mode != "animate_like":
#             raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

#         alpha0 = float(getattr(self.args, "train_ref_anchor_alpha0", 0.95))
#         warmup_ratio = float(getattr(self.args, "train_ref_anchor_warmup_ratio", 0.35))
#         alpha0 = max(0.0, min(1.0, alpha0))
#         warmup_ratio = max(0.0, min(1.0, warmup_ratio))
#         if warmup_ratio <= 0.0 or alpha0 <= 0.0:
#             return torch.zeros_like(t_norm, dtype=torch.float32)

#         start_t = 1.0 - warmup_ratio
#         alpha = torch.zeros_like(t_norm, dtype=torch.float32)
#         active = t_norm >= start_t
#         if not torch.any(active):
#             return alpha
#         u = ((t_norm[active] - start_t) / max(warmup_ratio, 1e-6)).clamp(0.0, 1.0)
#         alpha[active] = alpha0 * 0.5 * (1.0 - torch.cos(math.pi * u))
#         return alpha

#     @staticmethod
#     def _frames_to_latent_slots(frame_count: int, stride_t: int) -> int:
#         """像素帧数 -> latent 时间槽数（与 VAE 时间下采样保持一致）"""
#         f = max(0, int(frame_count))
#         if f <= 0:
#             return 0
#         return int((f - 1) // max(int(stride_t), 1) + 1)

#     def _encode_ref_image_to_latent(
#         self,
#         ref_img: Image.Image | None,
#         latent_h: int,
#         latent_w: int,
#         z_channels: int,
#     ) -> torch.Tensor:
#         """
#         将参考图编码为 1 帧 reference latent。
#         若 ref_img 缺失，返回零 reference latent。
#         """
#         if ref_img is None:
#             return torch.zeros(
#                 z_channels, 1, latent_h, latent_w,
#                 device=self.dev_dit, dtype=torch.float32,
#             )
#         target_h = int(latent_h * self.wan_config.vae_stride[1])
#         target_w = int(latent_w * self.wan_config.vae_stride[2])
#         ref_resized = ref_img.resize((target_w, target_h), Image.LANCZOS)
#         ref_np = np.array(ref_resized).astype(np.float32)
#         ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1) / 127.5 - 1.0
#         ref_5d = ref_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
#         with torch.no_grad():
#             ref_lat = self.wan.vae.encode([ref_5d])[0]
#         return ref_lat.float()

#     def _compute_wan_func_distill_loss(
#         self,
#         model_output: List[torch.Tensor],
#         x_inputs: List[torch.Tensor],
#         timesteps_wan: torch.Tensor,
#         max_seq_len: int,
#         t5_context: List[torch.Tensor],
#         mq_features: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         函数级蒸馏:
#             L_func = MSE( pred_mq(x_t,t), pred_t5(x_t,t) )
#         其中 pred_t5 由 frozen Wan + T5 条件生成（teacher no-grad）。
#         """
#         args = self.args
#         mode = str(getattr(args, "wan_func_teacher_mode", "t5_only")).strip().lower()
#         if mode not in {"t5_only", "t5_plus_mq"}:
#             raise ValueError(f"Unknown --wan_func_teacher_mode: {mode}")

#         teacher_context: List[torch.Tensor] = []
#         for i, t5_seq in enumerate(t5_context):
#             t5_feat = t5_seq.to(self.dev_dit, dtype=torch.bfloat16)
#             if mode == "t5_plus_mq":
#                 mq_feat = mq_features[i].detach().to(self.dev_dit, dtype=torch.bfloat16)
#                 t5_feat = torch.cat([mq_feat, t5_feat], dim=0)
#             teacher_context.append(t5_feat)

#         if not teacher_context:
#             return mq_features.new_zeros(())

#         teacher_text_len = max(int(ctx.shape[0]) for ctx in teacher_context)
#         cur_text_len = int(self.wan.model.text_len)
#         self.wan.model.text_len = teacher_text_len
#         try:
#             with torch.no_grad():
#                 with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#                     teacher_output = self.wan.model(
#                         x_inputs,
#                         t=timesteps_wan,
#                         context=teacher_context,
#                         seq_len=max_seq_len,
#                     )
#         finally:
#             self.wan.model.text_len = cur_text_len

#         loss = 0.0
#         valid = 0
#         for i in range(len(model_output)):
#             pred_mq = model_output[i].float()
#             pred_t5 = teacher_output[i].float()
#             loss = loss + F.mse_loss(pred_mq, pred_t5)
#             valid += 1
#         if valid <= 0:
#             return mq_features.new_zeros(())
#         return loss / valid

#     def _compute_loss(self, batch):
#         """
#         计算一个 batch 的 Flow Matching 损失。

#         训练默认使用 t2v 模式 (无第一帧蒙版/无首帧锚定)。
#         可通过 --train_ref_anchor_mode 在 x_t 注入 animate-like 首帧软锚定，
#         以缓解与 i2v 推理分布不一致问题。
#         """
#         args = self.args
#         captions = batch["caption"]
#         videos = batch["video"]         # list of [3, T, H, W]
#         mq_refs = batch["mq_ref_image"]  # list of PIL or None
#         B = len(captions)
#         self._last_loss_denoise = 0.0
#         self._last_loss_aux_align_total = 0.0
#         self._last_loss_aux_t5_l2 = 0.0
#         self._last_loss_aux_t5_cos = 0.0
#         self._last_loss_aux_t5_stats = 0.0
#         self._last_loss_aux_t5_gram = 0.0
#         self._last_loss_aux_t5_cka = 0.0
#         self._last_loss_aux_t5_ot = 0.0
#         self._last_loss_aux_image_preserve = 0.0
#         self._last_loss_aux_wan_func = 0.0

#         # ── 1. MetaQuery 编码 (在 encoder 设备上, 有梯度) ────────────────
#         mq_images = []
#         for ref in mq_refs:
#             if ref is not None:
#                 mq_images.append([ref])
#             else:
#                 mq_images.append(None)

#         all_none = all(img is None for img in mq_images)
#         if all_none:
#             mq_features = self.mq_encoder(captions, None)
#         else:
#             for i, img in enumerate(mq_images):
#                 if img is None:
#                     mq_images[i] = [Image.new("RGB", (224, 224))]
#             mq_features = self.mq_encoder(captions, mq_images)
#         # mq_features: [B, 256, 4096], 有梯度

#         # ── 2. MQ-only 注入 DiT context ─────────────────────────────────
#         augmented_context = []
#         for i in range(B):
#             mq_feat = mq_features[i].to(self.dev_dit, dtype=torch.bfloat16)
#             aug = mq_feat
#             if i == 0:
#                 self._verify_train_context_injection_once(mq_feat, aug)
#             augmented_context.append(aug)

#         # ── 4. VAE 编码视频 → latent (无梯度) ───────────────────────────
#         with torch.no_grad():
#             latents = self._encode_video(videos)
#             # latents: list of [C_z, T', H', W']

#         # ── 4. 采样噪声和时间步, 构建 Flow Matching 目标 ─────────────────
#         patch_size = self.wan_config.patch_size

#         x_inputs = []
#         timestep_rows = []
#         target_list = []
#         target_slots_list = []
#         max_seq_len = 0

#         for i, lat in enumerate(latents):
#             C, T, H, W = lat.shape
#             lat = lat.float()
#             x0_for_fm = lat

#             T_full = int(x0_for_fm.shape[1])
#             tokens_per_frame = int(math.ceil((H * W) / (patch_size[1] * patch_size[2])))
#             seq_len_i = int(tokens_per_frame * T_full)
#             max_seq_len = max(max_seq_len, seq_len_i)

#             t_val = torch.rand(1, device=self.dev_dit, dtype=torch.float32)
#             noise = torch.randn_like(x0_for_fm, dtype=torch.float32)

#             # Flow matching: x_t = (1-t) * x_0 + t * noise
#             sigma = t_val.view(-1, 1, 1, 1)
#             noisy_lat = (1.0 - sigma) * x0_for_fm + sigma * noise

#             # 目标: noise - x_0 (velocity)
#             velocity = noise - x0_for_fm

#             # token 级 timestep：MQ-only 下全部 token 共享 t
#             t_scalar = float((t_val * self.wan.num_train_timesteps).item())
#             t_row = torch.full((seq_len_i,), t_scalar, device=self.dev_dit, dtype=torch.float32)

#             x_inputs.append(noisy_lat)
#             target_list.append(velocity)
#             timestep_rows.append(t_row)
#             target_slots_list.append(T)

#         # 拼接 timestep → [B, max_seq_len]
#         padded_rows = []
#         for row in timestep_rows:
#             pad_len = max_seq_len - int(row.numel())
#             if pad_len > 0:
#                 pad_val = float(row[-1].item()) if row.numel() > 0 else 0.0
#                 row = torch.cat([row, row.new_full((pad_len,), pad_val)], dim=0)
#             padded_rows.append(row)
#         timesteps_wan = torch.stack(padded_rows, dim=0).to(self.dev_dit)

#         self._last_train_ref_anchor_alpha_mean = 0.0
#         self._last_train_ref_anchor_applied = 0
#         self._last_train_ref_anchor_effective_mode = "mq_only"
#         self._last_train_video_conditioning_mode = "mq_only"
#         self._last_train_prefix_latent_slots = 0
#         self._last_train_target_latent_slots = int(round(sum(target_slots_list) / max(len(target_slots_list), 1)))
#         self._last_train_prefix_loss_dropped = 0

#         # ── 5. MQ-only text_len + DiT forward ───────────────────────────
#         orig_text_len = self.wan.model.text_len
#         self.wan.model.text_len = self._aug_text_len

#         try:
#             with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#                 model_output = self.wan.model(
#                     x_inputs,
#                     t=timesteps_wan,
#                     context=augmented_context,
#                     seq_len=max_seq_len,
#                 )

#             # ── 6. 计算去噪主损失 ──────────────────────────────────────────
#             denoise_loss = 0.0
#             valid_terms = 0
#             for i in range(B):
#                 pred = model_output[i].float()
#                 target = target_list[i]
#                 loss = F.mse_loss(pred, target)
#                 denoise_loss += loss
#                 valid_terms += 1
#             if valid_terms <= 0:
#                 raise RuntimeError("无有效训练样本参与损失计算")
#             denoise_loss = denoise_loss / valid_terms

#             # 新版训练目标：仅保留原始去噪主损失（ground-truth latent velocity vs predicted velocity）
#             total_loss = denoise_loss
#             self._last_loss_denoise = float(denoise_loss.detach().item())
#             self._last_loss_aux_align_total = 0.0
#             self._last_loss_aux_t5_l2 = 0.0
#             self._last_loss_aux_t5_cos = 0.0
#             self._last_loss_aux_t5_stats = 0.0
#             self._last_loss_aux_t5_gram = 0.0
#             self._last_loss_aux_t5_cka = 0.0
#             self._last_loss_aux_t5_ot = 0.0
#             self._last_loss_aux_image_preserve = 0.0
#             self._last_loss_aux_wan_func = 0.0

#         finally:
#             self.wan.model.text_len = orig_text_len

#         return total_loss

#     def train(self):
#         """主训练循环。"""
#         args = self.args
#         self._audit_runtime_trainability(stage="train_start")

#         # 设置随机种子
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
#         np.random.seed(args.seed)

#         # 数据集（已完全收敛到 WanVideoDataset）
#         if WanDatasetClass is None:
#             raise RuntimeError("未能导入 WanVideoDataset，请检查 train_connector_for_wan.py 及其依赖")

#         dataset = WanDatasetClass(
#             seed=args.seed,
#             frame_num=args.frame_num,
#             max_area=args.max_area,
#             null_caption_prob=args.null_caption_prob,
#             null_image_prob=args.null_image_prob,
#             max_caption_tokens=args.max_caption_tokens,
#             caption_tokenizer_path=args.caption_tokenizer_path,
#             min_duration_sec=args.min_duration_sec,
#             max_duration_sec=args.max_duration_sec,
#             local_openvid_video_root=args.local_openvid_video_root,
#             local_openvid_csv_path=args.local_openvid_csv_path,
#             local_openvid_limit=args.local_openvid_limit,
#             local_openvid_hd_video_root=args.local_openvid_hd_video_root,
#             local_openvid_hd_csv_path=args.local_openvid_hd_csv_path,
#             local_openvid_hd_limit=args.local_openvid_hd_limit,
#             local_video_cache_dir=args.local_video_cache_dir,
#         )

#         if len(dataset) == 0:
#             raise RuntimeError("数据集为空！检查路径和 JSON 文件。")

#         # 由于视频尺寸可能不同, 使用 batch_size=1 避免 collate 问题
#         dataloader = DataLoader(
#             dataset,
#             batch_size=1,
#             shuffle=True,
#             num_workers=args.dataloader_num_workers,
#             pin_memory=True,
#             collate_fn=self._collate_fn,
#         )

#         # 训练循环
#         os.makedirs(args.output_dir, exist_ok=True)
#         output_dir = Path(args.output_dir).expanduser().resolve()
#         if not self._metrics_jsonl_path:
#             self._metrics_jsonl_path = str((output_dir / "logs" / "train_metrics.jsonl").expanduser().resolve())
#         args.output_dir = str(output_dir)
#         args.metrics_jsonl_path = self._metrics_jsonl_path
#         self._train_wall_start = time.perf_counter()

#         # 训练前快照（用于 verify_metaquery_chain before vs after）
#         self._train_before_checkpoint_path = str(output_dir / "checkpoint-before-training")
#         self._save_checkpoint(
#             self._train_before_checkpoint_path,
#             step=0,
#             extra_info={
#                 "is_before_training": True,
#                 "resume_mq_encoder_path": getattr(args, "resume_mq_encoder_path", None),
#                 "note": "trainable params snapshot before optimizer updates",
#             },
#         )
#         if self.is_main_process:
#             print(f"[VERIFY] 已保存训练前快照: {self._train_before_checkpoint_path}")

#         self.mq_encoder.train()
#         step = 0
#         running_loss = 0.0
#         early_stop_triggered = False
#         early_stop_reason = ""
#         early_stop_ckpt_path = ""
#         data_iter = iter(dataloader)

#         pbar = tqdm(total=args.num_train_steps, desc="Training")
#         self.optimizer.zero_grad(set_to_none=True)

#         while step < args.num_train_steps:
#             step_wall_start = time.perf_counter()
#             accum_loss = 0.0
#             accum_denoise_loss = 0.0
#             accum_align_loss = 0.0
#             accum_align_t5_l2 = 0.0
#             accum_align_t5_cos = 0.0
#             accum_align_t5_stats = 0.0
#             accum_align_t5_gram = 0.0
#             accum_align_t5_cka = 0.0
#             accum_align_t5_ot = 0.0
#             accum_align_img = 0.0
#             accum_align_wan_func = 0.0
#             skip_optimizer_step = False
#             had_fatal_cuda_error = False
#             backward_ok = 0
#             skip_reason = ""
#             self._current_train_ref_anchor_mode = self._resolve_train_ref_anchor_mode()

#             for accum_step in range(args.gradient_accumulation_steps):
#                 # 获取 batch
#                 try:
#                     batch = next(data_iter)
#                 except StopIteration:
#                     data_iter = iter(dataloader)
#                     batch = next(data_iter)

#                 try:
#                     loss = self._compute_loss(batch)
#                     loss = loss / args.gradient_accumulation_steps
#                     loss.backward()
#                     self._log_grad_health_once()
#                     accum_loss += loss.item()
#                     scale = 1.0 / max(float(args.gradient_accumulation_steps), 1.0)
#                     accum_denoise_loss += float(self._last_loss_denoise) * scale
#                     accum_align_loss += float(self._last_loss_aux_align_total) * scale
#                     accum_align_t5_l2 += float(self._last_loss_aux_t5_l2) * scale
#                     accum_align_t5_cos += float(self._last_loss_aux_t5_cos) * scale
#                     accum_align_t5_stats += float(self._last_loss_aux_t5_stats) * scale
#                     accum_align_t5_gram += float(self._last_loss_aux_t5_gram) * scale
#                     accum_align_t5_cka += float(self._last_loss_aux_t5_cka) * scale
#                     accum_align_t5_ot += float(self._last_loss_aux_t5_ot) * scale
#                     accum_align_img += float(self._last_loss_aux_image_preserve) * scale
#                     accum_align_wan_func += float(self._last_loss_aux_wan_func) * scale
#                     backward_ok += 1
#                 except Exception as e:
#                     err = str(e)
#                     bad_video = None
#                     try:
#                         bad_video = batch.get("video_path", None)
#                     except Exception:
#                         bad_video = None
#                     print(f"[WARN] step {step} accum_step {accum_step} 训练异常: {err}")
#                     if bad_video is not None:
#                         print(f"[WARN] step {step} accum_step {accum_step} bad_video={bad_video}")
#                     err_l = err.lower()
#                     is_illegal_access = "illegal memory access" in err_l
#                     is_device_assert = "device-side assert" in err_l
#                     if isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in err.lower()):
#                         skip_optimizer_step = True
#                         skip_reason = "oom"
#                         self.optimizer.zero_grad(set_to_none=True)
#                         if torch.cuda.is_available():
#                             torch.cuda.empty_cache()
#                         gc.collect()
#                         break
#                     if is_illegal_access or is_device_assert:
#                         had_fatal_cuda_error = True
#                         skip_optimizer_step = True
#                         skip_reason = "fatal_cuda"
#                         self.optimizer.zero_grad(set_to_none=True)
#                         if torch.cuda.is_available():
#                             torch.cuda.empty_cache()
#                         gc.collect()
#                         break
#                     # 其他异常也跳过本 step，避免残缺梯度进入 optimizer.step
#                     skip_optimizer_step = True
#                     skip_reason = "error"
#                     self.optimizer.zero_grad(set_to_none=True)
#                     break
#                     continue

#             if had_fatal_cuda_error:
#                 raise RuntimeError(
#                     f"Fatal CUDA kernel error at step={step}. "
#                     "检测到 illegal memory access/device-side assert，已中止训练。"
#                 )

#             if backward_ok == 0:
#                 self._skipped_step_count += 1
#                 if skip_reason == "oom":
#                     self._oom_skip_count += 1
#                 elif skip_reason and skip_reason != "fatal_cuda":
#                     self._error_skip_count += 1
#                 continue

#             if skip_optimizer_step:
#                 self._skipped_step_count += 1
#                 if skip_reason == "oom":
#                     self._oom_skip_count += 1
#                 else:
#                     self._error_skip_count += 1
#                 continue

#             # 梯度裁剪
#             grad_norm = torch.nn.utils.clip_grad_norm_(
#                 self._all_trainable_params(),
#                 args.max_grad_norm,
#             )

#             self.optimizer.step()
#             self.scheduler.step()
#             self.optimizer.zero_grad(set_to_none=True)
#             if args.aggressive_empty_cache:
#                 torch.cuda.empty_cache()

#             step += 1
#             step_time = max(time.perf_counter() - step_wall_start, 1e-6)
#             running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss
#             lr = self.scheduler.get_last_lr()[0]
#             grad_norm_value = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
#             effective_samples = int(max(backward_ok, 0) * max(args.batch_size, 1))
#             samples_per_sec = float(effective_samples / step_time)

#             metrics = {
#                 "train/loss_step": float(accum_loss),
#                 "train/loss_ema": float(running_loss),
#                 "train/loss_denoise": float(accum_denoise_loss),
#                 "train/loss_align_total": float(accum_align_loss),
#                 "train/loss_align_t5_l2": float(accum_align_t5_l2),
#                 "train/loss_align_t5_cos": float(accum_align_t5_cos),
#                 "train/loss_align_t5_stats": float(accum_align_t5_stats),
#                 "train/loss_align_t5_gram": float(accum_align_t5_gram),
#                 "train/loss_align_t5_cka": float(accum_align_t5_cka),
#                 "train/loss_align_t5_ot": float(accum_align_t5_ot),
#                 "train/loss_align_img_preserve": float(accum_align_img),
#                 "train/loss_align_wan_func": float(accum_align_wan_func),
#                 "train/lr": float(lr),
#                 "train/grad_norm": float(grad_norm_value),
#                 "train/step": int(step),
#                 "train/step_time_sec": float(step_time),
#                 "train/samples_per_sec": float(samples_per_sec),
#                 "train/backward_ok_microbatches": int(backward_ok),
#                 "train/effective_batch_samples": int(effective_samples),
#                 "train/skipped_step_count": int(self._skipped_step_count),
#                 "train/oom_skip_count": int(self._oom_skip_count),
#                 "train/error_skip_count": int(self._error_skip_count),
#                 "train/ref_anchor_alpha_mean": float(self._last_train_ref_anchor_alpha_mean),
#                 "train/ref_anchor_applied": int(self._last_train_ref_anchor_applied),
#                 "train/ref_anchor_mode_cfg": str(getattr(args, "train_ref_anchor_mode", "none")),
#                 "train/ref_anchor_mode_effective": str(self._last_train_ref_anchor_effective_mode),
#                 "train/ref_anchor_effective_is_animate": int(self._last_train_ref_anchor_effective_mode == "animate_like"),
#                 "train/video_conditioning_mode_cfg": str(getattr(args, "dit_condition_mode", "mq_only")),
#                 "train/video_conditioning_mode_effective": str(self._last_train_video_conditioning_mode),
#                 "train/prefix_latent_slots": int(self._last_train_prefix_latent_slots),
#                 "train/target_latent_slots": int(self._last_train_target_latent_slots),
#                 "train/prefix_loss_dropped": int(self._last_train_prefix_loss_dropped),
#             }
#             metrics.update(self._collect_trainability_metrics())
#             metrics.update(self._collect_cuda_memory_metrics())

#             should_log = bool(args.log_every_step or (step % args.log_steps == 0))
#             should_wandb_log = bool(
#                 self.wandb_run is not None and (args.wandb_log_every_step or should_log)
#             )

#             # 日志
#             if should_log:
#                 pbar.set_postfix({
#                     "loss": f"{accum_loss:.4f}",
#                     "denoise": f"{accum_denoise_loss:.4f}",
#                     "align": f"{accum_align_loss:.4f}",
#                     "func": f"{accum_align_wan_func:.4f}",
#                     "avg": f"{running_loss:.4f}",
#                     "lr": f"{lr:.2e}",
#                     "grad": f"{grad_norm_value:.2f}",
#                     "dP": f"{metrics['train/param_sample_abs_delta_mean']:.3e}",
#                 })
#                 print(
#                     f"\n[Step {step}/{args.num_train_steps}] "
#                     f"loss={accum_loss:.4f} denoise={accum_denoise_loss:.4f} align={accum_align_loss:.4f} func={accum_align_wan_func:.4f} "
#                     f"avg={running_loss:.4f} "
#                     f"lr={lr:.2e} grad_norm={grad_norm_value:.2f} "
#                     f"dt={step_time:.2f}s samp/s={samples_per_sec:.2f} "
#                     f"param_delta={metrics['train/param_sample_abs_delta_mean']:.3e} "
#                     f"skip(oom/err/total)={self._oom_skip_count}/{self._error_skip_count}/{self._skipped_step_count}"
#                 )
#             if should_wandb_log:
#                 self.wandb.log(metrics, step=step)
#             self._append_metrics_jsonl(metrics)
#             self._record_metrics(metrics)

#             # 保存
#             if step % args.save_steps == 0:
#                 self._save_checkpoint(output_dir / f"checkpoint-{step}", step)

#             pbar.update(1)

#             step_loss_for_early_stop = float(accum_denoise_loss)
#             if (
#                 bool(getattr(args, "enable_loss_early_stop", False))
#                 and step >= int(getattr(args, "loss_early_stop_min_step", 800))
#                 and step_loss_for_early_stop < float(getattr(args, "loss_early_stop_threshold", 0.25))
#             ):
#                 early_stop_triggered = True
#                 early_stop_reason = (
#                     f"train/loss_denoise={step_loss_for_early_stop:.6f} < {float(args.loss_early_stop_threshold):.6f} "
#                     f"at step={int(step)}"
#                 )
#                 early_stop_ckpt_path = str(
#                     output_dir / f"checkpoint-earlystop-step{int(step)}-denoise{step_loss_for_early_stop:.4f}"
#                 )
#                 self._save_checkpoint(
#                     early_stop_ckpt_path,
#                     step,
#                     extra_info={
#                         "early_stop": True,
#                         "early_stop_metric": "train/loss_denoise",
#                         "early_stop_loss": step_loss_for_early_stop,
#                         "early_stop_threshold": float(args.loss_early_stop_threshold),
#                         "early_stop_min_step": int(args.loss_early_stop_min_step),
#                     },
#                 )
#                 if self.is_main_process:
#                     print(f"[EARLY-STOP] 已触发: {early_stop_reason}")
#                     print(f"[EARLY-STOP] checkpoint: {early_stop_ckpt_path}")
#                 break

#         pbar.close()

#         # 最终保存
#         final_ckpt_path = str(output_dir / "checkpoint-final")
#         final_extra_info = None
#         if early_stop_triggered:
#             final_extra_info = {
#                 "early_stop": True,
#                 "early_stop_reason": early_stop_reason,
#                 "early_stop_checkpoint_path": early_stop_ckpt_path,
#             }
#         self._save_checkpoint(final_ckpt_path, step, extra_info=final_extra_info)
#         self._write_training_chain_manifest(output_dir, final_checkpoint_path=final_ckpt_path, final_step=step)
#         if early_stop_triggered and self.is_main_process:
#             print(f"[EARLY-STOP] 训练提前结束，最终步数: {step}")
#         print(f"\n✅ 训练完成！最终 checkpoint: {final_ckpt_path}")
#         if self.wandb_run is not None:
#             self.wandb.finish()

#     def _save_checkpoint(self, path, step, extra_info: Dict[str, Any] | None = None):
#         """保存 MQ 编码器 +（可选）Wan DiT 可训练子集（兼容增强格式）"""
#         if not self.is_main_process:
#             return
#         path = Path(path).expanduser().resolve()
#         module = self._mq_encoder_module()
#         ckpt_info = save_mq_checkpoint_bundle(
#             path=path,
#             module=module,
#             optimizer=self.optimizer,
#             scheduler=self.scheduler,
#             step=step,
#             args=self.args,
#             wan_module=getattr(self.wan, "model", None),
#             wan_train_mode=str(getattr(self, "_effective_wan_train_mode", "frozen")),
#             metrics_tail=self._metrics_history[-200:],
#             metrics_summary=self._build_metrics_summary(step=step),
#             extra_info={
#                 "before_checkpoint_path": self._train_before_checkpoint_path,
#                 "metrics_jsonl_path": self._metrics_jsonl_path,
#                 "wan_train_mode_effective": str(getattr(self, "_effective_wan_train_mode", "frozen")),
#                 "wan_trainable_tensor_count": int(len(getattr(self, "_wan_trainable_names", []))),
#                 "wan_trainable_name_preview": list(getattr(self, "_wan_trainable_names", [])[:64]),
#                 **(extra_info or {}),
#             },
#         )
#         print(f"  💾 Checkpoint 已保存: {ckpt_info['path']}")
#         if self.wandb_run is not None and self.args.wandb_log_checkpoint:
#             self.wandb.log(
#                 {
#                     "checkpoint/step": int(step),
#                     "checkpoint/path": str(ckpt_info["path"]),
#                 },
#                 step=step,
#             )

#     @staticmethod
#     def _collate_fn(batch):
#         """自定义 collate: 不 stack 不同尺寸的 tensor"""
#         result = {}
#         for key in batch[0].keys():
#             result[key] = [item[key] for item in batch]
#         return result


# # =============================================================================
# # Main
# # =============================================================================
# if __name__ == "__main__":
#     args = parse_args()
#     trainer = MetaQueryWanTrainer(args)
#     trainer.train()


















































# # 下面这个是，增加首帧作为wan参考条件的情况：
# """
# train_metaquery_wan.py
# =======================
# MetaQuery + Wan2.2 TI2V (Text+Image → Video) 联合训练脚本。

# ★ 核心思路:
#     复刻原始 MetaQuery 训练范式 —— 冻结 DiT，训练 Connector：
#     1. Qwen3-VL (冻结, 仅 MQ embeddings 可训练)
#     2. Connector: Qwen2Encoder(24L) + Linear + GELU + Linear + RMSNorm → dim=4096 (直接对齐 Wan)
#     3. to_wan_proj: 不再需要! Connector 直接输出 Wan text_dim=4096
#     4. Wan TI2V DiT (冻结): 接收 [MQ_tokens + T5_tokens] 作为 context
#     5. 计算 Flow Matching Loss → 反向传播更新 Connector + MQ Embeddings

# ★ 为什么选 WanTI2V (而非 I2V 或 Animate):
#     - TI2V 5B 是 Wan2.2 最新的 Text+Image→Video 统一模型
#     - 使用相同 DiT architecture 处理 t2v 和 i2v (model_type='ti2v')
#     - 不需要 CLIP encoder (I2V 需要 CLIP, Animate 需要 CLIP+Face+Pose)
#     - 参考图通过 VAE 编码后的 latent mask 注入 (最优雅的方式)
#     - 5B 参数量适中, 显存友好

# ★ 不需要 to_wan_proj:
#     直接让 Connector 输出 dim=4096 (Wan text_dim)
#     → 训练时 DiT 的 text_embedding 层直接消费 MQ 特征
#     → 无中间随机投影层, 梯度直接流过

# 用法:
#     # 单卡
#     python train_metaquery_wan.py --wan_checkpoint_dir /path/to/Wan2.2-TI2V-5B

#     # 多卡
#     torchrun --nproc_per_node=2 train_metaquery_wan.py
# """

# import os
# import sys
# import gc
# import json
# import math
# import time
# import argparse
# import random
# from pathlib import Path
# from datetime import datetime
# from contextlib import contextmanager
# from typing import Dict, Tuple, Any, List

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from PIL import Image
# from tqdm import tqdm

# # ── 路径设置 ─────────────────────────────────────────────────────────────────
# WAN_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(WAN_ROOT))

# METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
# sys.path.insert(0, METAQUERY_ROOT)


# # =============================================================================
# # 配置
# # =============================================================================
# def parse_args():
#     p = argparse.ArgumentParser(description="Train MetaQuery Connector for Wan TI2V")

#     # ── 模型路径 ──────────────────────────────────────────────────────────
#     p.add_argument("--wan_checkpoint_dir", type=str,
#                    default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
#                    help="Wan2.2 TI2V checkpoint 目录")
#     p.add_argument("--qwen3vl_model_id", type=str,
#                    default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking",
#                    help="Qwen3-VL 模型 ID 或本地路径")
#     p.add_argument("--output_dir", type=str,
#                    default="/home/liuzhirui/model/Wan2.2/metaquery_wan_ti2v_training",
#                    help="训练输出目录")

#     # ── 数据(OpenVid/WanVideoDataset) ───────────────────────────────────────
#     p.add_argument("--local_openvid_video_root", type=str, default=None,
#                    help="本地 OpenVid 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video")
#     p.add_argument("--local_openvid_csv_path", type=str, default=None,
#                    help="本地 OpenVid CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv")
#     p.add_argument("--local_openvid_limit", type=int, default=None,
#                    help="仅使用前N条本地匹配样本，默认使用全部已匹配样本")
#     p.add_argument("--local_openvid_hd_video_root", type=str, default=None,
#                    help="本地 OpenVid HD 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video_HD")
#     p.add_argument("--local_openvid_hd_csv_path", type=str, default=None,
#                    help="本地 OpenVid HD CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv")
#     p.add_argument("--local_openvid_hd_limit", type=int, default=None,
#                    help="仅使用前N条本地HD匹配样本，默认使用全部已匹配样本")
#     p.add_argument("--local_video_cache_dir", type=str, default=None,
#                    help="本地视频缓存目录（用于URL/字节解码缓存，可选）")
#     p.add_argument("--frame_num", type=int, default=41,
#                    help="每个视频片段采样帧数 (4n+1)")
#     p.add_argument("--max_area", type=int, default=480 * 832,
#                    help="视频最大面积 (宽×高)")
#     p.add_argument("--max_caption_tokens", type=int, default=512,
#                    help="超过该token长度的caption会被过滤")
#     p.add_argument("--caption_tokenizer_path", type=str, default="google/umt5-xxl",
#                    help="用于caption长度统计的tokenizer")
#     p.add_argument("--min_duration_sec", type=float, default=0.5,
#                    help="最短时长过滤阈值")
#     p.add_argument("--max_duration_sec", type=float, default=20.0,
#                    help="最长时长过滤阈值")

#     # ── 训练参数 ──────────────────────────────────────────────────────────
#     p.add_argument("--learning_rate", type=float, default=5e-5)
#     p.add_argument("--num_train_steps", type=int, default=5000)
#     p.add_argument("--warmup_steps", type=int, default=200)
#     p.add_argument(
#         "--lr_scheduler_type",
#         type=str,
#         default="cosine_with_warmup",
#         choices=["cosine_with_warmup", "constant_with_warmup", "warmup_hold_cooldown"],
#         help=(
#             "学习率调度器类型。"
#             "constant_with_warmup=warmup后恒定；"
#             "cosine_with_warmup=warmup后余弦衰减；"
#             "warmup_hold_cooldown=warmup线性升+中段恒定+末段线性降。"
#         ),
#     )
#     p.add_argument(
#         "--cooldown_steps",
#         type=int,
#         default=-1,
#         help="warmup_hold_cooldown 模式下末段降学习率步数。<0 表示使用 warmup_steps。",
#     )
#     p.add_argument(
#         "--lr_min_ratio",
#         type=float,
#         default=0.01,
#         help="cosine_with_warmup 模式下的最小学习率比例。",
#     )
#     p.add_argument("--batch_size", type=int, default=1)
#     p.add_argument("--gradient_accumulation_steps", type=int, default=4)
#     p.add_argument("--max_grad_norm", type=float, default=1.0)
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--save_steps", type=int, default=500)
#     p.add_argument("--log_steps", type=int, default=10)
#     p.add_argument("--enable_loss_early_stop", action="store_true", default=False,
#                    help="启用可选早停：当 step>=loss_early_stop_min_step 且 loss_step<loss_early_stop_threshold 时提前结束训练并保存 checkpoint。")
#     p.add_argument("--disable_loss_early_stop", action="store_false", dest="enable_loss_early_stop",
#                    help="关闭 loss 早停（默认）。")
#     p.add_argument("--loss_early_stop_min_step", type=int, default=800,
#                    help="loss 早停触发的最小 step（含）。")
#     p.add_argument("--loss_early_stop_threshold", type=float, default=0.25,
#                    help="loss 早停阈值：当 train/loss_step 小于该值时触发。")
#     p.add_argument("--log_every_step", action="store_true",
#                    help="每个优化 step 都打印详细训练日志")
#     p.add_argument("--wandb_log_every_step", action="store_true",
#                    help="每个优化 step 都写入 W&B（默认按 log_steps 写入）")
#     p.add_argument("--metrics_jsonl_path", type=str, default="",
#                    help="可选：将每步指标追加写入 JSONL 文件")
#     p.add_argument("--log_cuda_memory", action="store_true",
#                    help="记录并输出 CUDA 显存指标")
#     p.add_argument("--dataloader_num_workers", type=int, default=0)

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
#     p.add_argument("--mq_gradient_checkpointing", action="store_true",
#                    help="启用 MetaQuery 编码器梯度检查点，降低显存占用")
#     p.add_argument("--train_mq_input_embeddings", action="store_true", default=True,
#                    help="训练 Qwen 输入 embedding（默认开启）")
#     p.add_argument("--freeze_mq_input_embeddings", action="store_false", dest="train_mq_input_embeddings",
#                    help="冻结 Qwen 输入 embedding，仅训练 connector")
#     p.add_argument("--null_caption_prob", type=float, default=0.1)
#     p.add_argument("--null_image_prob", type=float, default=0.1)
#     p.add_argument("--enable_t5_alignment", action="store_true", default=True,
#                    help="启用 T5 对齐辅助损失（默认开启）：让 MQ 条件分布更接近 Wan 已适配的 T5 条件流形。")
#     p.add_argument("--disable_t5_alignment", action="store_false", dest="enable_t5_alignment",
#                    help="关闭 T5 对齐辅助损失，仅使用去噪主损失。")
#     p.add_argument(
#         "--t5_align_mode",
#         type=str,
#         default="gram_cka",
#         choices=["anchor", "gram_cka", "sinkhorn_ot"],
#         help=(
#             "T5 对齐方式。anchor=前K token 一一对齐；"
#             "gram_cka=基于 token 关系矩阵(Gram+CKA)的排列无关对齐；"
#             "sinkhorn_ot=基于 OT/Sinkhorn 的软匹配对齐。"
#         ),
#     )
#     p.add_argument("--t5_align_anchor_tokens", type=int, default=64,
#                    help="用于 T5 对齐的 anchor token 数（从 256 个 MQ token 前缀取）。")
#     p.add_argument("--lambda_t5_align_l2", type=float, default=0.2,
#                    help="T5 对齐主项权重：anchor 模式对应 token-L2；gram_cka 模式对应 Gram-L2；sinkhorn_ot 模式对应 OT 代价。")
#     p.add_argument("--lambda_t5_align_cos", type=float, default=0.1,
#                    help="T5 对齐次项权重：anchor 模式对应 token-cos；gram_cka 模式对应 CKA；sinkhorn_ot 模式默认忽略。")
#     p.add_argument("--lambda_t5_align_stats", type=float, default=0.02,
#                    help="T5 对齐的均值/方差统计损失权重。")
#     p.add_argument("--t5_align_ot_epsilon", type=float, default=0.05,
#                    help="Sinkhorn OT 熵正则温度 epsilon（越小越接近硬匹配）。")
#     p.add_argument("--t5_align_ot_iters", type=int, default=25,
#                    help="Sinkhorn OT 迭代次数。")
#     p.add_argument("--enable_mq_image_preserve", action="store_true", default=False,
#                    help="启用图像保持约束：有参考图时，MQ(cond) 与 MQ(text-only) 保持最小间隔。")
#     p.add_argument("--lambda_mq_image_preserve", type=float, default=0.02,
#                    help="图像保持约束权重。")
#     p.add_argument("--mq_image_preserve_margin", type=float, default=0.10,
#                    help="图像保持约束的最小间隔阈值（L2 均方根距离）。")
#     p.add_argument("--enable_wan_func_distill", action="store_true", default=False,
#                    help="启用 Wan 函数级蒸馏：约束 pred_mq(x_t,t) 贴近 pred_t5(x_t,t)。")
#     p.add_argument("--disable_wan_func_distill", action="store_false", dest="enable_wan_func_distill",
#                    help="关闭 Wan 函数级蒸馏。")
#     p.add_argument("--lambda_wan_func_distill", type=float, default=0.0,
#                    help="Wan 函数级蒸馏损失权重。")
#     p.add_argument(
#         "--wan_func_teacher_mode",
#         type=str,
#         default="t5_only",
#         choices=["t5_only", "t5_plus_mq"],
#         help="函数级蒸馏 teacher 条件。t5_only=仅 T5；t5_plus_mq=T5 与 MQ 拼接。",
#     )
#     p.add_argument("--enable_ti2v_first_frame_condition", action="store_true", default=True,
#                    help="启用 Wan 训练侧首帧参考条件（与 MQ 图像条件并行）。")
#     p.add_argument("--disable_ti2v_first_frame_condition", action="store_false",
#                    dest="enable_ti2v_first_frame_condition",
#                    help="关闭 Wan 训练侧首帧参考条件，仅保留 MQ 条件。")
#     p.add_argument("--train_video_conditioning_mode", type=str, default="legacy_t2v",
#                    choices=["legacy_t2v", "wan_animate_slot"],
#                    help=(
#                        "训练期视频条件注入方式: "
#                        "legacy_t2v=现有 TI2V 训练（可选首帧软锚定）；"
#                        "wan_animate_slot=参考图作为 preserved reference slot 注入，前缀 slot 不计入主损失"
#                    ))
#     p.add_argument("--train_animate_ref_frames", type=int, default=1,
#                    help="wan_animate_slot 模式下参考图保留帧数（像素帧数，内部按 VAE stride 映射到 latent slots）")
#     p.add_argument("--train_animate_temporal_frames", type=int, default=0,
#                    help="wan_animate_slot 模式下 temporal guidance 帧数（像素帧数；若无外部时序条件可保持 0）")
#     p.add_argument("--train_animate_conditional_frames", type=int, default=0,
#                    help="wan_animate_slot 模式下额外 conditional 帧数（像素帧数；无条件时保持 0，将注入全零 latent）")
#     p.add_argument("--train_animate_preserve_timestep_zero", action="store_true", default=True,
#                    help="wan_animate_slot: preserved prefix 对应 token 的 timestep 置 0（默认开启）")
#     p.add_argument("--train_animate_no_preserve_timestep_zero", action="store_false",
#                    dest="train_animate_preserve_timestep_zero",
#                    help="wan_animate_slot: 关闭 preserved prefix timestep=0")
#     p.add_argument("--train_animate_drop_prefix_loss", action="store_true", default=True,
#                    help="wan_animate_slot: 仅在 target frames 上计算损失，丢弃 reference/temporal/conditional prefix（默认开启）")
#     p.add_argument("--train_animate_no_drop_prefix_loss", action="store_false",
#                    dest="train_animate_drop_prefix_loss",
#                    help="wan_animate_slot: 不丢弃 prefix，整段都计入损失")
#     p.add_argument("--train_ref_anchor_mode", type=str, default="none",
#                    choices=["none", "animate_like", "mixed50"],
#                    help="训练时是否对 x_t 的首帧加入软参考锚定。none=保持原始 t2v；animate_like=全程启用软锚定；mixed50=约50%批次启用软锚定")
#     p.add_argument("--train_ref_anchor_alpha0", type=float, default=0.95,
#                    help="animate_like 模式的最大锚定强度 alpha0")
#     p.add_argument("--train_ref_anchor_warmup_ratio", type=float, default=0.35,
#                    help="animate_like 模式在高噪声区间启用锚定的占比（0~1）")

#     # ── 设备 ──────────────────────────────────────────────────────────────
#     p.add_argument("--dit_device", type=int, default=0,
#                    help="DiT + VAE + T5 所在 GPU")
#     p.add_argument("--encoder_device", type=int, default=1,
#                    help="Qwen3-VL + Connector 所在 GPU")
#     p.add_argument("--resume_mq_encoder_path", type=str, default=None,
#                    help="从已有mq_encoder权重继续训练")
#     p.add_argument("--t5_cpu", action="store_true",
#                    help="将Wan的T5文本编码器保留在CPU，显著降低GPU显存占用（速度会变慢）")
#     p.add_argument("--dit_fsdp", action="store_true",
#                    help="启用 Wan DiT 的 FSDP 参数分片，降低单卡模型权重占用")
#     p.add_argument("--t5_fsdp", action="store_true",
#                    help="启用 T5 编码器的 FSDP 参数分片")
#     p.add_argument("--use_sp", action="store_true",
#                    help="启用 sequence parallel（xDiT/USP 路径）")
#     p.add_argument("--no_init_on_cpu", action="store_true",
#                    help="关闭 init_on_cpu；默认开启以减小加载瞬时显存峰值")
#     p.add_argument("--convert_model_dtype", action="store_true",
#                    help="将 Wan DiT 参数显式转换到 config.param_dtype（仅非FSDP时生效）")
#     p.add_argument("--aggressive_empty_cache", action="store_true",
#                    help="每步训练后执行 torch.cuda.empty_cache()，缓解显存碎片")
#     p.add_argument("--wandb_enabled", action="store_true",
#                    help="启用 Weights & Biases 训练日志")
#     p.add_argument("--wandb_project", type=str, default="wan-metaquery",
#                    help="W&B project 名称")
#     p.add_argument("--wandb_entity", type=str, default="",
#                    help="W&B entity/team 名称")
#     p.add_argument("--wandb_run_name", type=str, default="",
#                    help="W&B run 名称, 留空自动生成")
#     p.add_argument("--wandb_tags", type=str, default="",
#                    help="W&B tags, 逗号分隔")
#     p.add_argument("--wandb_mode", type=str, default="online",
#                    choices=["online", "offline", "disabled"],
#                    help="W&B 模式")
#     p.add_argument("--wandb_api_key", type=str, default="",
#                    help="W&B API Key, 传入后会写入 WANDB_API_KEY 环境变量")
#     p.add_argument("--wandb_log_checkpoint", action="store_true",
#                    help="在 W&B 中记录 checkpoint 路径")
#     p.add_argument("--strict_freeze_check", action="store_true", default=True,
#                    help="启用严格冻结校验：若发现 Wan/T5/VAE 可训练或 optimizer 混入非 MQ 参数则中止")
#     p.add_argument("--no_strict_freeze_check", action="store_false", dest="strict_freeze_check",
#                    help="关闭严格冻结校验，仅打印告警")
#     p.add_argument(
#         "--wan_train_mode",
#         type=str,
#         default="auto",
#         choices=["auto", "frozen", "full", "cond_only"],
#         help=(
#             "Wan DiT 训练模式。auto=按显存策略自动在 full/cond_only 之间选择；"
#             "frozen=冻结；full=全量训练；cond_only=仅训 cross-attn + conditioning projection/AdaLN/modulation。"
#         ),
#     )
#     p.add_argument(
#         "--wan_auto_full_mem_gb",
#         type=float,
#         default=120.0,
#         help="auto 模式下，当 DiT 卡总显存 >= 该阈值时选择 full，否则选择 cond_only。",
#     )
#     p.add_argument(
#         "--wan_lr_ratio",
#         type=float,
#         default=1.0,
#         help="Wan 可训练参数学习率倍率（实际 lr = learning_rate * wan_lr_ratio）。",
#     )
#     p.add_argument(
#         "--wan_cond_name_pattern",
#         type=str,
#         default="",
#         help=(
#             "可选：自定义 cond_only 的参数名匹配关键字，逗号分隔。"
#             "为空时使用内置规则(cross_attn,text_embedding,time_projection,modulation,norm3,cross_attn_norm)。"
#         ),
#     )

#     return p.parse_args()


# def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
#     """兼容不同 torch 版本的安全加载。"""
#     try:
#         return torch.load(path, map_location=map_location, weights_only=True)
#     except TypeError:
#         return torch.load(path, map_location=map_location)


# def _extract_model_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
#     """从不同 checkpoint 负载中提取模型权重字典。"""
#     if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
#         return payload["model_state_dict"]
#     if isinstance(payload, dict):
#         tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
#         non_tensor_values = [v for v in payload.values() if not torch.is_tensor(v)]
#         if tensor_values and not non_tensor_values:
#             return payload
#     raise ValueError("无法从 checkpoint 提取 model_state_dict")


# def _to_cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     out = {}
#     for k, v in state_dict.items():
#         if torch.is_tensor(v):
#             out[k] = v.detach().cpu().contiguous()
#     return out


# def load_mq_encoder_state(path_or_dir: str, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, torch.Tensor], str]:
#     """
#     加载 MetaQuery encoder 权重:
#     - 支持传入单个文件: mq_encoder_full.pt / training_state.pt / model.safetensors
#     - 支持传入目录: 自动按优先级查找文件
#     """
#     path = Path(path_or_dir)
#     if not path.exists():
#         raise FileNotFoundError(f"checkpoint 路径不存在: {path}")

#     if path.is_dir():
#         candidates = [
#             path / "mq_encoder_full.pt",
#             path / "mq_encoder_full.safetensors",
#             path / "model.safetensors",
#             path / "pytorch_model.bin",
#             path / "training_state.pt",
#         ]
#         picked = next((p for p in candidates if p.exists()), None)
#         if picked is None:
#             raise FileNotFoundError(
#                 f"checkpoint 目录中未找到可加载权重文件: {path} "
#                 f"(expect one of {[c.name for c in candidates]})"
#             )
#         path = picked

#     suffix = path.suffix.lower()
#     if suffix == ".safetensors":
#         try:
#             from safetensors.torch import load_file
#         except Exception as e:
#             raise RuntimeError(
#                 f"检测到 safetensors 权重但未能导入 safetensors: {path}"
#             ) from e
#         state = load_file(str(path), device="cpu")
#     else:
#         payload = _safe_torch_load(path, map_location=map_location)
#         state = _extract_model_state_dict(payload)

#     return state, str(path.expanduser().resolve())


# def _write_json(path: Path, payload: Dict[str, Any]) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)


# def _to_jsonable(value: Any) -> Any:
#     if value is None or isinstance(value, (str, int, float, bool)):
#         return value
#     if isinstance(value, (list, tuple)):
#         return [_to_jsonable(v) for v in value]
#     if isinstance(value, dict):
#         return {str(k): _to_jsonable(v) for k, v in value.items()}
#     if isinstance(value, Path):
#         return str(value)
#     return str(value)


# def save_mq_checkpoint_bundle(
#     path: Path,
#     module: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LRScheduler,
#     step: int,
#     args: argparse.Namespace,
#     wan_module: nn.Module | None = None,
#     wan_train_mode: str = "frozen",
#     metrics_tail: List[Dict[str, Any]] | None = None,
#     metrics_summary: Dict[str, Any] | None = None,
#     extra_info: Dict[str, Any] | None = None,
# ) -> Dict[str, Any]:
#     """
#     保存“最小可用 + 兼容增强”的 checkpoint bundle。
#     兼容你当前推理脚本（mq_encoder_full.pt）并补充常见训练文件。
#     """
#     path = path.expanduser().resolve()
#     path.mkdir(parents=True, exist_ok=True)

#     full_state_cpu = _to_cpu_state_dict(module.state_dict())
#     name_to_param = dict(module.named_parameters())
#     trainable_state_cpu = {
#         name: tensor
#         for name, tensor in full_state_cpu.items()
#         if name_to_param.get(name, None) is not None
#         and name_to_param[name].requires_grad
#     }

#     torch.save(
#         {
#             "step": step,
#             "model_state_dict": trainable_state_cpu,
#             "optimizer_state_dict": optimizer.state_dict(),
#             "scheduler_state_dict": scheduler.state_dict(),
#         },
#         path / "training_state.pt",
#     )
#     torch.save(full_state_cpu, path / "mq_encoder_full.pt")
#     torch.save(trainable_state_cpu, path / "mq_encoder_trainable.pt")

#     wan_trainable_state_cpu: Dict[str, torch.Tensor] = {}
#     wan_trainable_param_count = 0
#     if wan_module is not None and isinstance(wan_module, nn.Module):
#         for name, p in wan_module.named_parameters():
#             if not p.requires_grad:
#                 continue
#             wan_trainable_state_cpu[name] = p.detach().cpu().contiguous()
#             wan_trainable_param_count += int(p.numel())
#     if wan_trainable_state_cpu:
#         torch.save(wan_trainable_state_cpu, path / "wan_dit_trainable.pt")

#     torch.save(vars(args), path / "training_args.bin")
#     _write_json(
#         path / "training_args.json",
#         {str(k): _to_jsonable(v) for k, v in vars(args).items()},
#     )
#     torch.save(optimizer.state_dict(), path / "optimizer.pt")
#     torch.save(scheduler.state_dict(), path / "scheduler.pt")

#     trainer_state = {
#         "global_step": int(step),
#         "checkpoint_format": "wan_metaquery_v2",
#         "has_full_pt": True,
#         "has_training_state": True,
#         "has_trainable_pt": True,
#         "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
#         "wan_train_mode": str(wan_train_mode),
#         "wan_trainable_param_count": int(wan_trainable_param_count),
#         "has_metrics_summary": bool(metrics_summary),
#         "metrics_tail_count": int(len(metrics_tail) if metrics_tail is not None else 0),
#     }
#     if extra_info:
#         trainer_state["extra_info"] = _to_jsonable(extra_info)
#     _write_json(path / "trainer_state.json", trainer_state)

#     config_payload = {
#         "format": "wan_metaquery_encoder",
#         "num_metaqueries": int(getattr(args, "num_metaqueries", 256)),
#         "connector_num_hidden_layers": int(getattr(args, "connector_num_hidden_layers", 24)),
#         "wan_text_dim": int(getattr(module, "wan_text_dim", 4096)),
#         "qwen3vl_model_id": str(getattr(args, "qwen3vl_model_id", "")),
#         "train_mq_input_embeddings": bool(getattr(args, "train_mq_input_embeddings", True)),
#         "wan_train_mode": str(wan_train_mode),
#         "wan_trainable_param_count": int(wan_trainable_param_count),
#         "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
#         "checkpoint_step": int(step),
#         "num_train_steps": int(getattr(args, "num_train_steps", 0)),
#         "save_steps": int(getattr(args, "save_steps", 0)),
#         "log_steps": int(getattr(args, "log_steps", 0)),
#         "enable_loss_early_stop": bool(getattr(args, "enable_loss_early_stop", False)),
#         "loss_early_stop_min_step": int(getattr(args, "loss_early_stop_min_step", 800)),
#         "loss_early_stop_threshold": float(getattr(args, "loss_early_stop_threshold", 0.25)),
#         "frame_num": int(getattr(args, "frame_num", 0)),
#         "max_area": int(getattr(args, "max_area", 0)),
#         "learning_rate": float(getattr(args, "learning_rate", 0.0)),
#         "warmup_steps": int(getattr(args, "warmup_steps", 0)),
#         "lr_scheduler_type": str(getattr(args, "lr_scheduler_type", "cosine_with_warmup")),
#         "cooldown_steps": int(getattr(args, "cooldown_steps", -1)),
#         "lr_min_ratio": float(getattr(args, "lr_min_ratio", 0.01)),
#         "enable_t5_alignment": bool(getattr(args, "enable_t5_alignment", True)),
#         "t5_align_mode": str(getattr(args, "t5_align_mode", "gram_cka")),
#         "t5_align_anchor_tokens": int(getattr(args, "t5_align_anchor_tokens", 64)),
#         "lambda_t5_align_l2": float(getattr(args, "lambda_t5_align_l2", 0.0)),
#         "lambda_t5_align_cos": float(getattr(args, "lambda_t5_align_cos", 0.0)),
#         "lambda_t5_align_stats": float(getattr(args, "lambda_t5_align_stats", 0.0)),
#         "t5_align_ot_epsilon": float(getattr(args, "t5_align_ot_epsilon", 0.05)),
#         "t5_align_ot_iters": int(getattr(args, "t5_align_ot_iters", 25)),
#         "enable_mq_image_preserve": bool(getattr(args, "enable_mq_image_preserve", False)),
#         "lambda_mq_image_preserve": float(getattr(args, "lambda_mq_image_preserve", 0.0)),
#         "mq_image_preserve_margin": float(getattr(args, "mq_image_preserve_margin", 0.0)),
#         "enable_wan_func_distill": bool(getattr(args, "enable_wan_func_distill", False)),
#         "lambda_wan_func_distill": float(getattr(args, "lambda_wan_func_distill", 0.0)),
#         "wan_func_teacher_mode": str(getattr(args, "wan_func_teacher_mode", "t5_only")),
#         "batch_size": int(getattr(args, "batch_size", 1)),
#         "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1)),
#         "null_caption_prob": float(getattr(args, "null_caption_prob", 0.0)),
#         "null_image_prob": float(getattr(args, "null_image_prob", 0.0)),
#         "wan_train_mode": str(getattr(args, "wan_train_mode", "auto")),
#         "wan_auto_full_mem_gb": float(getattr(args, "wan_auto_full_mem_gb", 120.0)),
#         "wan_lr_ratio": float(getattr(args, "wan_lr_ratio", 1.0)),
#         "wan_cond_name_pattern": str(getattr(args, "wan_cond_name_pattern", "")),
#     }
#     # 记录 MLLM embedding 行信息，便于推理期验证“新增 MQ token embedding 是否被保存/加载”。
#     try:
#         emb = module.mllm_model.mllm_backbone.get_input_embeddings()
#         if emb is not None and getattr(emb, "weight", None) is not None:
#             rows_total = int(emb.weight.shape[0])
#             rows_base = int(getattr(module.mllm_model, "num_embeddings", 0))
#             config_payload["mllm_embed_rows_total"] = rows_total
#             config_payload["mllm_embed_rows_base"] = rows_base
#             config_payload["mllm_embed_rows_added"] = max(rows_total - rows_base, 0)
#     except Exception:
#         pass
#     if extra_info:
#         config_payload["extra_info"] = _to_jsonable(extra_info)
#     _write_json(path / "config.json", config_payload)
#     if metrics_summary:
#         _write_json(path / "metrics_summary.json", {str(k): _to_jsonable(v) for k, v in metrics_summary.items()})
#     if metrics_tail is not None:
#         _write_json(
#             path / "metrics_tail.json",
#             {"records": [{str(k): _to_jsonable(v) for k, v in row.items()} for row in metrics_tail]},
#         )

#     try:
#         from safetensors.torch import save_file

#         save_file(full_state_cpu, str(path / "model.safetensors"))
#         save_file(trainable_state_cpu, str(path / "mq_encoder_trainable.safetensors"))
#         if wan_trainable_state_cpu:
#             save_file(wan_trainable_state_cpu, str(path / "wan_dit_trainable.safetensors"))
#     except Exception:
#         # safetensors 为增强项，不可用时保持兼容主流程
#         pass

#     # 兼容“latest”指针
#     try:
#         with open(path.parent / "latest", "w", encoding="utf-8") as f:
#             f.write(f"{path.name}\n")
#     except Exception:
#         pass

#     return {
#         "step": int(step),
#         "path": str(path),
#     }
# # =============================================================================
# # 数据集
# # =============================================================================
# try:
#     from train_connector_for_wan import WanVideoDataset as _DefaultWanVideoDataset
# except Exception:
#     _DefaultWanVideoDataset = None

# # 单一数据集入口：仅使用 WanVideoDataset。
# # 在 train_metaquery_wan_new.py 中可通过设置 base_ti2v.WanDatasetClass 进行覆写。
# WanDatasetClass = _DefaultWanVideoDataset


# # =============================================================================
# # Trainer
# # =============================================================================
# class MetaQueryWanTrainer:
#     """
#     MetaQuery + Wan TI2V 联合训练。

#     训练流程:
#         1. MetaQuery (Connector 可训练) → [B, 256, 4096]
#         2. T5 编码文本 → [B, text_len, 4096]
#         3. 拼接: [MQ + T5] → [B, 256+text_len, 4096]
#         4. VAE 编码视频帧 → latent
#         5. 采样噪声+时间步 → noisy_latent
#         6. 参考图 VAE 编码 → first frame mask
#         7. DiT (冻结) forward: 预测速度
#         8. Flow Matching Loss → 反向传播 Connector + MQ Embeddings
#     """

#     def __init__(self, args):
#         self.args = args
#         self.dev_dit = torch.device(f"cuda:{args.dit_device}")
#         self.dev_enc = torch.device(f"cuda:{args.encoder_device}")
#         self.wandb = None
#         self.wandb_run = None
#         self.is_main_process = self._is_main_process()
#         self._printed_grad_health = False
#         self._skipped_step_count = 0
#         self._oom_skip_count = 0
#         self._error_skip_count = 0
#         self._printed_context_inject_check = False
#         self._param_monitor = []
#         self._trainable_param_count = 0
#         self._init_trainable_norm = 0.0
#         self._init_param_sample_norm = 0.0
#         _metrics_jsonl = (args.metrics_jsonl_path or "").strip()
#         self._metrics_jsonl_path = str(Path(_metrics_jsonl).expanduser().resolve()) if _metrics_jsonl else ""
#         self._metrics_history: List[Dict[str, Any]] = []
#         self._train_before_checkpoint_path = ""
#         self._train_wall_start = 0.0
#         self._last_train_ref_anchor_alpha_mean = 0.0
#         self._last_train_ref_anchor_applied = 0
#         self._last_train_ref_anchor_effective_mode = "none"
#         self._train_ref_anchor_mixed_counter = 0
#         self._current_train_ref_anchor_mode = "none"
#         self._last_train_video_conditioning_mode = "mq_only"
#         self._last_train_prefix_latent_slots = 0
#         self._last_train_target_latent_slots = 0
#         self._last_train_prefix_loss_dropped = 0
#         self._last_loss_denoise = 0.0
#         self._last_loss_aux_align_total = 0.0
#         self._last_loss_aux_t5_l2 = 0.0
#         self._last_loss_aux_t5_cos = 0.0
#         self._last_loss_aux_t5_stats = 0.0
#         self._last_loss_aux_t5_gram = 0.0
#         self._last_loss_aux_t5_cka = 0.0
#         self._last_loss_aux_t5_ot = 0.0
#         self._last_loss_aux_image_preserve = 0.0
#         self._last_loss_aux_wan_func = 0.0
#         self._effective_wan_train_mode = "frozen"
#         self._wan_trainable_names: List[str] = []
#         self._wan_trainable_params_cache: List[torch.nn.Parameter] = []

#         print("\n" + "=" * 60)
#         print("  MetaQuery + Wan TI2V 联合训练")
#         print("=" * 60)
#         print(f"  DiT 设备       : {self.dev_dit}")
#         print(f"  Encoder 设备   : {self.dev_enc}")
#         print(f"  学习率         : {args.learning_rate}")
#         print(f"  LR 调度器      : {args.lr_scheduler_type}")
#         print(f"  Cooldown 步数  : {args.cooldown_steps} (-1 表示使用 warmup_steps)")
#         print(f"  训练步数       : {args.num_train_steps}")
#         print(f"  有效 batch     : {args.batch_size * args.gradient_accumulation_steps}")
#         print(
#             f"  Loss 早停       : enabled={int(bool(args.enable_loss_early_stop))} "
#             f"min_step={args.loss_early_stop_min_step} threshold={args.loss_early_stop_threshold}"
#         )
#         print(
#             f"  Wan 训练模式    : req={args.wan_train_mode} auto_full_mem_gb={args.wan_auto_full_mem_gb} "
#             f"wan_lr_ratio={args.wan_lr_ratio}"
#         )
#         print(
#             f"  T5 对齐(已禁用) : cfg_enabled={int(bool(args.enable_t5_alignment))} "
#             f"mode={args.t5_align_mode} "
#             f"anchor={args.t5_align_anchor_tokens} "
#             f"l2={args.lambda_t5_align_l2} cos={args.lambda_t5_align_cos} stats={args.lambda_t5_align_stats} "
#             f"ot_eps={args.t5_align_ot_epsilon} ot_iters={args.t5_align_ot_iters}"
#         )
#         print(
#             f"  图像保持(已禁用): cfg_enabled={int(bool(args.enable_mq_image_preserve))} "
#             f"lambda={args.lambda_mq_image_preserve} margin={args.mq_image_preserve_margin}"
#         )
#         print(
#             f"  函数蒸馏(已禁用): cfg_enabled={int(bool(args.enable_wan_func_distill))} "
#             f"lambda={args.lambda_wan_func_distill} teacher={args.wan_func_teacher_mode}"
#         )
#         print("  额外损失开关   : 当前版本固定仅使用 denoise MSE（其余辅助损失已禁用）")
#         print("=" * 60)

#         self._load_models()
#         self._log_runtime_topology()
#         self._setup_optimizer()
#         self._audit_runtime_trainability(stage="init")
#         self._init_trainability_monitor()
#         self._init_wandb()

#     def _is_main_process(self):
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             return torch.distributed.get_rank() == 0
#         rank_env = os.environ.get("RANK")
#         if rank_env is None:
#             return True
#         return int(rank_env) == 0

#     def _mq_encoder_module(self):
#         return self.mq_encoder.module if hasattr(self.mq_encoder, "module") else self.mq_encoder

#     def _mq_trainable_params(self):
#         module = self._mq_encoder_module()
#         if hasattr(module, "get_trainable_params"):
#             return module.get_trainable_params()
#         return [p for p in module.parameters() if p.requires_grad]

#     def _resolve_wan_train_mode(self) -> str:
#         mode = str(getattr(self.args, "wan_train_mode", "auto")).strip().lower()
#         if mode != "auto":
#             return mode
#         total_gb = 0.0
#         if self.dev_dit.type == "cuda" and torch.cuda.is_available():
#             try:
#                 props = torch.cuda.get_device_properties(self.dev_dit)
#                 total_gb = float(props.total_memory) / float(1024 ** 3)
#             except Exception:
#                 total_gb = 0.0
#         threshold = float(getattr(self.args, "wan_auto_full_mem_gb", 120.0))
#         return "full" if total_gb >= threshold else "cond_only"

#     def _wan_cond_keywords(self) -> List[str]:
#         custom = str(getattr(self.args, "wan_cond_name_pattern", "")).strip()
#         if custom:
#             return [k.strip().lower() for k in custom.split(",") if k.strip()]
#         return [
#             "cross_attn",
#             "cross-attn",
#             "crossattention",
#             "cross_attention",
#             "text_embedding",
#             "time_projection",
#             "modulation",
#             "cross_attn_norm",
#             "norm3",
#         ]

#     def _configure_wan_trainable_params(self) -> None:
#         wan_model = getattr(self.wan, "model", None)
#         if wan_model is None:
#             self._effective_wan_train_mode = "frozen"
#             self._wan_trainable_names = []
#             self._wan_trainable_params_cache = []
#             return

#         # 先全冻结，再按模式打开。
#         self._force_freeze(wan_model)
#         mode = self._resolve_wan_train_mode()
#         self._effective_wan_train_mode = mode
#         selected_names: List[str] = []
#         selected_params: List[torch.nn.Parameter] = []

#         if mode == "full":
#             for name, p in wan_model.named_parameters():
#                 p.requires_grad_(True)
#                 selected_names.append(name)
#                 selected_params.append(p)
#         elif mode == "cond_only":
#             kws = self._wan_cond_keywords()
#             for name, p in wan_model.named_parameters():
#                 lname = name.lower()
#                 if any(kw in lname for kw in kws):
#                     p.requires_grad_(True)
#                     selected_names.append(name)
#                     selected_params.append(p)
#         elif mode == "frozen":
#             pass
#         else:
#             raise ValueError(f"Unknown --wan_train_mode: {mode}")

#         self._wan_trainable_names = selected_names
#         self._wan_trainable_params_cache = selected_params
#         if selected_params:
#             wan_model.train()
#         else:
#             wan_model.eval()

#         if self.is_main_process:
#             total = sum(int(p.numel()) for p in selected_params)
#             print(
#                 f"[WAN-TRAIN] requested={self.args.wan_train_mode} effective={mode} "
#                 f"trainable_tensors={len(selected_params)} trainable_params={total:,}"
#             )
#             if mode == "cond_only":
#                 kws = self._wan_cond_keywords()
#                 preview = ", ".join(kws[:10])
#                 print(f"[WAN-TRAIN] cond_only keywords={preview}")
#             if selected_names:
#                 preview = ", ".join(selected_names[:8])
#                 more = "" if len(selected_names) <= 8 else f" ... +{len(selected_names)-8}"
#                 print(f"[WAN-TRAIN] selected preview: {preview}{more}")

#     def _wan_trainable_params(self) -> List[torch.nn.Parameter]:
#         return list(self._wan_trainable_params_cache)

#     def _all_trainable_params(self) -> List[torch.nn.Parameter]:
#         out: List[torch.nn.Parameter] = []
#         seen = set()
#         for p in self._mq_trainable_params():
#             if id(p) not in seen:
#                 out.append(p)
#                 seen.add(id(p))
#         for p in self._wan_trainable_params():
#             if id(p) not in seen:
#                 out.append(p)
#                 seen.add(id(p))
#         return out

#     @staticmethod
#     def _module_param_stats(module: nn.Module | None) -> Dict[str, int]:
#         total = 0
#         trainable = 0
#         if module is None or not isinstance(module, nn.Module):
#             return {"total": 0, "trainable": 0}
#         for p in module.parameters():
#             n = int(p.numel())
#             total += n
#             if p.requires_grad:
#                 trainable += n
#         return {"total": total, "trainable": trainable}

#     @staticmethod
#     def _named_param_id_map(module: nn.Module | None, prefix: str) -> Dict[int, str]:
#         out: Dict[int, str] = {}
#         if module is None or not isinstance(module, nn.Module):
#             return out
#         for name, p in module.named_parameters():
#             out[id(p)] = f"{prefix}.{name}"
#         return out

#     @staticmethod
#     def _force_freeze(module: nn.Module | None) -> None:
#         if module is None or not isinstance(module, nn.Module):
#             return
#         try:
#             module.eval()
#         except Exception:
#             pass
#         try:
#             module.requires_grad_(False)
#         except Exception:
#             for p in module.parameters():
#                 p.requires_grad_(False)

#     def _log_runtime_topology(self) -> None:
#         if not self.is_main_process:
#             return
#         args = self.args
#         same_gpu = (self.dev_dit == self.dev_enc)
#         print(
#             "[AUDIT][TOPO] "
#             f"dit_device={self.dev_dit} encoder_device={self.dev_enc} same_gpu={same_gpu} "
#             f"t5_cpu={args.t5_cpu} t5_fsdp={args.t5_fsdp} dit_fsdp={args.dit_fsdp} use_sp={args.use_sp} "
#             f"num_metaqueries={args.num_metaqueries} aug_text_len={getattr(self, '_aug_text_len', -1)} "
#             f"wan_mode_effective={getattr(self, '_effective_wan_train_mode', 'frozen')}"
#         )
#         if same_gpu:
#             print("[AUDIT][TOPO][WARN] DiT 与 Qwen/Connector 在同一 GPU，显存峰值风险较高。")
#         if (not args.t5_cpu) and (not args.t5_fsdp):
#             print("[AUDIT][TOPO] T5 文本编码器会在 DiT 卡上参与前向（no_grad）。")
#         try:
#             from wan.modules import attention as _attn
#             fa2 = bool(getattr(_attn, "FLASH_ATTN_2_AVAILABLE", False))
#             fa3 = bool(getattr(_attn, "FLASH_ATTN_3_AVAILABLE", False))
#             force_sdpa = bool(getattr(_attn, "_FORCE_SDPA", False))
#             print(
#                 "[AUDIT][ATTN] "
#                 f"flash_attn2={fa2} flash_attn3={fa3} force_sdpa={force_sdpa}"
#             )
#         except Exception as e:
#             print(f"[AUDIT][ATTN][WARN] 无法读取 attention backend 信息: {e}")

#     def _audit_runtime_trainability(self, stage: str = "runtime", strict: bool | None = None) -> None:
#         args = self.args
#         if strict is None:
#             strict = bool(getattr(args, "strict_freeze_check", True))

#         wan_mode = str(getattr(self, "_effective_wan_train_mode", "frozen"))
#         # Wan 是否冻结由 wan_train_mode 决定；T5/VAE 始终冻结。
#         t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
#         self._force_freeze(t5_model)
#         vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
#         if vae_model is None:
#             vae_model = getattr(self.wan, "vae", None)
#         self._force_freeze(vae_model)

#         stats_wan = self._module_param_stats(getattr(self.wan, "model", None))
#         stats_t5 = self._module_param_stats(t5_model)
#         stats_vae = self._module_param_stats(vae_model)

#         mq_module = self._mq_encoder_module()
#         stats_mq = self._module_param_stats(mq_module)
#         mq_trainable_params = self._mq_trainable_params()
#         wan_trainable_params = self._wan_trainable_params()
#         mq_trainable_ids = {id(p) for p in mq_trainable_params}
#         wan_trainable_ids = {id(p) for p in wan_trainable_params}
#         allowed_trainable_ids = mq_trainable_ids | wan_trainable_ids
#         emb_trainable = 0
#         emb_rows_total = 0
#         emb_rows_base = 0
#         emb_rows_added = 0
#         emb_hidden = 0
#         try:
#             backbone = mq_module.mllm_model.mllm_backbone
#             emb = backbone.get_input_embeddings()
#             if emb is not None and getattr(emb, "weight", None) is not None:
#                 w = emb.weight
#                 emb_rows_total = int(w.shape[0])
#                 emb_hidden = int(w.shape[1]) if w.ndim >= 2 else 0
#                 emb_rows_base = int(getattr(mq_module.mllm_model, "num_embeddings", 0))
#                 emb_rows_added = max(emb_rows_total - emb_rows_base, 0)
#                 if bool(w.requires_grad):
#                     emb_trainable = int(w.numel())
#         except Exception:
#             pass

#         opt_params = []
#         for g in self.optimizer.param_groups:
#             opt_params.extend(g.get("params", []))
#         opt_ids = [id(p) for p in opt_params]
#         opt_id_set = set(opt_ids)

#         outside_ids = [pid for pid in opt_ids if pid not in allowed_trainable_ids]
#         missing_mq_ids = [pid for pid in mq_trainable_ids if pid not in opt_id_set]
#         missing_wan_ids = [pid for pid in wan_trainable_ids if pid not in opt_id_set]
#         duplicate_count = max(len(opt_ids) - len(opt_id_set), 0)

#         name_map: Dict[int, str] = {}
#         name_map.update(self._named_param_id_map(getattr(self.wan, "model", None), "wan.model"))
#         name_map.update(self._named_param_id_map(t5_model, "wan.text_encoder.model"))
#         name_map.update(self._named_param_id_map(vae_model, "wan.vae.model"))
#         name_map.update(self._named_param_id_map(mq_module, "mq_encoder"))

#         unexpected_mq_names = []
#         for name, p in mq_module.named_parameters():
#             if not p.requires_grad:
#                 continue
#             lower = name.lower()
#             if ("connector" in lower) or ("embed" in lower):
#                 continue
#             unexpected_mq_names.append(name)

#         if self.is_main_process:
#             print(
#                 f"[AUDIT][FREEZE][{stage}] "
#                 f"wan_trainable={stats_wan['trainable']:,}/{stats_wan['total']:,} "
#                 f"t5_trainable={stats_t5['trainable']:,}/{stats_t5['total']:,} "
#                 f"vae_trainable={stats_vae['trainable']:,}/{stats_vae['total']:,} "
#                 f"mq_trainable={stats_mq['trainable']:,}/{stats_mq['total']:,} "
#                 f"mq_trainable_tensors={len(mq_trainable_params)} "
#                 f"wan_mode={wan_mode} wan_trainable_tensors={len(wan_trainable_params)}"
#             )
#             print(
#                 f"[AUDIT][OPT][{stage}] "
#                 f"optimizer_params={len(opt_ids)} "
#                 f"outside_allowed={len(outside_ids)} missing_mq={len(missing_mq_ids)} "
#                 f"missing_wan={len(missing_wan_ids)} duplicates={duplicate_count}"
#             )
#             print(
#                 f"[AUDIT][MQ-EMB][{stage}] "
#                 f"enabled={int(bool(args.train_mq_input_embeddings))} "
#                 f"embed_trainable={emb_trainable:,} "
#                 f"rows_total={emb_rows_total} rows_base={emb_rows_base} rows_added={emb_rows_added} "
#                 f"hidden={emb_hidden} expected_added≈num_metaqueries+2={int(args.num_metaqueries) + 2}"
#             )
#             if unexpected_mq_names:
#                 preview = ", ".join(unexpected_mq_names[:6])
#                 more = "" if len(unexpected_mq_names) <= 6 else f" ... +{len(unexpected_mq_names)-6}"
#                 print(
#                     "[AUDIT][MQ][WARN] 检测到非 connector/embed 命名的可训练参数: "
#                     f"{preview}{more}"
#                 )

#         errors = []
#         if wan_mode == "frozen" and stats_wan["trainable"] > 0:
#             errors.append(f"Wan DiT 期望冻结但仍有可训练参数: {stats_wan['trainable']}")
#         if wan_mode != "frozen" and len(wan_trainable_ids) == 0:
#             errors.append(f"Wan DiT 训练模式={wan_mode} 但未选中可训练参数")
#         if stats_t5["trainable"] > 0:
#             errors.append(f"Wan T5 仍有可训练参数: {stats_t5['trainable']}")
#         if stats_vae["trainable"] > 0:
#             errors.append(f"Wan VAE 仍有可训练参数: {stats_vae['trainable']}")
#         if len(mq_trainable_ids) == 0:
#             errors.append("MQ encoder 无可训练参数")
#         if bool(args.train_mq_input_embeddings) and emb_trainable <= 0:
#             errors.append("设置了 train_mq_input_embeddings，但输入 embedding 未开启训练")
#         if (not bool(args.train_mq_input_embeddings)) and emb_trainable > 0:
#             errors.append("设置了 freeze_mq_input_embeddings，但输入 embedding 仍可训练")
#         if outside_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in outside_ids[:8]]
#             errors.append(f"optimizer 含非允许参数(MQ+Wan): {names}")
#         if missing_mq_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_mq_ids[:8]]
#             errors.append(f"部分 MQ 可训练参数未进 optimizer: {names}")
#         if missing_wan_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_wan_ids[:8]]
#             errors.append(f"部分 Wan 可训练参数未进 optimizer: {names}")
#         if duplicate_count > 0:
#             errors.append(f"optimizer 参数重复引用: {duplicate_count}")
#         if wan_mode != "frozen" and torch.distributed.is_available() and torch.distributed.is_initialized():
#             ws = int(torch.distributed.get_world_size())
#             if ws > 1:
#                 dit_fsdp = bool(getattr(args, "dit_fsdp", False))
#                 use_sp = bool(getattr(args, "use_sp", False))
#                 if not (dit_fsdp or use_sp):
#                     errors.append(
#                         "Wan 可训练 + 多进程 需要启用 dit_fsdp 或 use_sp；"
#                         "当前仅 MQ-encoder 做了 DDP 包装，未启用时请使用单进程训练（WORLD_SIZE=1）"
#                     )
#                 # errors.append("当前仅实现 MQ-encoder 的 DDP 包装；Wan 可训练模式请使用单进程训练（WORLD_SIZE=1）")

#         if errors:
#             msg = " | ".join(errors)
#             if strict:
#                 raise RuntimeError(f"[AUDIT][FAIL][{stage}] {msg}")
#             if self.is_main_process:
#                 print(f"[AUDIT][WARN][{stage}] {msg}")

#     def post_wrap_ddp_audit(self) -> None:
#         # DDP 包装后再做一次 optimizer 与 trainable 参数一致性检查
#         if not hasattr(self.mq_encoder, "module"):
#             return
#         self._audit_runtime_trainability(stage="post_ddp")

#     def _log_grad_health_once(self):
#         if self._printed_grad_health:
#             return
#         module = self._mq_encoder_module()
#         connector_has_grad = False
#         mq_embed_has_grad = False
#         wan_has_grad = False
#         connector_grad_norm = 0.0
#         mq_embed_grad_norm = 0.0
#         wan_grad_norm = 0.0
#         mq_embed_added_grad_norm = 0.0
#         mq_embed_base_grad_norm = 0.0
#         mq_embed_boundary_grad_norm = 0.0
#         mq_embed_query_grad_norm = 0.0
#         try:
#             for _, p in module.mllm_model.connector.named_parameters():
#                 if p.grad is not None:
#                     connector_has_grad = True
#                     connector_grad_norm = float(p.grad.detach().float().norm().item())
#                     break
#             emb = module.mllm_model.mllm_backbone.get_input_embeddings()
#             if emb is not None and getattr(emb, "weight", None) is not None and emb.weight.grad is not None:
#                 mq_embed_has_grad = True
#                 g = emb.weight.grad.detach().float()
#                 mq_embed_grad_norm = float(g.norm().item())
#                 base_rows = int(getattr(module.mllm_model, "num_embeddings", 0))
#                 if g.ndim >= 2 and 0 < base_rows < int(g.shape[0]):
#                     mq_embed_base_grad_norm = float(g[:base_rows].norm().item())
#                     mq_embed_added_grad_norm = float(g[base_rows:].norm().item())
#                     boundary_end = min(base_rows + 2, int(g.shape[0]))
#                     query_end = min(boundary_end + int(self.args.num_metaqueries), int(g.shape[0]))
#                     if boundary_end > base_rows:
#                         mq_embed_boundary_grad_norm = float(g[base_rows:boundary_end].norm().item())
#                     if query_end > boundary_end:
#                         mq_embed_query_grad_norm = float(g[boundary_end:query_end].norm().item())
#         except Exception:
#             pass
#         try:
#             for p in self._wan_trainable_params():
#                 if p.grad is not None:
#                     wan_has_grad = True
#                     wan_grad_norm = float(p.grad.detach().float().norm().item())
#                     break
#         except Exception:
#             pass
#         print(
#             "[GRAD-CHECK] "
#             f"connector_has_grad={connector_has_grad} connector_grad_norm={connector_grad_norm:.4e} "
#             f"mq_embed_has_grad={mq_embed_has_grad} mq_embed_grad_norm={mq_embed_grad_norm:.4e} "
#             f"wan_has_grad={wan_has_grad} wan_grad_norm={wan_grad_norm:.4e} "
#             f"mq_embed_added_grad_norm={mq_embed_added_grad_norm:.4e} "
#             f"mq_embed_base_grad_norm={mq_embed_base_grad_norm:.4e} "
#             f"mq_embed_boundary_grad_norm={mq_embed_boundary_grad_norm:.4e} "
#             f"mq_embed_query_grad_norm={mq_embed_query_grad_norm:.4e}"
#         )
#         self._printed_grad_health = True

#     def _verify_train_context_injection_once(
#         self,
#         mq_feat: torch.Tensor,
#         aug_feat: torch.Tensor,
#     ) -> None:
#         if self._printed_context_inject_check:
#             return
#         mq_len = int(mq_feat.shape[0])
#         aug_len = int(aug_feat.shape[0])
#         if aug_len != mq_len:
#             raise RuntimeError(
#                 f"[VERIFY][TRAIN] MQ-only context 长度异常: aug={aug_len}, mq={mq_len}"
#             )
#         mq_ok = torch.allclose(
#             aug_feat.float(),
#             mq_feat.float(),
#             atol=1e-3,
#             rtol=1e-3,
#         )
#         if not mq_ok:
#             raise RuntimeError("[VERIFY][TRAIN] MQ-only context 未正确注入 Wan context")
#         if aug_len > self._aug_text_len:
#             raise RuntimeError(
#                 f"[VERIFY][TRAIN] aug_len 超出 text_len: aug={aug_len}, text_len={self._aug_text_len}"
#             )
#         print(
#             "[VERIFY][TRAIN] context 注入检查通过: "
#             f"mq_tokens={mq_len} aug_tokens={aug_len} model_text_len={self._aug_text_len}"
#         )
#         self._printed_context_inject_check = True

#     def _init_trainability_monitor(self):
#         self._param_monitor = []
#         total_sq = 0.0
#         sample_sq = 0.0
#         total_params = 0
#         named_params: List[Tuple[str, torch.nn.Parameter]] = []
#         mq_module = self._mq_encoder_module()
#         named_params.extend((f"mq_encoder.{n}", p) for n, p in mq_module.named_parameters() if p.requires_grad)
#         wan_model = getattr(self.wan, "model", None)
#         if isinstance(wan_model, nn.Module):
#             named_params.extend((f"wan.model.{n}", p) for n, p in wan_model.named_parameters() if p.requires_grad)
#         for name, p in named_params:
#             data = p.detach().float().view(-1)
#             numel = int(data.numel())
#             if numel <= 0:
#                 continue
#             sample_k = min(8, numel)
#             if sample_k == 1:
#                 idx = torch.zeros(1, dtype=torch.long)
#             else:
#                 idx = torch.linspace(0, numel - 1, steps=sample_k, dtype=torch.long)
#             init_vals = data.index_select(0, idx.to(data.device)).cpu()
#             self._param_monitor.append((name, p, idx.cpu(), init_vals))
#             total_sq += float(torch.sum(data * data).item())
#             sample_sq += float(torch.sum(init_vals * init_vals).item())
#             total_params += numel
#         self._trainable_param_count = total_params
#         self._init_trainable_norm = math.sqrt(max(total_sq, 0.0))
#         self._init_param_sample_norm = math.sqrt(max(sample_sq, 0.0))
#         if self.is_main_process:
#             print(
#                 "[VERIFY][TRAIN-INIT] "
#                 f"trainable_params={self._trainable_param_count:,} "
#                 f"init_param_norm={self._init_trainable_norm:.6f} "
#                 f"monitor_tensors={len(self._param_monitor)}"
#             )

#     def _collect_trainability_metrics(self):
#         sample_abs_sum = 0.0
#         sample_l2_sum = 0.0
#         sample_cur_sq_sum = 0.0
#         sample_count = 0
#         with torch.no_grad():
#             for _, p, idx_cpu, init_vals_cpu in self._param_monitor:
#                 data = p.detach().float().view(-1)
#                 idx = idx_cpu.to(data.device)
#                 now_vals = data.index_select(0, idx).cpu()
#                 diff = now_vals - init_vals_cpu
#                 sample_abs_sum += float(diff.abs().sum().item())
#                 sample_l2_sum += float(torch.sum(diff * diff).item())
#                 sample_cur_sq_sum += float(torch.sum(now_vals * now_vals).item())
#                 sample_count += int(diff.numel())
#         cur_sample_norm = math.sqrt(max(sample_cur_sq_sum, 0.0))
#         init_sample_norm = max(self._init_param_sample_norm, 1e-12)
#         return {
#             "train/param_sample_norm": float(cur_sample_norm),
#             "train/param_sample_norm_delta_ratio": float(abs(cur_sample_norm - self._init_param_sample_norm) / init_sample_norm),
#             "train/param_sample_abs_delta_mean": float(sample_abs_sum / max(sample_count, 1)),
#             "train/param_sample_l2_delta": float(math.sqrt(max(sample_l2_sum, 0.0))),
#             "train/trainable_param_count": int(self._trainable_param_count),
#         }

#     def _collect_cuda_memory_metrics(self):
#         if not (torch.cuda.is_available() and self.args.log_cuda_memory):
#             return {}
#         dit_idx = self.dev_dit.index if self.dev_dit.type == "cuda" else None
#         enc_idx = self.dev_enc.index if self.dev_enc.type == "cuda" else None

#         def _mem(prefix, dev_idx):
#             if dev_idx is None:
#                 return {}
#             return {
#                 f"train/cuda_{prefix}_alloc_mb": float(torch.cuda.memory_allocated(dev_idx) / 1024 / 1024),
#                 f"train/cuda_{prefix}_reserved_mb": float(torch.cuda.memory_reserved(dev_idx) / 1024 / 1024),
#                 f"train/cuda_{prefix}_max_alloc_mb": float(torch.cuda.max_memory_allocated(dev_idx) / 1024 / 1024),
#             }

#         metrics = {}
#         metrics.update(_mem("dit", dit_idx))
#         metrics.update(_mem("enc", enc_idx))
#         return metrics

#     def _append_metrics_jsonl(self, metrics):
#         if not self.is_main_process:
#             return
#         if not self._metrics_jsonl_path:
#             return
#         try:
#             path = Path(self._metrics_jsonl_path).expanduser().resolve()
#             path.parent.mkdir(parents=True, exist_ok=True)
#             with path.open("a", encoding="utf-8") as f:
#                 f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
#         except Exception as e:
#             print(f"[WARN] 写入 metrics_jsonl 失败: {e}")

#     def _record_metrics(self, metrics: Dict[str, Any]) -> None:
#         keep_keys = [
#             "train/step",
#             "train/loss_step",
#             "train/loss_denoise",
#             "train/loss_align_total",
#             "train/loss_align_t5_l2",
#             "train/loss_align_t5_cos",
#             "train/loss_align_t5_stats",
#             "train/loss_align_t5_gram",
#             "train/loss_align_t5_cka",
#             "train/loss_align_t5_ot",
#             "train/loss_align_img_preserve",
#             "train/loss_align_wan_func",
#             "train/loss_ema",
#             "train/lr",
#             "train/grad_norm",
#             "train/step_time_sec",
#             "train/samples_per_sec",
#             "train/param_sample_abs_delta_mean",
#             "train/param_sample_l2_delta",
#             "train/param_sample_norm_delta_ratio",
#             "train/skipped_step_count",
#             "train/oom_skip_count",
#             "train/error_skip_count",
#         ]
#         row = {k: metrics[k] for k in keep_keys if k in metrics}
#         self._metrics_history.append(row)

#     def _build_metrics_summary(self, step: int) -> Dict[str, Any]:
#         summary: Dict[str, Any] = {
#             "current_step": int(step),
#             "logged_steps": int(len(self._metrics_history)),
#             "metrics_jsonl_path": self._metrics_jsonl_path,
#             "skipped_step_count": int(self._skipped_step_count),
#             "oom_skip_count": int(self._oom_skip_count),
#             "error_skip_count": int(self._error_skip_count),
#         }
#         if self._metrics_history:
#             last = self._metrics_history[-1]
#             loss_vals = [float(m.get("train/loss_step", 0.0)) for m in self._metrics_history if "train/loss_step" in m]
#             step_time_vals = [float(m.get("train/step_time_sec", 0.0)) for m in self._metrics_history if "train/step_time_sec" in m]
#             sps_vals = [float(m.get("train/samples_per_sec", 0.0)) for m in self._metrics_history if "train/samples_per_sec" in m]
#             summary.update(
#                 {
#                     "step_first": int(self._metrics_history[0].get("train/step", 0)),
#                     "step_last": int(last.get("train/step", 0)),
#                     "loss_last": float(last.get("train/loss_step", 0.0)),
#                     "loss_ema_last": float(last.get("train/loss_ema", 0.0)),
#                     "lr_last": float(last.get("train/lr", 0.0)),
#                     "grad_norm_last": float(last.get("train/grad_norm", 0.0)),
#                     "loss_min": float(min(loss_vals) if loss_vals else 0.0),
#                     "loss_max": float(max(loss_vals) if loss_vals else 0.0),
#                     "step_time_sec_avg": float(sum(step_time_vals) / max(len(step_time_vals), 1)),
#                     "samples_per_sec_avg": float(sum(sps_vals) / max(len(sps_vals), 1)),
#                 }
#             )
#         if self._train_wall_start > 0:
#             summary["wall_time_sec"] = float(max(time.perf_counter() - self._train_wall_start, 0.0))
#         return summary

#     def _write_training_chain_manifest(self, output_dir: Path, final_checkpoint_path: str, final_step: int) -> None:
#         if not self.is_main_process:
#             return
#         output_dir = output_dir.expanduser().resolve()
#         payload = {
#             "before_checkpoint_path": self._train_before_checkpoint_path,
#             "final_checkpoint_path": str(Path(final_checkpoint_path).expanduser().resolve()),
#             "metrics_jsonl_path": self._metrics_jsonl_path,
#             "args": {str(k): _to_jsonable(v) for k, v in vars(self.args).items()},
#             "metrics_summary": self._build_metrics_summary(step=final_step),
#         }
#         _write_json(output_dir / "training_chain_manifest.json", payload)

#     def _wandb_config(self):
#         args = self.args
#         return {
#             "task": "wan_ti2v",
#             "learning_rate": args.learning_rate,
#             "num_train_steps": args.num_train_steps,
#             "warmup_steps": args.warmup_steps,
#             "lr_scheduler_type": args.lr_scheduler_type,
#             "cooldown_steps": args.cooldown_steps,
#             "lr_min_ratio": args.lr_min_ratio,
#             "enable_t5_alignment": args.enable_t5_alignment,
#             "t5_align_mode": args.t5_align_mode,
#             "t5_align_anchor_tokens": args.t5_align_anchor_tokens,
#             "lambda_t5_align_l2": args.lambda_t5_align_l2,
#             "lambda_t5_align_cos": args.lambda_t5_align_cos,
#             "lambda_t5_align_stats": args.lambda_t5_align_stats,
#             "t5_align_ot_epsilon": args.t5_align_ot_epsilon,
#             "t5_align_ot_iters": args.t5_align_ot_iters,
#             "enable_mq_image_preserve": args.enable_mq_image_preserve,
#             "lambda_mq_image_preserve": args.lambda_mq_image_preserve,
#             "mq_image_preserve_margin": args.mq_image_preserve_margin,
#             "enable_wan_func_distill": args.enable_wan_func_distill,
#             "lambda_wan_func_distill": args.lambda_wan_func_distill,
#             "wan_func_teacher_mode": args.wan_func_teacher_mode,
#             "batch_size": args.batch_size,
#             "gradient_accumulation_steps": args.gradient_accumulation_steps,
#             "max_grad_norm": args.max_grad_norm,
#             "frame_num": args.frame_num,
#             "max_area": args.max_area,
#             "num_metaqueries": args.num_metaqueries,
#             "connector_num_hidden_layers": args.connector_num_hidden_layers,
#             "dit_condition_mode": args.dit_condition_mode,
#             "mq_gradient_checkpointing": args.mq_gradient_checkpointing,
#             "train_mq_input_embeddings": args.train_mq_input_embeddings,
#             "null_caption_prob": args.null_caption_prob,
#             "null_image_prob": args.null_image_prob,
#             "wan_train_mode": args.wan_train_mode,
#             "wan_auto_full_mem_gb": args.wan_auto_full_mem_gb,
#             "wan_lr_ratio": args.wan_lr_ratio,
#             "wan_cond_name_pattern": args.wan_cond_name_pattern,
#             "t5_cpu": args.t5_cpu,
#             "dit_fsdp": args.dit_fsdp,
#             "t5_fsdp": args.t5_fsdp,
#             "use_sp": args.use_sp,
#             "aggressive_empty_cache": args.aggressive_empty_cache,
#             "seed": args.seed,
#             "save_steps": args.save_steps,
#             "log_steps": args.log_steps,
#             "enable_loss_early_stop": args.enable_loss_early_stop,
#             "loss_early_stop_min_step": args.loss_early_stop_min_step,
#             "loss_early_stop_threshold": args.loss_early_stop_threshold,
#             "log_every_step": args.log_every_step,
#             "wandb_log_every_step": args.wandb_log_every_step,
#             "metrics_jsonl_path": args.metrics_jsonl_path,
#             "log_cuda_memory": args.log_cuda_memory,
#             "output_dir": args.output_dir,
#             "local_openvid_video_root": args.local_openvid_video_root,
#             "local_openvid_csv_path": args.local_openvid_csv_path,
#             "local_openvid_limit": args.local_openvid_limit,
#             "local_openvid_hd_video_root": args.local_openvid_hd_video_root,
#             "local_openvid_hd_csv_path": args.local_openvid_hd_csv_path,
#             "local_openvid_hd_limit": args.local_openvid_hd_limit,
#             "wan_checkpoint_dir": args.wan_checkpoint_dir,
#             "qwen3vl_model_id": args.qwen3vl_model_id,
#         }

#     def _init_wandb(self):
#         args = self.args
#         if not getattr(args, "wandb_enabled", False):
#             return
#         if not self.is_main_process:
#             return
#         if args.wandb_api_key:
#             os.environ["WANDB_API_KEY"] = args.wandb_api_key
#         try:
#             import wandb
#         except ImportError:
#             print("[W&B] 未安装 wandb, 已跳过日志记录")
#             return
#         run_name = args.wandb_run_name.strip() or f"wan-ti2v-metaquery-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
#         tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
#         self.wandb = wandb
#         self.wandb_run = wandb.init(
#             project=args.wandb_project,
#             entity=args.wandb_entity or None,
#             name=run_name,
#             mode=args.wandb_mode,
#             config=self._wandb_config(),
#             tags=tags or None,
#         )
#         print(f"[W&B] 已初始化: project={args.wandb_project}, run={run_name}")

#     def _load_models(self):
#         """加载所有模型。"""
#         args = self.args

#         # ── 1. Wan TI2V Pipeline ─────────────────────────────────────────
#         print("\n[1/3] 加载 Wan TI2V Pipeline...")
#         from wan import WanTI2V
#         from wan.configs import WAN_CONFIGS

#         config = WAN_CONFIGS['ti2v-5B']
#         runtime_rank = (
#             torch.distributed.get_rank()
#             if torch.distributed.is_available() and torch.distributed.is_initialized()
#             else 0
#         )
#         self.wan = WanTI2V(
#             config=config,
#             checkpoint_dir=args.wan_checkpoint_dir,
#             device_id=args.dit_device,
#             rank=runtime_rank,
#             t5_fsdp=args.t5_fsdp,
#             dit_fsdp=args.dit_fsdp,
#             use_sp=args.use_sp,
#             t5_cpu=args.t5_cpu,
#             init_on_cpu=not args.no_init_on_cpu,
#             convert_model_dtype=args.convert_model_dtype,
#         )

#         # DiT 冻结；FSDP/SP 路径不再显式 .to，避免破坏分片包装
#         if not (args.dit_fsdp or args.use_sp):
#             self.wan.model.to(self.dev_dit)
#         self.wan.model.eval().requires_grad_(False)
#         t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
#         vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
#         if vae_model is None:
#             vae_model = getattr(self.wan, "vae", None)
#         self._force_freeze(t5_model)
#         self._force_freeze(vae_model)

#         self.wan_config = config
#         self.text_len = config.text_len  # 512
#         print(f"  ✅ Wan TI2V 5B 已加载, text_len={self.text_len}")

#         # ── 2. MetaQuery Encoder (直接输出 4096) ─────────────────────────
#         print("\n[2/3] 加载 MetaQuery Encoder (→4096)...")
#         # 统一使用 train_connector_for_wan.py 中的实现，避免同名类双份定义导致“改了不生效”。
#         from train_connector_for_wan import MetaQueryEncoderForWan as SharedMetaQueryEncoderForWan
#         self.mq_encoder = SharedMetaQueryEncoderForWan(
#             qwen3vl_model_id=args.qwen3vl_model_id,
#             num_metaqueries=args.num_metaqueries,
#             connector_num_hidden_layers=args.connector_num_hidden_layers,
#             gradient_checkpointing=args.mq_gradient_checkpointing,
#             train_input_embeddings=args.train_mq_input_embeddings,
#             dtype=torch.bfloat16,
#             device=f"cuda:{args.encoder_device}",
#         )
#         print(f"  ✅ Encoder实现来源: {self.mq_encoder.__class__.__module__}.{self.mq_encoder.__class__.__name__}")
#         self.mq_encoder.train()
#         if args.resume_mq_encoder_path:
#             state, resolved_path = load_mq_encoder_state(
#                 args.resume_mq_encoder_path,
#                 map_location="cpu",
#             )
#             missing, unexpected = self.mq_encoder.load_state_dict(state, strict=False)
#             print(f"  ✅ 已加载初始权重: {resolved_path}")
#             print(f"     missing={len(missing)}, unexpected={len(unexpected)}")
#         print(f"  ✅ MetaQuery Encoder 已加载")

#         # ── 3. 验证维度 ──────────────────────────────────────────────────
#         print("\n[3/3] 验证维度对齐...")
#         wan_text_dim = self.wan.model.text_dim  # 4096
#         mq_out_dim = self.mq_encoder.wan_text_dim  # 4096
#         assert wan_text_dim == mq_out_dim, (
#             f"维度不匹配! Wan text_dim={wan_text_dim}, MQ out={mq_out_dim}"
#         )
#         print(f"  ✅ MQ output dim = Wan text_dim = {wan_text_dim}")

#         # MQ-only: DiT text_len 仅容纳 MQ tokens
#         self._orig_text_len = self.wan.model.text_len
#         self._aug_text_len = args.num_metaqueries
#         print(f"  ✅ text_len(MQ-only): {self._orig_text_len} → {self._aug_text_len}")
#         self._configure_wan_trainable_params()

#     def _setup_optimizer(self):
#         """设置优化器和学习率调度。"""
#         args = self.args

#         mq_params = self._mq_trainable_params()
#         wan_params = self._wan_trainable_params()
#         trainable_params = self._all_trainable_params()
#         print(f"\n[Optimizer] 可训练参数组:")
#         print(f"  Connector + MQ Embeddings: {sum(p.numel() for p in mq_params) / 1e6:.1f}M")
#         print(f"  Wan DiT (mode={self._effective_wan_train_mode}): {sum(p.numel() for p in wan_params) / 1e6:.1f}M")
#         print(f"  Total trainable: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")
#         if len(trainable_params) <= 0:
#             raise RuntimeError("无可训练参数：请检查 MQ/Wan 训练配置。")

#         param_groups: List[Dict[str, Any]] = []
#         if mq_params:
#             param_groups.append(
#                 {
#                     "name": "mq",
#                     "params": mq_params,
#                     "lr": float(args.learning_rate),
#                 }
#             )
#         if wan_params:
#             param_groups.append(
#                 {
#                     "name": "wan",
#                     "params": wan_params,
#                     "lr": float(args.learning_rate) * float(getattr(args, "wan_lr_ratio", 1.0)),
#                 }
#             )

#         self.optimizer = torch.optim.AdamW(
#             param_groups,
#             betas=(0.9, 0.95),
#             weight_decay=0.1,
#             eps=1e-8,
#         )

#         def lr_lambda(step):
#             warmup = max(int(args.warmup_steps), 0)
#             total = max(int(args.num_train_steps), 1)
#             cooldown = int(getattr(args, "cooldown_steps", -1))
#             if cooldown < 0:
#                 cooldown = warmup
#             cooldown = max(cooldown, 0)
#             warmup = min(warmup, total)
#             cooldown = min(cooldown, max(total - warmup, 0))

#             if step < warmup:
#                 return step / max(1, warmup)
#             if args.lr_scheduler_type == "constant_with_warmup":
#                 return 1.0
#             if args.lr_scheduler_type == "warmup_hold_cooldown":
#                 cooldown_start = total - cooldown
#                 if cooldown <= 0 or step < cooldown_start:
#                     return 1.0
#                 progress = (step - cooldown_start) / max(1, cooldown)
#                 progress = min(max(progress, 0.0), 1.0)
#                 return 1.0 - (1.0 - float(args.lr_min_ratio)) * progress
#             progress = (step - warmup) / max(1, total - warmup)
#             return max(float(args.lr_min_ratio), 0.5 * (1.0 + math.cos(math.pi * progress)))

#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

#     def _encode_text(self, prompts):
#         """T5 编码文本"""
#         with torch.no_grad():
#             if not self.args.t5_cpu and not self.args.t5_fsdp:
#                 self.wan.text_encoder.model.to(self.dev_dit)
#                 context = self.wan.text_encoder(prompts, self.dev_dit)
#             else:
#                 context = self.wan.text_encoder(prompts, torch.device("cpu"))
#                 context = [t.to(self.dev_dit, dtype=torch.bfloat16) for t in context]
#         return context  # List[Tensor], each [text_len, 4096]

#     @staticmethod
#     def _resize_token_sequence(seq: torch.Tensor, out_tokens: int) -> torch.Tensor:
#         """
#         将 [L, D] token 序列重采样到 [out_tokens, D]。
#         使用线性插值仅做 teacher 侧长度对齐，不引入额外可训练参数。
#         """
#         if seq.dim() != 2:
#             raise ValueError(f"expect [L, D], got shape={tuple(seq.shape)}")
#         out_tokens = max(1, int(out_tokens))
#         if int(seq.shape[0]) == out_tokens:
#             return seq
#         # F.interpolate 期望 [N, C, L]
#         x = seq.transpose(0, 1).unsqueeze(0).float()
#         x = F.interpolate(x, size=out_tokens, mode="linear", align_corners=False)
#         return x.squeeze(0).transpose(0, 1)

#     @staticmethod
#     def _token_gram_matrix(tokens: torch.Tensor) -> torch.Tensor:
#         """
#         计算 token 关系矩阵（Gram）。
#         输入: [B, T, D]，输出: [B, T, T]
#         """
#         if tokens.dim() != 3:
#             raise ValueError(f"expect [B, T, D], got shape={tuple(tokens.shape)}")
#         x = tokens - tokens.mean(dim=1, keepdim=True)
#         x = F.normalize(x, p=2, dim=-1, eps=1e-6)
#         return torch.matmul(x, x.transpose(1, 2))

#     @staticmethod
#     def _linear_cka_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         线性 CKA 损失，返回 1-CKA（越小越好）。
#         输入: x/y [B, T, D]
#         """
#         if x.shape != y.shape:
#             raise ValueError(f"CKA shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")
#         x_c = x - x.mean(dim=1, keepdim=True)
#         y_c = y - y.mean(dim=1, keepdim=True)
#         kx = torch.matmul(x_c, x_c.transpose(1, 2))
#         ky = torch.matmul(y_c, y_c.transpose(1, 2))
#         hsic = (kx * ky).sum(dim=(1, 2))
#         denom = torch.sqrt(
#             kx.square().sum(dim=(1, 2)).clamp_min(1e-12)
#             * ky.square().sum(dim=(1, 2)).clamp_min(1e-12)
#         )
#         cka = hsic / denom.clamp_min(1e-12)
#         return (1.0 - cka.clamp(-1.0, 1.0)).mean()

#     @staticmethod
#     def _sinkhorn_ot_cost(
#         src_tokens: torch.Tensor,
#         tgt_tokens: torch.Tensor,
#         epsilon: float = 0.05,
#         iters: int = 25,
#     ) -> torch.Tensor:
#         """
#         Sinkhorn OT 软匹配代价（排列无关）。
#         输入: src/tgt [B, T, D]
#         输出: 标量（batch 平均 OT cost）
#         """
#         if src_tokens.dim() != 3 or tgt_tokens.dim() != 3:
#             raise ValueError(
#                 f"Sinkhorn expect [B,T,D], got src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
#             )
#         if int(src_tokens.shape[0]) != int(tgt_tokens.shape[0]) or int(src_tokens.shape[2]) != int(tgt_tokens.shape[2]):
#             raise ValueError(
#                 f"Sinkhorn shape mismatch: src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
#             )
#         bsz, n_tok, _ = src_tokens.shape
#         m_tok = int(tgt_tokens.shape[1])
#         if n_tok <= 0 or m_tok <= 0:
#             return src_tokens.new_zeros(())

#         cost = torch.cdist(src_tokens, tgt_tokens, p=2).pow(2)  # [B, N, M]
#         eps = max(float(epsilon), 1e-6)
#         kernel = torch.exp(-cost / eps).clamp_min(1e-12)
#         a = src_tokens.new_full((bsz, n_tok), 1.0 / float(n_tok))
#         b = src_tokens.new_full((bsz, m_tok), 1.0 / float(m_tok))
#         u = torch.ones_like(a)
#         v = torch.ones_like(b)
#         kernel_t = kernel.transpose(1, 2)

#         n_iter = max(int(iters), 1)
#         for _ in range(n_iter):
#             kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
#             u = a / kv
#             ktu = torch.bmm(kernel_t, u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
#             v = b / ktu

#         plan = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)  # [B, N, M]
#         return (plan * cost).sum(dim=(1, 2)).mean()

#     def _compute_mq_aux_losses(
#         self,
#         captions: List[str],
#         mq_refs: List[Any],
#         mq_features: torch.Tensor,
#         t5_context: List[torch.Tensor] | None = None,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         计算 MQ 辅助约束：
#         1) T5 对齐（支持 anchor/Gram+CKA/Sinkhorn）
#         2) T5 统计对齐（均值/方差）
#         3) 图像保持（可选）：有图条件需与 text-only MQ 保持最小间隔
#         """
#         device = self.dev_dit
#         zero = mq_features.new_zeros(())
#         out = {
#             "t5_l2": zero,
#             "t5_cos": zero,
#             "t5_stats": zero,
#             "t5_gram": zero,
#             "t5_cka": zero,
#             "t5_ot": zero,
#             "image_preserve": zero,
#             "total": zero,
#         }
#         args = self.args
#         need_t5 = bool(args.enable_t5_alignment) and (
#             float(args.lambda_t5_align_l2) > 0.0
#             or float(args.lambda_t5_align_cos) > 0.0
#             or float(args.lambda_t5_align_stats) > 0.0
#         )
#         need_img = bool(args.enable_mq_image_preserve) and float(args.lambda_mq_image_preserve) > 0.0
#         if not (need_t5 or need_img):
#             return out

#         mq_float = mq_features.to(device=device, dtype=torch.float32)
#         tokens = int(mq_float.shape[1])
#         hidden = int(mq_float.shape[2])
#         anchor_tokens = max(1, min(int(args.t5_align_anchor_tokens), tokens))

#         if need_t5:
#             with torch.no_grad():
#                 teacher_ctx = t5_context if t5_context is not None else self._encode_text(captions)
#                 pooled_t5 = []
#                 for t5_seq in teacher_ctx:
#                     # t5_seq: [L_t5, 4096]
#                     t5_seq_f = t5_seq.to(device=device, dtype=torch.float32)
#                     if int(t5_seq_f.shape[-1]) != hidden:
#                         raise RuntimeError(
#                             f"T5 hidden={int(t5_seq_f.shape[-1])} 与 MQ hidden={hidden} 不一致"
#                         )
#                     pooled_t5.append(self._resize_token_sequence(t5_seq_f, tokens))
#                 t5_teacher = torch.stack(pooled_t5, dim=0)  # [B, tokens, 4096]

#             align_mode = str(getattr(args, "t5_align_mode", "gram_cka")).strip().lower()
#             if align_mode == "anchor":
#                 mq_anchor = mq_float[:, :anchor_tokens, :]
#                 t5_anchor = t5_teacher[:, :anchor_tokens, :]
#                 out["t5_l2"] = F.mse_loss(mq_anchor, t5_anchor)

#                 mq_anchor_flat = mq_anchor.reshape(-1, hidden)
#                 t5_anchor_flat = t5_anchor.reshape(-1, hidden)
#                 cos_sim = F.cosine_similarity(mq_anchor_flat, t5_anchor_flat, dim=-1).mean()
#                 out["t5_cos"] = (1.0 - cos_sim)
#             elif align_mode == "gram_cka":
#                 mq_gram = self._token_gram_matrix(mq_float)
#                 t5_gram = self._token_gram_matrix(t5_teacher)
#                 out["t5_gram"] = F.mse_loss(mq_gram, t5_gram)
#                 out["t5_cka"] = self._linear_cka_loss(mq_float, t5_teacher)
#                 # 复用旧命名，保持日志/脚本兼容
#                 out["t5_l2"] = out["t5_gram"]
#                 out["t5_cos"] = out["t5_cka"]
#             elif align_mode == "sinkhorn_ot":
#                 out["t5_ot"] = self._sinkhorn_ot_cost(
#                     mq_float,
#                     t5_teacher,
#                     epsilon=float(getattr(args, "t5_align_ot_epsilon", 0.05)),
#                     iters=int(getattr(args, "t5_align_ot_iters", 25)),
#                 )
#                 out["t5_l2"] = out["t5_ot"]
#             else:
#                 raise ValueError(f"Unknown --t5_align_mode: {align_mode}")

#             mq_mean = mq_float.mean(dim=1)
#             mq_std = mq_float.std(dim=1, unbiased=False)
#             t5_mean = t5_teacher.mean(dim=1)
#             t5_std = t5_teacher.std(dim=1, unbiased=False)
#             out["t5_stats"] = F.mse_loss(mq_mean, t5_mean) + F.mse_loss(mq_std, t5_std)

#         if need_img:
#             has_ref = torch.tensor(
#                 [1 if ref is not None else 0 for ref in mq_refs],
#                 device=device,
#                 dtype=torch.bool,
#             )
#             if bool(torch.any(has_ref).item()):
#                 with torch.no_grad():
#                     mq_text_only = self.mq_encoder(captions, None).to(device=device, dtype=torch.float32)
#                 diff = mq_float[has_ref] - mq_text_only[has_ref]
#                 # 每样本的 token+channel RMS 距离
#                 rms = torch.sqrt(torch.mean(diff * diff, dim=(1, 2)) + 1e-8)
#                 margin = float(args.mq_image_preserve_margin)
#                 out["image_preserve"] = F.relu(margin - rms).mean()

#         out["total"] = (
#             float(args.lambda_t5_align_l2) * out["t5_l2"]
#             + float(args.lambda_t5_align_cos) * out["t5_cos"]
#             + float(args.lambda_t5_align_stats) * out["t5_stats"]
#             + float(args.lambda_mq_image_preserve) * out["image_preserve"]
#         )
#         return out

#     def _encode_video(self, video_tensors):
#         """VAE 编码视频 → latent"""
#         with torch.no_grad():
#             # video_tensors: [B, 3, T, H, W] or list of [3, T, H, W]
#             latents = []
#             for v in video_tensors:
#                 # v: [3, T, H, W] → VAE expects this format
#                 z = self.wan.vae.encode([v.to(self.dev_dit, dtype=torch.bfloat16)])
#                 latents.append(z[0])  # z[0]: [C_z, T', H', W']
#         return latents

#     def _encode_first_frame(self, first_frame_tensor):
#         """VAE 编码参考图第一帧 → i2v condition latent"""
#         with torch.no_grad():
#             # first_frame: [3, H, W] → [3, 1, H, W]
#             ff = first_frame_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
#             z = self.wan.vae.encode([ff])
#         return z[0]  # [C_z, 1, H', W']

#     def _resolve_train_ref_anchor_mode(self) -> str:
#         """
#         返回当前 batch 实际使用的锚定模式。
#         - none / animate_like: 直接使用
#         - mixed50: 按 optimizer step 交替 none / animate_like，保证长期约 50/50
#         """
#         mode = str(getattr(self.args, "train_ref_anchor_mode", "none")).strip().lower()
#         if mode in ("none", "animate_like"):
#             return mode
#         if mode == "mixed50":
#             use_animate = (self._train_ref_anchor_mixed_counter % 2 == 1)
#             self._train_ref_anchor_mixed_counter += 1
#             return "animate_like" if use_animate else "none"
#         raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

#     def _train_ref_anchor_alpha(self, t_norm: torch.Tensor, mode: str | None = None) -> torch.Tensor:
#         """
#         训练期首帧软锚定系数（0~1）。
#         说明：
#         - none: 始终 0，不改动训练行为
#         - animate_like: 高噪声(早期)强锚定，随后余弦衰减到 0
#         """
#         if mode is None:
#             mode = self._resolve_train_ref_anchor_mode()
#         if mode == "none":
#             return torch.zeros_like(t_norm, dtype=torch.float32)
#         if mode != "animate_like":
#             raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

#         alpha0 = float(getattr(self.args, "train_ref_anchor_alpha0", 0.95))
#         warmup_ratio = float(getattr(self.args, "train_ref_anchor_warmup_ratio", 0.35))
#         alpha0 = max(0.0, min(1.0, alpha0))
#         warmup_ratio = max(0.0, min(1.0, warmup_ratio))
#         if warmup_ratio <= 0.0 or alpha0 <= 0.0:
#             return torch.zeros_like(t_norm, dtype=torch.float32)

#         start_t = 1.0 - warmup_ratio
#         alpha = torch.zeros_like(t_norm, dtype=torch.float32)
#         active = t_norm >= start_t
#         if not torch.any(active):
#             return alpha
#         u = ((t_norm[active] - start_t) / max(warmup_ratio, 1e-6)).clamp(0.0, 1.0)
#         alpha[active] = alpha0 * 0.5 * (1.0 - torch.cos(math.pi * u))
#         return alpha

#     @staticmethod
#     def _frames_to_latent_slots(frame_count: int, stride_t: int) -> int:
#         """像素帧数 -> latent 时间槽数（与 VAE 时间下采样保持一致）"""
#         f = max(0, int(frame_count))
#         if f <= 0:
#             return 0
#         return int((f - 1) // max(int(stride_t), 1) + 1)

#     def _encode_ref_image_to_latent(
#         self,
#         ref_img: Image.Image | None,
#         latent_h: int,
#         latent_w: int,
#         z_channels: int,
#     ) -> torch.Tensor:
#         """
#         将参考图编码为 1 帧 reference latent。
#         若 ref_img 缺失，返回零 reference latent。
#         """
#         if ref_img is None:
#             return torch.zeros(
#                 z_channels, 1, latent_h, latent_w,
#                 device=self.dev_dit, dtype=torch.float32,
#             )
#         target_h = int(latent_h * self.wan_config.vae_stride[1])
#         target_w = int(latent_w * self.wan_config.vae_stride[2])
#         ref_resized = ref_img.resize((target_w, target_h), Image.LANCZOS)
#         ref_np = np.array(ref_resized).astype(np.float32)
#         ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1) / 127.5 - 1.0
#         ref_5d = ref_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
#         with torch.no_grad():
#             ref_lat = self.wan.vae.encode([ref_5d])[0]
#         return ref_lat.float()

#     def _compute_wan_func_distill_loss(
#         self,
#         model_output: List[torch.Tensor],
#         x_inputs: List[torch.Tensor],
#         timesteps_wan: torch.Tensor,
#         max_seq_len: int,
#         t5_context: List[torch.Tensor],
#         mq_features: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         函数级蒸馏:
#             L_func = MSE( pred_mq(x_t,t), pred_t5(x_t,t) )
#         其中 pred_t5 由 frozen Wan + T5 条件生成（teacher no-grad）。
#         """
#         args = self.args
#         mode = str(getattr(args, "wan_func_teacher_mode", "t5_only")).strip().lower()
#         if mode not in {"t5_only", "t5_plus_mq"}:
#             raise ValueError(f"Unknown --wan_func_teacher_mode: {mode}")

#         teacher_context: List[torch.Tensor] = []
#         for i, t5_seq in enumerate(t5_context):
#             t5_feat = t5_seq.to(self.dev_dit, dtype=torch.bfloat16)
#             if mode == "t5_plus_mq":
#                 mq_feat = mq_features[i].detach().to(self.dev_dit, dtype=torch.bfloat16)
#                 t5_feat = torch.cat([mq_feat, t5_feat], dim=0)
#             teacher_context.append(t5_feat)

#         if not teacher_context:
#             return mq_features.new_zeros(())

#         teacher_text_len = max(int(ctx.shape[0]) for ctx in teacher_context)
#         cur_text_len = int(self.wan.model.text_len)
#         self.wan.model.text_len = teacher_text_len
#         try:
#             with torch.no_grad():
#                 with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#                     teacher_output = self.wan.model(
#                         x_inputs,
#                         t=timesteps_wan,
#                         context=teacher_context,
#                         seq_len=max_seq_len,
#                     )
#         finally:
#             self.wan.model.text_len = cur_text_len

#         loss = 0.0
#         valid = 0
#         for i in range(len(model_output)):
#             pred_mq = model_output[i].float()
#             pred_t5 = teacher_output[i].float()
#             loss = loss + F.mse_loss(pred_mq, pred_t5)
#             valid += 1
#         if valid <= 0:
#             return mq_features.new_zeros(())
#         return loss / valid

#     def _compute_loss(self, batch):
#         """
#         计算一个 batch 的 Flow Matching 损失。

#         训练默认使用 t2v 模式 (无第一帧蒙版/无首帧锚定)。
#         可通过 --train_ref_anchor_mode 在 x_t 注入 animate-like 首帧软锚定，
#         以缓解与 i2v 推理分布不一致问题。
#         """
#         args = self.args
#         captions = batch["caption"]
#         videos = batch["video"]         # list of [3, T, H, W]
#         mq_refs = batch["mq_ref_image"]  # list of PIL or None
#         B = len(captions)
#         self._last_loss_denoise = 0.0
#         self._last_loss_aux_align_total = 0.0
#         self._last_loss_aux_t5_l2 = 0.0
#         self._last_loss_aux_t5_cos = 0.0
#         self._last_loss_aux_t5_stats = 0.0
#         self._last_loss_aux_t5_gram = 0.0
#         self._last_loss_aux_t5_cka = 0.0
#         self._last_loss_aux_t5_ot = 0.0
#         self._last_loss_aux_image_preserve = 0.0
#         self._last_loss_aux_wan_func = 0.0

#         # ── 1. MetaQuery 编码 (在 encoder 设备上, 有梯度) ────────────────
#         mq_images = []
#         for ref in mq_refs:
#             if ref is not None:
#                 mq_images.append([ref])
#             else:
#                 mq_images.append(None)

#         all_none = all(img is None for img in mq_images)
#         if all_none:
#             mq_features = self.mq_encoder(captions, None)
#         else:
#             for i, img in enumerate(mq_images):
#                 if img is None:
#                     mq_images[i] = [Image.new("RGB", (224, 224))]
#             mq_features = self.mq_encoder(captions, mq_images)
#         # mq_features: [B, 256, 4096], 有梯度

#         # ── 2. MQ-only 注入 DiT context ─────────────────────────────────
#         augmented_context = []
#         for i in range(B):
#             mq_feat = mq_features[i].to(self.dev_dit, dtype=torch.bfloat16)
#             aug = mq_feat
#             if i == 0:
#                 self._verify_train_context_injection_once(mq_feat, aug)
#             augmented_context.append(aug)

#         # ── 4. VAE 编码视频 → latent (无梯度) ───────────────────────────
#         with torch.no_grad():
#             latents = self._encode_video(videos)
#             # latents: list of [C_z, T', H', W']

#         # ── 4. 采样噪声和时间步, 构建 Flow Matching 目标 ─────────────────
#         patch_size = self.wan_config.patch_size
#         stride_t = int(self.wan_config.vae_stride[0])

#         first_frame_condition_enabled = bool(
#             getattr(args, "enable_ti2v_first_frame_condition", True)
#         )
#         mode_cfg = str(getattr(args, "train_video_conditioning_mode", "legacy_t2v")).strip().lower()
#         if mode_cfg not in ("legacy_t2v", "wan_animate_slot"):
#             mode_cfg = "legacy_t2v"
#         effective_video_mode = mode_cfg if first_frame_condition_enabled else "mq_only"

#         x_inputs = []
#         timestep_rows = []
#         target_list = []
#         prefix_slots_list = []
#         target_slots_list = []
#         max_seq_len = 0
#         ref_anchor_alpha_sum = 0.0
#         ref_anchor_applied = 0
#         ref_anchor_mode_effective = "none"

#         for i, lat in enumerate(latents):
#             C, T, H, W = lat.shape
#             lat = lat.float()
#             x0_for_fm = lat
#             prefix_slots_i = 0

#             # Wan 侧首帧条件：在训练阶段将参考图转换为 latent 并注入
#             ref_lat = None
#             if first_frame_condition_enabled:
#                 ref_lat = self._encode_ref_image_to_latent(
#                     mq_refs[i],
#                     latent_h=H,
#                     latent_w=W,
#                     z_channels=C,
#                 ).to(self.dev_dit, dtype=torch.float32)

#             if effective_video_mode == "wan_animate_slot":
#                 ref_slots = self._frames_to_latent_slots(
#                     int(getattr(args, "train_animate_ref_frames", 1)),
#                     stride_t=stride_t,
#                 )
#                 temporal_slots = self._frames_to_latent_slots(
#                     int(getattr(args, "train_animate_temporal_frames", 0)),
#                     stride_t=stride_t,
#                 )
#                 conditional_slots = self._frames_to_latent_slots(
#                     int(getattr(args, "train_animate_conditional_frames", 0)),
#                     stride_t=stride_t,
#                 )
#                 ref_slots = max(0, int(ref_slots))
#                 temporal_slots = max(0, int(temporal_slots))
#                 conditional_slots = max(0, int(conditional_slots))
#                 prefix_slots_i = ref_slots + temporal_slots + conditional_slots
#                 if prefix_slots_i > 0:
#                     prefix_chunks = []
#                     if ref_slots > 0:
#                         if ref_lat is None:
#                             ref_prefix = torch.zeros(
#                                 C, ref_slots, H, W, device=self.dev_dit, dtype=torch.float32
#                             )
#                         else:
#                             ref_prefix = ref_lat.repeat(1, ref_slots, 1, 1)
#                         prefix_chunks.append(ref_prefix)
#                     if temporal_slots > 0:
#                         prefix_chunks.append(
#                             torch.zeros(
#                                 C, temporal_slots, H, W, device=self.dev_dit, dtype=torch.float32
#                             )
#                         )
#                     if conditional_slots > 0:
#                         prefix_chunks.append(
#                             torch.zeros(
#                                 C, conditional_slots, H, W, device=self.dev_dit, dtype=torch.float32
#                             )
#                         )
#                     x0_prefix = torch.cat(prefix_chunks, dim=1)
#                     x0_for_fm = torch.cat([x0_prefix, lat], dim=1)

#             T_full = int(x0_for_fm.shape[1])
#             tokens_per_frame = int(math.ceil((H * W) / (patch_size[1] * patch_size[2])))
#             seq_len_i = int(tokens_per_frame * T_full)
#             max_seq_len = max(max_seq_len, seq_len_i)

#             t_val = torch.rand(1, device=self.dev_dit, dtype=torch.float32)
#             noise = torch.randn_like(x0_for_fm, dtype=torch.float32)

#             # Flow matching: x_t = (1-t) * x_0 + t * noise
#             sigma = t_val.view(-1, 1, 1, 1)
#             noisy_lat = (1.0 - sigma) * x0_for_fm + sigma * noise

#             if effective_video_mode == "legacy_t2v" and ref_lat is not None:
#                 ref_mode = str(getattr(self, "_current_train_ref_anchor_mode", "none")).strip().lower()
#                 if ref_mode not in ("none", "animate_like"):
#                     ref_mode = self._resolve_train_ref_anchor_mode()
#                 alpha_tensor = self._train_ref_anchor_alpha(t_val, mode=ref_mode)
#                 alpha_scalar = float(alpha_tensor.item())
#                 ref_anchor_mode_effective = ref_mode
#                 if alpha_scalar > 0.0:
#                     noisy_lat[:, :1] = (1.0 - alpha_scalar) * noisy_lat[:, :1] + alpha_scalar * ref_lat
#                     ref_anchor_alpha_sum += alpha_scalar
#                     ref_anchor_applied += 1

#             # 目标: noise - x_0 (velocity)
#             velocity = noise - x0_for_fm

#             # token 级 timestep：MQ-only 下全部 token 共享 t
#             t_scalar = float((t_val * self.wan.num_train_timesteps).item())
#             t_row = torch.full((seq_len_i,), t_scalar, device=self.dev_dit, dtype=torch.float32)
#             if (
#                 effective_video_mode == "wan_animate_slot"
#                 and prefix_slots_i > 0
#                 and bool(getattr(args, "train_animate_preserve_timestep_zero", True))
#             ):
#                 prefix_token_count = min(seq_len_i, int(prefix_slots_i * tokens_per_frame))
#                 if prefix_token_count > 0:
#                     t_row[:prefix_token_count] = 0.0

#             x_inputs.append(noisy_lat)
#             target_list.append(velocity)
#             prefix_slots_list.append(prefix_slots_i)
#             timestep_rows.append(t_row)
#             target_slots_list.append(T)

#         # 拼接 timestep → [B, max_seq_len]
#         padded_rows = []
#         for row in timestep_rows:
#             pad_len = max_seq_len - int(row.numel())
#             if pad_len > 0:
#                 pad_val = float(row[-1].item()) if row.numel() > 0 else 0.0
#                 row = torch.cat([row, row.new_full((pad_len,), pad_val)], dim=0)
#             padded_rows.append(row)
#         timesteps_wan = torch.stack(padded_rows, dim=0).to(self.dev_dit)

#         self._last_train_ref_anchor_alpha_mean = (
#             float(ref_anchor_alpha_sum / ref_anchor_applied) if ref_anchor_applied > 0 else 0.0
#         )
#         self._last_train_ref_anchor_applied = int(ref_anchor_applied)
#         self._last_train_ref_anchor_effective_mode = (
#             ref_anchor_mode_effective if ref_anchor_applied > 0 else "none"
#         )
#         self._last_train_video_conditioning_mode = str(effective_video_mode)
#         self._last_train_prefix_latent_slots = int(
#             round(sum(prefix_slots_list) / max(len(prefix_slots_list), 1))
#         )
#         self._last_train_target_latent_slots = int(round(sum(target_slots_list) / max(len(target_slots_list), 1)))
#         self._last_train_prefix_loss_dropped = 0

#         # ── 5. MQ-only text_len + DiT forward ───────────────────────────
#         orig_text_len = self.wan.model.text_len
#         self.wan.model.text_len = self._aug_text_len

#         try:
#             with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#                 model_output = self.wan.model(
#                     x_inputs,
#                     t=timesteps_wan,
#                     context=augmented_context,
#                     seq_len=max_seq_len,
#                 )

#             # ── 6. 计算去噪主损失 ──────────────────────────────────────────
#             denoise_loss = 0.0
#             valid_terms = 0
#             drop_prefix_loss = bool(getattr(args, "train_animate_drop_prefix_loss", True))
#             dropped_prefix_terms = 0
#             for i in range(B):
#                 pred = model_output[i].float()
#                 target = target_list[i]
#                 prefix_slots_i = int(prefix_slots_list[i]) if i < len(prefix_slots_list) else 0
#                 if (
#                     effective_video_mode == "wan_animate_slot"
#                     and drop_prefix_loss
#                     and prefix_slots_i > 0
#                 ):
#                     if pred.shape[1] <= prefix_slots_i or target.shape[1] <= prefix_slots_i:
#                         continue
#                     pred = pred[:, prefix_slots_i:, ...]
#                     target = target[:, prefix_slots_i:, ...]
#                     dropped_prefix_terms += 1
#                 loss = F.mse_loss(pred, target)
#                 denoise_loss += loss
#                 valid_terms += 1
#             if valid_terms <= 0:
#                 raise RuntimeError("无有效训练样本参与损失计算")
#             denoise_loss = denoise_loss / valid_terms
#             self._last_train_prefix_loss_dropped = int(dropped_prefix_terms)

#             # 新版训练目标：仅保留原始去噪主损失（ground-truth latent velocity vs predicted velocity）
#             total_loss = denoise_loss
#             self._last_loss_denoise = float(denoise_loss.detach().item())
#             self._last_loss_aux_align_total = 0.0
#             self._last_loss_aux_t5_l2 = 0.0
#             self._last_loss_aux_t5_cos = 0.0
#             self._last_loss_aux_t5_stats = 0.0
#             self._last_loss_aux_t5_gram = 0.0
#             self._last_loss_aux_t5_cka = 0.0
#             self._last_loss_aux_t5_ot = 0.0
#             self._last_loss_aux_image_preserve = 0.0
#             self._last_loss_aux_wan_func = 0.0

#         finally:
#             self.wan.model.text_len = orig_text_len

#         return total_loss

#     def train(self):
#         """主训练循环。"""
#         args = self.args
#         self._audit_runtime_trainability(stage="train_start")

#         # 设置随机种子
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
#         np.random.seed(args.seed)

#         # 数据集（已完全收敛到 WanVideoDataset）
#         if WanDatasetClass is None:
#             raise RuntimeError("未能导入 WanVideoDataset，请检查 train_connector_for_wan.py 及其依赖")

#         dataset = WanDatasetClass(
#             seed=args.seed,
#             frame_num=args.frame_num,
#             max_area=args.max_area,
#             null_caption_prob=args.null_caption_prob,
#             null_image_prob=args.null_image_prob,
#             max_caption_tokens=args.max_caption_tokens,
#             caption_tokenizer_path=args.caption_tokenizer_path,
#             min_duration_sec=args.min_duration_sec,
#             max_duration_sec=args.max_duration_sec,
#             local_openvid_video_root=args.local_openvid_video_root,
#             local_openvid_csv_path=args.local_openvid_csv_path,
#             local_openvid_limit=args.local_openvid_limit,
#             local_openvid_hd_video_root=args.local_openvid_hd_video_root,
#             local_openvid_hd_csv_path=args.local_openvid_hd_csv_path,
#             local_openvid_hd_limit=args.local_openvid_hd_limit,
#             local_video_cache_dir=args.local_video_cache_dir,
#         )

#         if len(dataset) == 0:
#             raise RuntimeError("数据集为空！检查路径和 JSON 文件。")

#         sampler = None
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=torch.distributed.get_world_size(),
#                 rank=torch.distributed.get_rank(),
#                 shuffle=True,
#                 drop_last=False,
#             )

#         # 由于视频尺寸可能不同, 使用 batch_size=1 避免 collate 问题
#         dataloader = DataLoader(
#             dataset,
#             batch_size=1,
#             shuffle=(sampler is None),
#             sampler=sampler,
#             num_workers=args.dataloader_num_workers,
#             pin_memory=True,
#             collate_fn=self._collate_fn,
#         )

#         # 训练循环
#         os.makedirs(args.output_dir, exist_ok=True)
#         output_dir = Path(args.output_dir).expanduser().resolve()
#         if not self._metrics_jsonl_path:
#             self._metrics_jsonl_path = str((output_dir / "logs" / "train_metrics.jsonl").expanduser().resolve())
#         args.output_dir = str(output_dir)
#         args.metrics_jsonl_path = self._metrics_jsonl_path
#         self._train_wall_start = time.perf_counter()
#         dist_enabled = torch.distributed.is_available() and torch.distributed.is_initialized()

#         def _dist_barrier() -> None:
#             if not dist_enabled:
#                 return
#             try:
#                 if torch.cuda.is_available():
#                     torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
#                 else:
#                     torch.distributed.barrier()
#             except TypeError:
#                 torch.distributed.barrier()

#         # 训练前快照（用于 verify_metaquery_chain before vs after）
#         self._train_before_checkpoint_path = str(output_dir / "checkpoint-before-training")
#         self._save_checkpoint(
#             self._train_before_checkpoint_path,
#             step=0,
#             extra_info={
#                 "is_before_training": True,
#                 "resume_mq_encoder_path": getattr(args, "resume_mq_encoder_path", None),
#                 "note": "trainable params snapshot before optimizer updates",
#             },
#         )
#         if self.is_main_process:
#             print(f"[VERIFY] 已保存训练前快照: {self._train_before_checkpoint_path}")
#         _dist_barrier()

#         self.mq_encoder.train()
#         step = 0
#         running_loss = 0.0
#         early_stop_triggered = False
#         early_stop_reason = ""
#         early_stop_ckpt_path = ""
#         data_epoch = 0
#         if sampler is not None:
#             sampler.set_epoch(data_epoch)
#         data_iter = iter(dataloader)

#         pbar = tqdm(total=args.num_train_steps, desc="Training")
#         self.optimizer.zero_grad(set_to_none=True)

#         while step < args.num_train_steps:
#             step_wall_start = time.perf_counter()
#             accum_loss = 0.0
#             accum_denoise_loss = 0.0
#             accum_align_loss = 0.0
#             accum_align_t5_l2 = 0.0
#             accum_align_t5_cos = 0.0
#             accum_align_t5_stats = 0.0
#             accum_align_t5_gram = 0.0
#             accum_align_t5_cka = 0.0
#             accum_align_t5_ot = 0.0
#             accum_align_img = 0.0
#             accum_align_wan_func = 0.0
#             skip_optimizer_step = False
#             had_fatal_cuda_error = False
#             backward_ok = 0
#             skip_reason = ""
#             self._current_train_ref_anchor_mode = self._resolve_train_ref_anchor_mode()

#             for accum_step in range(args.gradient_accumulation_steps):
#                 # 获取 batch
#                 try:
#                     batch = next(data_iter)
#                 except StopIteration:
#                     data_epoch += 1
#                     if sampler is not None:
#                         sampler.set_epoch(data_epoch)
#                     data_iter = iter(dataloader)
#                     batch = next(data_iter)

#                 try:
#                     loss = self._compute_loss(batch)
#                     loss = loss / args.gradient_accumulation_steps
#                     loss.backward()
#                     self._log_grad_health_once()
#                     accum_loss += loss.item()
#                     scale = 1.0 / max(float(args.gradient_accumulation_steps), 1.0)
#                     accum_denoise_loss += float(self._last_loss_denoise) * scale
#                     accum_align_loss += float(self._last_loss_aux_align_total) * scale
#                     accum_align_t5_l2 += float(self._last_loss_aux_t5_l2) * scale
#                     accum_align_t5_cos += float(self._last_loss_aux_t5_cos) * scale
#                     accum_align_t5_stats += float(self._last_loss_aux_t5_stats) * scale
#                     accum_align_t5_gram += float(self._last_loss_aux_t5_gram) * scale
#                     accum_align_t5_cka += float(self._last_loss_aux_t5_cka) * scale
#                     accum_align_t5_ot += float(self._last_loss_aux_t5_ot) * scale
#                     accum_align_img += float(self._last_loss_aux_image_preserve) * scale
#                     accum_align_wan_func += float(self._last_loss_aux_wan_func) * scale
#                     backward_ok += 1
#                 except Exception as e:
#                     err = str(e)
#                     bad_video = None
#                     try:
#                         bad_video = batch.get("video_path", None)
#                     except Exception:
#                         bad_video = None
#                     print(f"[WARN] step {step} accum_step {accum_step} 训练异常: {err}")
#                     if bad_video is not None:
#                         print(f"[WARN] step {step} accum_step {accum_step} bad_video={bad_video}")
#                     if dist_enabled:
#                         rank = -1
#                         try:
#                             rank = int(torch.distributed.get_rank())
#                         except Exception:
#                             rank = -1
#                         raise RuntimeError(
#                             f"[DIST_FAIL_FAST] rank={rank} step={step} accum_step={accum_step} "
#                             f"error={err}。多卡训练检测到异常，立即中止，"
#                             "避免 rank 步调不一致触发 NCCL timeout。"
#                         ) from e
#                     err_l = err.lower()
#                     is_illegal_access = "illegal memory access" in err_l
#                     is_device_assert = "device-side assert" in err_l
#                     if isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in err.lower()):
#                         skip_optimizer_step = True
#                         skip_reason = "oom"
#                         self.optimizer.zero_grad(set_to_none=True)
#                         if torch.cuda.is_available():
#                             torch.cuda.empty_cache()
#                         gc.collect()
#                         break
#                     if is_illegal_access or is_device_assert:
#                         had_fatal_cuda_error = True
#                         skip_optimizer_step = True
#                         skip_reason = "fatal_cuda"
#                         self.optimizer.zero_grad(set_to_none=True)
#                         if torch.cuda.is_available():
#                             torch.cuda.empty_cache()
#                         gc.collect()
#                         break
#                     # 其他异常也跳过本 step，避免残缺梯度进入 optimizer.step
#                     skip_optimizer_step = True
#                     skip_reason = "error"
#                     self.optimizer.zero_grad(set_to_none=True)
#                     break
#                     continue

#             if had_fatal_cuda_error:
#                 raise RuntimeError(
#                     f"Fatal CUDA kernel error at step={step}. "
#                     "检测到 illegal memory access/device-side assert，已中止训练。"
#                 )

#             if backward_ok == 0:
#                 self._skipped_step_count += 1
#                 if skip_reason == "oom":
#                     self._oom_skip_count += 1
#                 elif skip_reason and skip_reason != "fatal_cuda":
#                     self._error_skip_count += 1
#                 continue

#             if skip_optimizer_step:
#                 self._skipped_step_count += 1
#                 if skip_reason == "oom":
#                     self._oom_skip_count += 1
#                 else:
#                     self._error_skip_count += 1
#                 continue

#             # 梯度裁剪
#             grad_norm = torch.nn.utils.clip_grad_norm_(
#                 self._all_trainable_params(),
#                 args.max_grad_norm,
#             )

#             self.optimizer.step()
#             self.scheduler.step()
#             self.optimizer.zero_grad(set_to_none=True)
#             if args.aggressive_empty_cache:
#                 torch.cuda.empty_cache()

#             step += 1
#             step_time = max(time.perf_counter() - step_wall_start, 1e-6)
#             running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss
#             lr = self.scheduler.get_last_lr()[0]
#             grad_norm_value = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
#             effective_samples = int(max(backward_ok, 0) * max(args.batch_size, 1))
#             samples_per_sec = float(effective_samples / step_time)

#             metrics = {
#                 "train/loss_step": float(accum_loss),
#                 "train/loss_ema": float(running_loss),
#                 "train/loss_denoise": float(accum_denoise_loss),
#                 "train/loss_align_total": float(accum_align_loss),
#                 "train/loss_align_t5_l2": float(accum_align_t5_l2),
#                 "train/loss_align_t5_cos": float(accum_align_t5_cos),
#                 "train/loss_align_t5_stats": float(accum_align_t5_stats),
#                 "train/loss_align_t5_gram": float(accum_align_t5_gram),
#                 "train/loss_align_t5_cka": float(accum_align_t5_cka),
#                 "train/loss_align_t5_ot": float(accum_align_t5_ot),
#                 "train/loss_align_img_preserve": float(accum_align_img),
#                 "train/loss_align_wan_func": float(accum_align_wan_func),
#                 "train/lr": float(lr),
#                 "train/grad_norm": float(grad_norm_value),
#                 "train/step": int(step),
#                 "train/step_time_sec": float(step_time),
#                 "train/samples_per_sec": float(samples_per_sec),
#                 "train/backward_ok_microbatches": int(backward_ok),
#                 "train/effective_batch_samples": int(effective_samples),
#                 "train/skipped_step_count": int(self._skipped_step_count),
#                 "train/oom_skip_count": int(self._oom_skip_count),
#                 "train/error_skip_count": int(self._error_skip_count),
#                 "train/ref_anchor_alpha_mean": float(self._last_train_ref_anchor_alpha_mean),
#                 "train/ref_anchor_applied": int(self._last_train_ref_anchor_applied),
#                 "train/ref_anchor_mode_cfg": str(getattr(args, "train_ref_anchor_mode", "none")),
#                 "train/ref_anchor_mode_effective": str(self._last_train_ref_anchor_effective_mode),
#                 "train/ref_anchor_effective_is_animate": int(self._last_train_ref_anchor_effective_mode == "animate_like"),
#                 "train/video_conditioning_mode_cfg": str(getattr(args, "dit_condition_mode", "mq_only")),
#                 "train/video_conditioning_mode_effective": str(self._last_train_video_conditioning_mode),
#                 "train/prefix_latent_slots": int(self._last_train_prefix_latent_slots),
#                 "train/target_latent_slots": int(self._last_train_target_latent_slots),
#                 "train/prefix_loss_dropped": int(self._last_train_prefix_loss_dropped),
#             }
#             metrics.update(self._collect_trainability_metrics())
#             metrics.update(self._collect_cuda_memory_metrics())

#             should_log = bool(args.log_every_step or (step % args.log_steps == 0))
#             should_wandb_log = bool(
#                 self.wandb_run is not None and (args.wandb_log_every_step or should_log)
#             )

#             # 日志
#             if should_log:
#                 pbar.set_postfix({
#                     "loss": f"{accum_loss:.4f}",
#                     "denoise": f"{accum_denoise_loss:.4f}",
#                     "align": f"{accum_align_loss:.4f}",
#                     "func": f"{accum_align_wan_func:.4f}",
#                     "avg": f"{running_loss:.4f}",
#                     "lr": f"{lr:.2e}",
#                     "grad": f"{grad_norm_value:.2f}",
#                     "dP": f"{metrics['train/param_sample_abs_delta_mean']:.3e}",
#                 })
#                 print(
#                     f"\n[Step {step}/{args.num_train_steps}] "
#                     f"loss={accum_loss:.4f} denoise={accum_denoise_loss:.4f} align={accum_align_loss:.4f} func={accum_align_wan_func:.4f} "
#                     f"avg={running_loss:.4f} "
#                     f"lr={lr:.2e} grad_norm={grad_norm_value:.2f} "
#                     f"dt={step_time:.2f}s samp/s={samples_per_sec:.2f} "
#                     f"param_delta={metrics['train/param_sample_abs_delta_mean']:.3e} "
#                     f"skip(oom/err/total)={self._oom_skip_count}/{self._error_skip_count}/{self._skipped_step_count}"
#                 )
#             if should_wandb_log:
#                 self.wandb.log(metrics, step=step)
#             self._append_metrics_jsonl(metrics)
#             self._record_metrics(metrics)

#             # 保存
#             if step % args.save_steps == 0:
#                 self._save_checkpoint(output_dir / f"checkpoint-{step}", step)
#                 _dist_barrier()

#             pbar.update(1)

#             step_loss_for_early_stop = float(accum_denoise_loss)
#             if (
#                 bool(getattr(args, "enable_loss_early_stop", False))
#                 and step >= int(getattr(args, "loss_early_stop_min_step", 800))
#                 and step_loss_for_early_stop < float(getattr(args, "loss_early_stop_threshold", 0.25))
#             ):
#                 early_stop_triggered = True
#                 early_stop_reason = (
#                     f"train/loss_denoise={step_loss_for_early_stop:.6f} < {float(args.loss_early_stop_threshold):.6f} "
#                     f"at step={int(step)}"
#                 )
#                 early_stop_ckpt_path = str(
#                     output_dir / f"checkpoint-earlystop-step{int(step)}-denoise{step_loss_for_early_stop:.4f}"
#                 )
#                 self._save_checkpoint(
#                     early_stop_ckpt_path,
#                     step,
#                     extra_info={
#                         "early_stop": True,
#                         "early_stop_metric": "train/loss_denoise",
#                         "early_stop_loss": step_loss_for_early_stop,
#                         "early_stop_threshold": float(args.loss_early_stop_threshold),
#                         "early_stop_min_step": int(args.loss_early_stop_min_step),
#                     },
#                 )
#                 _dist_barrier()
#                 if self.is_main_process:
#                     print(f"[EARLY-STOP] 已触发: {early_stop_reason}")
#                     print(f"[EARLY-STOP] checkpoint: {early_stop_ckpt_path}")
#                 break

#         pbar.close()

#         # 最终保存
#         final_ckpt_path = str(output_dir / "checkpoint-final")
#         final_extra_info = None
#         if early_stop_triggered:
#             final_extra_info = {
#                 "early_stop": True,
#                 "early_stop_reason": early_stop_reason,
#                 "early_stop_checkpoint_path": early_stop_ckpt_path,
#             }
#         self._save_checkpoint(final_ckpt_path, step, extra_info=final_extra_info)
#         _dist_barrier()
#         self._write_training_chain_manifest(output_dir, final_checkpoint_path=final_ckpt_path, final_step=step)
#         if early_stop_triggered and self.is_main_process:
#             print(f"[EARLY-STOP] 训练提前结束，最终步数: {step}")
#         print(f"\n✅ 训练完成！最终 checkpoint: {final_ckpt_path}")
#         if self.wandb_run is not None:
#             self.wandb.finish()

#     def _save_checkpoint(self, path, step, extra_info: Dict[str, Any] | None = None):
#         """保存 MQ 编码器 +（可选）Wan DiT 可训练子集（兼容增强格式）"""
#         if not self.is_main_process:
#             return
#         path = Path(path).expanduser().resolve()
#         module = self._mq_encoder_module()
#         ckpt_info = save_mq_checkpoint_bundle(
#             path=path,
#             module=module,
#             optimizer=self.optimizer,
#             scheduler=self.scheduler,
#             step=step,
#             args=self.args,
#             wan_module=getattr(self.wan, "model", None),
#             wan_train_mode=str(getattr(self, "_effective_wan_train_mode", "frozen")),
#             metrics_tail=self._metrics_history[-200:],
#             metrics_summary=self._build_metrics_summary(step=step),
#             extra_info={
#                 "before_checkpoint_path": self._train_before_checkpoint_path,
#                 "metrics_jsonl_path": self._metrics_jsonl_path,
#                 "wan_train_mode_effective": str(getattr(self, "_effective_wan_train_mode", "frozen")),
#                 "wan_trainable_tensor_count": int(len(getattr(self, "_wan_trainable_names", []))),
#                 "wan_trainable_name_preview": list(getattr(self, "_wan_trainable_names", [])[:64]),
#                 **(extra_info or {}),
#             },
#         )
#         print(f"  💾 Checkpoint 已保存: {ckpt_info['path']}")
#         if self.wandb_run is not None and self.args.wandb_log_checkpoint:
#             self.wandb.log(
#                 {
#                     "checkpoint/step": int(step),
#                     "checkpoint/path": str(ckpt_info["path"]),
#                 },
#                 step=step,
#             )

#     @staticmethod
#     def _collate_fn(batch):
#         """自定义 collate: 不 stack 不同尺寸的 tensor"""
#         result = {}
#         for key in batch[0].keys():
#             result[key] = [item[key] for item in batch]
#         return result


# # =============================================================================
# # Main
# # =============================================================================
# if __name__ == "__main__":
#     args = parse_args()
#     trainer = MetaQueryWanTrainer(args)
#     trainer.train()
















































# 下面这个是为了适配fsdp情况下wan参数不匹配的情况：
# """
# train_metaquery_wan.py
# =======================
# MetaQuery + Wan2.2 TI2V (Text+Image → Video) 联合训练脚本。

# ★ 核心思路:
#     复刻原始 MetaQuery 训练范式 —— 冻结 DiT，训练 Connector：
#     1. Qwen3-VL (冻结, 仅 MQ embeddings 可训练)
#     2. Connector: Qwen2Encoder(24L) + Linear + GELU + Linear + RMSNorm → dim=4096 (直接对齐 Wan)
#     3. to_wan_proj: 不再需要! Connector 直接输出 Wan text_dim=4096
#     4. Wan TI2V DiT (冻结): 接收 [MQ_tokens + T5_tokens] 作为 context
#     5. 计算 Flow Matching Loss → 反向传播更新 Connector + MQ Embeddings

# ★ 为什么选 WanTI2V (而非 I2V 或 Animate):
#     - TI2V 5B 是 Wan2.2 最新的 Text+Image→Video 统一模型
#     - 使用相同 DiT architecture 处理 t2v 和 i2v (model_type='ti2v')
#     - 不需要 CLIP encoder (I2V 需要 CLIP, Animate 需要 CLIP+Face+Pose)
#     - 参考图通过 VAE 编码后的 latent mask 注入 (最优雅的方式)
#     - 5B 参数量适中, 显存友好

# ★ 不需要 to_wan_proj:
#     直接让 Connector 输出 dim=4096 (Wan text_dim)
#     → 训练时 DiT 的 text_embedding 层直接消费 MQ 特征
#     → 无中间随机投影层, 梯度直接流过

# 用法:
#     # 单卡
#     python train_metaquery_wan.py --wan_checkpoint_dir /path/to/Wan2.2-TI2V-5B

#     # 多卡
#     torchrun --nproc_per_node=2 train_metaquery_wan.py
# """

# import os
# import sys
# import gc
# import json
# import math
# import time
# import argparse
# import random
# from pathlib import Path
# from datetime import datetime
# from contextlib import contextmanager
# from typing import Dict, Tuple, Any, List, Sequence

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.utils.data import DataLoader
# from PIL import Image
# from tqdm import tqdm

# from wan_lora_utils import (
#     apply_lora_to_wan_model,
#     build_lora_config_dict,
#     collect_lora_state_dict,
# )

# # ── 路径设置 ─────────────────────────────────────────────────────────────────
# WAN_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(WAN_ROOT))

# METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
# sys.path.insert(0, METAQUERY_ROOT)


# # =============================================================================
# # 配置
# # =============================================================================
# def parse_args():
#     p = argparse.ArgumentParser(description="Train MetaQuery Connector for Wan TI2V")

#     # ── 模型路径 ──────────────────────────────────────────────────────────
#     p.add_argument("--wan_checkpoint_dir", type=str,
#                    default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
#                    help="Wan2.2 TI2V checkpoint 目录")
#     p.add_argument("--qwen3vl_model_id", type=str,
#                    default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking",
#                    help="Qwen3-VL 模型 ID 或本地路径")
#     p.add_argument("--output_dir", type=str,
#                    default="/home/liuzhirui/model/Wan2.2/metaquery_wan_ti2v_training",
#                    help="训练输出目录")

#     # ── 数据(OpenVid/WanVideoDataset) ───────────────────────────────────────
#     p.add_argument("--local_openvid_video_root", type=str, default=None,
#                    help="本地 OpenVid 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video")
#     p.add_argument("--local_openvid_csv_path", type=str, default=None,
#                    help="本地 OpenVid CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv")
#     p.add_argument("--local_openvid_limit", type=int, default=None,
#                    help="仅使用前N条本地匹配样本，默认使用全部已匹配样本")
#     p.add_argument("--local_openvid_hd_video_root", type=str, default=None,
#                    help="本地 OpenVid HD 视频目录，例如 /home/liuzhirui/dataset/OpenVid-1M/video_HD")
#     p.add_argument("--local_openvid_hd_csv_path", type=str, default=None,
#                    help="本地 OpenVid HD CSV 路径，例如 /home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv")
#     p.add_argument("--local_openvid_hd_limit", type=int, default=None,
#                    help="仅使用前N条本地HD匹配样本，默认使用全部已匹配样本")
#     p.add_argument("--local_video_cache_dir", type=str, default=None,
#                    help="本地视频缓存目录（用于URL/字节解码缓存，可选）")
#     p.add_argument("--frame_num", type=int, default=41,
#                    help="每个视频片段采样帧数 (4n+1)")
#     p.add_argument("--max_area", type=int, default=480 * 832,
#                    help="视频最大面积 (宽×高)")
#     p.add_argument("--max_caption_tokens", type=int, default=512,
#                    help="超过该token长度的caption会被过滤")
#     p.add_argument("--caption_tokenizer_path", type=str, default="google/umt5-xxl",
#                    help="用于caption长度统计的tokenizer")
#     p.add_argument("--min_duration_sec", type=float, default=0.5,
#                    help="最短时长过滤阈值")
#     p.add_argument("--max_duration_sec", type=float, default=20.0,
#                    help="最长时长过滤阈值")

#     # ── 训练参数 ──────────────────────────────────────────────────────────
#     p.add_argument("--learning_rate", type=float, default=5e-5)
#     p.add_argument("--num_train_steps", type=int, default=5000)
#     p.add_argument("--warmup_steps", type=int, default=200)
#     p.add_argument(
#         "--lr_scheduler_type",
#         type=str,
#         default="cosine_with_warmup",
#         choices=["cosine_with_warmup", "constant_with_warmup", "warmup_hold_cooldown"],
#         help=(
#             "学习率调度器类型。"
#             "constant_with_warmup=warmup后恒定；"
#             "cosine_with_warmup=warmup后余弦衰减；"
#             "warmup_hold_cooldown=warmup线性升+中段恒定+末段线性降。"
#         ),
#     )
#     p.add_argument(
#         "--cooldown_steps",
#         type=int,
#         default=-1,
#         help="warmup_hold_cooldown 模式下末段降学习率步数。<0 表示使用 warmup_steps。",
#     )
#     p.add_argument(
#         "--lr_min_ratio",
#         type=float,
#         default=0.01,
#         help="cosine_with_warmup 模式下的最小学习率比例。",
#     )
#     p.add_argument("--batch_size", type=int, default=1)
#     p.add_argument("--gradient_accumulation_steps", type=int, default=4)
#     p.add_argument("--max_grad_norm", type=float, default=1.0)
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--save_steps", type=int, default=500)
#     p.add_argument("--log_steps", type=int, default=10)
#     p.add_argument("--enable_loss_early_stop", action="store_true", default=False,
#                    help="启用可选早停：当 step>=loss_early_stop_min_step 且 loss_step<loss_early_stop_threshold 时提前结束训练并保存 checkpoint。")
#     p.add_argument("--disable_loss_early_stop", action="store_false", dest="enable_loss_early_stop",
#                    help="关闭 loss 早停（默认）。")
#     p.add_argument("--loss_early_stop_min_step", type=int, default=800,
#                    help="loss 早停触发的最小 step（含）。")
#     p.add_argument("--loss_early_stop_threshold", type=float, default=0.25,
#                    help="loss 早停阈值：当 train/loss_step 小于该值时触发。")
#     p.add_argument("--log_every_step", action="store_true",
#                    help="每个优化 step 都打印详细训练日志")
#     p.add_argument("--wandb_log_every_step", action="store_true",
#                    help="每个优化 step 都写入 W&B（默认按 log_steps 写入）")
#     p.add_argument("--metrics_jsonl_path", type=str, default="",
#                    help="可选：将每步指标追加写入 JSONL 文件")
#     p.add_argument("--log_cuda_memory", action="store_true",
#                    help="记录并输出 CUDA 显存指标")
#     p.add_argument("--dataloader_num_workers", type=int, default=0)

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
#     p.add_argument("--mq_gradient_checkpointing", action="store_true",
#                    help="启用 MetaQuery 编码器梯度检查点，降低显存占用")
#     p.add_argument("--train_mq_input_embeddings", action="store_true", default=True,
#                    help="训练 Qwen 输入 embedding（默认开启）")
#     p.add_argument("--freeze_mq_input_embeddings", action="store_false", dest="train_mq_input_embeddings",
#                    help="冻结 Qwen 输入 embedding，仅训练 connector")
#     p.add_argument("--null_caption_prob", type=float, default=0.1)
#     p.add_argument("--null_image_prob", type=float, default=0.1)
#     p.add_argument("--enable_t5_alignment", action="store_true", default=True,
#                    help="启用 T5 对齐辅助损失（默认开启）：让 MQ 条件分布更接近 Wan 已适配的 T5 条件流形。")
#     p.add_argument("--disable_t5_alignment", action="store_false", dest="enable_t5_alignment",
#                    help="关闭 T5 对齐辅助损失，仅使用去噪主损失。")
#     p.add_argument(
#         "--t5_align_mode",
#         type=str,
#         default="gram_cka",
#         choices=["anchor", "gram_cka", "sinkhorn_ot"],
#         help=(
#             "T5 对齐方式。anchor=前K token 一一对齐；"
#             "gram_cka=基于 token 关系矩阵(Gram+CKA)的排列无关对齐；"
#             "sinkhorn_ot=基于 OT/Sinkhorn 的软匹配对齐。"
#         ),
#     )
#     p.add_argument("--t5_align_anchor_tokens", type=int, default=64,
#                    help="用于 T5 对齐的 anchor token 数（从 256 个 MQ token 前缀取）。")
#     p.add_argument("--lambda_t5_align_l2", type=float, default=0.2,
#                    help="T5 对齐主项权重：anchor 模式对应 token-L2；gram_cka 模式对应 Gram-L2；sinkhorn_ot 模式对应 OT 代价。")
#     p.add_argument("--lambda_t5_align_cos", type=float, default=0.1,
#                    help="T5 对齐次项权重：anchor 模式对应 token-cos；gram_cka 模式对应 CKA；sinkhorn_ot 模式默认忽略。")
#     p.add_argument("--lambda_t5_align_stats", type=float, default=0.02,
#                    help="T5 对齐的均值/方差统计损失权重。")
#     p.add_argument("--t5_align_ot_epsilon", type=float, default=0.05,
#                    help="Sinkhorn OT 熵正则温度 epsilon（越小越接近硬匹配）。")
#     p.add_argument("--t5_align_ot_iters", type=int, default=25,
#                    help="Sinkhorn OT 迭代次数。")
#     p.add_argument("--enable_mq_image_preserve", action="store_true", default=False,
#                    help="启用图像保持约束：有参考图时，MQ(cond) 与 MQ(text-only) 保持最小间隔。")
#     p.add_argument("--lambda_mq_image_preserve", type=float, default=0.02,
#                    help="图像保持约束权重。")
#     p.add_argument("--mq_image_preserve_margin", type=float, default=0.10,
#                    help="图像保持约束的最小间隔阈值（L2 均方根距离）。")
#     p.add_argument("--mq_norm_probe_with_t5", action="store_true", default=True,
#                    help="训练时记录 MQ 与 T5 token RMS 范数比值（用于定位 MQ 条件被忽略问题）。")
#     p.add_argument("--disable_mq_norm_probe_with_t5", action="store_false", dest="mq_norm_probe_with_t5",
#                    help="关闭 MQ/T5 范数探针。")
#     p.add_argument("--mq_norm_probe_every_n_steps", type=int, default=20,
#                    help="每 N 次 _compute_loss 调用做一次 MQ/T5 范数探针。")
#     p.add_argument("--mq_norm_warn_ratio_low", type=float, default=0.25,
#                    help="当 MQ/T5 RMS 比值低于该阈值时打印警告。")
#     p.add_argument("--mq_norm_warn_ratio_high", type=float, default=4.0,
#                    help="当 MQ/T5 RMS 比值高于该阈值时打印警告。")
#     p.add_argument("--mq_norm_match_t5", action="store_true", default=False,
#                    help="将 MQ 特征按 token RMS 对齐到 T5 RMS（默认关闭，仅用于排查范数错配）。")
#     p.add_argument("--mq_norm_match_clip_min", type=float, default=0.25,
#                    help="mq_norm_match_t5 时缩放因子下限。")
#     p.add_argument("--mq_norm_match_clip_max", type=float, default=4.0,
#                    help="mq_norm_match_t5 时缩放因子上限。")
#     p.add_argument("--enable_wan_func_distill", action="store_true", default=False,
#                    help="启用 Wan 函数级蒸馏：约束 pred_mq(x_t,t) 贴近 pred_t5(x_t,t)。")
#     p.add_argument("--disable_wan_func_distill", action="store_false", dest="enable_wan_func_distill",
#                    help="关闭 Wan 函数级蒸馏。")
#     p.add_argument("--lambda_wan_func_distill", type=float, default=0.0,
#                    help="Wan 函数级蒸馏损失权重。")
#     p.add_argument(
#         "--wan_func_teacher_mode",
#         type=str,
#         default="t5_only",
#         choices=["t5_only", "t5_plus_mq"],
#         help="函数级蒸馏 teacher 条件。t5_only=仅 T5；t5_plus_mq=T5 与 MQ 拼接。",
#     )
#     p.add_argument("--enable_ti2v_first_frame_condition", action="store_true", default=True,
#                    help="启用 Wan 训练侧首帧参考条件（与 MQ 图像条件并行）。")
#     p.add_argument("--disable_ti2v_first_frame_condition", action="store_false",
#                    dest="enable_ti2v_first_frame_condition",
#                    help="关闭 Wan 训练侧首帧参考条件，仅保留 MQ 条件。")
#     p.add_argument("--train_video_conditioning_mode", type=str, default="legacy_t2v",
#                    choices=["legacy_t2v", "wan_animate_slot"],
#                    help=(
#                        "训练期视频条件注入方式: "
#                        "legacy_t2v=现有 TI2V 训练（可选首帧软锚定）；"
#                        "wan_animate_slot=参考图作为 preserved reference slot 注入，前缀 slot 不计入主损失"
#                    ))
#     p.add_argument("--train_animate_ref_frames", type=int, default=1,
#                    help="wan_animate_slot 模式下参考图保留帧数（像素帧数，内部按 VAE stride 映射到 latent slots）")
#     p.add_argument("--train_animate_temporal_frames", type=int, default=0,
#                    help="wan_animate_slot 模式下 temporal guidance 帧数（像素帧数；若无外部时序条件可保持 0）")
#     p.add_argument("--train_animate_conditional_frames", type=int, default=0,
#                    help="wan_animate_slot 模式下额外 conditional 帧数（像素帧数；无条件时保持 0，将注入全零 latent）")
#     p.add_argument("--train_animate_preserve_timestep_zero", action="store_true", default=True,
#                    help="wan_animate_slot: preserved prefix 对应 token 的 timestep 置 0（默认开启）")
#     p.add_argument("--train_animate_no_preserve_timestep_zero", action="store_false",
#                    dest="train_animate_preserve_timestep_zero",
#                    help="wan_animate_slot: 关闭 preserved prefix timestep=0")
#     p.add_argument("--train_animate_drop_prefix_loss", action="store_true", default=True,
#                    help="wan_animate_slot: 仅在 target frames 上计算损失，丢弃 reference/temporal/conditional prefix（默认开启）")
#     p.add_argument("--train_animate_no_drop_prefix_loss", action="store_false",
#                    dest="train_animate_drop_prefix_loss",
#                    help="wan_animate_slot: 不丢弃 prefix，整段都计入损失")
#     p.add_argument("--train_ref_anchor_mode", type=str, default="none",
#                    choices=["none", "animate_like", "mixed50"],
#                    help="训练时是否对 x_t 的首帧加入软参考锚定。none=保持原始 t2v；animate_like=全程启用软锚定；mixed50=约50%批次启用软锚定")
#     p.add_argument("--train_ref_anchor_alpha0", type=float, default=0.95,
#                    help="animate_like 模式的最大锚定强度 alpha0")
#     p.add_argument("--train_ref_anchor_warmup_ratio", type=float, default=0.35,
#                    help="animate_like 模式在高噪声区间启用锚定的占比（0~1）")

#     # ── 设备 ──────────────────────────────────────────────────────────────
#     p.add_argument("--dit_device", type=int, default=0,
#                    help="DiT + VAE + T5 所在 GPU")
#     p.add_argument("--encoder_device", type=int, default=1,
#                    help="Qwen3-VL + Connector 所在 GPU")
#     p.add_argument("--resume_mq_encoder_path", type=str, default=None,
#                    help="从已有mq_encoder权重继续训练")
#     p.add_argument("--t5_cpu", action="store_true",
#                    help="将Wan的T5文本编码器保留在CPU，显著降低GPU显存占用（速度会变慢）")
#     p.add_argument("--dit_fsdp", action="store_true",
#                    help="启用 Wan DiT 的 FSDP 参数分片，降低单卡模型权重占用")
#     p.add_argument("--t5_fsdp", action="store_true",
#                    help="启用 T5 编码器的 FSDP 参数分片")
#     p.add_argument("--use_sp", action="store_true",
#                    help="启用 sequence parallel（xDiT/USP 路径）")
#     p.add_argument("--no_init_on_cpu", action="store_true",
#                    help="关闭 init_on_cpu；默认开启以减小加载瞬时显存峰值")
#     p.add_argument("--convert_model_dtype", action="store_true",
#                    help="将 Wan DiT 参数显式转换到 config.param_dtype（仅非FSDP时生效）")
#     p.add_argument("--aggressive_empty_cache", action="store_true",
#                    help="每步训练后执行 torch.cuda.empty_cache()，缓解显存碎片")
#     p.add_argument("--wandb_enabled", action="store_true",
#                    help="启用 Weights & Biases 训练日志")
#     p.add_argument("--wandb_project", type=str, default="wan-metaquery",
#                    help="W&B project 名称")
#     p.add_argument("--wandb_entity", type=str, default="",
#                    help="W&B entity/team 名称")
#     p.add_argument("--wandb_run_name", type=str, default="",
#                    help="W&B run 名称, 留空自动生成")
#     p.add_argument("--wandb_tags", type=str, default="",
#                    help="W&B tags, 逗号分隔")
#     p.add_argument("--wandb_mode", type=str, default="online",
#                    choices=["online", "offline", "disabled"],
#                    help="W&B 模式")
#     p.add_argument("--wandb_api_key", type=str, default="",
#                    help="W&B API Key, 传入后会写入 WANDB_API_KEY 环境变量")
#     p.add_argument("--wandb_log_checkpoint", action="store_true",
#                    help="在 W&B 中记录 checkpoint 路径")
#     p.add_argument("--strict_freeze_check", action="store_true", default=True,
#                    help="启用严格冻结校验：若发现 Wan/T5/VAE 可训练或 optimizer 混入非 MQ 参数则中止")
#     p.add_argument("--no_strict_freeze_check", action="store_false", dest="strict_freeze_check",
#                    help="关闭严格冻结校验，仅打印告警")
#     p.add_argument(
#         "--wan_train_mode",
#         type=str,
#         default="auto",
#         choices=["auto", "frozen", "full", "cond_only", "lora"],
#         help=(
#             "Wan DiT 训练模式。auto=按显存策略自动在 full/cond_only 之间选择；"
#             "frozen=冻结；full=全量训练；cond_only=仅训 cross-attn + conditioning projection/AdaLN/modulation；"
#             "lora=LoRA 微调（可叠加额外小模块直训）。"
#         ),
#     )
#     p.add_argument(
#         "--wan_auto_full_mem_gb",
#         type=float,
#         default=120.0,
#         help="auto 模式下，当 DiT 卡总显存 >= 该阈值时选择 full，否则选择 cond_only。",
#     )
#     p.add_argument(
#         "--wan_lr_ratio",
#         type=float,
#         default=1.0,
#         help="Wan 可训练参数学习率倍率（实际 lr = learning_rate * wan_lr_ratio）。",
#     )
#     p.add_argument(
#         "--wan_cond_name_pattern",
#         type=str,
#         default="",
#         help=(
#             "可选：自定义 cond_only 的参数名匹配关键字，逗号分隔。"
#             "为空时使用内置规则(cross_attn,text_embedding,time_projection,modulation,norm3,cross_attn_norm)。"
#         ),
#     )
#     p.add_argument("--enable_wan_lora", action="store_true", default=False,
#                    help="启用 Wan DiT LoRA 微调（当前用于单进程/非FSDP 路径）")
#     p.add_argument("--disable_wan_lora", action="store_false", dest="enable_wan_lora",
#                    help="禁用 Wan DiT LoRA 微调")
#     p.add_argument("--wan_lora_rank", type=int, default=16,
#                    help="Wan LoRA rank")
#     p.add_argument("--wan_lora_alpha", type=float, default=16.0,
#                    help="Wan LoRA alpha")
#     p.add_argument("--wan_lora_dropout", type=float, default=0.0,
#                    help="Wan LoRA dropout")
#     p.add_argument("--wan_lora_targets", type=str, default="self_attn,cross_attn,ffn",
#                    help="Wan LoRA 目标模块类别，逗号分隔，可选: self_attn,cross_attn,ffn")
#     p.add_argument("--wan_lora_extra_name_pattern", type=str, default="",
#                    help="LoRA 模式下额外直训的小模块关键词，逗号分隔。例如 norm1,norm2,norm3,time_projection,modulation")

#     return p.parse_args()


# def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
#     """兼容不同 torch 版本的安全加载。"""
#     try:
#         return torch.load(path, map_location=map_location, weights_only=True)
#     except TypeError:
#         return torch.load(path, map_location=map_location)


# def _extract_model_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
#     """从不同 checkpoint 负载中提取模型权重字典。"""
#     if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
#         return payload["model_state_dict"]
#     if isinstance(payload, dict):
#         tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
#         non_tensor_values = [v for v in payload.values() if not torch.is_tensor(v)]
#         if tensor_values and not non_tensor_values:
#             return payload
#     raise ValueError("无法从 checkpoint 提取 model_state_dict")


# def _to_cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     out = {}
#     for k, v in state_dict.items():
#         if torch.is_tensor(v):
#             out[k] = v.detach().cpu().contiguous()
#     return out


# def load_mq_encoder_state(path_or_dir: str, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, torch.Tensor], str]:
#     """
#     加载 MetaQuery encoder 权重:
#     - 支持传入单个文件: mq_encoder_full.pt / training_state.pt / model.safetensors
#     - 支持传入目录: 自动按优先级查找文件
#     """
#     path = Path(path_or_dir)
#     if not path.exists():
#         raise FileNotFoundError(f"checkpoint 路径不存在: {path}")

#     if path.is_dir():
#         candidates = [
#             path / "mq_encoder_full.pt",
#             path / "mq_encoder_full.safetensors",
#             path / "model.safetensors",
#             path / "pytorch_model.bin",
#             path / "training_state.pt",
#         ]
#         picked = next((p for p in candidates if p.exists()), None)
#         if picked is None:
#             raise FileNotFoundError(
#                 f"checkpoint 目录中未找到可加载权重文件: {path} "
#                 f"(expect one of {[c.name for c in candidates]})"
#             )
#         path = picked

#     suffix = path.suffix.lower()
#     if suffix == ".safetensors":
#         try:
#             from safetensors.torch import load_file
#         except Exception as e:
#             raise RuntimeError(
#                 f"检测到 safetensors 权重但未能导入 safetensors: {path}"
#             ) from e
#         state = load_file(str(path), device="cpu")
#     else:
#         payload = _safe_torch_load(path, map_location=map_location)
#         state = _extract_model_state_dict(payload)

#     return state, str(path.expanduser().resolve())


# def _write_json(path: Path, payload: Dict[str, Any]) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)


# def _to_jsonable(value: Any) -> Any:
#     if value is None or isinstance(value, (str, int, float, bool)):
#         return value
#     if isinstance(value, (list, tuple)):
#         return [_to_jsonable(v) for v in value]
#     if isinstance(value, dict):
#         return {str(k): _to_jsonable(v) for k, v in value.items()}
#     if isinstance(value, Path):
#         return str(value)
#     return str(value)


# def save_mq_checkpoint_bundle(
#     path: Path,
#     module: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LRScheduler,
#     step: int,
#     args: argparse.Namespace,
#     wan_module: nn.Module | None = None,
#     wan_trainable_state_cpu: Dict[str, torch.Tensor] | None = None,
#     wan_lora_state_cpu: Dict[str, torch.Tensor] | None = None,
#     wan_lora_config: Dict[str, Any] | None = None,
#     wan_train_mode: str = "frozen",
#     metrics_tail: List[Dict[str, Any]] | None = None,
#     metrics_summary: Dict[str, Any] | None = None,
#     extra_info: Dict[str, Any] | None = None,
# ) -> Dict[str, Any]:
#     """
#     保存“最小可用 + 兼容增强”的 checkpoint bundle。
#     兼容你当前推理脚本（mq_encoder_full.pt）并补充常见训练文件。
#     """
#     path = path.expanduser().resolve()
#     path.mkdir(parents=True, exist_ok=True)

#     full_state_cpu = _to_cpu_state_dict(module.state_dict())
#     name_to_param = dict(module.named_parameters())
#     trainable_state_cpu = {
#         name: tensor
#         for name, tensor in full_state_cpu.items()
#         if name_to_param.get(name, None) is not None
#         and name_to_param[name].requires_grad
#     }

#     torch.save(
#         {
#             "step": step,
#             "model_state_dict": trainable_state_cpu,
#             "optimizer_state_dict": optimizer.state_dict(),
#             "scheduler_state_dict": scheduler.state_dict(),
#         },
#         path / "training_state.pt",
#     )
#     torch.save(full_state_cpu, path / "mq_encoder_full.pt")
#     torch.save(trainable_state_cpu, path / "mq_encoder_trainable.pt")

#     if wan_trainable_state_cpu is None:
#         wan_trainable_state_cpu = {}
#         if wan_module is not None and isinstance(wan_module, nn.Module):
#             for name, p in wan_module.named_parameters():
#                 if not p.requires_grad:
#                     continue
#                 wan_trainable_state_cpu[name] = p.detach().cpu().contiguous()
#     else:
#         wan_trainable_state_cpu = {
#             str(name): tensor.detach().cpu().contiguous()
#             for name, tensor in wan_trainable_state_cpu.items()
#             if torch.is_tensor(tensor)
#         }
#     if wan_lora_state_cpu is None:
#         wan_lora_state_cpu = {}
#     else:
#         wan_lora_state_cpu = {
#             str(name): tensor.detach().cpu().contiguous()
#             for name, tensor in wan_lora_state_cpu.items()
#             if torch.is_tensor(tensor)
#         }
#     wan_trainable_param_count = sum(int(t.numel()) for t in wan_trainable_state_cpu.values())
#     wan_lora_param_count = sum(int(t.numel()) for t in wan_lora_state_cpu.values())
#     if wan_trainable_state_cpu:
#         torch.save(wan_trainable_state_cpu, path / "wan_dit_trainable.pt")
#     if wan_lora_state_cpu:
#         torch.save(wan_lora_state_cpu, path / "wan_dit_lora.pt")

#     torch.save(vars(args), path / "training_args.bin")
#     _write_json(
#         path / "training_args.json",
#         {str(k): _to_jsonable(v) for k, v in vars(args).items()},
#     )
#     torch.save(optimizer.state_dict(), path / "optimizer.pt")
#     torch.save(scheduler.state_dict(), path / "scheduler.pt")

#     trainer_state = {
#         "global_step": int(step),
#         "checkpoint_format": "wan_metaquery_v2",
#         "has_full_pt": True,
#         "has_training_state": True,
#         "has_trainable_pt": True,
#         "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
#         "has_wan_dit_lora_pt": bool(len(wan_lora_state_cpu) > 0),
#         "wan_train_mode": str(wan_train_mode),
#         "wan_trainable_param_count": int(wan_trainable_param_count),
#         "wan_lora_param_count": int(wan_lora_param_count),
#         "has_metrics_summary": bool(metrics_summary),
#         "metrics_tail_count": int(len(metrics_tail) if metrics_tail is not None else 0),
#     }
#     if extra_info:
#         trainer_state["extra_info"] = _to_jsonable(extra_info)
#     _write_json(path / "trainer_state.json", trainer_state)

#     config_payload = {
#         "format": "wan_metaquery_encoder",
#         "num_metaqueries": int(getattr(args, "num_metaqueries", 256)),
#         "connector_num_hidden_layers": int(getattr(args, "connector_num_hidden_layers", 24)),
#         "wan_text_dim": int(getattr(module, "wan_text_dim", 4096)),
#         "qwen3vl_model_id": str(getattr(args, "qwen3vl_model_id", "")),
#         "train_mq_input_embeddings": bool(getattr(args, "train_mq_input_embeddings", True)),
#         "wan_train_mode": str(wan_train_mode),
#         "wan_trainable_param_count": int(wan_trainable_param_count),
#         "wan_lora_param_count": int(wan_lora_param_count),
#         "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
#         "has_wan_dit_lora_pt": bool(len(wan_lora_state_cpu) > 0),
#         "checkpoint_step": int(step),
#         "num_train_steps": int(getattr(args, "num_train_steps", 0)),
#         "save_steps": int(getattr(args, "save_steps", 0)),
#         "log_steps": int(getattr(args, "log_steps", 0)),
#         "enable_loss_early_stop": bool(getattr(args, "enable_loss_early_stop", False)),
#         "loss_early_stop_min_step": int(getattr(args, "loss_early_stop_min_step", 800)),
#         "loss_early_stop_threshold": float(getattr(args, "loss_early_stop_threshold", 0.25)),
#         "frame_num": int(getattr(args, "frame_num", 0)),
#         "max_area": int(getattr(args, "max_area", 0)),
#         "learning_rate": float(getattr(args, "learning_rate", 0.0)),
#         "warmup_steps": int(getattr(args, "warmup_steps", 0)),
#         "lr_scheduler_type": str(getattr(args, "lr_scheduler_type", "cosine_with_warmup")),
#         "cooldown_steps": int(getattr(args, "cooldown_steps", -1)),
#         "lr_min_ratio": float(getattr(args, "lr_min_ratio", 0.01)),
#         "enable_t5_alignment": bool(getattr(args, "enable_t5_alignment", True)),
#         "t5_align_mode": str(getattr(args, "t5_align_mode", "gram_cka")),
#         "t5_align_anchor_tokens": int(getattr(args, "t5_align_anchor_tokens", 64)),
#         "lambda_t5_align_l2": float(getattr(args, "lambda_t5_align_l2", 0.0)),
#         "lambda_t5_align_cos": float(getattr(args, "lambda_t5_align_cos", 0.0)),
#         "lambda_t5_align_stats": float(getattr(args, "lambda_t5_align_stats", 0.0)),
#         "t5_align_ot_epsilon": float(getattr(args, "t5_align_ot_epsilon", 0.05)),
#         "t5_align_ot_iters": int(getattr(args, "t5_align_ot_iters", 25)),
#         "enable_mq_image_preserve": bool(getattr(args, "enable_mq_image_preserve", False)),
#         "lambda_mq_image_preserve": float(getattr(args, "lambda_mq_image_preserve", 0.0)),
#         "mq_image_preserve_margin": float(getattr(args, "mq_image_preserve_margin", 0.0)),
#         "mq_norm_probe_with_t5": bool(getattr(args, "mq_norm_probe_with_t5", True)),
#         "mq_norm_probe_every_n_steps": int(getattr(args, "mq_norm_probe_every_n_steps", 20)),
#         "mq_norm_warn_ratio_low": float(getattr(args, "mq_norm_warn_ratio_low", 0.25)),
#         "mq_norm_warn_ratio_high": float(getattr(args, "mq_norm_warn_ratio_high", 4.0)),
#         "mq_norm_match_t5": bool(getattr(args, "mq_norm_match_t5", False)),
#         "mq_norm_match_clip_min": float(getattr(args, "mq_norm_match_clip_min", 0.25)),
#         "mq_norm_match_clip_max": float(getattr(args, "mq_norm_match_clip_max", 4.0)),
#         "enable_wan_func_distill": bool(getattr(args, "enable_wan_func_distill", False)),
#         "lambda_wan_func_distill": float(getattr(args, "lambda_wan_func_distill", 0.0)),
#         "wan_func_teacher_mode": str(getattr(args, "wan_func_teacher_mode", "t5_only")),
#         "batch_size": int(getattr(args, "batch_size", 1)),
#         "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1)),
#         "null_caption_prob": float(getattr(args, "null_caption_prob", 0.0)),
#         "null_image_prob": float(getattr(args, "null_image_prob", 0.0)),
#         "wan_train_mode": str(getattr(args, "wan_train_mode", "auto")),
#         "wan_auto_full_mem_gb": float(getattr(args, "wan_auto_full_mem_gb", 120.0)),
#         "wan_lr_ratio": float(getattr(args, "wan_lr_ratio", 1.0)),
#         "wan_cond_name_pattern": str(getattr(args, "wan_cond_name_pattern", "")),
#         "enable_wan_lora": bool(getattr(args, "enable_wan_lora", False)),
#         "wan_lora_rank": int(getattr(args, "wan_lora_rank", 16)),
#         "wan_lora_alpha": float(getattr(args, "wan_lora_alpha", 16.0)),
#         "wan_lora_dropout": float(getattr(args, "wan_lora_dropout", 0.0)),
#         "wan_lora_targets": str(getattr(args, "wan_lora_targets", "")),
#         "wan_lora_extra_name_pattern": str(getattr(args, "wan_lora_extra_name_pattern", "")),
#     }
#     if wan_lora_config:
#         config_payload["wan_lora"] = _to_jsonable(wan_lora_config)
#     # 记录 MLLM embedding 行信息，便于推理期验证“新增 MQ token embedding 是否被保存/加载”。
#     try:
#         emb = module.mllm_model.mllm_backbone.get_input_embeddings()
#         if emb is not None and getattr(emb, "weight", None) is not None:
#             rows_total = int(emb.weight.shape[0])
#             rows_base = int(getattr(module.mllm_model, "num_embeddings", 0))
#             config_payload["mllm_embed_rows_total"] = rows_total
#             config_payload["mllm_embed_rows_base"] = rows_base
#             config_payload["mllm_embed_rows_added"] = max(rows_total - rows_base, 0)
#     except Exception:
#         pass
#     if extra_info:
#         config_payload["extra_info"] = _to_jsonable(extra_info)
#     _write_json(path / "config.json", config_payload)
#     if metrics_summary:
#         _write_json(path / "metrics_summary.json", {str(k): _to_jsonable(v) for k, v in metrics_summary.items()})
#     if metrics_tail is not None:
#         _write_json(
#             path / "metrics_tail.json",
#             {"records": [{str(k): _to_jsonable(v) for k, v in row.items()} for row in metrics_tail]},
#         )

#     try:
#         from safetensors.torch import save_file

#         save_file(full_state_cpu, str(path / "model.safetensors"))
#         save_file(trainable_state_cpu, str(path / "mq_encoder_trainable.safetensors"))
#         if wan_trainable_state_cpu:
#             save_file(wan_trainable_state_cpu, str(path / "wan_dit_trainable.safetensors"))
#         if wan_lora_state_cpu:
#             save_file(wan_lora_state_cpu, str(path / "wan_dit_lora.safetensors"))
#     except Exception:
#         # safetensors 为增强项，不可用时保持兼容主流程
#         pass

#     # 兼容“latest”指针
#     try:
#         with open(path.parent / "latest", "w", encoding="utf-8") as f:
#             f.write(f"{path.name}\n")
#     except Exception:
#         pass

#     return {
#         "step": int(step),
#         "path": str(path),
#     }
# # =============================================================================
# # 数据集
# # =============================================================================
# try:
#     from train_connector_for_wan import WanVideoDataset as _DefaultWanVideoDataset
# except Exception:
#     _DefaultWanVideoDataset = None

# # 单一数据集入口：仅使用 WanVideoDataset。
# # 在 train_metaquery_wan_new.py 中可通过设置 base_ti2v.WanDatasetClass 进行覆写。
# WanDatasetClass = _DefaultWanVideoDataset


# # =============================================================================
# # Trainer
# # =============================================================================
# class MetaQueryWanTrainer:
#     """
#     MetaQuery + Wan TI2V 联合训练。

#     训练流程:
#         1. MetaQuery (Connector 可训练) → [B, 256, 4096]
#         2. T5 编码文本 → [B, text_len, 4096]
#         3. 拼接: [MQ + T5] → [B, 256+text_len, 4096]
#         4. VAE 编码视频帧 → latent
#         5. 采样噪声+时间步 → noisy_latent
#         6. 参考图 VAE 编码 → first frame mask
#         7. DiT (冻结) forward: 预测速度
#         8. Flow Matching Loss → 反向传播 Connector + MQ Embeddings
#     """

#     def __init__(self, args):
#         self.args = args
#         self.dev_dit = torch.device(f"cuda:{args.dit_device}")
#         self.dev_enc = torch.device(f"cuda:{args.encoder_device}")
#         self.wandb = None
#         self.wandb_run = None
#         self.is_main_process = self._is_main_process()
#         self._printed_grad_health = False
#         self._skipped_step_count = 0
#         self._oom_skip_count = 0
#         self._error_skip_count = 0
#         self._printed_context_inject_check = False
#         self._param_monitor = []
#         self._trainable_param_count = 0
#         self._init_trainable_norm = 0.0
#         self._init_param_sample_norm = 0.0
#         _metrics_jsonl = (args.metrics_jsonl_path or "").strip()
#         self._metrics_jsonl_path = str(Path(_metrics_jsonl).expanduser().resolve()) if _metrics_jsonl else ""
#         self._metrics_history: List[Dict[str, Any]] = []
#         self._train_before_checkpoint_path = ""
#         self._train_wall_start = 0.0
#         self._last_train_ref_anchor_alpha_mean = 0.0
#         self._last_train_ref_anchor_applied = 0
#         self._last_train_ref_anchor_effective_mode = "none"
#         self._train_ref_anchor_mixed_counter = 0
#         self._current_train_ref_anchor_mode = "none"
#         self._last_train_video_conditioning_mode = "mq_only"
#         self._last_train_prefix_latent_slots = 0
#         self._last_train_target_latent_slots = 0
#         self._last_train_prefix_loss_dropped = 0
#         self._last_loss_denoise = 0.0
#         self._last_loss_aux_align_total = 0.0
#         self._last_loss_aux_t5_l2 = 0.0
#         self._last_loss_aux_t5_cos = 0.0
#         self._last_loss_aux_t5_stats = 0.0
#         self._last_loss_aux_t5_gram = 0.0
#         self._last_loss_aux_t5_cka = 0.0
#         self._last_loss_aux_t5_ot = 0.0
#         self._last_loss_aux_image_preserve = 0.0
#         self._last_loss_aux_wan_func = 0.0
#         self._loss_call_count = 0
#         self._last_mq_rms = 0.0
#         self._last_t5_rms = 0.0
#         self._last_mq_t5_rms_ratio = 0.0
#         self._last_mq_norm_match_scale = 1.0
#         self._last_mq_norm_warn_flag = 0
#         self._effective_wan_train_mode = "frozen"
#         self._wan_trainable_names: List[str] = []
#         self._wan_trainable_params_cache: List[torch.nn.Parameter] = []
#         self._wan_lora_module_names: List[str] = []
#         self._wan_lora_extra_trainable_names: List[str] = []

#         print("\n" + "=" * 60)
#         print("  MetaQuery + Wan TI2V 联合训练")
#         print("=" * 60)
#         print(f"  DiT 设备       : {self.dev_dit}")
#         print(f"  Encoder 设备   : {self.dev_enc}")
#         print(f"  学习率         : {args.learning_rate}")
#         print(f"  LR 调度器      : {args.lr_scheduler_type}")
#         print(f"  Cooldown 步数  : {args.cooldown_steps} (-1 表示使用 warmup_steps)")
#         print(f"  训练步数       : {args.num_train_steps}")
#         print(f"  有效 batch     : {args.batch_size * args.gradient_accumulation_steps}")
#         print(
#             f"  Loss 早停       : enabled={int(bool(args.enable_loss_early_stop))} "
#             f"min_step={args.loss_early_stop_min_step} threshold={args.loss_early_stop_threshold}"
#         )
#         print(
#             f"  Wan 训练模式    : req={args.wan_train_mode} auto_full_mem_gb={args.wan_auto_full_mem_gb} "
#             f"wan_lr_ratio={args.wan_lr_ratio}"
#         )
#         print(
#             f"  Wan LoRA        : enabled={int(bool(getattr(args, 'enable_wan_lora', False)))} "
#             f"rank={getattr(args, 'wan_lora_rank', 16)} alpha={getattr(args, 'wan_lora_alpha', 16.0)} "
#             f"dropout={getattr(args, 'wan_lora_dropout', 0.0)} "
#             f"targets={getattr(args, 'wan_lora_targets', '')} "
#             f"extra={getattr(args, 'wan_lora_extra_name_pattern', '') or '<none>'}"
#         )
#         print(
#             f"  T5 对齐(已禁用) : cfg_enabled={int(bool(args.enable_t5_alignment))} "
#             f"mode={args.t5_align_mode} "
#             f"anchor={args.t5_align_anchor_tokens} "
#             f"l2={args.lambda_t5_align_l2} cos={args.lambda_t5_align_cos} stats={args.lambda_t5_align_stats} "
#             f"ot_eps={args.t5_align_ot_epsilon} ot_iters={args.t5_align_ot_iters}"
#         )
#         print(
#             f"  图像保持(已禁用): cfg_enabled={int(bool(args.enable_mq_image_preserve))} "
#             f"lambda={args.lambda_mq_image_preserve} margin={args.mq_image_preserve_margin}"
#         )
#         print(
#             f"  MQ/T5范数探针  : enabled={int(bool(args.mq_norm_probe_with_t5))} "
#             f"every={args.mq_norm_probe_every_n_steps} "
#             f"warn=[{args.mq_norm_warn_ratio_low},{args.mq_norm_warn_ratio_high}] "
#             f"match_t5={int(bool(args.mq_norm_match_t5))} "
#             f"clip=[{args.mq_norm_match_clip_min},{args.mq_norm_match_clip_max}]"
#         )
#         print(
#             f"  函数蒸馏(已禁用): cfg_enabled={int(bool(args.enable_wan_func_distill))} "
#             f"lambda={args.lambda_wan_func_distill} teacher={args.wan_func_teacher_mode}"
#         )
#         print("  额外损失开关   : 当前版本固定仅使用 denoise MSE（其余辅助损失已禁用）")
#         print("=" * 60)

#         self._load_models()
#         self._log_runtime_topology()
#         self._setup_optimizer()
#         self._audit_runtime_trainability(stage="init")
#         self._init_trainability_monitor()
#         self._init_wandb()

#     def _is_main_process(self):
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             return torch.distributed.get_rank() == 0
#         rank_env = os.environ.get("RANK")
#         if rank_env is None:
#             return True
#         return int(rank_env) == 0

#     def _mq_encoder_module(self):
#         return self.mq_encoder.module if hasattr(self.mq_encoder, "module") else self.mq_encoder

#     def _mq_trainable_params(self):
#         module = self._mq_encoder_module()
#         if hasattr(module, "get_trainable_params"):
#             return module.get_trainable_params()
#         return [p for p in module.parameters() if p.requires_grad]

#     def _wan_lora_enabled(self) -> bool:
#         return bool(getattr(self.args, "enable_wan_lora", False))

#     def _resolve_wan_train_mode(self) -> str:
#         if self._wan_lora_enabled():
#             return "lora"
#         mode = str(getattr(self.args, "wan_train_mode", "auto")).strip().lower()
#         if mode != "auto":
#             return mode
#         total_gb = 0.0
#         if self.dev_dit.type == "cuda" and torch.cuda.is_available():
#             try:
#                 props = torch.cuda.get_device_properties(self.dev_dit)
#                 total_gb = float(props.total_memory) / float(1024 ** 3)
#             except Exception:
#                 total_gb = 0.0
#         threshold = float(getattr(self.args, "wan_auto_full_mem_gb", 120.0))
#         return "full" if total_gb >= threshold else "cond_only"

#     def _wan_cond_keywords(self) -> List[str]:
#         custom = str(getattr(self.args, "wan_cond_name_pattern", "")).strip()
#         if custom:
#             return [k.strip().lower() for k in custom.split(",") if k.strip()]
#         return [
#             "cross_attn",
#             "cross-attn",
#             "crossattention",
#             "cross_attention",
#             "text_embedding",
#             "time_projection",
#             "modulation",
#             "cross_attn_norm",
#             "norm3",
#         ]

#     def _wan_lora_extra_keywords(self) -> List[str]:
#         custom = str(getattr(self.args, "wan_lora_extra_name_pattern", "")).strip()
#         if not custom:
#             return []
#         return [k.strip().lower() for k in custom.split(",") if k.strip()]

#     def _configure_wan_trainable_params(self) -> None:
#         wan_model = getattr(self.wan, "model", None)
#         if wan_model is None:
#             self._effective_wan_train_mode = "frozen"
#             self._wan_trainable_names = []
#             self._wan_trainable_params_cache = []
#             self._wan_lora_module_names = []
#             self._wan_lora_extra_trainable_names = []
#             return

#         # 先全冻结，再按模式打开。
#         self._force_freeze(wan_model)
#         mode = self._resolve_wan_train_mode()
#         self._effective_wan_train_mode = mode
#         selected_names: List[str] = []
#         selected_params: List[torch.nn.Parameter] = []
#         self._wan_lora_module_names = []
#         self._wan_lora_extra_trainable_names = []

#         if mode == "full":
#             for name, p in wan_model.named_parameters():
#                 p.requires_grad_(True)
#                 selected_names.append(name)
#                 selected_params.append(p)
#         elif mode == "cond_only":
#             kws = self._wan_cond_keywords()
#             for name, p in wan_model.named_parameters():
#                 lname = name.lower()
#                 if any(kw in lname for kw in kws):
#                     p.requires_grad_(True)
#                     selected_names.append(name)
#                     selected_params.append(p)
#         elif mode == "lora":
#             if bool(getattr(self.args, "dit_fsdp", False)) or bool(getattr(self.args, "use_sp", False)):
#                 raise RuntimeError("Wan LoRA 当前仅支持非 dit_fsdp / 非 use_sp 训练路径")
#             self._wan_lora_module_names = apply_lora_to_wan_model(
#                 wan_model,
#                 rank=int(getattr(self.args, "wan_lora_rank", 16)),
#                 alpha=float(getattr(self.args, "wan_lora_alpha", 16.0)),
#                 dropout=float(getattr(self.args, "wan_lora_dropout", 0.0)),
#                 target_types=getattr(self.args, "wan_lora_targets", "self_attn,cross_attn,ffn"),
#             )
#             if not self._wan_lora_module_names:
#                 raise RuntimeError("启用了 Wan LoRA，但未匹配到任何可注入的 Wan Linear")
#             extra_kws = self._wan_lora_extra_keywords()
#             for name, p in wan_model.named_parameters():
#                 lname = name.lower()
#                 is_lora = (".lora_a" in lname) or (".lora_b" in lname)
#                 is_extra = (not is_lora) and any(kw in lname for kw in extra_kws)
#                 if is_lora or is_extra:
#                     p.requires_grad_(True)
#                     selected_names.append(name)
#                     selected_params.append(p)
#                     if is_extra:
#                         self._wan_lora_extra_trainable_names.append(name)
#         elif mode == "frozen":
#             pass
#         else:
#             raise ValueError(f"Unknown --wan_train_mode: {mode}")

#         self._wan_trainable_names = selected_names
#         self._wan_trainable_params_cache = selected_params
#         if selected_params:
#             wan_model.train()
#         else:
#             wan_model.eval()

#         if self.is_main_process:
#             total = sum(int(p.numel()) for p in selected_params)
#             print(
#                 f"[WAN-TRAIN] requested={self.args.wan_train_mode} effective={mode} "
#                 f"trainable_tensors={len(selected_params)} trainable_params={total:,}"
#             )
#             if mode == "cond_only":
#                 kws = self._wan_cond_keywords()
#                 preview = ", ".join(kws[:10])
#                 print(f"[WAN-TRAIN] cond_only keywords={preview}")
#             if mode == "lora":
#                 lora_cfg = build_lora_config_dict(
#                     enabled=True,
#                     rank=int(getattr(self.args, "wan_lora_rank", 16)),
#                     alpha=float(getattr(self.args, "wan_lora_alpha", 16.0)),
#                     dropout=float(getattr(self.args, "wan_lora_dropout", 0.0)),
#                     targets=getattr(self.args, "wan_lora_targets", "self_attn,cross_attn,ffn"),
#                     module_names=self._wan_lora_module_names,
#                 )
#                 print(f"[WAN-TRAIN] lora_cfg={json.dumps(lora_cfg, ensure_ascii=False)}")
#                 if self._wan_lora_extra_trainable_names:
#                     preview = ", ".join(self._wan_lora_extra_trainable_names[:12])
#                     more = "" if len(self._wan_lora_extra_trainable_names) <= 12 else f" ... +{len(self._wan_lora_extra_trainable_names)-12}"
#                     print(f"[WAN-TRAIN] lora extra preview: {preview}{more}")
#             if selected_names:
#                 preview = ", ".join(selected_names[:8])
#                 more = "" if len(selected_names) <= 8 else f" ... +{len(selected_names)-8}"
#                 print(f"[WAN-TRAIN] selected preview: {preview}{more}")

#     def _wan_trainable_params(self) -> List[torch.nn.Parameter]:
#         return list(self._wan_trainable_params_cache)

#     def _all_trainable_params(self) -> List[torch.nn.Parameter]:
#         out: List[torch.nn.Parameter] = []
#         seen = set()
#         for p in self._mq_trainable_params():
#             if id(p) not in seen:
#                 out.append(p)
#                 seen.add(id(p))
#         for p in self._wan_trainable_params():
#             if id(p) not in seen:
#                 out.append(p)
#                 seen.add(id(p))
#         return out

#     @staticmethod
#     def _module_param_stats(module: nn.Module | None) -> Dict[str, int]:
#         total = 0
#         trainable = 0
#         if module is None or not isinstance(module, nn.Module):
#             return {"total": 0, "trainable": 0}
#         for p in module.parameters():
#             n = int(p.numel())
#             total += n
#             if p.requires_grad:
#                 trainable += n
#         return {"total": total, "trainable": trainable}

#     @staticmethod
#     def _named_param_id_map(module: nn.Module | None, prefix: str) -> Dict[int, str]:
#         out: Dict[int, str] = {}
#         if module is None or not isinstance(module, nn.Module):
#             return out
#         for name, p in module.named_parameters():
#             out[id(p)] = f"{prefix}.{name}"
#         return out

#     @staticmethod
#     def _force_freeze(module: nn.Module | None) -> None:
#         if module is None or not isinstance(module, nn.Module):
#             return
#         try:
#             module.eval()
#         except Exception:
#             pass
#         try:
#             module.requires_grad_(False)
#         except Exception:
#             for p in module.parameters():
#                 p.requires_grad_(False)

#     def _log_runtime_topology(self) -> None:
#         if not self.is_main_process:
#             return
#         args = self.args
#         same_gpu = (self.dev_dit == self.dev_enc)
#         print(
#             "[AUDIT][TOPO] "
#             f"dit_device={self.dev_dit} encoder_device={self.dev_enc} same_gpu={same_gpu} "
#             f"t5_cpu={args.t5_cpu} t5_fsdp={args.t5_fsdp} dit_fsdp={args.dit_fsdp} use_sp={args.use_sp} "
#             f"num_metaqueries={args.num_metaqueries} aug_text_len={getattr(self, '_aug_text_len', -1)} "
#             f"wan_mode_effective={getattr(self, '_effective_wan_train_mode', 'frozen')}"
#         )
#         if same_gpu:
#             print("[AUDIT][TOPO][WARN] DiT 与 Qwen/Connector 在同一 GPU，显存峰值风险较高。")
#         if (not args.t5_cpu) and (not args.t5_fsdp):
#             print("[AUDIT][TOPO] T5 文本编码器会在 DiT 卡上参与前向（no_grad）。")
#         try:
#             from wan.modules import attention as _attn
#             fa2 = bool(getattr(_attn, "FLASH_ATTN_2_AVAILABLE", False))
#             fa3 = bool(getattr(_attn, "FLASH_ATTN_3_AVAILABLE", False))
#             force_sdpa = bool(getattr(_attn, "_FORCE_SDPA", False))
#             print(
#                 "[AUDIT][ATTN] "
#                 f"flash_attn2={fa2} flash_attn3={fa3} force_sdpa={force_sdpa}"
#             )
#         except Exception as e:
#             print(f"[AUDIT][ATTN][WARN] 无法读取 attention backend 信息: {e}")

#     def _audit_runtime_trainability(self, stage: str = "runtime", strict: bool | None = None) -> None:
#         args = self.args
#         if strict is None:
#             strict = bool(getattr(args, "strict_freeze_check", True))

#         wan_mode = str(getattr(self, "_effective_wan_train_mode", "frozen"))
#         # Wan 是否冻结由 wan_train_mode 决定；T5/VAE 始终冻结。
#         t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
#         self._force_freeze(t5_model)
#         vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
#         if vae_model is None:
#             vae_model = getattr(self.wan, "vae", None)
#         self._force_freeze(vae_model)

#         stats_wan = self._module_param_stats(getattr(self.wan, "model", None))
#         stats_t5 = self._module_param_stats(t5_model)
#         stats_vae = self._module_param_stats(vae_model)

#         mq_module = self._mq_encoder_module()
#         stats_mq = self._module_param_stats(mq_module)
#         mq_trainable_params = self._mq_trainable_params()
#         wan_trainable_params = self._wan_trainable_params()
#         mq_trainable_ids = {id(p) for p in mq_trainable_params}
#         wan_trainable_ids = {id(p) for p in wan_trainable_params}
#         allowed_trainable_ids = mq_trainable_ids | wan_trainable_ids
#         emb_trainable = 0
#         emb_rows_total = 0
#         emb_rows_base = 0
#         emb_rows_added = 0
#         emb_hidden = 0
#         try:
#             backbone = mq_module.mllm_model.mllm_backbone
#             emb = backbone.get_input_embeddings()
#             if emb is not None and getattr(emb, "weight", None) is not None:
#                 w = emb.weight
#                 emb_rows_total = int(w.shape[0])
#                 emb_hidden = int(w.shape[1]) if w.ndim >= 2 else 0
#                 emb_rows_base = int(getattr(mq_module.mllm_model, "num_embeddings", 0))
#                 emb_rows_added = max(emb_rows_total - emb_rows_base, 0)
#                 if bool(w.requires_grad):
#                     emb_trainable = int(w.numel())
#         except Exception:
#             pass

#         opt_params = []
#         for g in self.optimizer.param_groups:
#             opt_params.extend(g.get("params", []))
#         opt_ids = [id(p) for p in opt_params]
#         opt_id_set = set(opt_ids)

#         outside_ids = [pid for pid in opt_ids if pid not in allowed_trainable_ids]
#         missing_mq_ids = [pid for pid in mq_trainable_ids if pid not in opt_id_set]
#         missing_wan_ids = [pid for pid in wan_trainable_ids if pid not in opt_id_set]
#         duplicate_count = max(len(opt_ids) - len(opt_id_set), 0)

#         name_map: Dict[int, str] = {}
#         name_map.update(self._named_param_id_map(getattr(self.wan, "model", None), "wan.model"))
#         name_map.update(self._named_param_id_map(t5_model, "wan.text_encoder.model"))
#         name_map.update(self._named_param_id_map(vae_model, "wan.vae.model"))
#         name_map.update(self._named_param_id_map(mq_module, "mq_encoder"))

#         unexpected_mq_names = []
#         for name, p in mq_module.named_parameters():
#             if not p.requires_grad:
#                 continue
#             lower = name.lower()
#             if ("connector" in lower) or ("embed" in lower):
#                 continue
#             unexpected_mq_names.append(name)

#         if self.is_main_process:
#             print(
#                 f"[AUDIT][FREEZE][{stage}] "
#                 f"wan_trainable={stats_wan['trainable']:,}/{stats_wan['total']:,} "
#                 f"t5_trainable={stats_t5['trainable']:,}/{stats_t5['total']:,} "
#                 f"vae_trainable={stats_vae['trainable']:,}/{stats_vae['total']:,} "
#                 f"mq_trainable={stats_mq['trainable']:,}/{stats_mq['total']:,} "
#                 f"mq_trainable_tensors={len(mq_trainable_params)} "
#                 f"wan_mode={wan_mode} wan_trainable_tensors={len(wan_trainable_params)}"
#             )
#             print(
#                 f"[AUDIT][OPT][{stage}] "
#                 f"optimizer_params={len(opt_ids)} "
#                 f"outside_allowed={len(outside_ids)} missing_mq={len(missing_mq_ids)} "
#                 f"missing_wan={len(missing_wan_ids)} duplicates={duplicate_count}"
#             )
#             print(
#                 f"[AUDIT][MQ-EMB][{stage}] "
#                 f"enabled={int(bool(args.train_mq_input_embeddings))} "
#                 f"embed_trainable={emb_trainable:,} "
#                 f"rows_total={emb_rows_total} rows_base={emb_rows_base} rows_added={emb_rows_added} "
#                 f"hidden={emb_hidden} expected_added≈num_metaqueries+2={int(args.num_metaqueries) + 2}"
#             )
#             if unexpected_mq_names:
#                 preview = ", ".join(unexpected_mq_names[:6])
#                 more = "" if len(unexpected_mq_names) <= 6 else f" ... +{len(unexpected_mq_names)-6}"
#                 print(
#                     "[AUDIT][MQ][WARN] 检测到非 connector/embed 命名的可训练参数: "
#                     f"{preview}{more}"
#                 )

#         errors = []
#         if wan_mode == "frozen" and stats_wan["trainable"] > 0:
#             errors.append(f"Wan DiT 期望冻结但仍有可训练参数: {stats_wan['trainable']}")
#         if wan_mode != "frozen" and len(wan_trainable_ids) == 0:
#             errors.append(f"Wan DiT 训练模式={wan_mode} 但未选中可训练参数")
#         if stats_t5["trainable"] > 0:
#             errors.append(f"Wan T5 仍有可训练参数: {stats_t5['trainable']}")
#         if stats_vae["trainable"] > 0:
#             errors.append(f"Wan VAE 仍有可训练参数: {stats_vae['trainable']}")
#         if len(mq_trainable_ids) == 0:
#             errors.append("MQ encoder 无可训练参数")
#         if bool(args.train_mq_input_embeddings) and emb_trainable <= 0:
#             errors.append("设置了 train_mq_input_embeddings，但输入 embedding 未开启训练")
#         if (not bool(args.train_mq_input_embeddings)) and emb_trainable > 0:
#             errors.append("设置了 freeze_mq_input_embeddings，但输入 embedding 仍可训练")
#         if outside_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in outside_ids[:8]]
#             errors.append(f"optimizer 含非允许参数(MQ+Wan): {names}")
#         if missing_mq_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_mq_ids[:8]]
#             errors.append(f"部分 MQ 可训练参数未进 optimizer: {names}")
#         if missing_wan_ids:
#             names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_wan_ids[:8]]
#             errors.append(f"部分 Wan 可训练参数未进 optimizer: {names}")
#         if duplicate_count > 0:
#             errors.append(f"optimizer 参数重复引用: {duplicate_count}")
#         if wan_mode != "frozen" and torch.distributed.is_available() and torch.distributed.is_initialized():
#             ws = int(torch.distributed.get_world_size())
#             if ws > 1:
#                 dit_fsdp_enabled = bool(getattr(args, "dit_fsdp", False))
#                 use_sp_enabled = bool(getattr(args, "use_sp", False))
#                 if not (dit_fsdp_enabled or use_sp_enabled):
#                     errors.append(
#                         "多进程 Wan 可训练模式需要启用 dit_fsdp 或 use_sp；"
#                         "否则当前仅有 MQ-encoder DDP 会导致 Wan 参数跨 rank 不一致。"
#                     )

#         if errors:
#             msg = " | ".join(errors)
#             if strict:
#                 raise RuntimeError(f"[AUDIT][FAIL][{stage}] {msg}")
#             if self.is_main_process:
#                 print(f"[AUDIT][WARN][{stage}] {msg}")

#     def post_wrap_ddp_audit(self) -> None:
#         # DDP 包装后再做一次 optimizer 与 trainable 参数一致性检查
#         if not hasattr(self.mq_encoder, "module"):
#             return
#         self._audit_runtime_trainability(stage="post_ddp")

#     def _log_grad_health_once(self):
#         if self._printed_grad_health:
#             return
#         module = self._mq_encoder_module()
#         connector_has_grad = False
#         mq_embed_has_grad = False
#         wan_has_grad = False
#         connector_grad_norm = 0.0
#         mq_embed_grad_norm = 0.0
#         wan_grad_norm = 0.0
#         mq_embed_added_grad_norm = 0.0
#         mq_embed_base_grad_norm = 0.0
#         mq_embed_boundary_grad_norm = 0.0
#         mq_embed_query_grad_norm = 0.0
#         try:
#             for _, p in module.mllm_model.connector.named_parameters():
#                 if p.grad is not None:
#                     connector_has_grad = True
#                     connector_grad_norm = float(p.grad.detach().float().norm().item())
#                     break
#             emb = module.mllm_model.mllm_backbone.get_input_embeddings()
#             if emb is not None and getattr(emb, "weight", None) is not None and emb.weight.grad is not None:
#                 mq_embed_has_grad = True
#                 g = emb.weight.grad.detach().float()
#                 mq_embed_grad_norm = float(g.norm().item())
#                 base_rows = int(getattr(module.mllm_model, "num_embeddings", 0))
#                 if g.ndim >= 2 and 0 < base_rows < int(g.shape[0]):
#                     mq_embed_base_grad_norm = float(g[:base_rows].norm().item())
#                     mq_embed_added_grad_norm = float(g[base_rows:].norm().item())
#                     boundary_end = min(base_rows + 2, int(g.shape[0]))
#                     query_end = min(boundary_end + int(self.args.num_metaqueries), int(g.shape[0]))
#                     if boundary_end > base_rows:
#                         mq_embed_boundary_grad_norm = float(g[base_rows:boundary_end].norm().item())
#                     if query_end > boundary_end:
#                         mq_embed_query_grad_norm = float(g[boundary_end:query_end].norm().item())
#         except Exception:
#             pass
#         try:
#             for p in self._wan_trainable_params():
#                 if p.grad is not None:
#                     wan_has_grad = True
#                     wan_grad_norm = float(p.grad.detach().float().norm().item())
#                     break
#         except Exception:
#             pass
#         print(
#             "[GRAD-CHECK] "
#             f"connector_has_grad={connector_has_grad} connector_grad_norm={connector_grad_norm:.4e} "
#             f"mq_embed_has_grad={mq_embed_has_grad} mq_embed_grad_norm={mq_embed_grad_norm:.4e} "
#             f"wan_has_grad={wan_has_grad} wan_grad_norm={wan_grad_norm:.4e} "
#             f"mq_embed_added_grad_norm={mq_embed_added_grad_norm:.4e} "
#             f"mq_embed_base_grad_norm={mq_embed_base_grad_norm:.4e} "
#             f"mq_embed_boundary_grad_norm={mq_embed_boundary_grad_norm:.4e} "
#             f"mq_embed_query_grad_norm={mq_embed_query_grad_norm:.4e}"
#         )
#         self._printed_grad_health = True

#     def _verify_train_context_injection_once(
#         self,
#         mq_feat: torch.Tensor,
#         aug_feat: torch.Tensor,
#     ) -> None:
#         if self._printed_context_inject_check:
#             return
#         mq_len = int(mq_feat.shape[0])
#         aug_len = int(aug_feat.shape[0])
#         if aug_len != mq_len:
#             raise RuntimeError(
#                 f"[VERIFY][TRAIN] MQ-only context 长度异常: aug={aug_len}, mq={mq_len}"
#             )
#         mq_ok = torch.allclose(
#             aug_feat.float(),
#             mq_feat.float(),
#             atol=1e-3,
#             rtol=1e-3,
#         )
#         if not mq_ok:
#             raise RuntimeError("[VERIFY][TRAIN] MQ-only context 未正确注入 Wan context")
#         if aug_len > self._aug_text_len:
#             raise RuntimeError(
#                 f"[VERIFY][TRAIN] aug_len 超出 text_len: aug={aug_len}, text_len={self._aug_text_len}"
#             )
#         print(
#             "[VERIFY][TRAIN] context 注入检查通过: "
#             f"mq_tokens={mq_len} aug_tokens={aug_len} model_text_len={self._aug_text_len}"
#         )
#         self._printed_context_inject_check = True

#     def _init_trainability_monitor(self):
#         self._param_monitor = []
#         total_sq = 0.0
#         sample_sq = 0.0
#         total_params = 0
#         named_params: List[Tuple[str, torch.nn.Parameter]] = []
#         mq_module = self._mq_encoder_module()
#         named_params.extend((f"mq_encoder.{n}", p) for n, p in mq_module.named_parameters() if p.requires_grad)
#         wan_model = getattr(self.wan, "model", None)
#         if isinstance(wan_model, nn.Module):
#             named_params.extend((f"wan.model.{n}", p) for n, p in wan_model.named_parameters() if p.requires_grad)
#         for name, p in named_params:
#             data = p.detach().float().view(-1)
#             numel = int(data.numel())
#             if numel <= 0:
#                 continue
#             sample_k = min(8, numel)
#             if sample_k == 1:
#                 idx = torch.zeros(1, dtype=torch.long)
#             else:
#                 idx = torch.linspace(0, numel - 1, steps=sample_k, dtype=torch.long)
#             init_vals = data.index_select(0, idx.to(data.device)).cpu()
#             self._param_monitor.append((name, p, idx.cpu(), init_vals))
#             total_sq += float(torch.sum(data * data).item())
#             sample_sq += float(torch.sum(init_vals * init_vals).item())
#             total_params += numel
#         self._trainable_param_count = total_params
#         self._init_trainable_norm = math.sqrt(max(total_sq, 0.0))
#         self._init_param_sample_norm = math.sqrt(max(sample_sq, 0.0))
#         if self.is_main_process:
#             print(
#                 "[VERIFY][TRAIN-INIT] "
#                 f"trainable_params={self._trainable_param_count:,} "
#                 f"init_param_norm={self._init_trainable_norm:.6f} "
#                 f"monitor_tensors={len(self._param_monitor)}"
#             )

#     def _collect_trainability_metrics(self):
#         sample_abs_sum = 0.0
#         sample_l2_sum = 0.0
#         sample_cur_sq_sum = 0.0
#         sample_count = 0
#         with torch.no_grad():
#             for _, p, idx_cpu, init_vals_cpu in self._param_monitor:
#                 data = p.detach().float().view(-1)
#                 idx = idx_cpu.to(data.device)
#                 now_vals = data.index_select(0, idx).cpu()
#                 diff = now_vals - init_vals_cpu
#                 sample_abs_sum += float(diff.abs().sum().item())
#                 sample_l2_sum += float(torch.sum(diff * diff).item())
#                 sample_cur_sq_sum += float(torch.sum(now_vals * now_vals).item())
#                 sample_count += int(diff.numel())
#         cur_sample_norm = math.sqrt(max(sample_cur_sq_sum, 0.0))
#         init_sample_norm = max(self._init_param_sample_norm, 1e-12)
#         return {
#             "train/param_sample_norm": float(cur_sample_norm),
#             "train/param_sample_norm_delta_ratio": float(abs(cur_sample_norm - self._init_param_sample_norm) / init_sample_norm),
#             "train/param_sample_abs_delta_mean": float(sample_abs_sum / max(sample_count, 1)),
#             "train/param_sample_l2_delta": float(math.sqrt(max(sample_l2_sum, 0.0))),
#             "train/trainable_param_count": int(self._trainable_param_count),
#         }

#     def _collect_cuda_memory_metrics(self):
#         if not (torch.cuda.is_available() and self.args.log_cuda_memory):
#             return {}
#         dit_idx = self.dev_dit.index if self.dev_dit.type == "cuda" else None
#         enc_idx = self.dev_enc.index if self.dev_enc.type == "cuda" else None

#         def _mem(prefix, dev_idx):
#             if dev_idx is None:
#                 return {}
#             return {
#                 f"train/cuda_{prefix}_alloc_mb": float(torch.cuda.memory_allocated(dev_idx) / 1024 / 1024),
#                 f"train/cuda_{prefix}_reserved_mb": float(torch.cuda.memory_reserved(dev_idx) / 1024 / 1024),
#                 f"train/cuda_{prefix}_max_alloc_mb": float(torch.cuda.max_memory_allocated(dev_idx) / 1024 / 1024),
#             }

#         metrics = {}
#         metrics.update(_mem("dit", dit_idx))
#         metrics.update(_mem("enc", enc_idx))
#         return metrics

#     def _append_metrics_jsonl(self, metrics):
#         if not self.is_main_process:
#             return
#         if not self._metrics_jsonl_path:
#             return
#         try:
#             path = Path(self._metrics_jsonl_path).expanduser().resolve()
#             path.parent.mkdir(parents=True, exist_ok=True)
#             with path.open("a", encoding="utf-8") as f:
#                 f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
#         except Exception as e:
#             print(f"[WARN] 写入 metrics_jsonl 失败: {e}")

#     def _record_metrics(self, metrics: Dict[str, Any]) -> None:
#         keep_keys = [
#             "train/step",
#             "train/loss_step",
#             "train/loss_denoise",
#             "train/loss_align_total",
#             "train/loss_align_t5_l2",
#             "train/loss_align_t5_cos",
#             "train/loss_align_t5_stats",
#             "train/loss_align_t5_gram",
#             "train/loss_align_t5_cka",
#             "train/loss_align_t5_ot",
#             "train/loss_align_img_preserve",
#             "train/loss_align_wan_func",
#             "train/loss_ema",
#             "train/lr",
#             "train/grad_norm",
#             "train/step_time_sec",
#             "train/samples_per_sec",
#             "train/param_sample_abs_delta_mean",
#             "train/param_sample_l2_delta",
#             "train/param_sample_norm_delta_ratio",
#             "train/skipped_step_count",
#             "train/oom_skip_count",
#             "train/error_skip_count",
#         ]
#         row = {k: metrics[k] for k in keep_keys if k in metrics}
#         self._metrics_history.append(row)

#     def _build_metrics_summary(self, step: int) -> Dict[str, Any]:
#         summary: Dict[str, Any] = {
#             "current_step": int(step),
#             "logged_steps": int(len(self._metrics_history)),
#             "metrics_jsonl_path": self._metrics_jsonl_path,
#             "skipped_step_count": int(self._skipped_step_count),
#             "oom_skip_count": int(self._oom_skip_count),
#             "error_skip_count": int(self._error_skip_count),
#         }
#         if self._metrics_history:
#             last = self._metrics_history[-1]
#             loss_vals = [float(m.get("train/loss_step", 0.0)) for m in self._metrics_history if "train/loss_step" in m]
#             step_time_vals = [float(m.get("train/step_time_sec", 0.0)) for m in self._metrics_history if "train/step_time_sec" in m]
#             sps_vals = [float(m.get("train/samples_per_sec", 0.0)) for m in self._metrics_history if "train/samples_per_sec" in m]
#             summary.update(
#                 {
#                     "step_first": int(self._metrics_history[0].get("train/step", 0)),
#                     "step_last": int(last.get("train/step", 0)),
#                     "loss_last": float(last.get("train/loss_step", 0.0)),
#                     "loss_ema_last": float(last.get("train/loss_ema", 0.0)),
#                     "lr_last": float(last.get("train/lr", 0.0)),
#                     "grad_norm_last": float(last.get("train/grad_norm", 0.0)),
#                     "loss_min": float(min(loss_vals) if loss_vals else 0.0),
#                     "loss_max": float(max(loss_vals) if loss_vals else 0.0),
#                     "step_time_sec_avg": float(sum(step_time_vals) / max(len(step_time_vals), 1)),
#                     "samples_per_sec_avg": float(sum(sps_vals) / max(len(sps_vals), 1)),
#                 }
#             )
#         if self._train_wall_start > 0:
#             summary["wall_time_sec"] = float(max(time.perf_counter() - self._train_wall_start, 0.0))
#         return summary

#     def _write_training_chain_manifest(self, output_dir: Path, final_checkpoint_path: str, final_step: int) -> None:
#         if not self.is_main_process:
#             return
#         output_dir = output_dir.expanduser().resolve()
#         payload = {
#             "before_checkpoint_path": self._train_before_checkpoint_path,
#             "final_checkpoint_path": str(Path(final_checkpoint_path).expanduser().resolve()),
#             "metrics_jsonl_path": self._metrics_jsonl_path,
#             "args": {str(k): _to_jsonable(v) for k, v in vars(self.args).items()},
#             "metrics_summary": self._build_metrics_summary(step=final_step),
#         }
#         _write_json(output_dir / "training_chain_manifest.json", payload)

#     def _wandb_config(self):
#         args = self.args
#         return {
#             "task": "wan_ti2v",
#             "learning_rate": args.learning_rate,
#             "num_train_steps": args.num_train_steps,
#             "warmup_steps": args.warmup_steps,
#             "lr_scheduler_type": args.lr_scheduler_type,
#             "cooldown_steps": args.cooldown_steps,
#             "lr_min_ratio": args.lr_min_ratio,
#             "enable_t5_alignment": args.enable_t5_alignment,
#             "t5_align_mode": args.t5_align_mode,
#             "t5_align_anchor_tokens": args.t5_align_anchor_tokens,
#             "lambda_t5_align_l2": args.lambda_t5_align_l2,
#             "lambda_t5_align_cos": args.lambda_t5_align_cos,
#             "lambda_t5_align_stats": args.lambda_t5_align_stats,
#             "t5_align_ot_epsilon": args.t5_align_ot_epsilon,
#             "t5_align_ot_iters": args.t5_align_ot_iters,
#             "enable_mq_image_preserve": args.enable_mq_image_preserve,
#             "lambda_mq_image_preserve": args.lambda_mq_image_preserve,
#             "mq_image_preserve_margin": args.mq_image_preserve_margin,
#             "mq_norm_probe_with_t5": args.mq_norm_probe_with_t5,
#             "mq_norm_probe_every_n_steps": args.mq_norm_probe_every_n_steps,
#             "mq_norm_warn_ratio_low": args.mq_norm_warn_ratio_low,
#             "mq_norm_warn_ratio_high": args.mq_norm_warn_ratio_high,
#             "mq_norm_match_t5": args.mq_norm_match_t5,
#             "mq_norm_match_clip_min": args.mq_norm_match_clip_min,
#             "mq_norm_match_clip_max": args.mq_norm_match_clip_max,
#             "enable_wan_func_distill": args.enable_wan_func_distill,
#             "lambda_wan_func_distill": args.lambda_wan_func_distill,
#             "wan_func_teacher_mode": args.wan_func_teacher_mode,
#             "batch_size": args.batch_size,
#             "gradient_accumulation_steps": args.gradient_accumulation_steps,
#             "max_grad_norm": args.max_grad_norm,
#             "frame_num": args.frame_num,
#             "max_area": args.max_area,
#             "num_metaqueries": args.num_metaqueries,
#             "connector_num_hidden_layers": args.connector_num_hidden_layers,
#             "dit_condition_mode": args.dit_condition_mode,
#             "mq_gradient_checkpointing": args.mq_gradient_checkpointing,
#             "train_mq_input_embeddings": args.train_mq_input_embeddings,
#             "null_caption_prob": args.null_caption_prob,
#             "null_image_prob": args.null_image_prob,
#             "wan_train_mode": args.wan_train_mode,
#             "wan_auto_full_mem_gb": args.wan_auto_full_mem_gb,
#             "wan_lr_ratio": args.wan_lr_ratio,
#             "wan_cond_name_pattern": args.wan_cond_name_pattern,
#             "enable_wan_lora": args.enable_wan_lora,
#             "wan_lora_rank": args.wan_lora_rank,
#             "wan_lora_alpha": args.wan_lora_alpha,
#             "wan_lora_dropout": args.wan_lora_dropout,
#             "wan_lora_targets": args.wan_lora_targets,
#             "wan_lora_extra_name_pattern": args.wan_lora_extra_name_pattern,
#             "t5_cpu": args.t5_cpu,
#             "dit_fsdp": args.dit_fsdp,
#             "t5_fsdp": args.t5_fsdp,
#             "use_sp": args.use_sp,
#             "aggressive_empty_cache": args.aggressive_empty_cache,
#             "seed": args.seed,
#             "save_steps": args.save_steps,
#             "log_steps": args.log_steps,
#             "enable_loss_early_stop": args.enable_loss_early_stop,
#             "loss_early_stop_min_step": args.loss_early_stop_min_step,
#             "loss_early_stop_threshold": args.loss_early_stop_threshold,
#             "log_every_step": args.log_every_step,
#             "wandb_log_every_step": args.wandb_log_every_step,
#             "metrics_jsonl_path": args.metrics_jsonl_path,
#             "log_cuda_memory": args.log_cuda_memory,
#             "output_dir": args.output_dir,
#             "local_openvid_video_root": args.local_openvid_video_root,
#             "local_openvid_csv_path": args.local_openvid_csv_path,
#             "local_openvid_limit": args.local_openvid_limit,
#             "local_openvid_hd_video_root": args.local_openvid_hd_video_root,
#             "local_openvid_hd_csv_path": args.local_openvid_hd_csv_path,
#             "local_openvid_hd_limit": args.local_openvid_hd_limit,
#             "wan_checkpoint_dir": args.wan_checkpoint_dir,
#             "qwen3vl_model_id": args.qwen3vl_model_id,
#         }

#     def _init_wandb(self):
#         args = self.args
#         if not getattr(args, "wandb_enabled", False):
#             return
#         if not self.is_main_process:
#             return
#         if args.wandb_api_key:
#             os.environ["WANDB_API_KEY"] = args.wandb_api_key
#         try:
#             import wandb
#         except ImportError:
#             print("[W&B] 未安装 wandb, 已跳过日志记录")
#             return
#         run_name = args.wandb_run_name.strip() or f"wan-ti2v-metaquery-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
#         tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
#         self.wandb = wandb
#         self.wandb_run = wandb.init(
#             project=args.wandb_project,
#             entity=args.wandb_entity or None,
#             name=run_name,
#             mode=args.wandb_mode,
#             config=self._wandb_config(),
#             tags=tags or None,
#         )
#         print(f"[W&B] 已初始化: project={args.wandb_project}, run={run_name}")

#     def _load_models(self):
#         """加载所有模型。"""
#         args = self.args

#         # ── 1. Wan TI2V Pipeline ─────────────────────────────────────────
#         print("\n[1/3] 加载 Wan TI2V Pipeline...")
#         from wan import WanTI2V
#         from wan.configs import WAN_CONFIGS

#         config = WAN_CONFIGS['ti2v-5B']
#         runtime_rank = (
#             torch.distributed.get_rank()
#             if torch.distributed.is_available() and torch.distributed.is_initialized()
#             else 0
#         )
#         self.wan = WanTI2V(
#             config=config,
#             checkpoint_dir=args.wan_checkpoint_dir,
#             device_id=args.dit_device,
#             rank=runtime_rank,
#             t5_fsdp=args.t5_fsdp,
#             dit_fsdp=args.dit_fsdp,
#             use_sp=args.use_sp,
#             t5_cpu=args.t5_cpu,
#             init_on_cpu=not args.no_init_on_cpu,
#             convert_model_dtype=args.convert_model_dtype,
#         )

#         # DiT 冻结；FSDP/SP 路径不再显式 .to，避免破坏分片包装
#         if not (args.dit_fsdp or args.use_sp):
#             self.wan.model.to(self.dev_dit)
#         self.wan.model.eval().requires_grad_(False)
#         t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
#         vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
#         if vae_model is None:
#             vae_model = getattr(self.wan, "vae", None)
#         self._force_freeze(t5_model)
#         self._force_freeze(vae_model)

#         self.wan_config = config
#         self.text_len = config.text_len  # 512
#         print(f"  ✅ Wan TI2V 5B 已加载, text_len={self.text_len}")

#         # ── 2. MetaQuery Encoder (直接输出 4096) ─────────────────────────
#         print("\n[2/3] 加载 MetaQuery Encoder (→4096)...")
#         # 统一使用 train_connector_for_wan.py 中的实现，避免同名类双份定义导致“改了不生效”。
#         from train_connector_for_wan import MetaQueryEncoderForWan as SharedMetaQueryEncoderForWan
#         self.mq_encoder = SharedMetaQueryEncoderForWan(
#             qwen3vl_model_id=args.qwen3vl_model_id,
#             num_metaqueries=args.num_metaqueries,
#             connector_num_hidden_layers=args.connector_num_hidden_layers,
#             gradient_checkpointing=args.mq_gradient_checkpointing,
#             train_input_embeddings=args.train_mq_input_embeddings,
#             dtype=torch.bfloat16,
#             device=f"cuda:{args.encoder_device}",
#         )
#         print(f"  ✅ Encoder实现来源: {self.mq_encoder.__class__.__module__}.{self.mq_encoder.__class__.__name__}")
#         self.mq_encoder.train()
#         if args.resume_mq_encoder_path:
#             state, resolved_path = load_mq_encoder_state(
#                 args.resume_mq_encoder_path,
#                 map_location="cpu",
#             )
#             missing, unexpected = self.mq_encoder.load_state_dict(state, strict=False)
#             print(f"  ✅ 已加载初始权重: {resolved_path}")
#             print(f"     missing={len(missing)}, unexpected={len(unexpected)}")
#         print(f"  ✅ MetaQuery Encoder 已加载")

#         # ── 3. 验证维度 ──────────────────────────────────────────────────
#         print("\n[3/3] 验证维度对齐...")
#         wan_text_dim = self.wan.model.text_dim  # 4096
#         mq_out_dim = self.mq_encoder.wan_text_dim  # 4096
#         assert wan_text_dim == mq_out_dim, (
#             f"维度不匹配! Wan text_dim={wan_text_dim}, MQ out={mq_out_dim}"
#         )
#         print(f"  ✅ MQ output dim = Wan text_dim = {wan_text_dim}")

#         # MQ-only: DiT text_len 仅容纳 MQ tokens
#         self._orig_text_len = self.wan.model.text_len
#         self._aug_text_len = args.num_metaqueries
#         print(f"  ✅ text_len(MQ-only): {self._orig_text_len} → {self._aug_text_len}")
#         self._configure_wan_trainable_params()

#     def _setup_optimizer(self):
#         """设置优化器和学习率调度。"""
#         args = self.args

#         mq_params = self._mq_trainable_params()
#         wan_params = self._wan_trainable_params()
#         trainable_params = self._all_trainable_params()
#         print(f"\n[Optimizer] 可训练参数组:")
#         print(f"  Connector + MQ Embeddings: {sum(p.numel() for p in mq_params) / 1e6:.1f}M")
#         print(f"  Wan DiT (mode={self._effective_wan_train_mode}): {sum(p.numel() for p in wan_params) / 1e6:.1f}M")
#         print(f"  Total trainable: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")
#         if len(trainable_params) <= 0:
#             raise RuntimeError("无可训练参数：请检查 MQ/Wan 训练配置。")

#         param_groups: List[Dict[str, Any]] = []
#         if mq_params:
#             param_groups.append(
#                 {
#                     "name": "mq",
#                     "params": mq_params,
#                     "lr": float(args.learning_rate),
#                 }
#             )
#         if wan_params:
#             param_groups.append(
#                 {
#                     "name": "wan",
#                     "params": wan_params,
#                     "lr": float(args.learning_rate) * float(getattr(args, "wan_lr_ratio", 1.0)),
#                 }
#             )

#         self.optimizer = torch.optim.AdamW(
#             param_groups,
#             betas=(0.9, 0.95),
#             weight_decay=0.1,
#             eps=1e-8,
#         )

#         def lr_lambda(step):
#             warmup = max(int(args.warmup_steps), 0)
#             total = max(int(args.num_train_steps), 1)
#             cooldown = int(getattr(args, "cooldown_steps", -1))
#             if cooldown < 0:
#                 cooldown = warmup
#             cooldown = max(cooldown, 0)
#             warmup = min(warmup, total)
#             cooldown = min(cooldown, max(total - warmup, 0))

#             if step < warmup:
#                 return step / max(1, warmup)
#             if args.lr_scheduler_type == "constant_with_warmup":
#                 return 1.0
#             if args.lr_scheduler_type == "warmup_hold_cooldown":
#                 cooldown_start = total - cooldown
#                 if cooldown <= 0 or step < cooldown_start:
#                     return 1.0
#                 progress = (step - cooldown_start) / max(1, cooldown)
#                 progress = min(max(progress, 0.0), 1.0)
#                 return 1.0 - (1.0 - float(args.lr_min_ratio)) * progress
#             progress = (step - warmup) / max(1, total - warmup)
#             return max(float(args.lr_min_ratio), 0.5 * (1.0 + math.cos(math.pi * progress)))

#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

#     def _encode_text(self, prompts):
#         """T5 编码文本"""
#         with torch.no_grad():
#             if not self.args.t5_cpu and not self.args.t5_fsdp:
#                 self.wan.text_encoder.model.to(self.dev_dit)
#                 context = self.wan.text_encoder(prompts, self.dev_dit)
#             else:
#                 context = self.wan.text_encoder(prompts, torch.device("cpu"))
#                 context = [t.to(self.dev_dit, dtype=torch.bfloat16) for t in context]
#         return context  # List[Tensor], each [text_len, 4096]

#     @staticmethod
#     def _resize_token_sequence(seq: torch.Tensor, out_tokens: int) -> torch.Tensor:
#         """
#         将 [L, D] token 序列重采样到 [out_tokens, D]。
#         使用线性插值仅做 teacher 侧长度对齐，不引入额外可训练参数。
#         """
#         if seq.dim() != 2:
#             raise ValueError(f"expect [L, D], got shape={tuple(seq.shape)}")
#         out_tokens = max(1, int(out_tokens))
#         if int(seq.shape[0]) == out_tokens:
#             return seq
#         # F.interpolate 期望 [N, C, L]
#         x = seq.transpose(0, 1).unsqueeze(0).float()
#         x = F.interpolate(x, size=out_tokens, mode="linear", align_corners=False)
#         return x.squeeze(0).transpose(0, 1)

#     @staticmethod
#     def _token_rms(feat: torch.Tensor) -> float:
#         x = feat.float()
#         return float(torch.sqrt(torch.mean(x * x)).item())

#     def _probe_and_optionally_match_mq_norm(
#         self,
#         captions: List[str],
#         mq_features: torch.Tensor,
#         t5_context: List[torch.Tensor] | None = None,
#     ) -> torch.Tensor:
#         args = self.args
#         enabled = bool(getattr(args, "mq_norm_probe_with_t5", True))
#         match_t5 = bool(getattr(args, "mq_norm_match_t5", False))
#         every = max(1, int(getattr(args, "mq_norm_probe_every_n_steps", 20)))
#         should_probe = enabled and ((int(self._loss_call_count) % every) == 0)
#         if not should_probe and not match_t5:
#             self._last_mq_norm_warn_flag = 0
#             self._last_mq_norm_match_scale = 1.0
#             return mq_features

#         try:
#             if t5_context is None:
#                 with torch.no_grad():
#                     t5_context = self._encode_text(captions)

#             t5_rms_vals = [self._token_rms(seq.to(self.dev_dit, dtype=torch.bfloat16)) for seq in t5_context]
#             t5_rms = float(sum(t5_rms_vals) / max(len(t5_rms_vals), 1)) if t5_rms_vals else 0.0
#             mq_rms = self._token_rms(mq_features)
#             ratio = float(mq_rms / (t5_rms + 1e-8))

#             self._last_mq_rms = float(mq_rms)
#             self._last_t5_rms = float(t5_rms)
#             self._last_mq_t5_rms_ratio = float(ratio)

#             low = float(getattr(args, "mq_norm_warn_ratio_low", 0.25))
#             high = float(getattr(args, "mq_norm_warn_ratio_high", 4.0))
#             warn_flag = int(ratio < low or ratio > high)
#             self._last_mq_norm_warn_flag = warn_flag
#             if warn_flag and self.is_main_process:
#                 print(
#                     "[MQ-NORM][WARN] "
#                     f"mq_rms={mq_rms:.6f} t5_rms={t5_rms:.6f} ratio={ratio:.6f} "
#                     f"outside=[{low}, {high}] loss_call={self._loss_call_count}"
#                 )

#             if match_t5 and t5_rms > 0:
#                 smin = float(getattr(args, "mq_norm_match_clip_min", 0.25))
#                 smax = float(getattr(args, "mq_norm_match_clip_max", 4.0))
#                 scale = float(t5_rms / (mq_rms + 1e-8))
#                 scale = float(max(smin, min(smax, scale)))
#                 self._last_mq_norm_match_scale = scale
#                 mq_features = mq_features * scale
#             else:
#                 self._last_mq_norm_match_scale = 1.0
#         except Exception as e:
#             self._last_mq_norm_warn_flag = 1
#             self._last_mq_norm_match_scale = 1.0
#             if self.is_main_process:
#                 print(f"[MQ-NORM][WARN] probe failed: {e}")

#         return mq_features

#     @staticmethod
#     def _token_gram_matrix(tokens: torch.Tensor) -> torch.Tensor:
#         """
#         计算 token 关系矩阵（Gram）。
#         输入: [B, T, D]，输出: [B, T, T]
#         """
#         if tokens.dim() != 3:
#             raise ValueError(f"expect [B, T, D], got shape={tuple(tokens.shape)}")
#         x = tokens - tokens.mean(dim=1, keepdim=True)
#         x = F.normalize(x, p=2, dim=-1, eps=1e-6)
#         return torch.matmul(x, x.transpose(1, 2))

#     @staticmethod
#     def _linear_cka_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         线性 CKA 损失，返回 1-CKA（越小越好）。
#         输入: x/y [B, T, D]
#         """
#         if x.shape != y.shape:
#             raise ValueError(f"CKA shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")
#         x_c = x - x.mean(dim=1, keepdim=True)
#         y_c = y - y.mean(dim=1, keepdim=True)
#         kx = torch.matmul(x_c, x_c.transpose(1, 2))
#         ky = torch.matmul(y_c, y_c.transpose(1, 2))
#         hsic = (kx * ky).sum(dim=(1, 2))
#         denom = torch.sqrt(
#             kx.square().sum(dim=(1, 2)).clamp_min(1e-12)
#             * ky.square().sum(dim=(1, 2)).clamp_min(1e-12)
#         )
#         cka = hsic / denom.clamp_min(1e-12)
#         return (1.0 - cka.clamp(-1.0, 1.0)).mean()

#     @staticmethod
#     def _sinkhorn_ot_cost(
#         src_tokens: torch.Tensor,
#         tgt_tokens: torch.Tensor,
#         epsilon: float = 0.05,
#         iters: int = 25,
#     ) -> torch.Tensor:
#         """
#         Sinkhorn OT 软匹配代价（排列无关）。
#         输入: src/tgt [B, T, D]
#         输出: 标量（batch 平均 OT cost）
#         """
#         if src_tokens.dim() != 3 or tgt_tokens.dim() != 3:
#             raise ValueError(
#                 f"Sinkhorn expect [B,T,D], got src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
#             )
#         if int(src_tokens.shape[0]) != int(tgt_tokens.shape[0]) or int(src_tokens.shape[2]) != int(tgt_tokens.shape[2]):
#             raise ValueError(
#                 f"Sinkhorn shape mismatch: src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
#             )
#         bsz, n_tok, _ = src_tokens.shape
#         m_tok = int(tgt_tokens.shape[1])
#         if n_tok <= 0 or m_tok <= 0:
#             return src_tokens.new_zeros(())

#         cost = torch.cdist(src_tokens, tgt_tokens, p=2).pow(2)  # [B, N, M]
#         eps = max(float(epsilon), 1e-6)
#         kernel = torch.exp(-cost / eps).clamp_min(1e-12)
#         a = src_tokens.new_full((bsz, n_tok), 1.0 / float(n_tok))
#         b = src_tokens.new_full((bsz, m_tok), 1.0 / float(m_tok))
#         u = torch.ones_like(a)
#         v = torch.ones_like(b)
#         kernel_t = kernel.transpose(1, 2)

#         n_iter = max(int(iters), 1)
#         for _ in range(n_iter):
#             kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
#             u = a / kv
#             ktu = torch.bmm(kernel_t, u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
#             v = b / ktu

#         plan = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)  # [B, N, M]
#         return (plan * cost).sum(dim=(1, 2)).mean()

#     def _compute_mq_aux_losses(
#         self,
#         captions: List[str],
#         mq_refs: List[Any],
#         mq_features: torch.Tensor,
#         t5_context: List[torch.Tensor] | None = None,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         计算 MQ 辅助约束：
#         1) T5 对齐（支持 anchor/Gram+CKA/Sinkhorn）
#         2) T5 统计对齐（均值/方差）
#         3) 图像保持（可选）：有图条件需与 text-only MQ 保持最小间隔
#         """
#         device = self.dev_dit
#         zero = mq_features.new_zeros(())
#         out = {
#             "t5_l2": zero,
#             "t5_cos": zero,
#             "t5_stats": zero,
#             "t5_gram": zero,
#             "t5_cka": zero,
#             "t5_ot": zero,
#             "image_preserve": zero,
#             "total": zero,
#         }
#         args = self.args
#         need_t5 = bool(args.enable_t5_alignment) and (
#             float(args.lambda_t5_align_l2) > 0.0
#             or float(args.lambda_t5_align_cos) > 0.0
#             or float(args.lambda_t5_align_stats) > 0.0
#         )
#         need_img = bool(args.enable_mq_image_preserve) and float(args.lambda_mq_image_preserve) > 0.0
#         if not (need_t5 or need_img):
#             return out

#         mq_float = mq_features.to(device=device, dtype=torch.float32)
#         tokens = int(mq_float.shape[1])
#         hidden = int(mq_float.shape[2])
#         anchor_tokens = max(1, min(int(args.t5_align_anchor_tokens), tokens))

#         if need_t5:
#             with torch.no_grad():
#                 teacher_ctx = t5_context if t5_context is not None else self._encode_text(captions)
#                 pooled_t5 = []
#                 for t5_seq in teacher_ctx:
#                     # t5_seq: [L_t5, 4096]
#                     t5_seq_f = t5_seq.to(device=device, dtype=torch.float32)
#                     if int(t5_seq_f.shape[-1]) != hidden:
#                         raise RuntimeError(
#                             f"T5 hidden={int(t5_seq_f.shape[-1])} 与 MQ hidden={hidden} 不一致"
#                         )
#                     pooled_t5.append(self._resize_token_sequence(t5_seq_f, tokens))
#                 t5_teacher = torch.stack(pooled_t5, dim=0)  # [B, tokens, 4096]

#             align_mode = str(getattr(args, "t5_align_mode", "gram_cka")).strip().lower()
#             if align_mode == "anchor":
#                 mq_anchor = mq_float[:, :anchor_tokens, :]
#                 t5_anchor = t5_teacher[:, :anchor_tokens, :]
#                 out["t5_l2"] = F.mse_loss(mq_anchor, t5_anchor)

#                 mq_anchor_flat = mq_anchor.reshape(-1, hidden)
#                 t5_anchor_flat = t5_anchor.reshape(-1, hidden)
#                 cos_sim = F.cosine_similarity(mq_anchor_flat, t5_anchor_flat, dim=-1).mean()
#                 out["t5_cos"] = (1.0 - cos_sim)
#             elif align_mode == "gram_cka":
#                 mq_gram = self._token_gram_matrix(mq_float)
#                 t5_gram = self._token_gram_matrix(t5_teacher)
#                 out["t5_gram"] = F.mse_loss(mq_gram, t5_gram)
#                 out["t5_cka"] = self._linear_cka_loss(mq_float, t5_teacher)
#                 # 复用旧命名，保持日志/脚本兼容
#                 out["t5_l2"] = out["t5_gram"]
#                 out["t5_cos"] = out["t5_cka"]
#             elif align_mode == "sinkhorn_ot":
#                 out["t5_ot"] = self._sinkhorn_ot_cost(
#                     mq_float,
#                     t5_teacher,
#                     epsilon=float(getattr(args, "t5_align_ot_epsilon", 0.05)),
#                     iters=int(getattr(args, "t5_align_ot_iters", 25)),
#                 )
#                 out["t5_l2"] = out["t5_ot"]
#             else:
#                 raise ValueError(f"Unknown --t5_align_mode: {align_mode}")

#             mq_mean = mq_float.mean(dim=1)
#             mq_std = mq_float.std(dim=1, unbiased=False)
#             t5_mean = t5_teacher.mean(dim=1)
#             t5_std = t5_teacher.std(dim=1, unbiased=False)
#             out["t5_stats"] = F.mse_loss(mq_mean, t5_mean) + F.mse_loss(mq_std, t5_std)

#         if need_img:
#             has_ref = torch.tensor(
#                 [1 if ref is not None else 0 for ref in mq_refs],
#                 device=device,
#                 dtype=torch.bool,
#             )
#             if bool(torch.any(has_ref).item()):
#                 with torch.no_grad():
#                     mq_text_only = self.mq_encoder(captions, None).to(device=device, dtype=torch.float32)
#                 diff = mq_float[has_ref] - mq_text_only[has_ref]
#                 # 每样本的 token+channel RMS 距离
#                 rms = torch.sqrt(torch.mean(diff * diff, dim=(1, 2)) + 1e-8)
#                 margin = float(args.mq_image_preserve_margin)
#                 out["image_preserve"] = F.relu(margin - rms).mean()

#         out["total"] = (
#             float(args.lambda_t5_align_l2) * out["t5_l2"]
#             + float(args.lambda_t5_align_cos) * out["t5_cos"]
#             + float(args.lambda_t5_align_stats) * out["t5_stats"]
#             + float(args.lambda_mq_image_preserve) * out["image_preserve"]
#         )
#         return out

#     def _encode_video(self, video_tensors):
#         """VAE 编码视频 → latent"""
#         with torch.no_grad():
#             # video_tensors: [B, 3, T, H, W] or list of [3, T, H, W]
#             latents = []
#             for v in video_tensors:
#                 # v: [3, T, H, W] → VAE expects this format
#                 z = self.wan.vae.encode([v.to(self.dev_dit, dtype=torch.bfloat16)])
#                 latents.append(z[0])  # z[0]: [C_z, T', H', W']
#         return latents

#     def _encode_first_frame(self, first_frame_tensor):
#         """VAE 编码参考图第一帧 → i2v condition latent"""
#         with torch.no_grad():
#             # first_frame: [3, H, W] → [3, 1, H, W]
#             ff = first_frame_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
#             z = self.wan.vae.encode([ff])
#         return z[0]  # [C_z, 1, H', W']

#     def _resolve_train_ref_anchor_mode(self) -> str:
#         """
#         返回当前 batch 实际使用的锚定模式。
#         - none / animate_like: 直接使用
#         - mixed50: 按 optimizer step 交替 none / animate_like，保证长期约 50/50
#         """
#         mode = str(getattr(self.args, "train_ref_anchor_mode", "none")).strip().lower()
#         if mode in ("none", "animate_like"):
#             return mode
#         if mode == "mixed50":
#             use_animate = (self._train_ref_anchor_mixed_counter % 2 == 1)
#             self._train_ref_anchor_mixed_counter += 1
#             return "animate_like" if use_animate else "none"
#         raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

#     def _train_ref_anchor_alpha(self, t_norm: torch.Tensor, mode: str | None = None) -> torch.Tensor:
#         """
#         训练期首帧软锚定系数（0~1）。
#         说明：
#         - none: 始终 0，不改动训练行为
#         - animate_like: 高噪声(早期)强锚定，随后余弦衰减到 0
#         """
#         if mode is None:
#             mode = self._resolve_train_ref_anchor_mode()
#         if mode == "none":
#             return torch.zeros_like(t_norm, dtype=torch.float32)
#         if mode != "animate_like":
#             raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

#         alpha0 = float(getattr(self.args, "train_ref_anchor_alpha0", 0.95))
#         warmup_ratio = float(getattr(self.args, "train_ref_anchor_warmup_ratio", 0.35))
#         alpha0 = max(0.0, min(1.0, alpha0))
#         warmup_ratio = max(0.0, min(1.0, warmup_ratio))
#         if warmup_ratio <= 0.0 or alpha0 <= 0.0:
#             return torch.zeros_like(t_norm, dtype=torch.float32)

#         start_t = 1.0 - warmup_ratio
#         alpha = torch.zeros_like(t_norm, dtype=torch.float32)
#         active = t_norm >= start_t
#         if not torch.any(active):
#             return alpha
#         u = ((t_norm[active] - start_t) / max(warmup_ratio, 1e-6)).clamp(0.0, 1.0)
#         alpha[active] = alpha0 * 0.5 * (1.0 - torch.cos(math.pi * u))
#         return alpha

#     @staticmethod
#     def _frames_to_latent_slots(frame_count: int, stride_t: int) -> int:
#         """像素帧数 -> latent 时间槽数（与 VAE 时间下采样保持一致）"""
#         f = max(0, int(frame_count))
#         if f <= 0:
#             return 0
#         return int((f - 1) // max(int(stride_t), 1) + 1)

#     def _encode_ref_image_to_latent(
#         self,
#         ref_img: Image.Image | None,
#         latent_h: int,
#         latent_w: int,
#         z_channels: int,
#     ) -> torch.Tensor:
#         """
#         将参考图编码为 1 帧 reference latent。
#         若 ref_img 缺失，返回零 reference latent。
#         """
#         if ref_img is None:
#             return torch.zeros(
#                 z_channels, 1, latent_h, latent_w,
#                 device=self.dev_dit, dtype=torch.float32,
#             )
#         target_h = int(latent_h * self.wan_config.vae_stride[1])
#         target_w = int(latent_w * self.wan_config.vae_stride[2])
#         ref_resized = ref_img.resize((target_w, target_h), Image.LANCZOS)
#         ref_np = np.array(ref_resized).astype(np.float32)
#         ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1) / 127.5 - 1.0
#         ref_5d = ref_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
#         with torch.no_grad():
#             ref_lat = self.wan.vae.encode([ref_5d])[0]
#         return ref_lat.float()

#     def _compute_wan_func_distill_loss(
#         self,
#         model_output: List[torch.Tensor],
#         x_inputs: List[torch.Tensor],
#         timesteps_wan: torch.Tensor,
#         max_seq_len: int,
#         t5_context: List[torch.Tensor],
#         mq_features: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         函数级蒸馏:
#             L_func = MSE( pred_mq(x_t,t), pred_t5(x_t,t) )
#         其中 pred_t5 由 frozen Wan + T5 条件生成（teacher no-grad）。
#         """
#         args = self.args
#         mode = str(getattr(args, "wan_func_teacher_mode", "t5_only")).strip().lower()
#         if mode not in {"t5_only", "t5_plus_mq"}:
#             raise ValueError(f"Unknown --wan_func_teacher_mode: {mode}")

#         teacher_context: List[torch.Tensor] = []
#         for i, t5_seq in enumerate(t5_context):
#             t5_feat = t5_seq.to(self.dev_dit, dtype=torch.bfloat16)
#             if mode == "t5_plus_mq":
#                 mq_feat = mq_features[i].detach().to(self.dev_dit, dtype=torch.bfloat16)
#                 t5_feat = torch.cat([mq_feat, t5_feat], dim=0)
#             teacher_context.append(t5_feat)

#         if not teacher_context:
#             return mq_features.new_zeros(())

#         teacher_text_len = max(int(ctx.shape[0]) for ctx in teacher_context)
#         cur_text_len = int(self.wan.model.text_len)
#         self.wan.model.text_len = teacher_text_len
#         try:
#             with torch.no_grad():
#                 with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#                     teacher_output = self.wan.model(
#                         x_inputs,
#                         t=timesteps_wan,
#                         context=teacher_context,
#                         seq_len=max_seq_len,
#                     )
#         finally:
#             self.wan.model.text_len = cur_text_len

#         loss = 0.0
#         valid = 0
#         for i in range(len(model_output)):
#             pred_mq = model_output[i].float()
#             pred_t5 = teacher_output[i].float()
#             loss = loss + F.mse_loss(pred_mq, pred_t5)
#             valid += 1
#         if valid <= 0:
#             return mq_features.new_zeros(())
#         return loss / valid

#     def _compute_loss(self, batch):
#         """
#         计算一个 batch 的 Flow Matching 损失。

#         训练默认使用 t2v 模式 (无第一帧蒙版/无首帧锚定)。
#         可通过 --train_ref_anchor_mode 在 x_t 注入 animate-like 首帧软锚定，
#         以缓解与 i2v 推理分布不一致问题。
#         """
#         args = self.args
#         captions = batch["caption"]
#         videos = batch["video"]         # list of [3, T, H, W]
#         mq_refs = batch["mq_ref_image"]  # list of PIL or None
#         B = len(captions)
#         self._last_loss_denoise = 0.0
#         self._last_loss_aux_align_total = 0.0
#         self._last_loss_aux_t5_l2 = 0.0
#         self._last_loss_aux_t5_cos = 0.0
#         self._last_loss_aux_t5_stats = 0.0
#         self._last_loss_aux_t5_gram = 0.0
#         self._last_loss_aux_t5_cka = 0.0
#         self._last_loss_aux_t5_ot = 0.0
#         self._last_loss_aux_image_preserve = 0.0
#         self._last_loss_aux_wan_func = 0.0
#         self._loss_call_count += 1

#         # ── 1. MetaQuery 编码 (在 encoder 设备上, 有梯度) ────────────────
#         mq_images = []
#         for ref in mq_refs:
#             if ref is not None:
#                 mq_images.append([ref])
#             else:
#                 mq_images.append(None)

#         all_none = all(img is None for img in mq_images)
#         if all_none:
#             mq_features = self.mq_encoder(captions, None)
#         else:
#             for i, img in enumerate(mq_images):
#                 if img is None:
#                     mq_images[i] = [Image.new("RGB", (224, 224))]
#             mq_features = self.mq_encoder(captions, mq_images)
#         # mq_features: [B, 256, 4096], 有梯度
#         mq_features = self._probe_and_optionally_match_mq_norm(
#             captions=captions,
#             mq_features=mq_features,
#             t5_context=None,
#         )

#         # ── 2. MQ-only 注入 DiT context ─────────────────────────────────
#         augmented_context = []
#         for i in range(B):
#             mq_feat = mq_features[i].to(self.dev_dit, dtype=torch.bfloat16)
#             aug = mq_feat
#             if i == 0:
#                 self._verify_train_context_injection_once(mq_feat, aug)
#             augmented_context.append(aug)

#         # ── 4. VAE 编码视频 → latent (无梯度) ───────────────────────────
#         with torch.no_grad():
#             latents = self._encode_video(videos)
#             # latents: list of [C_z, T', H', W']

#         # ── 4. 采样噪声和时间步, 构建 Flow Matching 目标 ─────────────────
#         patch_size = self.wan_config.patch_size
#         stride_t = int(self.wan_config.vae_stride[0])

#         first_frame_condition_enabled = bool(
#             getattr(args, "enable_ti2v_first_frame_condition", True)
#         )
#         mode_cfg = str(getattr(args, "train_video_conditioning_mode", "legacy_t2v")).strip().lower()
#         if mode_cfg not in ("legacy_t2v", "wan_animate_slot"):
#             mode_cfg = "legacy_t2v"
#         effective_video_mode = mode_cfg if first_frame_condition_enabled else "mq_only"

#         x_inputs = []
#         timestep_rows = []
#         target_list = []
#         prefix_slots_list = []
#         target_slots_list = []
#         max_seq_len = 0
#         ref_anchor_alpha_sum = 0.0
#         ref_anchor_applied = 0
#         ref_anchor_mode_effective = "none"

#         for i, lat in enumerate(latents):
#             C, T, H, W = lat.shape
#             lat = lat.float()
#             x0_for_fm = lat
#             prefix_slots_i = 0

#             # Wan 侧首帧条件：在训练阶段将参考图转换为 latent 并注入
#             ref_lat = None
#             if first_frame_condition_enabled:
#                 ref_lat = self._encode_ref_image_to_latent(
#                     mq_refs[i],
#                     latent_h=H,
#                     latent_w=W,
#                     z_channels=C,
#                 ).to(self.dev_dit, dtype=torch.float32)

#             if effective_video_mode == "wan_animate_slot":
#                 ref_slots = self._frames_to_latent_slots(
#                     int(getattr(args, "train_animate_ref_frames", 1)),
#                     stride_t=stride_t,
#                 )
#                 temporal_slots = self._frames_to_latent_slots(
#                     int(getattr(args, "train_animate_temporal_frames", 0)),
#                     stride_t=stride_t,
#                 )
#                 conditional_slots = self._frames_to_latent_slots(
#                     int(getattr(args, "train_animate_conditional_frames", 0)),
#                     stride_t=stride_t,
#                 )
#                 ref_slots = max(0, int(ref_slots))
#                 temporal_slots = max(0, int(temporal_slots))
#                 conditional_slots = max(0, int(conditional_slots))
#                 prefix_slots_i = ref_slots + temporal_slots + conditional_slots
#                 if prefix_slots_i > 0:
#                     prefix_chunks = []
#                     if ref_slots > 0:
#                         if ref_lat is None:
#                             ref_prefix = torch.zeros(
#                                 C, ref_slots, H, W, device=self.dev_dit, dtype=torch.float32
#                             )
#                         else:
#                             ref_prefix = ref_lat.repeat(1, ref_slots, 1, 1)
#                         prefix_chunks.append(ref_prefix)
#                     if temporal_slots > 0:
#                         prefix_chunks.append(
#                             torch.zeros(
#                                 C, temporal_slots, H, W, device=self.dev_dit, dtype=torch.float32
#                             )
#                         )
#                     if conditional_slots > 0:
#                         prefix_chunks.append(
#                             torch.zeros(
#                                 C, conditional_slots, H, W, device=self.dev_dit, dtype=torch.float32
#                             )
#                         )
#                     x0_prefix = torch.cat(prefix_chunks, dim=1)
#                     x0_for_fm = torch.cat([x0_prefix, lat], dim=1)

#             T_full = int(x0_for_fm.shape[1])
#             tokens_per_frame = int(math.ceil((H * W) / (patch_size[1] * patch_size[2])))
#             seq_len_i = int(tokens_per_frame * T_full)
#             max_seq_len = max(max_seq_len, seq_len_i)

#             t_val = torch.rand(1, device=self.dev_dit, dtype=torch.float32)
#             noise = torch.randn_like(x0_for_fm, dtype=torch.float32)

#             # Flow matching: x_t = (1-t) * x_0 + t * noise
#             sigma = t_val.view(-1, 1, 1, 1)
#             noisy_lat = (1.0 - sigma) * x0_for_fm + sigma * noise

#             if effective_video_mode == "legacy_t2v" and ref_lat is not None:
#                 ref_mode = str(getattr(self, "_current_train_ref_anchor_mode", "none")).strip().lower()
#                 if ref_mode not in ("none", "animate_like"):
#                     ref_mode = self._resolve_train_ref_anchor_mode()
#                 alpha_tensor = self._train_ref_anchor_alpha(t_val, mode=ref_mode)
#                 alpha_scalar = float(alpha_tensor.item())
#                 ref_anchor_mode_effective = ref_mode
#                 if alpha_scalar > 0.0:
#                     noisy_lat[:, :1] = (1.0 - alpha_scalar) * noisy_lat[:, :1] + alpha_scalar * ref_lat
#                     ref_anchor_alpha_sum += alpha_scalar
#                     ref_anchor_applied += 1

#             # 目标: noise - x_0 (velocity)
#             velocity = noise - x0_for_fm

#             # token 级 timestep：MQ-only 下全部 token 共享 t
#             t_scalar = float((t_val * self.wan.num_train_timesteps).item())
#             t_row = torch.full((seq_len_i,), t_scalar, device=self.dev_dit, dtype=torch.float32)
#             if (
#                 effective_video_mode == "wan_animate_slot"
#                 and prefix_slots_i > 0
#                 and bool(getattr(args, "train_animate_preserve_timestep_zero", True))
#             ):
#                 prefix_token_count = min(seq_len_i, int(prefix_slots_i * tokens_per_frame))
#                 if prefix_token_count > 0:
#                     t_row[:prefix_token_count] = 0.0

#             x_inputs.append(noisy_lat)
#             target_list.append(velocity)
#             prefix_slots_list.append(prefix_slots_i)
#             timestep_rows.append(t_row)
#             target_slots_list.append(T)

#         # 拼接 timestep → [B, max_seq_len]
#         padded_rows = []
#         for row in timestep_rows:
#             pad_len = max_seq_len - int(row.numel())
#             if pad_len > 0:
#                 pad_val = float(row[-1].item()) if row.numel() > 0 else 0.0
#                 row = torch.cat([row, row.new_full((pad_len,), pad_val)], dim=0)
#             padded_rows.append(row)
#         timesteps_wan = torch.stack(padded_rows, dim=0).to(self.dev_dit)

#         self._last_train_ref_anchor_alpha_mean = (
#             float(ref_anchor_alpha_sum / ref_anchor_applied) if ref_anchor_applied > 0 else 0.0
#         )
#         self._last_train_ref_anchor_applied = int(ref_anchor_applied)
#         self._last_train_ref_anchor_effective_mode = (
#             ref_anchor_mode_effective if ref_anchor_applied > 0 else "none"
#         )
#         self._last_train_video_conditioning_mode = str(effective_video_mode)
#         self._last_train_prefix_latent_slots = int(
#             round(sum(prefix_slots_list) / max(len(prefix_slots_list), 1))
#         )
#         self._last_train_target_latent_slots = int(round(sum(target_slots_list) / max(len(target_slots_list), 1)))
#         self._last_train_prefix_loss_dropped = 0

#         # ── 5. MQ-only text_len + DiT forward ───────────────────────────
#         orig_text_len = self.wan.model.text_len
#         self.wan.model.text_len = self._aug_text_len

#         try:
#             with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#                 model_output = self.wan.model(
#                     x_inputs,
#                     t=timesteps_wan,
#                     context=augmented_context,
#                     seq_len=max_seq_len,
#                 )

#             # ── 6. 计算去噪主损失 ──────────────────────────────────────────
#             denoise_loss = 0.0
#             valid_terms = 0
#             drop_prefix_loss = bool(getattr(args, "train_animate_drop_prefix_loss", True))
#             dropped_prefix_terms = 0
#             for i in range(B):
#                 pred = model_output[i].float()
#                 target = target_list[i]
#                 prefix_slots_i = int(prefix_slots_list[i]) if i < len(prefix_slots_list) else 0
#                 if (
#                     effective_video_mode == "wan_animate_slot"
#                     and drop_prefix_loss
#                     and prefix_slots_i > 0
#                 ):
#                     if pred.shape[1] <= prefix_slots_i or target.shape[1] <= prefix_slots_i:
#                         continue
#                     pred = pred[:, prefix_slots_i:, ...]
#                     target = target[:, prefix_slots_i:, ...]
#                     dropped_prefix_terms += 1
#                 loss = F.mse_loss(pred, target)
#                 denoise_loss += loss
#                 valid_terms += 1
#             if valid_terms <= 0:
#                 raise RuntimeError("无有效训练样本参与损失计算")
#             denoise_loss = denoise_loss / valid_terms
#             self._last_train_prefix_loss_dropped = int(dropped_prefix_terms)

#             # 新版训练目标：仅保留原始去噪主损失（ground-truth latent velocity vs predicted velocity）
#             total_loss = denoise_loss
#             self._last_loss_denoise = float(denoise_loss.detach().item())
#             self._last_loss_aux_align_total = 0.0
#             self._last_loss_aux_t5_l2 = 0.0
#             self._last_loss_aux_t5_cos = 0.0
#             self._last_loss_aux_t5_stats = 0.0
#             self._last_loss_aux_t5_gram = 0.0
#             self._last_loss_aux_t5_cka = 0.0
#             self._last_loss_aux_t5_ot = 0.0
#             self._last_loss_aux_image_preserve = 0.0
#             self._last_loss_aux_wan_func = 0.0

#         finally:
#             self.wan.model.text_len = orig_text_len

#         return total_loss

#     def train(self):
#         """主训练循环。"""
#         args = self.args
#         self._audit_runtime_trainability(stage="train_start")

#         # 设置随机种子
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
#         np.random.seed(args.seed)

#         # 数据集（已完全收敛到 WanVideoDataset）
#         if WanDatasetClass is None:
#             raise RuntimeError("未能导入 WanVideoDataset，请检查 train_connector_for_wan.py 及其依赖")

#         dataset = WanDatasetClass(
#             seed=args.seed,
#             frame_num=args.frame_num,
#             max_area=args.max_area,
#             null_caption_prob=args.null_caption_prob,
#             null_image_prob=args.null_image_prob,
#             max_caption_tokens=args.max_caption_tokens,
#             caption_tokenizer_path=args.caption_tokenizer_path,
#             min_duration_sec=args.min_duration_sec,
#             max_duration_sec=args.max_duration_sec,
#             local_openvid_video_root=args.local_openvid_video_root,
#             local_openvid_csv_path=args.local_openvid_csv_path,
#             local_openvid_limit=args.local_openvid_limit,
#             local_openvid_hd_video_root=args.local_openvid_hd_video_root,
#             local_openvid_hd_csv_path=args.local_openvid_hd_csv_path,
#             local_openvid_hd_limit=args.local_openvid_hd_limit,
#             local_video_cache_dir=args.local_video_cache_dir,
#         )

#         if len(dataset) == 0:
#             raise RuntimeError("数据集为空！检查路径和 JSON 文件。")

#         # 由于视频尺寸可能不同, 使用 batch_size=1 避免 collate 问题
#         dataloader = DataLoader(
#             dataset,
#             batch_size=1,
#             shuffle=True,
#             num_workers=args.dataloader_num_workers,
#             pin_memory=True,
#             collate_fn=self._collate_fn,
#         )

#         # 训练循环
#         os.makedirs(args.output_dir, exist_ok=True)
#         output_dir = Path(args.output_dir).expanduser().resolve()
#         if not self._metrics_jsonl_path:
#             self._metrics_jsonl_path = str((output_dir / "logs" / "train_metrics.jsonl").expanduser().resolve())
#         args.output_dir = str(output_dir)
#         args.metrics_jsonl_path = self._metrics_jsonl_path
#         self._train_wall_start = time.perf_counter()

#         # 训练前快照（用于 verify_metaquery_chain before vs after）
#         self._train_before_checkpoint_path = str(output_dir / "checkpoint-before-training")
#         self._save_checkpoint(
#             self._train_before_checkpoint_path,
#             step=0,
#             extra_info={
#                 "is_before_training": True,
#                 "resume_mq_encoder_path": getattr(args, "resume_mq_encoder_path", None),
#                 "note": "trainable params snapshot before optimizer updates",
#             },
#         )
#         if self.is_main_process:
#             print(f"[VERIFY] 已保存训练前快照: {self._train_before_checkpoint_path}")

#         self.mq_encoder.train()
#         step = 0
#         running_loss = 0.0
#         early_stop_triggered = False
#         early_stop_reason = ""
#         early_stop_ckpt_path = ""
#         data_iter = iter(dataloader)

#         pbar = tqdm(total=args.num_train_steps, desc="Training")
#         self.optimizer.zero_grad(set_to_none=True)

#         while step < args.num_train_steps:
#             step_wall_start = time.perf_counter()
#             accum_loss = 0.0
#             accum_denoise_loss = 0.0
#             accum_align_loss = 0.0
#             accum_align_t5_l2 = 0.0
#             accum_align_t5_cos = 0.0
#             accum_align_t5_stats = 0.0
#             accum_align_t5_gram = 0.0
#             accum_align_t5_cka = 0.0
#             accum_align_t5_ot = 0.0
#             accum_align_img = 0.0
#             accum_align_wan_func = 0.0
#             skip_optimizer_step = False
#             had_fatal_cuda_error = False
#             backward_ok = 0
#             skip_reason = ""
#             self._current_train_ref_anchor_mode = self._resolve_train_ref_anchor_mode()

#             for accum_step in range(args.gradient_accumulation_steps):
#                 # 获取 batch
#                 try:
#                     batch = next(data_iter)
#                 except StopIteration:
#                     data_iter = iter(dataloader)
#                     batch = next(data_iter)

#                 try:
#                     loss = self._compute_loss(batch)
#                     loss = loss / args.gradient_accumulation_steps
#                     loss.backward()
#                     self._log_grad_health_once()
#                     accum_loss += loss.item()
#                     scale = 1.0 / max(float(args.gradient_accumulation_steps), 1.0)
#                     accum_denoise_loss += float(self._last_loss_denoise) * scale
#                     accum_align_loss += float(self._last_loss_aux_align_total) * scale
#                     accum_align_t5_l2 += float(self._last_loss_aux_t5_l2) * scale
#                     accum_align_t5_cos += float(self._last_loss_aux_t5_cos) * scale
#                     accum_align_t5_stats += float(self._last_loss_aux_t5_stats) * scale
#                     accum_align_t5_gram += float(self._last_loss_aux_t5_gram) * scale
#                     accum_align_t5_cka += float(self._last_loss_aux_t5_cka) * scale
#                     accum_align_t5_ot += float(self._last_loss_aux_t5_ot) * scale
#                     accum_align_img += float(self._last_loss_aux_image_preserve) * scale
#                     accum_align_wan_func += float(self._last_loss_aux_wan_func) * scale
#                     backward_ok += 1
#                 except Exception as e:
#                     err = str(e)
#                     bad_video = None
#                     try:
#                         bad_video = batch.get("video_path", None)
#                     except Exception:
#                         bad_video = None
#                     print(f"[WARN] step {step} accum_step {accum_step} 训练异常: {err}")
#                     if bad_video is not None:
#                         print(f"[WARN] step {step} accum_step {accum_step} bad_video={bad_video}")
#                     err_l = err.lower()
#                     is_illegal_access = "illegal memory access" in err_l
#                     is_device_assert = "device-side assert" in err_l
#                     if isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in err.lower()):
#                         skip_optimizer_step = True
#                         skip_reason = "oom"
#                         self.optimizer.zero_grad(set_to_none=True)
#                         if torch.cuda.is_available():
#                             torch.cuda.empty_cache()
#                         gc.collect()
#                         break
#                     if is_illegal_access or is_device_assert:
#                         had_fatal_cuda_error = True
#                         skip_optimizer_step = True
#                         skip_reason = "fatal_cuda"
#                         self.optimizer.zero_grad(set_to_none=True)
#                         if torch.cuda.is_available():
#                             torch.cuda.empty_cache()
#                         gc.collect()
#                         break
#                     # 其他异常也跳过本 step，避免残缺梯度进入 optimizer.step
#                     skip_optimizer_step = True
#                     skip_reason = "error"
#                     self.optimizer.zero_grad(set_to_none=True)
#                     break
#                     continue

#             if had_fatal_cuda_error:
#                 raise RuntimeError(
#                     f"Fatal CUDA kernel error at step={step}. "
#                     "检测到 illegal memory access/device-side assert，已中止训练。"
#                 )

#             if backward_ok == 0:
#                 self._skipped_step_count += 1
#                 if skip_reason == "oom":
#                     self._oom_skip_count += 1
#                 elif skip_reason and skip_reason != "fatal_cuda":
#                     self._error_skip_count += 1
#                 continue

#             if skip_optimizer_step:
#                 self._skipped_step_count += 1
#                 if skip_reason == "oom":
#                     self._oom_skip_count += 1
#                 else:
#                     self._error_skip_count += 1
#                 continue

#             # 梯度裁剪
#             grad_norm = torch.nn.utils.clip_grad_norm_(
#                 self._all_trainable_params(),
#                 args.max_grad_norm,
#             )

#             self.optimizer.step()
#             self.scheduler.step()
#             self.optimizer.zero_grad(set_to_none=True)
#             if args.aggressive_empty_cache:
#                 torch.cuda.empty_cache()

#             step += 1
#             step_time = max(time.perf_counter() - step_wall_start, 1e-6)
#             running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss
#             lr = self.scheduler.get_last_lr()[0]
#             grad_norm_value = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
#             effective_samples = int(max(backward_ok, 0) * max(args.batch_size, 1))
#             samples_per_sec = float(effective_samples / step_time)

#             metrics = {
#                 "train/loss_step": float(accum_loss),
#                 "train/loss_ema": float(running_loss),
#                 "train/loss_denoise": float(accum_denoise_loss),
#                 "train/loss_align_total": float(accum_align_loss),
#                 "train/loss_align_t5_l2": float(accum_align_t5_l2),
#                 "train/loss_align_t5_cos": float(accum_align_t5_cos),
#                 "train/loss_align_t5_stats": float(accum_align_t5_stats),
#                 "train/loss_align_t5_gram": float(accum_align_t5_gram),
#                 "train/loss_align_t5_cka": float(accum_align_t5_cka),
#                 "train/loss_align_t5_ot": float(accum_align_t5_ot),
#                 "train/loss_align_img_preserve": float(accum_align_img),
#                 "train/loss_align_wan_func": float(accum_align_wan_func),
#                 "train/lr": float(lr),
#                 "train/grad_norm": float(grad_norm_value),
#                 "train/step": int(step),
#                 "train/step_time_sec": float(step_time),
#                 "train/samples_per_sec": float(samples_per_sec),
#                 "train/backward_ok_microbatches": int(backward_ok),
#                 "train/effective_batch_samples": int(effective_samples),
#                 "train/skipped_step_count": int(self._skipped_step_count),
#                 "train/oom_skip_count": int(self._oom_skip_count),
#                 "train/error_skip_count": int(self._error_skip_count),
#                 "train/ref_anchor_alpha_mean": float(self._last_train_ref_anchor_alpha_mean),
#                 "train/ref_anchor_applied": int(self._last_train_ref_anchor_applied),
#                 "train/ref_anchor_mode_cfg": str(getattr(args, "train_ref_anchor_mode", "none")),
#                 "train/ref_anchor_mode_effective": str(self._last_train_ref_anchor_effective_mode),
#                 "train/ref_anchor_effective_is_animate": int(self._last_train_ref_anchor_effective_mode == "animate_like"),
#                 "train/video_conditioning_mode_cfg": str(getattr(args, "dit_condition_mode", "mq_only")),
#                 "train/video_conditioning_mode_effective": str(self._last_train_video_conditioning_mode),
#                 "train/prefix_latent_slots": int(self._last_train_prefix_latent_slots),
#                 "train/target_latent_slots": int(self._last_train_target_latent_slots),
#                 "train/prefix_loss_dropped": int(self._last_train_prefix_loss_dropped),
#                 "train/mq_rms": float(self._last_mq_rms),
#                 "train/t5_rms_probe": float(self._last_t5_rms),
#                 "train/mq_t5_rms_ratio": float(self._last_mq_t5_rms_ratio),
#                 "train/mq_norm_warn": int(self._last_mq_norm_warn_flag),
#                 "train/mq_norm_match_scale": float(self._last_mq_norm_match_scale),
#             }
#             metrics.update(self._collect_trainability_metrics())
#             metrics.update(self._collect_cuda_memory_metrics())

#             should_log = bool(args.log_every_step or (step % args.log_steps == 0))
#             should_wandb_log = bool(
#                 self.wandb_run is not None and (args.wandb_log_every_step or should_log)
#             )

#             # 日志
#             if should_log:
#                 pbar.set_postfix({
#                     "loss": f"{accum_loss:.4f}",
#                     "denoise": f"{accum_denoise_loss:.4f}",
#                     "align": f"{accum_align_loss:.4f}",
#                     "func": f"{accum_align_wan_func:.4f}",
#                     "avg": f"{running_loss:.4f}",
#                     "lr": f"{lr:.2e}",
#                     "grad": f"{grad_norm_value:.2f}",
#                     "dP": f"{metrics['train/param_sample_abs_delta_mean']:.3e}",
#                 })
#                 print(
#                     f"\n[Step {step}/{args.num_train_steps}] "
#                     f"loss={accum_loss:.4f} denoise={accum_denoise_loss:.4f} align={accum_align_loss:.4f} func={accum_align_wan_func:.4f} "
#                     f"avg={running_loss:.4f} "
#                     f"lr={lr:.2e} grad_norm={grad_norm_value:.2f} "
#                     f"dt={step_time:.2f}s samp/s={samples_per_sec:.2f} "
#                     f"param_delta={metrics['train/param_sample_abs_delta_mean']:.3e} "
#                     f"skip(oom/err/total)={self._oom_skip_count}/{self._error_skip_count}/{self._skipped_step_count}"
#                 )
#             if should_wandb_log:
#                 self.wandb.log(metrics, step=step)
#             self._append_metrics_jsonl(metrics)
#             self._record_metrics(metrics)

#             # 保存
#             if step % args.save_steps == 0:
#                 self._save_checkpoint(output_dir / f"checkpoint-{step}", step)

#             pbar.update(1)

#             step_loss_for_early_stop = float(accum_denoise_loss)
#             if (
#                 bool(getattr(args, "enable_loss_early_stop", False))
#                 and step >= int(getattr(args, "loss_early_stop_min_step", 800))
#                 and step_loss_for_early_stop < float(getattr(args, "loss_early_stop_threshold", 0.25))
#             ):
#                 early_stop_triggered = True
#                 early_stop_reason = (
#                     f"train/loss_denoise={step_loss_for_early_stop:.6f} < {float(args.loss_early_stop_threshold):.6f} "
#                     f"at step={int(step)}"
#                 )
#                 early_stop_ckpt_path = str(
#                     output_dir / f"checkpoint-earlystop-step{int(step)}-denoise{step_loss_for_early_stop:.4f}"
#                 )
#                 self._save_checkpoint(
#                     early_stop_ckpt_path,
#                     step,
#                     extra_info={
#                         "early_stop": True,
#                         "early_stop_metric": "train/loss_denoise",
#                         "early_stop_loss": step_loss_for_early_stop,
#                         "early_stop_threshold": float(args.loss_early_stop_threshold),
#                         "early_stop_min_step": int(args.loss_early_stop_min_step),
#                     },
#                 )
#                 if self.is_main_process:
#                     print(f"[EARLY-STOP] 已触发: {early_stop_reason}")
#                     print(f"[EARLY-STOP] checkpoint: {early_stop_ckpt_path}")
#                 break

#         pbar.close()

#         # 最终保存
#         final_ckpt_path = str(output_dir / "checkpoint-final")
#         final_extra_info = None
#         if early_stop_triggered:
#             final_extra_info = {
#                 "early_stop": True,
#                 "early_stop_reason": early_stop_reason,
#                 "early_stop_checkpoint_path": early_stop_ckpt_path,
#             }
#         self._save_checkpoint(final_ckpt_path, step, extra_info=final_extra_info)
#         self._write_training_chain_manifest(output_dir, final_checkpoint_path=final_ckpt_path, final_step=step)
#         if early_stop_triggered and self.is_main_process:
#             print(f"[EARLY-STOP] 训练提前结束，最终步数: {step}")
#         print(f"\n✅ 训练完成！最终 checkpoint: {final_ckpt_path}")
#         if self.wandb_run is not None:
#             self.wandb.finish()

#     def _save_checkpoint(self, path, step, extra_info: Dict[str, Any] | None = None):
#         """保存 MQ 编码器 +（可选）Wan DiT 可训练子集（兼容增强格式）"""
#         path = Path(path).expanduser().resolve()
#         module = self._mq_encoder_module()
#         wan_state_cpu, wan_lora_state_cpu, wan_export_info = self._collect_wan_trainable_state_for_checkpoint()
#         if not self.is_main_process:
#             return
#         ckpt_info = save_mq_checkpoint_bundle(
#             path=path,
#             module=module,
#             optimizer=self.optimizer,
#             scheduler=self.scheduler,
#             step=step,
#             args=self.args,
#             wan_module=None,
#             wan_trainable_state_cpu=wan_state_cpu,
#             wan_lora_state_cpu=wan_lora_state_cpu,
#             wan_lora_config=build_lora_config_dict(
#                 enabled=self._wan_lora_enabled(),
#                 rank=int(getattr(self.args, "wan_lora_rank", 16)),
#                 alpha=float(getattr(self.args, "wan_lora_alpha", 16.0)),
#                 dropout=float(getattr(self.args, "wan_lora_dropout", 0.0)),
#                 targets=getattr(self.args, "wan_lora_targets", "self_attn,cross_attn,ffn"),
#                 module_names=self._wan_lora_module_names,
#             ) if self._wan_lora_enabled() else None,
#             wan_train_mode=str(getattr(self, "_effective_wan_train_mode", "frozen")),
#             metrics_tail=self._metrics_history[-200:],
#             metrics_summary=self._build_metrics_summary(step=step),
#             extra_info={
#                 "before_checkpoint_path": self._train_before_checkpoint_path,
#                 "metrics_jsonl_path": self._metrics_jsonl_path,
#                 "wan_train_mode_effective": str(getattr(self, "_effective_wan_train_mode", "frozen")),
#                 "wan_trainable_tensor_count": int(len(getattr(self, "_wan_trainable_names", []))),
#                 "wan_trainable_name_preview": list(getattr(self, "_wan_trainable_names", [])[:64]),
#                 "wan_lora_module_count": int(len(getattr(self, "_wan_lora_module_names", []))),
#                 "wan_lora_module_preview": list(getattr(self, "_wan_lora_module_names", [])[:64]),
#                 "wan_lora_extra_trainable_count": int(len(getattr(self, "_wan_lora_extra_trainable_names", []))),
#                 "wan_lora_extra_trainable_preview": list(getattr(self, "_wan_lora_extra_trainable_names", [])[:64]),
#                 **wan_export_info,
#                 **(extra_info or {}),
#             },
#         )
#         print(f"  💾 Checkpoint 已保存: {ckpt_info['path']}")
#         if self.wandb_run is not None and self.args.wandb_log_checkpoint:
#             self.wandb.log(
#                 {
#                     "checkpoint/step": int(step),
#                     "checkpoint/path": str(ckpt_info["path"]),
#                 },
#                 step=step,
#             )

#     @staticmethod
#     def _normalize_wan_state_key(name: str) -> str:
#         key = str(name)
#         while "_fsdp_wrapped_module." in key:
#             key = key.replace("_fsdp_wrapped_module.", "")
#         return key

#     def _collect_wan_trainable_state_for_checkpoint(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
#         """
#         导出可用于推理端加载的 Wan 可训练权重子集。

#         关键点：
#         - 非 FSDP: 直接按 requires_grad 导出。
#         - FSDP: 使用 FULL_STATE_DICT（所有 rank 参与）导出原始参数名，避免只保存 _flat_param。
#         """
#         wan_model = getattr(self.wan, "model", None)
#         mode = str(getattr(self, "_effective_wan_train_mode", "frozen")).strip().lower()
#         info: Dict[str, Any] = {
#             "wan_export_mode": "none",
#             "wan_export_model_mode": mode,
#             "wan_export_world_size": int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1,
#         }
#         if wan_model is None or not isinstance(wan_model, nn.Module) or mode == "frozen":
#             return {}, {}, info

#         if mode == "lora":
#             direct_state_cpu: Dict[str, torch.Tensor] = {}
#             for name, p in wan_model.named_parameters():
#                 if not p.requires_grad:
#                     continue
#                 norm_name = self._normalize_wan_state_key(name)
#                 lname = norm_name.lower()
#                 if ".lora_a" in lname or ".lora_b" in lname:
#                     continue
#                 direct_state_cpu[norm_name] = p.detach().cpu().contiguous()
#             lora_state_cpu = collect_lora_state_dict(wan_model)
#             info["wan_export_mode"] = "lora_plus_named_parameters"
#             info["wan_export_tensor_count"] = int(len(direct_state_cpu))
#             info["wan_export_param_count"] = int(sum(int(t.numel()) for t in direct_state_cpu.values()))
#             info["wan_export_lora_tensor_count"] = int(len(lora_state_cpu))
#             info["wan_export_lora_param_count"] = int(sum(int(t.numel()) for t in lora_state_cpu.values()))
#             info["wan_export_has_flat_param_key"] = 0
#             return direct_state_cpu, lora_state_cpu, info

#         fsdp_cls = None
#         fsdp_full_cfg = None
#         fsdp_state_type = None
#         try:
#             from torch.distributed.fsdp import FullyShardedDataParallel as _TorchFSDP
#             from torch.distributed.fsdp import FullStateDictConfig as _FullStateDictConfig
#             from torch.distributed.fsdp import StateDictType as _StateDictType

#             fsdp_cls = _TorchFSDP
#             fsdp_full_cfg = _FullStateDictConfig
#             fsdp_state_type = _StateDictType
#         except Exception:
#             fsdp_cls = None

#         has_fsdp = bool(fsdp_cls is not None and any(isinstance(m, fsdp_cls) for m in wan_model.modules()))
#         dist_ready = bool(dist.is_available() and dist.is_initialized())

#         if has_fsdp and dist_ready and fsdp_cls is not None and fsdp_full_cfg is not None and fsdp_state_type is not None:
#             if self.is_main_process:
#                 print("[WAN-SAVE] FSDP detected, exporting portable FULL_STATE_DICT ...")
#             cfg = fsdp_full_cfg(offload_to_cpu=True, rank0_only=True)
#             try:
#                 with fsdp_cls.state_dict_type(wan_model, fsdp_state_type.FULL_STATE_DICT, cfg):
#                     full_state = wan_model.state_dict()
#             except Exception as e:
#                 info["wan_export_mode"] = "fsdp_full_state_failed_fallback_named_parameters"
#                 info["wan_export_error"] = str(e)
#             else:
#                 if not self.is_main_process:
#                     info["wan_export_mode"] = "fsdp_full_state_non_main_rank"
#                     return {}, {}, info

#                 state_cpu: Dict[str, torch.Tensor] = {}
#                 for name, tensor in full_state.items():
#                     if not torch.is_tensor(tensor):
#                         continue
#                     norm_name = self._normalize_wan_state_key(name)
#                     state_cpu[norm_name] = tensor.detach().cpu().contiguous()

#                 if mode == "cond_only":
#                     kws = self._wan_cond_keywords()
#                     state_cpu = {
#                         n: t for n, t in state_cpu.items()
#                         if any(kw in n.lower() for kw in kws)
#                     }

#                 info["wan_export_mode"] = "fsdp_full_state"
#                 info["wan_export_tensor_count"] = int(len(state_cpu))
#                 info["wan_export_param_count"] = int(sum(int(t.numel()) for t in state_cpu.values()))
#                 info["wan_export_has_flat_param_key"] = int(any("_flat_param" in n for n in state_cpu.keys()))
#                 return state_cpu, {}, info

#         # 非 FSDP 或 FSDP full_state 导出失败时的兜底路径
#         state_cpu = {}
#         for name, p in wan_model.named_parameters():
#             if not p.requires_grad:
#                 continue
#             norm_name = self._normalize_wan_state_key(name)
#             state_cpu[norm_name] = p.detach().cpu().contiguous()

#         info["wan_export_mode"] = "named_parameters"
#         info["wan_export_tensor_count"] = int(len(state_cpu))
#         info["wan_export_param_count"] = int(sum(int(t.numel()) for t in state_cpu.values()))
#         info["wan_export_has_flat_param_key"] = int(any("_flat_param" in n for n in state_cpu.keys()))
#         return state_cpu, {}, info

#     @staticmethod
#     def _collate_fn(batch):
#         """自定义 collate: 不 stack 不同尺寸的 tensor"""
#         result = {}
#         for key in batch[0].keys():
#             result[key] = [item[key] for item in batch]
#         return result


# # =============================================================================
# # Main
# # =============================================================================
# if __name__ == "__main__":
#     args = parse_args()
#     trainer = MetaQueryWanTrainer(args)
#     trainer.train()























# 下面这个是MQ RMS适配 t5 token的情况
"""
train_metaquery_wan.py
=======================
MetaQuery + Wan2.2 TI2V (Text+Image → Video) 联合训练脚本。

★ 核心思路:
    复刻原始 MetaQuery 训练范式 —— 冻结 DiT，训练 Connector：
    1. Qwen3-VL (冻结, 仅 MQ embeddings 可训练)
    2. Connector: Qwen2Encoder(24L) + Linear + GELU + Linear + RMSNorm → dim=4096 (直接对齐 Wan)
    3. to_wan_proj: 不再需要! Connector 直接输出 Wan text_dim=4096
    4. Wan TI2V DiT (冻结): 接收 [MQ_tokens + T5_tokens] 作为 context
    5. 计算 Flow Matching Loss → 反向传播更新 Connector + MQ Embeddings

★ 为什么选 WanTI2V (而非 I2V 或 Animate):
    - TI2V 5B 是 Wan2.2 最新的 Text+Image→Video 统一模型
    - 使用相同 DiT architecture 处理 t2v 和 i2v (model_type='ti2v')
    - 不需要 CLIP encoder (I2V 需要 CLIP, Animate 需要 CLIP+Face+Pose)
    - 参考图通过 VAE 编码后的 latent mask 注入 (最优雅的方式)
    - 5B 参数量适中, 显存友好

★ 不需要 to_wan_proj:
    直接让 Connector 输出 dim=4096 (Wan text_dim)
    → 训练时 DiT 的 text_embedding 层直接消费 MQ 特征
    → 无中间随机投影层, 梯度直接流过

用法:
    # 单卡
    python train_metaquery_wan.py --wan_checkpoint_dir /path/to/Wan2.2-TI2V-5B

    # 多卡
    torchrun --nproc_per_node=2 train_metaquery_wan.py
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
from typing import Dict, Tuple, Any, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from wan_lora_utils import (
    apply_lora_to_wan_model,
    build_lora_config_dict,
    collect_lora_state_dict,
)

# ── 路径设置 ─────────────────────────────────────────────────────────────────
WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))

METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
sys.path.insert(0, METAQUERY_ROOT)


# =============================================================================
# 配置
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train MetaQuery Connector for Wan TI2V")

    # ── 模型路径 ──────────────────────────────────────────────────────────
    p.add_argument("--wan_checkpoint_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
                   help="Wan2.2 TI2V checkpoint 目录")
    p.add_argument("--qwen3vl_model_id", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking",
                   help="Qwen3-VL 模型 ID 或本地路径")
    p.add_argument("--output_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/metaquery_wan_ti2v_training",
                   help="训练输出目录")

    # ── 数据(OpenVid/WanVideoDataset) ───────────────────────────────────────
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
    p.add_argument("--frame_num", type=int, default=41,
                   help="每个视频片段采样帧数 (4n+1)")
    p.add_argument("--max_area", type=int, default=480 * 832,
                   help="视频最大面积 (宽×高)")
    p.add_argument("--max_caption_tokens", type=int, default=512,
                   help="超过该token长度的caption会被过滤")
    p.add_argument("--caption_tokenizer_path", type=str, default="google/umt5-xxl",
                   help="用于caption长度统计的tokenizer")
    p.add_argument("--min_duration_sec", type=float, default=0.5,
                   help="最短时长过滤阈值")
    p.add_argument("--max_duration_sec", type=float, default=20.0,
                   help="最长时长过滤阈值")

    # ── 训练参数 ──────────────────────────────────────────────────────────
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine_with_warmup",
        choices=["cosine_with_warmup", "constant_with_warmup", "warmup_hold_cooldown"],
        help=(
            "学习率调度器类型。"
            "constant_with_warmup=warmup后恒定；"
            "cosine_with_warmup=warmup后余弦衰减；"
            "warmup_hold_cooldown=warmup线性升+中段恒定+末段线性降。"
        ),
    )
    p.add_argument(
        "--cooldown_steps",
        type=int,
        default=-1,
        help="warmup_hold_cooldown 模式下末段降学习率步数。<0 表示使用 warmup_steps。",
    )
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.01,
        help="cosine_with_warmup 模式下的最小学习率比例。",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--enable_loss_early_stop", action="store_true", default=False,
                   help="启用可选早停：当 step>=loss_early_stop_min_step 且 loss_step<loss_early_stop_threshold 时提前结束训练并保存 checkpoint。")
    p.add_argument("--disable_loss_early_stop", action="store_false", dest="enable_loss_early_stop",
                   help="关闭 loss 早停（默认）。")
    p.add_argument("--loss_early_stop_min_step", type=int, default=800,
                   help="loss 早停触发的最小 step（含）。")
    p.add_argument("--loss_early_stop_threshold", type=float, default=0.25,
                   help="loss 早停阈值：当 train/loss_step 小于该值时触发。")
    p.add_argument("--log_every_step", action="store_true",
                   help="每个优化 step 都打印详细训练日志")
    p.add_argument("--wandb_log_every_step", action="store_true",
                   help="每个优化 step 都写入 W&B（默认按 log_steps 写入）")
    p.add_argument("--metrics_jsonl_path", type=str, default="",
                   help="可选：将每步指标追加写入 JSONL 文件")
    p.add_argument("--log_cuda_memory", action="store_true",
                   help="记录并输出 CUDA 显存指标")
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    # ── MetaQuery ─────────────────────────────────────────────────────────
    p.add_argument("--num_metaqueries", type=int, default=256)
    p.add_argument("--connector_num_hidden_layers", type=int, default=24)
    p.add_argument(
        "--dit_condition_mode",
        type=str,
        default="mq_only",
        choices=["mq_only"],
        help="DiT 显式条件注入模式。当前仅支持 mq_only（仅注入 MetaQuery tokens）。",
    )
    p.add_argument("--mq_gradient_checkpointing", action="store_true",
                   help="启用 MetaQuery 编码器梯度检查点，降低显存占用")
    p.add_argument("--train_mq_input_embeddings", action="store_true", default=True,
                   help="训练 Qwen 输入 embedding（默认开启）")
    p.add_argument("--freeze_mq_input_embeddings", action="store_false", dest="train_mq_input_embeddings",
                   help="冻结 Qwen 输入 embedding，仅训练 connector")
    p.add_argument("--mq_connector_norm_init_scale", type=float, default=1.0,
                   help="MQ connector 末层 RMSNorm 初始权重尺度。建议 1.0；过大可能导致 MQ/T5 范数失配。")
    p.add_argument("--null_caption_prob", type=float, default=0.1)
    p.add_argument("--null_image_prob", type=float, default=0.1)
    p.add_argument("--enable_t5_alignment", action="store_true", default=True,
                   help="启用 T5 对齐辅助损失（默认开启）：让 MQ 条件分布更接近 Wan 已适配的 T5 条件流形。")
    p.add_argument("--disable_t5_alignment", action="store_false", dest="enable_t5_alignment",
                   help="关闭 T5 对齐辅助损失，仅使用去噪主损失。")
    p.add_argument(
        "--t5_align_mode",
        type=str,
        default="gram_cka",
        choices=["anchor", "gram_cka", "sinkhorn_ot"],
        help=(
            "T5 对齐方式。anchor=前K token 一一对齐；"
            "gram_cka=基于 token 关系矩阵(Gram+CKA)的排列无关对齐；"
            "sinkhorn_ot=基于 OT/Sinkhorn 的软匹配对齐。"
        ),
    )
    p.add_argument("--t5_align_anchor_tokens", type=int, default=64,
                   help="用于 T5 对齐的 anchor token 数（从 256 个 MQ token 前缀取）。")
    p.add_argument("--lambda_t5_align_l2", type=float, default=0.2,
                   help="T5 对齐主项权重：anchor 模式对应 token-L2；gram_cka 模式对应 Gram-L2；sinkhorn_ot 模式对应 OT 代价。")
    p.add_argument("--lambda_t5_align_cos", type=float, default=0.1,
                   help="T5 对齐次项权重：anchor 模式对应 token-cos；gram_cka 模式对应 CKA；sinkhorn_ot 模式默认忽略。")
    p.add_argument("--lambda_t5_align_stats", type=float, default=0.02,
                   help="T5 对齐的均值/方差统计损失权重。")
    p.add_argument("--t5_align_ot_epsilon", type=float, default=0.05,
                   help="Sinkhorn OT 熵正则温度 epsilon（越小越接近硬匹配）。")
    p.add_argument("--t5_align_ot_iters", type=int, default=25,
                   help="Sinkhorn OT 迭代次数。")
    p.add_argument("--enable_mq_image_preserve", action="store_true", default=False,
                   help="启用图像保持约束：有参考图时，MQ(cond) 与 MQ(text-only) 保持最小间隔。")
    p.add_argument("--lambda_mq_image_preserve", type=float, default=0.02,
                   help="图像保持约束权重。")
    p.add_argument("--mq_image_preserve_margin", type=float, default=0.10,
                   help="图像保持约束的最小间隔阈值（L2 均方根距离）。")
    p.add_argument("--mq_norm_probe_with_t5", action="store_true", default=True,
                   help="训练时记录 MQ 与 T5 token RMS 范数比值（用于定位 MQ 条件被忽略问题）。")
    p.add_argument("--disable_mq_norm_probe_with_t5", action="store_false", dest="mq_norm_probe_with_t5",
                   help="关闭 MQ/T5 范数探针。")
    p.add_argument("--mq_norm_probe_every_n_steps", type=int, default=20,
                   help="每 N 次 _compute_loss 调用做一次 MQ/T5 范数探针。")
    p.add_argument("--mq_norm_warn_ratio_low", type=float, default=0.25,
                   help="当 MQ/T5 RMS 比值低于该阈值时打印警告。")
    p.add_argument("--mq_norm_warn_ratio_high", type=float, default=4.0,
                   help="当 MQ/T5 RMS 比值高于该阈值时打印警告。")
    p.add_argument("--mq_norm_match_t5", action="store_true", default=True,
                   help="将 MQ 特征按 token RMS 对齐到 T5 RMS（默认开启，降低 MQ/T5 范数失配风险）。")
    p.add_argument("--disable_mq_norm_match_t5", action="store_false", dest="mq_norm_match_t5",
                   help="关闭 MQ/T5 RMS 自动对齐。")
    p.add_argument("--mq_norm_match_clip_min", type=float, default=0.03,
                   help="mq_norm_match_t5 时缩放因子下限。")
    p.add_argument("--mq_norm_match_clip_max", type=float, default=4.0,
                   help="mq_norm_match_t5 时缩放因子上限。")
    p.add_argument("--enable_wan_func_distill", action="store_true", default=False,
                   help="启用 Wan 函数级蒸馏：约束 pred_mq(x_t,t) 贴近 pred_t5(x_t,t)。")
    p.add_argument("--disable_wan_func_distill", action="store_false", dest="enable_wan_func_distill",
                   help="关闭 Wan 函数级蒸馏。")
    p.add_argument("--lambda_wan_func_distill", type=float, default=0.0,
                   help="Wan 函数级蒸馏损失权重。")
    p.add_argument(
        "--wan_func_teacher_mode",
        type=str,
        default="t5_only",
        choices=["t5_only", "t5_plus_mq"],
        help="函数级蒸馏 teacher 条件。t5_only=仅 T5；t5_plus_mq=T5 与 MQ 拼接。",
    )
    p.add_argument("--enable_ti2v_first_frame_condition", action="store_true", default=True,
                   help="启用 Wan 训练侧首帧参考条件（与 MQ 图像条件并行）。")
    p.add_argument("--disable_ti2v_first_frame_condition", action="store_false",
                   dest="enable_ti2v_first_frame_condition",
                   help="关闭 Wan 训练侧首帧参考条件，仅保留 MQ 条件。")
    p.add_argument("--train_video_conditioning_mode", type=str, default="legacy_t2v",
                   choices=["legacy_t2v", "wan_animate_slot"],
                   help=(
                       "训练期视频条件注入方式: "
                       "legacy_t2v=现有 TI2V 训练（可选首帧软锚定）；"
                       "wan_animate_slot=参考图作为 preserved reference slot 注入，前缀 slot 不计入主损失"
                   ))
    p.add_argument("--train_animate_ref_frames", type=int, default=1,
                   help="wan_animate_slot 模式下参考图保留帧数（像素帧数，内部按 VAE stride 映射到 latent slots）")
    p.add_argument("--train_animate_temporal_frames", type=int, default=0,
                   help="wan_animate_slot 模式下 temporal guidance 帧数（像素帧数；若无外部时序条件可保持 0）")
    p.add_argument("--train_animate_conditional_frames", type=int, default=0,
                   help="wan_animate_slot 模式下额外 conditional 帧数（像素帧数；无条件时保持 0，将注入全零 latent）")
    p.add_argument("--train_animate_preserve_timestep_zero", action="store_true", default=True,
                   help="wan_animate_slot: preserved prefix 对应 token 的 timestep 置 0（默认开启）")
    p.add_argument("--train_animate_no_preserve_timestep_zero", action="store_false",
                   dest="train_animate_preserve_timestep_zero",
                   help="wan_animate_slot: 关闭 preserved prefix timestep=0")
    p.add_argument("--train_animate_drop_prefix_loss", action="store_true", default=True,
                   help="wan_animate_slot: 仅在 target frames 上计算损失，丢弃 reference/temporal/conditional prefix（默认开启）")
    p.add_argument("--train_animate_no_drop_prefix_loss", action="store_false",
                   dest="train_animate_drop_prefix_loss",
                   help="wan_animate_slot: 不丢弃 prefix，整段都计入损失")
    p.add_argument("--train_ref_anchor_mode", type=str, default="none",
                   choices=["none", "animate_like", "mixed50"],
                   help="训练时是否对 x_t 的首帧加入软参考锚定。none=保持原始 t2v；animate_like=全程启用软锚定；mixed50=约50%批次启用软锚定")
    p.add_argument("--train_ref_anchor_alpha0", type=float, default=0.95,
                   help="animate_like 模式的最大锚定强度 alpha0")
    p.add_argument("--train_ref_anchor_warmup_ratio", type=float, default=0.35,
                   help="animate_like 模式在高噪声区间启用锚定的占比（0~1）")

    # ── 设备 ──────────────────────────────────────────────────────────────
    p.add_argument("--dit_device", type=int, default=0,
                   help="DiT + VAE + T5 所在 GPU")
    p.add_argument("--encoder_device", type=int, default=1,
                   help="Qwen3-VL + Connector 所在 GPU")
    p.add_argument("--resume_mq_encoder_path", type=str, default=None,
                   help="从已有mq_encoder权重继续训练")
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
    p.add_argument("--strict_freeze_check", action="store_true", default=True,
                   help="启用严格冻结校验：若发现 Wan/T5/VAE 可训练或 optimizer 混入非 MQ 参数则中止")
    p.add_argument("--no_strict_freeze_check", action="store_false", dest="strict_freeze_check",
                   help="关闭严格冻结校验，仅打印告警")
    p.add_argument(
        "--wan_train_mode",
        type=str,
        default="auto",
        choices=["auto", "frozen", "full", "cond_only", "lora"],
        help=(
            "Wan DiT 训练模式。auto=按显存策略自动在 full/cond_only 之间选择；"
            "frozen=冻结；full=全量训练；cond_only=仅训 cross-attn + conditioning projection/AdaLN/modulation；"
            "lora=LoRA 微调（可叠加额外小模块直训）。"
        ),
    )
    p.add_argument(
        "--wan_auto_full_mem_gb",
        type=float,
        default=120.0,
        help="auto 模式下，当 DiT 卡总显存 >= 该阈值时选择 full，否则选择 cond_only。",
    )
    p.add_argument(
        "--wan_lr_ratio",
        type=float,
        default=1.0,
        help="Wan 可训练参数学习率倍率（实际 lr = learning_rate * wan_lr_ratio）。",
    )
    p.add_argument(
        "--wan_cond_name_pattern",
        type=str,
        default="",
        help=(
            "可选：自定义 cond_only 的参数名匹配关键字，逗号分隔。"
            "为空时使用内置规则(cross_attn,text_embedding,time_projection,modulation,norm3,cross_attn_norm)。"
        ),
    )
    p.add_argument("--enable_wan_lora", action="store_true", default=False,
                   help="启用 Wan DiT LoRA 微调（当前用于单进程/非FSDP 路径）")
    p.add_argument("--disable_wan_lora", action="store_false", dest="enable_wan_lora",
                   help="禁用 Wan DiT LoRA 微调")
    p.add_argument("--wan_lora_rank", type=int, default=16,
                   help="Wan LoRA rank")
    p.add_argument("--wan_lora_alpha", type=float, default=16.0,
                   help="Wan LoRA alpha")
    p.add_argument("--wan_lora_dropout", type=float, default=0.0,
                   help="Wan LoRA dropout")
    p.add_argument("--wan_lora_targets", type=str, default="self_attn,cross_attn,ffn",
                   help="Wan LoRA 目标模块类别，逗号分隔，可选: self_attn,cross_attn,ffn")
    p.add_argument("--wan_lora_extra_name_pattern", type=str, default="",
                   help="LoRA 模式下额外直训的小模块关键词，逗号分隔。例如 norm1,norm2,norm3,time_projection,modulation")

    return p.parse_args()


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """兼容不同 torch 版本的安全加载。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_model_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """从不同 checkpoint 负载中提取模型权重字典。"""
    if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
        non_tensor_values = [v for v in payload.values() if not torch.is_tensor(v)]
        if tensor_values and not non_tensor_values:
            return payload
    raise ValueError("无法从 checkpoint 提取 model_state_dict")


def _to_cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().contiguous()
    return out


def load_mq_encoder_state(path_or_dir: str, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, torch.Tensor], str]:
    """
    加载 MetaQuery encoder 权重:
    - 支持传入单个文件: mq_encoder_full.pt / training_state.pt / model.safetensors
    - 支持传入目录: 自动按优先级查找文件
    """
    path = Path(path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 路径不存在: {path}")

    if path.is_dir():
        candidates = [
            path / "mq_encoder_full.pt",
            path / "mq_encoder_full.safetensors",
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "training_state.pt",
        ]
        picked = next((p for p in candidates if p.exists()), None)
        if picked is None:
            raise FileNotFoundError(
                f"checkpoint 目录中未找到可加载权重文件: {path} "
                f"(expect one of {[c.name for c in candidates]})"
            )
        path = picked

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                f"检测到 safetensors 权重但未能导入 safetensors: {path}"
            ) from e
        state = load_file(str(path), device="cpu")
    else:
        payload = _safe_torch_load(path, map_location=map_location)
        state = _extract_model_state_dict(payload)

    return state, str(path.expanduser().resolve())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_mq_checkpoint_bundle(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    args: argparse.Namespace,
    wan_module: nn.Module | None = None,
    wan_trainable_state_cpu: Dict[str, torch.Tensor] | None = None,
    wan_lora_state_cpu: Dict[str, torch.Tensor] | None = None,
    wan_lora_config: Dict[str, Any] | None = None,
    wan_train_mode: str = "frozen",
    metrics_tail: List[Dict[str, Any]] | None = None,
    metrics_summary: Dict[str, Any] | None = None,
    extra_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    保存“最小可用 + 兼容增强”的 checkpoint bundle。
    兼容你当前推理脚本（mq_encoder_full.pt）并补充常见训练文件。
    """
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    full_state_cpu = _to_cpu_state_dict(module.state_dict())
    name_to_param = dict(module.named_parameters())
    trainable_state_cpu = {
        name: tensor
        for name, tensor in full_state_cpu.items()
        if name_to_param.get(name, None) is not None
        and name_to_param[name].requires_grad
    }

    torch.save(
        {
            "step": step,
            "model_state_dict": trainable_state_cpu,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path / "training_state.pt",
    )
    torch.save(full_state_cpu, path / "mq_encoder_full.pt")
    torch.save(trainable_state_cpu, path / "mq_encoder_trainable.pt")

    if wan_trainable_state_cpu is None:
        wan_trainable_state_cpu = {}
        if wan_module is not None and isinstance(wan_module, nn.Module):
            for name, p in wan_module.named_parameters():
                if not p.requires_grad:
                    continue
                wan_trainable_state_cpu[name] = p.detach().cpu().contiguous()
    else:
        wan_trainable_state_cpu = {
            str(name): tensor.detach().cpu().contiguous()
            for name, tensor in wan_trainable_state_cpu.items()
            if torch.is_tensor(tensor)
        }
    if wan_lora_state_cpu is None:
        wan_lora_state_cpu = {}
    else:
        wan_lora_state_cpu = {
            str(name): tensor.detach().cpu().contiguous()
            for name, tensor in wan_lora_state_cpu.items()
            if torch.is_tensor(tensor)
        }
    wan_trainable_param_count = sum(int(t.numel()) for t in wan_trainable_state_cpu.values())
    wan_lora_param_count = sum(int(t.numel()) for t in wan_lora_state_cpu.values())
    if wan_trainable_state_cpu:
        torch.save(wan_trainable_state_cpu, path / "wan_dit_trainable.pt")
    if wan_lora_state_cpu:
        torch.save(wan_lora_state_cpu, path / "wan_dit_lora.pt")

    torch.save(vars(args), path / "training_args.bin")
    _write_json(
        path / "training_args.json",
        {str(k): _to_jsonable(v) for k, v in vars(args).items()},
    )
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")

    trainer_state = {
        "global_step": int(step),
        "checkpoint_format": "wan_metaquery_v2",
        "has_full_pt": True,
        "has_training_state": True,
        "has_trainable_pt": True,
        "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
        "has_wan_dit_lora_pt": bool(len(wan_lora_state_cpu) > 0),
        "wan_train_mode": str(wan_train_mode),
        "wan_trainable_param_count": int(wan_trainable_param_count),
        "wan_lora_param_count": int(wan_lora_param_count),
        "has_metrics_summary": bool(metrics_summary),
        "metrics_tail_count": int(len(metrics_tail) if metrics_tail is not None else 0),
    }
    if extra_info:
        trainer_state["extra_info"] = _to_jsonable(extra_info)
    _write_json(path / "trainer_state.json", trainer_state)

    config_payload = {
        "format": "wan_metaquery_encoder",
        "num_metaqueries": int(getattr(args, "num_metaqueries", 256)),
        "connector_num_hidden_layers": int(getattr(args, "connector_num_hidden_layers", 24)),
        "wan_text_dim": int(getattr(module, "wan_text_dim", 4096)),
        "qwen3vl_model_id": str(getattr(args, "qwen3vl_model_id", "")),
        "train_mq_input_embeddings": bool(getattr(args, "train_mq_input_embeddings", True)),
        "mq_connector_norm_init_scale": float(getattr(args, "mq_connector_norm_init_scale", 1.0)),
        "wan_train_mode": str(wan_train_mode),
        "wan_trainable_param_count": int(wan_trainable_param_count),
        "wan_lora_param_count": int(wan_lora_param_count),
        "has_wan_dit_trainable_pt": bool(len(wan_trainable_state_cpu) > 0),
        "has_wan_dit_lora_pt": bool(len(wan_lora_state_cpu) > 0),
        "checkpoint_step": int(step),
        "num_train_steps": int(getattr(args, "num_train_steps", 0)),
        "save_steps": int(getattr(args, "save_steps", 0)),
        "log_steps": int(getattr(args, "log_steps", 0)),
        "enable_loss_early_stop": bool(getattr(args, "enable_loss_early_stop", False)),
        "loss_early_stop_min_step": int(getattr(args, "loss_early_stop_min_step", 800)),
        "loss_early_stop_threshold": float(getattr(args, "loss_early_stop_threshold", 0.25)),
        "frame_num": int(getattr(args, "frame_num", 0)),
        "max_area": int(getattr(args, "max_area", 0)),
        "learning_rate": float(getattr(args, "learning_rate", 0.0)),
        "warmup_steps": int(getattr(args, "warmup_steps", 0)),
        "lr_scheduler_type": str(getattr(args, "lr_scheduler_type", "cosine_with_warmup")),
        "cooldown_steps": int(getattr(args, "cooldown_steps", -1)),
        "lr_min_ratio": float(getattr(args, "lr_min_ratio", 0.01)),
        "enable_t5_alignment": bool(getattr(args, "enable_t5_alignment", True)),
        "t5_align_mode": str(getattr(args, "t5_align_mode", "gram_cka")),
        "t5_align_anchor_tokens": int(getattr(args, "t5_align_anchor_tokens", 64)),
        "lambda_t5_align_l2": float(getattr(args, "lambda_t5_align_l2", 0.0)),
        "lambda_t5_align_cos": float(getattr(args, "lambda_t5_align_cos", 0.0)),
        "lambda_t5_align_stats": float(getattr(args, "lambda_t5_align_stats", 0.0)),
        "t5_align_ot_epsilon": float(getattr(args, "t5_align_ot_epsilon", 0.05)),
        "t5_align_ot_iters": int(getattr(args, "t5_align_ot_iters", 25)),
        "enable_mq_image_preserve": bool(getattr(args, "enable_mq_image_preserve", False)),
        "lambda_mq_image_preserve": float(getattr(args, "lambda_mq_image_preserve", 0.0)),
        "mq_image_preserve_margin": float(getattr(args, "mq_image_preserve_margin", 0.0)),
        "mq_norm_probe_with_t5": bool(getattr(args, "mq_norm_probe_with_t5", True)),
        "mq_norm_probe_every_n_steps": int(getattr(args, "mq_norm_probe_every_n_steps", 20)),
        "mq_norm_warn_ratio_low": float(getattr(args, "mq_norm_warn_ratio_low", 0.25)),
        "mq_norm_warn_ratio_high": float(getattr(args, "mq_norm_warn_ratio_high", 4.0)),
        "mq_norm_match_t5": bool(getattr(args, "mq_norm_match_t5", True)),
        "mq_norm_match_clip_min": float(getattr(args, "mq_norm_match_clip_min", 0.03)),
        "mq_norm_match_clip_max": float(getattr(args, "mq_norm_match_clip_max", 4.0)),
        "enable_wan_func_distill": bool(getattr(args, "enable_wan_func_distill", False)),
        "lambda_wan_func_distill": float(getattr(args, "lambda_wan_func_distill", 0.0)),
        "wan_func_teacher_mode": str(getattr(args, "wan_func_teacher_mode", "t5_only")),
        "batch_size": int(getattr(args, "batch_size", 1)),
        "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1)),
        "null_caption_prob": float(getattr(args, "null_caption_prob", 0.0)),
        "null_image_prob": float(getattr(args, "null_image_prob", 0.0)),
        "wan_train_mode": str(getattr(args, "wan_train_mode", "auto")),
        "wan_auto_full_mem_gb": float(getattr(args, "wan_auto_full_mem_gb", 120.0)),
        "wan_lr_ratio": float(getattr(args, "wan_lr_ratio", 1.0)),
        "wan_cond_name_pattern": str(getattr(args, "wan_cond_name_pattern", "")),
        "enable_wan_lora": bool(getattr(args, "enable_wan_lora", False)),
        "wan_lora_rank": int(getattr(args, "wan_lora_rank", 16)),
        "wan_lora_alpha": float(getattr(args, "wan_lora_alpha", 16.0)),
        "wan_lora_dropout": float(getattr(args, "wan_lora_dropout", 0.0)),
        "wan_lora_targets": str(getattr(args, "wan_lora_targets", "")),
        "wan_lora_extra_name_pattern": str(getattr(args, "wan_lora_extra_name_pattern", "")),
    }
    if wan_lora_config:
        config_payload["wan_lora"] = _to_jsonable(wan_lora_config)
    # 记录 MLLM embedding 行信息，便于推理期验证“新增 MQ token embedding 是否被保存/加载”。
    try:
        emb = module.mllm_model.mllm_backbone.get_input_embeddings()
        if emb is not None and getattr(emb, "weight", None) is not None:
            rows_total = int(emb.weight.shape[0])
            rows_base = int(getattr(module.mllm_model, "num_embeddings", 0))
            config_payload["mllm_embed_rows_total"] = rows_total
            config_payload["mllm_embed_rows_base"] = rows_base
            config_payload["mllm_embed_rows_added"] = max(rows_total - rows_base, 0)
    except Exception:
        pass
    if extra_info:
        config_payload["extra_info"] = _to_jsonable(extra_info)
    _write_json(path / "config.json", config_payload)
    if metrics_summary:
        _write_json(path / "metrics_summary.json", {str(k): _to_jsonable(v) for k, v in metrics_summary.items()})
    if metrics_tail is not None:
        _write_json(
            path / "metrics_tail.json",
            {"records": [{str(k): _to_jsonable(v) for k, v in row.items()} for row in metrics_tail]},
        )

    try:
        from safetensors.torch import save_file

        save_file(full_state_cpu, str(path / "model.safetensors"))
        save_file(trainable_state_cpu, str(path / "mq_encoder_trainable.safetensors"))
        if wan_trainable_state_cpu:
            save_file(wan_trainable_state_cpu, str(path / "wan_dit_trainable.safetensors"))
        if wan_lora_state_cpu:
            save_file(wan_lora_state_cpu, str(path / "wan_dit_lora.safetensors"))
    except Exception:
        # safetensors 为增强项，不可用时保持兼容主流程
        pass

    # 兼容“latest”指针
    try:
        with open(path.parent / "latest", "w", encoding="utf-8") as f:
            f.write(f"{path.name}\n")
    except Exception:
        pass

    return {
        "step": int(step),
        "path": str(path),
    }
# =============================================================================
# 数据集
# =============================================================================
try:
    from train_connector_for_wan import WanVideoDataset as _DefaultWanVideoDataset
except Exception:
    _DefaultWanVideoDataset = None

# 单一数据集入口：仅使用 WanVideoDataset。
# 在 train_metaquery_wan_new.py 中可通过设置 base_ti2v.WanDatasetClass 进行覆写。
WanDatasetClass = _DefaultWanVideoDataset


# =============================================================================
# Trainer
# =============================================================================
class MetaQueryWanTrainer:
    """
    MetaQuery + Wan TI2V 联合训练。

    训练流程:
        1. MetaQuery (Connector 可训练) → [B, 256, 4096]
        2. T5 编码文本 → [B, text_len, 4096]
        3. 拼接: [MQ + T5] → [B, 256+text_len, 4096]
        4. VAE 编码视频帧 → latent
        5. 采样噪声+时间步 → noisy_latent
        6. 参考图 VAE 编码 → first frame mask
        7. DiT (冻结) forward: 预测速度
        8. Flow Matching Loss → 反向传播 Connector + MQ Embeddings
    """

    def __init__(self, args):
        self.args = args
        self.dev_dit = torch.device(f"cuda:{args.dit_device}")
        self.dev_enc = torch.device(f"cuda:{args.encoder_device}")
        self.wandb = None
        self.wandb_run = None
        self.is_main_process = self._is_main_process()
        self._printed_grad_health = False
        self._skipped_step_count = 0
        self._oom_skip_count = 0
        self._error_skip_count = 0
        self._printed_context_inject_check = False
        self._param_monitor = []
        self._trainable_param_count = 0
        self._init_trainable_norm = 0.0
        self._init_param_sample_norm = 0.0
        _metrics_jsonl = (args.metrics_jsonl_path or "").strip()
        self._metrics_jsonl_path = str(Path(_metrics_jsonl).expanduser().resolve()) if _metrics_jsonl else ""
        self._metrics_history: List[Dict[str, Any]] = []
        self._train_before_checkpoint_path = ""
        self._train_wall_start = 0.0
        self._last_train_ref_anchor_alpha_mean = 0.0
        self._last_train_ref_anchor_applied = 0
        self._last_train_ref_anchor_effective_mode = "none"
        self._train_ref_anchor_mixed_counter = 0
        self._current_train_ref_anchor_mode = "none"
        self._last_train_video_conditioning_mode = "mq_only"
        self._last_train_prefix_latent_slots = 0
        self._last_train_target_latent_slots = 0
        self._last_train_prefix_loss_dropped = 0
        self._last_loss_denoise = 0.0
        self._last_loss_aux_align_total = 0.0
        self._last_loss_aux_t5_l2 = 0.0
        self._last_loss_aux_t5_cos = 0.0
        self._last_loss_aux_t5_stats = 0.0
        self._last_loss_aux_t5_gram = 0.0
        self._last_loss_aux_t5_cka = 0.0
        self._last_loss_aux_t5_ot = 0.0
        self._last_loss_aux_image_preserve = 0.0
        self._last_loss_aux_wan_func = 0.0
        self._loss_call_count = 0
        self._last_mq_rms = 0.0
        self._last_t5_rms = 0.0
        self._last_mq_t5_rms_ratio = 0.0
        self._last_mq_norm_match_scale = 1.0
        self._last_mq_norm_warn_flag = 0
        self._effective_wan_train_mode = "frozen"
        self._wan_trainable_names: List[str] = []
        self._wan_trainable_params_cache: List[torch.nn.Parameter] = []
        self._wan_lora_module_names: List[str] = []
        self._wan_lora_extra_trainable_names: List[str] = []

        print("\n" + "=" * 60)
        print("  MetaQuery + Wan TI2V 联合训练")
        print("=" * 60)
        print(f"  DiT 设备       : {self.dev_dit}")
        print(f"  Encoder 设备   : {self.dev_enc}")
        print(f"  学习率         : {args.learning_rate}")
        print(f"  LR 调度器      : {args.lr_scheduler_type}")
        print(f"  Cooldown 步数  : {args.cooldown_steps} (-1 表示使用 warmup_steps)")
        print(f"  训练步数       : {args.num_train_steps}")
        print(f"  有效 batch     : {args.batch_size * args.gradient_accumulation_steps}")
        print(
            f"  Loss 早停       : enabled={int(bool(args.enable_loss_early_stop))} "
            f"min_step={args.loss_early_stop_min_step} threshold={args.loss_early_stop_threshold}"
        )
        print(
            f"  Wan 训练模式    : req={args.wan_train_mode} auto_full_mem_gb={args.wan_auto_full_mem_gb} "
            f"wan_lr_ratio={args.wan_lr_ratio}"
        )
        print(
            f"  Wan LoRA        : enabled={int(bool(getattr(args, 'enable_wan_lora', False)))} "
            f"rank={getattr(args, 'wan_lora_rank', 16)} alpha={getattr(args, 'wan_lora_alpha', 16.0)} "
            f"dropout={getattr(args, 'wan_lora_dropout', 0.0)} "
            f"targets={getattr(args, 'wan_lora_targets', '')} "
            f"extra={getattr(args, 'wan_lora_extra_name_pattern', '') or '<none>'}"
        )
        print(
            f"  T5 对齐(已禁用) : cfg_enabled={int(bool(args.enable_t5_alignment))} "
            f"mode={args.t5_align_mode} "
            f"anchor={args.t5_align_anchor_tokens} "
            f"l2={args.lambda_t5_align_l2} cos={args.lambda_t5_align_cos} stats={args.lambda_t5_align_stats} "
            f"ot_eps={args.t5_align_ot_epsilon} ot_iters={args.t5_align_ot_iters}"
        )
        print(
            f"  图像保持(已禁用): cfg_enabled={int(bool(args.enable_mq_image_preserve))} "
            f"lambda={args.lambda_mq_image_preserve} margin={args.mq_image_preserve_margin}"
        )
        print(
            f"  MQ/T5范数探针  : enabled={int(bool(args.mq_norm_probe_with_t5))} "
            f"every={args.mq_norm_probe_every_n_steps} "
            f"warn=[{args.mq_norm_warn_ratio_low},{args.mq_norm_warn_ratio_high}] "
            f"match_t5={int(bool(args.mq_norm_match_t5))} "
            f"clip=[{args.mq_norm_match_clip_min},{args.mq_norm_match_clip_max}]"
        )
        print(f"  MQ connector norm init: {args.mq_connector_norm_init_scale}")
        print(
            f"  函数蒸馏(已禁用): cfg_enabled={int(bool(args.enable_wan_func_distill))} "
            f"lambda={args.lambda_wan_func_distill} teacher={args.wan_func_teacher_mode}"
        )
        print("  额外损失开关   : 当前版本固定仅使用 denoise MSE（其余辅助损失已禁用）")
        print("=" * 60)

        self._load_models()
        self._log_runtime_topology()
        self._setup_optimizer()
        self._audit_runtime_trainability(stage="init")
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

    def _wan_lora_enabled(self) -> bool:
        return bool(getattr(self.args, "enable_wan_lora", False))

    def _resolve_wan_train_mode(self) -> str:
        if self._wan_lora_enabled():
            return "lora"
        mode = str(getattr(self.args, "wan_train_mode", "auto")).strip().lower()
        if mode != "auto":
            return mode
        total_gb = 0.0
        if self.dev_dit.type == "cuda" and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.dev_dit)
                total_gb = float(props.total_memory) / float(1024 ** 3)
            except Exception:
                total_gb = 0.0
        threshold = float(getattr(self.args, "wan_auto_full_mem_gb", 120.0))
        return "full" if total_gb >= threshold else "cond_only"

    def _wan_cond_keywords(self) -> List[str]:
        custom = str(getattr(self.args, "wan_cond_name_pattern", "")).strip()
        if custom:
            return [k.strip().lower() for k in custom.split(",") if k.strip()]
        return [
            "cross_attn",
            "cross-attn",
            "crossattention",
            "cross_attention",
            "text_embedding",
            "time_projection",
            "modulation",
            "cross_attn_norm",
            "norm3",
        ]

    def _wan_lora_extra_keywords(self) -> List[str]:
        custom = str(getattr(self.args, "wan_lora_extra_name_pattern", "")).strip()
        if not custom:
            return []
        return [k.strip().lower() for k in custom.split(",") if k.strip()]

    def _configure_wan_trainable_params(self) -> None:
        wan_model = getattr(self.wan, "model", None)
        if wan_model is None:
            self._effective_wan_train_mode = "frozen"
            self._wan_trainable_names = []
            self._wan_trainable_params_cache = []
            self._wan_lora_module_names = []
            self._wan_lora_extra_trainable_names = []
            return

        # 先全冻结，再按模式打开。
        self._force_freeze(wan_model)
        mode = self._resolve_wan_train_mode()
        self._effective_wan_train_mode = mode
        selected_names: List[str] = []
        selected_params: List[torch.nn.Parameter] = []
        self._wan_lora_module_names = []
        self._wan_lora_extra_trainable_names = []

        if mode == "full":
            for name, p in wan_model.named_parameters():
                p.requires_grad_(True)
                selected_names.append(name)
                selected_params.append(p)
        elif mode == "cond_only":
            kws = self._wan_cond_keywords()
            for name, p in wan_model.named_parameters():
                lname = name.lower()
                if any(kw in lname for kw in kws):
                    p.requires_grad_(True)
                    selected_names.append(name)
                    selected_params.append(p)
        elif mode == "lora":
            if bool(getattr(self.args, "dit_fsdp", False)) or bool(getattr(self.args, "use_sp", False)):
                raise RuntimeError("Wan LoRA 当前仅支持非 dit_fsdp / 非 use_sp 训练路径")
            self._wan_lora_module_names = apply_lora_to_wan_model(
                wan_model,
                rank=int(getattr(self.args, "wan_lora_rank", 16)),
                alpha=float(getattr(self.args, "wan_lora_alpha", 16.0)),
                dropout=float(getattr(self.args, "wan_lora_dropout", 0.0)),
                target_types=getattr(self.args, "wan_lora_targets", "self_attn,cross_attn,ffn"),
            )
            if not self._wan_lora_module_names:
                raise RuntimeError("启用了 Wan LoRA，但未匹配到任何可注入的 Wan Linear")
            extra_kws = self._wan_lora_extra_keywords()
            for name, p in wan_model.named_parameters():
                lname = name.lower()
                is_lora = (".lora_a" in lname) or (".lora_b" in lname)
                is_extra = (not is_lora) and any(kw in lname for kw in extra_kws)
                if is_lora or is_extra:
                    p.requires_grad_(True)
                    selected_names.append(name)
                    selected_params.append(p)
                    if is_extra:
                        self._wan_lora_extra_trainable_names.append(name)
        elif mode == "frozen":
            pass
        else:
            raise ValueError(f"Unknown --wan_train_mode: {mode}")

        self._wan_trainable_names = selected_names
        self._wan_trainable_params_cache = selected_params
        if selected_params:
            wan_model.train()
        else:
            wan_model.eval()

        if self.is_main_process:
            total = sum(int(p.numel()) for p in selected_params)
            print(
                f"[WAN-TRAIN] requested={self.args.wan_train_mode} effective={mode} "
                f"trainable_tensors={len(selected_params)} trainable_params={total:,}"
            )
            if mode == "cond_only":
                kws = self._wan_cond_keywords()
                preview = ", ".join(kws[:10])
                print(f"[WAN-TRAIN] cond_only keywords={preview}")
            if mode == "lora":
                lora_cfg = build_lora_config_dict(
                    enabled=True,
                    rank=int(getattr(self.args, "wan_lora_rank", 16)),
                    alpha=float(getattr(self.args, "wan_lora_alpha", 16.0)),
                    dropout=float(getattr(self.args, "wan_lora_dropout", 0.0)),
                    targets=getattr(self.args, "wan_lora_targets", "self_attn,cross_attn,ffn"),
                    module_names=self._wan_lora_module_names,
                )
                print(f"[WAN-TRAIN] lora_cfg={json.dumps(lora_cfg, ensure_ascii=False)}")
                if self._wan_lora_extra_trainable_names:
                    preview = ", ".join(self._wan_lora_extra_trainable_names[:12])
                    more = "" if len(self._wan_lora_extra_trainable_names) <= 12 else f" ... +{len(self._wan_lora_extra_trainable_names)-12}"
                    print(f"[WAN-TRAIN] lora extra preview: {preview}{more}")
            if selected_names:
                preview = ", ".join(selected_names[:8])
                more = "" if len(selected_names) <= 8 else f" ... +{len(selected_names)-8}"
                print(f"[WAN-TRAIN] selected preview: {preview}{more}")

    def _wan_trainable_params(self) -> List[torch.nn.Parameter]:
        return list(self._wan_trainable_params_cache)

    def _all_trainable_params(self) -> List[torch.nn.Parameter]:
        out: List[torch.nn.Parameter] = []
        seen = set()
        for p in self._mq_trainable_params():
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
        for p in self._wan_trainable_params():
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
        return out

    @staticmethod
    def _module_param_stats(module: nn.Module | None) -> Dict[str, int]:
        total = 0
        trainable = 0
        if module is None or not isinstance(module, nn.Module):
            return {"total": 0, "trainable": 0}
        for p in module.parameters():
            n = int(p.numel())
            total += n
            if p.requires_grad:
                trainable += n
        return {"total": total, "trainable": trainable}

    @staticmethod
    def _named_param_id_map(module: nn.Module | None, prefix: str) -> Dict[int, str]:
        out: Dict[int, str] = {}
        if module is None or not isinstance(module, nn.Module):
            return out
        for name, p in module.named_parameters():
            out[id(p)] = f"{prefix}.{name}"
        return out

    @staticmethod
    def _force_freeze(module: nn.Module | None) -> None:
        if module is None or not isinstance(module, nn.Module):
            return
        try:
            module.eval()
        except Exception:
            pass
        try:
            module.requires_grad_(False)
        except Exception:
            for p in module.parameters():
                p.requires_grad_(False)

    def _log_runtime_topology(self) -> None:
        if not self.is_main_process:
            return
        args = self.args
        same_gpu = (self.dev_dit == self.dev_enc)
        print(
            "[AUDIT][TOPO] "
            f"dit_device={self.dev_dit} encoder_device={self.dev_enc} same_gpu={same_gpu} "
            f"t5_cpu={args.t5_cpu} t5_fsdp={args.t5_fsdp} dit_fsdp={args.dit_fsdp} use_sp={args.use_sp} "
            f"num_metaqueries={args.num_metaqueries} aug_text_len={getattr(self, '_aug_text_len', -1)} "
            f"wan_mode_effective={getattr(self, '_effective_wan_train_mode', 'frozen')}"
        )
        if same_gpu:
            print("[AUDIT][TOPO][WARN] DiT 与 Qwen/Connector 在同一 GPU，显存峰值风险较高。")
        if (not args.t5_cpu) and (not args.t5_fsdp):
            print("[AUDIT][TOPO] T5 文本编码器会在 DiT 卡上参与前向（no_grad）。")
        try:
            from wan.modules import attention as _attn
            fa2 = bool(getattr(_attn, "FLASH_ATTN_2_AVAILABLE", False))
            fa3 = bool(getattr(_attn, "FLASH_ATTN_3_AVAILABLE", False))
            force_sdpa = bool(getattr(_attn, "_FORCE_SDPA", False))
            print(
                "[AUDIT][ATTN] "
                f"flash_attn2={fa2} flash_attn3={fa3} force_sdpa={force_sdpa}"
            )
        except Exception as e:
            print(f"[AUDIT][ATTN][WARN] 无法读取 attention backend 信息: {e}")

    def _audit_runtime_trainability(self, stage: str = "runtime", strict: bool | None = None) -> None:
        args = self.args
        if strict is None:
            strict = bool(getattr(args, "strict_freeze_check", True))

        wan_mode = str(getattr(self, "_effective_wan_train_mode", "frozen"))
        # Wan 是否冻结由 wan_train_mode 决定；T5/VAE 始终冻结。
        t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
        self._force_freeze(t5_model)
        vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
        if vae_model is None:
            vae_model = getattr(self.wan, "vae", None)
        self._force_freeze(vae_model)

        stats_wan = self._module_param_stats(getattr(self.wan, "model", None))
        stats_t5 = self._module_param_stats(t5_model)
        stats_vae = self._module_param_stats(vae_model)

        mq_module = self._mq_encoder_module()
        stats_mq = self._module_param_stats(mq_module)
        mq_trainable_params = self._mq_trainable_params()
        wan_trainable_params = self._wan_trainable_params()
        mq_trainable_ids = {id(p) for p in mq_trainable_params}
        wan_trainable_ids = {id(p) for p in wan_trainable_params}
        allowed_trainable_ids = mq_trainable_ids | wan_trainable_ids
        emb_trainable = 0
        emb_rows_total = 0
        emb_rows_base = 0
        emb_rows_added = 0
        emb_hidden = 0
        try:
            backbone = mq_module.mllm_model.mllm_backbone
            emb = backbone.get_input_embeddings()
            if emb is not None and getattr(emb, "weight", None) is not None:
                w = emb.weight
                emb_rows_total = int(w.shape[0])
                emb_hidden = int(w.shape[1]) if w.ndim >= 2 else 0
                emb_rows_base = int(getattr(mq_module.mllm_model, "num_embeddings", 0))
                emb_rows_added = max(emb_rows_total - emb_rows_base, 0)
                if bool(w.requires_grad):
                    emb_trainable = int(w.numel())
        except Exception:
            pass

        opt_params = []
        for g in self.optimizer.param_groups:
            opt_params.extend(g.get("params", []))
        opt_ids = [id(p) for p in opt_params]
        opt_id_set = set(opt_ids)

        outside_ids = [pid for pid in opt_ids if pid not in allowed_trainable_ids]
        missing_mq_ids = [pid for pid in mq_trainable_ids if pid not in opt_id_set]
        missing_wan_ids = [pid for pid in wan_trainable_ids if pid not in opt_id_set]
        duplicate_count = max(len(opt_ids) - len(opt_id_set), 0)

        name_map: Dict[int, str] = {}
        name_map.update(self._named_param_id_map(getattr(self.wan, "model", None), "wan.model"))
        name_map.update(self._named_param_id_map(t5_model, "wan.text_encoder.model"))
        name_map.update(self._named_param_id_map(vae_model, "wan.vae.model"))
        name_map.update(self._named_param_id_map(mq_module, "mq_encoder"))

        unexpected_mq_names = []
        for name, p in mq_module.named_parameters():
            if not p.requires_grad:
                continue
            lower = name.lower()
            if ("connector" in lower) or ("embed" in lower):
                continue
            unexpected_mq_names.append(name)

        if self.is_main_process:
            print(
                f"[AUDIT][FREEZE][{stage}] "
                f"wan_trainable={stats_wan['trainable']:,}/{stats_wan['total']:,} "
                f"t5_trainable={stats_t5['trainable']:,}/{stats_t5['total']:,} "
                f"vae_trainable={stats_vae['trainable']:,}/{stats_vae['total']:,} "
                f"mq_trainable={stats_mq['trainable']:,}/{stats_mq['total']:,} "
                f"mq_trainable_tensors={len(mq_trainable_params)} "
                f"wan_mode={wan_mode} wan_trainable_tensors={len(wan_trainable_params)}"
            )
            print(
                f"[AUDIT][OPT][{stage}] "
                f"optimizer_params={len(opt_ids)} "
                f"outside_allowed={len(outside_ids)} missing_mq={len(missing_mq_ids)} "
                f"missing_wan={len(missing_wan_ids)} duplicates={duplicate_count}"
            )
            print(
                f"[AUDIT][MQ-EMB][{stage}] "
                f"enabled={int(bool(args.train_mq_input_embeddings))} "
                f"embed_trainable={emb_trainable:,} "
                f"rows_total={emb_rows_total} rows_base={emb_rows_base} rows_added={emb_rows_added} "
                f"hidden={emb_hidden} expected_added≈num_metaqueries+2={int(args.num_metaqueries) + 2}"
            )
            if unexpected_mq_names:
                preview = ", ".join(unexpected_mq_names[:6])
                more = "" if len(unexpected_mq_names) <= 6 else f" ... +{len(unexpected_mq_names)-6}"
                print(
                    "[AUDIT][MQ][WARN] 检测到非 connector/embed 命名的可训练参数: "
                    f"{preview}{more}"
                )

        errors = []
        if wan_mode == "frozen" and stats_wan["trainable"] > 0:
            errors.append(f"Wan DiT 期望冻结但仍有可训练参数: {stats_wan['trainable']}")
        if wan_mode != "frozen" and len(wan_trainable_ids) == 0:
            errors.append(f"Wan DiT 训练模式={wan_mode} 但未选中可训练参数")
        if stats_t5["trainable"] > 0:
            errors.append(f"Wan T5 仍有可训练参数: {stats_t5['trainable']}")
        if stats_vae["trainable"] > 0:
            errors.append(f"Wan VAE 仍有可训练参数: {stats_vae['trainable']}")
        if len(mq_trainable_ids) == 0:
            errors.append("MQ encoder 无可训练参数")
        if bool(args.train_mq_input_embeddings) and emb_trainable <= 0:
            errors.append("设置了 train_mq_input_embeddings，但输入 embedding 未开启训练")
        if (not bool(args.train_mq_input_embeddings)) and emb_trainable > 0:
            errors.append("设置了 freeze_mq_input_embeddings，但输入 embedding 仍可训练")
        if outside_ids:
            names = [name_map.get(pid, f"<unknown:{pid}>") for pid in outside_ids[:8]]
            errors.append(f"optimizer 含非允许参数(MQ+Wan): {names}")
        if missing_mq_ids:
            names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_mq_ids[:8]]
            errors.append(f"部分 MQ 可训练参数未进 optimizer: {names}")
        if missing_wan_ids:
            names = [name_map.get(pid, f"<unknown:{pid}>") for pid in missing_wan_ids[:8]]
            errors.append(f"部分 Wan 可训练参数未进 optimizer: {names}")
        if duplicate_count > 0:
            errors.append(f"optimizer 参数重复引用: {duplicate_count}")
        if wan_mode != "frozen" and torch.distributed.is_available() and torch.distributed.is_initialized():
            ws = int(torch.distributed.get_world_size())
            if ws > 1:
                dit_fsdp_enabled = bool(getattr(args, "dit_fsdp", False))
                use_sp_enabled = bool(getattr(args, "use_sp", False))
                if not (dit_fsdp_enabled or use_sp_enabled):
                    errors.append(
                        "多进程 Wan 可训练模式需要启用 dit_fsdp 或 use_sp；"
                        "否则当前仅有 MQ-encoder DDP 会导致 Wan 参数跨 rank 不一致。"
                    )

        if errors:
            msg = " | ".join(errors)
            if strict:
                raise RuntimeError(f"[AUDIT][FAIL][{stage}] {msg}")
            if self.is_main_process:
                print(f"[AUDIT][WARN][{stage}] {msg}")

    def post_wrap_ddp_audit(self) -> None:
        # DDP 包装后再做一次 optimizer 与 trainable 参数一致性检查
        if not hasattr(self.mq_encoder, "module"):
            return
        self._audit_runtime_trainability(stage="post_ddp")

    def _log_grad_health_once(self):
        if self._printed_grad_health:
            return
        module = self._mq_encoder_module()
        connector_has_grad = False
        mq_embed_has_grad = False
        wan_has_grad = False
        connector_grad_norm = 0.0
        mq_embed_grad_norm = 0.0
        wan_grad_norm = 0.0
        mq_embed_added_grad_norm = 0.0
        mq_embed_base_grad_norm = 0.0
        mq_embed_boundary_grad_norm = 0.0
        mq_embed_query_grad_norm = 0.0
        try:
            for _, p in module.mllm_model.connector.named_parameters():
                if p.grad is not None:
                    connector_has_grad = True
                    connector_grad_norm = float(p.grad.detach().float().norm().item())
                    break
            emb = module.mllm_model.mllm_backbone.get_input_embeddings()
            if emb is not None and getattr(emb, "weight", None) is not None and emb.weight.grad is not None:
                mq_embed_has_grad = True
                g = emb.weight.grad.detach().float()
                mq_embed_grad_norm = float(g.norm().item())
                base_rows = int(getattr(module.mllm_model, "num_embeddings", 0))
                if g.ndim >= 2 and 0 < base_rows < int(g.shape[0]):
                    mq_embed_base_grad_norm = float(g[:base_rows].norm().item())
                    mq_embed_added_grad_norm = float(g[base_rows:].norm().item())
                    boundary_end = min(base_rows + 2, int(g.shape[0]))
                    query_end = min(boundary_end + int(self.args.num_metaqueries), int(g.shape[0]))
                    if boundary_end > base_rows:
                        mq_embed_boundary_grad_norm = float(g[base_rows:boundary_end].norm().item())
                    if query_end > boundary_end:
                        mq_embed_query_grad_norm = float(g[boundary_end:query_end].norm().item())
        except Exception:
            pass
        try:
            for p in self._wan_trainable_params():
                if p.grad is not None:
                    wan_has_grad = True
                    wan_grad_norm = float(p.grad.detach().float().norm().item())
                    break
        except Exception:
            pass
        print(
            "[GRAD-CHECK] "
            f"connector_has_grad={connector_has_grad} connector_grad_norm={connector_grad_norm:.4e} "
            f"mq_embed_has_grad={mq_embed_has_grad} mq_embed_grad_norm={mq_embed_grad_norm:.4e} "
            f"wan_has_grad={wan_has_grad} wan_grad_norm={wan_grad_norm:.4e} "
            f"mq_embed_added_grad_norm={mq_embed_added_grad_norm:.4e} "
            f"mq_embed_base_grad_norm={mq_embed_base_grad_norm:.4e} "
            f"mq_embed_boundary_grad_norm={mq_embed_boundary_grad_norm:.4e} "
            f"mq_embed_query_grad_norm={mq_embed_query_grad_norm:.4e}"
        )
        self._printed_grad_health = True

    def _verify_train_context_injection_once(
        self,
        mq_feat: torch.Tensor,
        aug_feat: torch.Tensor,
    ) -> None:
        if self._printed_context_inject_check:
            return
        mq_len = int(mq_feat.shape[0])
        aug_len = int(aug_feat.shape[0])
        if aug_len != mq_len:
            raise RuntimeError(
                f"[VERIFY][TRAIN] MQ-only context 长度异常: aug={aug_len}, mq={mq_len}"
            )
        mq_ok = torch.allclose(
            aug_feat.float(),
            mq_feat.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        if not mq_ok:
            raise RuntimeError("[VERIFY][TRAIN] MQ-only context 未正确注入 Wan context")
        if aug_len > self._aug_text_len:
            raise RuntimeError(
                f"[VERIFY][TRAIN] aug_len 超出 text_len: aug={aug_len}, text_len={self._aug_text_len}"
            )
        print(
            "[VERIFY][TRAIN] context 注入检查通过: "
            f"mq_tokens={mq_len} aug_tokens={aug_len} model_text_len={self._aug_text_len}"
        )
        self._printed_context_inject_check = True

    def _init_trainability_monitor(self):
        self._param_monitor = []
        total_sq = 0.0
        sample_sq = 0.0
        total_params = 0
        named_params: List[Tuple[str, torch.nn.Parameter]] = []
        mq_module = self._mq_encoder_module()
        named_params.extend((f"mq_encoder.{n}", p) for n, p in mq_module.named_parameters() if p.requires_grad)
        wan_model = getattr(self.wan, "model", None)
        if isinstance(wan_model, nn.Module):
            named_params.extend((f"wan.model.{n}", p) for n, p in wan_model.named_parameters() if p.requires_grad)
        for name, p in named_params:
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

    def _record_metrics(self, metrics: Dict[str, Any]) -> None:
        keep_keys = [
            "train/step",
            "train/loss_step",
            "train/loss_denoise",
            "train/loss_align_total",
            "train/loss_align_t5_l2",
            "train/loss_align_t5_cos",
            "train/loss_align_t5_stats",
            "train/loss_align_t5_gram",
            "train/loss_align_t5_cka",
            "train/loss_align_t5_ot",
            "train/loss_align_img_preserve",
            "train/loss_align_wan_func",
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
        row = {k: metrics[k] for k in keep_keys if k in metrics}
        self._metrics_history.append(row)

    def _build_metrics_summary(self, step: int) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
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

    def _write_training_chain_manifest(self, output_dir: Path, final_checkpoint_path: str, final_step: int) -> None:
        if not self.is_main_process:
            return
        output_dir = output_dir.expanduser().resolve()
        payload = {
            "before_checkpoint_path": self._train_before_checkpoint_path,
            "final_checkpoint_path": str(Path(final_checkpoint_path).expanduser().resolve()),
            "metrics_jsonl_path": self._metrics_jsonl_path,
            "args": {str(k): _to_jsonable(v) for k, v in vars(self.args).items()},
            "metrics_summary": self._build_metrics_summary(step=final_step),
        }
        _write_json(output_dir / "training_chain_manifest.json", payload)

    def _wandb_config(self):
        args = self.args
        return {
            "task": "wan_ti2v",
            "learning_rate": args.learning_rate,
            "num_train_steps": args.num_train_steps,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "cooldown_steps": args.cooldown_steps,
            "lr_min_ratio": args.lr_min_ratio,
            "enable_t5_alignment": args.enable_t5_alignment,
            "t5_align_mode": args.t5_align_mode,
            "t5_align_anchor_tokens": args.t5_align_anchor_tokens,
            "lambda_t5_align_l2": args.lambda_t5_align_l2,
            "lambda_t5_align_cos": args.lambda_t5_align_cos,
            "lambda_t5_align_stats": args.lambda_t5_align_stats,
            "t5_align_ot_epsilon": args.t5_align_ot_epsilon,
            "t5_align_ot_iters": args.t5_align_ot_iters,
            "enable_mq_image_preserve": args.enable_mq_image_preserve,
            "lambda_mq_image_preserve": args.lambda_mq_image_preserve,
            "mq_image_preserve_margin": args.mq_image_preserve_margin,
            "mq_norm_probe_with_t5": args.mq_norm_probe_with_t5,
            "mq_norm_probe_every_n_steps": args.mq_norm_probe_every_n_steps,
            "mq_norm_warn_ratio_low": args.mq_norm_warn_ratio_low,
            "mq_norm_warn_ratio_high": args.mq_norm_warn_ratio_high,
            "mq_norm_match_t5": args.mq_norm_match_t5,
            "mq_norm_match_clip_min": args.mq_norm_match_clip_min,
            "mq_norm_match_clip_max": args.mq_norm_match_clip_max,
            "enable_wan_func_distill": args.enable_wan_func_distill,
            "lambda_wan_func_distill": args.lambda_wan_func_distill,
            "wan_func_teacher_mode": args.wan_func_teacher_mode,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "frame_num": args.frame_num,
            "max_area": args.max_area,
            "num_metaqueries": args.num_metaqueries,
            "connector_num_hidden_layers": args.connector_num_hidden_layers,
            "dit_condition_mode": args.dit_condition_mode,
            "mq_gradient_checkpointing": args.mq_gradient_checkpointing,
            "train_mq_input_embeddings": args.train_mq_input_embeddings,
            "mq_connector_norm_init_scale": args.mq_connector_norm_init_scale,
            "null_caption_prob": args.null_caption_prob,
            "null_image_prob": args.null_image_prob,
            "wan_train_mode": args.wan_train_mode,
            "wan_auto_full_mem_gb": args.wan_auto_full_mem_gb,
            "wan_lr_ratio": args.wan_lr_ratio,
            "wan_cond_name_pattern": args.wan_cond_name_pattern,
            "enable_wan_lora": args.enable_wan_lora,
            "wan_lora_rank": args.wan_lora_rank,
            "wan_lora_alpha": args.wan_lora_alpha,
            "wan_lora_dropout": args.wan_lora_dropout,
            "wan_lora_targets": args.wan_lora_targets,
            "wan_lora_extra_name_pattern": args.wan_lora_extra_name_pattern,
            "t5_cpu": args.t5_cpu,
            "dit_fsdp": args.dit_fsdp,
            "t5_fsdp": args.t5_fsdp,
            "use_sp": args.use_sp,
            "aggressive_empty_cache": args.aggressive_empty_cache,
            "seed": args.seed,
            "save_steps": args.save_steps,
            "log_steps": args.log_steps,
            "enable_loss_early_stop": args.enable_loss_early_stop,
            "loss_early_stop_min_step": args.loss_early_stop_min_step,
            "loss_early_stop_threshold": args.loss_early_stop_threshold,
            "log_every_step": args.log_every_step,
            "wandb_log_every_step": args.wandb_log_every_step,
            "metrics_jsonl_path": args.metrics_jsonl_path,
            "log_cuda_memory": args.log_cuda_memory,
            "output_dir": args.output_dir,
            "local_openvid_video_root": args.local_openvid_video_root,
            "local_openvid_csv_path": args.local_openvid_csv_path,
            "local_openvid_limit": args.local_openvid_limit,
            "local_openvid_hd_video_root": args.local_openvid_hd_video_root,
            "local_openvid_hd_csv_path": args.local_openvid_hd_csv_path,
            "local_openvid_hd_limit": args.local_openvid_hd_limit,
            "wan_checkpoint_dir": args.wan_checkpoint_dir,
            "qwen3vl_model_id": args.qwen3vl_model_id,
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
        run_name = args.wandb_run_name.strip() or f"wan-ti2v-metaquery-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        """加载所有模型。"""
        args = self.args

        # ── 1. Wan TI2V Pipeline ─────────────────────────────────────────
        print("\n[1/3] 加载 Wan TI2V Pipeline...")
        from wan import WanTI2V
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['ti2v-5B']
        runtime_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        self.wan = WanTI2V(
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

        # DiT 冻结；FSDP/SP 路径不再显式 .to，避免破坏分片包装
        if not (args.dit_fsdp or args.use_sp):
            self.wan.model.to(self.dev_dit)
        self.wan.model.eval().requires_grad_(False)
        t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
        vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
        if vae_model is None:
            vae_model = getattr(self.wan, "vae", None)
        self._force_freeze(t5_model)
        self._force_freeze(vae_model)

        self.wan_config = config
        self.text_len = config.text_len  # 512
        print(f"  ✅ Wan TI2V 5B 已加载, text_len={self.text_len}")

        # ── 2. MetaQuery Encoder (直接输出 4096) ─────────────────────────
        print("\n[2/3] 加载 MetaQuery Encoder (→4096)...")
        # 统一使用 train_connector_for_wan.py 中的实现，避免同名类双份定义导致“改了不生效”。
        from train_connector_for_wan import MetaQueryEncoderForWan as SharedMetaQueryEncoderForWan
        self.mq_encoder = SharedMetaQueryEncoderForWan(
            qwen3vl_model_id=args.qwen3vl_model_id,
            num_metaqueries=args.num_metaqueries,
            connector_num_hidden_layers=args.connector_num_hidden_layers,
            gradient_checkpointing=args.mq_gradient_checkpointing,
            train_input_embeddings=args.train_mq_input_embeddings,
            connector_norm_init_scale=args.mq_connector_norm_init_scale,
            dtype=torch.bfloat16,
            device=f"cuda:{args.encoder_device}",
        )
        print(f"  ✅ Encoder实现来源: {self.mq_encoder.__class__.__module__}.{self.mq_encoder.__class__.__name__}")
        self.mq_encoder.train()
        if args.resume_mq_encoder_path:
            state, resolved_path = load_mq_encoder_state(
                args.resume_mq_encoder_path,
                map_location="cpu",
            )
            missing, unexpected = self.mq_encoder.load_state_dict(state, strict=False)
            print(f"  ✅ 已加载初始权重: {resolved_path}")
            print(f"     missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"  ✅ MetaQuery Encoder 已加载")

        # ── 3. 验证维度 ──────────────────────────────────────────────────
        print("\n[3/3] 验证维度对齐...")
        wan_text_dim = self.wan.model.text_dim  # 4096
        mq_out_dim = self.mq_encoder.wan_text_dim  # 4096
        assert wan_text_dim == mq_out_dim, (
            f"维度不匹配! Wan text_dim={wan_text_dim}, MQ out={mq_out_dim}"
        )
        print(f"  ✅ MQ output dim = Wan text_dim = {wan_text_dim}")

        # MQ-only: DiT text_len 仅容纳 MQ tokens
        self._orig_text_len = self.wan.model.text_len
        self._aug_text_len = args.num_metaqueries
        print(f"  ✅ text_len(MQ-only): {self._orig_text_len} → {self._aug_text_len}")
        self._configure_wan_trainable_params()

    def _setup_optimizer(self):
        """设置优化器和学习率调度。"""
        args = self.args

        mq_params = self._mq_trainable_params()
        wan_params = self._wan_trainable_params()
        trainable_params = self._all_trainable_params()
        print(f"\n[Optimizer] 可训练参数组:")
        print(f"  Connector + MQ Embeddings: {sum(p.numel() for p in mq_params) / 1e6:.1f}M")
        print(f"  Wan DiT (mode={self._effective_wan_train_mode}): {sum(p.numel() for p in wan_params) / 1e6:.1f}M")
        print(f"  Total trainable: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")
        if len(trainable_params) <= 0:
            raise RuntimeError("无可训练参数：请检查 MQ/Wan 训练配置。")

        param_groups: List[Dict[str, Any]] = []
        if mq_params:
            param_groups.append(
                {
                    "name": "mq",
                    "params": mq_params,
                    "lr": float(args.learning_rate),
                }
            )
        if wan_params:
            param_groups.append(
                {
                    "name": "wan",
                    "params": wan_params,
                    "lr": float(args.learning_rate) * float(getattr(args, "wan_lr_ratio", 1.0)),
                }
            )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8,
        )

        def lr_lambda(step):
            warmup = max(int(args.warmup_steps), 0)
            total = max(int(args.num_train_steps), 1)
            cooldown = int(getattr(args, "cooldown_steps", -1))
            if cooldown < 0:
                cooldown = warmup
            cooldown = max(cooldown, 0)
            warmup = min(warmup, total)
            cooldown = min(cooldown, max(total - warmup, 0))

            if step < warmup:
                return step / max(1, warmup)
            if args.lr_scheduler_type == "constant_with_warmup":
                return 1.0
            if args.lr_scheduler_type == "warmup_hold_cooldown":
                cooldown_start = total - cooldown
                if cooldown <= 0 or step < cooldown_start:
                    return 1.0
                progress = (step - cooldown_start) / max(1, cooldown)
                progress = min(max(progress, 0.0), 1.0)
                return 1.0 - (1.0 - float(args.lr_min_ratio)) * progress
            progress = (step - warmup) / max(1, total - warmup)
            return max(float(args.lr_min_ratio), 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _encode_text(self, prompts):
        """T5 编码文本"""
        with torch.no_grad():
            if not self.args.t5_cpu and not self.args.t5_fsdp:
                self.wan.text_encoder.model.to(self.dev_dit)
                context = self.wan.text_encoder(prompts, self.dev_dit)
            else:
                context = self.wan.text_encoder(prompts, torch.device("cpu"))
                context = [t.to(self.dev_dit, dtype=torch.bfloat16) for t in context]
        return context  # List[Tensor], each [text_len, 4096]

    @staticmethod
    def _resize_token_sequence(seq: torch.Tensor, out_tokens: int) -> torch.Tensor:
        """
        将 [L, D] token 序列重采样到 [out_tokens, D]。
        使用线性插值仅做 teacher 侧长度对齐，不引入额外可训练参数。
        """
        if seq.dim() != 2:
            raise ValueError(f"expect [L, D], got shape={tuple(seq.shape)}")
        out_tokens = max(1, int(out_tokens))
        if int(seq.shape[0]) == out_tokens:
            return seq
        # F.interpolate 期望 [N, C, L]
        x = seq.transpose(0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=out_tokens, mode="linear", align_corners=False)
        return x.squeeze(0).transpose(0, 1)

    @staticmethod
    def _token_rms(feat: torch.Tensor) -> float:
        x = feat.float()
        return float(torch.sqrt(torch.mean(x * x)).item())

    def _probe_and_optionally_match_mq_norm(
        self,
        captions: List[str],
        mq_features: torch.Tensor,
        t5_context: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        args = self.args
        enabled = bool(getattr(args, "mq_norm_probe_with_t5", True))
        match_t5 = bool(getattr(args, "mq_norm_match_t5", True))
        every = max(1, int(getattr(args, "mq_norm_probe_every_n_steps", 20)))
        should_probe = enabled and ((int(self._loss_call_count) % every) == 0)
        if not should_probe and not match_t5:
            self._last_mq_norm_warn_flag = 0
            self._last_mq_norm_match_scale = 1.0
            return mq_features

        try:
            if t5_context is None:
                with torch.no_grad():
                    t5_context = self._encode_text(captions)

            t5_rms_vals = [self._token_rms(seq.to(self.dev_dit, dtype=torch.bfloat16)) for seq in t5_context]
            t5_rms = float(sum(t5_rms_vals) / max(len(t5_rms_vals), 1)) if t5_rms_vals else 0.0
            mq_rms = self._token_rms(mq_features)
            ratio = float(mq_rms / (t5_rms + 1e-8))

            self._last_mq_rms = float(mq_rms)
            self._last_t5_rms = float(t5_rms)
            self._last_mq_t5_rms_ratio = float(ratio)

            low = float(getattr(args, "mq_norm_warn_ratio_low", 0.25))
            high = float(getattr(args, "mq_norm_warn_ratio_high", 4.0))
            warn_flag = int(ratio < low or ratio > high)
            self._last_mq_norm_warn_flag = warn_flag
            if warn_flag and self.is_main_process:
                print(
                    "[MQ-NORM][WARN] "
                    f"mq_rms={mq_rms:.6f} t5_rms={t5_rms:.6f} ratio={ratio:.6f} "
                    f"outside=[{low}, {high}] loss_call={self._loss_call_count}. "
                    "建议检查: mq_connector_norm_init_scale / mq_norm_match_t5 / clip_min"
                )

            if match_t5 and t5_rms > 0:
                smin = float(getattr(args, "mq_norm_match_clip_min", 0.03))
                smax = float(getattr(args, "mq_norm_match_clip_max", 4.0))
                scale = float(t5_rms / (mq_rms + 1e-8))
                scale = float(max(smin, min(smax, scale)))
                self._last_mq_norm_match_scale = scale
                mq_features = mq_features * scale
                if self.is_main_process and warn_flag:
                    post_ratio = ratio * scale
                    print(
                        "[MQ-NORM][ADJUST] "
                        f"applied_scale={scale:.6f} post_ratio≈{post_ratio:.6f} "
                        f"(raw_target_scale={t5_rms / (mq_rms + 1e-8):.6f}, clip=[{smin},{smax}])"
                    )
            else:
                self._last_mq_norm_match_scale = 1.0
        except Exception as e:
            self._last_mq_norm_warn_flag = 1
            self._last_mq_norm_match_scale = 1.0
            if self.is_main_process:
                print(f"[MQ-NORM][WARN] probe failed: {e}")

        return mq_features

    @staticmethod
    def _token_gram_matrix(tokens: torch.Tensor) -> torch.Tensor:
        """
        计算 token 关系矩阵（Gram）。
        输入: [B, T, D]，输出: [B, T, T]
        """
        if tokens.dim() != 3:
            raise ValueError(f"expect [B, T, D], got shape={tuple(tokens.shape)}")
        x = tokens - tokens.mean(dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=-1, eps=1e-6)
        return torch.matmul(x, x.transpose(1, 2))

    @staticmethod
    def _linear_cka_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        线性 CKA 损失，返回 1-CKA（越小越好）。
        输入: x/y [B, T, D]
        """
        if x.shape != y.shape:
            raise ValueError(f"CKA shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")
        x_c = x - x.mean(dim=1, keepdim=True)
        y_c = y - y.mean(dim=1, keepdim=True)
        kx = torch.matmul(x_c, x_c.transpose(1, 2))
        ky = torch.matmul(y_c, y_c.transpose(1, 2))
        hsic = (kx * ky).sum(dim=(1, 2))
        denom = torch.sqrt(
            kx.square().sum(dim=(1, 2)).clamp_min(1e-12)
            * ky.square().sum(dim=(1, 2)).clamp_min(1e-12)
        )
        cka = hsic / denom.clamp_min(1e-12)
        return (1.0 - cka.clamp(-1.0, 1.0)).mean()

    @staticmethod
    def _sinkhorn_ot_cost(
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        epsilon: float = 0.05,
        iters: int = 25,
    ) -> torch.Tensor:
        """
        Sinkhorn OT 软匹配代价（排列无关）。
        输入: src/tgt [B, T, D]
        输出: 标量（batch 平均 OT cost）
        """
        if src_tokens.dim() != 3 or tgt_tokens.dim() != 3:
            raise ValueError(
                f"Sinkhorn expect [B,T,D], got src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
            )
        if int(src_tokens.shape[0]) != int(tgt_tokens.shape[0]) or int(src_tokens.shape[2]) != int(tgt_tokens.shape[2]):
            raise ValueError(
                f"Sinkhorn shape mismatch: src={tuple(src_tokens.shape)} tgt={tuple(tgt_tokens.shape)}"
            )
        bsz, n_tok, _ = src_tokens.shape
        m_tok = int(tgt_tokens.shape[1])
        if n_tok <= 0 or m_tok <= 0:
            return src_tokens.new_zeros(())

        cost = torch.cdist(src_tokens, tgt_tokens, p=2).pow(2)  # [B, N, M]
        eps = max(float(epsilon), 1e-6)
        kernel = torch.exp(-cost / eps).clamp_min(1e-12)
        a = src_tokens.new_full((bsz, n_tok), 1.0 / float(n_tok))
        b = src_tokens.new_full((bsz, m_tok), 1.0 / float(m_tok))
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        kernel_t = kernel.transpose(1, 2)

        n_iter = max(int(iters), 1)
        for _ in range(n_iter):
            kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            u = a / kv
            ktu = torch.bmm(kernel_t, u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            v = b / ktu

        plan = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)  # [B, N, M]
        return (plan * cost).sum(dim=(1, 2)).mean()

    def _compute_mq_aux_losses(
        self,
        captions: List[str],
        mq_refs: List[Any],
        mq_features: torch.Tensor,
        t5_context: List[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 MQ 辅助约束：
        1) T5 对齐（支持 anchor/Gram+CKA/Sinkhorn）
        2) T5 统计对齐（均值/方差）
        3) 图像保持（可选）：有图条件需与 text-only MQ 保持最小间隔
        """
        device = self.dev_dit
        zero = mq_features.new_zeros(())
        out = {
            "t5_l2": zero,
            "t5_cos": zero,
            "t5_stats": zero,
            "t5_gram": zero,
            "t5_cka": zero,
            "t5_ot": zero,
            "image_preserve": zero,
            "total": zero,
        }
        args = self.args
        need_t5 = bool(args.enable_t5_alignment) and (
            float(args.lambda_t5_align_l2) > 0.0
            or float(args.lambda_t5_align_cos) > 0.0
            or float(args.lambda_t5_align_stats) > 0.0
        )
        need_img = bool(args.enable_mq_image_preserve) and float(args.lambda_mq_image_preserve) > 0.0
        if not (need_t5 or need_img):
            return out

        mq_float = mq_features.to(device=device, dtype=torch.float32)
        tokens = int(mq_float.shape[1])
        hidden = int(mq_float.shape[2])
        anchor_tokens = max(1, min(int(args.t5_align_anchor_tokens), tokens))

        if need_t5:
            with torch.no_grad():
                teacher_ctx = t5_context if t5_context is not None else self._encode_text(captions)
                pooled_t5 = []
                for t5_seq in teacher_ctx:
                    # t5_seq: [L_t5, 4096]
                    t5_seq_f = t5_seq.to(device=device, dtype=torch.float32)
                    if int(t5_seq_f.shape[-1]) != hidden:
                        raise RuntimeError(
                            f"T5 hidden={int(t5_seq_f.shape[-1])} 与 MQ hidden={hidden} 不一致"
                        )
                    pooled_t5.append(self._resize_token_sequence(t5_seq_f, tokens))
                t5_teacher = torch.stack(pooled_t5, dim=0)  # [B, tokens, 4096]

            align_mode = str(getattr(args, "t5_align_mode", "gram_cka")).strip().lower()
            if align_mode == "anchor":
                mq_anchor = mq_float[:, :anchor_tokens, :]
                t5_anchor = t5_teacher[:, :anchor_tokens, :]
                out["t5_l2"] = F.mse_loss(mq_anchor, t5_anchor)

                mq_anchor_flat = mq_anchor.reshape(-1, hidden)
                t5_anchor_flat = t5_anchor.reshape(-1, hidden)
                cos_sim = F.cosine_similarity(mq_anchor_flat, t5_anchor_flat, dim=-1).mean()
                out["t5_cos"] = (1.0 - cos_sim)
            elif align_mode == "gram_cka":
                mq_gram = self._token_gram_matrix(mq_float)
                t5_gram = self._token_gram_matrix(t5_teacher)
                out["t5_gram"] = F.mse_loss(mq_gram, t5_gram)
                out["t5_cka"] = self._linear_cka_loss(mq_float, t5_teacher)
                # 复用旧命名，保持日志/脚本兼容
                out["t5_l2"] = out["t5_gram"]
                out["t5_cos"] = out["t5_cka"]
            elif align_mode == "sinkhorn_ot":
                out["t5_ot"] = self._sinkhorn_ot_cost(
                    mq_float,
                    t5_teacher,
                    epsilon=float(getattr(args, "t5_align_ot_epsilon", 0.05)),
                    iters=int(getattr(args, "t5_align_ot_iters", 25)),
                )
                out["t5_l2"] = out["t5_ot"]
            else:
                raise ValueError(f"Unknown --t5_align_mode: {align_mode}")

            mq_mean = mq_float.mean(dim=1)
            mq_std = mq_float.std(dim=1, unbiased=False)
            t5_mean = t5_teacher.mean(dim=1)
            t5_std = t5_teacher.std(dim=1, unbiased=False)
            out["t5_stats"] = F.mse_loss(mq_mean, t5_mean) + F.mse_loss(mq_std, t5_std)

        if need_img:
            has_ref = torch.tensor(
                [1 if ref is not None else 0 for ref in mq_refs],
                device=device,
                dtype=torch.bool,
            )
            if bool(torch.any(has_ref).item()):
                with torch.no_grad():
                    mq_text_only = self.mq_encoder(captions, None).to(device=device, dtype=torch.float32)
                diff = mq_float[has_ref] - mq_text_only[has_ref]
                # 每样本的 token+channel RMS 距离
                rms = torch.sqrt(torch.mean(diff * diff, dim=(1, 2)) + 1e-8)
                margin = float(args.mq_image_preserve_margin)
                out["image_preserve"] = F.relu(margin - rms).mean()

        out["total"] = (
            float(args.lambda_t5_align_l2) * out["t5_l2"]
            + float(args.lambda_t5_align_cos) * out["t5_cos"]
            + float(args.lambda_t5_align_stats) * out["t5_stats"]
            + float(args.lambda_mq_image_preserve) * out["image_preserve"]
        )
        return out

    def _encode_video(self, video_tensors):
        """VAE 编码视频 → latent"""
        with torch.no_grad():
            # video_tensors: [B, 3, T, H, W] or list of [3, T, H, W]
            latents = []
            for v in video_tensors:
                # v: [3, T, H, W] → VAE expects this format
                z = self.wan.vae.encode([v.to(self.dev_dit, dtype=torch.bfloat16)])
                latents.append(z[0])  # z[0]: [C_z, T', H', W']
        return latents

    def _encode_first_frame(self, first_frame_tensor):
        """VAE 编码参考图第一帧 → i2v condition latent"""
        with torch.no_grad():
            # first_frame: [3, H, W] → [3, 1, H, W]
            ff = first_frame_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
            z = self.wan.vae.encode([ff])
        return z[0]  # [C_z, 1, H', W']

    def _resolve_train_ref_anchor_mode(self) -> str:
        """
        返回当前 batch 实际使用的锚定模式。
        - none / animate_like: 直接使用
        - mixed50: 按 optimizer step 交替 none / animate_like，保证长期约 50/50
        """
        mode = str(getattr(self.args, "train_ref_anchor_mode", "none")).strip().lower()
        if mode in ("none", "animate_like"):
            return mode
        if mode == "mixed50":
            use_animate = (self._train_ref_anchor_mixed_counter % 2 == 1)
            self._train_ref_anchor_mixed_counter += 1
            return "animate_like" if use_animate else "none"
        raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

    def _train_ref_anchor_alpha(self, t_norm: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """
        训练期首帧软锚定系数（0~1）。
        说明：
        - none: 始终 0，不改动训练行为
        - animate_like: 高噪声(早期)强锚定，随后余弦衰减到 0
        """
        if mode is None:
            mode = self._resolve_train_ref_anchor_mode()
        if mode == "none":
            return torch.zeros_like(t_norm, dtype=torch.float32)
        if mode != "animate_like":
            raise ValueError(f"Unknown --train_ref_anchor_mode: {mode}")

        alpha0 = float(getattr(self.args, "train_ref_anchor_alpha0", 0.95))
        warmup_ratio = float(getattr(self.args, "train_ref_anchor_warmup_ratio", 0.35))
        alpha0 = max(0.0, min(1.0, alpha0))
        warmup_ratio = max(0.0, min(1.0, warmup_ratio))
        if warmup_ratio <= 0.0 or alpha0 <= 0.0:
            return torch.zeros_like(t_norm, dtype=torch.float32)

        start_t = 1.0 - warmup_ratio
        alpha = torch.zeros_like(t_norm, dtype=torch.float32)
        active = t_norm >= start_t
        if not torch.any(active):
            return alpha
        u = ((t_norm[active] - start_t) / max(warmup_ratio, 1e-6)).clamp(0.0, 1.0)
        alpha[active] = alpha0 * 0.5 * (1.0 - torch.cos(math.pi * u))
        return alpha

    @staticmethod
    def _frames_to_latent_slots(frame_count: int, stride_t: int) -> int:
        """像素帧数 -> latent 时间槽数（与 VAE 时间下采样保持一致）"""
        f = max(0, int(frame_count))
        if f <= 0:
            return 0
        return int((f - 1) // max(int(stride_t), 1) + 1)

    def _encode_ref_image_to_latent(
        self,
        ref_img: Image.Image | None,
        latent_h: int,
        latent_w: int,
        z_channels: int,
    ) -> torch.Tensor:
        """
        将参考图编码为 1 帧 reference latent。
        若 ref_img 缺失，返回零 reference latent。
        """
        if ref_img is None:
            return torch.zeros(
                z_channels, 1, latent_h, latent_w,
                device=self.dev_dit, dtype=torch.float32,
            )
        target_h = int(latent_h * self.wan_config.vae_stride[1])
        target_w = int(latent_w * self.wan_config.vae_stride[2])
        ref_resized = ref_img.resize((target_w, target_h), Image.LANCZOS)
        ref_np = np.array(ref_resized).astype(np.float32)
        ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1) / 127.5 - 1.0
        ref_5d = ref_tensor.unsqueeze(1).to(self.dev_dit, dtype=torch.bfloat16)
        with torch.no_grad():
            ref_lat = self.wan.vae.encode([ref_5d])[0]
        return ref_lat.float()

    def _compute_wan_func_distill_loss(
        self,
        model_output: List[torch.Tensor],
        x_inputs: List[torch.Tensor],
        timesteps_wan: torch.Tensor,
        max_seq_len: int,
        t5_context: List[torch.Tensor],
        mq_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        函数级蒸馏:
            L_func = MSE( pred_mq(x_t,t), pred_t5(x_t,t) )
        其中 pred_t5 由 frozen Wan + T5 条件生成（teacher no-grad）。
        """
        args = self.args
        mode = str(getattr(args, "wan_func_teacher_mode", "t5_only")).strip().lower()
        if mode not in {"t5_only", "t5_plus_mq"}:
            raise ValueError(f"Unknown --wan_func_teacher_mode: {mode}")

        teacher_context: List[torch.Tensor] = []
        for i, t5_seq in enumerate(t5_context):
            t5_feat = t5_seq.to(self.dev_dit, dtype=torch.bfloat16)
            if mode == "t5_plus_mq":
                mq_feat = mq_features[i].detach().to(self.dev_dit, dtype=torch.bfloat16)
                t5_feat = torch.cat([mq_feat, t5_feat], dim=0)
            teacher_context.append(t5_feat)

        if not teacher_context:
            return mq_features.new_zeros(())

        teacher_text_len = max(int(ctx.shape[0]) for ctx in teacher_context)
        cur_text_len = int(self.wan.model.text_len)
        self.wan.model.text_len = teacher_text_len
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    teacher_output = self.wan.model(
                        x_inputs,
                        t=timesteps_wan,
                        context=teacher_context,
                        seq_len=max_seq_len,
                    )
        finally:
            self.wan.model.text_len = cur_text_len

        loss = 0.0
        valid = 0
        for i in range(len(model_output)):
            pred_mq = model_output[i].float()
            pred_t5 = teacher_output[i].float()
            loss = loss + F.mse_loss(pred_mq, pred_t5)
            valid += 1
        if valid <= 0:
            return mq_features.new_zeros(())
        return loss / valid

    def _compute_loss(self, batch):
        """
        计算一个 batch 的 Flow Matching 损失。

        训练默认使用 t2v 模式 (无第一帧蒙版/无首帧锚定)。
        可通过 --train_ref_anchor_mode 在 x_t 注入 animate-like 首帧软锚定，
        以缓解与 i2v 推理分布不一致问题。
        """
        args = self.args
        captions = batch["caption"]
        videos = batch["video"]         # list of [3, T, H, W]
        mq_refs = batch["mq_ref_image"]  # list of PIL or None
        B = len(captions)
        self._last_loss_denoise = 0.0
        self._last_loss_aux_align_total = 0.0
        self._last_loss_aux_t5_l2 = 0.0
        self._last_loss_aux_t5_cos = 0.0
        self._last_loss_aux_t5_stats = 0.0
        self._last_loss_aux_t5_gram = 0.0
        self._last_loss_aux_t5_cka = 0.0
        self._last_loss_aux_t5_ot = 0.0
        self._last_loss_aux_image_preserve = 0.0
        self._last_loss_aux_wan_func = 0.0
        self._loss_call_count += 1

        # ── 1. MetaQuery 编码 (在 encoder 设备上, 有梯度) ────────────────
        mq_images = []
        for ref in mq_refs:
            if ref is not None:
                mq_images.append([ref])
            else:
                mq_images.append(None)

        all_none = all(img is None for img in mq_images)
        if all_none:
            mq_features = self.mq_encoder(captions, None)
        else:
            for i, img in enumerate(mq_images):
                if img is None:
                    mq_images[i] = [Image.new("RGB", (224, 224))]
            mq_features = self.mq_encoder(captions, mq_images)
        # mq_features: [B, 256, 4096], 有梯度
        mq_features = self._probe_and_optionally_match_mq_norm(
            captions=captions,
            mq_features=mq_features,
            t5_context=None,
        )

        # ── 2. MQ-only 注入 DiT context ─────────────────────────────────
        augmented_context = []
        for i in range(B):
            mq_feat = mq_features[i].to(self.dev_dit, dtype=torch.bfloat16)
            aug = mq_feat
            if i == 0:
                self._verify_train_context_injection_once(mq_feat, aug)
            augmented_context.append(aug)

        # ── 4. VAE 编码视频 → latent (无梯度) ───────────────────────────
        with torch.no_grad():
            latents = self._encode_video(videos)
            # latents: list of [C_z, T', H', W']

        # ── 4. 采样噪声和时间步, 构建 Flow Matching 目标 ─────────────────
        patch_size = self.wan_config.patch_size
        stride_t = int(self.wan_config.vae_stride[0])

        first_frame_condition_enabled = bool(
            getattr(args, "enable_ti2v_first_frame_condition", True)
        )
        mode_cfg = str(getattr(args, "train_video_conditioning_mode", "legacy_t2v")).strip().lower()
        if mode_cfg not in ("legacy_t2v", "wan_animate_slot"):
            mode_cfg = "legacy_t2v"
        effective_video_mode = mode_cfg if first_frame_condition_enabled else "mq_only"

        x_inputs = []
        timestep_rows = []
        target_list = []
        prefix_slots_list = []
        target_slots_list = []
        max_seq_len = 0
        ref_anchor_alpha_sum = 0.0
        ref_anchor_applied = 0
        ref_anchor_mode_effective = "none"

        for i, lat in enumerate(latents):
            C, T, H, W = lat.shape
            lat = lat.float()
            x0_for_fm = lat
            prefix_slots_i = 0

            # Wan 侧首帧条件：在训练阶段将参考图转换为 latent 并注入
            ref_lat = None
            if first_frame_condition_enabled:
                ref_lat = self._encode_ref_image_to_latent(
                    mq_refs[i],
                    latent_h=H,
                    latent_w=W,
                    z_channels=C,
                ).to(self.dev_dit, dtype=torch.float32)

            if effective_video_mode == "wan_animate_slot":
                ref_slots = self._frames_to_latent_slots(
                    int(getattr(args, "train_animate_ref_frames", 1)),
                    stride_t=stride_t,
                )
                temporal_slots = self._frames_to_latent_slots(
                    int(getattr(args, "train_animate_temporal_frames", 0)),
                    stride_t=stride_t,
                )
                conditional_slots = self._frames_to_latent_slots(
                    int(getattr(args, "train_animate_conditional_frames", 0)),
                    stride_t=stride_t,
                )
                ref_slots = max(0, int(ref_slots))
                temporal_slots = max(0, int(temporal_slots))
                conditional_slots = max(0, int(conditional_slots))
                prefix_slots_i = ref_slots + temporal_slots + conditional_slots
                if prefix_slots_i > 0:
                    prefix_chunks = []
                    if ref_slots > 0:
                        if ref_lat is None:
                            ref_prefix = torch.zeros(
                                C, ref_slots, H, W, device=self.dev_dit, dtype=torch.float32
                            )
                        else:
                            ref_prefix = ref_lat.repeat(1, ref_slots, 1, 1)
                        prefix_chunks.append(ref_prefix)
                    if temporal_slots > 0:
                        prefix_chunks.append(
                            torch.zeros(
                                C, temporal_slots, H, W, device=self.dev_dit, dtype=torch.float32
                            )
                        )
                    if conditional_slots > 0:
                        prefix_chunks.append(
                            torch.zeros(
                                C, conditional_slots, H, W, device=self.dev_dit, dtype=torch.float32
                            )
                        )
                    x0_prefix = torch.cat(prefix_chunks, dim=1)
                    x0_for_fm = torch.cat([x0_prefix, lat], dim=1)

            T_full = int(x0_for_fm.shape[1])
            tokens_per_frame = int(math.ceil((H * W) / (patch_size[1] * patch_size[2])))
            seq_len_i = int(tokens_per_frame * T_full)
            max_seq_len = max(max_seq_len, seq_len_i)

            t_val = torch.rand(1, device=self.dev_dit, dtype=torch.float32)
            noise = torch.randn_like(x0_for_fm, dtype=torch.float32)

            # Flow matching: x_t = (1-t) * x_0 + t * noise
            sigma = t_val.view(-1, 1, 1, 1)
            noisy_lat = (1.0 - sigma) * x0_for_fm + sigma * noise

            if effective_video_mode == "legacy_t2v" and ref_lat is not None:
                ref_mode = str(getattr(self, "_current_train_ref_anchor_mode", "none")).strip().lower()
                if ref_mode not in ("none", "animate_like"):
                    ref_mode = self._resolve_train_ref_anchor_mode()
                alpha_tensor = self._train_ref_anchor_alpha(t_val, mode=ref_mode)
                alpha_scalar = float(alpha_tensor.item())
                ref_anchor_mode_effective = ref_mode
                if alpha_scalar > 0.0:
                    noisy_lat[:, :1] = (1.0 - alpha_scalar) * noisy_lat[:, :1] + alpha_scalar * ref_lat
                    ref_anchor_alpha_sum += alpha_scalar
                    ref_anchor_applied += 1

            # 目标: noise - x_0 (velocity)
            velocity = noise - x0_for_fm

            # token 级 timestep：MQ-only 下全部 token 共享 t
            t_scalar = float((t_val * self.wan.num_train_timesteps).item())
            t_row = torch.full((seq_len_i,), t_scalar, device=self.dev_dit, dtype=torch.float32)
            if (
                effective_video_mode == "wan_animate_slot"
                and prefix_slots_i > 0
                and bool(getattr(args, "train_animate_preserve_timestep_zero", True))
            ):
                prefix_token_count = min(seq_len_i, int(prefix_slots_i * tokens_per_frame))
                if prefix_token_count > 0:
                    t_row[:prefix_token_count] = 0.0

            x_inputs.append(noisy_lat)
            target_list.append(velocity)
            prefix_slots_list.append(prefix_slots_i)
            timestep_rows.append(t_row)
            target_slots_list.append(T)

        # 拼接 timestep → [B, max_seq_len]
        padded_rows = []
        for row in timestep_rows:
            pad_len = max_seq_len - int(row.numel())
            if pad_len > 0:
                pad_val = float(row[-1].item()) if row.numel() > 0 else 0.0
                row = torch.cat([row, row.new_full((pad_len,), pad_val)], dim=0)
            padded_rows.append(row)
        timesteps_wan = torch.stack(padded_rows, dim=0).to(self.dev_dit)

        self._last_train_ref_anchor_alpha_mean = (
            float(ref_anchor_alpha_sum / ref_anchor_applied) if ref_anchor_applied > 0 else 0.0
        )
        self._last_train_ref_anchor_applied = int(ref_anchor_applied)
        self._last_train_ref_anchor_effective_mode = (
            ref_anchor_mode_effective if ref_anchor_applied > 0 else "none"
        )
        self._last_train_video_conditioning_mode = str(effective_video_mode)
        self._last_train_prefix_latent_slots = int(
            round(sum(prefix_slots_list) / max(len(prefix_slots_list), 1))
        )
        self._last_train_target_latent_slots = int(round(sum(target_slots_list) / max(len(target_slots_list), 1)))
        self._last_train_prefix_loss_dropped = 0

        # ── 5. MQ-only text_len + DiT forward ───────────────────────────
        orig_text_len = self.wan.model.text_len
        self.wan.model.text_len = self._aug_text_len

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model_output = self.wan.model(
                    x_inputs,
                    t=timesteps_wan,
                    context=augmented_context,
                    seq_len=max_seq_len,
                )

            # ── 6. 计算去噪主损失 ──────────────────────────────────────────
            denoise_loss = 0.0
            valid_terms = 0
            drop_prefix_loss = bool(getattr(args, "train_animate_drop_prefix_loss", True))
            dropped_prefix_terms = 0
            for i in range(B):
                pred = model_output[i].float()
                target = target_list[i]
                prefix_slots_i = int(prefix_slots_list[i]) if i < len(prefix_slots_list) else 0
                if (
                    effective_video_mode == "wan_animate_slot"
                    and drop_prefix_loss
                    and prefix_slots_i > 0
                ):
                    if pred.shape[1] <= prefix_slots_i or target.shape[1] <= prefix_slots_i:
                        continue
                    pred = pred[:, prefix_slots_i:, ...]
                    target = target[:, prefix_slots_i:, ...]
                    dropped_prefix_terms += 1
                loss = F.mse_loss(pred, target)
                denoise_loss += loss
                valid_terms += 1
            if valid_terms <= 0:
                raise RuntimeError("无有效训练样本参与损失计算")
            denoise_loss = denoise_loss / valid_terms
            self._last_train_prefix_loss_dropped = int(dropped_prefix_terms)

            # 新版训练目标：仅保留原始去噪主损失（ground-truth latent velocity vs predicted velocity）
            total_loss = denoise_loss
            self._last_loss_denoise = float(denoise_loss.detach().item())
            self._last_loss_aux_align_total = 0.0
            self._last_loss_aux_t5_l2 = 0.0
            self._last_loss_aux_t5_cos = 0.0
            self._last_loss_aux_t5_stats = 0.0
            self._last_loss_aux_t5_gram = 0.0
            self._last_loss_aux_t5_cka = 0.0
            self._last_loss_aux_t5_ot = 0.0
            self._last_loss_aux_image_preserve = 0.0
            self._last_loss_aux_wan_func = 0.0

        finally:
            self.wan.model.text_len = orig_text_len

        return total_loss

    def train(self):
        """主训练循环。"""
        args = self.args
        self._audit_runtime_trainability(stage="train_start")

        # 设置随机种子
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # 数据集（已完全收敛到 WanVideoDataset）
        if WanDatasetClass is None:
            raise RuntimeError("未能导入 WanVideoDataset，请检查 train_connector_for_wan.py 及其依赖")

        dataset = WanDatasetClass(
            seed=args.seed,
            frame_num=args.frame_num,
            max_area=args.max_area,
            null_caption_prob=args.null_caption_prob,
            null_image_prob=args.null_image_prob,
            max_caption_tokens=args.max_caption_tokens,
            caption_tokenizer_path=args.caption_tokenizer_path,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            local_openvid_video_root=args.local_openvid_video_root,
            local_openvid_csv_path=args.local_openvid_csv_path,
            local_openvid_limit=args.local_openvid_limit,
            local_openvid_hd_video_root=args.local_openvid_hd_video_root,
            local_openvid_hd_csv_path=args.local_openvid_hd_csv_path,
            local_openvid_hd_limit=args.local_openvid_hd_limit,
            local_video_cache_dir=args.local_video_cache_dir,
        )

        if len(dataset) == 0:
            raise RuntimeError("数据集为空！检查路径和 JSON 文件。")

        # 由于视频尺寸可能不同, 使用 batch_size=1 避免 collate 问题
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

        # 训练循环
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = Path(args.output_dir).expanduser().resolve()
        if not self._metrics_jsonl_path:
            self._metrics_jsonl_path = str((output_dir / "logs" / "train_metrics.jsonl").expanduser().resolve())
        args.output_dir = str(output_dir)
        args.metrics_jsonl_path = self._metrics_jsonl_path
        self._train_wall_start = time.perf_counter()

        # 训练前快照（用于 verify_metaquery_chain before vs after）
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
        early_stop_triggered = False
        early_stop_reason = ""
        early_stop_ckpt_path = ""
        data_iter = iter(dataloader)

        pbar = tqdm(total=args.num_train_steps, desc="Training")
        self.optimizer.zero_grad(set_to_none=True)

        while step < args.num_train_steps:
            step_wall_start = time.perf_counter()
            accum_loss = 0.0
            accum_denoise_loss = 0.0
            accum_align_loss = 0.0
            accum_align_t5_l2 = 0.0
            accum_align_t5_cos = 0.0
            accum_align_t5_stats = 0.0
            accum_align_t5_gram = 0.0
            accum_align_t5_cka = 0.0
            accum_align_t5_ot = 0.0
            accum_align_img = 0.0
            accum_align_wan_func = 0.0
            skip_optimizer_step = False
            had_fatal_cuda_error = False
            backward_ok = 0
            skip_reason = ""
            self._current_train_ref_anchor_mode = self._resolve_train_ref_anchor_mode()

            for accum_step in range(args.gradient_accumulation_steps):
                # 获取 batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                try:
                    loss = self._compute_loss(batch)
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    self._log_grad_health_once()
                    accum_loss += loss.item()
                    scale = 1.0 / max(float(args.gradient_accumulation_steps), 1.0)
                    accum_denoise_loss += float(self._last_loss_denoise) * scale
                    accum_align_loss += float(self._last_loss_aux_align_total) * scale
                    accum_align_t5_l2 += float(self._last_loss_aux_t5_l2) * scale
                    accum_align_t5_cos += float(self._last_loss_aux_t5_cos) * scale
                    accum_align_t5_stats += float(self._last_loss_aux_t5_stats) * scale
                    accum_align_t5_gram += float(self._last_loss_aux_t5_gram) * scale
                    accum_align_t5_cka += float(self._last_loss_aux_t5_cka) * scale
                    accum_align_t5_ot += float(self._last_loss_aux_t5_ot) * scale
                    accum_align_img += float(self._last_loss_aux_image_preserve) * scale
                    accum_align_wan_func += float(self._last_loss_aux_wan_func) * scale
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
                    if isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in err.lower()):
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
                    # 其他异常也跳过本 step，避免残缺梯度进入 optimizer.step
                    skip_optimizer_step = True
                    skip_reason = "error"
                    self.optimizer.zero_grad(set_to_none=True)
                    break
                    continue

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

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._all_trainable_params(),
                args.max_grad_norm,
            )

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
            effective_samples = int(max(backward_ok, 0) * max(args.batch_size, 1))
            samples_per_sec = float(effective_samples / step_time)

            metrics = {
                "train/loss_step": float(accum_loss),
                "train/loss_ema": float(running_loss),
                "train/loss_denoise": float(accum_denoise_loss),
                "train/loss_align_total": float(accum_align_loss),
                "train/loss_align_t5_l2": float(accum_align_t5_l2),
                "train/loss_align_t5_cos": float(accum_align_t5_cos),
                "train/loss_align_t5_stats": float(accum_align_t5_stats),
                "train/loss_align_t5_gram": float(accum_align_t5_gram),
                "train/loss_align_t5_cka": float(accum_align_t5_cka),
                "train/loss_align_t5_ot": float(accum_align_t5_ot),
                "train/loss_align_img_preserve": float(accum_align_img),
                "train/loss_align_wan_func": float(accum_align_wan_func),
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
                "train/ref_anchor_alpha_mean": float(self._last_train_ref_anchor_alpha_mean),
                "train/ref_anchor_applied": int(self._last_train_ref_anchor_applied),
                "train/ref_anchor_mode_cfg": str(getattr(args, "train_ref_anchor_mode", "none")),
                "train/ref_anchor_mode_effective": str(self._last_train_ref_anchor_effective_mode),
                "train/ref_anchor_effective_is_animate": int(self._last_train_ref_anchor_effective_mode == "animate_like"),
                "train/video_conditioning_mode_cfg": str(getattr(args, "dit_condition_mode", "mq_only")),
                "train/video_conditioning_mode_effective": str(self._last_train_video_conditioning_mode),
                "train/prefix_latent_slots": int(self._last_train_prefix_latent_slots),
                "train/target_latent_slots": int(self._last_train_target_latent_slots),
                "train/prefix_loss_dropped": int(self._last_train_prefix_loss_dropped),
                "train/mq_rms": float(self._last_mq_rms),
                "train/t5_rms_probe": float(self._last_t5_rms),
                "train/mq_t5_rms_ratio": float(self._last_mq_t5_rms_ratio),
                "train/mq_norm_warn": int(self._last_mq_norm_warn_flag),
                "train/mq_norm_match_scale": float(self._last_mq_norm_match_scale),
            }
            metrics.update(self._collect_trainability_metrics())
            metrics.update(self._collect_cuda_memory_metrics())

            should_log = bool(args.log_every_step or (step % args.log_steps == 0))
            should_wandb_log = bool(
                self.wandb_run is not None and (args.wandb_log_every_step or should_log)
            )

            # 日志
            if should_log:
                pbar.set_postfix({
                    "loss": f"{accum_loss:.4f}",
                    "denoise": f"{accum_denoise_loss:.4f}",
                    "align": f"{accum_align_loss:.4f}",
                    "func": f"{accum_align_wan_func:.4f}",
                    "avg": f"{running_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "grad": f"{grad_norm_value:.2f}",
                    "dP": f"{metrics['train/param_sample_abs_delta_mean']:.3e}",
                })
                print(
                    f"\n[Step {step}/{args.num_train_steps}] "
                    f"loss={accum_loss:.4f} denoise={accum_denoise_loss:.4f} align={accum_align_loss:.4f} func={accum_align_wan_func:.4f} "
                    f"avg={running_loss:.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm_value:.2f} "
                    f"dt={step_time:.2f}s samp/s={samples_per_sec:.2f} "
                    f"param_delta={metrics['train/param_sample_abs_delta_mean']:.3e} "
                    f"skip(oom/err/total)={self._oom_skip_count}/{self._error_skip_count}/{self._skipped_step_count}"
                )
            if should_wandb_log:
                self.wandb.log(metrics, step=step)
            self._append_metrics_jsonl(metrics)
            self._record_metrics(metrics)

            # 保存
            if step % args.save_steps == 0:
                self._save_checkpoint(output_dir / f"checkpoint-{step}", step)

            pbar.update(1)

            step_loss_for_early_stop = float(accum_denoise_loss)
            if (
                bool(getattr(args, "enable_loss_early_stop", False))
                and step >= int(getattr(args, "loss_early_stop_min_step", 800))
                and step_loss_for_early_stop < float(getattr(args, "loss_early_stop_threshold", 0.25))
            ):
                early_stop_triggered = True
                early_stop_reason = (
                    f"train/loss_denoise={step_loss_for_early_stop:.6f} < {float(args.loss_early_stop_threshold):.6f} "
                    f"at step={int(step)}"
                )
                early_stop_ckpt_path = str(
                    output_dir / f"checkpoint-earlystop-step{int(step)}-denoise{step_loss_for_early_stop:.4f}"
                )
                self._save_checkpoint(
                    early_stop_ckpt_path,
                    step,
                    extra_info={
                        "early_stop": True,
                        "early_stop_metric": "train/loss_denoise",
                        "early_stop_loss": step_loss_for_early_stop,
                        "early_stop_threshold": float(args.loss_early_stop_threshold),
                        "early_stop_min_step": int(args.loss_early_stop_min_step),
                    },
                )
                if self.is_main_process:
                    print(f"[EARLY-STOP] 已触发: {early_stop_reason}")
                    print(f"[EARLY-STOP] checkpoint: {early_stop_ckpt_path}")
                break

        pbar.close()

        # 最终保存
        final_ckpt_path = str(output_dir / "checkpoint-final")
        final_extra_info = None
        if early_stop_triggered:
            final_extra_info = {
                "early_stop": True,
                "early_stop_reason": early_stop_reason,
                "early_stop_checkpoint_path": early_stop_ckpt_path,
            }
        self._save_checkpoint(final_ckpt_path, step, extra_info=final_extra_info)
        self._write_training_chain_manifest(output_dir, final_checkpoint_path=final_ckpt_path, final_step=step)
        if early_stop_triggered and self.is_main_process:
            print(f"[EARLY-STOP] 训练提前结束，最终步数: {step}")
        print(f"\n✅ 训练完成！最终 checkpoint: {final_ckpt_path}")
        if self.wandb_run is not None:
            self.wandb.finish()

    def _save_checkpoint(self, path, step, extra_info: Dict[str, Any] | None = None):
        """保存 MQ 编码器 +（可选）Wan DiT 可训练子集（兼容增强格式）"""
        path = Path(path).expanduser().resolve()
        module = self._mq_encoder_module()
        wan_state_cpu, wan_lora_state_cpu, wan_export_info = self._collect_wan_trainable_state_for_checkpoint()
        if not self.is_main_process:
            return
        ckpt_info = save_mq_checkpoint_bundle(
            path=path,
            module=module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            args=self.args,
            wan_module=None,
            wan_trainable_state_cpu=wan_state_cpu,
            wan_lora_state_cpu=wan_lora_state_cpu,
            wan_lora_config=build_lora_config_dict(
                enabled=self._wan_lora_enabled(),
                rank=int(getattr(self.args, "wan_lora_rank", 16)),
                alpha=float(getattr(self.args, "wan_lora_alpha", 16.0)),
                dropout=float(getattr(self.args, "wan_lora_dropout", 0.0)),
                targets=getattr(self.args, "wan_lora_targets", "self_attn,cross_attn,ffn"),
                module_names=self._wan_lora_module_names,
            ) if self._wan_lora_enabled() else None,
            wan_train_mode=str(getattr(self, "_effective_wan_train_mode", "frozen")),
            metrics_tail=self._metrics_history[-200:],
            metrics_summary=self._build_metrics_summary(step=step),
            extra_info={
                "before_checkpoint_path": self._train_before_checkpoint_path,
                "metrics_jsonl_path": self._metrics_jsonl_path,
                "wan_train_mode_effective": str(getattr(self, "_effective_wan_train_mode", "frozen")),
                "wan_trainable_tensor_count": int(len(getattr(self, "_wan_trainable_names", []))),
                "wan_trainable_name_preview": list(getattr(self, "_wan_trainable_names", [])[:64]),
                "wan_lora_module_count": int(len(getattr(self, "_wan_lora_module_names", []))),
                "wan_lora_module_preview": list(getattr(self, "_wan_lora_module_names", [])[:64]),
                "wan_lora_extra_trainable_count": int(len(getattr(self, "_wan_lora_extra_trainable_names", []))),
                "wan_lora_extra_trainable_preview": list(getattr(self, "_wan_lora_extra_trainable_names", [])[:64]),
                **wan_export_info,
                **(extra_info or {}),
            },
        )
        print(f"  💾 Checkpoint 已保存: {ckpt_info['path']}")
        if self.wandb_run is not None and self.args.wandb_log_checkpoint:
            self.wandb.log(
                {
                    "checkpoint/step": int(step),
                    "checkpoint/path": str(ckpt_info["path"]),
                },
                step=step,
            )

    @staticmethod
    def _normalize_wan_state_key(name: str) -> str:
        key = str(name)
        while "_fsdp_wrapped_module." in key:
            key = key.replace("_fsdp_wrapped_module.", "")
        return key

    def _collect_wan_trainable_state_for_checkpoint(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        导出可用于推理端加载的 Wan 可训练权重子集。

        关键点：
        - 非 FSDP: 直接按 requires_grad 导出。
        - FSDP: 使用 FULL_STATE_DICT（所有 rank 参与）导出原始参数名，避免只保存 _flat_param。
        """
        wan_model = getattr(self.wan, "model", None)
        mode = str(getattr(self, "_effective_wan_train_mode", "frozen")).strip().lower()
        info: Dict[str, Any] = {
            "wan_export_mode": "none",
            "wan_export_model_mode": mode,
            "wan_export_world_size": int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1,
        }
        if wan_model is None or not isinstance(wan_model, nn.Module) or mode == "frozen":
            return {}, {}, info

        if mode == "lora":
            direct_state_cpu: Dict[str, torch.Tensor] = {}
            for name, p in wan_model.named_parameters():
                if not p.requires_grad:
                    continue
                norm_name = self._normalize_wan_state_key(name)
                lname = norm_name.lower()
                if ".lora_a" in lname or ".lora_b" in lname:
                    continue
                direct_state_cpu[norm_name] = p.detach().cpu().contiguous()
            lora_state_cpu = collect_lora_state_dict(wan_model)
            info["wan_export_mode"] = "lora_plus_named_parameters"
            info["wan_export_tensor_count"] = int(len(direct_state_cpu))
            info["wan_export_param_count"] = int(sum(int(t.numel()) for t in direct_state_cpu.values()))
            info["wan_export_lora_tensor_count"] = int(len(lora_state_cpu))
            info["wan_export_lora_param_count"] = int(sum(int(t.numel()) for t in lora_state_cpu.values()))
            info["wan_export_has_flat_param_key"] = 0
            return direct_state_cpu, lora_state_cpu, info

        fsdp_cls = None
        fsdp_full_cfg = None
        fsdp_state_type = None
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as _TorchFSDP
            from torch.distributed.fsdp import FullStateDictConfig as _FullStateDictConfig
            from torch.distributed.fsdp import StateDictType as _StateDictType

            fsdp_cls = _TorchFSDP
            fsdp_full_cfg = _FullStateDictConfig
            fsdp_state_type = _StateDictType
        except Exception:
            fsdp_cls = None

        has_fsdp = bool(fsdp_cls is not None and any(isinstance(m, fsdp_cls) for m in wan_model.modules()))
        dist_ready = bool(dist.is_available() and dist.is_initialized())

        if has_fsdp and dist_ready and fsdp_cls is not None and fsdp_full_cfg is not None and fsdp_state_type is not None:
            if self.is_main_process:
                print("[WAN-SAVE] FSDP detected, exporting portable FULL_STATE_DICT ...")
            cfg = fsdp_full_cfg(offload_to_cpu=True, rank0_only=True)
            try:
                with fsdp_cls.state_dict_type(wan_model, fsdp_state_type.FULL_STATE_DICT, cfg):
                    full_state = wan_model.state_dict()
            except Exception as e:
                info["wan_export_mode"] = "fsdp_full_state_failed_fallback_named_parameters"
                info["wan_export_error"] = str(e)
            else:
                if not self.is_main_process:
                    info["wan_export_mode"] = "fsdp_full_state_non_main_rank"
                    return {}, {}, info

                state_cpu: Dict[str, torch.Tensor] = {}
                for name, tensor in full_state.items():
                    if not torch.is_tensor(tensor):
                        continue
                    norm_name = self._normalize_wan_state_key(name)
                    state_cpu[norm_name] = tensor.detach().cpu().contiguous()

                if mode == "cond_only":
                    kws = self._wan_cond_keywords()
                    state_cpu = {
                        n: t for n, t in state_cpu.items()
                        if any(kw in n.lower() for kw in kws)
                    }

                info["wan_export_mode"] = "fsdp_full_state"
                info["wan_export_tensor_count"] = int(len(state_cpu))
                info["wan_export_param_count"] = int(sum(int(t.numel()) for t in state_cpu.values()))
                info["wan_export_has_flat_param_key"] = int(any("_flat_param" in n for n in state_cpu.keys()))
                return state_cpu, {}, info

        # 非 FSDP 或 FSDP full_state 导出失败时的兜底路径
        state_cpu = {}
        for name, p in wan_model.named_parameters():
            if not p.requires_grad:
                continue
            norm_name = self._normalize_wan_state_key(name)
            state_cpu[norm_name] = p.detach().cpu().contiguous()

        info["wan_export_mode"] = "named_parameters"
        info["wan_export_tensor_count"] = int(len(state_cpu))
        info["wan_export_param_count"] = int(sum(int(t.numel()) for t in state_cpu.values()))
        info["wan_export_has_flat_param_key"] = int(any("_flat_param" in n for n in state_cpu.keys()))
        return state_cpu, {}, info

    @staticmethod
    def _collate_fn(batch):
        """自定义 collate: 不 stack 不同尺寸的 tensor"""
        result = {}
        for key in batch[0].keys():
            result[key] = [item[key] for item in batch]
        return result


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    trainer = MetaQueryWanTrainer(args)
    trainer.train()
