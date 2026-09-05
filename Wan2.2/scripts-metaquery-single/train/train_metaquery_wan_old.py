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
from transformers import AutoTokenizer

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

    # ── 数据 ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root", type=str,
                   default="/home/liuzhirui/model/SCAIL/dataset/output_tom_and_jerry_720x1280",
                   help="数据集根目录 (含多个 S*E* 子目录)")
    p.add_argument("--caption_json_root", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/dataset/Tom_and_Jerry_720x1280",
                   help="caption JSON 所在根目录")
    p.add_argument("--manifest_path", type=str, default=None,
                   help="统一清单文件路径(JSON/JSONL), 提供后优先使用")
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
    p.add_argument("--probe_missing_meta", action="store_true",
                   help="若缺失时长/帧数信息则探测视频元数据")

    # ── 训练参数 ──────────────────────────────────────────────────────────
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)

    # ── MetaQuery ─────────────────────────────────────────────────────────
    p.add_argument("--num_metaqueries", type=int, default=256)
    p.add_argument("--connector_num_hidden_layers", type=int, default=24)
    p.add_argument("--null_caption_prob", type=float, default=0.1)
    p.add_argument("--null_image_prob", type=float, default=0.1)

    # ── 设备 ──────────────────────────────────────────────────────────────
    p.add_argument("--dit_device", type=int, default=0,
                   help="DiT + VAE + T5 所在 GPU")
    p.add_argument("--encoder_device", type=int, default=1,
                   help="Qwen3-VL + Connector 所在 GPU")
    p.add_argument("--resume_mq_encoder_path", type=str, default=None,
                   help="从已有mq_encoder权重继续训练")

    return p.parse_args()


# =============================================================================
# MetaQuery Encoder (直接输出 Wan text_dim=4096, 无需 to_wan_proj)
# =============================================================================
class MetaQueryEncoderForWan(nn.Module):
    """
    Qwen3-VL → Connector → dim=4096 (直接匹配 Wan text_dim)

    与之前 MetaQueryEncoder 的关键区别:
    - Connector 最终输出维度直接设为 4096 (而非 Sana 的 2240)
    - 不需要额外的 to_wan_proj 层
    - Connector 从零初始化, 会和 Wan DiT 一起训练对齐
    """

    WAN_TEXT_DIM = 4096

    def __init__(
        self,
        qwen3vl_model_id: str,
        num_metaqueries: int = 256,
        connector_num_hidden_layers: int = 24,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_metaqueries = num_metaqueries
        self.wan_text_dim = self.WAN_TEXT_DIM
        self.dtype = dtype
        self.device = torch.device(device)

        print("=" * 60)
        print(f"[MetaQueryEncoderForWan] 初始化")
        print(f"  Qwen3-VL : {qwen3vl_model_id}")
        print(f"  MQ tokens: {num_metaqueries}")
        print(f"  目标 dim : {self.wan_text_dim} (Wan text_dim, 无 to_wan_proj)")
        print("=" * 60)

        # ── 加载 MLLMInContext ────────────────────────────────────────────
        from models.model import MLLMInContextConfig, MLLMInContext

        # 仅加载 MLLM 主干和 tokenizer，不再加载任何扩散模型权重。
        # connector 输出维度直接指定为 Wan 所需的 4096。
        config = MLLMInContextConfig(
            mllm_id=qwen3vl_model_id,
            diffusion_model_id="none",
            connector_out_dim_override=self.wan_text_dim,
            num_metaqueries=num_metaqueries,
            _gradient_checkpointing=False,
            connector_num_hidden_layers=connector_num_hidden_layers,
        )

        mllm_model = MLLMInContext(config)
        mllm_model = mllm_model.to(device=self.device, dtype=dtype)

        self.mllm_model = mllm_model
        self.tokenizer = mllm_model.get_tokenizer()
        self.tokenize = mllm_model.get_tokenize_fn()

        # 记录当前 connector 输出维度（此处已由 override 设为 4096）
        self._orig_connector_out_dim = mllm_model.connector_out_dim
        mllm_hidden_size = mllm_model.mllm_hidden_size

        print(f"  MLLM hidden: {mllm_hidden_size}")
        print(f"  原始 Connector out: {self._orig_connector_out_dim}")

        # ── 替换 Connector: 直接输出 Wan text_dim=4096 ──────────────────
        # 原始 connector: Qwen2Encoder(1536) → Linear(1536→2240) → GELU → Linear(2240→2240) → RMSNorm(2240)
        # 新 connector:   Qwen2Encoder(1536) → Linear(1536→4096) → GELU → Linear(4096→4096) → RMSNorm(4096)
        from models.transformer_encoder import Qwen2Encoder
        from transformers import Qwen2Config
        from diffusers.models.normalization import RMSNorm

        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=mllm_hidden_size,
                intermediate_size=mllm_hidden_size * 4,
                num_hidden_layers=connector_num_hidden_layers,
                num_attention_heads=mllm_hidden_size // 64,
                num_key_value_heads=mllm_hidden_size // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )

        input_scale = math.sqrt(5.5)  # 与 MetaQuery 原始训练一致
        norm = RMSNorm(self.wan_text_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(input_scale)

        new_connector = nn.Sequential(
            encoder,
            nn.Linear(mllm_hidden_size, self.wan_text_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.wan_text_dim, self.wan_text_dim),
            norm,
        ).to(device=self.device, dtype=dtype)

        # 替换 mllm_model 内部的 connector
        self.mllm_model.connector = new_connector
        self.mllm_model.connector_out_dim = self.wan_text_dim

        print(f"  新 Connector out: {self.wan_text_dim} (直接匹配 Wan)")

        # ── 冻结 MLLM backbone, 只有 MQ embeddings 和 Connector 可训练 ───
        self.mllm_model.mllm_backbone.requires_grad_(False)

        # 解冻 embed_tokens (MQ token 部分通过 freeze_hook 控制)
        embed_tokens = self.mllm_model.mllm_backbone.get_input_embeddings()
        embed_tokens.requires_grad_(True)

        # Connector 可训练
        self.mllm_model.connector.requires_grad_(True)

        # ── 删除不需要的 transformer ─────────────────────────────────────
        if hasattr(self.mllm_model, 'transformer') and self.mllm_model.transformer is not None:
            del self.mllm_model.transformer
            self.mllm_model.transformer = None
            torch.cuda.empty_cache()

        # ── 统计 ─────────────────────────────────────────────────────────
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  总参数: {total_params / 1e6:.1f}M")
        print(f"  可训练: {trainable_params / 1e6:.1f}M")
        print("[MetaQueryEncoderForWan] ✅ 初始化完成\n")

    def get_trainable_params(self):
        """返回所有可训练参数 (Connector + MQ Embeddings)"""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, captions, input_images=None):
        """
        前向传播 (训练模式, 保留梯度)

        Args:
            captions: List[str], 长度 B
            input_images: List[List[PIL.Image]] 或 None

        Returns:
            Tensor [B, num_metaqueries, 4096]
        """
        # ── 分词 ─────────────────────────────────────────────────────────
        if input_images is not None:
            input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
                self.tokenizer, captions, input_images
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device, self.dtype)
                if self.mllm_model.mllm_type in ("qwenvl", "qwen3vl"):
                    pixel_values = pixel_values.squeeze(0)
            if image_sizes is not None:
                image_sizes = image_sizes.to(self.device)
        else:
            input_ids, attention_mask = self.tokenize(
                self.tokenizer, captions
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values = None
            image_sizes = None

        # ── MLLM forward ─────────────────────────────────────────────────
        mq_features, mq_mask = self.mllm_model.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        # mq_features: [B, num_metaqueries, 4096] — 已经过 Connector

        return mq_features

    @torch.no_grad()
    def encode(self, captions, input_images=None):
        """推理模式 (无梯度)"""
        return self.forward(captions, input_images)


# =============================================================================
# 数据集
# =============================================================================
class TomAndJerryVideoDataset(Dataset):
    """
    加载 Tom & Jerry 视频数据集。

    每条数据:
        - segment_video_path: 视频片段路径
        - character_crop_path: 角色裁剪图 (作为参考图 / MetaQuery 输入)
        - caption: 文本描述
    """

    def __init__(
        self,
        caption_json_root: str,
        manifest_path: str = None,
        frame_num: int = 41,
        max_area: int = 480 * 832,
        null_caption_prob: float = 0.1,
        null_image_prob: float = 0.1,
        max_caption_tokens: int = 512,
        caption_tokenizer_path: str = "google/umt5-xxl",
        min_duration_sec: float = 0.5,
        max_duration_sec: float = 20.0,
        probe_missing_meta: bool = False,
    ):
        self.frame_num = frame_num
        self.max_area = max_area
        self.null_caption_prob = null_caption_prob
        self.null_image_prob = null_image_prob
        self.max_caption_tokens = max_caption_tokens
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.probe_missing_meta = probe_missing_meta
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained(caption_tokenizer_path)

        if manifest_path:
            self._load_from_manifest(manifest_path)
        else:
            caption_root = Path(caption_json_root)
            json_files = sorted(caption_root.rglob("captions_output.json"))

            for jf in json_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        items = json.load(f)
                except Exception as e:
                    print(f"[Dataset] 跳过 {jf}: {e}")
                    continue

                seen = {}
                for item in items:
                    vp = item.get("segment_video_path") or item.get("video_path", "")
                    cp = item.get("character_crop_path", "")
                    cap = item.get("caption", "")
                    if not vp or not cap:
                        continue
                    key = (vp, cp)
                    if key not in seen or len(cap) > len(seen[key]["caption"]):
                        seen[key] = {
                            "video_path": vp,
                            "ref_image_path": cp,
                            "caption": cap,
                            "fps": item.get("fps", None),
                            "frame_count": item.get("frame_count", None),
                            "duration": item.get("duration", None),
                        }
                self.samples.extend(seen.values())

        self._apply_filters()

        print(f"[Dataset] 加载 {len(self.samples)} 条样本 from {caption_json_root}")

    def _load_from_manifest(self, manifest_path: str):
        mp = Path(manifest_path)
        items = []
        if mp.suffix.lower() == ".jsonl":
            with open(mp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        else:
            with open(mp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "samples" in data:
                items = data["samples"]
            else:
                raise ValueError(f"manifest格式不支持: {manifest_path}")

        for item in items:
            vp = item.get("video_path") or item.get("segment_video_path")
            cp = item.get("ref_image_path") or item.get("character_crop_path", "")
            cap = item.get("caption", "")
            if not vp or not cap:
                continue
            self.samples.append({
                "video_path": vp,
                "ref_image_path": cp,
                "caption": cap,
                "fps": item.get("fps", None),
                "frame_count": item.get("frame_count", None),
                "duration": item.get("duration_sec", item.get("duration", None)),
            })

    def _caption_token_len(self, text: str) -> int:
        ids = self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        return len(ids)

    @staticmethod
    def _probe_video_meta(video_path: str):
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps is None or fps <= 0:
            return None, None, None
        return float(fps), int(frame_count), float(frame_count) / float(fps)

    def _apply_filters(self):
        filtered = []
        drop_long = 0
        drop_short = 0
        drop_long_duration = 0
        drop_too_few_frames = 0
        drop_missing_video = 0
        for s in self.samples:
            cap_len = self._caption_token_len(s["caption"])
            if cap_len > self.max_caption_tokens:
                drop_long += 1
                continue

            fps = s.get("fps", None)
            frame_count = s.get("frame_count", None)
            duration = s.get("duration", None)
            if (duration is None or frame_count is None) and self.probe_missing_meta:
                fps_p, fc_p, d_p = self._probe_video_meta(s["video_path"])
                if duration is None:
                    duration = d_p
                if frame_count is None:
                    frame_count = fc_p
                if fps is None:
                    fps = fps_p

            if duration is not None:
                if duration < self.min_duration_sec:
                    drop_short += 1
                    continue
                if duration > self.max_duration_sec:
                    drop_long_duration += 1
                    continue

            if frame_count is not None and frame_count < self.frame_num:
                drop_too_few_frames += 1
                continue

            if not os.path.exists(s["video_path"]):
                drop_missing_video += 1
                continue

            filtered.append(s)

        self.samples = filtered
        print(
            f"[Dataset] 过滤统计: caption过长={drop_long}, 时长过短={drop_short}, "
            f"时长过长={drop_long_duration}, 帧数不足={drop_too_few_frames}, 视频缺失={drop_missing_video}"
        )

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, video_path, num_frames):
        """从视频中均匀采样 num_frames 帧"""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        if len(frames) < num_frames:
            return None
        return frames

    def _resize_frame(self, frame, target_h, target_w):
        """将帧 resize 到目标尺寸 (numpy array, HWC)"""
        import cv2
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _compute_target_size(self, h, w):
        """计算满足 VAE stride 和 patch size 约束的目标尺寸。
        要求: H, W 都必须整除 vae_stride * patch_size = 16*2 = 32"""
        ALIGN = 32  # vae_stride(16) * patch_size(2)
        # 按 max_area 缩放
        area = h * w
        if area > self.max_area:
            scale = (self.max_area / area) ** 0.5
            h = int(h * scale)
            w = int(w * scale)
        # 对齐到 ALIGN
        h = max(ALIGN, (h // ALIGN) * ALIGN)
        w = max(ALIGN, (w // ALIGN) * ALIGN)
        return h, w

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        caption = sample["caption"]
        video_path = sample["video_path"]
        ref_path = sample["ref_image_path"]

        # ── 加载视频帧 ───────────────────────────────────────────────────
        frames = self._load_video_frames(video_path, self.frame_num)
        if frames is None:
            # fallback: 尝试下一个样本
            return self.__getitem__((idx + 1) % len(self.samples))

        # ── 计算目标尺寸并 resize ────────────────────────────────────────
        orig_h, orig_w = frames[0].shape[:2]
        target_h, target_w = self._compute_target_size(orig_h, orig_w)
        if (orig_h, orig_w) != (target_h, target_w):
            frames = [self._resize_frame(f, target_h, target_w) for f in frames]

        # ── 加载参考图 ───────────────────────────────────────────────────
        ref_image = None
        if ref_path and os.path.exists(ref_path):
            try:
                ref_image = Image.open(ref_path).convert("RGB")
            except Exception:
                ref_image = None

        if ref_image is None:
            # fallback: 使用视频第一帧
            ref_image = Image.fromarray(frames[0])

        # ── CFG 数据增强: 独立 drop caption 和 image ─────────────────────
        mq_ref_image = ref_image
        if random.random() < self.null_caption_prob:
            caption = ""
        if random.random() < self.null_image_prob:
            mq_ref_image = None

        # ── 转换帧为 tensor ───────────────────────────────────────────────
        # [T, H, W, 3] → [3, T, H, W], 归一化到 [-1, 1]
        frame_tensors = []
        for f in frames:
            t = torch.from_numpy(f).float().permute(2, 0, 1) / 127.5 - 1.0
            frame_tensors.append(t)
        video_tensor = torch.stack(frame_tensors, dim=1)  # [3, T, H, W]

        return {
            "caption": caption,
            "video": video_tensor,          # [3, T, H, W]
            "ref_image": ref_image,         # PIL Image (for i2v first frame)
            "mq_ref_image": mq_ref_image,   # PIL Image or None (for MetaQuery)
            "video_path": video_path,
        }


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

        print("\n" + "=" * 60)
        print("  MetaQuery + Wan TI2V 联合训练")
        print("=" * 60)
        print(f"  DiT 设备       : {self.dev_dit}")
        print(f"  Encoder 设备   : {self.dev_enc}")
        print(f"  学习率         : {args.learning_rate}")
        print(f"  训练步数       : {args.num_train_steps}")
        print(f"  有效 batch     : {args.batch_size * args.gradient_accumulation_steps}")
        print("=" * 60)

        self._load_models()
        self._setup_optimizer()

    def _load_models(self):
        """加载所有模型。"""
        args = self.args

        # ── 1. Wan TI2V Pipeline ─────────────────────────────────────────
        print("\n[1/3] 加载 Wan TI2V Pipeline...")
        from wan import WanTI2V
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['ti2v-5B']
        self.wan = WanTI2V(
            config=config,
            checkpoint_dir=args.wan_checkpoint_dir,
            device_id=args.dit_device,
            rank=0,
            t5_cpu=False,
            init_on_cpu=True,
        )

        # DiT 冻结 (已经在 WanTI2V.__init__ 中设为 eval+no_grad)
        self.wan.model.to(self.dev_dit)
        self.wan.model.eval().requires_grad_(False)

        self.wan_config = config
        self.text_len = config.text_len  # 512
        print(f"  ✅ Wan TI2V 5B 已加载, text_len={self.text_len}")

        # ── 2. MetaQuery Encoder (直接输出 4096) ─────────────────────────
        print("\n[2/3] 加载 MetaQuery Encoder (→4096)...")
        self.mq_encoder = MetaQueryEncoderForWan(
            qwen3vl_model_id=args.qwen3vl_model_id,
            num_metaqueries=args.num_metaqueries,
            connector_num_hidden_layers=args.connector_num_hidden_layers,
            dtype=torch.bfloat16,
            device=f"cuda:{args.encoder_device}",
        )
        self.mq_encoder.train()
        if args.resume_mq_encoder_path:
            ckpt = torch.load(args.resume_mq_encoder_path, map_location="cpu")
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            missing, unexpected = self.mq_encoder.load_state_dict(state, strict=False)
            print(f"  ✅ 已加载初始权重: {args.resume_mq_encoder_path}")
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

        # 扩展 DiT text_len 以容纳 MQ tokens
        self._orig_text_len = self.wan.model.text_len
        self._aug_text_len = self._orig_text_len + args.num_metaqueries
        print(f"  ✅ text_len: {self._orig_text_len} → {self._aug_text_len}")

    def _setup_optimizer(self):
        """设置优化器和学习率调度。"""
        args = self.args

        trainable_params = self.mq_encoder.get_trainable_params()
        print(f"\n[Optimizer] 可训练参数组:")
        print(f"  Connector + MQ Embeddings: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8,
        )

        # Cosine decay with warmup
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, args.num_train_steps - args.warmup_steps)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _encode_text(self, prompts):
        """T5 编码文本"""
        with torch.no_grad():
            self.wan.text_encoder.model.to(self.dev_dit)
            context = self.wan.text_encoder(prompts, self.dev_dit)
        return context  # List[Tensor], each [text_len, 4096]

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

    def _compute_loss(self, batch):
        """
        计算一个 batch 的 Flow Matching 损失。

        训练使用 t2v 模式 (无第一帧蒙版):
        - MetaQuery 参考图是角色裁剪图, 不是视频第一帧
        - 所有 latent 位置都参与去噪和损失计算
        - 推理时可灵活选择 t2v 或 i2v 模式
        """
        args = self.args
        captions = batch["caption"]
        videos = batch["video"]         # list of [3, T, H, W]
        mq_refs = batch["mq_ref_image"]  # list of PIL or None
        B = len(captions)

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

        # ── 2. T5 编码文本 (无梯度) ──────────────────────────────────────
        t5_context = self._encode_text(captions)  # List[Tensor [text_len, 4096]]

        # ── 3. 拼接 MQ + T5 → augmented context ─────────────────────────
        augmented_context = []
        for i in range(B):
            mq_feat = mq_features[i].to(self.dev_dit, dtype=torch.bfloat16)
            t5_feat = t5_context[i].to(self.dev_dit, dtype=torch.bfloat16)
            aug = torch.cat([mq_feat, t5_feat], dim=0)  # [256+512, 4096]
            augmented_context.append(aug)

        # ── 4. VAE 编码视频 → latent (无梯度) ───────────────────────────
        with torch.no_grad():
            latents = self._encode_video(videos)
            # latents: list of [C_z, T', H', W']

        # ── 5. 采样噪声和时间步, 构建 Flow Matching 目标 ─────────────────
        patch_size = self.wan_config.patch_size
        x_inputs = []
        timestep_list = []
        target_list = []
        max_seq_len = 0

        for lat in latents:
            C, T, H, W = lat.shape
            seq_len_i = math.ceil((H * W) / (patch_size[1] * patch_size[2]) * T)
            max_seq_len = max(max_seq_len, seq_len_i)

            t_val = torch.rand(1, device=self.dev_dit, dtype=torch.float32)
            noise = torch.randn_like(lat, dtype=torch.float32)

            # Flow matching: x_t = (1-t) * x_0 + t * noise
            sigma = t_val.view(-1, 1, 1, 1)
            noisy_lat = (1.0 - sigma) * lat.float() + sigma * noise

            # 目标: noise - x_0 (velocity)
            velocity = noise - lat.float()

            x_inputs.append(noisy_lat)
            timestep_list.append(t_val * self.wan.num_train_timesteps)
            target_list.append(velocity)

        # 拼接 timestep → [B]
        timesteps_wan = torch.cat(timestep_list).to(self.dev_dit)

        # ── 6. 扩展 text_len → DiT forward ──────────────────────────────
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

            # ── 7. 计算 MSE loss ──────────────────────────────────────────
            total_loss = 0.0
            for i in range(B):
                pred = model_output[i].float()
                target = target_list[i]
                loss = F.mse_loss(pred, target)
                total_loss += loss
            total_loss = total_loss / B

        finally:
            self.wan.model.text_len = orig_text_len

        return total_loss

    def train(self):
        """主训练循环。"""
        args = self.args

        # 设置随机种子
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # 数据集
        dataset = TomAndJerryVideoDataset(
            caption_json_root=args.caption_json_root,
            manifest_path=args.manifest_path,
            frame_num=args.frame_num,
            max_area=args.max_area,
            null_caption_prob=args.null_caption_prob,
            null_image_prob=args.null_image_prob,
            max_caption_tokens=args.max_caption_tokens,
            caption_tokenizer_path=args.caption_tokenizer_path,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            probe_missing_meta=args.probe_missing_meta,
        )

        if len(dataset) == 0:
            raise RuntimeError("数据集为空！检查路径和 JSON 文件。")

        # 由于视频尺寸可能不同, 使用 batch_size=1 避免 collate 问题
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

        # 训练循环
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = Path(args.output_dir)

        self.mq_encoder.train()
        step = 0
        running_loss = 0.0
        data_iter = iter(dataloader)

        pbar = tqdm(total=args.num_train_steps, desc="Training")
        self.optimizer.zero_grad()

        while step < args.num_train_steps:
            accum_loss = 0.0

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
                    accum_loss += loss.item()
                except Exception as e:
                    print(f"[WARN] step {step} 训练异常: {e}")
                    continue

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.mq_encoder.get_trainable_params(),
                args.max_grad_norm,
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step += 1
            running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss

            # 日志
            if step % args.log_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{accum_loss:.4f}",
                    "avg": f"{running_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "grad": f"{grad_norm:.2f}" if isinstance(grad_norm, float) else f"{grad_norm.item():.2f}",
                })
                print(
                    f"\n[Step {step}/{args.num_train_steps}] "
                    f"loss={accum_loss:.4f} avg={running_loss:.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm:.2f}"
                )

            # 保存
            if step % args.save_steps == 0:
                self._save_checkpoint(output_dir / f"checkpoint-{step}", step)

            pbar.update(1)

        pbar.close()

        # 最终保存
        self._save_checkpoint(output_dir / f"checkpoint-final", step)
        print(f"\n✅ 训练完成！最终 checkpoint: {output_dir / 'checkpoint-final'}")

    def _save_checkpoint(self, path, step):
        """保存 Connector + MQ Embeddings"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存所有可训练参数
        trainable_state = {}
        for name, param in self.mq_encoder.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.data.cpu()

        torch.save({
            "step": step,
            "model_state_dict": trainable_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path / "training_state.pt")

        # 也保存完整 mq_encoder state_dict (方便推理加载)
        torch.save(
            self.mq_encoder.state_dict(),
            path / "mq_encoder_full.pt",
        )

        print(f"  💾 Checkpoint 已保存: {path}")

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
