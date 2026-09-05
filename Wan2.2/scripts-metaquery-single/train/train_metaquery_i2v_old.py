"""
train_metaquery_i2v.py
=======================
MetaQuery + Wan2.2 I2V (双模型: high_noise + low_noise) 联合训练脚本。

★ 核心架构分析 (WanI2V):
    - 使用标准 WanModel, model_type='i2v'
    - in_dim=16, 但 i2v 模式下 x = concat(x, y) → 有效 ch = 16 + 16+4(mask) = 36
    - dim=5120, text_dim=4096, text_len=512
    - NO CLIP — 只使用 T5 作为 context
    - 双模型架构:
        - high_noise_model: 处理 t >= boundary (0.9 * 1000 = 900)
        - low_noise_model:  处理 t < boundary
    - VAE: Wan2_1_VAE, stride=(4,8,8)
    - y = concat(mask, VAE_first_frame_padded_with_zeros)

★ 本训练方案:
    - MetaQuery features (256, 4096) 拼接到 T5 context 前面
    - 两个 DiT 模型都冻结, 只训练 Connector
    - text_len 扩展: 512 → 768 (两个模型都需要扩展)
    - 训练时根据随机采样的 timestep 选择正确的模型

★ 与 TI2V 方案的关键区别:
    - TI2V: 单模型, 使用 Wan2_2_VAE (stride 4,16,16)
    - I2V:  双模型, 使用 Wan2_1_VAE (stride 4,8,8)
    - I2V: 有 y 条件 (第一帧 VAE latent + mask)
    - I2V: 更大的 latent (8x 而非 16x 空间下采样)

用法:
    python train_metaquery_i2v.py \
        --wan_checkpoint_dir /path/to/Wan2.2-I2V-A14B \
        --qwen3vl_model_id /path/to/Qwen3-VL-2B-Thinking
"""

import os
import sys
import gc
import json
import math
import argparse
import random
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ── 路径设置 ─────────────────────────────────────────────────────────────────
WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))
METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
sys.path.insert(0, METAQUERY_ROOT)

# 复用 MetaQuery Encoder 和数据集
from train_metaquery_wan import MetaQueryEncoderForWan, TomAndJerryVideoDataset


# =============================================================================
# 配置
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train MetaQuery for Wan I2V (Dual Model)")

    p.add_argument("--wan_checkpoint_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B")
    p.add_argument("--qwen3vl_model_id", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking")
    p.add_argument("--output_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/metaquery_i2v_training")
    p.add_argument("--caption_json_root", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/dataset/Tom_and_Jerry_720x1280")
    p.add_argument("--manifest_path", type=str, default=None)

    # 训练
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--frame_num", type=int, default=81,
                   help="4n+1, I2V 通常使用 81 帧")
    p.add_argument("--max_area", type=int, default=720 * 1280,
                   help="最大面积, I2V 默认 720x1280")
    p.add_argument("--max_caption_tokens", type=int, default=512)
    p.add_argument("--caption_tokenizer_path", type=str, default="google/umt5-xxl")
    p.add_argument("--min_duration_sec", type=float, default=0.5)
    p.add_argument("--max_duration_sec", type=float, default=20.0)
    p.add_argument("--probe_missing_meta", action="store_true")

    # MetaQuery
    p.add_argument("--num_metaqueries", type=int, default=256)
    p.add_argument("--connector_num_hidden_layers", type=int, default=24)
    p.add_argument("--null_caption_prob", type=float, default=0.1)
    p.add_argument("--null_image_prob", type=float, default=0.1)

    # 设备
    p.add_argument("--dit_device", type=int, default=0)
    p.add_argument("--encoder_device", type=int, default=1)
    p.add_argument("--resume_mq_encoder_path", type=str, default=None)

    return p.parse_args()


# =============================================================================
# Trainer
# =============================================================================
class MetaQueryI2VTrainer:
    """
    MetaQuery + Wan I2V 联合训练 (双模型架构)。

    ★ I2V 双模型架构:
        - high_noise_model: 处理 t >= boundary*1000=900 的高噪声时间步
        - low_noise_model:  处理 t < 900 的低噪声时间步
        - 两个模型的 forward 签名都是: forward(x, t, context, seq_len, y)

    ★ 训练策略:
        - 均匀采样 t ∈ [0, 1000)
        - 根据 t 选择对应模型
        - 两个模型都冻结, 各自的 text_len 都扩展
        - 只训练 MetaQuery Connector

    ★ y 条件构建 (I2V 标准):
        - 第一帧: 真实图像 VAE 编码
        - 其余帧: 全零填充
        - mask: 标记第一帧为 1, 其余为 0
        - y = concat(mask[4ch], latent[16ch]) → [20, T_lat, H_lat, W_lat]

    ★ context 构建:
        - context = [MQ_features(256, 4096) | T5_features(512, 4096)] → [768, 4096]
        - 经过 WanModel.text_embedding (Linear 4096→5120, GELU, Linear 5120→5120)
    """

    def __init__(self, args):
        self.args = args
        self.dev_dit = torch.device(f"cuda:{args.dit_device}")
        self.dev_enc = torch.device(f"cuda:{args.encoder_device}")

        print("\n" + "=" * 60)
        print("  MetaQuery + Wan I2V 联合训练 (双模型架构)")
        print("=" * 60)
        self._load_models()
        self._setup_optimizer()

    def _load_models(self):
        args = self.args

        # ── 1. Wan I2V Pipeline ──────────────────────────────────────────
        print("\n[1/3] 加载 Wan I2V Pipeline (双模型)...")
        from wan import WanI2V
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['i2v-A14B']
        self.wan = WanI2V(
            config=config,
            checkpoint_dir=args.wan_checkpoint_dir,
            device_id=args.dit_device,
            rank=0,
            t5_cpu=False,
            init_on_cpu=True,
        )

        # 两个模型都加载到 GPU
        self.wan.high_noise_model.to(self.dev_dit)
        self.wan.low_noise_model.to(self.dev_dit)
        self.wan.high_noise_model.eval().requires_grad_(False)
        self.wan.low_noise_model.eval().requires_grad_(False)

        self.wan_config = config
        self.text_len = config.text_len  # 512
        self.boundary = config.boundary * config.num_train_timesteps  # 0.9 * 1000 = 900

        print(f"  ✅ Wan I2V A14B 双模型已加载")
        print(f"    high_noise_model text_dim={self.wan.high_noise_model.text_dim}")
        print(f"    low_noise_model  text_dim={self.wan.low_noise_model.text_dim}")
        print(f"    text_len={self.text_len}")
        print(f"    boundary={self.boundary} (t >= {self.boundary} → high noise)")

        # ── 2. MetaQuery Encoder → 4096 ─────────────────────────────────
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

        # ── 3. 验证 ─────────────────────────────────────────────────────
        print("\n[3/3] 验证维度对齐...")
        assert self.wan.high_noise_model.text_dim == 4096
        assert self.wan.low_noise_model.text_dim == 4096
        assert self.mq_encoder.wan_text_dim == 4096

        self._orig_text_len = self.text_len
        self._aug_text_len = self._orig_text_len + args.num_metaqueries
        print(f"  ✅ text_len: {self._orig_text_len} → {self._aug_text_len}")

    def _setup_optimizer(self):
        args = self.args
        trainable_params = self.mq_encoder.get_trainable_params()
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

    def _compute_loss(self, batch):
        """
        计算 Flow Matching 损失。

        ★ 关键: 根据采样到的 timestep 自动选择 high/low noise 模型。
        """
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
            self.wan.text_encoder.model.to(self.dev_dit)
            t5_context = self.wan.text_encoder(captions, self.dev_dit)

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
                # z[0]: [16, T_lat, H_lat, W_lat]

        # ── 5. 构建 y (I2V 标准: mask + first_frame_padded) ─────────────
        with torch.no_grad():
            y_list = []
            for i in range(B):
                lat = latents[i]
                C, T_lat, H_lat, W_lat = lat.shape

                # 参考图: 视频第一帧或独立参考图
                ref_img = ref_images[i]
                # 将参考图 resize 到匹配 latent 对应的像素尺寸
                target_h = H_lat * self.wan_config.vae_stride[1]  # H_lat * 8
                target_w = W_lat * self.wan_config.vae_stride[2]  # W_lat * 8
                ref_resized = ref_img.resize((target_w, target_h), Image.LANCZOS)
                ref_np = np.array(ref_resized).astype(np.float32)
                ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1) / 127.5 - 1.0
                # [3, H, W]

                # I2V 的 y 构建: 第一帧为参考图, 其余填零, 然后 VAE 编码
                F = (T_lat - 1) * self.wan_config.vae_stride[0] + 1
                padded_video = torch.cat([
                    ref_tensor.unsqueeze(1),  # [3, 1, H, W]
                    torch.zeros(3, F - 1, target_h, target_w),
                ], dim=1).to(self.dev_dit, dtype=torch.bfloat16)

                y_latent = self.wan.vae.encode([padded_video])[0]
                # y_latent: [16, T_lat, H_lat, W_lat]

                # mask: 标记第一帧
                msk = torch.ones(1, F, H_lat, W_lat, device=self.dev_dit)
                msk[:, 1:] = 0
                msk = torch.concat([
                    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                    msk[:, 1:]
                ], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, H_lat, W_lat)
                msk = msk.transpose(1, 2)[0]
                # msk: [4, T_lat, H_lat, W_lat]

                y = torch.cat([msk, y_latent], dim=0).to(torch.bfloat16)
                # y: [20, T_lat, H_lat, W_lat]
                y_list.append(y)

        # ── 6. Flow Matching: 采样 t, 选择模型 ──────────────────────────
        vae_stride = self.wan_config.vae_stride
        patch_size = self.wan_config.patch_size

        total_loss = 0.0
        for i in range(B):
            lat = latents[i]
            C, T_lat, H_lat, W_lat = lat.shape

            seq_len = T_lat * H_lat * W_lat // (patch_size[1] * patch_size[2])
            seq_len = int(math.ceil(seq_len))

            # 采样 timestep
            t_val = torch.rand(1, device=self.dev_dit, dtype=torch.float32)
            t_wan = t_val * self.wan.num_train_timesteps

            # 选择模型
            if t_wan.item() >= self.boundary:
                model = self.wan.high_noise_model
                model_name = "high_noise"
            else:
                model = self.wan.low_noise_model
                model_name = "low_noise"

            # 加噪
            noise = torch.randn_like(lat, dtype=torch.float32)
            sigma = t_val.view(-1, 1, 1, 1)
            noisy_lat = (1.0 - sigma) * lat.float() + sigma * noise
            velocity = noise - lat.float()

            # 扩展 text_len
            orig_text_len = model.text_len
            model.text_len = self._aug_text_len

            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(
                        [noisy_lat],
                        t=t_wan,
                        context=[augmented_context[i]],
                        seq_len=seq_len,
                        y=[y_list[i]],
                    )[0]

                loss = F.mse_loss(pred.float(), velocity)
                total_loss += loss
            finally:
                model.text_len = orig_text_len

        total_loss /= B
        return total_loss

    def train(self):
        args = self.args
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

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
            raise RuntimeError("数据集为空！")

        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=2,
            pin_memory=True, collate_fn=self._collate_fn,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.mq_encoder.train()
        step = 0
        running_loss = 0.0
        data_iter = iter(dataloader)
        pbar = tqdm(total=args.num_train_steps, desc="I2V Training")
        self.optimizer.zero_grad()

        # 统计高/低噪声使用频率
        high_noise_count = 0
        low_noise_count = 0

        while step < args.num_train_steps:
            accum_loss = 0.0
            for _ in range(args.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                try:
                    loss = self._compute_loss(batch)
                    (loss / args.gradient_accumulation_steps).backward()
                    accum_loss += loss.item() / args.gradient_accumulation_steps
                except Exception as e:
                    print(f"[WARN] step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.mq_encoder.get_trainable_params(), args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step += 1
            running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss

            if step % args.log_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{accum_loss:.4f}", avg=f"{running_loss:.4f}", lr=f"{lr:.2e}")
                print(f"\n[Step {step}] loss={accum_loss:.4f} avg={running_loss:.4f} lr={lr:.2e}")

            if step % args.save_steps == 0:
                self._save_checkpoint(output_dir / f"checkpoint-{step}", step)
            pbar.update(1)

        pbar.close()
        self._save_checkpoint(output_dir / "checkpoint-final", step)

    def _save_checkpoint(self, path, step):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        trainable = {n: p.data.cpu() for n, p in self.mq_encoder.named_parameters() if p.requires_grad}
        torch.save({"step": step, "model_state_dict": trainable,
                     "optimizer_state_dict": self.optimizer.state_dict(),
                     "scheduler_state_dict": self.scheduler.state_dict()},
                    path / "training_state.pt")
        torch.save(self.mq_encoder.state_dict(), path / "mq_encoder_full.pt")
        print(f"  💾 Saved: {path}")

    @staticmethod
    def _collate_fn(batch):
        return {k: [item[k] for item in batch] for k in batch[0]}


if __name__ == "__main__":
    args = parse_args()
    trainer = MetaQueryI2VTrainer(args)
    trainer.train()
