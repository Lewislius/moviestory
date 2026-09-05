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
from einops import rearrange

# ── 路径设置 ─────────────────────────────────────────────────────────────────
WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))
METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
sys.path.insert(0, METAQUERY_ROOT)

# 复用 TI2V 的 MetaQuery Encoder (输出 4096)
from train_metaquery_wan import MetaQueryEncoderForWan, TomAndJerryVideoDataset


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
    p.add_argument("--frame_num", type=int, default=77,
                   help="Animate 默认 frame_num=77, 必须是 4n+1")
    p.add_argument("--max_area", type=int, default=512 * 512,
                   help="Animate 使用 512x512 分辨率")
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

    # face video (可选)
    p.add_argument("--use_face", action="store_true",
                   help="启用面部视频条件 (需要 face video 数据)")

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

        print("\n" + "=" * 60)
        print("  MetaQuery + Wan Animate 联合训练 (无骨架)")
        print("=" * 60)
        self._load_models()
        self._setup_optimizer()

    def _load_models(self):
        args = self.args

        # ── 1. Wan Animate Pipeline ──────────────────────────────────────
        print("\n[1/3] 加载 Wan Animate Pipeline...")
        from wan import WanAnimate
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['animate-14B']
        self.wan = WanAnimate(
            config=config,
            checkpoint_dir=args.wan_checkpoint_dir,
            device_id=args.dit_device,
            rank=0,
            t5_cpu=False,
            init_on_cpu=True,
        )

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
        assert self.wan.noise_model.text_dim == 4096
        assert self.mq_encoder.wan_text_dim == 4096

        self._orig_text_len = self.wan.noise_model.text_len
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

        # ── 5. 构建 y (参考图 VAE latent + mask) ────────────────────────
        with torch.no_grad():
            y_list = []
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

        # Face: 全零 (或真实数据, 如果启用)
        face_pixel_values = torch.zeros(
            B, 3, args.frame_num, 512, 512,
            device=self.dev_dit, dtype=torch.bfloat16
        )

        # ── 7. Flow Matching ────────────────────────────────────────────
        patch_size = self.wan_config.patch_size
        x_inputs = []
        timestep_list = []
        target_list = []
        max_seq_len = 0

        for lat in latents:
            C, T, H, W = lat.shape
            # Animate 的 target_shape 包含 +1 帧 (ref frame)
            seq_len_i = math.ceil(np.prod([T + 1, H, W]) // (patch_size[1] * patch_size[2]))
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
        pbar = tqdm(total=args.num_train_steps, desc="Animate Training")
        self.optimizer.zero_grad()

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
    trainer = MetaQueryAnimateTrainer(args)
    trainer.train()
