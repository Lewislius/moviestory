"""
inference_metaquery_animate.py
==============================
MetaQuery + Wan2.2 Animate (无骨架/面部) 推理脚本。

★ WanAnimateModel.forward 签名:
    forward(x, t, clip_fea, context, seq_len, y, pose_latents, face_pixel_values)
    - x: TensorList of noise latents
    - t: timestep [B]
    - clip_fea: [B, 257, 1280] → img_emb → [B, 257, 5120] → 拼到 context 前面
    - context: List[Tensor [text_len, 4096]] → text_embedding → [B, text_len, 5120]
    - y: List[Tensor [20, T+1, H', W']] → 参考图 + 时序参考
    - pose_latents: [B, 16, T, H', W'] → pose_patch_embedding → 加到 patch 上
    - face_pixel_values: [B, 3, F, H_face, W_face] → face pipeline

★ 本推理方案 (简化 Animate):
    - MQ features 拼接到 T5 context 前面 (text_len 扩展)
    - clip_fea = 零向量 (MQ 覆盖了 CLIP visual 的语义角色)
    - pose_latents = 全零 (不使用骨架条件)
    - face_pixel_values = 全零 (不使用面部条件)
    - y = 正常构建 (参考图 VAE latent + mask, Animate 核心条件)
    - 解码时跳过第一帧: vae.decode([x0[:, 1:]])

用法:
    python inference_metaquery_animate.py \
        --checkpoint_path /path/to/checkpoint-final/mq_encoder_full.pt \
        --prompt "Tom chases Jerry across the kitchen" \
        --ref_image ./reference.png \
        --output_path output_animate.mp4
"""

import os
import sys
import gc
import math
import random
import argparse
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from einops import rearrange

# ── 路径设置 ─────────────────────────────────────────────────────────────────
WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))
METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
sys.path.insert(0, METAQUERY_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Inference: MetaQuery + Wan Animate (no skeleton)")

    # ── 模型路径 ──────────────────────────────────────────────────────────
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="checkpoint 文件或目录路径（支持 mq_encoder_full.pt / checkpoint-final/）")
    p.add_argument("--wan_checkpoint_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B")
    p.add_argument("--qwen3vl_model_id", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking")

    # ── 输入 ──────────────────────────────────────────────────────────────
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--ref_image", type=str, required=True,
                   help="参考图路径 (角色参考 + Animate 条件)")
    p.add_argument("--mq_ref_only", action="store_true",
                   help="仅将参考图用于 MetaQuery 编码，不将参考图传给 Wan 的 y 条件")
    p.add_argument("--no_ref_condition", action="store_true",
                   help="参考图既不用于 MetaQuery，也不用于 Wan 的 y 条件")
    p.add_argument("--negative_prompt", type=str, default="")

    # ── 生成参数 ──────────────────────────────────────────────────────────
    p.add_argument("--frame_num", type=int, default=81,
                   help="生成帧数 (必须是 4n+1)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--guide_scale", type=float, default=1.0,
                   help="Animate 默认 guide_scale=1.0 (无 CFG)")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample_solver", type=str, default="unipc",
                   choices=["unipc", "dpm++"])
    p.add_argument("--seed", type=int, default=42)

    # ── 输出 ──────────────────────────────────────────────────────────────
    p.add_argument("--output_path", type=str, default="output_animate_metaquery.mp4")

    # ── MetaQuery ─────────────────────────────────────────────────────────
    p.add_argument("--num_metaqueries", type=int, default=256)
    p.add_argument("--connector_num_hidden_layers", type=int, default=24)

    # ── 设备 ──────────────────────────────────────────────────────────────
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--offload_model", action="store_true")

    return p.parse_args()


# =============================================================================
# MetaQuery Encoder (推理模式, 复用训练脚本的类)
# =============================================================================
class MetaQueryEncoderForAnimateInference(nn.Module):
    """加载 Animate 版训练好的 MetaQuery Encoder"""

    WAN_TEXT_DIM = 4096

    def __init__(
        self,
        qwen3vl_model_id: str,
        checkpoint_path: str,
        num_metaqueries: int = 256,
        connector_num_hidden_layers: int = 24,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)

        print("=" * 60)
        print("[MetaQuery Inference] 加载 Animate 版 Encoder")
        print(f"  Checkpoint: {checkpoint_path}")
        print("=" * 60)

        from train_metaquery_wan import MetaQueryEncoderForWan, load_mq_encoder_state
        encoder = MetaQueryEncoderForWan(
            qwen3vl_model_id=qwen3vl_model_id,
            num_metaqueries=num_metaqueries,
            connector_num_hidden_layers=connector_num_hidden_layers,
            dtype=dtype,
            device=device,
        )

        state_dict, resolved_path = load_mq_encoder_state(
            checkpoint_path,
            map_location=self.device,
        )
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        print(f"  Resolved ckpt: {resolved_path}")
        print(f"  Missing keys : {len(missing)}")
        print(f"  Unexpected   : {len(unexpected)}")

        self.encoder = encoder
        self.encoder.eval()
        print("[MetaQuery Inference] ✅ 加载完成")

    @torch.no_grad()
    def encode(self, caption, ref_image=None):
        captions = [caption]
        images = [[ref_image]] if ref_image is not None else None
        return self.encoder(captions, images)  # [1, 256, 4096]


# =============================================================================
# MetaQuery + Wan Animate 推理管线
# =============================================================================
class MetaQueryAnimatePipeline:
    """
    MetaQuery 增强的 Wan Animate 推理管线 (无骨架/面部)。

    ★ 上下文构建:
        text context = [MQ_tokens(256, 4096) | T5_tokens(512, 4096)]
        → text_embedding → [B, 768, 5120]
        然后 Animate 会再拼接 CLIP tokens (257 个零):
        final_context = [CLIP_zero(257, 5120) | MQ+T5(768, 5120)]

    ★ 条件信号:
        y = [y_ref | y_reft]: 参考图 + 时序参考的 VAE latent + mask
        pose_latents = 全零 (不使用骨架)
        face_pixel_values = 全零 (不使用面部)
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}")
        self._load_pipeline()
        self._load_mq_encoder()

    def _load_pipeline(self):
        """加载 Wan Animate Pipeline"""
        from wan import WanAnimate
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['animate-14B']
        self.wan = WanAnimate(
            config=config,
            checkpoint_dir=self.args.wan_checkpoint_dir,
            device_id=self.args.device,
            rank=0,
            t5_cpu=False,
            init_on_cpu=True,
        )
        self.wan.noise_model.to(self.device)
        self.wan_config = config
        self._orig_text_len = self.wan.noise_model.text_len
        print(f"[Pipeline] Wan Animate 14B 已加载")
        print(f"  text_len={self._orig_text_len}, in_dim={self.wan.noise_model.in_dim}")
        print(f"  dim={self.wan.noise_model.dim}")

    def _load_mq_encoder(self):
        self.mq_encoder = MetaQueryEncoderForAnimateInference(
            qwen3vl_model_id=self.args.qwen3vl_model_id,
            checkpoint_path=self.args.checkpoint_path,
            num_metaqueries=self.args.num_metaqueries,
            connector_num_hidden_layers=self.args.connector_num_hidden_layers,
            dtype=torch.bfloat16,
            device=f"cuda:{self.args.device}",
        )

    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, device="cuda"):
        """生成 Animate i2v mask (参考 WanAnimate.get_i2v_mask)"""
        msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        msk[:, :mask_len] = 1
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, lat_t, lat_h, lat_w]
        return msk

    def generate(
        self,
        prompt: str,
        ref_image: Image.Image,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 1.0,
        seed: int = 42,
    ):
        """
        MetaQuery 增强的 Animate 生成。

        Args:
            prompt: 文本描述
            ref_image: 参考图 (默认同时用于 MQ 编码和 Animate y 条件)
            negative_prompt: 负面文本
            height, width: 视频分辨率
            frame_num: 帧数 (4n+1)
            guide_scale: Animate 默认 1.0 (不使用 CFG)
            seed: 随机种子
        """
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from wan.modules.animate.animate_utils import TensorList

        device = self.device
        num_mq = self.args.num_metaqueries

        # ── 1. MetaQuery 编码 ────────────────────────────────────────────
        use_mq_ref = not self.args.no_ref_condition
        mq_ref_image = ref_image if use_mq_ref else None
        if use_mq_ref:
            print("[Generate] MetaQuery 编码 (with ref image)...")
        else:
            print("[Generate] MetaQuery 编码 (text-only, no ref image)...")
        mq_feat = self.mq_encoder.encode(prompt, mq_ref_image)  # [1, 256, 4096]
        mq_feat = mq_feat[0].to(device, dtype=torch.bfloat16)  # [256, 4096]
        mq_feat_null = torch.zeros_like(mq_feat)

        # ── 2. T5 编码 ──────────────────────────────────────────────────
        print("[Generate] T5 编码...")
        self.wan.text_encoder.model.to(device)
        t5_context = self.wan.text_encoder([prompt], device)
        if not negative_prompt:
            negative_prompt = self.wan.sample_neg_prompt
        t5_null = self.wan.text_encoder([negative_prompt], device)
        if self.args.offload_model:
            self.wan.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # ── 3. 拼接 context = [MQ + T5] ─────────────────────────────────
        aug_context = [torch.cat([mq_feat, t5_context[0]], dim=0)]   # [768, 4096]
        aug_null    = [torch.cat([mq_feat_null, t5_null[0]], dim=0)] # [768, 4096]

        print(f"  MQ features  : {mq_feat.shape}")
        print(f"  T5 features  : {t5_context[0].shape}")
        print(f"  Aug context  : {aug_context[0].shape}")

        # ── 4. 计算 latent 尺寸 ─────────────────────────────────────────
        H, W = height, width
        T = frame_num
        vae_stride = self.wan_config.vae_stride  # (4, 8, 8)
        lat_t = (T - 1) // vae_stride[0] + 1
        lat_h = H // vae_stride[1]
        lat_w = W // vae_stride[2]
        z_dim = 16  # Wan2_1_VAE z_dim

        # target_shape 包含 +1 帧 (Animate 的 ref frame slot)
        target_shape = (z_dim, lat_t + 1, lat_h, lat_w)
        max_seq_len = int(math.ceil(np.prod(target_shape[1:]) / 4))
        # 对齐 patch_size
        patch_size = self.wan_config.patch_size
        max_seq_len = int(math.ceil(
            np.prod(target_shape) // (patch_size[1] * patch_size[2])))

        print(f"  Latent: T={lat_t}(+1ref), H={lat_h}, W={lat_w}")
        print(f"  seq_len: {max_seq_len}")

        # ── 5. 生成噪声 ─────────────────────────────────────────────────
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed if seed >= 0 else random.randint(0, sys.maxsize))
        noise = [
            torch.randn(
                z_dim, target_shape[1], target_shape[2], target_shape[3],
                dtype=torch.float32, device=device, generator=seed_g)
        ]

        # ── 6. 参考图 → VAE → y 条件 ────────────────────────────────────
        print("[Generate] 构建参考图条件 y...")
        use_wan_ref = (not self.args.mq_ref_only) and (not self.args.no_ref_condition)
        if use_wan_ref:
            print("  y condition : use ref image (Wan + MQ)")
        else:
            if self.args.no_ref_condition:
                print("  y condition : disable ref image for Wan (no-ref mode)")
            else:
                print("  y condition : disable ref image for Wan (MQ only)")
        with torch.no_grad():
            if use_wan_ref:
                # 参考图预处理
                img_resized = ref_image.resize((W, H), Image.LANCZOS)
                img_np = np.array(img_resized).astype(np.float32)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) / 127.5 - 1.0
                img_tensor = img_tensor.unsqueeze(1).to(device, dtype=torch.bfloat16)
                # [3, 1, H, W]

                ref_latents = self.wan.vae.encode([img_tensor])
                ref_latents = torch.stack(ref_latents)  # [1, 16, 1, lat_h, lat_w]

                # y_ref: [4+16=20, 1, lat_h, lat_w]
                mask_ref = self.get_i2v_mask(1, lat_h, lat_w, mask_len=1, device=device)
                y_ref = torch.cat([mask_ref, ref_latents[0]], dim=0).to(torch.bfloat16)
            else:
                ref_latents = torch.zeros(
                    1, 16, 1, lat_h, lat_w, device=device, dtype=torch.bfloat16
                )
                mask_ref = self.get_i2v_mask(1, lat_h, lat_w, mask_len=0, device=device)
                y_ref = torch.cat([mask_ref, ref_latents[0]], dim=0).to(torch.bfloat16)

            # y_reft: 全零 (不使用时序参考)
            msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_len=0, device=device)
            zero_reft = torch.zeros(16, lat_t, lat_h, lat_w, device=device, dtype=torch.bfloat16)
            y_reft = torch.cat([msk_reft, zero_reft], dim=0).to(torch.bfloat16)

            # y = [y_ref | y_reft] → [20, 1+lat_t, lat_h, lat_w]
            y = torch.cat([y_ref, y_reft], dim=1)

        # ── 7. CLIP (零向量), Pose (零), Face (零) ──────────────────────
        clip_fea = torch.zeros(1, 257, 1280, device=device, dtype=torch.bfloat16)

        pose_latents = torch.zeros(
            1, 16, lat_t, lat_h, lat_w, device=device, dtype=torch.bfloat16)

        # 关键优化:
        # 不使用 face 条件时必须传 None，而不是全零张量。
        # 否则会触发 motion_encoder 分支，带来显著显存和计算开销。
        face_pixel_values = None

        # 关键优化:
        # 去噪阶段不需要 VAE 编码器参数驻留 GPU，先迁移到 CPU，解码前再搬回。
        try:
            self.wan.vae.model.cpu()
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ── 8. 去噪循环 ─────────────────────────────────────────────────
        aug_text_len = self._orig_text_len + num_mq
        self.wan.noise_model.text_len = aug_text_len
        print(f"  text_len: {self._orig_text_len} → {aug_text_len}")

        with (
            torch.autocast(device_type=str(device).split(':')[0], dtype=torch.bfloat16),
            torch.no_grad(),
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

            latents = noise

            arg_c = {
                "context": aug_context,
                "seq_len": max_seq_len,
                "clip_fea": clip_fea,
                "y": [y],
                "pose_latents": pose_latents,
                "face_pixel_values": face_pixel_values,
            }

            # CFG 无条件参数 (guide_scale > 1 时使用)
            if guide_scale > 1:
                arg_null = {
                    "context": aug_null,
                    "seq_len": max_seq_len,
                    "clip_fea": clip_fea,
                    "y": [y],
                    "pose_latents": pose_latents,
                    "face_pixel_values": None,
                }

            print(f"[Generate] 开始去噪 ({len(timesteps)} steps)...")
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = torch.stack([t])

                noise_pred_cond = TensorList(
                    self.wan.noise_model(
                        TensorList(latent_model_input),
                        t=timestep,
                        **arg_c
                    )
                )

                if guide_scale > 1:
                    noise_pred_uncond = TensorList(
                        self.wan.noise_model(
                            TensorList(latent_model_input),
                            t=timestep,
                            **arg_null
                        )
                    )
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                temp_x0 = scheduler.step(
                    noise_pred[0].unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents[0] = temp_x0.squeeze(0)

            x0 = latents
            x0 = [x.to(dtype=torch.float32) for x in x0]

            if self.args.offload_model:
                self.wan.noise_model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # ★ Animate 解码: 跳过第一帧 (ref frame slot)
            print("[Generate] VAE 解码 (跳过 ref frame)...")
            try:
                self.wan.vae.model.to(device)
                torch.cuda.empty_cache()
            except Exception:
                pass
            out_frames = torch.stack(self.wan.vae.decode([x0[0][:, 1:]]))

        # 恢复 text_len
        self.wan.noise_model.text_len = self._orig_text_len

        del noise, latents, x0, y, ref_latents
        gc.collect()
        torch.cuda.empty_cache()

        return out_frames[0]  # [3, T, H, W]


# =============================================================================
# 视频保存
# =============================================================================
def save_video(video_tensor, output_path, fps=24):
    import cv2
    video = video_tensor.cpu().float()
    if video.min() < 0:
        video = (video + 1.0) / 2.0
    video = video.clamp(0, 1) * 255
    video = video.byte()
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

    pipeline = MetaQueryAnimatePipeline(args)

    ref_image = Image.open(args.ref_image).convert("RGB")
    print(f"[Main] 参考图: {args.ref_image} ({ref_image.size})")

    video = pipeline.generate(
        prompt=args.prompt,
        ref_image=ref_image,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )

    save_video(video, args.output_path)
