"""
inference_metaquery_i2v.py
===========================
MetaQuery + Wan2.2 I2V (双模型: high_noise + low_noise) 推理脚本。

★ 核心架构:
    - WanI2V 使用两个 WanModel:
      - high_noise_model: 处理 t >= boundary (0.9 * 1000 = 900) 的高噪声步
      - low_noise_model:  处理 t < boundary 的低噪声步
    - 推理时根据当前 timestep 自动切换模型
    - y = concat(mask, VAE_first_frame_padded) 作为图像到视频的条件
    - context = [MQ_features(256, 4096) | T5_features(512, 4096)]

★ 与原生 WanI2V 的区别:
    - 原生: context = T5 only, text_len=512
    - 本方案: context = MQ + T5, text_len=768
    - 两个模型的 text_len 都需要扩展

用法:
    python inference_metaquery_i2v.py \
        --checkpoint_path /path/to/checkpoint-final/mq_encoder_full.pt \
        --prompt "Tom chases Jerry across the kitchen" \
        --ref_image ./reference.png \
        --output_path output_i2v.mp4
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
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ── 路径设置 ─────────────────────────────────────────────────────────────────
WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))
METAQUERY_ROOT = str(WAN_ROOT.parent / "Qwen3-VL-main" / "metaquery-main")
sys.path.insert(0, METAQUERY_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Inference: MetaQuery + Wan I2V (Dual Model)")

    # ── 模型路径 ──────────────────────────────────────────────────────────
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="checkpoint 文件或目录路径（支持 mq_encoder_full.pt / checkpoint-final/）")
    p.add_argument("--wan_checkpoint_dir", type=str,
                   default="/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B")
    p.add_argument("--qwen3vl_model_id", type=str,
                   default="/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking")

    # ── 输入 ──────────────────────────────────────────────────────────────
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--ref_image", type=str, required=True,
                   help="参考图路径 (MQ 编码 + I2V 首帧条件)")
    p.add_argument("--negative_prompt", type=str, default="")

    # ── 生成参数 ──────────────────────────────────────────────────────────
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--max_area", type=int, default=720 * 1280)
    p.add_argument("--sampling_steps", type=int, default=40,
                   help="I2V 默认 40 步")
    p.add_argument("--guide_scale", type=float, nargs='+', default=[3.5, 3.5],
                   help="CFG scale: [low_noise, high_noise], I2V 默认 (3.5, 3.5)")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample_solver", type=str, default="unipc",
                   choices=["unipc", "dpm++"])
    p.add_argument("--seed", type=int, default=42)

    # ── 输出 ──────────────────────────────────────────────────────────────
    p.add_argument("--output_path", type=str, default="output_i2v_metaquery.mp4")

    # ── MetaQuery ─────────────────────────────────────────────────────────
    p.add_argument("--num_metaqueries", type=int, default=256)
    p.add_argument("--connector_num_hidden_layers", type=int, default=24)

    # ── 设备 ──────────────────────────────────────────────────────────────
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--offload_model", action="store_true",
                   help="切换模型时 offload 另一个到 CPU (节省 VRAM)")

    return p.parse_args()


# =============================================================================
# MetaQuery Encoder (推理模式)
# =============================================================================
class MetaQueryEncoderForI2VInference(nn.Module):
    """加载 I2V 版训练好的 MetaQuery Encoder"""

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
        print("[MetaQuery Inference] 加载 I2V 版 Encoder")
        print(f"  Checkpoint: {checkpoint_path}")
        print("=" * 60)

        from train_connector_for_wan import MetaQueryEncoderForWan
        from train_metaquery_wan import load_mq_encoder_state
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
# MetaQuery + Wan I2V 推理管线
# =============================================================================
class MetaQueryI2VPipeline:
    """
    MetaQuery 增强的 Wan I2V 推理管线 (双模型)。

    ★ 去噪流程:
        for each timestep t:
            if t >= boundary (900):
                model = high_noise_model (guide_scale[1])
            else:
                model = low_noise_model (guide_scale[0])
            model.text_len = 768 (扩展)
            pred = model(latent, t, context=[MQ+T5], y=[mask+first_frame])
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}")
        self._load_pipeline()
        self._load_mq_encoder()

    def _load_pipeline(self):
        from wan import WanI2V
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['i2v-A14B']
        self.wan = WanI2V(
            config=config,
            checkpoint_dir=self.args.wan_checkpoint_dir,
            device_id=self.args.device,
            rank=0,
            t5_cpu=False,
            init_on_cpu=True,
        )
        self.wan_config = config
        self.boundary = config.boundary * config.num_train_timesteps  # 900

        self._orig_text_len_low = self.wan.low_noise_model.text_len
        self._orig_text_len_high = self.wan.high_noise_model.text_len

        print(f"[Pipeline] Wan I2V A14B 双模型已加载")
        print(f"  boundary={self.boundary}")
        print(f"  low_noise text_len={self._orig_text_len_low}")
        print(f"  high_noise text_len={self._orig_text_len_high}")

    def _load_mq_encoder(self):
        self.mq_encoder = MetaQueryEncoderForI2VInference(
            qwen3vl_model_id=self.args.qwen3vl_model_id,
            checkpoint_path=self.args.checkpoint_path,
            num_metaqueries=self.args.num_metaqueries,
            connector_num_hidden_layers=self.args.connector_num_hidden_layers,
            dtype=torch.bfloat16,
            device=f"cuda:{self.args.device}",
        )

    def generate(
        self,
        prompt: str,
        ref_image: Image.Image,
        negative_prompt: str = "",
        max_area: int = 720 * 1280,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 40,
        guide_scale: tuple = (3.5, 3.5),
        seed: int = 42,
    ):
        """
        MetaQuery 增强的 I2V 生成。

        Args:
            prompt: 文本描述
            ref_image: 参考图 (MQ 编码 + I2V 第一帧条件)
            guide_scale: (low_noise_scale, high_noise_scale)
        """
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        device = self.device
        num_mq = self.args.num_metaqueries
        offload_model = self.args.offload_model
        if not negative_prompt:
            negative_prompt = self.wan.sample_neg_prompt

        # ── 1. MetaQuery 编码 ────────────────────────────────────────────
        print("[Generate] MetaQuery 编码...")
        mq_feat = self.mq_encoder.encode(prompt, ref_image)  # [1, 256, 4096]
        mq_feat = mq_feat[0].to(device, dtype=torch.bfloat16)
        null_prompt = negative_prompt
        null_ref_image = Image.new("RGB", ref_image.size)
        mq_feat_null = self.mq_encoder.encode(null_prompt, null_ref_image)[0].to(
            device, dtype=torch.bfloat16
        )

        # ── 2. T5 编码 ──────────────────────────────────────────────────
        print("[Generate] T5 编码...")
        self.wan.text_encoder.model.to(device)
        t5_context = self.wan.text_encoder([prompt], device)
        t5_null = self.wan.text_encoder([negative_prompt], device)
        if offload_model:
            self.wan.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # ── 3. 拼接 context = [MQ + T5] ─────────────────────────────────
        aug_context = [torch.cat([mq_feat, t5_context[0]], dim=0)]   # [768, 4096]
        aug_null    = [torch.cat([mq_feat_null, t5_null[0]], dim=0)] # [768, 4096]
        print(f"  Aug context shape: {aug_context[0].shape}")

        # ── 4. 图像预处理 + latent 尺寸 ─────────────────────────────────
        F = frame_num
        img = TF.to_tensor(ref_image).sub_(0.5).div_(0.5).to(device)
        h, w = img.shape[1:]
        aspect_ratio = h / w

        vae_stride = self.wan_config.vae_stride
        patch_size = self.wan_config.patch_size

        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
            patch_size[1] * patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
            patch_size[2] * patch_size[2])
        H = lat_h * vae_stride[1]
        W = lat_w * vae_stride[2]

        max_seq_len = ((F - 1) // vae_stride[0] + 1) * lat_h * lat_w // (
            patch_size[1] * patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len))

        print(f"  Video: {W}x{H}, {F} frames")
        print(f"  Latent: lat_h={lat_h}, lat_w={lat_w}, seq_len={max_seq_len}")

        # ── 5. 生成噪声 ─────────────────────────────────────────────────
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // vae_stride[0] + 1, lat_h, lat_w,
            dtype=torch.float32, device=device, generator=seed_g)

        # ── 6. 构建 y (mask + VAE first frame) ──────────────────────────
        print("[Generate] 构建 y 条件...")
        with torch.no_grad():
            # mask: 第一帧标记为 1
            msk = torch.ones(1, F, lat_h, lat_w, device=device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]
            # msk: [4, T_lat, H_lat, W_lat]

            # 参考图 → VAE 编码 (第一帧有值, 其余帧全零)
            y = self.wan.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(H, W), mode='bicubic'
                    ).transpose(0, 1),
                    torch.zeros(3, F - 1, H, W)
                ], dim=1).to(device)
            ])[0]
            # y: [16, T_lat, H_lat, W_lat]
            y = torch.concat([msk, y])
            # y: [20, T_lat, H_lat, W_lat]

        # ── 7. 扩展 text_len (两个模型都要) ─────────────────────────────
        aug_text_len = self._orig_text_len_low + num_mq
        self.wan.low_noise_model.text_len = aug_text_len
        self.wan.high_noise_model.text_len = aug_text_len
        print(f"  text_len: {self._orig_text_len_low} → {aug_text_len} (both models)")

        # ── 8. 去噪循环 ─────────────────────────────────────────────────
        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low = getattr(self.wan.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high = getattr(self.wan.high_noise_model, 'no_sync', noop_no_sync)

        with (
            torch.amp.autocast('cuda', dtype=self.wan.param_dtype),
            torch.no_grad(),
            no_sync_low(),
            no_sync_high(),
        ):
            boundary = self.boundary

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

            latent = noise

            arg_c = {
                'context': aug_context,
                'seq_len': max_seq_len,
                'y': [y],
            }
            arg_null = {
                'context': aug_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            print(f"[Generate] 开始去噪 ({len(timesteps)} steps, boundary={boundary})...")
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(device)]
                timestep = torch.stack([t]).to(device)

                # ★ 核心: 根据 t 选择模型
                model = self.wan._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]

                if offload_model:
                    torch.cuda.empty_cache()

                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]

                if offload_model:
                    torch.cuda.empty_cache()

                noise_pred = noise_pred_uncond + sample_guide * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                del latent_model_input, timestep

            x0 = [latent]

            if offload_model:
                self.wan.low_noise_model.cpu()
                self.wan.high_noise_model.cpu()
                torch.cuda.empty_cache()

            print("[Generate] VAE 解码...")
            videos = self.wan.vae.decode(x0)

        # 恢复 text_len
        self.wan.low_noise_model.text_len = self._orig_text_len_low
        self.wan.high_noise_model.text_len = self._orig_text_len_high

        del noise, latent, x0, y
        gc.collect()
        torch.cuda.empty_cache()

        return videos[0]  # [3, T, H, W]


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

    # 确保 guide_scale 是 tuple
    if isinstance(args.guide_scale, list):
        if len(args.guide_scale) == 1:
            args.guide_scale = (args.guide_scale[0], args.guide_scale[0])
        else:
            args.guide_scale = tuple(args.guide_scale[:2])

    pipeline = MetaQueryI2VPipeline(args)

    ref_image = Image.open(args.ref_image).convert("RGB")
    print(f"[Main] 参考图: {args.ref_image} ({ref_image.size})")

    video = pipeline.generate(
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

    save_video(video, args.output_path)
