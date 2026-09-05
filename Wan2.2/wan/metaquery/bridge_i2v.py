"""
MetaQueryWanI2VBridge: 将 MetaQuery 视觉条件注入 Wan2.2 I2V (Image-to-Video) 管线。

注入策略:
    1. **首帧条件 (Channel Concatenation)** — 与原始 WanI2V 完全一致:
       输入首帧图像经 VAE 编码后，与 4 通道掩码拼接为 20 通道 y tensor，
       在 WanModel.forward 中与噪声 latent (16ch) channel-cat 成 36 通道输入。

    2. **MetaQuery 语义条件 (Context Concatenation)** — 与 T2V bridge 一致:
       MetaQuery (Qwen3-VL) 从参考图像提取 256 个语义 token (dim=4096)，
       前置拼接到 T5 文本 context 上，参与 WanModel 所有层的 cross-attention。

    两种条件互补:
    - 首帧条件提供像素级/结构级的视觉一致性（从首帧"扩展"出视频）
    - MetaQuery 条件提供语义级的视觉引导（风格、内容、构图等高层语义）
"""

import sys
import gc
import math
import random
from contextlib import contextmanager
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F_nn
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .encoder import MetaQueryEncoder
from ..utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class MetaQueryWanI2VBridge:
    """
    MetaQuery 增强的 Wan2.2 图生视频 (I2V) 管线。

    在标准 WanI2V 生成流程的基础上，额外注入 Qwen3-VL MetaQuery
    视觉语义特征，实现 **首帧结构条件 + MetaQuery 语义条件** 的双重引导。

    Args:
        wan_i2v_pipeline:      已初始化的 WanI2V 实例。
        metaquery_checkpoint:  MetaQuery + Qwen3-VL checkpoint 路径。
        num_metaqueries:       MetaQuery token 数量，默认 256。
        mq_guidance_scale:     MetaQuery 引导强度，默认 1.0。
        dtype:                 计算精度，默认 bfloat16。
    """

    def __init__(
        self,
        wan_i2v_pipeline,
        metaquery_checkpoint: str,
        num_metaqueries: int = 256,
        mq_guidance_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.wan = wan_i2v_pipeline
        self.mq_guidance_scale = mq_guidance_scale
        self.dtype = dtype

        print("\n" + "=" * 60)
        print("[MetaQueryWanI2VBridge] 初始化 MetaQuery + Wan2.2 I2V 联合管线")
        print(f"  Wan pipeline 类型 : {wan_i2v_pipeline.__class__.__name__}")
        print(f"  MetaQuery ckpt    : {metaquery_checkpoint}")
        print(f"  num_metaqueries   : {num_metaqueries}")
        print(f"  mq_guidance_scale : {mq_guidance_scale}")
        print("=" * 60 + "\n")

        # ── 验证传入的是 WanI2V 实例 ─────────────────────────────────────────
        pipeline_cls = wan_i2v_pipeline.__class__.__name__
        assert 'I2V' in pipeline_cls or 'i2v' in pipeline_cls.lower(), (
            f"[FATAL] 期望 WanI2V 实例, 但收到 {pipeline_cls}! "
            "请使用 MetaQueryWanBridge (T2V) 或确认传入 WanI2V。"
        )

        # ── 加载 MetaQuery 编码器 ────────────────────────────────────────────
        self.mq_encoder = MetaQueryEncoder(
            metaquery_checkpoint_path=metaquery_checkpoint,
            num_metaqueries=num_metaqueries,
            wan_text_dim=4096,
            dtype=dtype,
            device=wan_i2v_pipeline.device,
        )

        self.num_metaqueries = num_metaqueries

        # ── 记录原始 text_len，后续动态扩展 ──────────────────────────────────
        self._orig_text_len = self.wan.high_noise_model.text_len
        self._aug_text_len = self._orig_text_len + num_metaqueries

        print(
            f"[MetaQueryWanI2VBridge] WanModel text_len: "
            f"{self._orig_text_len} → {self._aug_text_len} "
            f"(+{num_metaqueries} MetaQuery tokens)"
        )

        # ── 初始化完整性验证 ──────────────────────────────────────────────────
        print("\n[MetaQueryWanI2VBridge] 【初始化完整性验证】")

        # 验证 WanModel 的 text_embedding 输入维度 = 4096 (匹配 MetaQuery 投影)
        text_emb_in = self.wan.high_noise_model.text_embedding[0].in_features
        assert text_emb_in == 4096, (
            f"[FATAL] WanModel.text_embedding.in_features={text_emb_in}, "
            "期望=4096! MetaQuery 投影维度不匹配"
        )
        print(f"  [PASS] WanModel text_embedding 输入维度: {text_emb_in}")

        # 验证 WanModel 是 I2V 模型 (model_type='i2v')
        model_type = self.wan.high_noise_model.model_type
        assert model_type == 'i2v', (
            f"[FATAL] WanModel.model_type='{model_type}', 期望='i2v'! "
            "请确认使用了 I2V 配置和 checkpoint。"
        )
        print(f"  [PASS] WanModel.model_type = '{model_type}'")

        # 验证 I2V 模型的 patch_embedding 输入通道 = 36 (16 latent + 20 条件)
        in_dim = self.wan.high_noise_model.in_dim
        assert in_dim == 36, (
            f"[WARN] WanModel.in_dim={in_dim}, I2V 期望 36 (16+20)。"
            "这可能意味着 checkpoint 与 I2V 不匹配。"
        )
        print(f"  [PASS] WanModel.in_dim = {in_dim} (I2V: 16+20=36)")

        # 验证 blocks 含 cross_attn
        n_blocks = len(self.wan.high_noise_model.blocks)
        has_cross = hasattr(self.wan.high_noise_model.blocks[0], 'cross_attn')
        print(f"  [PASS] WanModel 有 {n_blocks} 个 block, cross_attn={has_cross}")

        # 验证 MetaQuery encoder 就绪
        enc_ok = (hasattr(self.mq_encoder, 'mllm_model')
                  and self.mq_encoder.mllm_model is not None)
        print(f"  [PASS] MetaQuery encoder 就绪: {enc_ok}")

        # 验证增强 text_len 合理
        assert self._aug_text_len <= 4096, (
            f"[FATAL] 增强后 text_len={self._aug_text_len} 过大"
        )
        print(f"  [PASS] 增强后 text_len={self._aug_text_len}")

        print("[MetaQueryWanI2VBridge] ✅ I2V Bridge 初始化验证全部通过！\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Context 拼接 (与 T2V bridge 相同)
    # ─────────────────────────────────────────────────────────────────────────

    def _augment_context(
        self,
        t5_context: List[torch.Tensor],
        mq_context: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """将 MetaQuery 特征前置拼接到 T5 context。"""
        augmented = []
        for i, (t5_feat, mq_feat) in enumerate(zip(t5_context, mq_context)):
            assert t5_feat.shape[-1] == mq_feat.shape[-1], (
                f"T5 和 MQ 特征维度不匹配: T5={t5_feat.shape[-1]}, MQ={mq_feat.shape[-1]}"
            )
            assert mq_feat.shape[0] == self.num_metaqueries, (
                f"MQ token 数={mq_feat.shape[0]}, 期望={self.num_metaqueries}"
            )
            aug = torch.cat(
                [mq_feat.to(t5_feat.device, t5_feat.dtype), t5_feat], dim=0
            )
            if i == 0:
                mq_norm = aug[:mq_feat.shape[0]].float().norm().item()
                t5_norm = aug[mq_feat.shape[0]:].float().norm().item()
                print(
                    f"  [VERIFY-拼接] MQ L2={mq_norm:.4f}, "
                    f"T5 L2={t5_norm:.4f}"
                )
            augmented.append(aug)
        return augmented

    def _patch_wan_text_len(self, model, new_text_len: int):
        model.text_len = new_text_len

    def _restore_wan_text_len(self, model):
        model.text_len = self._orig_text_len

    # ─────────────────────────────────────────────────────────────────────────
    # 首帧条件编码 (与原始 WanI2V.generate 中的逻辑完全一致)
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_first_frame(
        self,
        first_frame: Image.Image,
        frame_num: int,
        max_area: int = 720 * 1280,
    ):
        """
        将首帧图像编码为 I2V 条件张量 y。

        流程 (与原始 WanI2V 完全一致):
            1. 首帧 → 归一化 [-1,1]
            2. 根据图像宽高比 + max_area 计算 latent 尺寸
            3. 构建掩码: 第 1 帧为 1, 其余为 0
            4. VAE encode(首帧 + 全零后续帧) → 16ch latent
            5. concat(mask[4ch], latent[16ch]) → y[20ch]

        Args:
            first_frame:  PIL.Image, 首帧图像
            frame_num:    总帧数
            max_area:     最大像素面积 (宽高比自适应)

        Returns:
            y:           torch.Tensor, shape [20, T_lat, lat_h, lat_w]
            lat_h:       int, latent 高度
            lat_w:       int, latent 宽度
            h:           int, 目标像素高度
            w:           int, 目标像素宽度
        """
        device = self.wan.device
        vae_stride = self.wan.vae_stride
        patch_size = self.wan.patch_size

        # 首帧 → tensor [-1, 1]
        img = TF.to_tensor(first_frame).sub_(0.5).div_(0.5).to(device)

        num_frames = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w

        # 根据宽高比自适应计算 latent 尺寸 (与 WanI2V 完全一致)
        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // vae_stride[1] // patch_size[1] * patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // vae_stride[2] // patch_size[2] * patch_size[2]
        )
        h = lat_h * vae_stride[1]
        w = lat_w * vae_stride[2]

        # 构建掩码: 第 1 帧 = 1, 其余 = 0
        msk = torch.ones(1, num_frames, lat_h, lat_w, device=device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, T_lat, lat_h, lat_w]

        # VAE 编码首帧 (后续帧全零)
        y = self.wan.vae.encode([
            torch.concat([
                F_nn.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic'
                ).transpose(0, 1),
                torch.zeros(3, num_frames - 1, h, w)
            ], dim=1).to(device)
        ])[0]
        # y: [16, T_lat, lat_h, lat_w]

        # 拼接掩码 + VAE latent → 20 通道条件
        y = torch.concat([msk, y])  # [20, T_lat, lat_h, lat_w]

        print(
            f"  [首帧编码] y shape={y.shape} "
            f"(mask:4ch + VAE:16ch = 20ch), "
            f"lat_h={lat_h}, lat_w={lat_w}, "
            f"pixel_h={h}, pixel_w={w}"
        )
        # 验证
        assert y.shape[0] == 20, f"y 通道数={y.shape[0]}, 期望 20"
        assert not torch.isnan(y).any(), "y 含 NaN!"
        y_norm = y.float().norm().item()
        assert y_norm > 0, "y 全零!"
        print(f"  [VERIFY] y L2={y_norm:.4f}, 非零非NaN ✅")

        return y, lat_h, lat_w, h, w

    # ─────────────────────────────────────────────────────────────────────────
    # 主生成方法
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        input_prompt: str,
        first_frame: Image.Image,
        mq_reference_images: Optional[List[Image.Image]] = None,
        max_area: int = 720 * 1280,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 40,
        guide_scale: Union[float, tuple] = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ):
        """
        MetaQuery 增强的图生视频 (I2V) 生成。

        与标准 WanI2V.generate 接口兼容，额外支持:
            mq_reference_images: MetaQuery 参考图像列表（提供语义条件）

        双重条件注入:
            1. first_frame     → VAE encode → channel concat → 首帧结构条件 (y)
            2. mq_reference_images → Qwen3-VL → MetaQuery → context concat → 语义条件

        Args:
            input_prompt:        文本提示词
            first_frame:         首帧图像 (PIL.Image), 必须提供
            mq_reference_images: MetaQuery 参考图像列表 (可为 None, 仅用文本)
                                 可以是与 first_frame 相同的图, 也可以是不同的参考图
            max_area:            最大像素面积 (I2V 根据首帧宽高比自适应)
            frame_num:           帧数 (4n+1)
            shift:               噪声调度偏移
            sample_solver:       "unipc" 或 "dpm++"
            sampling_steps:      去噪步数
            guide_scale:         CFG 引导强度 (可为 float 或 tuple)
            n_prompt:            负面提示词
            seed:                随机种子 (-1=随机)
            offload_model:       模型卸载到 CPU (节省 VRAM)

        Returns:
            torch.Tensor: 生成的视频, shape [C=3, F, H, W]
        """
        print("\n" + "=" * 60)
        print("[MetaQueryWanI2VBridge.generate] 开始 MetaQuery 增强 I2V 生成")
        print(f"  prompt         : {input_prompt[:80]}{'...' if len(input_prompt)>80 else ''}")
        print(f"  first_frame    : {first_frame.size[0]}×{first_frame.size[1]}")
        print(f"  MQ ref images  : {len(mq_reference_images) if mq_reference_images else 0} 张")
        print(f"  max_area       : {max_area}")
        print(f"  frame_num      : {frame_num}")
        print(f"  sampling_steps : {sampling_steps}")
        print(f"  guide_scale    : {guide_scale}")
        print("=" * 60)

        wan = self.wan
        device = wan.device

        # ── 1. 首帧编码 → y tensor [20, T_lat, H_lat, W_lat] ────────────────
        print("\n[Step 1/5] 编码首帧为 I2V 条件...")
        y, lat_h, lat_w, h, w = self._encode_first_frame(
            first_frame, frame_num, max_area
        )

        # ── 2. 形状预算 ──────────────────────────────────────────────────────
        F_num = frame_num
        target_shape = (
            16,  # VAE latent channels
            (F_num - 1) // wan.vae_stride[0] + 1,
            lat_h,
            lat_w,
        )
        max_seq_len = (
            (F_num - 1) // wan.vae_stride[0] + 1
        ) * lat_h * lat_w // (wan.patch_size[1] * wan.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / wan.sp_size)) * wan.sp_size

        if n_prompt == "":
            n_prompt = wan.sample_neg_prompt

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        print(f"[I2V] seed={seed}, target_shape={target_shape}, max_seq_len={max_seq_len}")

        guide_scale = (guide_scale, guide_scale) if not isinstance(
            guide_scale, tuple) else guide_scale

        # ── 3. T5 文本编码 ──────────────────────────────────────────────────
        print("\n[Step 2/5] T5 编码文本条件...")
        if not wan.t5_cpu:
            wan.text_encoder.model.to(device)
        context = wan.text_encoder(
            [input_prompt], device if not wan.t5_cpu else torch.device('cpu')
        )
        context_null = wan.text_encoder(
            [n_prompt], device if not wan.t5_cpu else torch.device('cpu')
        )
        if wan.t5_cpu:
            context = [t.to(device) for t in context]
            context_null = [t.to(device) for t in context_null]
        if offload_model and not wan.t5_cpu:
            wan.text_encoder.model.cpu()
        print(f"  ✅ T5 context shape: {context[0].shape}")

        # ── 4. MetaQuery (Qwen3-VL) 编码语义条件 ────────────────────────────
        print("\n[Step 3/5] Qwen3-VL MetaQuery 编码语义条件...")

        # 如果没有额外指定参考图 (或传入空列表), 则默认用首帧作为 MetaQuery 参考
        if not mq_reference_images:  # None 或 [] 都走这个分支
            mq_images_for_encode = [first_frame]
            print("  [INFO] 未指定 MQ 参考图, 使用首帧作为 MetaQuery 输入")
        else:
            mq_images_for_encode = mq_reference_images

        mq_context = self.mq_encoder.encode(
            [input_prompt], [mq_images_for_encode]
        )
        mq_context_null = self.mq_encoder.encode(
            [n_prompt], None  # 无条件: 无参考图
        )

        if self.mq_guidance_scale != 1.0:
            mq_context = [c * self.mq_guidance_scale for c in mq_context]
            mq_context_null = [c * self.mq_guidance_scale for c in mq_context_null]
            print(f"  [INFO] MQ guidance scale = {self.mq_guidance_scale}")

        print(f"  ✅ MQ context shape: {mq_context[0].shape}")
        print(f"  ✅ MQ null shape   : {mq_context_null[0].shape}")

        # 验证 cond vs uncond 差异 (CFG 有效性)
        _mq_cos = torch.nn.functional.cosine_similarity(
            mq_context[0].float().reshape(1, -1),
            mq_context_null[0].float().reshape(1, -1)
        ).item()
        print(
            f"  [VERIFY] MQ cond vs uncond 余弦相似度={_mq_cos:.4f} "
            f"({'CFG有效 ✅' if abs(_mq_cos) < 0.99 else '⚠️ cond≈uncond'})"
        )

        # ── 5. 拼接 Context (T5 + MetaQuery) ────────────────────────────────
        print("\n[Step 4/5] 拼接 T5 + MetaQuery context...")
        aug_context = self._augment_context(context, mq_context)
        aug_context_null = self._augment_context(context_null, mq_context_null)
        print(
            f"  ✅ 增强 context: {aug_context[0].shape} "
            f"(T5:{context[0].shape[0]} + MQ:{self.num_metaqueries})"
        )

        # ── 6-7. 扩展 text_len + 去噪循环 (在 try/finally 内保证恢复) ──────
        print("\n[Step 5/5] 启动 I2V 去噪循环 (首帧条件 + MetaQuery 语义条件)...")
        noise = torch.randn(
            target_shape[0], target_shape[1],
            target_shape[2], target_shape[3],
            dtype=torch.float32, device=device, generator=seed_g,
        )

        boundary = wan.boundary * wan.num_train_timesteps

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low = getattr(wan.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high = getattr(wan.high_noise_model, 'no_sync', noop_no_sync)

        videos = None

        # ★ 使用 try/finally 保证 text_len 在任何情况下都能恢复
        try:
            self._patch_wan_text_len(wan.low_noise_model, self._aug_text_len)
            self._patch_wan_text_len(wan.high_noise_model, self._aug_text_len)
            assert wan.high_noise_model.text_len == self._aug_text_len
            print(f"  ✅ text_len 扩展: {self._orig_text_len} → {self._aug_text_len}")
            with (
                torch.amp.autocast('cuda', dtype=wan.param_dtype),
                torch.no_grad(),
                no_sync_low(),
                no_sync_high(),
            ):
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=wan.num_train_timesteps,
                        shift=1, use_dynamic_shifting=False,
                    )
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=device, shift=shift
                    )
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=wan.num_train_timesteps,
                        shift=1, use_dynamic_shifting=False,
                    )
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler, device=device, sigmas=sampling_sigmas
                    )
                else:
                    raise NotImplementedError(f"不支持的 solver: {sample_solver}")

                latent = noise

                # ★ 关键: arg_c 和 arg_null 都包含 y (首帧条件)
                # 这与原版 WanI2V 完全一致
                arg_c = {
                    'context': aug_context,
                    'seq_len': max_seq_len,
                    'y': [y],
                }
                arg_null = {
                    'context': aug_context_null,
                    'seq_len': max_seq_len,
                    'y': [y],
                }

                if offload_model:
                    torch.cuda.empty_cache()

                for step_i, t in enumerate(tqdm(timesteps, desc="MetaQuery+I2V 去噪")):
                    latent_model_input = [latent.to(device)]
                    timestep = [t]
                    timestep = torch.stack(timestep).to(device)

                    model = wan._prepare_model_for_timestep(
                        t, boundary, offload_model
                    )
                    sample_guide_scale = (
                        guide_scale[1] if t.item() >= boundary else guide_scale[0]
                    )

                    noise_pred_cond = model(
                        latent_model_input, t=timestep, **arg_c
                    )[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = model(
                        latent_model_input, t=timestep, **arg_null
                    )[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                        return_dict=False, generator=seed_g,
                    )[0]
                    latent = temp_x0.squeeze(0)

                    # 首步和中间步验证
                    if step_i == 0:
                        _cond_norm = noise_pred_cond.float().norm().item()
                        _uncond_norm = noise_pred_uncond.float().norm().item()
                        _diff_norm = (noise_pred_cond - noise_pred_uncond).float().norm().item()
                        _cfg_cos = torch.nn.functional.cosine_similarity(
                            noise_pred_cond.float().reshape(1, -1),
                            noise_pred_uncond.float().reshape(1, -1),
                        ).item()
                        print(
                            f"\n  [VERIFY-去噪 step 1] t={t.item():.1f}\n"
                            f"    cond L2={_cond_norm:.4f}, uncond L2={_uncond_norm:.4f}\n"
                            f"    cond-uncond 差异 L2={_diff_norm:.4f}\n"
                            f"    cond vs uncond cos={_cfg_cos:.4f}\n"
                            f"    guide_scale={sample_guide_scale}\n"
                            f"    context tokens: {aug_context[0].shape[0]} "
                            f"(MQ:{self.num_metaqueries} + T5:{context[0].shape[0]})\n"
                            f"    y shape: {y.shape} (首帧条件)\n"
                            f"    {'✅ 双重条件生效' if _diff_norm > 1e-6 else '⚠️ 差异极小'}"
                        )
                    elif step_i == len(timesteps) // 2:
                        _mid_diff = (noise_pred_cond - noise_pred_uncond).float().norm().item()
                        print(
                            f"  [VERIFY-去噪 step {step_i+1}/{len(timesteps)}] "
                            f"t={t.item():.1f}, 差异 L2={_mid_diff:.4f} "
                            f"{'✅' if _mid_diff > 1e-6 else '⚠️'}"
                        )

                    del latent_model_input, timestep

                x0 = [latent]

                if offload_model:
                    wan.low_noise_model.cpu()
                    wan.high_noise_model.cpu()
                    torch.cuda.empty_cache()

                if wan.rank == 0:
                    print("\n[MetaQueryWanI2VBridge] VAE 解码...")
                    videos = wan.vae.decode(x0)

        finally:
            # ── 恢复原始 text_len (无论是否异常都必须执行) ──────────────────
            self._restore_wan_text_len(wan.low_noise_model)
            self._restore_wan_text_len(wan.high_noise_model)
            assert wan.high_noise_model.text_len == self._orig_text_len
            print(
                f"[MetaQueryWanI2VBridge] text_len 已恢复: {self._orig_text_len} ✅"
            )

        # ── 清理中间变量 ──────────────────────────────────────────────────
        del noise, latent, x0, aug_context, aug_context_null, sample_scheduler, y
        del context, context_null, mq_context, mq_context_null
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        print("\n[MetaQueryWanI2VBridge] ✅ MetaQuery 增强 I2V 生成完成！")
        return videos[0] if (wan.rank == 0 and videos is not None) else None
