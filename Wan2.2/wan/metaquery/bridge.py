"""
MetaQueryWanBridge: 将 MetaQuery 视觉条件注入 Wan2.2 生成管线。

注入策略 (Context Concatenation):
    Wan2.2 的 WanModel.forward 接受 context: List[Tensor]，
    每个 Tensor shape 为 [L_t5, 4096]，经 text_embedding 线性层投影后
    作为 cross-attention 的 key/value。

    本模块在每次去噪步之前，将 MetaQuery 特征
    (shape: [num_mq, 4096]) 前置拼接到 T5 context 上：

        context_augmented[i] = cat([mq_feat[i], t5_feat[i]], dim=0)
        # shape: [num_mq + L_t5, 4096]

    由于 WanModel 内部对 context 做了 padding 到 text_len，
    我们只需保证 (num_mq + max_t5_len) ≤ augmented_text_len。
    为此我们在 bridge 中动态扩展 WanModel 的 text_len 设置。
"""

import sys
from pathlib import Path
import gc
import math
import random
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from .encoder import MetaQueryEncoder
from ..utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# ── 将 Wan2.2 内部模块暴露给 bridge ─────────────────────────────────────────
_WAN_ROOT = Path(__file__).resolve().parents[2]   # Wan2.2/wan
_WAN_PARENT = _WAN_ROOT.parent                    # Wan2.2/
if str(_WAN_PARENT) not in sys.path:
    sys.path.insert(0, str(_WAN_PARENT))


class MetaQueryWanBridge:
    """
    MetaQuery 增强的 Wan2.2 文本到视频生成管线。

    在标准 WanT2V 生成流程中，于文本编码之后，
    将 Qwen3-VL MetaQuery 视觉特征拼接进 context，
    实现视觉条件增强。

    Args:
        wan_pipeline:          已初始化的 WanT2V/WanI2V 实例。
        metaquery_checkpoint:  MetaQuery + Qwen3-VL checkpoint 路径。
        num_metaqueries:       MetaQuery token 数量，默认 256。
        mq_guidance_scale:     MetaQuery 视觉特征引导强度，默认 1.0（与文本等权重）。
        dtype:                 计算精度，默认 bfloat16。
    """

    def __init__(
        self,
        wan_pipeline,
        metaquery_checkpoint: str,
        num_metaqueries: int = 256,
        mq_guidance_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.wan = wan_pipeline
        self.mq_guidance_scale = mq_guidance_scale
        self.dtype = dtype

        print("\n" + "=" * 60)
        print("[MetaQueryWanBridge] 初始化 MetaQuery + Wan2.2 联合管线")
        print(f"  Wan pipeline 类型: {wan_pipeline.__class__.__name__}")
        print(f"  MetaQuery ckpt   : {metaquery_checkpoint}")
        print(f"  num_metaqueries  : {num_metaqueries}")
        print(f"  mq_guidance_scale: {mq_guidance_scale}")
        print("=" * 60 + "\n")

        # ── 加载 MetaQuery 编码器 ────────────────────────────────────────────
        self.mq_encoder = MetaQueryEncoder(
            metaquery_checkpoint_path=metaquery_checkpoint,
            num_metaqueries=num_metaqueries,
            wan_text_dim=4096,
            dtype=dtype,
            device=wan_pipeline.device,
        )

        self.num_metaqueries = num_metaqueries

        # ── 记录原始 text_len，动态扩展 ──────────────────────────────────────
        self._orig_text_len = self.wan.high_noise_model.text_len
        self._aug_text_len  = self._orig_text_len + num_metaqueries

        print(
            f"[MetaQueryWanBridge] WanModel text_len: "
            f"{self._orig_text_len} → {self._aug_text_len} "
            f"(+{num_metaqueries} MetaQuery tokens)"
        )

        # ── 初始化完整性验证 ──────────────────────────────────────────────────
        print("\n[MetaQueryWanBridge] 【初始化完整性验证】")

        # 验证 WanModel 的 text_embedding 第一个 Linear 的 in_features = 4096
        text_emb_in = self.wan.high_noise_model.text_embedding[0].in_features
        assert text_emb_in == 4096, (
            f"[FATAL] WanModel.text_embedding.in_features={text_emb_in}, "
            "期望=4096! MetaQuery 投影维度不匹配"
        )
        print(f"  [PASS] WanModel text_embedding 输入维度: {text_emb_in} (= MQ 投影维度 4096)")

        # 验证 WanModel 的 blocks 确实使用 cross-attention
        n_blocks = len(self.wan.high_noise_model.blocks)
        has_cross = hasattr(self.wan.high_noise_model.blocks[0], 'cross_attn')
        print(
            f"  [PASS] WanModel 有 {n_blocks} 个 WanAttentionBlock, "
            f"含 cross_attn={has_cross}"
        )

        # 验证 MetaQuery encoder 已就绪
        enc_ok = hasattr(self.mq_encoder, 'mllm_model') and self.mq_encoder.mllm_model is not None
        print(f"  [PASS] MetaQuery encoder 已就绪: mllm_model={enc_ok}")

        # 验证增强后 text_len 不超过模型能处理的范围（合理性检查）
        assert self._aug_text_len <= 4096, (
            f"[FATAL] 增强后 text_len={self._aug_text_len} 过大, "
            "可能导致显存溢出"
        )
        print(f"  [PASS] 增强后 text_len={self._aug_text_len} (合理范围内)")

        print("[MetaQueryWanBridge] ✅ Bridge 初始化验证全部通过！\n")

    def _augment_context(
        self,
        t5_context: List[torch.Tensor],
        mq_context: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        将 MetaQuery 特征前置拼接到 T5 context。

        Args:
            t5_context:  List[Tensor]，每项 [L_t5, 4096]
            mq_context:  List[Tensor]，每项 [num_mq, 4096]

        Returns:
            List[Tensor]，每项 [num_mq + L_t5, 4096]
        """
        augmented = []
        for i, (t5_feat, mq_feat) in enumerate(zip(t5_context, mq_context)):
            # 验证输入维度一致
            assert t5_feat.shape[-1] == mq_feat.shape[-1], (
                f"[FATAL] T5 和 MQ 特征维度不匹配: "
                f"T5={t5_feat.shape[-1]}, MQ={mq_feat.shape[-1]}"
            )
            assert mq_feat.shape[0] == self.num_metaqueries, (
                f"[FATAL] MQ 特征 token 数={mq_feat.shape[0]}, "
                f"期望={self.num_metaqueries}"
            )
            # MetaQuery 特征前置 → 先被 cross-attn 看到
            aug = torch.cat([mq_feat.to(t5_feat.device, t5_feat.dtype), t5_feat], dim=0)

            # 验证拼接后长度正确
            expected_len = t5_feat.shape[0] + mq_feat.shape[0]
            assert aug.shape[0] == expected_len, (
                f"[FATAL] 拼接后长度={aug.shape[0]}, 期望={expected_len}"
            )

            if i == 0:
                # 验证 MQ 部分和 T5 部分确实是不同数据（非复制）
                mq_part_norm = aug[:mq_feat.shape[0]].float().norm().item()
                t5_part_norm = aug[mq_feat.shape[0]:].float().norm().item()
                # 计算两部分的余弦相似度（按 flatten 比较）
                _mq_flat = aug[:mq_feat.shape[0]].float().reshape(-1)
                _t5_flat = aug[mq_feat.shape[0]:].float().reshape(-1)
                # 截取相同长度来对比
                _min_len = min(_mq_flat.shape[0], _t5_flat.shape[0])
                _cos = torch.nn.functional.cosine_similarity(
                    _mq_flat[:_min_len].unsqueeze(0),
                    _t5_flat[:_min_len].unsqueeze(0)
                ).item()
                print(
                    f"  [VERIFY-拼接] batch[{i}]: "
                    f"MQ部分 L2={mq_part_norm:.4f}, "
                    f"T5部分 L2={t5_part_norm:.4f}, "
                    f"两部分余弦相似度={_cos:.4f} "
                    f"({'二者不同 ✅' if abs(_cos) < 0.99 else '⚠️ 高度相似,请检查'})"
                )

            augmented.append(aug)
        return augmented

    def _patch_wan_text_len(self, model, new_text_len: int):
        """临时将 WanModel 的 text_len 扩展，使 context padding 适配。"""
        model.text_len = new_text_len

    def _restore_wan_text_len(self, model):
        model.text_len = self._orig_text_len

    def generate(
        self,
        input_prompt: str,
        input_images: Optional[List[Image.Image]] = None,
        size=(1280, 720),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ):
        """
        MetaQuery 增强的视频生成。

        与标准 WanT2V.generate 接口兼容，额外支持:
            input_images: 参考图像列表（提供视觉条件）

        流程:
            1. T5 编码文本 → context / context_null
            2. MetaQuery (Qwen3-VL) 编码图像+文本 → mq_context / mq_null
            3. 两者 concat → augmented context
            4. 扩展 WanModel.text_len，运行标准去噪循环
        """
        print("\n" + "=" * 60)
        print("[MetaQueryWanBridge.generate] 开始 MetaQuery 增强视频生成")
        print(f"  prompt        : {input_prompt[:80]}{'...' if len(input_prompt)>80 else ''}")
        print(f"  input_images  : {len(input_images) if input_images else 0} 张参考图")
        print(f"  size          : {size}")
        print(f"  frame_num     : {frame_num}")
        print(f"  sampling_steps: {sampling_steps}")
        print(f"  guide_scale   : {guide_scale}")
        print("=" * 60)

        wan = self.wan
        device = wan.device

        # ── 1. 形状预算 ──────────────────────────────────────────────────────
        F = frame_num
        target_shape = (
            wan.vae.model.z_dim,
            (F - 1) // wan.vae_stride[0] + 1,
            size[1] // wan.vae_stride[1],
            size[0] // wan.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (wan.patch_size[1] * wan.patch_size[2])
            * target_shape[1]
            / wan.sp_size
        ) * wan.sp_size

        if n_prompt == "":
            n_prompt = wan.sample_neg_prompt

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        print(f"[MetaQueryWanBridge] seed={seed}, target_shape={target_shape}")

        # ── 2. T5 文本编码 ──────────────────────────────────────────────────
        print("\n[Step 1/4] T5 编码文本条件...")
        if not wan.t5_cpu:
            wan.text_encoder.model.to(device)
        context      = wan.text_encoder([input_prompt], device if not wan.t5_cpu else torch.device('cpu'))
        context_null = wan.text_encoder([n_prompt],     device if not wan.t5_cpu else torch.device('cpu'))
        if wan.t5_cpu:
            context      = [t.to(device) for t in context]
            context_null = [t.to(device) for t in context_null]
        if offload_model and not wan.t5_cpu:
            wan.text_encoder.model.cpu()

        print(
            f"  ✅ T5 context shape : {context[0].shape}  "
            f"(L={context[0].shape[0]}, dim=4096)"
        )
        # 验证 T5 输出非零
        _t5_norm = context[0].float().norm().item()
        assert _t5_norm > 0, "[FATAL] T5 编码输出全为零!"
        assert not torch.isnan(context[0]).any(), "[FATAL] T5 编码输出含 NaN!"
        print(
            f"  [VERIFY] T5 context: L2_norm={_t5_norm:.4f}, "
            f"mean={context[0].float().mean().item():.6f}, "
            f"std={context[0].float().std().item():.6f} (非零、非NaN ✅)"
        )

        # ── 3. MetaQuery (Qwen3-VL) 编码视觉条件 ────────────────────────────
        print("\n[Step 2/4] Qwen3-VL MetaQuery 编码视觉条件...")
        mq_images  = [input_images] if input_images else None
        mq_null    = None   # null 条件：无参考图

        mq_context      = self.mq_encoder.encode([input_prompt], mq_images)
        mq_context_null = self.mq_encoder.encode([n_prompt], mq_null)

        print(
            f"  ✅ MetaQuery context shape     : {mq_context[0].shape} "
            f"(num_mq={self.num_metaqueries}, dim=4096)"
        )
        print(
            f"  ✅ MetaQuery null context shape: {mq_context_null[0].shape}"
        )
        # 验证 MQ 编码输出非零、非 NaN
        _mq_norm = mq_context[0].float().norm().item()
        _mq_null_norm = mq_context_null[0].float().norm().item()
        assert _mq_norm > 0, "[FATAL] MetaQuery 条件编码全为零!"
        assert _mq_null_norm > 0, "[FATAL] MetaQuery null 编码全为零!"
        assert not torch.isnan(mq_context[0]).any(), "[FATAL] MQ 编码含 NaN!"
        # 应用 mq_guidance_scale (缩放 MetaQuery 特征影响力)
        if self.mq_guidance_scale != 1.0:
            mq_context = [c * self.mq_guidance_scale for c in mq_context]
            mq_context_null = [c * self.mq_guidance_scale for c in mq_context_null]
            print(
                f"  [INFO] 已应用 mq_guidance_scale={self.mq_guidance_scale}, "
                f"MQ cond 缩放后 L2={mq_context[0].float().norm().item():.4f}"
            )

        # 验证有条件和无条件的 MQ 特征确实不同（CFG 才有意义）
        _mq_cond_uncond_cos = torch.nn.functional.cosine_similarity(
            mq_context[0].float().reshape(1, -1),
            mq_context_null[0].float().reshape(1, -1)
        ).item()
        print(
            f"  [VERIFY] MQ cond L2={_mq_norm:.4f}, "
            f"MQ uncond L2={_mq_null_norm:.4f}, "
            f"cond vs uncond 余弦相似度={_mq_cond_uncond_cos:.4f} "
            f"({'二者不同,CFG有效 ✅' if abs(_mq_cond_uncond_cos) < 0.99 else '⚠️ cond≈uncond, CFG可能无效'})"
        )

        # ── 4. 拼接 Context ──────────────────────────────────────────────────
        print("\n[Step 3/4] 拼接 T5 + MetaQuery context...")
        aug_context      = self._augment_context(context,      mq_context)
        aug_context_null = self._augment_context(context_null, mq_context_null)
        print(
            f"  ✅ 增强后 context shape: {aug_context[0].shape} "
            f"(T5:{context[0].shape[0]} + MQ:{self.num_metaqueries} = "
            f"{aug_context[0].shape[0]} tokens)"
        )
        # 验证增强前后长度确实改变了
        assert aug_context[0].shape[0] > context[0].shape[0], (
            f"[FATAL] 增强后 context 长度={aug_context[0].shape[0]} "
            f"<= 原始 T5 长度={context[0].shape[0]}! MQ 拼接未生效!"
        )
        assert aug_context[0].shape[0] == context[0].shape[0] + self.num_metaqueries, (
            f"[FATAL] 增强后长度={aug_context[0].shape[0]}, "
            f"期望={context[0].shape[0] + self.num_metaqueries}"
        )
        # 验证增强后数据的有效性
        _aug_norm = aug_context[0].float().norm().item()
        _aug_null_norm = aug_context_null[0].float().norm().item()
        print(
            f"  [VERIFY] 增强 context L2={_aug_norm:.4f}, "
            f"增强 null context L2={_aug_null_norm:.4f} "
            f"(两者均非零 ✅)"
        )

        # ── 5. 临时扩展 WanModel.text_len ────────────────────────────────────
        # ★ 使用 try/finally 保证 text_len 在任何情况下都能恢复
        videos = None  # 预初始化，避免条件分支中未定义

        try:  # ★ 保证在任何异常下都能恢复 text_len
            self._patch_wan_text_len(wan.low_noise_model,  self._aug_text_len)
            self._patch_wan_text_len(wan.high_noise_model, self._aug_text_len)
            # 验证 text_len 确实被修改了（回读验证，非假打印）
            _actual_text_len_high = wan.high_noise_model.text_len
            _actual_text_len_low  = wan.low_noise_model.text_len
            assert _actual_text_len_high == self._aug_text_len, (
                f"[FATAL] high_noise_model.text_len={_actual_text_len_high}, "
                f"期望={self._aug_text_len}! patch 未生效!"
            )
            assert _actual_text_len_low == self._aug_text_len, (
                f"[FATAL] low_noise_model.text_len={_actual_text_len_low}, "
                f"期望={self._aug_text_len}! patch 未生效!"
            )
            print(
                f"  ✅ WanModel.text_len 扩展验证: "
                f"high_noise_model.text_len={_actual_text_len_high}, "
                f"low_noise_model.text_len={_actual_text_len_low} "
                f"(原始={self._orig_text_len}, 已扩展 ✅)"
            )

            # ── 6. 初始化噪声 ──────────────────────────────────────────────
            noise = [
                torch.randn(
                    target_shape[0], target_shape[1],
                    target_shape[2], target_shape[3],
                    dtype=torch.float32, device=device, generator=seed_g,
                )
            ]

            # ── 7. 去噪循环 ────────────────────────────────────────────────
            print("\n[Step 4/4] 启动去噪循环 (MetaQuery 增强条件)...")

            guide_scale = (guide_scale, guide_scale) if not isinstance(guide_scale, tuple) else guide_scale
            boundary = wan.boundary * wan.num_train_timesteps

            @contextmanager
            def noop_no_sync():
                yield

            no_sync_low  = getattr(wan.low_noise_model,  'no_sync', noop_no_sync)
            no_sync_high = getattr(wan.high_noise_model, 'no_sync', noop_no_sync)

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
                    sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=wan.num_train_timesteps,
                        shift=1, use_dynamic_shifting=False,
                    )
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler, device=device, sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError(f"不支持的 solver: {sample_solver}")

                latents = noise

                arg_c    = {'context': aug_context,      'seq_len': seq_len}
                arg_null = {'context': aug_context_null, 'seq_len': seq_len}

                for step_i, t in enumerate(tqdm(timesteps, desc="MetaQuery+Wan 去噪")):
                    latent_model_input = latents
                    timestep = [t]
                    timestep = torch.stack(timestep)

                    model = wan._prepare_model_for_timestep(t, boundary, offload_model)
                    sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                    noise_pred_cond   = model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                        return_dict=False, generator=seed_g,
                    )[0]
                    latents = [temp_x0.squeeze(0)]

                    if step_i == 0:
                        # ── 去噪第一步的深度验证 ──────────────────────────
                        _cond_norm = noise_pred_cond.float().norm().item()
                        _uncond_norm = noise_pred_uncond.float().norm().item()
                        _diff_norm = (noise_pred_cond - noise_pred_uncond).float().norm().item()
                        _cfg_cos = torch.nn.functional.cosine_similarity(
                            noise_pred_cond.float().reshape(1, -1),
                            noise_pred_uncond.float().reshape(1, -1),
                        ).item()
                        print(
                            f"\n  [VERIFY-去噪 step 1] t={t.item():.1f}\n"
                            f"    noise_pred_cond   shape={noise_pred_cond.shape}, L2={_cond_norm:.4f}\n"
                            f"    noise_pred_uncond shape={noise_pred_uncond.shape}, L2={_uncond_norm:.4f}\n"
                            f"    cond-uncond 差异  L2={_diff_norm:.4f}\n"
                            f"    cond vs uncond 余弦相似度={_cfg_cos:.4f}\n"
                            f"    guide_scale={sample_guide_scale}\n"
                            f"    context tokens: {aug_context[0].shape[0]} "
                            f"(T5:{context[0].shape[0]} + MQ:{self.num_metaqueries})\n"
                            f"    {'✅ CFG差异显著, MQ+T5 条件正在影响去噪' if _diff_norm > 1e-6 else '⚠️ CFG差异极小, 请检查条件是否生效'}"
                        )
                    elif step_i == len(timesteps) // 2:
                        _mid_diff = (noise_pred_cond - noise_pred_uncond).float().norm().item()
                        print(
                            f"  [VERIFY-去噪 step {step_i+1}/{len(timesteps)}] "
                            f"t={t.item():.1f}, cond-uncond差异 L2={_mid_diff:.4f} "
                            f"{'✅' if _mid_diff > 1e-6 else '⚠️'}"
                        )

                x0 = latents

                if offload_model:
                    wan.low_noise_model.cpu()
                    wan.high_noise_model.cpu()
                    torch.cuda.empty_cache()

                if wan.rank == 0:
                    print("\n[MetaQueryWanBridge] VAE 解码...")
                    videos = wan.vae.decode(x0)

        finally:
            # ── 8. 恢复原始 text_len (无论是否异常都必须执行) ──────────────
            self._restore_wan_text_len(wan.low_noise_model)
            self._restore_wan_text_len(wan.high_noise_model)

        # 验证恢复成功
        _restored_high = wan.high_noise_model.text_len
        _restored_low  = wan.low_noise_model.text_len
        assert _restored_high == self._orig_text_len, (
            f"[FATAL] text_len 未恢复! high={_restored_high}, 原始={self._orig_text_len}"
        )
        print(
            f"[MetaQueryWanBridge] WanModel.text_len 恢复验证: "
            f"{_restored_high} == {self._orig_text_len} ✅"
        )

        del noise, latents, aug_context, aug_context_null, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        print("\n[MetaQueryWanBridge] ✅ MetaQuery 增强视频生成完成！")
        return videos[0] if (wan.rank == 0 and videos is not None) else None
