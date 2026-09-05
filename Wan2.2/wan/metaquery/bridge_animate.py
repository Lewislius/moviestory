"""
MetaQueryWanAnimateBridge: 将 MetaQuery 视觉条件注入 Wan2.2 Animate (人物动画) 管线。

注入策略 (三重条件):
    1. **参考图条件 (Channel Concatenation)** — 与原始 WanAnimate 一致:
       参考人物图经 VAE 编码 → 20ch y tensor，在 WanAnimateModel.forward
       中与噪声 latent (16ch) channel-cat 成 36ch 输入。

    2. **CLIP 视觉条件 (Image Embedding)** — 与原始 WanAnimate 一致:
       参考图经 CLIP ViT-H/14 编码得全局 257 token，在 WanAnimateCrossAttention
       中独立的 k_img/v_img 处理后与文本注意力输出相加。

    3. **MetaQuery 语义条件 (Context Concatenation)** — 与 T2V/I2V bridge 一致:
       MetaQuery (Qwen3-VL) 从参考图像提取 256 个语义 token (dim=4096)，
       前置拼接到 T5 文本 context 上，参与 WanAnimateModel 所有层的 cross-attention。

    4. **面部条件 (Face Adapter)** — 与原始 WanAnimate 一致:
       面部视频帧经 motion_encoder → face_encoder → face_adapter，
       每 5 个 transformer block 通过交叉注意力注入面部动作特征。

    注意: 本版本 **不使用** Body Adapter / 骨架姿态信息。
    pose_latents 传入全零张量，不提取视频人物骨架。

    多条件互补:
    - 参考图条件提供像素级结构一致性
    - CLIP 条件提供全局视觉语义
    - MetaQuery 条件提供细粒度视觉语义引导 (风格、内容、构图等)
    - 面部条件提供表情/面部动作驱动
"""

import sys
import gc
import math
import random
from contextlib import contextmanager
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F_nn
import torchvision.transforms.functional as TF
from copy import deepcopy
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from .encoder import MetaQueryEncoder
from ..modules.animate.animate_utils import TensorList
from ..utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class MetaQueryWanAnimateBridge:
    """
    MetaQuery 增强的 Wan2.2 人物动画 (Animate) 管线 — 无骨架 / 有面部。

    在标准 WanAnimate 流程基础上:
      - 保留: 参考图 (I2V-style)、CLIP 视觉编码、面部 Adapter
      - 移除: 骨架 (pose) 条件 — 传零 pose latent
      - 新增: MetaQuery (Qwen3-VL) 语义条件 — context concat 注入

    Args:
        wan_animate_pipeline:   已初始化的 WanAnimate 实例。
        metaquery_checkpoint:   MetaQuery + Qwen3-VL checkpoint 路径。
        num_metaqueries:        MetaQuery token 数量，默认 256。
        mq_guidance_scale:      MetaQuery 引导强度，默认 1.0。
        dtype:                  计算精度，默认 bfloat16。
    """

    def __init__(
        self,
        wan_animate_pipeline,
        metaquery_checkpoint: str,
        num_metaqueries: int = 256,
        mq_guidance_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.wan = wan_animate_pipeline
        self.mq_guidance_scale = mq_guidance_scale
        self.dtype = dtype

        print("\n" + "=" * 60)
        print("[MetaQueryWanAnimateBridge] 初始化 MetaQuery + Wan2.2 Animate 联合管线")
        print(f"  Wan pipeline 类型     : {wan_animate_pipeline.__class__.__name__}")
        print(f"  MetaQuery ckpt        : {metaquery_checkpoint}")
        print(f"  num_metaqueries       : {num_metaqueries}")
        print(f"  mq_guidance_scale     : {mq_guidance_scale}")
        print(f"  模式: 无骨架 + 有面部 + MetaQuery 语义")
        print("=" * 60 + "\n")

        # ── 验证传入的是 WanAnimate 实例 ─────────────────────────────────────
        pipeline_cls = wan_animate_pipeline.__class__.__name__
        assert 'Animate' in pipeline_cls or 'animate' in pipeline_cls.lower(), (
            f"[FATAL] 期望 WanAnimate 实例, 但收到 {pipeline_cls}! "
            "请确认传入了 WanAnimate 对象。"
        )

        # ── 加载 MetaQuery 编码器 ────────────────────────────────────────────
        self.mq_encoder = MetaQueryEncoder(
            metaquery_checkpoint_path=metaquery_checkpoint,
            num_metaqueries=num_metaqueries,
            wan_text_dim=4096,
            dtype=dtype,
            device=wan_animate_pipeline.device,
        )

        self.num_metaqueries = num_metaqueries

        # ── 记录原始 text_len，后续动态扩展 ──────────────────────────────────
        self._orig_text_len = self.wan.noise_model.text_len
        self._aug_text_len = self._orig_text_len + num_metaqueries

        print(
            f"[MetaQueryWanAnimateBridge] WanAnimateModel text_len: "
            f"{self._orig_text_len} → {self._aug_text_len} "
            f"(+{num_metaqueries} MetaQuery tokens)"
        )

        # ── 初始化完整性验证 ──────────────────────────────────────────────────
        print("\n[MetaQueryWanAnimateBridge] 【初始化完整性验证】")

        # 验证 text_embedding 输入维度 = 4096
        text_emb_in = self.wan.noise_model.text_embedding[0].in_features
        assert text_emb_in == 4096, (
            f"[FATAL] WanAnimateModel.text_embedding.in_features={text_emb_in}, "
            "期望=4096! MetaQuery 投影维度不匹配"
        )
        print(f"  [PASS] WanAnimateModel text_embedding 输入维度: {text_emb_in}")

        # 验证 in_dim = 36 (I2V-style: 16 noise + 20 condition)
        in_dim = self.wan.noise_model.in_dim
        assert in_dim == 36, (
            f"[WARN] WanAnimateModel.in_dim={in_dim}, Animate 期望 36 (16+20)。"
        )
        print(f"  [PASS] WanAnimateModel.in_dim = {in_dim} (16+20=36)")

        # 验证含 cross_attn 和 face_adapter
        n_blocks = len(self.wan.noise_model.blocks)
        has_cross = hasattr(self.wan.noise_model.blocks[0], 'cross_attn')
        has_face = hasattr(self.wan.noise_model, 'face_adapter')
        has_motion = hasattr(self.wan.noise_model, 'motion_encoder')
        has_pose = hasattr(self.wan.noise_model, 'pose_patch_embedding')
        print(f"  [PASS] {n_blocks} blocks, cross_attn={has_cross}")
        print(f"  [PASS] face_adapter={has_face}, motion_encoder={has_motion}")
        print(f"  [INFO] pose_patch_embedding={has_pose} (将传零 pose，不使用骨架)")

        # 验证 CLIP 编码器
        has_clip = hasattr(self.wan, 'clip')
        print(f"  [PASS] CLIP 编码器: {has_clip}")

        # 验证 img_emb (CLIP → DiT dim 投影)
        has_img_emb = hasattr(self.wan.noise_model, 'img_emb')
        print(f"  [PASS] img_emb (CLIP投影): {has_img_emb}")

        # 验证 MetaQuery encoder
        enc_ok = (hasattr(self.mq_encoder, 'mllm_model')
                  and self.mq_encoder.mllm_model is not None)
        print(f"  [PASS] MetaQuery encoder 就绪: {enc_ok}")

        # 验证增强 text_len 合理
        assert self._aug_text_len <= 4096, (
            f"[FATAL] 增强后 text_len={self._aug_text_len} 过大"
        )
        print(f"  [PASS] 增强后 text_len={self._aug_text_len}")

        print("[MetaQueryWanAnimateBridge] ✅ Animate Bridge 初始化验证全部通过！\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Context 拼接
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
    # 辅助方法
    # ─────────────────────────────────────────────────────────────────────────

    def _padding_resize(
        self,
        img_ori: np.ndarray,
        height: int = 512,
        width: int = 512,
        padding_color=(0, 0, 0),
    ) -> np.ndarray:
        """将图片等比缩放并居中填充到目标尺寸。"""
        ori_h, ori_w = img_ori.shape[:2]
        channel = img_ori.shape[2] if len(img_ori.shape) > 2 else 1

        img_pad = np.zeros((height, width, channel if channel > 1 else 1), dtype=np.uint8)
        for c_idx in range(min(channel, 3)):
            img_pad[:, :, c_idx] = padding_color[c_idx] if c_idx < len(padding_color) else 0

        if (ori_h / ori_w) > (height / width):
            new_w = int(height / ori_h * ori_w)
            img = cv2.resize(img_ori, (new_w, height), interpolation=cv2.INTER_LINEAR)
            pad = (width - new_w) // 2
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            img_pad[:, pad:pad + new_w, :] = img
        else:
            new_h = int(width / ori_w * ori_h)
            img = cv2.resize(img_ori, (width, new_h), interpolation=cv2.INTER_LINEAR)
            pad = (height - new_h) // 2
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            img_pad[pad:pad + new_h, :, :] = img

        return np.uint8(img_pad)

    @staticmethod
    def _inputs_padding(array: list, target_len: int) -> list:
        """循环镜像填充序列至目标长度 (与 WanAnimate.inputs_padding 一致)。"""
        idx = 0
        flip = False
        target_array = []
        while len(target_array) < target_len:
            target_array.append(deepcopy(array[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return target_array[:target_len]

    def _get_i2v_mask(
        self,
        lat_t: int,
        lat_h: int,
        lat_w: int,
        mask_len: int = 1,
        device: str = "cuda",
    ) -> torch.Tensor:
        """构造 I2V 条件掩码 (与 WanAnimate.get_i2v_mask 一致)。"""
        msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        msk[:, :mask_len] = 1
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def _load_face_video(
        self,
        face_source,
        target_len: int,
    ) -> list:
        """
        加载面部视频帧。

        Args:
            face_source: 可以是:
                - str: 面部视频路径 (.mp4)
                - List[np.ndarray]: 已裁剪的 512×512 面部帧列表
                - List[PIL.Image]: PIL 面部图像列表
            target_len: 需要的目标帧数

        Returns:
            list[np.ndarray]: 面部帧列表 (H=512, W=512, C=3, dtype=uint8)
        """
        if isinstance(face_source, str):
            from decord import VideoReader
            vr = VideoReader(face_source)
            face_idxs = list(range(len(vr)))
            face_frames = vr.get_batch(face_idxs).asnumpy()
            face_frames = list(face_frames)
        elif isinstance(face_source, list):
            if len(face_source) > 0 and isinstance(face_source[0], Image.Image):
                face_frames = [np.array(f) for f in face_source]
            else:
                face_frames = list(face_source)
        else:
            raise TypeError(
                f"face_source 类型不支持: {type(face_source)}. "
                "请传入 str(视频路径), List[np.ndarray], 或 List[PIL.Image]"
            )

        # 确保每帧是 512×512
        resized = []
        for frame in face_frames:
            if frame.shape[0] != 512 or frame.shape[1] != 512:
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            resized.append(frame)

        # 填充到目标长度
        if len(resized) < target_len:
            resized = self._inputs_padding(resized, target_len)
        else:
            resized = resized[:target_len]

        return resized

    # ─────────────────────────────────────────────────────────────────────────
    # 主生成方法
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        input_prompt: str = "",
        ref_image: Optional[Image.Image] = None,
        face_source=None,
        mq_reference_images: Optional[List[Image.Image]] = None,
        frame_num: int = 77,
        clip_len: int = 77,
        refert_num: int = 1,
        shift: float = 5.0,
        sample_solver: str = "dpm++",
        sampling_steps: int = 20,
        guide_scale: float = 1.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ):
        """
        MetaQuery 增强的人物动画生成 (无骨架 + 有面部 + MetaQuery 语义)。

        三/四重条件注入:
            1. ref_image     → VAE encode → channel concat → 参考图条件 (y)
            2. ref_image     → CLIP ViT-H/14 → img_emb → CLIP 视觉条件
            3. mq_reference_images → Qwen3-VL → MetaQuery → context concat → 语义条件
            4. face_source   → motion_encoder → face_encoder → face_adapter → 面部条件

        无骨架: pose_latents 传零张量，不起实际引导作用。

        Args:
            input_prompt:        文本提示词 (默认使用配置中的固定提示)
            ref_image:           参考人物图 (PIL.Image)，必须提供
            face_source:         面部条件源，可以是:
                                 - str: 面部视频路径 (.mp4, 512×512 裁剪面部)
                                 - List[np.ndarray]: 面部帧列表
                                 - List[PIL.Image]: PIL 面部图像列表
                                 - None: 不使用面部条件 (传全零)
            mq_reference_images: MetaQuery 参考图像列表 (可为 None, 默认用 ref_image)
            frame_num:           总帧数 (应为 4n+1)
            clip_len:            每 clip 帧数 (默认 77, 应为 4n+1)
            refert_num:          时序引导帧数 (1 或 5)
            shift:               噪声调度偏移
            sample_solver:       "unipc" 或 "dpm++"
            sampling_steps:      去噪步数
            guide_scale:         CFG 引导强度 (1.0=无 CFG, 仅面部表情调节时>1)
            n_prompt:            负面提示词
            seed:                随机种子 (-1=随机)
            offload_model:       模型卸载到 CPU (节省 VRAM)

        Returns:
            torch.Tensor: 生成的视频, shape [C=3, F, H, W] 或 None (非 rank 0)
        """
        assert refert_num == 1 or refert_num == 5, "refert_num 应为 1 或 5"
        assert ref_image is not None, "ref_image (参考人物图) 必须提供!"
        assert (frame_num - 1) % 4 == 0, f"frame_num={frame_num} 应为 4n+1 格式!"
        assert (clip_len - 1) % 4 == 0, f"clip_len={clip_len} 应为 4n+1 格式!"

        print("\n" + "=" * 60)
        print("[MetaQueryWanAnimateBridge.generate] 开始 MetaQuery 增强 Animate 生成")
        print(f"  prompt         : {input_prompt[:80] if input_prompt else '(默认)'}")
        print(f"  ref_image      : {ref_image.size[0]}×{ref_image.size[1]}")
        print(f"  face_source    : {type(face_source).__name__}")
        print(f"  MQ ref images  : {len(mq_reference_images) if mq_reference_images else 0} 张")
        print(f"  frame_num      : {frame_num}")
        print(f"  clip_len       : {clip_len}")
        print(f"  refert_num     : {refert_num}")
        print(f"  guide_scale    : {guide_scale} {'(CFG 启用)' if guide_scale > 1 else '(无 CFG)'}")
        print(f"  骨架 (pose)    : 禁用 (传零 latent)")
        print(f"  面部 (face)    : {'启用' if face_source is not None else '禁用 (传零)'}")
        print("=" * 60)

        wan = self.wan
        device = wan.device

        # 使用默认 prompt
        if input_prompt == "":
            input_prompt = wan.sample_prompt
        if n_prompt == "":
            n_prompt = wan.sample_neg_prompt

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        print(f"[Animate] seed={seed}")

        # ── 1. 准备参考图像 ──────────────────────────────────────────────────
        print("\n[Step 1/7] 准备参考人物图...")
        ref_np = np.array(ref_image)
        # 确定生成尺寸 (取参考图原始尺寸，但对齐到 8 的倍数)
        height = (ref_np.shape[0] // 8) * 8
        width = (ref_np.shape[1] // 8) * 8
        if height == 0:
            height = 512
        if width == 0:
            width = 512
        ref_np = self._padding_resize(ref_np, height=height, width=width)
        print(f"  参考图尺寸: {height}×{width}")

        # ── 2. 准备面部帧 ────────────────────────────────────────────────────
        print("\n[Step 2/7] 准备面部视频帧...")
        if face_source is not None:
            face_frames = self._load_face_video(face_source, frame_num)
            print(f"  面部帧数: {len(face_frames)}, 尺寸: {face_frames[0].shape}")
        else:
            # 无面部条件，全零 uint8 → 经 /127.5-1 后变全 -1.0
            # 这与 CFG 无条件分支 (face*0-1) 一致，使 face_adapter 不起作用
            face_frames = [np.zeros((512, 512, 3), dtype=np.uint8)] * frame_num
            print("  未提供面部视频，使用全零面部帧 (归一化后=-1, 面部条件不起作用)")

        # ── 3. T5 文本编码 ──────────────────────────────────────────────────
        print("\n[Step 3/7] T5 编码文本条件...")
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
        print("\n[Step 4/7] Qwen3-VL MetaQuery 编码语义条件...")

        if not mq_reference_images:
            mq_images_for_encode = [ref_image]
            print("  [INFO] 未指定 MQ 参考图, 使用参考人物图作为 MetaQuery 输入")
        else:
            mq_images_for_encode = mq_reference_images

        mq_context = self.mq_encoder.encode(
            [input_prompt], [mq_images_for_encode]
        )
        mq_context_null = self.mq_encoder.encode(
            [n_prompt], None
        )

        if self.mq_guidance_scale != 1.0:
            mq_context = [c * self.mq_guidance_scale for c in mq_context]
            mq_context_null = [c * self.mq_guidance_scale for c in mq_context_null]
            print(f"  [INFO] MQ guidance scale = {self.mq_guidance_scale}")
        print(f"  ✅ MQ context shape: {mq_context[0].shape}")
        print(f"  ✅ MQ null shape   : {mq_context_null[0].shape}")

        # CFG 验证
        _mq_cos = torch.nn.functional.cosine_similarity(
            mq_context[0].float().reshape(1, -1),
            mq_context_null[0].float().reshape(1, -1)
        ).item()
        print(
            f"  [VERIFY] MQ cond vs uncond cos={_mq_cos:.4f} "
            f"({'CFG有效 ✅' if abs(_mq_cos) < 0.99 else '⚠️ cond≈uncond'})"
        )

        # ── 5. 拼接 Context (T5 + MetaQuery) ────────────────────────────────
        print("\n[Step 5/7] 拼接 T5 + MetaQuery context...")
        aug_context = self._augment_context(context, mq_context)
        aug_context_null = self._augment_context(context_null, mq_context_null)
        print(
            f"  ✅ 增强 context: {aug_context[0].shape} "
            f"(T5:{context[0].shape[0]} + MQ:{self.num_metaqueries})"
        )

        # ── 8. 逐 clip 去噪生成 ──────────────────────────────────────────────
        print("\n[Step 7/7] 启动逐 clip 去噪循环...")
        print(f"  (参考图条件 + CLIP + MetaQuery 语义 + 面部动作，无骨架)")

        # 对齐帧数到 clip 边界
        real_clip_len = clip_len - refert_num
        last_clip_rem = (frame_num - refert_num) % real_clip_len
        extra = 0 if last_clip_rem == 0 else (real_clip_len - last_clip_rem)
        target_len = frame_num + extra

        if len(face_frames) < target_len:
            face_frames = self._inputs_padding(face_frames, target_len)

        lat_h = height // 8
        lat_w = width // 8

        start = 0
        end = clip_len
        all_out_frames = []
        out_frames = None
        clip_idx = 0

        # 预置循环内变量为 None，防止零迭代时 del 异常
        noise = None
        latents = None
        x0 = None
        face_pixel_values = None
        pose_latents = None
        sample_scheduler = None
        clip_context = None

        try:
            # ── 6. 扩展 WanAnimateModel.text_len (在 try 内，保证 finally 恢复) ─
            self._patch_wan_text_len(wan.noise_model, self._aug_text_len)
            assert wan.noise_model.text_len == self._aug_text_len
            print(f"  ✅ text_len 扩展: {self._orig_text_len} → {self._aug_text_len}")

            # ── 7. CLIP 编码参考图 ────────────────────────────────────────────
            print("\n[Step 6/7] CLIP ViT-H/14 编码参考图...")
            ref_tensor = torch.tensor(ref_np / 127.5 - 1, dtype=torch.bfloat16, device=device)
            ref_tensor = rearrange(ref_tensor, "h w c -> c h w")
            # CLIP.visual 期望 [B, C, T, H, W] 形式 → 我们传 [C, T=1, H, W]
            clip_context = wan.clip.visual([ref_tensor[:, None, :, :]]).to(
                dtype=torch.bfloat16, device=device
            )
            print(f"  ✅ CLIP context: {clip_context.shape} (257 tokens)")
            with (
                torch.autocast(
                    device_type=str(device).split(':')[0],
                    dtype=torch.bfloat16,
                    enabled=True,
                ),
                torch.no_grad(),
            ):
                while True:
                    if start + refert_num >= target_len:
                        break

                    clip_idx += 1
                    mask_reft_len = 0 if start == 0 else refert_num

                    print(
                        f"\n  --- Clip {clip_idx}: frames [{start}:{end}], "
                        f"mask_reft_len={mask_reft_len} ---"
                    )

                    # ---- 构建面部张量 ----
                    face_clip = face_frames[start:end]
                    if len(face_clip) < clip_len:
                        face_clip = self._inputs_padding(face_clip, clip_len)
                    face_pixel_values = rearrange(
                        torch.tensor(
                            np.stack(face_clip) / 127.5 - 1,
                            dtype=torch.bfloat16
                        ),
                        "t h w c -> 1 c t h w",
                    ).to(device)

                    # ---- 参考图 VAE 编码 ----
                    ref_pv = rearrange(
                        torch.tensor(ref_np / 127.5 - 1, dtype=torch.bfloat16),
                        "h w c -> 1 c h w",
                    ).to(device)
                    ref_pv_5d = rearrange(ref_pv, "b c h w -> b c 1 h w")
                    ref_latents = wan.vae.encode(ref_pv_5d.to(torch.bfloat16))
                    ref_latents = torch.stack(ref_latents)

                    T = clip_len
                    lat_t = T // 4 + 1
                    target_shape = [lat_t + 1, lat_h, lat_w]

                    # y_ref: 参考图条件 (mask + VAE latent)
                    mask_ref = self._get_i2v_mask(1, lat_h, lat_w, 1, device=device)
                    y_ref = torch.concat([mask_ref, ref_latents[0]]).to(
                        dtype=torch.bfloat16, device=device
                    )

                    # y_reft: 时序引导帧条件
                    if mask_reft_len > 0:
                        # 使用前一 clip 最后 refert_num 帧
                        # refer_t_pv: [refert_num, 3, H, W] (t c h w)
                        refer_t_pv = rearrange(
                            out_frames[0, :, -refert_num:].clone().detach(),
                            "c t h w -> t c h w",
                        )
                        # refer_t_pv[0, :, :mask_reft_len] → [3, reft, H, W]
                        # 注意: refer_t_pv 已是 [T, C, H, W]，取前 mask_reft_len 帧后
                        # permute(1,0,2,3) → [C, reft, H, W] 给 interpolate 做空间缩放
                        reft_frames = F_nn.interpolate(
                            refer_t_pv[:mask_reft_len].cpu().permute(1, 0, 2, 3),
                            size=(height, width), mode="bicubic"
                        )  # → [3, mask_reft_len, height, width]
                        y_reft = wan.vae.encode([
                            torch.concat([
                                reft_frames,
                                torch.zeros(3, T - mask_reft_len, height, width),
                            ], dim=1).to(device)
                        ])[0]
                        msk_reft = self._get_i2v_mask(
                            lat_t, lat_h, lat_w, mask_reft_len, device=device
                        )
                    else:
                        y_reft = wan.vae.encode([
                            torch.zeros(3, T, height, width).to(device)
                        ])[0]
                        msk_reft = self._get_i2v_mask(
                            lat_t, lat_h, lat_w, 0, device=device
                        )

                    y_reft = torch.concat([msk_reft, y_reft]).to(
                        dtype=torch.bfloat16, device=device
                    )
                    y = torch.concat([y_ref, y_reft], dim=1)

                    print(f"    y shape: {y.shape} (ref:1帧 + reft:{lat_t}帧)")

                    # ---- 零 pose latent (不使用骨架) ----
                    pose_latents = torch.zeros(
                        1, 16, lat_t, lat_h, lat_w,
                        dtype=torch.bfloat16, device=device
                    )

                    # ---- 噪声初始化 ----
                    noise = [
                        torch.randn(
                            16, target_shape[0], target_shape[1], target_shape[2],
                            dtype=torch.float32, device=device, generator=seed_g,
                        )
                    ]

                    max_seq_len = int(
                        math.ceil(np.prod(target_shape) // 4 / wan.sp_size)
                    ) * wan.sp_size
                    if max_seq_len % wan.sp_size != 0:
                        raise ValueError(
                            f"max_seq_len {max_seq_len} 不能被 "
                            f"sp_size {wan.sp_size} 整除"
                        )

                    # ---- Solver 设置 ----
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

                    latents = noise

                    # ---- 去噪参数 ----
                    arg_c = {
                        "context": aug_context,
                        "seq_len": max_seq_len,
                        "clip_fea": clip_context,
                        "y": [y],
                        "pose_latents": pose_latents,
                        "face_pixel_values": face_pixel_values,
                    }

                    if guide_scale > 1:
                        face_pv_uncond = face_pixel_values * 0 - 1
                        arg_null = {
                            "context": aug_context_null,
                            "seq_len": max_seq_len,
                            "clip_fea": clip_context,
                            "y": [y],
                            "pose_latents": pose_latents,
                            "face_pixel_values": face_pv_uncond,
                        }

                    # ---- 去噪循环 ----
                    desc = f"MQ+Animate Clip {clip_idx}"
                    for step_i, t in enumerate(tqdm(timesteps, desc=desc)):
                        latent_model_input = latents
                        timestep = torch.stack([t])

                        noise_pred_cond = TensorList(
                            wan.noise_model(
                                TensorList(latent_model_input),
                                t=timestep,
                                **arg_c,
                            )
                        )

                        if guide_scale > 1:
                            noise_pred_uncond = TensorList(
                                wan.noise_model(
                                    TensorList(latent_model_input),
                                    t=timestep,
                                    **arg_null,
                                )
                            )
                            noise_pred = noise_pred_uncond + guide_scale * (
                                noise_pred_cond - noise_pred_uncond
                            )
                        else:
                            noise_pred = noise_pred_cond

                        temp_x0 = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g,
                        )[0]
                        latents[0] = temp_x0.squeeze(0)

                        # 首步验证
                        if step_i == 0 and clip_idx == 1:
                            _cn = noise_pred_cond[0].float().norm().item()
                            print(
                                f"\n    [VERIFY-去噪 clip1/step1] t={t.item():.1f}\n"
                                f"      cond L2={_cn:.4f}\n"
                                f"      context tokens: {aug_context[0].shape[0]} "
                                f"(MQ:{self.num_metaqueries} + T5:{context[0].shape[0]})\n"
                                f"      y shape: {y.shape}\n"
                                f"      face_pixel_values: {face_pixel_values.shape}\n"
                                f"      pose_latents: zeros (骨架禁用)"
                            )

                    # ---- VAE 解码 ----
                    x0 = latents
                    x0 = [x.to(dtype=torch.float32) for x in x0]
                    out_frames = torch.stack(wan.vae.decode([x0[0][:, 1:]]))

                    if start != 0:
                        out_frames = out_frames[:, :, refert_num:]

                    all_out_frames.append(out_frames.cpu())
                    print(f"    ✅ Clip {clip_idx} 解码完成, 帧数: {out_frames.shape[2]}")

                    start += clip_len - refert_num
                    end += clip_len - refert_num

        finally:
            # ── 恢复原始 text_len (无论是否异常都必须执行) ──────────────────
            self._restore_wan_text_len(wan.noise_model)
            assert wan.noise_model.text_len == self._orig_text_len
            print(
                f"[MetaQueryWanAnimateBridge] text_len 已恢复: "
                f"{self._orig_text_len} ✅"
            )

        # ── 拼接所有 clip ────────────────────────────────────────────────────
        videos = torch.cat(all_out_frames, dim=2)[:, :, :frame_num]

        # ── 清理 ─────────────────────────────────────────────────────────────
        del aug_context, aug_context_null
        del context, context_null, mq_context, mq_context_null
        del clip_context
        # 循环内变量已预置为 None，安全删除
        del noise, latents, x0, face_pixel_values, pose_latents, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        print(
            f"\n[MetaQueryWanAnimateBridge] ✅ MetaQuery 增强 Animate 生成完成！"
            f" 总帧数: {videos.shape[2]}"
        )
        return videos[0] if wan.rank == 0 else None
