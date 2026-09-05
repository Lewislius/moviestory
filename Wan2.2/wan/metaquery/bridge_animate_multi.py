"""
MultiGPU MetaQueryWanAnimateBridge
==================================
将 MetaQuery + Wan2.2 Animate 各组件分散到多张 GPU，解决单卡 OOM 问题。

★ 核心特性: 支持 DiT 14B 跨双卡拆分
  DiT 14B 权重 ~28GB，推理时激活 ~5-10GB，单张 48GB 可能不够。
  本方案将 DiT 的 40 个 Transformer block 拆分到 2 张卡:
    - GPU 0: embeddings + blocks 0-19 + motion_encoder + face_adapter(前半)
    - GPU 1: blocks 20-39 + face_adapter(后半) + head

显存分配策略 (4×4090 48GB 推荐):
    GPU 0 (dit_0) : DiT 前半 (~15GB)
    GPU 1 (dit_1) : DiT 后半 (~15GB)
    GPU 2 (aux)   : T5-XXL (~10GB) + CLIP (~1.2GB) + VAE (~0.5GB)
    GPU 3 (mq)    : MetaQuery Qwen3-VL + Connector (~5GB)

也支持 DiT 不拆分 (3 卡方案):
    GPU 0 (dit)   : DiT 全部 (~35GB)
    GPU 1 (aux)   : T5 + CLIP + VAE
    GPU 2 (mq)    : MetaQuery

GPU_MAP 配置:
    dit: int 或 [int, int]
      - int:       DiT 全部放在该 GPU
      - [int,int]: DiT 前半放 [0], 后半放 [1]
    t5, clip_vae, mq: int (各放一张卡)
"""

import sys
import gc
import math
import random
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F_nn
from copy import deepcopy
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from .encoder import MetaQueryEncoder
from ..modules.animate.animate_utils import TensorList
from ..modules.model import sinusoidal_embedding_1d
from ..utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ─────────────────────────────────────────────────────────────────────────────
# DiT 双卡拆分
# ─────────────────────────────────────────────────────────────────────────────

def split_dit_across_gpus(noise_model, dit_gpus: List[int], split_idx: int = 20):
    """
    将 WanAnimateModel 的 40 个 block 拆分到 2 张 GPU。

    布局:
        dit_gpus[0]: patch_embedding, pose_patch_embedding, text_embedding,
                     time_embedding, time_projection, img_emb,
                     motion_encoder, face_encoder,
                     blocks[0:split_idx], face_adapter.fuser_blocks[前半]
        dit_gpus[1]: blocks[split_idx:40], face_adapter.fuser_blocks[后半],
                     head

    Args:
        noise_model:  WanAnimateModel 实例
        dit_gpus:     [gpu_id_first, gpu_id_second]
        split_idx:    拆分位置 (默认 20, 即前后各 20 blocks)

    Returns:
        (dev_first, dev_second, split_idx)
    """
    dev0 = torch.device(f"cuda:{dit_gpus[0]}")
    dev1 = torch.device(f"cuda:{dit_gpus[1]}")

    n_blocks = len(noise_model.blocks)
    assert 0 < split_idx < n_blocks, f"split_idx={split_idx} 超出范围 [1, {n_blocks-1}]"

    print(f"\n[DiT Split] 将 {n_blocks} 个 block 拆分:")
    print(f"  cuda:{dit_gpus[0]}: 嵌入层 + blocks[0:{split_idx}] + motion_encoder")
    print(f"  cuda:{dit_gpus[1]}: blocks[{split_idx}:{n_blocks}] + head")

    # ── 嵌入层 → dev0 ─────────────────────────────────────────────────────
    noise_model.patch_embedding.to(dev0)
    noise_model.pose_patch_embedding.to(dev0)
    noise_model.text_embedding.to(dev0)
    noise_model.time_embedding.to(dev0)
    noise_model.time_projection.to(dev0)
    noise_model.img_emb.to(dev0)
    noise_model.motion_encoder.to(dev0)
    noise_model.face_encoder.to(dev0)

    # ── blocks 拆分 ───────────────────────────────────────────────────────
    for i, block in enumerate(noise_model.blocks):
        dev = dev0 if i < split_idx else dev1
        block.to(dev)

    # ── face_adapter.fuser_blocks 拆分 ────────────────────────────────────
    # fuser_blocks[j] 在 block_idx = j*5 时被调用
    for j, fuser_block in enumerate(noise_model.face_adapter.fuser_blocks):
        block_idx = j * 5
        dev = dev0 if block_idx < split_idx else dev1
        fuser_block.to(dev)

    # ── head → dev1 ──────────────────────────────────────────────────────
    noise_model.head.to(dev1)

    # ── freqs 复制到两个设备 ──────────────────────────────────────────────
    noise_model._freqs_dev0 = noise_model.freqs.to(dev0)
    noise_model._freqs_dev1 = noise_model.freqs.to(dev1)

    torch.cuda.empty_cache()

    # 打印分配结果
    mem0 = torch.cuda.memory_allocated(dit_gpus[0]) / 1024**3
    mem1 = torch.cuda.memory_allocated(dit_gpus[1]) / 1024**3
    print(f"  cuda:{dit_gpus[0]} 已分配: {mem0:.1f} GB")
    print(f"  cuda:{dit_gpus[1]} 已分配: {mem1:.1f} GB")
    print("[DiT Split] 拆分完成 ✅\n")

    return dev0, dev1, split_idx


def install_split_forward(noise_model, split_idx: int, dev0, dev1):
    """
    Monkey-patch WanAnimateModel.forward，使其支持跨双卡推理。

    关键: 在 block 循环中，当 idx == split_idx 时，将所有张量从 dev0 迁移到 dev1。
    """
    # 保存原始 forward 以便恢复
    noise_model._original_forward = noise_model.forward

    def split_forward(
        x, t, clip_fea, context, seq_len,
        y=None, pose_latents=None, face_pixel_values=None,
    ):
        model = noise_model

        # ── Phase 1: 嵌入 (全在 dev0) ──────────────────────────────────
        model.freqs = model._freqs_dev0

        # 确保输入在 dev0
        if y is not None:
            x = [torch.cat([u.to(dev0), v.to(dev0)], dim=0) for u, v in zip(x, y)]
        else:
            x = [u.to(dev0) for u in x]
        t = t.to(dev0)
        clip_fea = clip_fea.to(dev0)
        context = [c.to(dev0) for c in context]
        if pose_latents is not None:
            pose_latents = pose_latents.to(dev0)
        if face_pixel_values is not None:
            face_pixel_values = face_pixel_values.to(dev0)

        # patch embedding
        x = [model.patch_embedding(u.unsqueeze(0)) for u in x]
        x, motion_vec = model.after_patch_embedding(x, pose_latents, face_pixel_values)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # time embedding
        with amp.autocast(dtype=torch.float32):
            e = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim, t).float()
            )
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))

        # context embedding
        context_lens = None
        context = model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        if model.use_img_emb:
            context_clip = model.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=model.freqs,
            context=context,
            context_lens=context_lens,
        )

        # ── Phase 2: Block 循环 (跨双卡) ───────────────────────────────
        for idx, block in enumerate(model.blocks):
            if idx == split_idx:
                # ★ 跨 GPU 传输: dev0 → dev1
                x = x.to(dev1)
                e = e.to(dev1)
                e0 = e0.to(dev1)
                context = context.to(dev1)
                motion_vec = motion_vec.to(dev1)
                model.freqs = model._freqs_dev1
                kwargs = dict(
                    e=e0,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=model.freqs,
                    context=context,
                    context_lens=context_lens,
                )

            x = block(x, **kwargs)
            x = model.after_transformer_block(idx, x, motion_vec)

        # ── Phase 3: Head (在 dev1) ─────────────────────────────────────
        x = model.head(x, e)

        # unpatchify
        x = model.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    noise_model.forward = split_forward
    print("[DiT Split] 已安装跨双卡 forward ✅")


def uninstall_split_forward(noise_model):
    """恢复原始 forward。"""
    if hasattr(noise_model, '_original_forward'):
        noise_model.forward = noise_model._original_forward
        del noise_model._original_forward


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数: 将 WanAnimate 子模块迁移到目标 GPU
# ─────────────────────────────────────────────────────────────────────────────

def redistribute_wan_components(wan, gpu_map: Dict[str, Union[int, List[int]]]):
    """
    将 WanAnimate 的各子模块迁移到 gpu_map 指定的 GPU。

    Args:
        wan:      已初始化的 WanAnimate 实例。
        gpu_map:  GPU 分配字典:
                  - "dit": int 或 [int, int] — 当为 list 时, DiT 拆分到 2 张卡
                  - "t5": int
                  - "clip_vae": int
                  - "mq": int

    Returns:
        (dev_dit_input, is_split)
          dev_dit_input: DiT 接收输入的设备 (即 dit 或 dit[0])
          is_split: 是否已拆分 DiT
    """
    dit_val      = gpu_map['dit']
    dev_t5       = torch.device(f"cuda:{gpu_map['t5']}")
    dev_clip_vae = torch.device(f"cuda:{gpu_map['clip_vae']}")
    is_split     = isinstance(dit_val, (list, tuple))

    print("\n[MultiGPU] 重新分配模型到各 GPU...")

    # ── T5 ────────────────────────────────────────────────────────────────
    wan.text_encoder.model.to(dev_t5)
    wan.text_encoder.device = dev_t5
    print(f"  T5-XXL          → cuda:{gpu_map['t5']}")

    # ── CLIP ──────────────────────────────────────────────────────────────
    wan.clip.model.to(dev_clip_vae)
    wan.clip.device = dev_clip_vae
    print(f"  CLIP ViT-H/14   → cuda:{gpu_map['clip_vae']}")

    # ── VAE ───────────────────────────────────────────────────────────────
    wan.vae.model.to(dev_clip_vae)
    wan.vae.device = dev_clip_vae
    wan.vae.mean = wan.vae.mean.to(dev_clip_vae)
    wan.vae.std  = wan.vae.std.to(dev_clip_vae)
    wan.vae.scale = [wan.vae.mean, 1.0 / wan.vae.std]
    print(f"  VAE             → cuda:{gpu_map['clip_vae']}")

    # ── DiT ───────────────────────────────────────────────────────────────
    if is_split:
        # DiT 拆分到 2 张卡
        dev_first, dev_second, split_idx = split_dit_across_gpus(
            wan.noise_model, dit_val, split_idx=20
        )
        install_split_forward(wan.noise_model, split_idx, dev_first, dev_second)
        wan.device = dev_first  # Pipeline 设备指向 DiT 输入端
        dev_dit_input = dev_first
    else:
        dev_dit = torch.device(f"cuda:{dit_val}")
        wan.noise_model.to(dev_dit)
        wan.device = dev_dit
        dev_dit_input = dev_dit
        print(f"  DiT (14B)       → cuda:{dit_val}")

    torch.cuda.empty_cache()
    print("[MultiGPU] 模型重分配完成 ✅\n")
    return dev_dit_input, is_split


# ─────────────────────────────────────────────────────────────────────────────
# 多卡 Bridge
# ─────────────────────────────────────────────────────────────────────────────

class MultiGPUMetaQueryAnimateBridge:
    """
    多 GPU 版 MetaQuery + Wan2.2 Animate 管线。

    与单卡版 MetaQueryWanAnimateBridge 功能相同，但各模型组件
    分布在不同 GPU 上，通过显式 .to(device) 在编码/去噪/解码阶段
    将张量传输到正确设备。

    ★ 支持 DiT 双卡拆分: gpu_map["dit"] = [0, 1] 时，40 个 block
      前 20 个在 GPU 0，后 20 个在 GPU 1，单卡显存需求降至 ~15GB。

    Args:
        wan_animate_pipeline:   已初始化的 WanAnimate
        gpu_map:                GPU 分配字典:
                                  dit: int 或 [int,int] — 支持双卡拆分
                                  t5: int
                                  clip_vae: int
                                  mq: int
        metaquery_checkpoint:   MetaQuery checkpoint 路径
        num_metaqueries:        MetaQuery token 数
        mq_guidance_scale:      MetaQuery 引导强度
        dtype:                  计算精度
        mllm_id:                Qwen3-VL 模型 ID
        diffusion_model_id:     Diffusion model ID
        connector_num_hidden_layers: Connector 层数
    """

    def __init__(
        self,
        wan_animate_pipeline,
        gpu_map: Dict[str, int],
        metaquery_checkpoint: Optional[str] = None,
        num_metaqueries: int = 256,
        mq_guidance_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        mllm_id: Optional[str] = None,
        diffusion_model_id: Optional[str] = None,
        connector_num_hidden_layers: int = 24,
    ):
        self.wan = wan_animate_pipeline
        self.gpu_map = gpu_map
        self.mq_guidance_scale = mq_guidance_scale
        self.dtype = dtype
        self.num_metaqueries = num_metaqueries

        # ── 设备映射 ─────────────────────────────────────────────────────
        dit_val = gpu_map['dit']
        self.dit_is_split = isinstance(dit_val, (list, tuple))
        self.dev_t5       = torch.device(f"cuda:{gpu_map['t5']}")
        self.dev_clip_vae = torch.device(f"cuda:{gpu_map['clip_vae']}")
        self.dev_mq       = torch.device(f"cuda:{gpu_map['mq']}")

        print("\n" + "=" * 60)
        print("[MultiGPU Bridge] 初始化 MetaQuery + Wan2.2 Animate (多卡)")
        print(f"  GPU 分配:")
        if self.dit_is_split:
            print(f"    DiT 前半      → cuda:{dit_val[0]}  (blocks 0-19)")
            print(f"    DiT 后半      → cuda:{dit_val[1]}  (blocks 20-39)")
        else:
            print(f"    DiT (14B)     → cuda:{dit_val}")
        print(f"    T5-XXL        → cuda:{gpu_map['t5']}")
        print(f"    CLIP + VAE    → cuda:{gpu_map['clip_vae']}")
        print(f"    MetaQuery     → cuda:{gpu_map['mq']}")
        print(f"  MetaQuery ckpt  : {metaquery_checkpoint or '(无, 直接初始化)'}")
        print("=" * 60 + "\n")

        # ── 1. 重新分配 WanAnimate 组件到各 GPU ──────────────────────────
        dev_dit_input, _ = redistribute_wan_components(wan_animate_pipeline, gpu_map)
        self.dev_dit = dev_dit_input  # DiT 输入端设备 (即 dit 或 dit[0])

        # ── 2. 在指定 GPU 上加载 MetaQuery 编码器 ────────────────────────
        self.mq_encoder = MetaQueryEncoder(
            metaquery_checkpoint_path=metaquery_checkpoint,
            num_metaqueries=num_metaqueries,
            wan_text_dim=4096,
            dtype=dtype,
            device=self.dev_mq,   # ★ MetaQuery 独占一张 GPU
            mllm_id=mllm_id,
            diffusion_model_id=diffusion_model_id,
            connector_num_hidden_layers=connector_num_hidden_layers,
        )

        # ── 3. text_len 管理 ─────────────────────────────────────────────
        self._orig_text_len = self.wan.noise_model.text_len
        self._aug_text_len  = self._orig_text_len + num_metaqueries

        print(
            f"[MultiGPU Bridge] text_len: "
            f"{self._orig_text_len} → {self._aug_text_len} "
            f"(+{num_metaqueries} MQ tokens)"
        )

        # ── 4. 打印各 GPU 显存占用 ───────────────────────────────────────
        self._print_gpu_memory()
        print("[MultiGPU Bridge] ✅ 初始化完成!\n")

    # ─────────────────────────────────────────────────────────────────────
    # 辅助方法 (与单卡版相同)
    # ─────────────────────────────────────────────────────────────────────

    def _print_gpu_memory(self):
        """打印各 GPU 的显存占用。"""
        gpu_map = self.gpu_map
        dit_val = gpu_map['dit']
        all_ids = set()
        if isinstance(dit_val, (list, tuple)):
            all_ids.update(dit_val)
        else:
            all_ids.add(dit_val)
        all_ids.add(gpu_map['t5'])
        all_ids.add(gpu_map['clip_vae'])
        all_ids.add(gpu_map['mq'])

        print("\n[MultiGPU] 各 GPU 显存使用:")
        for gpu_id in sorted(all_ids):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved  = torch.cuda.memory_reserved(gpu_id) / 1024**3
            roles = []
            if isinstance(dit_val, (list, tuple)):
                if gpu_id == dit_val[0]:
                    roles.append("dit_front")
                if gpu_id == dit_val[1]:
                    roles.append("dit_back")
            elif gpu_id == dit_val:
                roles.append("dit")
            for key in ('t5', 'clip_vae', 'mq'):
                if gpu_map[key] == gpu_id:
                    roles.append(key)
            print(
                f"  cuda:{gpu_id} ({', '.join(roles)}): "
                f"已分配 {allocated:.1f} GB / 已预留 {reserved:.1f} GB"
            )

    def _augment_context(
        self,
        t5_context: List[torch.Tensor],
        mq_context: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """将 MetaQuery 特征前置拼接到 T5 context (两者应已在同一设备)。"""
        augmented = []
        for i, (t5_feat, mq_feat) in enumerate(zip(t5_context, mq_context)):
            aug = torch.cat(
                [mq_feat.to(t5_feat.device, t5_feat.dtype), t5_feat], dim=0
            )
            augmented.append(aug)
        return augmented

    def _patch_wan_text_len(self, model, new_text_len: int):
        model.text_len = new_text_len

    def _restore_wan_text_len(self, model):
        model.text_len = self._orig_text_len

    def _padding_resize(
        self,
        img_ori: np.ndarray,
        height: int = 512,
        width: int = 512,
        padding_color=(0, 0, 0),
    ) -> np.ndarray:
        """等比缩放 + 居中填充到目标尺寸。"""
        ori_h, ori_w = img_ori.shape[:2]
        channel = img_ori.shape[2] if len(img_ori.shape) > 2 else 1
        img_pad = np.zeros(
            (height, width, channel if channel > 1 else 1), dtype=np.uint8
        )
        for c_idx in range(min(channel, 3)):
            img_pad[:, :, c_idx] = (
                padding_color[c_idx] if c_idx < len(padding_color) else 0
            )
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
        """循环镜像填充序列至目标长度。"""
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
        self, lat_t, lat_h, lat_w, mask_len=1, device="cuda"
    ) -> torch.Tensor:
        """构造 I2V 条件掩码。"""
        msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        msk[:, :mask_len] = 1
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def _load_face_video(self, face_source, target_len: int) -> list:
        """加载面部视频帧 → list[np.ndarray(512,512,3)]。"""
        if isinstance(face_source, str):
            from decord import VideoReader
            vr = VideoReader(face_source)
            face_frames = list(vr.get_batch(list(range(len(vr)))).asnumpy())
        elif isinstance(face_source, list):
            if len(face_source) > 0 and isinstance(face_source[0], Image.Image):
                face_frames = [np.array(f) for f in face_source]
            else:
                face_frames = list(face_source)
        else:
            raise TypeError(f"face_source 类型不支持: {type(face_source)}")

        resized = []
        for frame in face_frames:
            if frame.shape[0] != 512 or frame.shape[1] != 512:
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            resized.append(frame)

        if len(resized) < target_len:
            resized = self._inputs_padding(resized, target_len)
        else:
            resized = resized[:target_len]
        return resized

    # ─────────────────────────────────────────────────────────────────────
    # 主生成方法 (多 GPU 版)
    # ─────────────────────────────────────────────────────────────────────

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
        offload_model: bool = False,  # 多卡下默认不需要 offload
    ):
        """
        多 GPU 版 MetaQuery 增强人物动画生成。

        各编码器在各自 GPU 上运行，编码结果通过 .to(dev_dit) 传输到
        DiT GPU，去噪循环只在 DiT GPU 上进行。

        Args:
            (与单卡版 generate() 参数完全相同)

        Returns:
            torch.Tensor: [C=3, F, H, W] 或 None
        """
        assert refert_num in (1, 5), "refert_num 应为 1 或 5"
        assert ref_image is not None, "ref_image 必须提供!"
        assert (frame_num - 1) % 4 == 0, f"frame_num={frame_num} 应为 4n+1!"
        assert (clip_len - 1) % 4 == 0, f"clip_len={clip_len} 应为 4n+1!"

        wan = self.wan
        dev_dit      = self.dev_dit
        dev_t5       = self.dev_t5
        dev_clip_vae = self.dev_clip_vae
        dev_mq       = self.dev_mq

        print("\n" + "=" * 60)
        print("[MultiGPU Generate] 开始 MetaQuery 增强 Animate 生成")
        dit_val = self.gpu_map['dit']
        if self.dit_is_split:
            print(f"  DiT      → cuda:{dit_val[0]} + cuda:{dit_val[1]} (双卡拆分)")
        else:
            print(f"  DiT      → cuda:{dit_val}")
        print(f"  T5       → cuda:{self.gpu_map['t5']}")
        print(f"  CLIP+VAE → cuda:{self.gpu_map['clip_vae']}")
        print(f"  MetaQuery→ cuda:{self.gpu_map['mq']}")
        print(f"  frame_num: {frame_num}, guide_scale: {guide_scale}")
        print("=" * 60)

        if input_prompt == "":
            input_prompt = wan.sample_prompt
        if n_prompt == "":
            n_prompt = wan.sample_neg_prompt

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=dev_dit)
        seed_g.manual_seed(seed)
        print(f"[MultiGPU] seed={seed}")

        # ── Step 1: 准备参考图 (CPU) ────────────────────────────────────
        print("\n[Step 1/7] 准备参考人物图...")
        ref_np = np.array(ref_image)
        height = max((ref_np.shape[0] // 8) * 8, 512)
        width  = max((ref_np.shape[1] // 8) * 8, 512)
        ref_np = self._padding_resize(ref_np, height=height, width=width)
        print(f"  参考图尺寸: {height}×{width}")

        # ── Step 2: 准备面部帧 (CPU) ────────────────────────────────────
        print("\n[Step 2/7] 准备面部视频帧...")
        if face_source is not None:
            face_frames = self._load_face_video(face_source, frame_num)
            print(f"  面部帧数: {len(face_frames)}")
        else:
            face_frames = [np.zeros((512, 512, 3), dtype=np.uint8)] * frame_num
            print("  未提供面部视频，使用全零帧")

        # ── Step 3: T5 编码 (dev_t5) → 结果传到 dev_dit ────────────────
        print(f"\n[Step 3/7] T5 编码文本 (cuda:{self.gpu_map['t5']})...")
        context = wan.text_encoder([input_prompt], dev_t5)
        context_null = wan.text_encoder([n_prompt], dev_t5)
        # ★ 传输到 DiT GPU
        context      = [c.to(dev_dit) for c in context]
        context_null = [c.to(dev_dit) for c in context_null]
        print(f"  ✅ T5 context → cuda:{self.gpu_map['dit']}, shape: {context[0].shape}")

        # ── Step 4: MetaQuery 编码 (dev_mq) → 结果传到 dev_dit ─────────
        print(f"\n[Step 4/7] MetaQuery 编码 (cuda:{self.gpu_map['mq']})...")
        mq_images = mq_reference_images if mq_reference_images else [ref_image]
        mq_context      = self.mq_encoder.encode([input_prompt], [mq_images])
        mq_context_null = self.mq_encoder.encode([n_prompt], None)

        if self.mq_guidance_scale != 1.0:
            mq_context      = [c * self.mq_guidance_scale for c in mq_context]
            mq_context_null = [c * self.mq_guidance_scale for c in mq_context_null]

        # ★ 传输到 DiT GPU
        mq_context      = [c.to(dev_dit) for c in mq_context]
        mq_context_null = [c.to(dev_dit) for c in mq_context_null]
        print(f"  ✅ MQ context → cuda:{self.gpu_map['dit']}, shape: {mq_context[0].shape}")

        # ── Step 5: 拼接 Context (dev_dit) ──────────────────────────────
        print("\n[Step 5/7] 拼接 T5 + MetaQuery context...")
        aug_context      = self._augment_context(context, mq_context)
        aug_context_null = self._augment_context(context_null, mq_context_null)
        print(
            f"  ✅ 增强 context shape: {aug_context[0].shape} "
            f"(MQ:{self.num_metaqueries} + T5:{context[0].shape[0]})"
        )

        # ── Step 6: CLIP 编码 (dev_clip_vae) → 结果传到 dev_dit ────────
        print(f"\n[Step 6/7] CLIP 编码 (cuda:{self.gpu_map['clip_vae']})...")
        ref_tensor_clip = torch.tensor(
            ref_np / 127.5 - 1, dtype=torch.bfloat16, device=dev_clip_vae
        )
        ref_tensor_clip = rearrange(ref_tensor_clip, "h w c -> c h w")
        clip_context = wan.clip.visual(
            [ref_tensor_clip[:, None, :, :]]
        ).to(dtype=torch.bfloat16, device=dev_dit)  # ★ 传到 DiT GPU
        del ref_tensor_clip
        print(f"  ✅ CLIP context → cuda:{self.gpu_map['dit']}, shape: {clip_context.shape}")

        # ── Step 7: 逐 clip 去噪循环 ────────────────────────────────────
        print(f"\n[Step 7/7] 去噪循环 (DiT on cuda:{self.gpu_map['dit']})...")

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
        out_frames_on_vae = None  # VAE 解码输出, 在 dev_clip_vae 上
        clip_idx = 0

        noise = latents = x0 = face_pixel_values = pose_latents = None
        sample_scheduler = None

        try:
            self._patch_wan_text_len(wan.noise_model, self._aug_text_len)
            print(f"  text_len 扩展: {self._orig_text_len} → {self._aug_text_len}")

            with (
                torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
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
                        f"mask_reft={mask_reft_len} ---"
                    )

                    # ── 面部张量 → dev_dit ────────────────────────────────
                    face_clip = face_frames[start:end]
                    if len(face_clip) < clip_len:
                        face_clip = self._inputs_padding(face_clip, clip_len)
                    face_pixel_values = rearrange(
                        torch.tensor(
                            np.stack(face_clip) / 127.5 - 1,
                            dtype=torch.bfloat16,
                        ),
                        "t h w c -> 1 c t h w",
                    ).to(dev_dit)

                    # ── 参考图 VAE 编码 (dev_clip_vae) → dev_dit ─────────
                    ref_pv = rearrange(
                        torch.tensor(ref_np / 127.5 - 1, dtype=torch.bfloat16),
                        "h w c -> 1 c h w",
                    ).to(dev_clip_vae)
                    ref_pv_5d = rearrange(ref_pv, "b c h w -> b c 1 h w")
                    ref_latents = wan.vae.encode(ref_pv_5d.to(torch.bfloat16))
                    ref_latents = torch.stack(ref_latents).to(dev_dit)  # ★ → DiT
                    del ref_pv, ref_pv_5d

                    T = clip_len
                    lat_t = T // 4 + 1
                    target_shape = [lat_t + 1, lat_h, lat_w]

                    # y_ref: 参考图条件
                    mask_ref = self._get_i2v_mask(1, lat_h, lat_w, 1, device=dev_dit)
                    y_ref = torch.concat([mask_ref, ref_latents[0]]).to(
                        dtype=torch.bfloat16, device=dev_dit
                    )

                    # y_reft: 时序引导帧
                    if mask_reft_len > 0:
                        # 从上一 clip VAE 解码结果取最后几帧 (在 dev_clip_vae 上)
                        refer_t_pv = rearrange(
                            out_frames_on_vae[0, :, -refert_num:].clone().detach(),
                            "c t h w -> t c h w",
                        )
                        reft_frames = F_nn.interpolate(
                            refer_t_pv[:mask_reft_len].cpu().permute(1, 0, 2, 3),
                            size=(height, width),
                            mode="bicubic",
                        )
                        y_reft = wan.vae.encode([
                            torch.concat([
                                reft_frames,
                                torch.zeros(3, T - mask_reft_len, height, width),
                            ], dim=1).to(dev_clip_vae)
                        ])[0].to(dev_dit)  # ★ → DiT
                        msk_reft = self._get_i2v_mask(
                            lat_t, lat_h, lat_w, mask_reft_len, device=dev_dit
                        )
                    else:
                        y_reft = wan.vae.encode([
                            torch.zeros(3, T, height, width).to(dev_clip_vae)
                        ])[0].to(dev_dit)  # ★ → DiT
                        msk_reft = self._get_i2v_mask(
                            lat_t, lat_h, lat_w, 0, device=dev_dit
                        )

                    y_reft = torch.concat([msk_reft, y_reft]).to(
                        dtype=torch.bfloat16, device=dev_dit
                    )
                    y = torch.concat([y_ref, y_reft], dim=1)

                    # ── 零 pose latent → dev_dit ──────────────────────────
                    pose_latents = torch.zeros(
                        1, 16, lat_t, lat_h, lat_w,
                        dtype=torch.bfloat16, device=dev_dit,
                    )

                    # ── 噪声初始化 → dev_dit ──────────────────────────────
                    noise = [
                        torch.randn(
                            16, target_shape[0], target_shape[1], target_shape[2],
                            dtype=torch.float32, device=dev_dit, generator=seed_g,
                        )
                    ]

                    max_seq_len = int(
                        math.ceil(np.prod(target_shape) // 4 / wan.sp_size)
                    ) * wan.sp_size

                    # ── Solver ─────────────────────────────────────────────
                    if sample_solver == "unipc":
                        sample_scheduler = FlowUniPCMultistepScheduler(
                            num_train_timesteps=wan.num_train_timesteps,
                            shift=1, use_dynamic_shifting=False,
                        )
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=dev_dit, shift=shift
                        )
                        timesteps = sample_scheduler.timesteps
                    elif sample_solver == "dpm++":
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=wan.num_train_timesteps,
                            shift=1, use_dynamic_shifting=False,
                        )
                        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler, device=dev_dit, sigmas=sampling_sigmas
                        )
                    else:
                        raise NotImplementedError(f"不支持: {sample_solver}")

                    latents = noise

                    # ── 去噪参数 (全部在 dev_dit) ─────────────────────────
                    arg_c = {
                        "context": aug_context,
                        "seq_len": max_seq_len,
                        "clip_fea": clip_context,
                        "y": [y],
                        "pose_latents": pose_latents,
                        "face_pixel_values": face_pixel_values,
                    }

                    arg_null = None
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

                    # ── 去噪循环 (全在 dev_dit) ───────────────────────────
                    if self.dit_is_split:
                        dit_desc = f"GPU:{dit_val[0]}+{dit_val[1]}"
                    else:
                        dit_desc = f"GPU:{dit_val}"
                    desc = f"[{dit_desc}] Clip {clip_idx}"
                    for step_i, t in enumerate(tqdm(timesteps, desc=desc)):
                        latent_model_input = latents
                        timestep = torch.stack([t])

                        # ★ 当 DiT 双卡拆分时, forward 输出在 dev1,
                        #   需要 .to(dev_dit) 移回 dev0 与 latents 对齐。
                        #   非拆分时 .to(dev_dit) 是 no-op。
                        noise_pred_cond = TensorList(
                            [u.to(dev_dit) for u in wan.noise_model(
                                TensorList(latent_model_input),
                                t=timestep,
                                **arg_c,
                            )]
                        )

                        if guide_scale > 1:
                            noise_pred_uncond = TensorList(
                                [u.to(dev_dit) for u in wan.noise_model(
                                    TensorList(latent_model_input),
                                    t=timestep,
                                    **arg_null,
                                )]
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

                    # ── VAE 解码: latent → dev_clip_vae → 解码 ───────────
                    x0 = [x.to(dtype=torch.float32, device=dev_clip_vae) for x in latents]
                    out_frames_on_vae = torch.stack(
                        wan.vae.decode([x0[0][:, 1:]])
                    )  # 在 dev_clip_vae 上

                    if start != 0:
                        out_frames_on_vae = out_frames_on_vae[:, :, refert_num:]

                    all_out_frames.append(out_frames_on_vae.cpu())
                    # out_frames_on_vae 留在 dev_clip_vae 供下个 clip 取时序引导帧
                    print(
                        f"    ✅ Clip {clip_idx} 完成, 帧数: {out_frames_on_vae.shape[2]}"
                    )

                    start += clip_len - refert_num
                    end   += clip_len - refert_num

        finally:
            self._restore_wan_text_len(wan.noise_model)
            print(f"[MultiGPU] text_len 已恢复: {self._orig_text_len}")

        # ── 拼接所有 clip ────────────────────────────────────────────────
        videos = torch.cat(all_out_frames, dim=2)[:, :, :frame_num]

        # ── 清理 ─────────────────────────────────────────────────────────
        del aug_context, aug_context_null
        del context, context_null, mq_context, mq_context_null
        del clip_context, out_frames_on_vae
        del noise, latents, x0, face_pixel_values, pose_latents, sample_scheduler
        gc.collect()
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        self._print_gpu_memory()
        print(
            f"\n[MultiGPU] ✅ 生成完成! 总帧数: {videos.shape[2]}"
        )
        return videos[0] if wan.rank == 0 else None
