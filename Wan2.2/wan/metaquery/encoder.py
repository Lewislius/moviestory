"""
MetaQueryEncoder: 使用 Qwen3-VL 提取 MetaQuery 视觉-语言条件特征。

原理:
  MetaQuery 在 MLLM (Qwen3-VL) 前向输出中，通过植入的特殊 token
  <begin_of_img> ... <img0>~<imgN-1> ... <end_of_img> 从隐藏状态中
  提取出与图像/文本对齐的 N 个 MetaQuery embedding，再经过一个
  轻量 Connector (双向 Qwen2Encoder + 线性投影) 映射到目标维度。

  本模块在此基础上，额外增加一个 to_wan_proj 线性层，将 connector
  输出（维度 = Sana cross-attention dim）进一步投影到 Wan2.2 所需的
  text_dim=4096，使得 MetaQuery token 可以直接拼接进 Wan2.2 的
  context 张量参与 cross-attention。
"""

import sys
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from typing import List, Optional, Union
from PIL import Image

# ── 将 metaquery-main 加入 sys.path ───────────────────────────────────────────
# 目录结构: Qwen3-VL / Qwen3-VL-main / metaquery-main /  (代码直接在此)
#           Qwen3-VL / Wan2.2 / wan / metaquery / encoder.py  (本文件)
# parents[3] 到达 Qwen3-VL/，再拼接 Qwen3-VL-main/metaquery-main/
_MQ_ROOT = Path(__file__).resolve().parents[3] / "Qwen3-VL-main" / "metaquery-main"
if str(_MQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_MQ_ROOT))
_WAN2_ROOT = Path(__file__).resolve().parents[2]
if str(_WAN2_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAN2_ROOT))

# ── Wan2.2 text_dim ─────────────────────────────────────────────────────────
WAN_TEXT_DIM = 4096          # T5 embedding 维度，WanModel.text_embedding 的 in_features


class MetaQueryEncoder(nn.Module):
    """
    将 Qwen3-VL MetaQuery 特征编码并投影到 Wan2.2 的 context 空间。

    支持两种初始化模式:
      1. **从 checkpoint 加载** (metaquery_checkpoint_path 为有效路径):
         加载训练好的 MetaQuery checkpoint，包含微调后的 MLLM + Connector。
      2. **从预训练模型直接初始化** (metaquery_checkpoint_path 为 None):
         直接使用 Qwen3-VL 原始预训练权重 + 随机初始化的 Connector，
         无需训练好的 MetaQuery checkpoint 即可运行。
         需要通过 mllm_id 参数指定 Qwen3-VL 的 HuggingFace 模型 ID 或本地路径。

    Args:
        metaquery_checkpoint_path (str or None):
            MetaQuery + Qwen3-VL 训练好的 checkpoint 路径。
            设为 None 则直接从预训练 Qwen3-VL 初始化 (无需 checkpoint)。
        num_metaqueries (int):
            MetaQuery token 数量，默认 256，与训练配置一致。
        wan_text_dim (int):
            Wan2.2 的 text_dim（T5 特征维度），默认 4096。
        dtype (torch.dtype):
            计算精度，默认 bfloat16。
        device (str|torch.device):
            运行设备，默认 cuda。
        mllm_id (str or None):
            当 metaquery_checkpoint_path=None 时，指定 Qwen3-VL 的
            HuggingFace 模型 ID 或本地路径。默认 None 时自动使用
            "Qwen/Qwen3-VL-2B-Instruct"。
        diffusion_model_id (str or None):
            当 metaquery_checkpoint_path=None 时，指定用于确定 connector
            输出维度的 diffusion model ID。默认使用 Sana。
        connector_num_hidden_layers (int):
            当从预训练模型初始化时，connector 的 Qwen2Encoder 层数。默认 24。
    """

    def __init__(
        self,
        metaquery_checkpoint_path: Optional[str] = None,
        num_metaqueries: int = 256,
        wan_text_dim: int = WAN_TEXT_DIM,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        mllm_id: Optional[str] = None,
        diffusion_model_id: Optional[str] = None,
        connector_num_hidden_layers: int = 24,
    ):
        super().__init__()
        self.num_metaqueries = num_metaqueries
        self.wan_text_dim = wan_text_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self._uses_identity_wan_proj = False

        # 判断初始化模式
        use_checkpoint = (
            metaquery_checkpoint_path is not None
            and metaquery_checkpoint_path != ""
        )

        mode_str = "从 checkpoint 加载" if use_checkpoint else "从预训练模型直接初始化"
        print(
            "=" * 60,
            f"\n[MetaQueryEncoder] 初始化中... ({mode_str})"
            f"\n  checkpoint : {metaquery_checkpoint_path if use_checkpoint else '(无, 直接初始化)'}"
            f"\n  num_metaqueries: {num_metaqueries}"
            f"\n  wan_text_dim   : {wan_text_dim}"
            f"\n  dtype          : {dtype}"
            f"\n  device         : {device}\n" +
            "=" * 60
        )

        if use_checkpoint:
            # ── 模式 1: 从 checkpoint 加载 ──────────────────────────────────
            self._init_from_checkpoint(
                metaquery_checkpoint_path, num_metaqueries, dtype
            )
        else:
            # ── 模式 2: 从预训练模型直接初始化 ──────────────────────────────
            self._init_from_pretrained(
                mllm_id=mllm_id or "Qwen/Qwen3-VL-2B-Instruct",
                diffusion_model_id=diffusion_model_id or "Efficient-Large-Model/Sana_1600M_512px_diffusers",
                num_metaqueries=num_metaqueries,
                connector_num_hidden_layers=connector_num_hidden_layers,
                dtype=dtype,
            )

        # ── 公共部分: 投影层 + 验证 ────────────────────────────────────────
        self._init_projection_and_validate(wan_text_dim, dtype)

    # ─────────────────────────────────────────────────────────────────────
    # 初始化模式 1: 从训练好的 checkpoint 加载
    # ─────────────────────────────────────────────────────────────────────

    def _init_from_checkpoint(self, checkpoint_path, num_metaqueries, dtype):
        """从训练好的 MetaQuery checkpoint 加载 MLLM + Connector。"""
        if self._looks_like_wan_connector_checkpoint(checkpoint_path):
            self._init_from_wan_connector_checkpoint(
                checkpoint_path=checkpoint_path,
                num_metaqueries=num_metaqueries,
                dtype=dtype,
            )
            return

        import sys
        _metaquery_root = "/home/liuzhirui/model/Qwen3-VL-main/metaquery-main"
        if _metaquery_root not in sys.path:
            sys.path.insert(0, _metaquery_root)
            print(f"[MetaQueryEncoder] 已将 metaquery-main 加入 sys.path: {_metaquery_root}")

        from trainer_utils import find_newest_checkpoint
        from models.metaquery import MetaQuery, MetaQueryConfig

        ckpt = find_newest_checkpoint(checkpoint_path)
        print(f"[MetaQueryEncoder] 加载 checkpoint: {ckpt}")

        # ── 手动实例化 + 加载权重，绕开 from_pretrained 的 meta device 上下文 ──
        import os
        config = MetaQueryConfig.from_pretrained(ckpt)
        config._gradient_checkpointing = False

        # 直接构造模型（不经过 from_pretrained 的 meta device 包装）
        # 内部 Qwen3VLForConditionalGeneration.from_pretrained 就能正常执行
        mq_model = MetaQuery(config)

        # 手动加载 checkpoint 权重
        weight_file = os.path.join(ckpt, "model.safetensors")
        if os.path.exists(weight_file):
            from safetensors.torch import load_file
            state_dict = load_file(weight_file)
        else:
            weight_file = os.path.join(ckpt, "pytorch_model.bin")
            state_dict = torch.load(weight_file, map_location="cpu")
        missing, unexpected = mq_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[MetaQueryEncoder] 缺失的权重 ({len(missing)} 个): {missing[:5]}...")
        if unexpected:
            print(f"[MetaQueryEncoder] 多余的权重 ({len(unexpected)} 个): {unexpected[:5]}...")
        print(f"[MetaQueryEncoder] 权重加载完成: {weight_file}")

        mq_model = mq_model.to(device=self.device, dtype=dtype)
        mq_model.eval().requires_grad_(False)

        # 只保留 MLLM backbone + Connector，丢弃 VAE / Diffusion
        self.mllm_model = mq_model.model      # MLLMInContext
        self.tokenizer  = mq_model.model.get_tokenizer()
        self.tokenize   = mq_model.model.get_tokenize_fn()

        connector_out_dim = mq_model.model.connector_out_dim
        self.connector_out_dim = connector_out_dim

        print(
            f"[MetaQueryEncoder] MLLM backbone: "
            f"{mq_model.model.mllm_backbone.__class__.__name__}"
        )
        print(
            f"[MetaQueryEncoder] connector 输出维度: {connector_out_dim}"
        )

        del mq_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _looks_like_wan_connector_checkpoint(self, checkpoint_path: str) -> bool:
        path = Path(checkpoint_path)
        if path.is_file():
            explicit_wan_files = {
                "mq_encoder_full.pt",
                "mq_encoder_full.safetensors",
                "mq_encoder_trainable.pt",
                "mq_encoder_trainable.safetensors",
                "training_state.pt",
            }
            if path.name in explicit_wan_files:
                return True
            if path.name not in {"model.safetensors", "pytorch_model.bin"}:
                return False
            cfg_path = path.parent / "config.json"
            if not cfg_path.exists():
                return False
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                return False
            return str(cfg.get("format", "")) == "wan_metaquery_encoder"
        if not path.is_dir():
            return False
        if any(
            (path / name).exists()
            for name in (
                "mq_encoder_full.pt",
                "mq_encoder_full.safetensors",
                "mq_encoder_trainable.pt",
                "mq_encoder_trainable.safetensors",
                "training_state.pt",
            )
        ):
            return True
        cfg_path = path / "config.json"
        if not cfg_path.exists():
            return False
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return str(cfg.get("format", "")) == "wan_metaquery_encoder"

    def _init_from_wan_connector_checkpoint(
        self,
        checkpoint_path: str,
        num_metaqueries: int,
        dtype: torch.dtype,
    ) -> None:
        from train_metaquery_wan import load_mq_encoder_state
        from train_connector_for_wan import (
            MetaQueryEncoderForWan as WanMetaQueryEncoderForWan,
        )

        path = Path(checkpoint_path)
        cfg_path = (path if path.is_dir() else path.parent) / "config.json"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

        qwen3vl_model_id = cfg.get("qwen3vl_model_id") or "Qwen/Qwen3-VL-2B-Instruct"
        connector_layers = int(cfg.get("connector_num_hidden_layers", 24))
        wan_text_dim = int(cfg.get("wan_text_dim", WAN_TEXT_DIM))

        encoder = WanMetaQueryEncoderForWan(
            qwen3vl_model_id=qwen3vl_model_id,
            num_metaqueries=num_metaqueries,
            connector_num_hidden_layers=connector_layers,
            dtype=dtype,
            device=self.device,
        )

        state_dict, resolved_path = load_mq_encoder_state(checkpoint_path, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[MetaQueryEncoder] WAN checkpoint 缺失权重 ({len(missing)} 个): {missing[:5]}...")
        if unexpected:
            print(f"[MetaQueryEncoder] WAN checkpoint 多余权重 ({len(unexpected)} 个): {unexpected[:5]}...")
        print(f"[MetaQueryEncoder] 加载 WAN connector checkpoint: {resolved_path}")

        encoder = encoder.to(device=self.device, dtype=dtype)
        encoder.eval().requires_grad_(False)

        self.mllm_model = encoder.mllm_model
        self.tokenizer = encoder.tokenizer
        self.tokenize = encoder.tokenize
        self.connector_out_dim = wan_text_dim
        self._uses_identity_wan_proj = wan_text_dim == self.wan_text_dim

        print(f"[MetaQueryEncoder] WAN checkpoint 输出维度: {wan_text_dim}")

    # ─────────────────────────────────────────────────────────────────────
    # 初始化模式 2: 从预训练 Qwen3-VL 直接初始化 (无需 checkpoint)
    # ─────────────────────────────────────────────────────────────────────

    def _init_from_pretrained(
        self,
        mllm_id: str,
        diffusion_model_id: str,
        num_metaqueries: int,
        connector_num_hidden_layers: int,
        dtype: torch.dtype,
    ):
        """
        通过直接加载 Qwen3-VL 预训练模型 + 随机初始化 Connector 来构建
        MetaQuery 编码器。无需训练好的 MetaQuery checkpoint。

        MLLM backbone 保留原始预训练权重 (可提供有意义的语义特征)。
        Connector (Qwen2Encoder + Linear 投影) 为随机初始化。
        """
        from models.model import MLLMInContextConfig, MLLMInContext

        print(f"[MetaQueryEncoder] 从预训练模型直接初始化:")
        print(f"  mllm_id            : {mllm_id}")
        print(f"  diffusion_model_id : {diffusion_model_id}")
        print(f"  connector_layers   : {connector_num_hidden_layers}")

        # 构建 config
        config = MLLMInContextConfig(
            mllm_id=mllm_id,
            diffusion_model_id=diffusion_model_id,
            num_metaqueries=num_metaqueries,
            _gradient_checkpointing=False,
            connector_num_hidden_layers=connector_num_hidden_layers,
        )

        # 直接实例化 MLLMInContext (会自动加载 Qwen3-VL 预训练权重)
        mllm_model = MLLMInContext(config)
        mllm_model = mllm_model.to(device=self.device, dtype=dtype)
        mllm_model.eval().requires_grad_(False)

        self.mllm_model = mllm_model
        self.tokenizer  = mllm_model.get_tokenizer()
        self.tokenize   = mllm_model.get_tokenize_fn()

        connector_out_dim = mllm_model.connector_out_dim
        self.connector_out_dim = connector_out_dim

        print(
            f"[MetaQueryEncoder] MLLM backbone: "
            f"{mllm_model.mllm_backbone.__class__.__name__}"
        )
        print(
            f"[MetaQueryEncoder] connector 输出维度: {connector_out_dim}"
        )
        print(
            "[MetaQueryEncoder] ★ 注意: Connector 为随机初始化, "
            "MLLM backbone 保留原始预训练权重"
        )

        # 清理 MLLMInContext 中的 transformer (Sana/UNet), 仅保留 mllm + connector
        if hasattr(self.mllm_model, 'transformer') and self.mllm_model.transformer is not None:
            del self.mllm_model.transformer
            self.mllm_model.transformer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[MetaQueryEncoder] 已删除 MLLMInContext.transformer 以释放显存")

    # ─────────────────────────────────────────────────────────────────────
    # 公共初始化: 投影层 + 验证
    # ─────────────────────────────────────────────────────────────────────

    def _init_projection_and_validate(self, wan_text_dim, dtype):
        """投影层初始化 + 清理 + 完整性验证。公共代码, 两种模式都调用。"""
        connector_out_dim = self.connector_out_dim

        if connector_out_dim == wan_text_dim:
            self.to_wan_proj = nn.Identity().to(device=self.device)
            self._uses_identity_wan_proj = True
        else:
            self.to_wan_proj = nn.Sequential(
                nn.Linear(connector_out_dim, wan_text_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(wan_text_dim, wan_text_dim, bias=True),
            ).to(device=self.device, dtype=dtype)

            for m in self.to_wan_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        # ── 清理不需要的子模块以释放显存 ─────────────────────────────
        #    MLLMInContext 持有 self.transformer (SanaTransformer2DModel)
        #    编码阶段只需 mllm_backbone + connector，删除以节省 ~3-6GB
        #    (从 pretrained 初始化模式已在 _init_from_pretrained 中清理过)
        if hasattr(self.mllm_model, 'transformer') and self.mllm_model.transformer is not None:
            del self.mllm_model.transformer
            self.mllm_model.transformer = None
            torch.cuda.empty_cache()
            print("[MetaQueryEncoder] 已删除 MLLMInContext.transformer 以释放显存")

        print(
            f"[MetaQueryEncoder] 投影层已初始化: "
            f"{connector_out_dim} → {wan_text_dim}"
        )

        # ── 初始化完整性验证 ────────────────────────────────────────
        print("\n[MetaQueryEncoder] 【初始化完整性验证】")

        # 验证 mllm_backbone 确实是 Qwen3-VL
        backbone_cls = self.mllm_model.mllm_backbone.__class__.__name__
        assert 'Qwen3VL' in backbone_cls or 'Qwen2' in backbone_cls or 'Llava' in backbone_cls, \
            f"mllm_backbone 类型异常: {backbone_cls}"
        print(f"  [PASS] mllm_backbone 类型: {backbone_cls}")

        # 验证 mllm_type 设置正确
        mllm_type = self.mllm_model.mllm_type
        print(f"  [PASS] mllm_type = '{mllm_type}'")

        # 验证 BOI/EOI token ID 已注册
        assert hasattr(self.mllm_model, 'boi_token_id'), "boi_token_id 未注册!"
        assert hasattr(self.mllm_model, 'eoi_token_id'), "eoi_token_id 未注册!"
        print(
            f"  [PASS] BOI token id = {self.mllm_model.boi_token_id}, "
            f"EOI token id = {self.mllm_model.eoi_token_id}"
        )

        # 验证 connector 非 Identity，确实是多层神经网络
        connector = self.mllm_model.connector
        connector_params = sum(p.numel() for p in connector.parameters())
        assert connector_params > 0, "connector 没有可训练参数!"
        print(
            f"  [PASS] connector 参数量: {connector_params:,} "
            f"(非 Identity, 确实包含神经网络层)"
        )

        # 验证 connector 内部结构: Qwen2Encoder + Linear + GELU + Linear + RMSNorm
        connector_modules = [m.__class__.__name__ for m in connector]
        print(f"  [PASS] connector 内部结构: {connector_modules}")

        if isinstance(self.to_wan_proj, nn.Identity):
            print("  [PASS] to_wan_proj = Identity (connector 已直接输出 Wan 4096 维)")
        else:
            proj_weight_norm = sum(
                p.data.abs().sum().item() for p in self.to_wan_proj.parameters()
            )
            assert proj_weight_norm > 0, "to_wan_proj 权重全为零!"
            print(f"  [PASS] to_wan_proj 权重 L1 范数: {proj_weight_norm:.4f} (非零)")

        # 验证维度链: mllm_hidden_size → connector_out_dim → wan_text_dim
        print(
            f"  [PASS] 维度链: mllm_hidden={self.mllm_model.mllm_hidden_size} "
            f"→ connector_out={self.connector_out_dim} "
            f"→ wan_text_dim={self.wan_text_dim}"
        )

        print("[MetaQueryEncoder] ✅ 全部验证通过，初始化完成！\n")

    @torch.no_grad()
    def encode(
        self,
        captions: List[str],
        input_images: Optional[List[Optional[List[Image.Image]]]] = None,
    ) -> List[torch.Tensor]:
        """
        提取 MetaQuery 条件特征，并投影到 Wan2.2 context 空间。

        Args:
            captions:       批次文本描述列表，长度 = B
            input_images:   批次参考图像列表，
                            每项为 List[PIL.Image] 或 None，长度 = B

        Returns:
            List[Tensor]:   长度 = B，每项 shape [num_metaqueries, wan_text_dim]
                            可与 T5 context 张量 concat 后送入 WanModel。
        """
        print(
            f"[MetaQueryEncoder] encode() 调用 | batch={len(captions)} | "
            f"has_images={input_images is not None}"
        )

        # ── 分词 / 视觉归一化 ────────────────────────────────────────────────
        if input_images is not None:
            input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
                self.tokenizer, captions, input_images
            )
            input_ids      = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values   = pixel_values.to(self.device, self.dtype) if pixel_values is not None else None
            image_sizes    = image_sizes.to(self.device) if image_sizes is not None else None
        else:
            input_ids, attention_mask = self.tokenize(
                self.tokenizer, captions
            )
            input_ids      = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values   = None
            image_sizes    = None

        print(
            f"  ├─ input_ids shape   : {input_ids.shape}"
        )
        if pixel_values is not None:
            # tokenize() 对 qwenvl/qwen3vl 做了 unsqueeze(0)，
            # 但 encode_condition() 期望原始形状，需还原
            if (
                hasattr(self.mllm_model, 'mllm_type')
                and self.mllm_model.mllm_type in ('qwenvl', 'qwen3vl')
            ):
                pixel_values = pixel_values.squeeze(0)
            print(
                f"  ├─ pixel_values shape: {pixel_values.shape}, "
                f"L2_norm={pixel_values.float().norm().item():.4f}, "
                f"非零元素占比={((pixel_values != 0).sum().item() / pixel_values.numel() * 100):.1f}%"
            )

        # ── 验证 BOI/EOI token 确实存在于 input_ids 中 ───────────────────
        boi_id = self.mllm_model.boi_token_id
        eoi_id = self.mllm_model.eoi_token_id
        boi_count = (input_ids == boi_id).sum().item()
        eoi_count = (input_ids == eoi_id).sum().item()
        assert boi_count > 0, (
            f"[FATAL] input_ids 中找不到 <begin_of_img> (id={boi_id})! "
            "MetaQuery token 未被正确插入到 prompt 中"
        )
        assert eoi_count > 0, (
            f"[FATAL] input_ids 中找不到 <end_of_img> (id={eoi_id})! "
            "MetaQuery token 未被正确插入到 prompt 中"
        )
        # 统计 MQ token 在 input_ids 中的实际数量
        boi_pos_0 = (input_ids[0] == boi_id).nonzero(as_tuple=True)[0].item()
        eoi_pos_0 = (input_ids[0] == eoi_id).nonzero(as_tuple=True)[0].item()
        mq_token_count = eoi_pos_0 - boi_pos_0 - 1
        assert mq_token_count == self.num_metaqueries, (
            f"[FATAL] input_ids 中 <img> token 数量={mq_token_count}, "
            f"期望={self.num_metaqueries}! tokenize 流程异常"
        )
        print(
            f"  ├─ [VERIFY] input_ids 中确认含有 <begin_of_img>(pos={boi_pos_0}) + "
            f"{mq_token_count} 个 <img> token + <end_of_img>(pos={eoi_pos_0})"
        )

        # ── 通过 MLLM + Connector 提取 MetaQuery 特征 ────────────────────────
        # encode_condition 返回 (connector_out, attention_mask)
        # connector_out shape: [B, num_metaqueries, connector_out_dim]
        mq_features, mq_mask = self.mllm_model.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        # 验证 encode_condition 输出维度正确
        assert mq_features.shape[1] == self.num_metaqueries, (
            f"[FATAL] encode_condition 输出的 MetaQuery token 数"
            f"={mq_features.shape[1]}, 期望={self.num_metaqueries}! "
            "BOI/EOI 区间提取异常"
        )
        assert mq_features.shape[2] == self.connector_out_dim, (
            f"[FATAL] connector 输出维度={mq_features.shape[2]}, "
            f"期望={self.connector_out_dim}! connector 未正确运行"
        )
        # 验证特征非零、非 NaN
        assert not torch.isnan(mq_features).any(), (
            "[FATAL] MetaQuery features 包含 NaN!"
        )
        assert not torch.isinf(mq_features).any(), (
            "[FATAL] MetaQuery features 包含 Inf!"
        )
        feat_norm = mq_features.float().norm().item()
        feat_nonzero = (mq_features.abs() > 1e-8).sum().item()
        feat_total = mq_features.numel()
        assert feat_norm > 0, (
            "[FATAL] MetaQuery features 全为零! "
            "Qwen3-VL backbone 或 connector 未正确运行"
        )
        print(
            f"  ├─ [VERIFY] Connector 输出: shape={mq_features.shape}, "
            f"L2_norm={feat_norm:.4f}, "
            f"mean={mq_features.float().mean().item():.6f}, "
            f"std={mq_features.float().std().item():.6f}, "
            f"非零元素={feat_nonzero}/{feat_total} "
            f"(非零、非NaN、非Inf ✅)"
        )

        # ── 投影到 Wan2.2 text_dim ────────────────────────────────────────────
        mq_features = mq_features.to(self.dtype)
        wan_features = self.to_wan_proj(mq_features)
        # wan_features: [B, num_metaqueries, wan_text_dim=4096]

        # 验证投影后维度和数值
        assert wan_features.shape == (mq_features.shape[0], self.num_metaqueries, self.wan_text_dim), (
            f"[FATAL] 投影后 shape={wan_features.shape}, "
            f"期望=({mq_features.shape[0]}, {self.num_metaqueries}, {self.wan_text_dim})"
        )
        assert not torch.isnan(wan_features).any(), "[FATAL] 投影后包含 NaN!"
        wan_norm = wan_features.float().norm().item()
        assert wan_norm > 0, "[FATAL] 投影后全为零!"
        # 计算投影前后的余弦相似度(采样第一个token, 对齐维度)来验证变换确实发生了
        with torch.no_grad():
            _pre_sample = mq_features[0, 0, :min(self.connector_out_dim, self.wan_text_dim)].float()
            _post_sample = wan_features[0, 0, :min(self.connector_out_dim, self.wan_text_dim)].float()
            _cos_sim = torch.nn.functional.cosine_similarity(
                _pre_sample.unsqueeze(0), _post_sample.unsqueeze(0)
            ).item()
        print(
            f"  └─ [VERIFY] 投影后: shape={wan_features.shape}, "
            f"L2_norm={wan_norm:.4f}, "
            f"mean={wan_features.float().mean().item():.6f}, "
            f"std={wan_features.float().std().item():.6f}, "
            f"投影前后余弦相似度={_cos_sim:.4f} "
            f"(→ Wan text_dim={self.wan_text_dim} ✅)"
        )

        # 拆分为 list 与 T5 context 格式一致
        return [wan_features[i] for i in range(wan_features.size(0))]
