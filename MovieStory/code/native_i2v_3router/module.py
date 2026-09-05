"""MovieStory's composed Qwen + MetaQuery + native Wan2.2 I2V-A14B module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Type

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .contracts import build_native_i2v_condition, native_flow_matching_pair


@dataclass
class NativeI2VTrainingOutput:
    loss: torch.Tensor
    predictions: List[torch.Tensor]
    targets: List[torch.Tensor]
    normalized_timestep: torch.Tensor
    wan_timestep: torch.Tensor
    branch: str
    context_tokens: int
    condition_shapes: List[tuple[int, ...]]


class MetaQueryQwenWanI2VA14B(nn.Module):
    """External composition module; the two native Wan DiTs stay unmodified.

    Qwen/MetaQuery produces Wan cross-attention context.  The reference image is
    injected only through Wan's existing ``y`` argument.  The target/noise pair
    and MSE follow the native rectified-flow training contract.
    """

    DEFAULT_COND_KEYWORDS = (
        "cross_attn",
        "cross-attn",
        "crossattention",
        "cross_attention",
        "text_embedding",
        "time_projection",
        "modulation",
        "cross_attn_norm",
        "norm3",
    )

    def __init__(
        self,
        *,
        wan_checkpoint_dir: str,
        qwen3vl_model_id: str,
        metaquery_encoder_class: Type[nn.Module],
        num_metaqueries: int = 256,
        connector_num_hidden_layers: int = 24,
        mq_gradient_checkpointing: bool = True,
        train_mq_input_embeddings: bool = False,
        connector_norm_init_scale: float = 1.0,
        device_id: int = 0,
        rank: int = 0,
        dit_fsdp: bool = False,
        t5_cpu: bool = True,
        context_text_len: int = 768,
        wan_train_mode: str = "frozen",
        wan_cond_name_pattern: str = "",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from wan import WanI2V
        from wan.configs import WAN_CONFIGS

        self.device_id = int(device_id)
        self.dit_device = torch.device(f"cuda:{self.device_id}")
        self.encoder_device = self.dit_device
        self.dtype = dtype
        self.context_text_len = int(context_text_len)
        self.wan_train_mode = str(wan_train_mode).strip().lower()
        if self.wan_train_mode not in ("frozen", "cond_only", "full"):
            raise ValueError(
                f"wan_train_mode must be frozen/cond_only/full, got {wan_train_mode}"
            )

        config = WAN_CONFIGS["i2v-A14B"]
        # WanI2V is a pipeline container rather than nn.Module.  Its native
        # high/low WanModel objects are deliberately not re-parented or wrapped.
        self.wan = WanI2V(
            config=config,
            checkpoint_dir=wan_checkpoint_dir,
            device_id=self.device_id,
            rank=int(rank),
            t5_fsdp=False,
            dit_fsdp=bool(dit_fsdp),
            use_sp=False,
            t5_cpu=bool(t5_cpu),
            init_on_cpu=not bool(dit_fsdp),
            convert_model_dtype=not bool(dit_fsdp),
        )
        if not dit_fsdp:
            self.wan.low_noise_model.to(self.dit_device)
            self.wan.high_noise_model.to(self.dit_device)
        # Register both native DiTs as children of the composed nn.Module while
        # keeping the WanI2V pipeline's references to the exact same objects.
        self.low_noise_model = self.wan.low_noise_model
        self.high_noise_model = self.wan.high_noise_model

        self.config = config
        self.boundary_timestep = float(config.boundary * config.num_train_timesteps)
        self.mq_encoder = metaquery_encoder_class(
            qwen3vl_model_id=qwen3vl_model_id,
            num_metaqueries=int(num_metaqueries),
            connector_num_hidden_layers=int(connector_num_hidden_layers),
            gradient_checkpointing=bool(mq_gradient_checkpointing),
            train_input_embeddings=bool(train_mq_input_embeddings),
            connector_norm_init_scale=float(connector_norm_init_scale),
            dtype=dtype,
            device=str(self.encoder_device),
        )
        if hasattr(self.mq_encoder, "bind_t5_provider"):
            self.mq_encoder.bind_t5_provider(self.encode_frozen_t5)

        self._configure_wan_text_len()
        self._audit_native_i2v_models()
        self._configure_wan_trainability(wan_cond_name_pattern)
        self.last_audit: Dict[str, object] = {}

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        current = model
        seen = set()
        while id(current) not in seen:
            seen.add(id(current))
            child = getattr(current, "module", None)
            if child is None:
                child = getattr(current, "_fsdp_wrapped_module", None)
            if child is None or child is current:
                break
            current = child
        return current

    def _configure_wan_text_len(self) -> None:
        if self.context_text_len <= 0:
            raise ValueError("context_text_len must be positive")
        for model in (self.low_noise_model, self.high_noise_model):
            raw = self._unwrap_model(model)
            raw.text_len = self.context_text_len
            try:
                model.text_len = self.context_text_len
            except Exception:
                pass

    def _audit_native_i2v_models(self) -> None:
        for branch, wrapped in (
            ("low_noise", self.low_noise_model),
            ("high_noise", self.high_noise_model),
        ):
            model = self._unwrap_model(wrapped)
            expected = {
                "model_type": "i2v",
                "in_dim": 36,
                "out_dim": 16,
                "text_dim": 4096,
            }
            actual = {key: getattr(model, key, None) for key in expected}
            bad = {
                key: (actual[key], value)
                for key, value in expected.items()
                if actual[key] != value
            }
            if bad:
                raise RuntimeError(f"{branch} is not native Wan I2V-A14B: {bad}")
            if int(getattr(model, "text_len")) != self.context_text_len:
                raise RuntimeError(f"{branch} text_len was not configured externally")
        if tuple(self.config.vae_stride) != (4, 8, 8):
            raise RuntimeError(f"I2V-A14B requires VAE stride (4,8,8), got {self.config.vae_stride}")
        if tuple(self.config.patch_size) != (1, 2, 2):
            raise RuntimeError(
                f"I2V-A14B requires patch size (1,2,2), got {self.config.patch_size}"
            )

    def _configure_wan_trainability(self, pattern: str) -> None:
        keywords = tuple(
            value.strip().lower() for value in pattern.split(",") if value.strip()
        ) or self.DEFAULT_COND_KEYWORDS
        self._wan_trainable_names: List[str] = []
        self._wan_trainable_parameters: List[nn.Parameter] = []
        seen = set()
        for branch, model in (
            ("low_noise", self.low_noise_model),
            ("high_noise", self.high_noise_model),
        ):
            for name, parameter in model.named_parameters():
                selected = self.wan_train_mode == "full" or (
                    self.wan_train_mode == "cond_only"
                    and any(keyword in name.lower() for keyword in keywords)
                )
                if parameter.requires_grad != selected:
                    parameter.requires_grad_(selected)
                if selected and id(parameter) not in seen:
                    seen.add(id(parameter))
                    self._wan_trainable_names.append(f"{branch}.{name}")
                    self._wan_trainable_parameters.append(parameter)
            model.train(self.wan_train_mode != "frozen")
        self._wan_cond_keywords = keywords

    def wan_trainable_parameters(self) -> List[nn.Parameter]:
        return list(self._wan_trainable_parameters)

    def mq_trainable_parameters(self) -> List[nn.Parameter]:
        module = self.mq_encoder.module if hasattr(self.mq_encoder, "module") else self.mq_encoder
        if hasattr(module, "get_trainable_params"):
            return list(module.get_trainable_params())
        return [parameter for parameter in module.parameters() if parameter.requires_grad]

    def trainable_parameters(self) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        seen = set()
        for parameter in (*self.mq_trainable_parameters(), *self.wan_trainable_parameters()):
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        return parameters

    def train(self, mode: bool = True):
        super().train(mode)
        wan_training = bool(mode and self.wan_train_mode != "frozen")
        self.low_noise_model.train(wan_training)
        self.high_noise_model.train(wan_training)
        # VAE and T5 are native frozen encoders and must remain in eval mode.
        self.wan.vae.model.eval()
        self.wan.text_encoder.model.eval()
        return self

    @torch.no_grad()
    def encode_frozen_t5(self, captions: Sequence[str]) -> list[torch.Tensor]:
        rows = self.wan.text_encoder(list(captions), torch.device("cpu"))
        return [row.detach() for row in rows]

    def _shared_normalized_timestep(self) -> torch.Tensor:
        value = torch.rand(1, device=self.dit_device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(value, src=0)
        return value

    def _context(self, captions: Sequence[str], mq_references: Sequence[Optional[Image.Image]]):
        qwen_images = [
            [reference.convert("RGB")]
            if isinstance(reference, Image.Image)
            else [Image.new("RGB", (224, 224))]
            for reference in mq_references
        ]
        features = self.mq_encoder(list(captions), qwen_images)
        if features.ndim != 3 or features.shape[0] != len(captions) or features.shape[2] != 4096:
            raise RuntimeError(f"invalid MQ context shape: {tuple(features.shape)}")
        if features.shape[1] > self.context_text_len:
            raise RuntimeError(
                f"context has {features.shape[1]} tokens but Wan text_len={self.context_text_len}"
            )
        return [row.to(device=self.dit_device, dtype=self.dtype) for row in features]

    @torch.no_grad()
    def _target_latents(self, videos: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        latents = []
        for index, video in enumerate(videos):
            if not torch.is_tensor(video) or video.ndim != 4 or video.shape[0] != 3:
                raise ValueError(
                    f"video {index} must be [3,F,H,W], got "
                    f"{tuple(video.shape) if torch.is_tensor(video) else type(video).__name__}"
                )
            latent = self.wan.vae.encode(
                [video.to(device=self.dit_device)]
            )[0]
            latents.append(latent)
        return latents

    def compute_native_i2v_loss(
        self,
        *,
        captions: Sequence[str],
        videos: Sequence[torch.Tensor],
        first_frames: Sequence[Image.Image],
        mq_references: Sequence[Optional[Image.Image]],
        normalized_timestep: Optional[torch.Tensor] = None,
        noises: Optional[Sequence[torch.Tensor]] = None,
    ) -> NativeI2VTrainingOutput:
        batch = len(captions)
        if not (batch == len(videos) == len(first_frames) == len(mq_references)):
            raise ValueError("caption/video/I2V-reference/MQ-reference batch sizes differ")
        if batch <= 0:
            raise ValueError("empty batch")
        frame_counts = {int(video.shape[1]) for video in videos}
        if len(frame_counts) != 1:
            raise ValueError(f"all videos in one batch must share frame_num, got {frame_counts}")
        frame_num = next(iter(frame_counts))

        context = self._context(captions, mq_references)
        target_latents = self._target_latents(videos)
        conditions = [
            build_native_i2v_condition(
                vae=self.wan.vae,
                first_frame=reference,
                frame_num=frame_num,
                latent_shape=latent.shape,
                vae_stride=self.config.vae_stride,
                device=self.dit_device,
                dtype=torch.float32,
            )
            for reference, latent in zip(first_frames, target_latents)
        ]

        t = self._shared_normalized_timestep() if normalized_timestep is None else normalized_timestep
        t = t.to(device=self.dit_device, dtype=torch.float32).reshape(1)
        if dist.is_available() and dist.is_initialized() and normalized_timestep is not None:
            dist.broadcast(t, src=0)
        if not bool(((t >= 0.0) & (t < 1.0)).all()):
            raise ValueError(f"normalized timestep must be in [0,1), got {t.item()}")
        wan_t = t * float(self.config.num_train_timesteps)
        if float(wan_t.item()) >= self.boundary_timestep:
            model = self.high_noise_model
            branch = "high_noise"
        else:
            model = self.low_noise_model
            branch = "low_noise"

        if noises is None:
            noise_rows = [torch.randn_like(latent, dtype=torch.float32) for latent in target_latents]
        else:
            if len(noises) != batch:
                raise ValueError("noise batch size mismatch")
            noise_rows = [noise.to(self.dit_device, torch.float32) for noise in noises]
        noisy_inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for latent, noise in zip(target_latents, noise_rows):
            noisy, target = native_flow_matching_pair(latent, noise, t)
            noisy_inputs.append(noisy)
            targets.append(target)

        patch = self.config.patch_size
        sequence_lengths = []
        for latent in target_latents:
            token_numerator = int(latent.shape[1] * latent.shape[2] * latent.shape[3])
            token_denominator = int(patch[1] * patch[2])
            if token_numerator % token_denominator:
                raise RuntimeError("native I2V latent is not divisible by its spatial patch area")
            sequence_lengths.append(token_numerator // token_denominator)
        model_t = wan_t.expand(batch)
        with torch.amp.autocast("cuda", dtype=self.dtype):
            predictions = model(
                noisy_inputs,
                t=model_t,
                context=context,
                seq_len=max(sequence_lengths),
                y=[condition.y for condition in conditions],
            )
        loss = torch.stack(
            [F.mse_loss(prediction.float(), target) for prediction, target in zip(predictions, targets)]
        ).mean()

        self.last_audit = {
            "branch": branch,
            "normalized_timestep": float(t.item()),
            "wan_timestep": float(wan_t.item()),
            "boundary_timestep": self.boundary_timestep,
            "context_tokens": int(context[0].shape[0]),
            "y_channels": [int(condition.y.shape[0]) for condition in conditions],
            "condition_shapes": [tuple(condition.y.shape) for condition in conditions],
            "loss_contract": "mean_mse(predicted_velocity, noise-clean_latent)",
            "image_injection": "native_y_concat(mask4,vae(first_frame+zeros))",
        }
        return NativeI2VTrainingOutput(
            loss=loss,
            predictions=predictions,
            targets=targets,
            normalized_timestep=t,
            wan_timestep=wan_t,
            branch=branch,
            context_tokens=int(context[0].shape[0]),
            condition_shapes=[tuple(condition.y.shape) for condition in conditions],
        )

    def forward(self, batch: Dict[str, Sequence[object]]) -> NativeI2VTrainingOutput:
        return self.compute_native_i2v_loss(
            captions=batch["caption"],  # type: ignore[arg-type]
            videos=batch["video"],  # type: ignore[arg-type]
            first_frames=batch["ref_image"],  # type: ignore[arg-type]
            mq_references=batch["mq_ref_image"],  # type: ignore[arg-type]
        )

    def architecture_metadata(self) -> Dict[str, object]:
        encoder = self.mq_encoder.module if hasattr(self.mq_encoder, "module") else self.mq_encoder
        return {
            "module": self.__class__.__name__,
            "wan_task": "i2v-A14B",
            "wan_internal_structure_changed": False,
            "wan_branches": ["low_noise_model", "high_noise_model"],
            "boundary_timestep": self.boundary_timestep,
            "vae_stride": tuple(self.config.vae_stride),
            "patch_size": tuple(self.config.patch_size),
            "context_text_len": self.context_text_len,
            "wan_train_mode": self.wan_train_mode,
            "wan_trainable_tensors": len(self._wan_trainable_parameters),
            "native_image_injection": "y=concat(mask[4],vae(first_frame+zeros)[16])",
            "native_loss": "MSE(model(x_t,t,context,y), noise-x0)",
            "router": getattr(encoder, "get_three_router_metadata", lambda: {})(),
            "conditioning": getattr(encoder, "get_conditioning_metadata", lambda: {})(),
        }
