"""
Determined/torchrun friendly FSDP inference entry for MetaQuery + Wan Animate.

用法示例:
  torchrun --nproc_per_node=4 inference_metaquery_animate_fsdp.py \
    --distributed --dit_fsdp \
    --checkpoint_path /path/to/checkpoint-final \
    --wan_checkpoint_dir /path/to/Wan2.2-Animate-14B \
    --qwen3vl_model_id /path/to/Qwen3-VL-2B-Thinking \
    --prompt "a person dancing" \
    --ref_image /path/to/ref.png \
    --output_path /tmp/out.mp4
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from PIL import Image

import inference_metaquery_animate as base


@dataclass
class DistEnv:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int


class _BroadcastMQEncoder:
    """Only rank0 loads Qwen/MQ encoder; other ranks receive broadcasted MQ features."""

    def __init__(
        self,
        real_encoder,
        env: DistEnv,
        device: torch.device,
        num_metaqueries: int,
        out_dim: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.real_encoder = real_encoder
        self.env = env
        self.device = device
        self.num_metaqueries = int(num_metaqueries)
        self.out_dim = int(out_dim)
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, caption, ref_image=None):
        if not self.env.enabled or self.env.world_size <= 1:
            if self.real_encoder is None:
                raise RuntimeError("real_encoder is None in non-distributed mode")
            return self.real_encoder.encode(caption, ref_image)

        if self.env.rank == 0:
            if self.real_encoder is None:
                raise RuntimeError("rank0 real_encoder is None")
            feat = self.real_encoder.encode(caption, ref_image)
            feat = feat[0].to(self.device, dtype=self.dtype)
        else:
            feat = torch.empty(
                (self.num_metaqueries, self.out_dim),
                device=self.device,
                dtype=self.dtype,
            )
        dist.broadcast(feat, src=0)
        return feat.unsqueeze(0)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--distributed", action="store_true")
    pre.add_argument("--dit_fsdp", action="store_true")
    pre.add_argument("--t5_fsdp", action="store_true")
    pre.add_argument("--use_sp", action="store_true")
    pre.add_argument("--t5_cpu", action="store_true")
    pre.add_argument("--no_init_on_cpu", action="store_true")
    pre.add_argument(
        "--dist_timeout_sec",
        type=int,
        default=int(os.environ.get("DIST_TIMEOUT_SEC", "1800")),
    )
    pre.add_argument(
        "--dist_warmup",
        type=str,
        choices=["none", "barrier", "all_reduce"],
        default=os.environ.get("WAN_DIST_WARMUP", "none"),
    )
    pre.add_argument(
        "--load_stagger_sec",
        type=float,
        default=float(os.environ.get("WAN_LOAD_STAGGER_SEC", "0")),
    )
    pre_args, remain = pre.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remain
        args = base.parse_args()
    finally:
        sys.argv = original_argv

    args.distributed = bool(pre_args.distributed)
    args.dit_fsdp = bool(pre_args.dit_fsdp)
    args.t5_fsdp = bool(pre_args.t5_fsdp)
    args.use_sp = bool(pre_args.use_sp)
    args.t5_cpu = bool(pre_args.t5_cpu)
    args.init_on_cpu = not bool(pre_args.no_init_on_cpu)
    args.dist_timeout_sec = int(pre_args.dist_timeout_sec)
    args.dist_warmup = str(pre_args.dist_warmup)
    args.load_stagger_sec = float(pre_args.load_stagger_sec)
    return args


def init_dist(args) -> DistEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = bool(args.distributed or world_size > 1)

    if enabled and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("distributed 推理需要 CUDA")
        cuda_count = torch.cuda.device_count()
        if local_rank < 0 or local_rank >= cuda_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} 超出可见 GPU 数量范围 [0, {cuda_count - 1}]"
            )
        torch.cuda.set_device(local_rank)
        timeout = timedelta(seconds=max(60, int(args.dist_timeout_sec)))
        backend = "nccl"
        print(
            f"[DIST][init] rank={rank} local_rank={local_rank} world_size={world_size} "
            f"backend={backend} timeout_sec={int(timeout.total_seconds())} "
            f"master={os.environ.get('MASTER_ADDR', '<unset>')}:{os.environ.get('MASTER_PORT', '<unset>')} "
            f"cuda_count={cuda_count}",
            flush=True,
        )
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timeout,
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        except TypeError:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timeout,
            )
        print(f"[DIST][init-ok] rank={rank}", flush=True)
        if args.dist_warmup == "barrier":
            try:
                dist.barrier(device_ids=[local_rank])
            except TypeError:
                dist.barrier()
            print(f"[DIST][warmup=barrier-ok] rank={rank}", flush=True)
        elif args.dist_warmup == "all_reduce":
            warm = torch.zeros(1, device=torch.device(f"cuda:{local_rank}"))
            dist.all_reduce(warm)
            print(f"[DIST][warmup=all_reduce-ok] rank={rank}", flush=True)
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    if enabled:
        args.device = local_rank
    return DistEnv(enabled=enabled, world_size=world_size, rank=rank, local_rank=local_rank)


def destroy_dist(env: DistEnv):
    if env.enabled and dist.is_initialized():
        try:
            dist.barrier(device_ids=[env.local_rank] if torch.cuda.is_available() else None)
        except Exception:
            pass
        dist.destroy_process_group()


class MetaQueryAnimatePipelineFSDP(base.MetaQueryAnimatePipeline):
    def __init__(self, args, env: DistEnv):
        self._dist_env = env
        self.args = args
        self.device = torch.device(f"cuda:{args.device}")
        print(f"[INIT] rank={self._dist_env.rank} wan_load_start", flush=True)
        self._load_pipeline()
        print(f"[INIT] rank={self._dist_env.rank} wan_load_done", flush=True)
        if (not self._dist_env.enabled) or self._dist_env.rank == 0:
            print(f"[INIT] rank={self._dist_env.rank} mq_load_start", flush=True)
            self._load_mq_encoder()
            self._mq_encoder_impl = self.mq_encoder
            print(f"[INIT] rank={self._dist_env.rank} mq_load_done", flush=True)
        else:
            self._mq_encoder_impl = None
            print(
                f"[INIT] rank={self._dist_env.rank} mq_load_skip(non-zero rank, use broadcast)",
                flush=True,
            )
        self.mq_encoder = _BroadcastMQEncoder(
            real_encoder=self._mq_encoder_impl,
            env=self._dist_env,
            device=self.device,
            num_metaqueries=self.args.num_metaqueries,
            out_dim=4096,
            dtype=torch.bfloat16,
        )

    def _load_pipeline(self):
        from wan import WanAnimate
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS["animate-14B"]
        self.wan = WanAnimate(
            config=config,
            checkpoint_dir=self.args.wan_checkpoint_dir,
            device_id=self.args.device,
            rank=self._dist_env.rank,
            t5_fsdp=self.args.t5_fsdp,
            dit_fsdp=self.args.dit_fsdp,
            use_sp=self.args.use_sp,
            t5_cpu=self.args.t5_cpu,
            init_on_cpu=self.args.init_on_cpu,
        )
        self.wan_config = config
        self._orig_text_len = self.wan.noise_model.text_len

        # 本推理路径使用零 clip 特征，不需要 CLIP 视觉塔常驻 GPU。
        try:
            if hasattr(self.wan, "clip") and hasattr(self.wan.clip, "model"):
                self.wan.clip.model.cpu()
                torch.cuda.empty_cache()
        except Exception:
            pass

        # FSDP 模式不做 offload，以避免分片参数反复迁移
        if self.args.dit_fsdp or self.args.t5_fsdp or self.args.use_sp:
            self.args.offload_model = False

        if self._dist_env.rank == 0:
            print(
                f"[FSDP] Animate loaded | rank={self._dist_env.rank} "
                f"world_size={self._dist_env.world_size} "
                f"device={self.args.device} dit_fsdp={self.args.dit_fsdp} "
                f"t5_fsdp={self.args.t5_fsdp} use_sp={self.args.use_sp}"
            )


def main():
    args = parse_args()
    env = init_dist(args)
    try:
        print(
            f"[STAGE] rank={env.rank} local_rank={env.local_rank} dist_ready=1 "
            f"world_size={env.world_size} device={args.device}",
            flush=True,
        )
        if env.enabled and args.load_stagger_sec > 0:
            delay = env.rank * args.load_stagger_sec
            print(f"[LOAD] rank={env.rank} stagger_sleep={delay:.1f}s", flush=True)
            time.sleep(delay)

        print(f"[STAGE] rank={env.rank} pipeline_init_start", flush=True)
        pipeline = MetaQueryAnimatePipelineFSDP(args, env)
        print(f"[STAGE] rank={env.rank} pipeline_init_done", flush=True)

        ref_image = Image.open(args.ref_image).convert("RGB")
        print(
            f"[STAGE] rank={env.rank} generate_start frame_num={args.frame_num} "
            f"steps={args.sampling_steps} size={args.width}x{args.height}",
            flush=True,
        )
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
        print(f"[STAGE] rank={env.rank} generate_done", flush=True)

        if env.rank == 0:
            base.save_video(video, args.output_path)
            print(f"[DONE] saved={args.output_path}")
    finally:
        destroy_dist(env)


if __name__ == "__main__":
    main()
