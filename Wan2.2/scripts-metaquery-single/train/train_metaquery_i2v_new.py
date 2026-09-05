import argparse
import os
import sys
import importlib.util
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))

try:
    from train_connector_for_wan import WanVideoDataset
    _DATASET_IMPORT_ERROR = None
except Exception as _e:
    try:
        _env_connector = os.environ.get("WAN_CONNECTOR_FILE", "").strip()
        _candidates = []
        if _env_connector:
            _candidates.append(Path(_env_connector))
        _candidates.extend(
            [
                Path(WAN_ROOT) / "train_connector_for_wan.py",
                Path(WAN_ROOT) / "Wan2.2" / "train_connector_for_wan.py",
                Path.cwd() / "train_connector_for_wan.py",
                Path.cwd() / "Wan2.2" / "train_connector_for_wan.py",
            ]
        )
        _connector_file = next((p for p in _candidates if p.exists()), None)
        if _connector_file is not None:
            _spec = importlib.util.spec_from_file_location("wan_train_connector_for_wan", str(_connector_file))
            if _spec is None or _spec.loader is None:
                raise RuntimeError(f"无法构建导入规格: {_connector_file}")
            _module = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_module)
            WanVideoDataset = _module.WanVideoDataset
            _DATASET_IMPORT_ERROR = None
            print(f"[IMPORT] WanVideoDataset fallback_file={_connector_file}")
        else:
            WanVideoDataset = None
            _DATASET_IMPORT_ERROR = RuntimeError(
                "未找到 train_connector_for_wan.py; "
                f"candidates={[str(p) for p in _candidates]} cwd={Path.cwd()}"
            )
    except Exception as _e2:
        WanVideoDataset = None
        _DATASET_IMPORT_ERROR = _e2


HF_RUNTIME = {}
BASE_I2V = None
DEFAULT_WAN_CKPT = "/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B"
DEFAULT_QWEN_CKPT = "/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking"


class HFWanDatasetAdapter(WanVideoDataset if WanVideoDataset is not None else object):
    def __init__(
        self,
        caption_json_root=None,
        manifest_path=None,
        frame_num=81,
        max_area=720 * 1280,
        null_caption_prob=0.1,
        null_image_prob=0.1,
        max_caption_tokens=512,
        caption_tokenizer_path="google/umt5-xxl",
        min_duration_sec=0.5,
        max_duration_sec=20.0,
        probe_missing_meta=False,
        seed=42,
        local_openvid_video_root=None,
        local_openvid_csv_path=None,
        local_openvid_limit=None,
        local_openvid_hd_video_root=None,
        local_openvid_hd_csv_path=None,
        local_openvid_hd_limit=None,
        local_video_cache_dir=None,
    ):
        if WanVideoDataset is None:
            raise RuntimeError(f"加载 WanVideoDataset 失败: {_DATASET_IMPORT_ERROR}")
        super().__init__(
            frame_num=frame_num,
            max_area=max_area,
            null_caption_prob=null_caption_prob,
            null_image_prob=null_image_prob,
            max_caption_tokens=max_caption_tokens,
            caption_tokenizer_path=caption_tokenizer_path,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            hf_stage=HF_RUNTIME.get("hf_stage", "stage1"),
            hf_dataset_name=HF_RUNTIME.get("hf_dataset_name", None),
            hf_split=HF_RUNTIME.get("hf_split", "train"),
            hf_subset_ratio=HF_RUNTIME.get("hf_subset_ratio", 0.01),
            hf_subset_size=HF_RUNTIME.get("hf_subset_size", None),
            hf_scan_factor=HF_RUNTIME.get("hf_scan_factor", 30),
            hf_subset_cache_dir=HF_RUNTIME.get("hf_subset_cache_dir", None),
            hf_subset_use_cache=HF_RUNTIME.get("hf_subset_use_cache", True),
            hf_cache_dir=HF_RUNTIME.get("hf_cache_dir", None),
            hf_streaming=HF_RUNTIME.get("hf_streaming", True),
            hf_shuffle_buffer=HF_RUNTIME.get("hf_shuffle_buffer", 10000),
            seed=HF_RUNTIME.get("hf_seed", seed),
            local_video_cache_dir=HF_RUNTIME.get("local_video_cache_dir", local_video_cache_dir),
            local_openvid_video_root=local_openvid_video_root,
            local_openvid_csv_path=local_openvid_csv_path,
            local_openvid_limit=local_openvid_limit,
            local_openvid_hd_video_root=local_openvid_hd_video_root,
            local_openvid_hd_csv_path=local_openvid_hd_csv_path,
            local_openvid_hd_limit=local_openvid_hd_limit,
        )


def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=False)
    # 两阶段总控开关：开启后自动串行跑 stage1->stage2
    parser.add_argument("--run_two_stage", action="store_true")
    parser.add_argument("--output_root", type=str, default="./metaquery_i2v_hf_two_stage")
    parser.add_argument("--stage1_steps", type=int, default=5000)
    parser.add_argument("--stage2_steps", type=int, default=3000)
    # 仅检查参数和路径，不实际进入训练
    parser.add_argument("--check_only", action="store_true")
    # 多卡并行参数：适配 torchrun / Determined 的环境变量
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--auto_device_map", action="store_true")
    parser.add_argument("--gpus_per_process", type=int, default=1)
    parser.add_argument("--hf_stage", type=str, default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--hf_dataset_name", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_subset_ratio", type=float, default=0.01)
    parser.add_argument("--hf_subset_size", type=int, default=None)
    parser.add_argument("--hf_scan_factor", type=int, default=30)
    parser.add_argument("--hf_subset_cache_dir", type=str, default=None)
    parser.add_argument("--hf_subset_use_cache", action="store_true")
    parser.add_argument("--hf_no_subset_cache", action="store_true")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--hf_streaming", action="store_true")
    parser.add_argument("--hf_no_streaming", action="store_true")
    parser.add_argument("--hf_shuffle_buffer", type=int, default=10000)
    parser.add_argument("--hf_seed", type=int, default=42)
    parser.add_argument("--local_video_cache_dir", type=str, default=None)
    known, remain = parser.parse_known_args()
    if known.hf_no_streaming:
        known.hf_streaming = False
    elif known.hf_streaming:
        known.hf_streaming = True
    else:
        known.hf_streaming = False
    known.hf_subset_use_cache = True if not known.hf_no_subset_cache else False
    return known, remain


def _set_or_append(argv, key, value):
    if key in argv:
        idx = argv.index(key)
        if idx + 1 < len(argv):
            argv[idx + 1] = str(value)
        else:
            argv.append(str(value))
    else:
        argv.extend([key, str(value)])
    return argv


def _get_arg_value(argv, key, default=""):
    if key in argv:
        idx = argv.index(key)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default


def _has_flag(argv, key):
    return key in argv


def _with_stage_wandb(argv, stage_name):
    out = list(argv)
    if not _has_flag(out, "--wandb_enabled"):
        return out
    run_name = str(_get_arg_value(out, "--wandb_run_name", "")).strip()
    if run_name:
        out = _set_or_append(out, "--wandb_run_name", f"{run_name}-{stage_name}")
    else:
        out = _set_or_append(out, "--wandb_run_name", f"wan-i2v-metaquery-{stage_name}")
    tags = str(_get_arg_value(out, "--wandb_tags", "")).strip()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if stage_name not in tag_list:
        tag_list.append(stage_name)
    out = _set_or_append(out, "--wandb_tags", ",".join(tag_list))
    return out


def _ensure_defaults(argv):
    # 用户未显式传入时，使用你提供的默认模型路径
    if "--wan_checkpoint_dir" not in argv:
        argv.extend(["--wan_checkpoint_dir", DEFAULT_WAN_CKPT])
    if "--qwen3vl_model_id" not in argv:
        argv.extend(["--qwen3vl_model_id", DEFAULT_QWEN_CKPT])
    return argv


def _print_plan(runtime, remain):
    output_root = runtime.output_root
    stage1_out = os.path.join(output_root, "stage1_openvid_1pct")
    stage2_out = os.path.join(output_root, "stage2_opens2v_1pct")
    stage1_final = os.path.join(stage1_out, "checkpoint-final", "mq_encoder_full.pt")
    stage2_final = os.path.join(stage2_out, "checkpoint-final", "mq_encoder_full.pt")
    print("[CHECK_ONLY] I2V 两阶段计划")
    print(f"[CHECK_ONLY] run_two_stage={runtime.run_two_stage}")
    print(f"[CHECK_ONLY] stage1_steps={runtime.stage1_steps}, stage2_steps={runtime.stage2_steps}")
    print(f"[CHECK_ONLY] stage1_output={stage1_out}")
    print(f"[CHECK_ONLY] stage2_output={stage2_out}")
    print(f"[CHECK_ONLY] stage1_final_exists={os.path.exists(stage1_final)}")
    print(f"[CHECK_ONLY] stage2_final_exists={os.path.exists(stage2_final)}")
    print(f"[CHECK_ONLY] passthrough_args={' '.join(remain)}")
    print(f"[CHECK_ONLY] auto_device_map={runtime.auto_device_map} gpus_per_process={runtime.gpus_per_process}")


def _dist_info(runtime):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = runtime.distributed or world_size > 1
    return enabled, world_size, rank, local_rank


def _resolve_process_device(runtime, local_rank):
    if not torch.cuda.is_available():
        return None
    cuda_count = torch.cuda.device_count()
    if cuda_count <= 0:
        return None
    if runtime.auto_device_map and int(runtime.gpus_per_process) > 1:
        start = local_rank * int(runtime.gpus_per_process)
        prefer = os.environ.get("WAN_DIST_PROCESS_DEVICE", "encoder").strip().lower()
        if prefer in ("dit", "model", "0"):
            device_idx = start
        else:
            device_idx = start + 1
    else:
        device_idx = local_rank
    if device_idx < 0 or device_idx >= cuda_count:
        raise ValueError(
            f"[DIST] 非法设备映射: local_rank={local_rank}, "
            f"gpus_per_process={runtime.gpus_per_process}, cuda_count={cuda_count}, "
            f"computed_device={device_idx}"
        )
    return int(device_idx)


def _init_dist_if_needed(runtime):
    enabled, world_size, rank, local_rank = _dist_info(runtime)
    process_device = _resolve_process_device(runtime, local_rank)
    if enabled and not dist.is_initialized():
        if process_device is not None:
            torch.cuda.set_device(process_device)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if process_device is not None and torch.cuda.is_available():
            try:
                dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{process_device}"))
            except TypeError:
                dist.init_process_group(backend=backend)
            _warm = torch.zeros(1, device=torch.device(f"cuda:{process_device}"))
            dist.all_reduce(_warm)
        else:
            dist.init_process_group(backend=backend)
    if enabled:
        print(
            f"[DIST] enabled=True world_size={world_size} rank={rank} local_rank={local_rank} "
            f"process_device={process_device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
    return enabled, world_size, rank, local_rank, process_device


def _destroy_dist_if_needed(enabled):
    if enabled and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[DIST][WARN] destroy_process_group failed: {e}")


def _resolve_device_args(runtime, remain, local_rank):
    argv = list(remain)
    if "--dit_device" in argv and "--encoder_device" in argv:
        return argv
    if runtime.auto_device_map:
        start = local_rank * max(1, runtime.gpus_per_process)
        dit = start
        enc = start + 1 if runtime.gpus_per_process > 1 else start
        argv = _set_or_append(argv, "--dit_device", dit)
        argv = _set_or_append(argv, "--encoder_device", enc)
    return argv


def _patch_trainer_for_ddp(trainer, enabled, rank):
    if not enabled:
        return trainer
    enc_device = int(trainer.args.encoder_device)
    trainer.mq_encoder = DDP(
        trainer.mq_encoder,
        device_ids=[enc_device],
        output_device=enc_device,
        find_unused_parameters=False,
    )
    original_save = trainer._save_checkpoint

    def _save_rank0_only(*args, **kwargs):
        if rank == 0:
            original_save(*args, **kwargs)

    trainer._save_checkpoint = _save_rank0_only
    return trainer


def _run_single_train(runtime, remain):
    enabled, _, rank, local_rank, process_device = _init_dist_if_needed(runtime)
    try:
        base_i2v = _load_base_i2v()
        base_i2v.TomAndJerryVideoDataset = HFWanDatasetAdapter
        resolved_argv = _resolve_device_args(runtime, remain, local_rank)
        print(f"[PIPELINE] resolved_argv={' '.join(map(str, resolved_argv))}")
        print(
            f"[PIPELINE] hf_stage={HF_RUNTIME.get('hf_stage')} "
            f"hf_dataset_name={HF_RUNTIME.get('hf_dataset_name')} "
            f"subset_ratio={HF_RUNTIME.get('hf_subset_ratio')} "
            f"subset_size={HF_RUNTIME.get('hf_subset_size')}"
        )
        original = list(sys.argv)
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        if enabled and rank != 0:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            sys.argv = [sys.argv[0]] + resolved_argv
            args = base_i2v.parse_args()
            args.seed = int(args.seed) + rank
            print(
                f"[TRAIN] rank={rank} dit_device={args.dit_device} encoder_device={args.encoder_device} "
                f"num_train_steps={args.num_train_steps} output_dir={args.output_dir}"
            )
            trainer = base_i2v.MetaQueryI2VTrainer(args)
            if enabled and rank != 0:
                if old_hf_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_hf_offline
                if old_tf_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
            trainer = _patch_trainer_for_ddp(trainer, enabled, rank)
            trainer.train()
            if enabled and dist.is_initialized():
                if process_device is not None and torch.cuda.is_available():
                    dist.barrier(device_ids=[process_device])
                else:
                    dist.barrier()
        finally:
            if enabled and rank != 0:
                if old_hf_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_hf_offline
                if old_tf_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
            sys.argv = original
    finally:
        _destroy_dist_if_needed(enabled)


def _run_one_stage(base_argv, runtime, stage_name, output_dir, steps, resume_path=None):
    HF_RUNTIME.update(
        {
            "hf_stage": stage_name,
            "hf_dataset_name": None,
            "hf_split": runtime.hf_split,
            "hf_subset_ratio": runtime.hf_subset_ratio,
            "hf_subset_size": runtime.hf_subset_size,
            "hf_scan_factor": runtime.hf_scan_factor,
            "hf_subset_cache_dir": runtime.hf_subset_cache_dir,
            "hf_subset_use_cache": runtime.hf_subset_use_cache,
            "hf_cache_dir": runtime.hf_cache_dir,
            "hf_streaming": runtime.hf_streaming,
            "hf_shuffle_buffer": runtime.hf_shuffle_buffer,
            "hf_seed": runtime.hf_seed,
            "local_video_cache_dir": runtime.local_video_cache_dir,
        }
    )
    argv = list(base_argv)
    argv = _with_stage_wandb(argv, stage_name)
    argv = _set_or_append(argv, "--output_dir", output_dir)
    argv = _set_or_append(argv, "--num_train_steps", steps)
    if resume_path is not None:
        argv = _set_or_append(argv, "--resume_mq_encoder_path", resume_path)
    _run_single_train(runtime, argv)


def _load_base_i2v():
    global BASE_I2V
    if BASE_I2V is None:
        import train_metaquery_i2v as base_i2v
        BASE_I2V = base_i2v
    return BASE_I2V


if __name__ == "__main__":
    runtime, remain = parse_runtime_args()
    remain = _ensure_defaults(remain)
    if runtime.check_only:
        _print_plan(runtime, remain)
        sys.exit(0)
    if not runtime.run_two_stage:
        HF_RUNTIME.update(vars(runtime))
        _run_single_train(runtime, remain)
    else:
        output_root = runtime.output_root
        stage1_out = os.path.join(output_root, "stage1_openvid_1pct")
        stage2_out = os.path.join(output_root, "stage2_opens2v_1pct")
        stage1_final = os.path.join(stage1_out, "checkpoint-final", "mq_encoder_full.pt")
        stage2_final = os.path.join(stage2_out, "checkpoint-final", "mq_encoder_full.pt")
        run_stage1 = not os.path.exists(stage1_final)
        print(f"[TWO_STAGE] stage1_path={stage1_out} exists={not run_stage1}")
        if run_stage1:
            _run_one_stage(remain, runtime, "stage1", stage1_out, runtime.stage1_steps, None)
        run_stage2 = not os.path.exists(stage2_final)
        print(f"[TWO_STAGE] stage2_path={stage2_out} exists={not run_stage2}")
        if run_stage2:
            _run_one_stage(remain, runtime, "stage2", stage2_out, runtime.stage2_steps, stage1_final)
