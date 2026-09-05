# import sys
# import argparse
# import os
# import importlib.util
# from pathlib import Path
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# WAN_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(WAN_ROOT))

# import train_metaquery_wan as base_ti2v
# try:
#     from train_connector_for_wan import WanVideoDataset
#     _DATASET_IMPORT_ERROR = None
# except Exception as _e:
#     try:
#         _env_connector = os.environ.get("WAN_CONNECTOR_FILE", "").strip()
#         _candidates = []
#         if _env_connector:
#             _candidates.append(Path(_env_connector))
#         _candidates.extend(
#             [
#                 Path(WAN_ROOT) / "train_connector_for_wan.py",
#                 Path(WAN_ROOT) / "Wan2.2" / "train_connector_for_wan.py",
#                 Path.cwd() / "train_connector_for_wan.py",
#                 Path.cwd() / "Wan2.2" / "train_connector_for_wan.py",
#             ]
#         )
#         _connector_file = next((p for p in _candidates if p.exists()), None)
#         if _connector_file is not None:
#             _spec = importlib.util.spec_from_file_location("wan_train_connector_for_wan", str(_connector_file))
#             if _spec is None or _spec.loader is None:
#                 raise RuntimeError(f"无法构建导入规格: {_connector_file}")
#             _module = importlib.util.module_from_spec(_spec)
#             _spec.loader.exec_module(_module)
#             WanVideoDataset = _module.WanVideoDataset
#             _DATASET_IMPORT_ERROR = None
#             print(f"[IMPORT] WanVideoDataset fallback_file={_connector_file}")
#         else:
#             WanVideoDataset = None
#             _DATASET_IMPORT_ERROR = RuntimeError(
#                 "未找到 train_connector_for_wan.py; "
#                 f"candidates={[str(p) for p in _candidates]} cwd={Path.cwd()}"
#             )
#     except Exception as _e2:
#         WanVideoDataset = None
#         _DATASET_IMPORT_ERROR = _e2


# HF_RUNTIME = {}


# class HFWanDatasetAdapter(WanVideoDataset if WanVideoDataset is not None else object):
#     def __init__(
#         self,
#         frame_num=41,
#         max_area=480 * 832,
#         null_caption_prob=0.1,
#         null_image_prob=0.1,
#         max_caption_tokens=512,
#         caption_tokenizer_path="google/umt5-xxl",
#         min_duration_sec=0.5,
#         max_duration_sec=20.0,
#         seed=42,
#         local_openvid_video_root=None,
#         local_openvid_csv_path=None,
#         local_openvid_limit=None,
#         local_openvid_hd_video_root=None,
#         local_openvid_hd_csv_path=None,
#         local_openvid_hd_limit=None,
#         local_video_cache_dir=None,
#     ):
#         if WanVideoDataset is None:
#             raise RuntimeError(f"加载 WanVideoDataset 失败: {_DATASET_IMPORT_ERROR}")
#         super().__init__(
#             frame_num=frame_num,
#             max_area=max_area,
#             null_caption_prob=null_caption_prob,
#             null_image_prob=null_image_prob,
#             max_caption_tokens=max_caption_tokens,
#             caption_tokenizer_path=caption_tokenizer_path,
#             min_duration_sec=min_duration_sec,
#             max_duration_sec=max_duration_sec,
#             hf_stage=HF_RUNTIME.get("hf_stage", "stage1"),
#             hf_dataset_name=HF_RUNTIME.get("hf_dataset_name", None),
#             hf_split=HF_RUNTIME.get("hf_split", "train"),
#             hf_subset_ratio=HF_RUNTIME.get("hf_subset_ratio", 0.01),
#             hf_subset_size=HF_RUNTIME.get("hf_subset_size", None),
#             hf_scan_factor=HF_RUNTIME.get("hf_scan_factor", 30),
#             hf_subset_cache_dir=HF_RUNTIME.get("hf_subset_cache_dir", None),
#             hf_subset_use_cache=HF_RUNTIME.get("hf_subset_use_cache", True),
#             hf_cache_dir=HF_RUNTIME.get("hf_cache_dir", None),
#             hf_streaming=HF_RUNTIME.get("hf_streaming", True),
#             hf_shuffle_buffer=HF_RUNTIME.get("hf_shuffle_buffer", 10000),
#             seed=HF_RUNTIME.get("hf_seed", seed),
#             local_video_cache_dir=HF_RUNTIME.get("local_video_cache_dir", local_video_cache_dir),
#             local_openvid_video_root=local_openvid_video_root,
#             local_openvid_csv_path=local_openvid_csv_path,
#             local_openvid_limit=local_openvid_limit,
#             local_openvid_hd_video_root=local_openvid_hd_video_root,
#             local_openvid_hd_csv_path=local_openvid_hd_csv_path,
#             local_openvid_hd_limit=local_openvid_hd_limit,
#         )


# def parse_runtime_args():
#     parser = argparse.ArgumentParser(add_help=False)
#     parser.add_argument("--distributed", action="store_true")
#     parser.add_argument("--auto_device_map", action="store_true")
#     parser.add_argument("--gpus_per_process", type=int, default=1)
#     parser.add_argument("--check_only", action="store_true")
#     parser.add_argument("--hf_stage", type=str, default="stage1", choices=["stage1", "stage2"])
#     parser.add_argument("--hf_dataset_name", type=str, default=None)
#     parser.add_argument("--hf_split", type=str, default="train")
#     parser.add_argument("--hf_subset_ratio", type=float, default=0.01)
#     parser.add_argument("--hf_subset_size", type=int, default=None)
#     parser.add_argument("--hf_scan_factor", type=int, default=30)
#     parser.add_argument("--hf_subset_cache_dir", type=str, default=None)
#     parser.add_argument("--hf_subset_use_cache", action="store_true")
#     parser.add_argument("--hf_no_subset_cache", action="store_true")
#     parser.add_argument("--hf_cache_dir", type=str, default=None)
#     parser.add_argument("--hf_streaming", action="store_true")
#     parser.add_argument("--hf_no_streaming", action="store_true")
#     parser.add_argument("--hf_shuffle_buffer", type=int, default=10000)
#     parser.add_argument("--hf_seed", type=int, default=42)
#     parser.add_argument("--local_video_cache_dir", type=str, default=None)
#     known, remain = parser.parse_known_args()
#     if known.hf_no_streaming:
#         known.hf_streaming = False
#     elif known.hf_streaming:
#         known.hf_streaming = True
#     else:
#         known.hf_streaming = False
#     known.hf_subset_use_cache = True if not known.hf_no_subset_cache else False
#     return known, remain


# def _set_or_append(argv, key, value):
#     if key in argv:
#         idx = argv.index(key)
#         if idx + 1 < len(argv):
#             argv[idx + 1] = str(value)
#         else:
#             argv.append(str(value))
#     else:
#         argv.extend([key, str(value)])
#     return argv


# def _dist_info(runtime):
#     world_size = int(os.environ.get("WORLD_SIZE", "1"))
#     rank = int(os.environ.get("RANK", "0"))
#     local_rank = int(os.environ.get("LOCAL_RANK", "0"))
#     enabled = runtime.distributed or world_size > 1
#     return enabled, world_size, rank, local_rank


# def _resolve_process_device(runtime, local_rank):
#     if not torch.cuda.is_available():
#         return None
#     cuda_count = torch.cuda.device_count()
#     if cuda_count <= 0:
#         return None
#     if runtime.auto_device_map and int(runtime.gpus_per_process) > 1:
#         start = local_rank * int(runtime.gpus_per_process)
#         prefer = os.environ.get("WAN_DIST_PROCESS_DEVICE", "encoder").strip().lower()
#         if prefer in ("dit", "model", "0"):
#             device_idx = start
#         else:
#             # 默认绑定 encoder 卡，兼容当前 DDP(mq_encoder) 路径
#             device_idx = start + 1
#     else:
#         device_idx = local_rank
#     if device_idx < 0 or device_idx >= cuda_count:
#         raise ValueError(
#             f"[DIST] 非法设备映射: local_rank={local_rank}, "
#             f"gpus_per_process={runtime.gpus_per_process}, cuda_count={cuda_count}, "
#             f"computed_device={device_idx}"
#         )
#     return int(device_idx)


# def _init_dist_if_needed(runtime):
#     enabled, world_size, rank, local_rank = _dist_info(runtime)
#     process_device = _resolve_process_device(runtime, local_rank)
#     if enabled and not dist.is_initialized():
#         if process_device is not None:
#             torch.cuda.set_device(process_device)
#         backend = "nccl" if torch.cuda.is_available() else "gloo"
#         if process_device is not None and torch.cuda.is_available():
#             try:
#                 dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{process_device}"))
#             except TypeError:
#                 dist.init_process_group(backend=backend)
#             # 触发一次轻量通信，让 PG 记录本 rank 的设备，避免后续 barrier 警告。
#             _warm = torch.zeros(1, device=torch.device(f"cuda:{process_device}"))
#             dist.all_reduce(_warm)
#         else:
#             dist.init_process_group(backend=backend)
#     if enabled:
#         print(
#             f"[DIST] enabled=True world_size={world_size} rank={rank} local_rank={local_rank} "
#             f"process_device={process_device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
#             f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
#         )
#     return enabled, world_size, rank, local_rank, process_device


# def _destroy_dist_if_needed(enabled):
#     if enabled and dist.is_initialized():
#         try:
#             dist.destroy_process_group()
#         except Exception as e:
#             print(f"[DIST][WARN] destroy_process_group failed: {e}")


# def _resolve_device_args(runtime, remain, local_rank):
#     argv = list(remain)
#     if "--dit_device" in argv and "--encoder_device" in argv:
#         return argv
#     if runtime.auto_device_map:
#         if int(runtime.gpus_per_process) <= 1:
#             dit = local_rank
#             enc = local_rank
#         else:
#             start = local_rank * max(1, runtime.gpus_per_process)
#             dit = start
#             enc = start + 1
#         argv = _set_or_append(argv, "--dit_device", dit)
#         argv = _set_or_append(argv, "--encoder_device", enc)
#     return argv


# def _patch_trainer_for_ddp(trainer, enabled, rank):
#     if not enabled:
#         return trainer
#     enc_device = int(trainer.args.encoder_device)
#     bucket_cap_mb = int(os.environ.get("WAN_DDP_BUCKET_CAP_MB", "25"))
#     grad_bucket_view = os.environ.get("WAN_DDP_GRAD_BUCKET_VIEW", "1") == "1"
#     ddp_kwargs = dict(
#         device_ids=[enc_device],
#         output_device=enc_device,
#         find_unused_parameters=False,
#         bucket_cap_mb=bucket_cap_mb,
#         gradient_as_bucket_view=grad_bucket_view,
#     )
#     try:
#         trainer.mq_encoder = DDP(trainer.mq_encoder, **ddp_kwargs)
#     except TypeError:
#         ddp_kwargs.pop("gradient_as_bucket_view", None)
#         trainer.mq_encoder = DDP(trainer.mq_encoder, **ddp_kwargs)
#     print(
#         f"[DIST][DDP] encoder_device={enc_device} "
#         f"bucket_cap_mb={bucket_cap_mb} gradient_as_bucket_view={grad_bucket_view}"
#     )
#     if hasattr(trainer, "_save_checkpoint"):
#         original_save = trainer._save_checkpoint

#         def _save_rank0_only(*args, **kwargs):
#             if rank == 0:
#                 original_save(*args, **kwargs)

#         trainer._save_checkpoint = _save_rank0_only
#     if hasattr(trainer, "post_wrap_ddp_audit"):
#         trainer.post_wrap_ddp_audit()
#     return trainer


# def _run_single_train(runtime, remain):
#     enabled, _, rank, local_rank, process_device = _init_dist_if_needed(runtime)
#     try:
#         base_ti2v.WanDatasetClass = HFWanDatasetAdapter
#         resolved_argv = _resolve_device_args(runtime, remain, local_rank)
#         print(f"[PIPELINE] resolved_argv={' '.join(map(str, resolved_argv))}")
#         original = list(sys.argv)
#         old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
#         old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
#         if enabled and rank != 0:
#             os.environ["HF_HUB_OFFLINE"] = "1"
#             os.environ["TRANSFORMERS_OFFLINE"] = "1"
#         try:
#             sys.argv = [sys.argv[0]] + resolved_argv
#             args = base_ti2v.parse_args()
#             args.seed = int(args.seed) + rank
#             print(
#                 f"[TRAIN] rank={rank} dit_device={args.dit_device} encoder_device={args.encoder_device} "
#                 f"num_train_steps={args.num_train_steps} output_dir={args.output_dir}"
#             )
#             trainer = base_ti2v.MetaQueryWanTrainer(args)
#             if enabled and rank != 0:
#                 if old_hf_offline is None:
#                     os.environ.pop("HF_HUB_OFFLINE", None)
#                 else:
#                     os.environ["HF_HUB_OFFLINE"] = old_hf_offline
#                 if old_tf_offline is None:
#                     os.environ.pop("TRANSFORMERS_OFFLINE", None)
#                 else:
#                     os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
#             trainer = _patch_trainer_for_ddp(trainer, enabled, rank)
#             trainer.train()
#             if enabled and dist.is_initialized():
#                 if process_device is not None and torch.cuda.is_available():
#                     dist.barrier(device_ids=[process_device])
#                 else:
#                     dist.barrier()
#         finally:
#             if enabled and rank != 0:
#                 if old_hf_offline is None:
#                     os.environ.pop("HF_HUB_OFFLINE", None)
#                 else:
#                     os.environ["HF_HUB_OFFLINE"] = old_hf_offline
#                 if old_tf_offline is None:
#                     os.environ.pop("TRANSFORMERS_OFFLINE", None)
#                 else:
#                     os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
#             sys.argv = original
#     finally:
#         _destroy_dist_if_needed(enabled)


# if __name__ == "__main__":
#     runtime, remain = parse_runtime_args()
#     if runtime.check_only:
#         print(
#             "[CHECK_ONLY] "
#             f"distributed={runtime.distributed} auto_device_map={runtime.auto_device_map} "
#             f"gpus_per_process={runtime.gpus_per_process} passthrough_args={' '.join(remain)}"
#         )
#         sys.exit(0)
#     HF_RUNTIME.update(vars(runtime))
#     _run_single_train(runtime, remain)


















# # 上面是正常的t5 + mq作为DIT条件，这个是只mq
# import sys
# import argparse
# import os
# import importlib
# import importlib.util
# from pathlib import Path
# from datetime import timedelta
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# WAN_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(WAN_ROOT))

# def _load_base_ti2v_module():
#     """
#     仅加载明确允许的基础 TI2V 训练模块。
#     默认使用 train_metaquery_wan（无 animate-like/hard-lock 专用逻辑）。
#     可选通过 WAN_BASE_TI2V_MODULE 显式指定允许模块：
#       - train_metaquery_wan
#       - train_metaquery_wan_animate_like_v2
#     WAN_BASE_TI2V_FILE 仅作为同名模块的文件路径兜底。
#     """
#     default_name = "train_metaquery_wan"
#     allowed_names = {"train_metaquery_wan", "train_metaquery_wan_animate_like_v2"}
#     prefer = (os.environ.get("WAN_BASE_TI2V_MODULE", "") or "").strip() or default_name
#     if prefer not in allowed_names:
#         raise RuntimeError(
#             f"仅允许模块 {sorted(allowed_names)}，当前 WAN_BASE_TI2V_MODULE={prefer}"
#         )

#     last_err = None
#     try:
#         mod = importlib.import_module(prefer)
#         print(f"[IMPORT] base_ti2v_module={prefer}")
#         return mod
#     except Exception as e:
#         last_err = e

#     env_file = (os.environ.get("WAN_BASE_TI2V_FILE", "") or "").strip()
#     if env_file:
#         path = Path(env_file).expanduser().resolve()
#         if path.exists():
#             try:
#                 spec = importlib.util.spec_from_file_location("wan_base_ti2v", str(path))
#                 if spec is None or spec.loader is None:
#                     raise RuntimeError(f"无法构建导入规格: {path}")
#                 mod = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(mod)
#                 mod_name = str(getattr(mod, "__name__", ""))
#                 if prefer not in mod_name and path.stem != prefer:
#                     raise RuntimeError(
#                         f"WAN_BASE_TI2V_FILE 必须对应 {prefer}，当前文件={path}"
#                     )
#                 print(f"[IMPORT] base_ti2v_file={path}")
#                 return mod
#             except Exception as e:
#                 last_err = e

#     raise RuntimeError(
#         "无法加载基础 TI2V 训练模块。"
#         f"目标模块={prefer}，WAN_BASE_TI2V_FILE={env_file or '<unset>'}，last_err={last_err}"
#     )

# base_ti2v = _load_base_ti2v_module()
# if not hasattr(base_ti2v, "DistributedSampler"):
#     base_ti2v.DistributedSampler = torch.utils.data.distributed.DistributedSampler
#     print("[PATCH] Injected torch.utils.data.distributed.DistributedSampler into base_ti2v")
# try:
#     from train_connector_for_wan import WanVideoDataset
#     _DATASET_IMPORT_ERROR = None
# except Exception as _e:
#     try:
#         _env_connector = os.environ.get("WAN_CONNECTOR_FILE", "").strip()
#         _candidates = []
#         if _env_connector:
#             _candidates.append(Path(_env_connector))
#         _candidates.extend(
#             [
#                 Path(WAN_ROOT) / "train_connector_for_wan.py",
#                 Path(WAN_ROOT) / "Wan2.2" / "train_connector_for_wan.py",
#                 Path.cwd() / "train_connector_for_wan.py",
#                 Path.cwd() / "Wan2.2" / "train_connector_for_wan.py",
#             ]
#         )
#         _connector_file = next((p for p in _candidates if p.exists()), None)
#         if _connector_file is not None:
#             _spec = importlib.util.spec_from_file_location("wan_train_connector_for_wan", str(_connector_file))
#             if _spec is None or _spec.loader is None:
#                 raise RuntimeError(f"无法构建导入规格: {_connector_file}")
#             _module = importlib.util.module_from_spec(_spec)
#             _spec.loader.exec_module(_module)
#             WanVideoDataset = _module.WanVideoDataset
#             _DATASET_IMPORT_ERROR = None
#             print(f"[IMPORT] WanVideoDataset fallback_file={_connector_file}")
#         else:
#             WanVideoDataset = None
#             _DATASET_IMPORT_ERROR = RuntimeError(
#                 "未找到 train_connector_for_wan.py; "
#                 f"candidates={[str(p) for p in _candidates]} cwd={Path.cwd()}"
#             )
#     except Exception as _e2:
#         WanVideoDataset = None
#         _DATASET_IMPORT_ERROR = _e2


# HF_RUNTIME = {}


# class HFWanDatasetAdapter(WanVideoDataset if WanVideoDataset is not None else object):
#     def __init__(
#         self,
#         frame_num=41,
#         max_area=480 * 832,
#         null_caption_prob=0.1,
#         null_image_prob=0.1,
#         max_caption_tokens=512,
#         caption_tokenizer_path="google/umt5-xxl",
#         min_duration_sec=0.5,
#         max_duration_sec=20.0,
#         seed=42,
#         local_openvid_video_root=None,
#         local_openvid_csv_path=None,
#         local_openvid_limit=None,
#         local_openvid_hd_video_root=None,
#         local_openvid_hd_csv_path=None,
#         local_openvid_hd_limit=None,
#         local_video_cache_dir=None,
#     ):
#         if WanVideoDataset is None:
#             raise RuntimeError(f"加载 WanVideoDataset 失败: {_DATASET_IMPORT_ERROR}")
#         super().__init__(
#             frame_num=frame_num,
#             max_area=max_area,
#             null_caption_prob=null_caption_prob,
#             null_image_prob=null_image_prob,
#             max_caption_tokens=max_caption_tokens,
#             caption_tokenizer_path=caption_tokenizer_path,
#             min_duration_sec=min_duration_sec,
#             max_duration_sec=max_duration_sec,
#             hf_stage=HF_RUNTIME.get("hf_stage", "stage1"),
#             hf_dataset_name=HF_RUNTIME.get("hf_dataset_name", None),
#             hf_split=HF_RUNTIME.get("hf_split", "train"),
#             hf_subset_ratio=HF_RUNTIME.get("hf_subset_ratio", 0.01),
#             hf_subset_size=HF_RUNTIME.get("hf_subset_size", None),
#             hf_scan_factor=HF_RUNTIME.get("hf_scan_factor", 30),
#             hf_subset_cache_dir=HF_RUNTIME.get("hf_subset_cache_dir", None),
#             hf_subset_use_cache=HF_RUNTIME.get("hf_subset_use_cache", True),
#             hf_cache_dir=HF_RUNTIME.get("hf_cache_dir", None),
#             hf_streaming=HF_RUNTIME.get("hf_streaming", True),
#             hf_shuffle_buffer=HF_RUNTIME.get("hf_shuffle_buffer", 10000),
#             seed=HF_RUNTIME.get("hf_seed", seed),
#             local_video_cache_dir=HF_RUNTIME.get("local_video_cache_dir", local_video_cache_dir),
#             local_openvid_video_root=local_openvid_video_root,
#             local_openvid_csv_path=local_openvid_csv_path,
#             local_openvid_limit=local_openvid_limit,
#             local_openvid_hd_video_root=local_openvid_hd_video_root,
#             local_openvid_hd_csv_path=local_openvid_hd_csv_path,
#             local_openvid_hd_limit=local_openvid_hd_limit,
#         )


# def parse_runtime_args():
#     parser = argparse.ArgumentParser(add_help=False)
#     parser.add_argument("--distributed", action="store_true")
#     parser.add_argument("--auto_device_map", action="store_true")
#     parser.add_argument("--gpus_per_process", type=int, default=1)
#     parser.add_argument("--check_only", action="store_true")
#     parser.add_argument("--hf_stage", type=str, default="stage1", choices=["stage1", "stage2"])
#     parser.add_argument("--hf_dataset_name", type=str, default=None)
#     parser.add_argument("--hf_split", type=str, default="train")
#     parser.add_argument("--hf_subset_ratio", type=float, default=0.01)
#     parser.add_argument("--hf_subset_size", type=int, default=None)
#     parser.add_argument("--hf_scan_factor", type=int, default=30)
#     parser.add_argument("--hf_subset_cache_dir", type=str, default=None)
#     parser.add_argument("--hf_subset_use_cache", action="store_true")
#     parser.add_argument("--hf_no_subset_cache", action="store_true")
#     parser.add_argument("--hf_cache_dir", type=str, default=None)
#     parser.add_argument("--hf_streaming", action="store_true")
#     parser.add_argument("--hf_no_streaming", action="store_true")
#     parser.add_argument("--hf_shuffle_buffer", type=int, default=10000)
#     parser.add_argument("--hf_seed", type=int, default=42)
#     parser.add_argument("--local_video_cache_dir", type=str, default=None)
#     known, remain = parser.parse_known_args()
#     if known.hf_no_streaming:
#         known.hf_streaming = False
#     elif known.hf_streaming:
#         known.hf_streaming = True
#     else:
#         known.hf_streaming = False
#     known.hf_subset_use_cache = True if not known.hf_no_subset_cache else False
#     return known, remain


# def _set_or_append(argv, key, value):
#     if key in argv:
#         idx = argv.index(key)
#         if idx + 1 < len(argv):
#             argv[idx + 1] = str(value)
#         else:
#             argv.append(str(value))
#     else:
#         argv.extend([key, str(value)])
#     return argv


# def _dist_info(runtime):
#     world_size = int(os.environ.get("WORLD_SIZE", "1"))
#     rank = int(os.environ.get("RANK", "0"))
#     local_rank = int(os.environ.get("LOCAL_RANK", "0"))
#     enabled = runtime.distributed or world_size > 1
#     return enabled, world_size, rank, local_rank


# def _resolve_process_device(runtime, local_rank):
#     if not torch.cuda.is_available():
#         return None
#     cuda_count = torch.cuda.device_count()
#     if cuda_count <= 0:
#         return None
#     if runtime.auto_device_map and int(runtime.gpus_per_process) > 1:
#         start = local_rank * int(runtime.gpus_per_process)
#         prefer = os.environ.get("WAN_DIST_PROCESS_DEVICE", "encoder").strip().lower()
#         if prefer in ("dit", "model", "0"):
#             device_idx = start
#         else:
#             # 默认绑定 encoder 卡，兼容当前 DDP(mq_encoder) 路径
#             device_idx = start + 1
#     else:
#         device_idx = local_rank
#     if device_idx < 0 or device_idx >= cuda_count:
#         raise ValueError(
#             f"[DIST] 非法设备映射: local_rank={local_rank}, "
#             f"gpus_per_process={runtime.gpus_per_process}, cuda_count={cuda_count}, "
#             f"computed_device={device_idx}"
#         )
#     return int(device_idx)


# def _init_dist_if_needed(runtime):
#     enabled, world_size, rank, local_rank = _dist_info(runtime)
#     process_device = _resolve_process_device(runtime, local_rank)
#     if enabled and not dist.is_initialized():
#         timeout_sec = max(60, int(os.environ.get("WAN_DIST_TIMEOUT_SEC", "3600")))
#         pg_timeout = timedelta(seconds=timeout_sec)
#         if process_device is not None:
#             torch.cuda.set_device(process_device)
#         backend = "nccl" if torch.cuda.is_available() else "gloo"
#         if process_device is not None and torch.cuda.is_available():
#             try:
#                 dist.init_process_group(
#                     backend=backend,
#                     device_id=torch.device(f"cuda:{process_device}"),
#                     timeout=pg_timeout,
#                 )
#             except TypeError:
#                 dist.init_process_group(backend=backend, timeout=pg_timeout)
#             # 触发一次轻量通信，让 PG 记录本 rank 的设备，避免后续 barrier 警告。
#             _warm = torch.zeros(1, device=torch.device(f"cuda:{process_device}"))
#             dist.all_reduce(_warm)
#         else:
#             dist.init_process_group(backend=backend, timeout=pg_timeout)
#     if enabled:
#         print(
#             f"[DIST] enabled=True world_size={world_size} rank={rank} local_rank={local_rank} "
#             f"process_device={process_device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
#             f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
#             f"timeout_sec={os.environ.get('WAN_DIST_TIMEOUT_SEC', '3600')}"
#         )
#     return enabled, world_size, rank, local_rank, process_device


# def _destroy_dist_if_needed(enabled):
#     if enabled and dist.is_initialized():
#         try:
#             dist.destroy_process_group()
#         except Exception as e:
#             print(f"[DIST][WARN] destroy_process_group failed: {e}")


# def _resolve_device_args(runtime, remain, local_rank):
#     argv = list(remain)
#     if not ("--dit_device" in argv and "--encoder_device" in argv) and runtime.auto_device_map:
#         if int(runtime.gpus_per_process) <= 1:
#             dit = local_rank
#             enc = local_rank
#         else:
#             start = local_rank * max(1, runtime.gpus_per_process)
#             dit = start
#             enc = start + 1
#         argv = _set_or_append(argv, "--dit_device", dit)
#         argv = _set_or_append(argv, "--encoder_device", enc)
#     # 统一强制 MQ-only DiT 条件注入，避免调用方遗漏参数。
#     argv = _set_or_append(argv, "--dit_condition_mode", "mq_only")
#     return argv


# def _patch_trainer_for_ddp(trainer, enabled, rank):
#     if not enabled:
#         return trainer
#     enc_device = int(trainer.args.encoder_device)
#     bucket_cap_mb = int(os.environ.get("WAN_DDP_BUCKET_CAP_MB", "25"))
#     grad_bucket_view = os.environ.get("WAN_DDP_GRAD_BUCKET_VIEW", "1") == "1"
#     ddp_kwargs = dict(
#         device_ids=[enc_device],
#         output_device=enc_device,
#         find_unused_parameters=False,
#         broadcast_buffers=False,
#         bucket_cap_mb=bucket_cap_mb,
#         gradient_as_bucket_view=grad_bucket_view,
#     )
#     try:
#         trainer.mq_encoder = DDP(trainer.mq_encoder, **ddp_kwargs)
#     except TypeError:
#         ddp_kwargs.pop("gradient_as_bucket_view", None)
#         trainer.mq_encoder = DDP(trainer.mq_encoder, **ddp_kwargs)
#     print(
#         f"[DIST][DDP] encoder_device={enc_device} "
#         f"bucket_cap_mb={bucket_cap_mb} gradient_as_bucket_view={grad_bucket_view}"
#     )
#     if hasattr(trainer, "_save_checkpoint"):
#         original_save = trainer._save_checkpoint

#         def _save_rank0_only(*args, **kwargs):
#             if rank == 0:
#                 original_save(*args, **kwargs)

#         trainer._save_checkpoint = _save_rank0_only
#     if hasattr(trainer, "post_wrap_ddp_audit"):
#         trainer.post_wrap_ddp_audit()
#     return trainer


# def _run_single_train(runtime, remain):
#     enabled, _, rank, local_rank, process_device = _init_dist_if_needed(runtime)
#     try:
#         base_ti2v.WanDatasetClass = HFWanDatasetAdapter
#         resolved_argv = _resolve_device_args(runtime, remain, local_rank)
#         print(f"[PIPELINE] resolved_argv={' '.join(map(str, resolved_argv))}")
#         original = list(sys.argv)
#         old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
#         old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
#         if enabled and rank != 0:
#             os.environ["HF_HUB_OFFLINE"] = "1"
#             os.environ["TRANSFORMERS_OFFLINE"] = "1"
#         try:
#             sys.argv = [sys.argv[0]] + resolved_argv
#             args = base_ti2v.parse_args()
#             args.seed = int(args.seed) + rank
#             print(
#                 f"[TRAIN] rank={rank} dit_device={args.dit_device} encoder_device={args.encoder_device} "
#                 f"num_train_steps={args.num_train_steps} output_dir={args.output_dir}"
#             )
#             trainer = base_ti2v.MetaQueryWanTrainer(args)
#             if enabled and rank != 0:
#                 if old_hf_offline is None:
#                     os.environ.pop("HF_HUB_OFFLINE", None)
#                 else:
#                     os.environ["HF_HUB_OFFLINE"] = old_hf_offline
#                 if old_tf_offline is None:
#                     os.environ.pop("TRANSFORMERS_OFFLINE", None)
#                 else:
#                     os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
#             trainer = _patch_trainer_for_ddp(trainer, enabled, rank)
#             trainer.train()
#             if enabled and dist.is_initialized():
#                 if process_device is not None and torch.cuda.is_available():
#                     dist.barrier(device_ids=[process_device])
#                 else:
#                     dist.barrier()
#         finally:
#             if enabled and rank != 0:
#                 if old_hf_offline is None:
#                     os.environ.pop("HF_HUB_OFFLINE", None)
#                 else:
#                     os.environ["HF_HUB_OFFLINE"] = old_hf_offline
#                 if old_tf_offline is None:
#                     os.environ.pop("TRANSFORMERS_OFFLINE", None)
#                 else:
#                     os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
#             sys.argv = original
#     finally:
#         _destroy_dist_if_needed(enabled)


# if __name__ == "__main__":
#     runtime, remain = parse_runtime_args()
#     if runtime.check_only:
#         print(
#             "[CHECK_ONLY] "
#             f"distributed={runtime.distributed} auto_device_map={runtime.auto_device_map} "
#             f"gpus_per_process={runtime.gpus_per_process} passthrough_args={' '.join(remain)}"
#         )
#         sys.exit(0)
#     HF_RUNTIME.update(vars(runtime))
#     _run_single_train(runtime, remain)
























# 下面是适配fsdp的情况：
import sys
import argparse
import os
import importlib
import importlib.util
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

WAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WAN_ROOT))

def _load_base_ti2v_module():
    """
    仅加载明确允许的基础 TI2V 训练模块。
    默认使用 train_metaquery_wan（无 animate-like/hard-lock 专用逻辑）。
    可选通过 WAN_BASE_TI2V_MODULE 显式指定允许模块：
      - train_metaquery_wan
      - train_metaquery_wan_animate_like_v2
    WAN_BASE_TI2V_FILE 仅作为同名模块的文件路径兜底。
    """
    default_name = "train_metaquery_wan"
    allowed_names = {"train_metaquery_wan", "train_metaquery_wan_animate_like_v2"}
    prefer = (os.environ.get("WAN_BASE_TI2V_MODULE", "") or "").strip() or default_name
    if prefer not in allowed_names:
        raise RuntimeError(
            f"仅允许模块 {sorted(allowed_names)}，当前 WAN_BASE_TI2V_MODULE={prefer}"
        )

    last_err = None
    try:
        mod = importlib.import_module(prefer)
        print(f"[IMPORT] base_ti2v_module={prefer}")
        return mod
    except Exception as e:
        last_err = e

    env_file = (os.environ.get("WAN_BASE_TI2V_FILE", "") or "").strip()
    if env_file:
        path = Path(env_file).expanduser().resolve()
        if path.exists():
            try:
                spec = importlib.util.spec_from_file_location("wan_base_ti2v", str(path))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"无法构建导入规格: {path}")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod_name = str(getattr(mod, "__name__", ""))
                if prefer not in mod_name and path.stem != prefer:
                    raise RuntimeError(
                        f"WAN_BASE_TI2V_FILE 必须对应 {prefer}，当前文件={path}"
                    )
                print(f"[IMPORT] base_ti2v_file={path}")
                return mod
            except Exception as e:
                last_err = e

    raise RuntimeError(
        "无法加载基础 TI2V 训练模块。"
        f"目标模块={prefer}，WAN_BASE_TI2V_FILE={env_file or '<unset>'}，last_err={last_err}"
    )

base_ti2v = _load_base_ti2v_module()
if not hasattr(base_ti2v, "DistributedSampler"):
    base_ti2v.DistributedSampler = torch.utils.data.distributed.DistributedSampler
    print("[PATCH] Injected torch.utils.data.distributed.DistributedSampler into base_ti2v")
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


class HFWanDatasetAdapter(WanVideoDataset if WanVideoDataset is not None else object):
    def __init__(
        self,
        frame_num=41,
        max_area=480 * 832,
        null_caption_prob=0.1,
        null_image_prob=0.1,
        max_caption_tokens=512,
        caption_tokenizer_path="google/umt5-xxl",
        min_duration_sec=0.5,
        max_duration_sec=20.0,
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
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--auto_device_map", action="store_true")
    parser.add_argument("--gpus_per_process", type=int, default=1)
    parser.add_argument("--check_only", action="store_true")
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
            # 默认绑定 encoder 卡，兼容当前 DDP(mq_encoder) 路径
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
        timeout_sec = max(60, int(os.environ.get("WAN_DIST_TIMEOUT_SEC", "3600")))
        pg_timeout = timedelta(seconds=timeout_sec)
        if process_device is not None:
            torch.cuda.set_device(process_device)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if process_device is not None and torch.cuda.is_available():
            try:
                dist.init_process_group(
                    backend=backend,
                    device_id=torch.device(f"cuda:{process_device}"),
                    timeout=pg_timeout,
                )
            except TypeError:
                dist.init_process_group(backend=backend, timeout=pg_timeout)
            # 触发一次轻量通信，让 PG 记录本 rank 的设备，避免后续 barrier 警告。
            _warm = torch.zeros(1, device=torch.device(f"cuda:{process_device}"))
            dist.all_reduce(_warm)
        else:
            dist.init_process_group(backend=backend, timeout=pg_timeout)
    if enabled:
        print(
            f"[DIST] enabled=True world_size={world_size} rank={rank} local_rank={local_rank} "
            f"process_device={process_device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
            f"timeout_sec={os.environ.get('WAN_DIST_TIMEOUT_SEC', '3600')}"
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
    if not ("--dit_device" in argv and "--encoder_device" in argv) and runtime.auto_device_map:
        if int(runtime.gpus_per_process) <= 1:
            dit = local_rank
            enc = local_rank
        else:
            start = local_rank * max(1, runtime.gpus_per_process)
            dit = start
            enc = start + 1
        argv = _set_or_append(argv, "--dit_device", dit)
        argv = _set_or_append(argv, "--encoder_device", enc)
    # 统一强制 MQ-only DiT 条件注入，避免调用方遗漏参数。
    argv = _set_or_append(argv, "--dit_condition_mode", "mq_only")
    return argv


def _patch_trainer_for_ddp(trainer, enabled, rank):
    if not enabled:
        return trainer
    enc_device = int(trainer.args.encoder_device)
    bucket_cap_mb = int(os.environ.get("WAN_DDP_BUCKET_CAP_MB", "25"))
    grad_bucket_view = os.environ.get("WAN_DDP_GRAD_BUCKET_VIEW", "1") == "1"
    ddp_kwargs = dict(
        device_ids=[enc_device],
        output_device=enc_device,
        find_unused_parameters=False,
        broadcast_buffers=False,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=grad_bucket_view,
    )
    try:
        trainer.mq_encoder = DDP(trainer.mq_encoder, **ddp_kwargs)
    except TypeError:
        ddp_kwargs.pop("gradient_as_bucket_view", None)
        trainer.mq_encoder = DDP(trainer.mq_encoder, **ddp_kwargs)
    print(
        f"[DIST][DDP] encoder_device={enc_device} "
        f"bucket_cap_mb={bucket_cap_mb} gradient_as_bucket_view={grad_bucket_view}"
    )
    # 不再在这里强制 rank0-only 保存：
    # train_metaquery_wan.py 的 _save_checkpoint 内部已自行处理主从写盘，
    # 且 FSDP 的 portable full-state 导出需要所有 rank 参与 collective。
    if hasattr(trainer, "post_wrap_ddp_audit"):
        trainer.post_wrap_ddp_audit()
    return trainer


def _run_single_train(runtime, remain):
    enabled, _, rank, local_rank, process_device = _init_dist_if_needed(runtime)
    try:
        base_ti2v.WanDatasetClass = HFWanDatasetAdapter
        resolved_argv = _resolve_device_args(runtime, remain, local_rank)
        print(f"[PIPELINE] resolved_argv={' '.join(map(str, resolved_argv))}")
        original = list(sys.argv)
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        if enabled and rank != 0:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            sys.argv = [sys.argv[0]] + resolved_argv
            args = base_ti2v.parse_args()
            args.seed = int(args.seed) + rank
            print(
                f"[TRAIN] rank={rank} dit_device={args.dit_device} encoder_device={args.encoder_device} "
                f"num_train_steps={args.num_train_steps} output_dir={args.output_dir}"
            )
            trainer = base_ti2v.MetaQueryWanTrainer(args)
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


if __name__ == "__main__":
    runtime, remain = parse_runtime_args()
    if runtime.check_only:
        print(
            "[CHECK_ONLY] "
            f"distributed={runtime.distributed} auto_device_map={runtime.auto_device_map} "
            f"gpus_per_process={runtime.gpus_per_process} passthrough_args={' '.join(remain)}"
        )
        sys.exit(0)
    HF_RUNTIME.update(vars(runtime))
    _run_single_train(runtime, remain)
