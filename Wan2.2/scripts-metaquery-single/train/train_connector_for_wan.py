# import hashlib
# import math
# import os
# import pickle
# import random
# import time
# import zipfile
# import csv
# from collections import defaultdict
# from pathlib import Path

# import numpy as np
# import requests
# import torch
# import torch.nn as nn
# from PIL import Image
# from datasets import load_dataset
# from huggingface_hub import snapshot_download
# from requests.adapters import HTTPAdapter
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer


# DEFAULT_STAGE_DATASET = {
#     "stage1": "nkp37/OpenVid-1M",
#     "stage2": "BestWishYsh/OpenS2V-5M",
# }

# DEFAULT_STAGE_TOTAL = {
#     "nkp37/OpenVid-1M": 1_000_000,
#     "BestWishYsh/OpenS2V-5M": 5_000_000,
# }


# class MetaQueryEncoderForWan(nn.Module):
#     WAN_TEXT_DIM = 4096

#     def __init__(
#         self,
#         qwen3vl_model_id: str,
#         num_metaqueries: int = 256,
#         connector_num_hidden_layers: int = 24,
#         gradient_checkpointing: bool = False,
#         dtype: torch.dtype = torch.bfloat16,
#         device: str = "cuda",
#     ):
#         super().__init__()
#         self.num_metaqueries = num_metaqueries
#         self.wan_text_dim = self.WAN_TEXT_DIM
#         self.dtype = dtype
#         self.device = torch.device(device)
#         self._printed_forward_stats = False

#         from diffusers.models.normalization import RMSNorm
#         from models.model import MLLMInContext, MLLMInContextConfig
#         from models.transformer_encoder import Qwen2Encoder
#         from transformers import Qwen2Config

#         # 关键点：
#         # 1) diffusion_model_id 设为 "none" -> 不会加载 Sana/SD 的扩散骨干；
#         # 2) connector_out_dim_override 直接指定为 4096，与 Wan text_dim 对齐。
#         config = MLLMInContextConfig(
#             mllm_id=qwen3vl_model_id,
#             diffusion_model_id="none",
#             connector_out_dim_override=self.wan_text_dim,
#             num_metaqueries=num_metaqueries,
#             _gradient_checkpointing=gradient_checkpointing,
#             connector_num_hidden_layers=connector_num_hidden_layers,
#         )

#         mllm_model = MLLMInContext(config).to(device=self.device, dtype=dtype)
#         self.mllm_model = mllm_model
#         self.tokenizer = mllm_model.get_tokenizer()
#         self.tokenize = mllm_model.get_tokenize_fn()
#         # 开启梯度检查点时关闭 KV cache，减少显存并避免反复告警。
#         try:
#             if hasattr(self.mllm_model.mllm_backbone, "config"):
#                 self.mllm_model.mllm_backbone.config.use_cache = False
#             if hasattr(self.mllm_model.mllm_backbone, "generation_config"):
#                 self.mllm_model.mllm_backbone.generation_config.use_cache = False
#         except Exception:
#             pass
#         print(
#             f"[MetaQueryEncoderForWan] mllm_type={self.mllm_model.mllm_type} "
#             f"transformer_loaded={self.mllm_model.transformer is not None}"
#         )

#         mllm_hidden_size = mllm_model.mllm_hidden_size
#         print(
#             f"[MetaQueryEncoderForWan] mllm_hidden={mllm_hidden_size} "
#             f"target_wan_text_dim={self.wan_text_dim}"
#         )
#         encoder = Qwen2Encoder(
#             Qwen2Config(
#                 hidden_size=mllm_hidden_size,
#                 intermediate_size=mllm_hidden_size * 4,
#                 num_hidden_layers=connector_num_hidden_layers,
#                 num_attention_heads=mllm_hidden_size // 64,
#                 num_key_value_heads=mllm_hidden_size // 64,
#                 initializer_range=0.014,
#                 use_cache=False,
#                 rope=True,
#                 qk_norm=True,
#             )
#         )
#         # 兼容自定义 Qwen2Encoder 的梯度检查点调用。
#         if hasattr(encoder, "gradient_checkpointing"):
#             encoder.gradient_checkpointing = bool(gradient_checkpointing)
#         if gradient_checkpointing and not hasattr(encoder, "_gradient_checkpointing_func"):
#             encoder._gradient_checkpointing_func = (
#                 lambda func, *gc_args: torch.utils.checkpoint.checkpoint(
#                     func, *gc_args, use_reentrant=False
#                 )
#             )
#         norm = RMSNorm(self.wan_text_dim, eps=1e-5, elementwise_affine=True)
#         with torch.no_grad():
#             norm.weight.fill_(math.sqrt(5.5))
#         new_connector = nn.Sequential(
#             encoder,
#             nn.Linear(mllm_hidden_size, self.wan_text_dim),
#             nn.GELU(approximate="tanh"),
#             nn.Linear(self.wan_text_dim, self.wan_text_dim),
#             norm,
#         ).to(device=self.device, dtype=dtype)
#         self.mllm_model.connector = new_connector
#         self.mllm_model.connector_out_dim = self.wan_text_dim
#         print(
#             f"[MetaQueryEncoderForWan] connector_out={self.mllm_model.connector_out_dim} "
#             f"num_metaqueries={self.num_metaqueries}"
#         )

#         self.mllm_model.mllm_backbone.requires_grad_(False)
#         self.mllm_model.connector.requires_grad_(True)
#         self.mllm_model.mllm_backbone.get_input_embeddings().requires_grad_(True)

#         if hasattr(self.mllm_model, "transformer"):
#             del self.mllm_model.transformer
#             self.mllm_model.transformer = None
#         torch.cuda.empty_cache()

#     def get_trainable_params(self):
#         return [p for p in self.parameters() if p.requires_grad]

#     def forward(self, captions, input_images=None):
#         if input_images is not None:
#             input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
#                 self.tokenizer, captions, input_images
#             )
#             input_ids = input_ids.to(self.device)
#             attention_mask = attention_mask.to(self.device)
#             if pixel_values is not None:
#                 pixel_values = pixel_values.to(self.device, self.dtype)
#                 if self.mllm_model.mllm_type in ("qwenvl", "qwen3vl"):
#                     pixel_values = pixel_values.squeeze(0)
#             if image_sizes is not None:
#                 image_sizes = image_sizes.to(self.device)
#         else:
#             input_ids, attention_mask = self.tokenize(self.tokenizer, captions)
#             input_ids = input_ids.to(self.device)
#             attention_mask = attention_mask.to(self.device)
#             pixel_values = None
#             image_sizes = None

#         mq_features, _ = self.mllm_model.encode_condition(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             pixel_values=pixel_values,
#             image_sizes=image_sizes,
#         )
#         if not self._printed_forward_stats:
#             print(
#                 f"[MetaQueryEncoderForWan] forward_once "
#                 f"input_ids={tuple(input_ids.shape)} "
#                 f"attention_mask={tuple(attention_mask.shape)} "
#                 f"mq_features={tuple(mq_features.shape)} "
#                 f"dtype={mq_features.dtype}"
#             )
#             self._printed_forward_stats = True
#         return mq_features


# def resolve_hf_stage_config(stage: str, dataset_name: str | None, subset_ratio: float):
#     stage_name = stage.lower()
#     ds = dataset_name or DEFAULT_STAGE_DATASET.get(stage_name, DEFAULT_STAGE_DATASET["stage1"])
#     total_hint = DEFAULT_STAGE_TOTAL.get(ds, None)
#     subset_size = int(total_hint * subset_ratio) if total_hint is not None else None
#     return ds, subset_size, total_hint


# class WanVideoDataset(Dataset):
#     def __init__(
#         self,
#         frame_num: int = 81,
#         max_area: int = 720 * 1280,
#         null_caption_prob: float = 0.1,
#         null_image_prob: float = 0.1,
#         max_caption_tokens: int = 512,
#         caption_tokenizer_path: str = "google/umt5-xxl",
#         min_duration_sec: float = 0.5,
#         max_duration_sec: float = 20.0,
#         hf_stage: str = "stage1",
#         hf_dataset_name: str | None = None,
#         hf_split: str = "train",
#         hf_subset_ratio: float = 0.01,
#         hf_subset_size: int | None = None,
#         hf_scan_factor: int = 30,
#         hf_subset_cache_dir: str | None = None,
#         hf_subset_use_cache: bool = True,
#         hf_cache_dir: str | None = None,
#         hf_streaming: bool = True,
#         hf_shuffle_buffer: int = 10000,
#         seed: int = 42,
#         local_video_cache_dir: str | None = None,
#         local_openvid_video_root: str | None = None,
#         local_openvid_csv_path: str | None = None,
#         local_openvid_limit: int | None = None,
#         local_openvid_hd_video_root: str | None = None,
#         local_openvid_hd_csv_path: str | None = None,
#         local_openvid_hd_limit: int | None = None,
#     ):
#         self.frame_num = frame_num
#         self.max_area = max_area
#         self.null_caption_prob = null_caption_prob
#         self.null_image_prob = null_image_prob
#         self.max_caption_tokens = max_caption_tokens
#         self.min_duration_sec = min_duration_sec
#         self.max_duration_sec = max_duration_sec
#         self.hf_split = hf_split
#         self.hf_cache_dir = hf_cache_dir
#         self.hf_streaming = hf_streaming
#         self.hf_shuffle_buffer = hf_shuffle_buffer
#         self.seed = seed
#         self.scan_factor = max(5, hf_scan_factor)
#         self._printed_sample_info = False
#         self._last_good_sample = None
#         self._failure_stats = defaultdict(int)
#         self._warned_hf_token = False
#         self._subset_scanned = 0
#         self._subset_accepted = 0
#         self._subset_rejected_stats = defaultdict(int)
#         self.quick_fail = os.environ.get("WAN_DATA_QUICK_FAIL", "1").strip().lower() not in (
#             "0",
#             "false",
#             "off",
#         )
#         self.max_trials_cap = max(10, int(os.environ.get("WAN_DATA_MAX_TRIALS", "400")))
#         self.trial_log_interval = max(
#             1, int(os.environ.get("WAN_DATA_TRIAL_LOG_INTERVAL", "100"))
#         )
#         self.url_fallback_limit = max(
#             1, int(os.environ.get("WAN_DATA_PATH_URL_FALLBACK_LIMIT", "2"))
#         )
#         self.http_retry_total = max(0, int(os.environ.get("WAN_DATA_HTTP_RETRY_TOTAL", "1")))
#         self.http_timeout_sec = max(3, int(os.environ.get("WAN_DATA_HTTP_TIMEOUT_SEC", "12")))
#         self.lock_timeout_sec = max(3, int(os.environ.get("WAN_DATA_LOCK_TIMEOUT_SEC", "20")))
#         self.preclean_enabled = os.environ.get("WAN_DATA_PRECLEAN", "1").strip().lower() not in (
#             "0",
#             "false",
#             "off",
#         )
#         self.preclean_log_interval = max(
#             1, int(os.environ.get("WAN_DATA_PRECLEAN_LOG_INTERVAL", "200"))
#         )
#         self.preclean_scan_multiplier = max(
#             1, int(os.environ.get("WAN_DATA_PRECLEAN_SCAN_MULTIPLIER", "1"))
#         )
#         self.preclean_scan_cap = max(
#             1000, int(os.environ.get("WAN_DATA_PRECLEAN_SCAN_CAP", "200000"))
#         )
#         self.preclean_zero_accept_abort_scan = max(
#             1000, int(os.environ.get("WAN_DATA_PRECLEAN_ZERO_ACCEPT_ABORT_SCAN", "20000"))
#         )
#         self.local_overfit_debug = os.environ.get("WAN_LOCAL_OVERFIT_DEBUG", "0").strip().lower() in (
#             "1",
#             "true",
#             "yes",
#             "on",
#         )
#         self.hf_subset_ratio = hf_subset_ratio
#         self.hf_subset_size = hf_subset_size
#         self.hf_subset_use_cache = hf_subset_use_cache
#         self.hf_subset_cache_dir = Path(
#             hf_subset_cache_dir or (Path(hf_cache_dir or ".hf_cache") / "subset_cache")
#         )
#         self.hf_subset_cache_dir.mkdir(parents=True, exist_ok=True)
#         self.local_video_cache_dir = Path(
#             local_video_cache_dir or (Path(hf_cache_dir or ".hf_cache") / "video_cache")
#         )
#         self.local_video_cache_dir.mkdir(parents=True, exist_ok=True)
#         if local_openvid_video_root is None:
#             env_video_root = os.environ.get("OPENVID_LOCAL_VIDEO_ROOT", "").strip()
#             local_openvid_video_root = env_video_root or None
#         if local_openvid_csv_path is None:
#             env_csv_path = os.environ.get("OPENVID_LOCAL_CSV_PATH", "").strip()
#             local_openvid_csv_path = env_csv_path or None
#         if local_openvid_limit is None:
#             env_limit = os.environ.get("OPENVID_LOCAL_LIMIT", "").strip()
#             if env_limit:
#                 try:
#                     local_openvid_limit = int(env_limit)
#                 except Exception:
#                     local_openvid_limit = None
#         if local_openvid_hd_video_root is None:
#             env_hd_video_root = os.environ.get("OPENVID_HD_LOCAL_VIDEO_ROOT", "").strip()
#             local_openvid_hd_video_root = env_hd_video_root or None
#         if local_openvid_hd_csv_path is None:
#             env_hd_csv_path = os.environ.get("OPENVID_HD_LOCAL_CSV_PATH", "").strip()
#             local_openvid_hd_csv_path = env_hd_csv_path or None
#         if local_openvid_hd_limit is None:
#             env_hd_limit = os.environ.get("OPENVID_HD_LOCAL_LIMIT", "").strip()
#             if env_hd_limit:
#                 try:
#                     local_openvid_hd_limit = int(env_hd_limit)
#                 except Exception:
#                     local_openvid_hd_limit = None
#         env_total_limit = os.environ.get("OPENVID_LOCAL_TOTAL_LIMIT", "").strip()
#         local_openvid_total_limit = None
#         if env_total_limit:
#             try:
#                 local_openvid_total_limit = int(env_total_limit)
#             except Exception:
#                 local_openvid_total_limit = None

#         def _to_path(v):
#             return Path(v).expanduser().resolve() if v else None

#         def _to_limit(v):
#             if v is None:
#                 return None
#             try:
#                 iv = int(v)
#             except Exception:
#                 return None
#             return iv if iv > 0 else None

#         self.local_openvid_video_root = _to_path(local_openvid_video_root)
#         self.local_openvid_csv_path = _to_path(local_openvid_csv_path)
#         self.local_openvid_limit = _to_limit(local_openvid_limit)
#         self.local_openvid_hd_video_root = _to_path(local_openvid_hd_video_root)
#         self.local_openvid_hd_csv_path = _to_path(local_openvid_hd_csv_path)
#         self.local_openvid_hd_limit = _to_limit(local_openvid_hd_limit)
#         self.local_openvid_total_limit = _to_limit(local_openvid_total_limit)

#         self.local_openvid_sources = []
#         if self.local_openvid_video_root is not None and self.local_openvid_csv_path is not None:
#             self.local_openvid_sources.append(
#                 {
#                     "name": "openvid",
#                     "video_root": self.local_openvid_video_root,
#                     "csv_path": self.local_openvid_csv_path,
#                     "limit": self.local_openvid_limit,
#                 }
#             )
#         elif self.local_openvid_video_root is not None or self.local_openvid_csv_path is not None:
#             print(
#                 "[WanVideoDataset] warning: openvid 普通源参数不完整，"
#                 "需同时提供 local_openvid_video_root 与 local_openvid_csv_path，已忽略该源"
#             )

#         if self.local_openvid_hd_video_root is not None and self.local_openvid_hd_csv_path is not None:
#             self.local_openvid_sources.append(
#                 {
#                     "name": "openvid_hd",
#                     "video_root": self.local_openvid_hd_video_root,
#                     "csv_path": self.local_openvid_hd_csv_path,
#                     "limit": self.local_openvid_hd_limit,
#                 }
#             )
#         elif self.local_openvid_hd_video_root is not None or self.local_openvid_hd_csv_path is not None:
#             print(
#                 "[WanVideoDataset] warning: openvid HD 源参数不完整，"
#                 "需同时提供 local_openvid_hd_video_root 与 local_openvid_hd_csv_path，已忽略该源"
#             )

#         self.local_openvid_enabled = len(self.local_openvid_sources) > 0
#         tokenizer_local_only = os.environ.get("WAN_TOKENIZER_LOCAL_ONLY", "0").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 caption_tokenizer_path,
#                 local_files_only=tokenizer_local_only,
#             )
#         except Exception as e:
#             if tokenizer_local_only:
#                 raise RuntimeError(
#                     f"[WanVideoDataset] 本地加载 tokenizer 失败: {caption_tokenizer_path}. "
#                     "请确认该路径可读，或关闭 WAN_TOKENIZER_LOCAL_ONLY。"
#                 ) from e
#             # 兜底：网络异常时自动尝试仅本地缓存，避免 Determined 环境因外网抖动直接失败
#             try:
#                 self.tokenizer = AutoTokenizer.from_pretrained(
#                     caption_tokenizer_path,
#                     local_files_only=True,
#                 )
#                 print(
#                     f"[WanVideoDataset] tokenizer remote load failed, fallback local cache only: "
#                     f"path={caption_tokenizer_path} err={e}"
#                 )
#             except Exception as e2:
#                 raise RuntimeError(
#                     f"[WanVideoDataset] tokenizer 加载失败: {caption_tokenizer_path}. "
#                     "网络访问异常且本地缓存不可用。建议在 .sh 中设置 CAPTION_TOKENIZER_PATH 为本地目录，"
#                     "并开启 TOKENIZER_LOCAL_ONLY=1。"
#                 ) from e2

#         resolved_ds, stage_subset_size, total_hint = resolve_hf_stage_config(
#             stage=hf_stage,
#             dataset_name=hf_dataset_name,
#             subset_ratio=hf_subset_ratio,
#         )
#         if self.local_openvid_enabled:
#             self.dataset_name = "local/OpenVid-1M+HD"
#             self.total_hint = None
#             self.is_openvid = True
#         else:
#             self.dataset_name = resolved_ds
#             self.total_hint = total_hint
#             self.is_openvid = self.dataset_name.lower() == "nkp37/openvid-1m"
#         self.openvid_record_streaming = os.environ.get("OPENVID_RECORD_STREAMING", "1").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         openvid_root = os.environ.get("OPENVID_VIDEO_ROOT", "").strip()
#         if self.local_openvid_enabled:
#             openvid_root = str(self.local_openvid_sources[0]["video_root"])
#         self.openvid_video_root = Path(openvid_root) if openvid_root else None
#         openvid_archive_root = os.environ.get("OPENVID_ARCHIVE_ROOT", "").strip()
#         if self.local_openvid_enabled:
#             openvid_archive_root = ""
#         self.openvid_archive_root = Path(openvid_archive_root) if openvid_archive_root else None
#         self.openvid_snapshot_download = os.environ.get("OPENVID_SNAPSHOT_DOWNLOAD", "0").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         if self.local_openvid_enabled:
#             self.openvid_snapshot_download = False
#         self.openvid_snapshot_dir = Path(
#             os.environ.get("OPENVID_SNAPSHOT_DIR", str(Path(hf_cache_dir or ".hf_cache") / "openvid_repo"))
#         )
#         self.openvid_snapshot_dir.mkdir(parents=True, exist_ok=True)
#         self.openvid_snapshot_patterns = [
#             p.strip()
#             for p in os.environ.get(
#                 "OPENVID_SNAPSHOT_PATTERNS",
#                 "Openvid_part*.zip,Openvid_part*.part*,OpenVidHD.csv,data/*",
#             ).split(",")
#             if p.strip()
#         ]
#         self.openvid_allow_http_guess = os.environ.get("OPENVID_ALLOW_HTTP_GUESS", "0").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         self._openvid_archive_index_built = False
#         self._openvid_archive_map = {}
#         self._openvid_archive_miss = set()
#         self._openvid_autofallback_done = False
#         self._openvid_archive_max_scan = max(
#             0, int(os.environ.get("OPENVID_ARCHIVE_MAX_SCAN_FILES", "0"))
#         )
#         self.openvid_auto_join_parts = os.environ.get("OPENVID_AUTO_JOIN_PARTS", "1").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         self.openvid_joined_archive_dir = Path(
#             os.environ.get("OPENVID_JOINED_ARCHIVE_DIR", str(self.local_video_cache_dir / "openvid_joined"))
#         )
#         self.openvid_joined_archive_dir.mkdir(parents=True, exist_ok=True)
#         self.openvid_extracted_cache_dir = self.local_video_cache_dir / "openvid_extracted"
#         self.openvid_extracted_cache_dir.mkdir(parents=True, exist_ok=True)
#         self._subset_streaming = (
#             (not self.local_openvid_enabled)
#             and (self.hf_streaming or (self.is_openvid and self.openvid_record_streaming))
#         )
#         if (
#             self.is_openvid
#             and (not self.local_openvid_enabled)
#             and self.openvid_record_streaming
#             and self.openvid_snapshot_download
#             and self.openvid_archive_root is None
#         ):
#             print("[WanVideoDataset] openvid_record_streaming=1, skip snapshot_download")
#             self.openvid_snapshot_download = False
#         if (
#             self.is_openvid
#             and (not self.local_openvid_enabled)
#             and self.openvid_archive_root is None
#             and self.openvid_snapshot_download
#         ):
#             self.openvid_archive_root = self._prepare_openvid_archive_root_from_hf()
#         if self.local_openvid_enabled:
#             per_source_limits = [src["limit"] for src in self.local_openvid_sources]
#             if per_source_limits and all(v is not None for v in per_source_limits):
#                 self.target_subset_size = sum(int(v) for v in per_source_limits)
#             else:
#                 self.target_subset_size = 0
#         else:
#             self.target_subset_size = hf_subset_size or stage_subset_size or 10000
#         self.samples = self._build_subset()
#         if len(self.samples) == 0:
#             raise RuntimeError(f"数据集可用样本为0: {self.dataset_name}")
#         print(
#             f"[WanVideoDataset] dataset={self.dataset_name} split={self.hf_split} "
#             f"target={self.target_subset_size} loaded={len(self.samples)} "
#             f"scanned={self._subset_scanned} scan_factor={self.scan_factor} "
#             f"streaming={self.hf_streaming} ratio={self.hf_subset_ratio} "
#             f"subset_size_override={self.hf_subset_size} cache_dir={self.hf_cache_dir}"
#         )
#         if self.is_openvid:
#             print(
#                 f"[WanVideoDataset] openvid_mode=1 "
#                 f"openvid_video_root={str(self.openvid_video_root) if self.openvid_video_root else 'unset'} "
#                 f"openvid_archive_root={str(self.openvid_archive_root) if self.openvid_archive_root else 'unset'} "
#                 f"snapshot_download={self.openvid_snapshot_download} "
#                 f"record_streaming={self.openvid_record_streaming} "
#                 f"auto_join_parts={self.openvid_auto_join_parts} "
#                 f"allow_http_guess={self.openvid_allow_http_guess}"
#             )
#         if self.local_openvid_enabled:
#             for src in self.local_openvid_sources:
#                 print(
#                     f"[WanVideoDataset] local_openvid_source "
#                     f"name={src['name']} video_root={src['video_root']} "
#                     f"csv_path={src['csv_path']} "
#                     f"limit={src['limit'] if src['limit'] else 'all'}"
#                 )

#     def _subset_cache_path(self):
#         if self.local_openvid_enabled:
#             source_parts = []
#             for src in self.local_openvid_sources:
#                 source_parts.append(
#                     f"{src['name']}:{src['video_root']}:{src['csv_path']}:limit={src['limit']}"
#                 )
#             key = (
#                 f"local_openvid_multi|{'|'.join(source_parts)}|"
#                 f"f={self.frame_num}|min={self.min_duration_sec}|"
#                 f"max={self.max_duration_sec}|tok={self.max_caption_tokens}|seed={self.seed}|"
#                 f"preclean={self.preclean_enabled}"
#             )
#             name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pkl"
#             return self.hf_subset_cache_dir / name
#         key = (
#             f"{self.dataset_name}|{self.hf_split}|{self.target_subset_size}|"
#             f"{self.scan_factor}|{self.seed}|{self.hf_streaming}|"
#             f"preclean={self.preclean_enabled}|f={self.frame_num}|min={self.min_duration_sec}|"
#             f"max={self.max_duration_sec}|tok={self.max_caption_tokens}"
#         )
#         name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pkl"
#         return self.hf_subset_cache_dir / name

#     def _build_subset(self):
#         cache_path = self._subset_cache_path()
#         if self.local_openvid_enabled:
#             return self._build_local_openvid_subset(cache_path)
#         if self.hf_subset_use_cache and cache_path.exists():
#             try:
#                 with open(cache_path, "rb") as f:
#                     payload = pickle.load(f)
#                 self._subset_scanned = int(payload.get("scanned", 0))
#                 samples = payload.get("samples", [])
#                 self._subset_accepted = len(samples)
#                 rejected_stats = payload.get("rejected_stats", {})
#                 self._subset_rejected_stats = defaultdict(
#                     int, {str(k): int(v) for k, v in rejected_stats.items()}
#                 )
#                 print(f"[WanVideoDataset] subset_cache_hit={cache_path} loaded={len(samples)}")
#                 if len(samples) > 0:
#                     return samples
#             except Exception:
#                 pass
#         ds = load_dataset(
#             self.dataset_name,
#             split=self.hf_split,
#             streaming=self._subset_streaming,
#             cache_dir=self.hf_cache_dir,
#         )
#         if self.hf_shuffle_buffer > 0 and hasattr(ds, "shuffle"):
#             try:
#                 if self._subset_streaming:
#                     ds = ds.shuffle(seed=self.seed, buffer_size=self.hf_shuffle_buffer)
#                 else:
#                     ds = ds.shuffle(seed=self.seed)
#             except TypeError:
#                 ds = ds.shuffle(seed=self.seed)

#         samples = []
#         rejected_stats = defaultdict(int)
#         scan_multiplier = self.preclean_scan_multiplier if self.preclean_enabled else 1
#         max_scan = self.target_subset_size * self.scan_factor * scan_multiplier
#         max_scan = min(max_scan, self.preclean_scan_cap)
#         scanned = 0
#         for row in ds:
#             scanned += 1
#             parsed = self._extract_row(row)
#             if parsed is not None:
#                 if self.preclean_enabled:
#                     ok, reject_reason = self._preclean_sample(parsed)
#                     if ok:
#                         samples.append(parsed)
#                     else:
#                         rejected_stats[reject_reason] += 1
#                 else:
#                     samples.append(parsed)
#             else:
#                 rejected_stats["extract_failed"] += 1
#             if scanned % self.preclean_log_interval == 0:
#                 rejected = sum(rejected_stats.values())
#                 top_reject = ", ".join(
#                     f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:4]
#                 ) or "none"
#                 top_io = ", ".join(
#                     f"{k}={v}" for k, v in sorted(self._failure_stats.items(), key=lambda x: -x[1])[:4]
#                 ) or "none"
#                 print(
#                     f"[WanVideoDataset] preclean_progress scanned={scanned} "
#                     f"accepted={len(samples)} rejected={rejected} top=[{top_reject}] io=[{top_io}] "
#                     f"target={self.target_subset_size} max_scan={max_scan}"
#                 )
#             if (
#                 self.preclean_enabled
#                 and len(samples) == 0
#                 and scanned >= self.preclean_zero_accept_abort_scan
#             ):
#                 print(
#                     f"[WanVideoDataset] preclean_early_abort scanned={scanned} accepted=0 "
#                     f"reason=zero_accepted_until_threshold({self.preclean_zero_accept_abort_scan})"
#                 )
#                 break
#             if len(samples) >= self.target_subset_size:
#                 break
#             if scanned >= max_scan:
#                 break
#         self._subset_scanned = scanned
#         self._subset_accepted = len(samples)
#         self._subset_rejected_stats = rejected_stats
#         rejected = sum(rejected_stats.values())
#         top_reject = ", ".join(
#             f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:8]
#         ) or "none"
#         top_io = ", ".join(
#             f"{k}={v}" for k, v in sorted(self._failure_stats.items(), key=lambda x: -x[1])[:8]
#         ) or "none"
#         print(
#             f"[WanVideoDataset] preclean_done scanned={scanned} accepted={len(samples)} "
#             f"rejected={rejected} top=[{top_reject}] io=[{top_io}] "
#             f"preclean={self.preclean_enabled} target={self.target_subset_size} max_scan={max_scan}"
#         )
#         if self.hf_subset_use_cache:
#             try:
#                 with open(cache_path, "wb") as f:
#                     pickle.dump(
#                         {
#                             "scanned": scanned,
#                             "samples": samples,
#                             "rejected_stats": dict(rejected_stats),
#                         },
#                         f,
#                     )
#                 print(f"[WanVideoDataset] subset_cache_write={cache_path} saved={len(samples)}")
#             except Exception:
#                 pass
#         if len(samples) == 0:
#             if (
#                 self.is_openvid
#                 and (not self._openvid_autofallback_done)
#                 and self.openvid_archive_root is None
#                 and self._failure_stats.get("openvid_filename_without_local_root", 0) > 0
#                 and self.openvid_snapshot_dir is not None
#             ):
#                 self._openvid_autofallback_done = True
#                 print(
#                     "[WanVideoDataset] openvid_record_streaming_detected_filename_only=1, "
#                     "auto_fallback_to_snapshot_download=1"
#                 )
#                 self.openvid_snapshot_download = True
#                 self.openvid_archive_root = self._prepare_openvid_archive_root_from_hf()
#                 if self.openvid_archive_root is not None:
#                     self._openvid_archive_index_built = False
#                     self._openvid_archive_map = {}
#                     self._openvid_archive_miss = set()
#                     return self._build_subset()
#             raise RuntimeError(
#                 f"预清洗后可用样本为0: dataset={self.dataset_name} scanned={scanned} "
#                 f"rejected={rejected} top=[{top_reject}] io=[{top_io}] "
#                 f"请检查OPENVID_SNAPSHOT_DIR中是否含Openvid_part*.zip/part*并开启OPENVID_AUTO_JOIN_PARTS=1"
#             )
#         return samples

#     @staticmethod
#     def _normalize_local_openvid_key(value):
#         if value is None:
#             return ""
#         out = str(value).strip().replace("\\", "/")
#         while out.startswith("./"):
#             out = out[2:]
#         out = out.lstrip("/")
#         return out.lower()

#     def _iter_local_openvid_files(self, video_root: Path):
#         if video_root is None:
#             return []
#         if not video_root.exists():
#             raise RuntimeError(
#                 f"local_openvid_video_root 不存在: {video_root}"
#             )
#         exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
#         files = []
#         for p in video_root.rglob("*"):
#             if not p.is_file():
#                 continue
#             if p.suffix.lower() in exts:
#                 files.append(p)
#         files.sort()
#         return files

#     def _load_local_openvid_caption_index(self, csv_path: Path, source_name: str):
#         if csv_path is None:
#             raise RuntimeError("local_openvid_csv_path 未设置")
#         if not csv_path.exists():
#             raise RuntimeError(f"local_openvid_csv_path 不存在: {csv_path}")
#         path_to_caption = {}
#         name_to_caption = {}
#         row_count = 0
#         drop_no_video = 0
#         drop_no_caption = 0
#         selected_video_col = None
#         selected_caption_col = None

#         def _keep_longer(mapping, key, caption):
#             old = mapping.get(key, None)
#             if old is None or len(caption) > len(old):
#                 mapping[key] = caption

#         with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
#             reader = csv.DictReader(f)
#             fieldnames = [str(x) for x in (reader.fieldnames or [])]
#             if not fieldnames:
#                 raise RuntimeError(f"CSV 无表头: {csv_path}")
#             lowered = {name.lower(): name for name in fieldnames}
#             for key in ("video", "video_path", "filename", "file", "path"):
#                 if key in lowered:
#                     selected_video_col = lowered[key]
#                     break
#             for key in ("caption", "text", "description", "prompt", "summary"):
#                 if key in lowered:
#                     selected_caption_col = lowered[key]
#                     break
#             if selected_video_col is None or selected_caption_col is None:
#                 raise RuntimeError(
#                     "CSV 缺少 video/caption 列。"
#                     f"当前列: {fieldnames}"
#                 )
#             for row in reader:
#                 row_count += 1
#                 video_val = str(row.get(selected_video_col, "") or "").strip()
#                 caption_val = str(row.get(selected_caption_col, "") or "").strip()
#                 if not video_val:
#                     drop_no_video += 1
#                     continue
#                 if not caption_val:
#                     drop_no_caption += 1
#                     continue
#                 norm_path = self._normalize_local_openvid_key(video_val)
#                 if norm_path:
#                     _keep_longer(path_to_caption, norm_path, caption_val)
#                 basename = self._normalize_local_openvid_key(Path(video_val).name)
#                 if basename:
#                     _keep_longer(name_to_caption, basename, caption_val)
#         print(
#             f"[WanVideoDataset][local_openvid][{source_name}] csv_index_done rows={row_count} "
#             f"video_col={selected_video_col} caption_col={selected_caption_col} "
#             f"path_keys={len(path_to_caption)} name_keys={len(name_to_caption)} "
#             f"drop_no_video={drop_no_video} drop_no_caption={drop_no_caption}"
#         )
#         return path_to_caption, name_to_caption

#     def _lookup_local_openvid_caption(
#         self,
#         video_path: Path,
#         video_root: Path,
#         path_to_caption,
#         name_to_caption,
#     ):
#         rel_path = str(video_path.relative_to(video_root)).replace("\\", "/")
#         rel_key = self._normalize_local_openvid_key(rel_path)
#         cap = path_to_caption.get(rel_key, None)
#         if cap:
#             return cap, rel_path, "rel_path"
#         name_key = self._normalize_local_openvid_key(video_path.name)
#         cap = name_to_caption.get(name_key, None)
#         if cap:
#             return cap, rel_path, "basename"
#         return None, rel_path, "missing"

#     def _build_local_openvid_subset(self, cache_path):
#         if self.hf_subset_use_cache and cache_path.exists():
#             try:
#                 with open(cache_path, "rb") as f:
#                     payload = pickle.load(f)
#                 self._subset_scanned = int(payload.get("scanned", 0))
#                 samples = payload.get("samples", [])
#                 self._subset_accepted = len(samples)
#                 rejected_stats = payload.get("rejected_stats", {})
#                 self._subset_rejected_stats = defaultdict(
#                     int, {str(k): int(v) for k, v in rejected_stats.items()}
#                 )
#                 print(f"[WanVideoDataset] local_openvid_subset_cache_hit={cache_path} loaded={len(samples)}")
#                 if len(samples) > 0:
#                     return samples
#             except Exception:
#                 pass

#         paired = []
#         source_pair_stats = []
#         max_missing_print = max(0, int(os.environ.get("WAN_LOCAL_MISSING_CAPTION_PRINT_MAX", "200")))
#         for src in self.local_openvid_sources:
#             src_name = str(src["name"])
#             src_video_root = src["video_root"]
#             src_csv_path = src["csv_path"]
#             src_limit = src["limit"]
#             video_files = self._iter_local_openvid_files(src_video_root)
#             path_to_caption, name_to_caption = self._load_local_openvid_caption_index(
#                 src_csv_path, src_name
#             )
#             source_paired = []
#             missing_caption = 0
#             match_by_rel = 0
#             match_by_name = 0
#             for vf in video_files:
#                 caption, rel_path, matched_by = self._lookup_local_openvid_caption(
#                     vf, src_video_root, path_to_caption, name_to_caption
#                 )
#                 if not caption:
#                     missing_caption += 1
#                     if missing_caption <= max_missing_print:
#                         print(
#                             f"[WanVideoDataset][local_openvid][{src_name}] "
#                             f"missing_caption_skip video={rel_path}"
#                         )
#                     continue
#                 if matched_by == "rel_path":
#                     match_by_rel += 1
#                 elif matched_by == "basename":
#                     match_by_name += 1
#                 source_paired.append(
#                     {
#                         "caption": caption,
#                         "video_spec": {"kind": "path", "value": str(vf)},
#                         "raw": {
#                             "video": rel_path,
#                             "matched_by": matched_by,
#                             "source_name": src_name,
#                         },
#                     }
#                 )
#             if missing_caption > max_missing_print:
#                 print(
#                     f"[WanVideoDataset][local_openvid][{src_name}] missing_caption_skip_more="
#                     f"{missing_caption - max_missing_print}"
#                 )
#             if src_limit and src_limit > 0 and len(source_paired) > src_limit:
#                 rng = random.Random(self.seed + (abs(hash(src_name)) % 10007))
#                 rng.shuffle(source_paired)
#                 source_paired = source_paired[:src_limit]
#             source_pair_stats.append(
#                 {
#                     "name": src_name,
#                     "local_videos": len(video_files),
#                     "paired": len(source_paired),
#                     "missing_caption": missing_caption,
#                     "matched_by_rel": match_by_rel,
#                     "matched_by_name": match_by_name,
#                     "limit": src_limit,
#                 }
#             )
#             paired.extend(source_paired)

#         rng = random.Random(self.seed)
#         rng.shuffle(paired)
#         if len(paired) == 0:
#             raise RuntimeError(
#                 "本地OpenVid(含HD)配对后样本数为0，请检查视频目录与CSV是否匹配。"
#                 f" sources={[(str(s['video_root']), str(s['csv_path'])) for s in self.local_openvid_sources]}"
#             )
#         for st in source_pair_stats:
#             print(
#                 f"[WanVideoDataset][local_openvid][{st['name']}] pair_done "
#                 f"local_videos={st['local_videos']} paired={st['paired']} "
#                 f"missing_caption={st['missing_caption']} matched_by_rel={st['matched_by_rel']} "
#                 f"matched_by_name={st['matched_by_name']} "
#                 f"target_limit={st['limit'] if st['limit'] else 'all'}"
#             )
#         print(f"[WanVideoDataset][local_openvid] pair_merged_total={len(paired)}")
#         if self.local_openvid_total_limit and len(paired) > self.local_openvid_total_limit:
#             paired = paired[: self.local_openvid_total_limit]
#             print(
#                 f"[WanVideoDataset][local_openvid] pair_merged_capped={len(paired)} "
#                 f"total_limit={self.local_openvid_total_limit}"
#             )

#         samples = []
#         rejected_stats = defaultdict(int)
#         scanned = 0
#         for parsed in paired:
#             scanned += 1
#             raw_video = str(parsed.get("raw", {}).get("video", ""))
#             video_name = Path(raw_video).name if raw_video else ""
#             caption_text = str(parsed.get("caption", "")).replace("\n", " ").strip()
#             if self.local_overfit_debug:
#                 print(f"[WanVideoDataset][local_overfit_debug] video={video_name} caption={caption_text}")

#             if self.preclean_enabled:
#                 ok, reject_reason = self._preclean_sample(parsed)
#                 if ok:
#                     samples.append(parsed)
#                     if self.local_overfit_debug:
#                         print(
#                             f"[WanVideoDataset][local_openvid][KEEP] video={raw_video or video_name} "
#                             f"caption={caption_text}"
#                         )
#                 else:
#                     rejected_stats[reject_reason] += 1
#                     if self.local_overfit_debug:
#                         print(
#                             f"[WanVideoDataset][local_openvid][DROP] video={raw_video or video_name} "
#                             f"reason={reject_reason} caption={caption_text}"
#                         )
#             else:
#                 samples.append(parsed)
#                 if self.local_overfit_debug:
#                     print(
#                         f"[WanVideoDataset][local_openvid][KEEP_NO_PRECLEAN] video={raw_video or video_name} "
#                         f"caption={caption_text}"
#                     )
#             if scanned % self.preclean_log_interval == 0:
#                 rejected = sum(rejected_stats.values())
#                 top_reject = ", ".join(
#                     f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:4]
#                 ) or "none"
#                 print(
#                     f"[WanVideoDataset][local_openvid] preclean_progress scanned={scanned} "
#                     f"accepted={len(samples)} rejected={rejected} top=[{top_reject}]"
#                 )
#         self._subset_scanned = scanned
#         self._subset_accepted = len(samples)
#         self._subset_rejected_stats = rejected_stats
#         rejected = sum(rejected_stats.values())
#         top_reject = ", ".join(
#             f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:8]
#         ) or "none"
#         print(
#             f"[WanVideoDataset][local_openvid] preclean_done scanned={scanned} "
#             f"accepted={len(samples)} rejected={rejected} top=[{top_reject}]"
#         )
#         if self.hf_subset_use_cache:
#             try:
#                 with open(cache_path, "wb") as f:
#                     pickle.dump(
#                         {
#                             "scanned": scanned,
#                             "samples": samples,
#                             "rejected_stats": dict(rejected_stats),
#                         },
#                         f,
#                     )
#                 print(f"[WanVideoDataset] local_openvid_subset_cache_write={cache_path} saved={len(samples)}")
#             except Exception:
#                 pass
#         if len(samples) == 0:
#             raise RuntimeError(
#                 "本地OpenVid(含HD)样本在预清洗后为0，请检查视频可解码性或放宽过滤阈值。"
#             )
#         return samples

#     def _preclean_sample(self, sample):
#         caption = sample["caption"]
#         token_ids = self.tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"]
#         if len(token_ids) > self.max_caption_tokens:
#             return False, "caption_too_long"
#         video_path = self._materialize_video(sample["video_spec"])
#         if video_path is None:
#             return False, "materialize_failed"
#         if not self._probe_video_quick(video_path):
#             try:
#                 p = Path(video_path)
#                 if self.local_video_cache_dir in p.parents:
#                     p.unlink(missing_ok=True)
#             except Exception:
#                 pass
#             return False, "probe_failed"
#         return True, "ok"

#     def _probe_video_quick(self, video_path):
#         try:
#             import cv2

#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 cap.release()
#                 return False
#             total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
#             if total <= 0 or total < self.frame_num:
#                 cap.release()
#                 return False
#             duration = (total / fps) if fps > 0 else None
#             if duration is not None and duration < self.min_duration_sec:
#                 cap.release()
#                 return False
#             if duration is not None and duration > self.max_duration_sec:
#                 cap.release()
#                 return False
#             ok, frame = cap.read()
#             cap.release()
#             if not ok or frame is None:
#                 return False
#             return True
#         except Exception:
#             return False

#     def _extract_row(self, row):
#         caption = self._extract_caption(row)
#         if caption is None:
#             return None
#         video_spec = self._extract_video_spec(row)
#         if video_spec is None:
#             return None
#         return {
#             "caption": caption,
#             "video_spec": video_spec,
#             "raw": row,
#         }

#     def _extract_caption(self, row):
#         keys = (
#             "caption",
#             "text",
#             "description",
#             "prompt",
#             "sentence",
#             "summary",
#         )
#         for k in keys:
#             v = row.get(k, None)
#             if isinstance(v, str) and v.strip():
#                 return v.strip()
#         return None

#     def _extract_video_spec(self, row):
#         keys = (
#             "video",
#             "mp4",
#             "file",
#             "video_path",
#             "path",
#             "url",
#             "video_name",
#             "filename",
#         )
#         candidates = []
#         for k in keys:
#             if k not in row:
#                 continue
#             value = row[k]
#             candidates.extend(self._collect_video_specs(value))
#         if not candidates:
#             return None
#         return max(candidates, key=self._video_spec_priority)

#     def _collect_video_specs(self, value, depth=0):
#         if depth > 4:
#             return []
#         out = []
#         spec = self._parse_video_value(value)
#         if spec is not None:
#             out.append(spec)
#         if isinstance(value, dict):
#             preferred_keys = (
#                 "bytes",
#                 "url",
#                 "download_url",
#                 "video_url",
#                 "href",
#                 "path",
#                 "file",
#                 "video",
#             )
#             for k in preferred_keys:
#                 if k in value:
#                     out.extend(self._collect_video_specs(value.get(k), depth + 1))
#             for k, v in value.items():
#                 if k in preferred_keys:
#                     continue
#                 out.extend(self._collect_video_specs(v, depth + 1))
#         elif isinstance(value, (list, tuple)):
#             for v in value:
#                 out.extend(self._collect_video_specs(v, depth + 1))
#         uniq = {}
#         for item in out:
#             kind = item.get("kind")
#             val = item.get("value")
#             if kind == "bytes":
#                 key = ("bytes", hashlib.sha1(bytes(val)).hexdigest())
#             else:
#                 key = (kind, str(val))
#             uniq[key] = item
#         return list(uniq.values())

#     def _video_spec_priority(self, spec):
#         kind = spec.get("kind")
#         value = spec.get("value")
#         if kind == "bytes":
#             return 300
#         if kind == "url":
#             return 200
#         if kind == "path":
#             try:
#                 p = Path(value)
#                 if p.exists():
#                     return 100
#             except Exception:
#                 pass
#             return 10
#         return 0

#     def _parse_video_value(self, value):
#         if value is None:
#             return None
#         if isinstance(value, str):
#             value = value.strip()
#             if not value:
#                 return None
#             if value.startswith("http://") or value.startswith("https://"):
#                 return {"kind": "url", "value": value}
#             if value.startswith("hf://"):
#                 return {"kind": "url", "value": value}
#             return {"kind": "path", "value": value}
#         if isinstance(value, (bytes, bytearray)):
#             return {"kind": "bytes", "value": value}
#         return None

#     def _normalize_rel_path(self, value: str):
#         path = str(value).strip().replace("\\", "/")
#         if path.startswith("hf://"):
#             return path
#         while path.startswith("./"):
#             path = path[2:]
#         return path.lstrip("/")

#     def _candidate_urls_from_missing_path(self, value: str):
#         path = self._normalize_rel_path(value)
#         if path.startswith("http://") or path.startswith("https://"):
#             return [path]
#         if path.startswith("hf://datasets/"):
#             return [
#                 path.replace(
#                     f"hf://datasets/{self.dataset_name}/",
#                     f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/",
#                 )
#             ]
#         if self.is_openvid and (not self.openvid_allow_http_guess):
#             return []
#         ds_name = self.dataset_name
#         split = str(self.hf_split).strip("/")
#         candidates = [
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/data/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/videos/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/{split}/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/files/{path}",
#         ]
#         uniq = []
#         seen = set()
#         for u in candidates:
#             if u in seen:
#                 continue
#             seen.add(u)
#             uniq.append(u)
#         return uniq

#     def _resolve_openvid_local_path(self, value: str):
#         normalized = self._normalize_rel_path(value)
#         if normalized.startswith("http://") or normalized.startswith("https://"):
#             return None
#         if normalized.startswith("hf://"):
#             return None
#         if self.openvid_video_root:
#             p = self.openvid_video_root / normalized
#             if p.exists():
#                 return p
#             try:
#                 p2 = self.openvid_video_root / Path(normalized).name
#                 if p2.exists():
#                     return p2
#             except Exception:
#                 pass
#         return self._extract_openvid_from_archives(normalized)

#     def _prepare_openvid_archive_root_from_hf(self):
#         try:
#             print(
#                 f"[WanVideoDataset] openvid_snapshot_download_start "
#                 f"repo={self.dataset_name} local_dir={self.openvid_snapshot_dir} "
#                 f"patterns={self.openvid_snapshot_patterns}"
#             )
#             path = snapshot_download(
#                 repo_id=self.dataset_name,
#                 repo_type="dataset",
#                 local_dir=str(self.openvid_snapshot_dir),
#                 allow_patterns=self.openvid_snapshot_patterns,
#             )
#             print(f"[WanVideoDataset] openvid_snapshot_download_done local_dir={path}")
#             return Path(path)
#         except Exception:
#             self._failure_stats["openvid_snapshot_download_error"] += 1
#             return None

#     def _iter_openvid_archive_files(self):
#         if not self.openvid_archive_root or (not self.openvid_archive_root.exists()):
#             return []
#         archives = sorted(self.openvid_archive_root.rglob("*.zip"))
#         if self.openvid_auto_join_parts:
#             archives.extend(self._join_openvid_part_archives())
#             archives = sorted(set(archives))
#         if self._openvid_archive_max_scan > 0:
#             archives = archives[: self._openvid_archive_max_scan]
#         return archives

#     def _join_openvid_part_archives(self):
#         part_files = sorted(self.openvid_archive_root.rglob("*.part*"))
#         groups = defaultdict(list)
#         for part in part_files:
#             name = part.name
#             if ".part" not in name:
#                 continue
#             prefix, suffix = name.split(".part", 1)
#             if not suffix:
#                 continue
#             groups[prefix].append((suffix, part))
#         joined_archives = []
#         for prefix, items in groups.items():
#             items = sorted(items, key=lambda x: x[0])
#             if len(items) < 2:
#                 continue
#             out_zip = self.openvid_joined_archive_dir / f"{prefix}.zip"
#             if not out_zip.exists():
#                 lock_path = Path(f"{out_zip}.lock")
#                 if not self._acquire_lock(lock_path, timeout_sec=max(self.lock_timeout_sec, 120)):
#                     continue
#                 try:
#                     if not out_zip.exists():
#                         tmp_path = Path(
#                             f"{out_zip}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
#                         )
#                         try:
#                             with open(tmp_path, "wb") as wf:
#                                 for _, part_path in items:
#                                     with open(part_path, "rb") as rf:
#                                         while True:
#                                             chunk = rf.read(1024 * 1024)
#                                             if not chunk:
#                                                 break
#                                             wf.write(chunk)
#                             os.replace(tmp_path, out_zip)
#                             print(
#                                 f"[WanVideoDataset] openvid_join_parts_done prefix={prefix} parts={len(items)} out={out_zip}"
#                             )
#                         except Exception:
#                             self._failure_stats["openvid_join_parts_error"] += 1
#                             try:
#                                 tmp_path.unlink(missing_ok=True)
#                             except Exception:
#                                 pass
#                 finally:
#                     self._release_lock(lock_path)
#             if out_zip.exists():
#                 joined_archives.append(out_zip)
#         return joined_archives

#     def _build_openvid_archive_index(self):
#         if self._openvid_archive_index_built:
#             return
#         self._openvid_archive_index_built = True
#         archives = self._iter_openvid_archive_files()
#         if not archives:
#             return
#         indexed = 0
#         for i, archive in enumerate(archives):
#             try:
#                 with zipfile.ZipFile(archive, "r") as zf:
#                     for name in zf.namelist():
#                         if not name.lower().endswith(".mp4"):
#                             continue
#                         base = Path(name).name
#                         if base not in self._openvid_archive_map:
#                             self._openvid_archive_map[base] = (archive, name)
#                             indexed += 1
#             except Exception:
#                 self._failure_stats["openvid_archive_read_error"] += 1
#             if (i + 1) % 5 == 0:
#                 print(
#                     f"[WanVideoDataset] openvid_archive_index_progress scanned_archives={i + 1} "
#                     f"indexed_videos={indexed}"
#                 )
#         print(
#             f"[WanVideoDataset] openvid_archive_index_done archives={len(archives)} "
#             f"indexed_videos={indexed}"
#         )

#     def _extract_openvid_from_archives(self, normalized: str):
#         if not self.openvid_archive_root:
#             return None
#         file_name = Path(normalized).name
#         if (not file_name) or (not file_name.lower().endswith(".mp4")):
#             return None
#         if file_name in self._openvid_archive_miss:
#             return None
#         target_path = self.openvid_extracted_cache_dir / file_name
#         if target_path.exists() and self._is_cache_file_valid(target_path):
#             return target_path
#         self._build_openvid_archive_index()
#         arc = self._openvid_archive_map.get(file_name, None)
#         if arc is None:
#             self._openvid_archive_miss.add(file_name)
#             self._failure_stats["openvid_archive_member_missing"] += 1
#             return None
#         archive_path, member_name = arc
#         lock_path = Path(f"{target_path}.lock")
#         if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
#             self._failure_stats["lock_timeout"] += 1
#             if target_path.exists() and self._is_cache_file_valid(target_path):
#                 return target_path
#             return None
#         try:
#             if target_path.exists() and self._is_cache_file_valid(target_path):
#                 return target_path
#             tmp_path = Path(
#                 f"{target_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
#             )
#             try:
#                 with zipfile.ZipFile(archive_path, "r") as zf:
#                     with zf.open(member_name, "r") as src, open(tmp_path, "wb") as dst:
#                         while True:
#                             chunk = src.read(1024 * 1024)
#                             if not chunk:
#                                 break
#                             dst.write(chunk)
#                 os.replace(tmp_path, target_path)
#             except Exception:
#                 self._failure_stats["openvid_archive_extract_error"] += 1
#                 try:
#                     tmp_path.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 return None
#             if not self._is_cache_file_valid(target_path):
#                 self._failure_stats["openvid_archive_extracted_invalid"] += 1
#                 try:
#                     target_path.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 return None
#             self._failure_stats["openvid_archive_extract_hit"] += 1
#             return target_path
#         finally:
#             self._release_lock(lock_path)

#     def _cached_file_path(self, key: str, suffix=".mp4"):
#         file_name = hashlib.sha1(key.encode("utf-8")).hexdigest() + suffix
#         return self.local_video_cache_dir / file_name

#     def _build_http_session(self):
#         session = requests.Session()
#         try:
#             from urllib3.util.retry import Retry

#             retry = Retry(
#                 total=self.http_retry_total,
#                 connect=self.http_retry_total,
#                 read=self.http_retry_total,
#                 backoff_factor=0.3,
#                 status_forcelist=(429, 500, 502, 503, 504),
#                 allowed_methods=frozenset(["GET"]),
#                 raise_on_status=False,
#             )
#             adapter = HTTPAdapter(max_retries=retry)
#             session.mount("http://", adapter)
#             session.mount("https://", adapter)
#         except Exception:
#             pass
#         return session

#     def _acquire_lock(self, lock_path: Path, timeout_sec: float = 120.0):
#         deadline = time.time() + timeout_sec
#         while time.time() < deadline:
#             try:
#                 fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
#                 with os.fdopen(fd, "w", encoding="utf-8") as f:
#                     f.write(f"{os.getpid()} {time.time()}")
#                 return True
#             except FileExistsError:
#                 try:
#                     age = time.time() - lock_path.stat().st_mtime
#                     if age > 300:
#                         lock_path.unlink(missing_ok=True)
#                         continue
#                 except Exception:
#                     pass
#                 time.sleep(0.2)
#             except Exception:
#                 time.sleep(0.2)
#         return False

#     def _release_lock(self, lock_path: Path):
#         try:
#             lock_path.unlink(missing_ok=True)
#         except Exception:
#             pass

#     def _is_cache_file_valid(self, file_path: Path):
#         if not file_path.exists():
#             return False
#         try:
#             if file_path.stat().st_size < 4096:
#                 return False
#         except Exception:
#             return False
#         try:
#             import cv2

#             cap = cv2.VideoCapture(str(file_path))
#             if not cap.isOpened():
#                 cap.release()
#                 return False
#             total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             cap.release()
#             return total > 0
#         except Exception:
#             return False

#     def _download_url_to_file(self, url: str, target_path: Path, headers):
#         tmp_path = Path(f"{target_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}")
#         session = self._build_http_session()
#         try:
#             with session.get(url, stream=True, timeout=self.http_timeout_sec, headers=headers) as r:
#                 status = int(getattr(r, "status_code", 0) or 0)
#                 if status >= 400:
#                     return False, f"http_{status}"
#                 r.raise_for_status()
#                 bytes_written = 0
#                 with open(tmp_path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=1024 * 1024):
#                         if not chunk:
#                             continue
#                         f.write(chunk)
#                         bytes_written += len(chunk)
#             if bytes_written < 4096:
#                 try:
#                     tmp_path.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 return False, "download_too_small"
#             os.replace(tmp_path, target_path)
#             return True, "ok"
#         except requests.exceptions.Timeout:
#             return False, "timeout"
#         except requests.exceptions.HTTPError:
#             return False, "http_error"
#         except Exception:
#             return False, "download_exception"
#         finally:
#             try:
#                 tmp_path.unlink(missing_ok=True)
#             except Exception:
#                 pass
#             try:
#                 session.close()
#             except Exception:
#                 pass

#     def _materialize_video(self, video_spec):
#         kind = video_spec["kind"]
#         value = video_spec["value"]
#         if kind == "path":
#             if isinstance(value, str) and (
#                 value.startswith("http://") or value.startswith("https://")
#             ):
#                 return self._materialize_video({"kind": "url", "value": value})
#             if self.is_openvid:
#                 local_openvid = self._resolve_openvid_local_path(str(value))
#                 if local_openvid is not None:
#                     if not self._is_cache_file_valid(local_openvid):
#                         self._failure_stats["openvid_local_invalid"] += 1
#                         return None
#                     return str(local_openvid)
#             p = Path(value)
#             if p.exists():
#                 if not self._is_cache_file_valid(p):
#                     self._failure_stats["path_invalid"] += 1
#                     return None
#                 return str(p)
#             if self.is_openvid:
#                 normalized = self._normalize_rel_path(str(value))
#                 if normalized.lower().endswith(".mp4"):
#                     if (self.openvid_video_root is None) and (self.openvid_archive_root is None):
#                         self._failure_stats["openvid_filename_without_local_root"] += 1
#                     else:
#                         self._failure_stats["openvid_local_missing"] += 1
#                     return None
#             candidate_urls = self._candidate_urls_from_missing_path(str(value))
#             if self.quick_fail:
#                 candidate_urls = candidate_urls[: self.url_fallback_limit]
#             for url in candidate_urls:
#                 url_video = self._materialize_video({"kind": "url", "value": url})
#                 if url_video is not None:
#                     self._failure_stats["path_to_url_fallback_hit"] += 1
#                     return url_video
#             self._failure_stats["path_missing"] += 1
#             return None
#         if kind == "url":
#             local_path = self._cached_file_path(value, suffix=".mp4")
#             lock_path = Path(f"{local_path}.lock")
#             if local_path.exists() and self._is_cache_file_valid(local_path):
#                 return str(local_path)
#             if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
#                 self._failure_stats["lock_timeout"] += 1
#                 if local_path.exists() and self._is_cache_file_valid(local_path):
#                     return str(local_path)
#                 return None
#             try:
#                 headers = None
#                 hf_token = os.environ.get("HF_TOKEN", "").strip()
#                 if hf_token and "huggingface.co" in value:
#                     headers = {"Authorization": f"Bearer {hf_token}"}
#                 if "huggingface.co" in value and (not hf_token) and (not self._warned_hf_token):
#                     print("[WanVideoDataset] warning: huggingface url without HF_TOKEN")
#                     self._warned_hf_token = True
#                 if local_path.exists() and not self._is_cache_file_valid(local_path):
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                 if local_path.exists() and self._is_cache_file_valid(local_path):
#                     return str(local_path)
#                 ok, reason = self._download_url_to_file(value, local_path, headers=headers)
#                 if not ok:
#                     self._failure_stats[reason] += 1
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 if not self._is_cache_file_valid(local_path):
#                     self._failure_stats["cache_invalid_after_download"] += 1
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 return str(local_path)
#             finally:
#                 self._release_lock(lock_path)
#         if kind == "bytes":
#             if not isinstance(value, (bytes, bytearray)):
#                 self._failure_stats["bytes_invalid_type"] += 1
#                 return None
#             key = hashlib.sha1(value).hexdigest()
#             local_path = self._cached_file_path(key, suffix=".mp4")
#             lock_path = Path(f"{local_path}.lock")
#             if local_path.exists() and self._is_cache_file_valid(local_path):
#                 return str(local_path)
#             if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
#                 self._failure_stats["lock_timeout"] += 1
#                 if local_path.exists() and self._is_cache_file_valid(local_path):
#                     return str(local_path)
#                 return None
#             try:
#                 if local_path.exists() and not self._is_cache_file_valid(local_path):
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                 try:
#                     tmp_path = Path(
#                         f"{local_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
#                     )
#                     with open(tmp_path, "wb") as f:
#                         f.write(bytes(value))
#                     os.replace(tmp_path, local_path)
#                 except Exception:
#                     self._failure_stats["bytes_write_failed"] += 1
#                     try:
#                         tmp_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 if not self._is_cache_file_valid(local_path):
#                     self._failure_stats["bytes_cache_invalid"] += 1
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 return str(local_path)
#             finally:
#                 self._release_lock(lock_path)
#         return None

#     def _load_video_frames(self, video_path):
#         import cv2

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return None, None, None
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
#         if total <= 0:
#             cap.release()
#             return None, None, None
#         if total < self.frame_num:
#             cap.release()
#             return None, None, None
#         duration = (total / fps) if fps > 0 else None
#         if duration is not None and duration < self.min_duration_sec:
#             cap.release()
#             return None, None, None
#         if duration is not None and duration > self.max_duration_sec:
#             cap.release()
#             return None, None, None

#         indices = np.linspace(0, total - 1, self.frame_num, dtype=int)
#         frames = []
#         for idx in indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#         cap.release()
#         if len(frames) < self.frame_num:
#             return None, None, None
#         return frames, fps, duration

#     def _process_frames(self, frames):
#         import cv2

#         h, w = frames[0].shape[:2]
#         area = h * w
#         if area > self.max_area:
#             scale = math.sqrt(self.max_area / area)
#             h = int(h * scale)
#             w = int(w * scale)
#         h = max(32, (h // 32) * 32)
#         w = max(32, (w // 32) * 32)
#         resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
#         tensor = torch.stack([torch.from_numpy(f) for f in resized]).float() / 127.5 - 1.0
#         tensor = tensor.permute(3, 0, 1, 2)
#         return tensor, resized[0]

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         n = len(self.samples)
#         default_trials = min(max(60, n // 4), n)
#         if self.quick_fail:
#             max_trials = min(default_trials, self.max_trials_cap, n)
#             if n >= 20:
#                 max_trials = max(20, max_trials)
#         else:
#             max_trials = default_trials
#         if max_trials <= 0:
#             raise RuntimeError("数据集为空，无法取样")
#         trial_stats = defaultdict(int)
#         for trial in range(max_trials):
#             if trial < 10:
#                 sample_idx = (idx + trial) % n
#             else:
#                 sample_idx = random.randint(0, n - 1)
#             sample = self.samples[sample_idx]
#             caption = sample["caption"]
#             token_ids = self.tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"]
#             if len(token_ids) > self.max_caption_tokens:
#                 trial_stats["caption_too_long"] += 1
#                 continue

#             video_path = self._materialize_video(sample["video_spec"])
#             if video_path is None:
#                 trial_stats["materialize_failed"] += 1
#                 if (trial + 1) % self.trial_log_interval == 0:
#                     trial_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(trial_stats.items())
#                     ) or "none"
#                     io_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(self._failure_stats.items())
#                     ) or "none"
#                     print(
#                         f"[WanVideoDataset] retry_progress idx={idx} tried={trial + 1}/{max_trials} "
#                         f"trial_stats=[{trial_detail}] io_stats=[{io_detail}]"
#                     )
#                 continue
#             frames, _, _ = self._load_video_frames(video_path)
#             if frames is None:
#                 trial_stats["decode_failed"] += 1
#                 try:
#                     p = Path(video_path)
#                     if self.local_video_cache_dir in p.parents:
#                         p.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 if (trial + 1) % self.trial_log_interval == 0:
#                     trial_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(trial_stats.items())
#                     ) or "none"
#                     io_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(self._failure_stats.items())
#                     ) or "none"
#                     print(
#                         f"[WanVideoDataset] retry_progress idx={idx} tried={trial + 1}/{max_trials} "
#                         f"trial_stats=[{trial_detail}] io_stats=[{io_detail}]"
#                     )
#                 continue
#             video_tensor, first_frame_np = self._process_frames(frames)
#             ref_image = Image.fromarray(first_frame_np)
#             mq_ref_image = ref_image
#             out_caption = caption
#             # if random.random() < self.null_caption_prob:
#             #     out_caption = ""
#             # if random.random() < self.null_image_prob:
#             #     mq_ref_image = None
#             if not self._printed_sample_info:
#                 print(
#                     f"[WanVideoDataset] sample_once "
#                     f"video_tensor={tuple(video_tensor.shape)} "
#                     f"caption_tokens={len(token_ids)} "
#                     f"mq_ref_is_none={mq_ref_image is None} "
#                     f"video_path={video_path}"
#                 )
#                 self._printed_sample_info = True
#             result = {
#                 "caption": out_caption,
#                 "video": video_tensor,
#                 "ref_image": ref_image,
#                 "mq_ref_image": mq_ref_image,
#                 "video_path": video_path,
#             }
#             self._last_good_sample = result
#             return result
#         if self._last_good_sample is not None:
#             return self._last_good_sample
#         trial_detail = ", ".join(f"{k}={v}" for k, v in sorted(trial_stats.items())) or "none"
#         io_detail = ", ".join(f"{k}={v}" for k, v in sorted(self._failure_stats.items())) or "none"
#         hf_token_ready = bool(os.environ.get("HF_TOKEN", "").strip())
#         raise RuntimeError(
#             f"样本解码失败，trials={max_trials}，dataset={self.dataset_name}，"
#             f"trial_stats=[{trial_detail}] io_stats=[{io_detail}] "
#             f"streaming={self.hf_streaming} hf_token={'set' if hf_token_ready else 'unset'} "
#             f"openvid_root={str(self.openvid_video_root) if self.openvid_video_root else 'unset'} "
#             f"openvid_archive_root={str(self.openvid_archive_root) if self.openvid_archive_root else 'unset'}，"
#             f"建议设置HF_TOKEN并优先使用--hf_no_streaming；OpenVid请配置OPENVID_VIDEO_ROOT或OPENVID_ARCHIVE_ROOT，"
#             f"或开启OPENVID_SNAPSHOT_DOWNLOAD=1自动从HF拉取归档"
#         )

#     @staticmethod
#     def collate_fn(batch):
#         return {k: [item[k] for item in batch] for k in batch[0]}

# import hashlib
# import math
# import os
# import pickle
# import random
# import time
# import zipfile
# import csv
# from collections import defaultdict
# from pathlib import Path

# import numpy as np
# import requests
# import torch
# import torch.nn as nn
# from PIL import Image
# from datasets import load_dataset
# from huggingface_hub import snapshot_download
# from requests.adapters import HTTPAdapter
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer


# DEFAULT_STAGE_DATASET = {
#     "stage1": "nkp37/OpenVid-1M",
#     "stage2": "BestWishYsh/OpenS2V-5M",
# }

# DEFAULT_STAGE_TOTAL = {
#     "nkp37/OpenVid-1M": 1_000_000,
#     "BestWishYsh/OpenS2V-5M": 5_000_000,
# }


# class MetaQueryEncoderForWan(nn.Module):
#     WAN_TEXT_DIM = 4096

#     def __init__(
#         self,
#         qwen3vl_model_id: str,
#         num_metaqueries: int = 256,
#         connector_num_hidden_layers: int = 24,
#         gradient_checkpointing: bool = False,
#         train_input_embeddings: bool = True,
#         dtype: torch.dtype = torch.bfloat16,
#         device: str = "cuda",
#     ):
#         super().__init__()
#         self.num_metaqueries = num_metaqueries
#         self.wan_text_dim = self.WAN_TEXT_DIM
#         self.dtype = dtype
#         self.device = torch.device(device)
#         self.train_input_embeddings = bool(train_input_embeddings)
#         self._printed_forward_stats = False

#         from diffusers.models.normalization import RMSNorm
#         from models.model import MLLMInContext, MLLMInContextConfig
#         from models.transformer_encoder import Qwen2Encoder
#         from transformers import Qwen2Config

#         # 关键点：
#         # 1) diffusion_model_id 设为 "none" -> 不会加载 Sana/SD 的扩散骨干；
#         # 2) connector_out_dim_override 直接指定为 4096，与 Wan text_dim 对齐。
#         config = MLLMInContextConfig(
#             mllm_id=qwen3vl_model_id,
#             diffusion_model_id="none",
#             connector_out_dim_override=self.wan_text_dim,
#             num_metaqueries=num_metaqueries,
#             _gradient_checkpointing=gradient_checkpointing,
#             connector_num_hidden_layers=connector_num_hidden_layers,
#         )

#         mllm_model = MLLMInContext(config).to(device=self.device, dtype=dtype)
#         self.mllm_model = mllm_model
#         self.tokenizer = mllm_model.get_tokenizer()
#         self.tokenize = mllm_model.get_tokenize_fn()
#         # 开启梯度检查点时关闭 KV cache，减少显存并避免反复告警。
#         try:
#             if hasattr(self.mllm_model.mllm_backbone, "config"):
#                 self.mllm_model.mllm_backbone.config.use_cache = False
#             if hasattr(self.mllm_model.mllm_backbone, "generation_config"):
#                 self.mllm_model.mllm_backbone.generation_config.use_cache = False
#         except Exception:
#             pass
#         print(
#             f"[MetaQueryEncoderForWan] mllm_type={self.mllm_model.mllm_type} "
#             f"transformer_loaded={self.mllm_model.transformer is not None}"
#         )

#         mllm_hidden_size = mllm_model.mllm_hidden_size
#         print(
#             f"[MetaQueryEncoderForWan] mllm_hidden={mllm_hidden_size} "
#             f"target_wan_text_dim={self.wan_text_dim}"
#         )
#         encoder = Qwen2Encoder(
#             Qwen2Config(
#                 hidden_size=mllm_hidden_size,
#                 intermediate_size=mllm_hidden_size * 4,
#                 num_hidden_layers=connector_num_hidden_layers,
#                 num_attention_heads=mllm_hidden_size // 64,
#                 num_key_value_heads=mllm_hidden_size // 64,
#                 initializer_range=0.014,
#                 use_cache=False,
#                 rope=True,
#                 qk_norm=True,
#             )
#         )
#         # 兼容自定义 Qwen2Encoder 的梯度检查点调用。
#         if hasattr(encoder, "gradient_checkpointing"):
#             encoder.gradient_checkpointing = bool(gradient_checkpointing)
#         if gradient_checkpointing and not hasattr(encoder, "_gradient_checkpointing_func"):
#             encoder._gradient_checkpointing_func = (
#                 lambda func, *gc_args: torch.utils.checkpoint.checkpoint(
#                     func, *gc_args, use_reentrant=False
#                 )
#             )
#         norm = RMSNorm(self.wan_text_dim, eps=1e-5, elementwise_affine=True)
#         with torch.no_grad():
#             norm.weight.fill_(math.sqrt(5.5))
#         new_connector = nn.Sequential(
#             encoder,
#             nn.Linear(mllm_hidden_size, self.wan_text_dim),
#             nn.GELU(approximate="tanh"),
#             nn.Linear(self.wan_text_dim, self.wan_text_dim),
#             norm,
#         ).to(device=self.device, dtype=dtype)
#         self.mllm_model.connector = new_connector
#         self.mllm_model.connector_out_dim = self.wan_text_dim
#         print(
#             f"[MetaQueryEncoderForWan] connector_out={self.mllm_model.connector_out_dim} "
#             f"num_metaqueries={self.num_metaqueries}"
#         )

#         self.mllm_model.mllm_backbone.requires_grad_(False)
#         self.mllm_model.connector.requires_grad_(True)
#         self.mllm_model.mllm_backbone.get_input_embeddings().requires_grad_(self.train_input_embeddings)
#         print(
#             f"[MetaQueryEncoderForWan] train_connector=True "
#             f"train_input_embeddings={self.train_input_embeddings}"
#         )

#         if hasattr(self.mllm_model, "transformer"):
#             del self.mllm_model.transformer
#             self.mllm_model.transformer = None
#         torch.cuda.empty_cache()

#     def get_trainable_params(self):
#         return [p for p in self.parameters() if p.requires_grad]

#     def forward(self, captions, input_images=None):
#         if input_images is not None:
#             input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
#                 self.tokenizer, captions, input_images
#             )
#             input_ids = input_ids.to(self.device)
#             attention_mask = attention_mask.to(self.device)
#             if pixel_values is not None:
#                 pixel_values = pixel_values.to(self.device, self.dtype)
#                 if self.mllm_model.mllm_type in ("qwenvl", "qwen3vl"):
#                     pixel_values = pixel_values.squeeze(0)
#             if image_sizes is not None:
#                 image_sizes = image_sizes.to(self.device)
#         else:
#             input_ids, attention_mask = self.tokenize(self.tokenizer, captions)
#             input_ids = input_ids.to(self.device)
#             attention_mask = attention_mask.to(self.device)
#             pixel_values = None
#             image_sizes = None

#         mq_features, _ = self.mllm_model.encode_condition(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             pixel_values=pixel_values,
#             image_sizes=image_sizes,
#         )
#         if not self._printed_forward_stats:
#             print(
#                 f"[MetaQueryEncoderForWan] forward_once "
#                 f"input_ids={tuple(input_ids.shape)} "
#                 f"attention_mask={tuple(attention_mask.shape)} "
#                 f"mq_features={tuple(mq_features.shape)} "
#                 f"dtype={mq_features.dtype}"
#             )
#             self._printed_forward_stats = True
#         return mq_features


# def resolve_hf_stage_config(stage: str, dataset_name: str | None, subset_ratio: float):
#     stage_name = stage.lower()
#     ds = dataset_name or DEFAULT_STAGE_DATASET.get(stage_name, DEFAULT_STAGE_DATASET["stage1"])
#     total_hint = DEFAULT_STAGE_TOTAL.get(ds, None)
#     subset_size = int(total_hint * subset_ratio) if total_hint is not None else None
#     return ds, subset_size, total_hint


# class WanVideoDataset(Dataset):
#     def __init__(
#         self,
#         frame_num: int = 81,
#         max_area: int = 720 * 1280,
#         null_caption_prob: float = 0.0,
#         null_image_prob: float = 0.5,
#         max_caption_tokens: int = 512,
#         caption_tokenizer_path: str = "google/umt5-xxl",
#         min_duration_sec: float = 0.5,
#         max_duration_sec: float = 20.0,
#         hf_stage: str = "stage1",
#         hf_dataset_name: str | None = None,
#         hf_split: str = "train",
#         hf_subset_ratio: float = 0.01,
#         hf_subset_size: int | None = None,
#         hf_scan_factor: int = 30,
#         hf_subset_cache_dir: str | None = None,
#         hf_subset_use_cache: bool = True,
#         hf_cache_dir: str | None = None,
#         hf_streaming: bool = True,
#         hf_shuffle_buffer: int = 10000,
#         seed: int = 42,
#         local_video_cache_dir: str | None = None,
#         local_openvid_video_root: str | None = None,
#         local_openvid_csv_path: str | None = None,
#         local_openvid_limit: int | None = None,
#         local_openvid_hd_video_root: str | None = None,
#         local_openvid_hd_csv_path: str | None = None,
#         local_openvid_hd_limit: int | None = None,
#     ):
#         self.frame_num = frame_num
#         self.max_area = max_area
#         self.null_caption_prob = null_caption_prob
#         self.null_image_prob = null_image_prob
#         self.max_caption_tokens = max_caption_tokens
#         self.min_duration_sec = min_duration_sec
#         self.max_duration_sec = max_duration_sec
#         self.hf_split = hf_split
#         self.hf_cache_dir = hf_cache_dir
#         self.hf_streaming = hf_streaming
#         self.hf_shuffle_buffer = hf_shuffle_buffer
#         self.seed = seed
#         self.scan_factor = max(5, hf_scan_factor)
#         self._printed_sample_info = False
#         self._last_good_sample = None
#         self._failure_stats = defaultdict(int)
#         self._warned_hf_token = False
#         self._subset_scanned = 0
#         self._subset_accepted = 0
#         self._subset_rejected_stats = defaultdict(int)
#         self.quick_fail = os.environ.get("WAN_DATA_QUICK_FAIL", "1").strip().lower() not in (
#             "0",
#             "false",
#             "off",
#         )
#         self.max_trials_cap = max(10, int(os.environ.get("WAN_DATA_MAX_TRIALS", "400")))
#         self.trial_log_interval = max(
#             1, int(os.environ.get("WAN_DATA_TRIAL_LOG_INTERVAL", "100"))
#         )
#         self.url_fallback_limit = max(
#             1, int(os.environ.get("WAN_DATA_PATH_URL_FALLBACK_LIMIT", "2"))
#         )
#         self.http_retry_total = max(0, int(os.environ.get("WAN_DATA_HTTP_RETRY_TOTAL", "1")))
#         self.http_timeout_sec = max(3, int(os.environ.get("WAN_DATA_HTTP_TIMEOUT_SEC", "12")))
#         self.lock_timeout_sec = max(3, int(os.environ.get("WAN_DATA_LOCK_TIMEOUT_SEC", "20")))
#         self.preclean_enabled = os.environ.get("WAN_DATA_PRECLEAN", "1").strip().lower() not in (
#             "0",
#             "false",
#             "off",
#         )
#         self.preclean_log_interval = max(
#             1, int(os.environ.get("WAN_DATA_PRECLEAN_LOG_INTERVAL", "200"))
#         )
#         self.preclean_scan_multiplier = max(
#             1, int(os.environ.get("WAN_DATA_PRECLEAN_SCAN_MULTIPLIER", "1"))
#         )
#         self.preclean_scan_cap = max(
#             1000, int(os.environ.get("WAN_DATA_PRECLEAN_SCAN_CAP", "200000"))
#         )
#         self.preclean_zero_accept_abort_scan = max(
#             1000, int(os.environ.get("WAN_DATA_PRECLEAN_ZERO_ACCEPT_ABORT_SCAN", "20000"))
#         )
#         self.hf_subset_ratio = hf_subset_ratio
#         self.hf_subset_size = hf_subset_size
#         self.hf_subset_use_cache = hf_subset_use_cache
#         self.hf_subset_cache_dir = Path(
#             hf_subset_cache_dir or (Path(hf_cache_dir or ".hf_cache") / "subset_cache")
#         )
#         self.hf_subset_cache_dir.mkdir(parents=True, exist_ok=True)
#         self.local_video_cache_dir = Path(
#             local_video_cache_dir or (Path(hf_cache_dir or ".hf_cache") / "video_cache")
#         )
#         self.local_video_cache_dir.mkdir(parents=True, exist_ok=True)
#         if local_openvid_video_root is None:
#             env_video_root = os.environ.get("OPENVID_LOCAL_VIDEO_ROOT", "").strip()
#             local_openvid_video_root = env_video_root or None
#         if local_openvid_csv_path is None:
#             env_csv_path = os.environ.get("OPENVID_LOCAL_CSV_PATH", "").strip()
#             local_openvid_csv_path = env_csv_path or None
#         if local_openvid_limit is None:
#             env_limit = os.environ.get("OPENVID_LOCAL_LIMIT", "").strip()
#             if env_limit:
#                 try:
#                     local_openvid_limit = int(env_limit)
#                 except Exception:
#                     local_openvid_limit = None
#         if local_openvid_hd_video_root is None:
#             env_hd_video_root = os.environ.get("OPENVID_HD_LOCAL_VIDEO_ROOT", "").strip()
#             local_openvid_hd_video_root = env_hd_video_root or None
#         if local_openvid_hd_csv_path is None:
#             env_hd_csv_path = os.environ.get("OPENVID_HD_LOCAL_CSV_PATH", "").strip()
#             local_openvid_hd_csv_path = env_hd_csv_path or None
#         if local_openvid_hd_limit is None:
#             env_hd_limit = os.environ.get("OPENVID_HD_LOCAL_LIMIT", "").strip()
#             if env_hd_limit:
#                 try:
#                     local_openvid_hd_limit = int(env_hd_limit)
#                 except Exception:
#                     local_openvid_hd_limit = None
#         env_total_limit = os.environ.get("OPENVID_LOCAL_TOTAL_LIMIT", "").strip()
#         local_openvid_total_limit = None
#         if env_total_limit:
#             try:
#                 local_openvid_total_limit = int(env_total_limit)
#             except Exception:
#                 local_openvid_total_limit = None

#         def _to_path(v):
#             return Path(v).expanduser().resolve() if v else None

#         def _to_limit(v):
#             if v is None:
#                 return None
#             try:
#                 iv = int(v)
#             except Exception:
#                 return None
#             return iv if iv > 0 else None

#         self.local_openvid_video_root = _to_path(local_openvid_video_root)
#         self.local_openvid_csv_path = _to_path(local_openvid_csv_path)
#         self.local_openvid_limit = _to_limit(local_openvid_limit)
#         self.local_openvid_hd_video_root = _to_path(local_openvid_hd_video_root)
#         self.local_openvid_hd_csv_path = _to_path(local_openvid_hd_csv_path)
#         self.local_openvid_hd_limit = _to_limit(local_openvid_hd_limit)
#         self.local_openvid_total_limit = _to_limit(local_openvid_total_limit)

#         self.local_openvid_sources = []
#         if self.local_openvid_video_root is not None and self.local_openvid_csv_path is not None:
#             self.local_openvid_sources.append(
#                 {
#                     "name": "openvid",
#                     "video_root": self.local_openvid_video_root,
#                     "csv_path": self.local_openvid_csv_path,
#                     "limit": self.local_openvid_limit,
#                 }
#             )
#         elif self.local_openvid_video_root is not None or self.local_openvid_csv_path is not None:
#             print(
#                 "[WanVideoDataset] warning: openvid 普通源参数不完整，"
#                 "需同时提供 local_openvid_video_root 与 local_openvid_csv_path，已忽略该源"
#             )

#         if self.local_openvid_hd_video_root is not None and self.local_openvid_hd_csv_path is not None:
#             self.local_openvid_sources.append(
#                 {
#                     "name": "openvid_hd",
#                     "video_root": self.local_openvid_hd_video_root,
#                     "csv_path": self.local_openvid_hd_csv_path,
#                     "limit": self.local_openvid_hd_limit,
#                 }
#             )
#         elif self.local_openvid_hd_video_root is not None or self.local_openvid_hd_csv_path is not None:
#             print(
#                 "[WanVideoDataset] warning: openvid HD 源参数不完整，"
#                 "需同时提供 local_openvid_hd_video_root 与 local_openvid_hd_csv_path，已忽略该源"
#             )

#         self.local_openvid_enabled = len(self.local_openvid_sources) > 0
#         tokenizer_local_only = os.environ.get("WAN_TOKENIZER_LOCAL_ONLY", "0").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 caption_tokenizer_path,
#                 local_files_only=tokenizer_local_only,
#             )
#         except Exception as e:
#             if tokenizer_local_only:
#                 raise RuntimeError(
#                     f"[WanVideoDataset] 本地加载 tokenizer 失败: {caption_tokenizer_path}. "
#                     "请确认该路径可读，或关闭 WAN_TOKENIZER_LOCAL_ONLY。"
#                 ) from e
#             # 兜底：网络异常时自动尝试仅本地缓存，避免 Determined 环境因外网抖动直接失败
#             try:
#                 self.tokenizer = AutoTokenizer.from_pretrained(
#                     caption_tokenizer_path,
#                     local_files_only=True,
#                 )
#                 print(
#                     f"[WanVideoDataset] tokenizer remote load failed, fallback local cache only: "
#                     f"path={caption_tokenizer_path} err={e}"
#                 )
#             except Exception as e2:
#                 raise RuntimeError(
#                     f"[WanVideoDataset] tokenizer 加载失败: {caption_tokenizer_path}. "
#                     "网络访问异常且本地缓存不可用。建议在 .sh 中设置 CAPTION_TOKENIZER_PATH 为本地目录，"
#                     "并开启 TOKENIZER_LOCAL_ONLY=1。"
#                 ) from e2

#         resolved_ds, stage_subset_size, total_hint = resolve_hf_stage_config(
#             stage=hf_stage,
#             dataset_name=hf_dataset_name,
#             subset_ratio=hf_subset_ratio,
#         )
#         if self.local_openvid_enabled:
#             self.dataset_name = "local/OpenVid-1M+HD"
#             self.total_hint = None
#             self.is_openvid = True
#         else:
#             self.dataset_name = resolved_ds
#             self.total_hint = total_hint
#             self.is_openvid = self.dataset_name.lower() == "nkp37/openvid-1m"
#         self.openvid_record_streaming = os.environ.get("OPENVID_RECORD_STREAMING", "1").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         openvid_root = os.environ.get("OPENVID_VIDEO_ROOT", "").strip()
#         if self.local_openvid_enabled:
#             openvid_root = str(self.local_openvid_sources[0]["video_root"])
#         self.openvid_video_root = Path(openvid_root) if openvid_root else None
#         openvid_archive_root = os.environ.get("OPENVID_ARCHIVE_ROOT", "").strip()
#         if self.local_openvid_enabled:
#             openvid_archive_root = ""
#         self.openvid_archive_root = Path(openvid_archive_root) if openvid_archive_root else None
#         self.openvid_snapshot_download = os.environ.get("OPENVID_SNAPSHOT_DOWNLOAD", "0").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         if self.local_openvid_enabled:
#             self.openvid_snapshot_download = False
#         self.openvid_snapshot_dir = Path(
#             os.environ.get("OPENVID_SNAPSHOT_DIR", str(Path(hf_cache_dir or ".hf_cache") / "openvid_repo"))
#         )
#         self.openvid_snapshot_dir.mkdir(parents=True, exist_ok=True)
#         self.openvid_snapshot_patterns = [
#             p.strip()
#             for p in os.environ.get(
#                 "OPENVID_SNAPSHOT_PATTERNS",
#                 "Openvid_part*.zip,Openvid_part*.part*,OpenVidHD.csv,data/*",
#             ).split(",")
#             if p.strip()
#         ]
#         self.openvid_allow_http_guess = os.environ.get("OPENVID_ALLOW_HTTP_GUESS", "0").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         self._openvid_archive_index_built = False
#         self._openvid_archive_map = {}
#         self._openvid_archive_miss = set()
#         self._openvid_autofallback_done = False
#         self._openvid_archive_max_scan = max(
#             0, int(os.environ.get("OPENVID_ARCHIVE_MAX_SCAN_FILES", "0"))
#         )
#         self.openvid_auto_join_parts = os.environ.get("OPENVID_AUTO_JOIN_PARTS", "1").strip().lower() in (
#             "1",
#             "true",
#             "on",
#         )
#         self.openvid_joined_archive_dir = Path(
#             os.environ.get("OPENVID_JOINED_ARCHIVE_DIR", str(self.local_video_cache_dir / "openvid_joined"))
#         )
#         self.openvid_joined_archive_dir.mkdir(parents=True, exist_ok=True)
#         self.openvid_extracted_cache_dir = self.local_video_cache_dir / "openvid_extracted"
#         self.openvid_extracted_cache_dir.mkdir(parents=True, exist_ok=True)
#         self._subset_streaming = (
#             (not self.local_openvid_enabled)
#             and (self.hf_streaming or (self.is_openvid and self.openvid_record_streaming))
#         )
#         if (
#             self.is_openvid
#             and (not self.local_openvid_enabled)
#             and self.openvid_record_streaming
#             and self.openvid_snapshot_download
#             and self.openvid_archive_root is None
#         ):
#             print("[WanVideoDataset] openvid_record_streaming=1, skip snapshot_download")
#             self.openvid_snapshot_download = False
#         if (
#             self.is_openvid
#             and (not self.local_openvid_enabled)
#             and self.openvid_archive_root is None
#             and self.openvid_snapshot_download
#         ):
#             self.openvid_archive_root = self._prepare_openvid_archive_root_from_hf()
#         if self.local_openvid_enabled:
#             per_source_limits = [src["limit"] for src in self.local_openvid_sources]
#             if per_source_limits and all(v is not None for v in per_source_limits):
#                 self.target_subset_size = sum(int(v) for v in per_source_limits)
#             else:
#                 self.target_subset_size = 0
#         else:
#             self.target_subset_size = hf_subset_size or stage_subset_size or 10000
#         self.samples = self._build_subset()
#         if len(self.samples) == 0:
#             raise RuntimeError(f"数据集可用样本为0: {self.dataset_name}")
#         print(
#             f"[WanVideoDataset] dataset={self.dataset_name} split={self.hf_split} "
#             f"target={self.target_subset_size} loaded={len(self.samples)} "
#             f"scanned={self._subset_scanned} scan_factor={self.scan_factor} "
#             f"streaming={self.hf_streaming} ratio={self.hf_subset_ratio} "
#             f"subset_size_override={self.hf_subset_size} cache_dir={self.hf_cache_dir}"
#         )
#         if self.is_openvid:
#             print(
#                 f"[WanVideoDataset] openvid_mode=1 "
#                 f"openvid_video_root={str(self.openvid_video_root) if self.openvid_video_root else 'unset'} "
#                 f"openvid_archive_root={str(self.openvid_archive_root) if self.openvid_archive_root else 'unset'} "
#                 f"snapshot_download={self.openvid_snapshot_download} "
#                 f"record_streaming={self.openvid_record_streaming} "
#                 f"auto_join_parts={self.openvid_auto_join_parts} "
#                 f"allow_http_guess={self.openvid_allow_http_guess}"
#             )
#         if self.local_openvid_enabled:
#             for src in self.local_openvid_sources:
#                 print(
#                     f"[WanVideoDataset] local_openvid_source "
#                     f"name={src['name']} video_root={src['video_root']} "
#                     f"csv_path={src['csv_path']} "
#                     f"limit={src['limit'] if src['limit'] else 'all'}"
#                 )

#     def _subset_cache_path(self):
#         if self.local_openvid_enabled:
#             source_parts = []
#             for src in self.local_openvid_sources:
#                 source_parts.append(
#                     f"{src['name']}:{src['video_root']}:{src['csv_path']}:limit={src['limit']}"
#                 )
#             key = (
#                 f"local_openvid_multi|{'|'.join(source_parts)}|"
#                 f"f={self.frame_num}|min={self.min_duration_sec}|"
#                 f"max={self.max_duration_sec}|tok={self.max_caption_tokens}|seed={self.seed}|"
#                 f"preclean={self.preclean_enabled}"
#             )
#             name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pkl"
#             return self.hf_subset_cache_dir / name
#         key = (
#             f"{self.dataset_name}|{self.hf_split}|{self.target_subset_size}|"
#             f"{self.scan_factor}|{self.seed}|{self.hf_streaming}|"
#             f"preclean={self.preclean_enabled}|f={self.frame_num}|min={self.min_duration_sec}|"
#             f"max={self.max_duration_sec}|tok={self.max_caption_tokens}"
#         )
#         name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pkl"
#         return self.hf_subset_cache_dir / name

#     def _build_subset(self):
#         cache_path = self._subset_cache_path()
#         if self.local_openvid_enabled:
#             return self._build_local_openvid_subset(cache_path)
#         if self.hf_subset_use_cache and cache_path.exists():
#             try:
#                 with open(cache_path, "rb") as f:
#                     payload = pickle.load(f)
#                 self._subset_scanned = int(payload.get("scanned", 0))
#                 samples = payload.get("samples", [])
#                 self._subset_accepted = len(samples)
#                 rejected_stats = payload.get("rejected_stats", {})
#                 self._subset_rejected_stats = defaultdict(
#                     int, {str(k): int(v) for k, v in rejected_stats.items()}
#                 )
#                 print(f"[WanVideoDataset] subset_cache_hit={cache_path} loaded={len(samples)}")
#                 if len(samples) > 0:
#                     return samples
#             except Exception:
#                 pass
#         ds = load_dataset(
#             self.dataset_name,
#             split=self.hf_split,
#             streaming=self._subset_streaming,
#             cache_dir=self.hf_cache_dir,
#         )
#         if self.hf_shuffle_buffer > 0 and hasattr(ds, "shuffle"):
#             try:
#                 if self._subset_streaming:
#                     ds = ds.shuffle(seed=self.seed, buffer_size=self.hf_shuffle_buffer)
#                 else:
#                     ds = ds.shuffle(seed=self.seed)
#             except TypeError:
#                 ds = ds.shuffle(seed=self.seed)

#         samples = []
#         rejected_stats = defaultdict(int)
#         scan_multiplier = self.preclean_scan_multiplier if self.preclean_enabled else 1
#         max_scan = self.target_subset_size * self.scan_factor * scan_multiplier
#         max_scan = min(max_scan, self.preclean_scan_cap)
#         scanned = 0
#         for row in ds:
#             scanned += 1
#             parsed = self._extract_row(row)
#             if parsed is not None:
#                 if self.preclean_enabled:
#                     ok, reject_reason = self._preclean_sample(parsed)
#                     if ok:
#                         samples.append(parsed)
#                     else:
#                         rejected_stats[reject_reason] += 1
#                 else:
#                     samples.append(parsed)
#             else:
#                 rejected_stats["extract_failed"] += 1
#             if scanned % self.preclean_log_interval == 0:
#                 rejected = sum(rejected_stats.values())
#                 top_reject = ", ".join(
#                     f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:4]
#                 ) or "none"
#                 top_io = ", ".join(
#                     f"{k}={v}" for k, v in sorted(self._failure_stats.items(), key=lambda x: -x[1])[:4]
#                 ) or "none"
#                 print(
#                     f"[WanVideoDataset] preclean_progress scanned={scanned} "
#                     f"accepted={len(samples)} rejected={rejected} top=[{top_reject}] io=[{top_io}] "
#                     f"target={self.target_subset_size} max_scan={max_scan}"
#                 )
#             if (
#                 self.preclean_enabled
#                 and len(samples) == 0
#                 and scanned >= self.preclean_zero_accept_abort_scan
#             ):
#                 print(
#                     f"[WanVideoDataset] preclean_early_abort scanned={scanned} accepted=0 "
#                     f"reason=zero_accepted_until_threshold({self.preclean_zero_accept_abort_scan})"
#                 )
#                 break
#             if len(samples) >= self.target_subset_size:
#                 break
#             if scanned >= max_scan:
#                 break
#         self._subset_scanned = scanned
#         self._subset_accepted = len(samples)
#         self._subset_rejected_stats = rejected_stats
#         rejected = sum(rejected_stats.values())
#         top_reject = ", ".join(
#             f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:8]
#         ) or "none"
#         top_io = ", ".join(
#             f"{k}={v}" for k, v in sorted(self._failure_stats.items(), key=lambda x: -x[1])[:8]
#         ) or "none"
#         print(
#             f"[WanVideoDataset] preclean_done scanned={scanned} accepted={len(samples)} "
#             f"rejected={rejected} top=[{top_reject}] io=[{top_io}] "
#             f"preclean={self.preclean_enabled} target={self.target_subset_size} max_scan={max_scan}"
#         )
#         if self.hf_subset_use_cache:
#             try:
#                 with open(cache_path, "wb") as f:
#                     pickle.dump(
#                         {
#                             "scanned": scanned,
#                             "samples": samples,
#                             "rejected_stats": dict(rejected_stats),
#                         },
#                         f,
#                     )
#                 print(f"[WanVideoDataset] subset_cache_write={cache_path} saved={len(samples)}")
#             except Exception:
#                 pass
#         if len(samples) == 0:
#             if (
#                 self.is_openvid
#                 and (not self._openvid_autofallback_done)
#                 and self.openvid_archive_root is None
#                 and self._failure_stats.get("openvid_filename_without_local_root", 0) > 0
#                 and self.openvid_snapshot_dir is not None
#             ):
#                 self._openvid_autofallback_done = True
#                 print(
#                     "[WanVideoDataset] openvid_record_streaming_detected_filename_only=1, "
#                     "auto_fallback_to_snapshot_download=1"
#                 )
#                 self.openvid_snapshot_download = True
#                 self.openvid_archive_root = self._prepare_openvid_archive_root_from_hf()
#                 if self.openvid_archive_root is not None:
#                     self._openvid_archive_index_built = False
#                     self._openvid_archive_map = {}
#                     self._openvid_archive_miss = set()
#                     return self._build_subset()
#             raise RuntimeError(
#                 f"预清洗后可用样本为0: dataset={self.dataset_name} scanned={scanned} "
#                 f"rejected={rejected} top=[{top_reject}] io=[{top_io}] "
#                 f"请检查OPENVID_SNAPSHOT_DIR中是否含Openvid_part*.zip/part*并开启OPENVID_AUTO_JOIN_PARTS=1"
#             )
#         return samples

#     @staticmethod
#     def _normalize_local_openvid_key(value):
#         if value is None:
#             return ""
#         out = str(value).strip().replace("\\", "/")
#         while out.startswith("./"):
#             out = out[2:]
#         out = out.lstrip("/")
#         return out.lower()

#     def _iter_local_openvid_files(self, video_root: Path):
#         if video_root is None:
#             return []
#         if not video_root.exists():
#             raise RuntimeError(
#                 f"local_openvid_video_root 不存在: {video_root}"
#             )
#         exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
#         files = []
#         for p in video_root.rglob("*"):
#             if not p.is_file():
#                 continue
#             if p.suffix.lower() in exts:
#                 files.append(p)
#         files.sort()
#         return files

#     def _load_local_openvid_caption_index(self, csv_path: Path, source_name: str):
#         if csv_path is None:
#             raise RuntimeError("local_openvid_csv_path 未设置")
#         if not csv_path.exists():
#             raise RuntimeError(f"local_openvid_csv_path 不存在: {csv_path}")
#         path_to_caption = {}
#         name_to_caption = {}
#         row_count = 0
#         drop_no_video = 0
#         drop_no_caption = 0
#         selected_video_col = None
#         selected_caption_col = None

#         def _keep_longer(mapping, key, caption):
#             old = mapping.get(key, None)
#             if old is None or len(caption) > len(old):
#                 mapping[key] = caption

#         with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
#             reader = csv.DictReader(f)
#             fieldnames = [str(x) for x in (reader.fieldnames or [])]
#             if not fieldnames:
#                 raise RuntimeError(f"CSV 无表头: {csv_path}")
#             lowered = {name.lower(): name for name in fieldnames}
#             for key in ("video", "video_path", "filename", "file", "path"):
#                 if key in lowered:
#                     selected_video_col = lowered[key]
#                     break
#             for key in ("caption", "text", "description", "prompt", "summary"):
#                 if key in lowered:
#                     selected_caption_col = lowered[key]
#                     break
#             if selected_video_col is None or selected_caption_col is None:
#                 raise RuntimeError(
#                     "CSV 缺少 video/caption 列。"
#                     f"当前列: {fieldnames}"
#                 )
#             for row in reader:
#                 row_count += 1
#                 video_val = str(row.get(selected_video_col, "") or "").strip()
#                 caption_val = str(row.get(selected_caption_col, "") or "").strip()
#                 if not video_val:
#                     drop_no_video += 1
#                     continue
#                 if not caption_val:
#                     drop_no_caption += 1
#                     continue
#                 norm_path = self._normalize_local_openvid_key(video_val)
#                 if norm_path:
#                     _keep_longer(path_to_caption, norm_path, caption_val)
#                 basename = self._normalize_local_openvid_key(Path(video_val).name)
#                 if basename:
#                     _keep_longer(name_to_caption, basename, caption_val)
#         print(
#             f"[WanVideoDataset][local_openvid][{source_name}] csv_index_done rows={row_count} "
#             f"video_col={selected_video_col} caption_col={selected_caption_col} "
#             f"path_keys={len(path_to_caption)} name_keys={len(name_to_caption)} "
#             f"drop_no_video={drop_no_video} drop_no_caption={drop_no_caption}"
#         )
#         return path_to_caption, name_to_caption

#     def _lookup_local_openvid_caption(
#         self,
#         video_path: Path,
#         video_root: Path,
#         path_to_caption,
#         name_to_caption,
#     ):
#         rel_path = str(video_path.relative_to(video_root)).replace("\\", "/")
#         rel_key = self._normalize_local_openvid_key(rel_path)
#         cap = path_to_caption.get(rel_key, None)
#         if cap:
#             return cap, rel_path, "rel_path"
#         name_key = self._normalize_local_openvid_key(video_path.name)
#         cap = name_to_caption.get(name_key, None)
#         if cap:
#             return cap, rel_path, "basename"
#         return None, rel_path, "missing"

#     def _build_local_openvid_subset(self, cache_path):
#         if self.hf_subset_use_cache and cache_path.exists():
#             try:
#                 with open(cache_path, "rb") as f:
#                     payload = pickle.load(f)
#                 self._subset_scanned = int(payload.get("scanned", 0))
#                 samples = payload.get("samples", [])
#                 self._subset_accepted = len(samples)
#                 rejected_stats = payload.get("rejected_stats", {})
#                 self._subset_rejected_stats = defaultdict(
#                     int, {str(k): int(v) for k, v in rejected_stats.items()}
#                 )
#                 print(f"[WanVideoDataset] local_openvid_subset_cache_hit={cache_path} loaded={len(samples)}")
#                 if len(samples) > 0:
#                     return samples
#             except Exception:
#                 pass

#         paired = []
#         source_pair_stats = []
#         max_missing_print = max(0, int(os.environ.get("WAN_LOCAL_MISSING_CAPTION_PRINT_MAX", "200")))
#         for src in self.local_openvid_sources:
#             src_name = str(src["name"])
#             src_video_root = src["video_root"]
#             src_csv_path = src["csv_path"]
#             src_limit = src["limit"]
#             video_files = self._iter_local_openvid_files(src_video_root)
#             path_to_caption, name_to_caption = self._load_local_openvid_caption_index(
#                 src_csv_path, src_name
#             )
#             source_paired = []
#             missing_caption = 0
#             match_by_rel = 0
#             match_by_name = 0
#             for vf in video_files:
#                 caption, rel_path, matched_by = self._lookup_local_openvid_caption(
#                     vf, src_video_root, path_to_caption, name_to_caption
#                 )
#                 if not caption:
#                     missing_caption += 1
#                     if missing_caption <= max_missing_print:
#                         print(
#                             f"[WanVideoDataset][local_openvid][{src_name}] "
#                             f"missing_caption_skip video={rel_path}"
#                         )
#                     continue
#                 if matched_by == "rel_path":
#                     match_by_rel += 1
#                 elif matched_by == "basename":
#                     match_by_name += 1
#                 source_paired.append(
#                     {
#                         "caption": caption,
#                         "video_spec": {"kind": "path", "value": str(vf)},
#                         "raw": {
#                             "video": rel_path,
#                             "matched_by": matched_by,
#                             "source_name": src_name,
#                         },
#                     }
#                 )
#             if missing_caption > max_missing_print:
#                 print(
#                     f"[WanVideoDataset][local_openvid][{src_name}] missing_caption_skip_more="
#                     f"{missing_caption - max_missing_print}"
#                 )
#             if src_limit and src_limit > 0 and len(source_paired) > src_limit:
#                 rng = random.Random(self.seed + (abs(hash(src_name)) % 10007))
#                 rng.shuffle(source_paired)
#                 source_paired = source_paired[:src_limit]
#             source_pair_stats.append(
#                 {
#                     "name": src_name,
#                     "local_videos": len(video_files),
#                     "paired": len(source_paired),
#                     "missing_caption": missing_caption,
#                     "matched_by_rel": match_by_rel,
#                     "matched_by_name": match_by_name,
#                     "limit": src_limit,
#                 }
#             )
#             paired.extend(source_paired)

#         rng = random.Random(self.seed)
#         rng.shuffle(paired)
#         if len(paired) == 0:
#             raise RuntimeError(
#                 "本地OpenVid(含HD)配对后样本数为0，请检查视频目录与CSV是否匹配。"
#                 f" sources={[(str(s['video_root']), str(s['csv_path'])) for s in self.local_openvid_sources]}"
#             )
#         for st in source_pair_stats:
#             print(
#                 f"[WanVideoDataset][local_openvid][{st['name']}] pair_done "
#                 f"local_videos={st['local_videos']} paired={st['paired']} "
#                 f"missing_caption={st['missing_caption']} matched_by_rel={st['matched_by_rel']} "
#                 f"matched_by_name={st['matched_by_name']} "
#                 f"target_limit={st['limit'] if st['limit'] else 'all'}"
#             )
#         print(f"[WanVideoDataset][local_openvid] pair_merged_total={len(paired)}")
#         if self.local_openvid_total_limit and len(paired) > self.local_openvid_total_limit:
#             paired = paired[: self.local_openvid_total_limit]
#             print(
#                 f"[WanVideoDataset][local_openvid] pair_merged_capped={len(paired)} "
#                 f"total_limit={self.local_openvid_total_limit}"
#             )

#         samples = []
#         rejected_stats = defaultdict(int)
#         scanned = 0
#         for parsed in paired:
#             scanned += 1
#             if self.preclean_enabled:
#                 ok, reject_reason = self._preclean_sample(parsed)
#                 if ok:
#                     samples.append(parsed)
#                 else:
#                     rejected_stats[reject_reason] += 1
#             else:
#                 samples.append(parsed)
#             if scanned % self.preclean_log_interval == 0:
#                 rejected = sum(rejected_stats.values())
#                 top_reject = ", ".join(
#                     f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:4]
#                 ) or "none"
#                 print(
#                     f"[WanVideoDataset][local_openvid] preclean_progress scanned={scanned} "
#                     f"accepted={len(samples)} rejected={rejected} top=[{top_reject}]"
#                 )
#         self._subset_scanned = scanned
#         self._subset_accepted = len(samples)
#         self._subset_rejected_stats = rejected_stats
#         rejected = sum(rejected_stats.values())
#         top_reject = ", ".join(
#             f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:8]
#         ) or "none"
#         print(
#             f"[WanVideoDataset][local_openvid] preclean_done scanned={scanned} "
#             f"accepted={len(samples)} rejected={rejected} top=[{top_reject}]"
#         )
#         if self.hf_subset_use_cache:
#             try:
#                 with open(cache_path, "wb") as f:
#                     pickle.dump(
#                         {
#                             "scanned": scanned,
#                             "samples": samples,
#                             "rejected_stats": dict(rejected_stats),
#                         },
#                         f,
#                     )
#                 print(f"[WanVideoDataset] local_openvid_subset_cache_write={cache_path} saved={len(samples)}")
#             except Exception:
#                 pass
#         if len(samples) == 0:
#             raise RuntimeError(
#                 "本地OpenVid(含HD)样本在预清洗后为0，请检查视频可解码性或放宽过滤阈值。"
#             )
#         return samples

#     def _preclean_sample(self, sample):
#         caption = sample["caption"]
#         token_ids = self.tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"]
#         if len(token_ids) > self.max_caption_tokens:
#             return False, "caption_too_long"
#         video_path = self._materialize_video(sample["video_spec"])
#         if video_path is None:
#             return False, "materialize_failed"
#         if not self._probe_video_quick(video_path):
#             try:
#                 p = Path(video_path)
#                 if self.local_video_cache_dir in p.parents:
#                     p.unlink(missing_ok=True)
#             except Exception:
#                 pass
#             return False, "probe_failed"
#         return True, "ok"

#     def _probe_video_quick(self, video_path):
#         try:
#             import cv2

#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 cap.release()
#                 return False
#             total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
#             if total <= 0 or total < self.frame_num:
#                 cap.release()
#                 return False
#             duration = (total / fps) if fps > 0 else None
#             if duration is not None and duration < self.min_duration_sec:
#                 cap.release()
#                 return False
#             if duration is not None and duration > self.max_duration_sec:
#                 cap.release()
#                 return False
#             ok, frame = cap.read()
#             cap.release()
#             if not ok or frame is None:
#                 return False
#             return True
#         except Exception:
#             return False

#     def _extract_row(self, row):
#         caption = self._extract_caption(row)
#         if caption is None:
#             return None
#         video_spec = self._extract_video_spec(row)
#         if video_spec is None:
#             return None
#         return {
#             "caption": caption,
#             "video_spec": video_spec,
#             "raw": row,
#         }

#     def _extract_caption(self, row):
#         keys = (
#             "caption",
#             "text",
#             "description",
#             "prompt",
#             "sentence",
#             "summary",
#         )
#         for k in keys:
#             v = row.get(k, None)
#             if isinstance(v, str) and v.strip():
#                 return v.strip()
#         return None

#     def _extract_video_spec(self, row):
#         keys = (
#             "video",
#             "mp4",
#             "file",
#             "video_path",
#             "path",
#             "url",
#             "video_name",
#             "filename",
#         )
#         candidates = []
#         for k in keys:
#             if k not in row:
#                 continue
#             value = row[k]
#             candidates.extend(self._collect_video_specs(value))
#         if not candidates:
#             return None
#         return max(candidates, key=self._video_spec_priority)

#     def _collect_video_specs(self, value, depth=0):
#         if depth > 4:
#             return []
#         out = []
#         spec = self._parse_video_value(value)
#         if spec is not None:
#             out.append(spec)
#         if isinstance(value, dict):
#             preferred_keys = (
#                 "bytes",
#                 "url",
#                 "download_url",
#                 "video_url",
#                 "href",
#                 "path",
#                 "file",
#                 "video",
#             )
#             for k in preferred_keys:
#                 if k in value:
#                     out.extend(self._collect_video_specs(value.get(k), depth + 1))
#             for k, v in value.items():
#                 if k in preferred_keys:
#                     continue
#                 out.extend(self._collect_video_specs(v, depth + 1))
#         elif isinstance(value, (list, tuple)):
#             for v in value:
#                 out.extend(self._collect_video_specs(v, depth + 1))
#         uniq = {}
#         for item in out:
#             kind = item.get("kind")
#             val = item.get("value")
#             if kind == "bytes":
#                 key = ("bytes", hashlib.sha1(bytes(val)).hexdigest())
#             else:
#                 key = (kind, str(val))
#             uniq[key] = item
#         return list(uniq.values())

#     def _video_spec_priority(self, spec):
#         kind = spec.get("kind")
#         value = spec.get("value")
#         if kind == "bytes":
#             return 300
#         if kind == "url":
#             return 200
#         if kind == "path":
#             try:
#                 p = Path(value)
#                 if p.exists():
#                     return 100
#             except Exception:
#                 pass
#             return 10
#         return 0

#     def _parse_video_value(self, value):
#         if value is None:
#             return None
#         if isinstance(value, str):
#             value = value.strip()
#             if not value:
#                 return None
#             if value.startswith("http://") or value.startswith("https://"):
#                 return {"kind": "url", "value": value}
#             if value.startswith("hf://"):
#                 return {"kind": "url", "value": value}
#             return {"kind": "path", "value": value}
#         if isinstance(value, (bytes, bytearray)):
#             return {"kind": "bytes", "value": value}
#         return None

#     def _normalize_rel_path(self, value: str):
#         path = str(value).strip().replace("\\", "/")
#         if path.startswith("hf://"):
#             return path
#         while path.startswith("./"):
#             path = path[2:]
#         return path.lstrip("/")

#     def _candidate_urls_from_missing_path(self, value: str):
#         path = self._normalize_rel_path(value)
#         if path.startswith("http://") or path.startswith("https://"):
#             return [path]
#         if path.startswith("hf://datasets/"):
#             return [
#                 path.replace(
#                     f"hf://datasets/{self.dataset_name}/",
#                     f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/",
#                 )
#             ]
#         if self.is_openvid and (not self.openvid_allow_http_guess):
#             return []
#         ds_name = self.dataset_name
#         split = str(self.hf_split).strip("/")
#         candidates = [
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/data/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/videos/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/{split}/{path}",
#             f"https://huggingface.co/datasets/{ds_name}/resolve/main/files/{path}",
#         ]
#         uniq = []
#         seen = set()
#         for u in candidates:
#             if u in seen:
#                 continue
#             seen.add(u)
#             uniq.append(u)
#         return uniq

#     def _resolve_openvid_local_path(self, value: str):
#         normalized = self._normalize_rel_path(value)
#         if normalized.startswith("http://") or normalized.startswith("https://"):
#             return None
#         if normalized.startswith("hf://"):
#             return None
#         if self.openvid_video_root:
#             p = self.openvid_video_root / normalized
#             if p.exists():
#                 return p
#             try:
#                 p2 = self.openvid_video_root / Path(normalized).name
#                 if p2.exists():
#                     return p2
#             except Exception:
#                 pass
#         return self._extract_openvid_from_archives(normalized)

#     def _prepare_openvid_archive_root_from_hf(self):
#         try:
#             print(
#                 f"[WanVideoDataset] openvid_snapshot_download_start "
#                 f"repo={self.dataset_name} local_dir={self.openvid_snapshot_dir} "
#                 f"patterns={self.openvid_snapshot_patterns}"
#             )
#             path = snapshot_download(
#                 repo_id=self.dataset_name,
#                 repo_type="dataset",
#                 local_dir=str(self.openvid_snapshot_dir),
#                 allow_patterns=self.openvid_snapshot_patterns,
#             )
#             print(f"[WanVideoDataset] openvid_snapshot_download_done local_dir={path}")
#             return Path(path)
#         except Exception:
#             self._failure_stats["openvid_snapshot_download_error"] += 1
#             return None

#     def _iter_openvid_archive_files(self):
#         if not self.openvid_archive_root or (not self.openvid_archive_root.exists()):
#             return []
#         archives = sorted(self.openvid_archive_root.rglob("*.zip"))
#         if self.openvid_auto_join_parts:
#             archives.extend(self._join_openvid_part_archives())
#             archives = sorted(set(archives))
#         if self._openvid_archive_max_scan > 0:
#             archives = archives[: self._openvid_archive_max_scan]
#         return archives

#     def _join_openvid_part_archives(self):
#         part_files = sorted(self.openvid_archive_root.rglob("*.part*"))
#         groups = defaultdict(list)
#         for part in part_files:
#             name = part.name
#             if ".part" not in name:
#                 continue
#             prefix, suffix = name.split(".part", 1)
#             if not suffix:
#                 continue
#             groups[prefix].append((suffix, part))
#         joined_archives = []
#         for prefix, items in groups.items():
#             items = sorted(items, key=lambda x: x[0])
#             if len(items) < 2:
#                 continue
#             out_zip = self.openvid_joined_archive_dir / f"{prefix}.zip"
#             if not out_zip.exists():
#                 lock_path = Path(f"{out_zip}.lock")
#                 if not self._acquire_lock(lock_path, timeout_sec=max(self.lock_timeout_sec, 120)):
#                     continue
#                 try:
#                     if not out_zip.exists():
#                         tmp_path = Path(
#                             f"{out_zip}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
#                         )
#                         try:
#                             with open(tmp_path, "wb") as wf:
#                                 for _, part_path in items:
#                                     with open(part_path, "rb") as rf:
#                                         while True:
#                                             chunk = rf.read(1024 * 1024)
#                                             if not chunk:
#                                                 break
#                                             wf.write(chunk)
#                             os.replace(tmp_path, out_zip)
#                             print(
#                                 f"[WanVideoDataset] openvid_join_parts_done prefix={prefix} parts={len(items)} out={out_zip}"
#                             )
#                         except Exception:
#                             self._failure_stats["openvid_join_parts_error"] += 1
#                             try:
#                                 tmp_path.unlink(missing_ok=True)
#                             except Exception:
#                                 pass
#                 finally:
#                     self._release_lock(lock_path)
#             if out_zip.exists():
#                 joined_archives.append(out_zip)
#         return joined_archives

#     def _build_openvid_archive_index(self):
#         if self._openvid_archive_index_built:
#             return
#         self._openvid_archive_index_built = True
#         archives = self._iter_openvid_archive_files()
#         if not archives:
#             return
#         indexed = 0
#         for i, archive in enumerate(archives):
#             try:
#                 with zipfile.ZipFile(archive, "r") as zf:
#                     for name in zf.namelist():
#                         if not name.lower().endswith(".mp4"):
#                             continue
#                         base = Path(name).name
#                         if base not in self._openvid_archive_map:
#                             self._openvid_archive_map[base] = (archive, name)
#                             indexed += 1
#             except Exception:
#                 self._failure_stats["openvid_archive_read_error"] += 1
#             if (i + 1) % 5 == 0:
#                 print(
#                     f"[WanVideoDataset] openvid_archive_index_progress scanned_archives={i + 1} "
#                     f"indexed_videos={indexed}"
#                 )
#         print(
#             f"[WanVideoDataset] openvid_archive_index_done archives={len(archives)} "
#             f"indexed_videos={indexed}"
#         )

#     def _extract_openvid_from_archives(self, normalized: str):
#         if not self.openvid_archive_root:
#             return None
#         file_name = Path(normalized).name
#         if (not file_name) or (not file_name.lower().endswith(".mp4")):
#             return None
#         if file_name in self._openvid_archive_miss:
#             return None
#         target_path = self.openvid_extracted_cache_dir / file_name
#         if target_path.exists() and self._is_cache_file_valid(target_path):
#             return target_path
#         self._build_openvid_archive_index()
#         arc = self._openvid_archive_map.get(file_name, None)
#         if arc is None:
#             self._openvid_archive_miss.add(file_name)
#             self._failure_stats["openvid_archive_member_missing"] += 1
#             return None
#         archive_path, member_name = arc
#         lock_path = Path(f"{target_path}.lock")
#         if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
#             self._failure_stats["lock_timeout"] += 1
#             if target_path.exists() and self._is_cache_file_valid(target_path):
#                 return target_path
#             return None
#         try:
#             if target_path.exists() and self._is_cache_file_valid(target_path):
#                 return target_path
#             tmp_path = Path(
#                 f"{target_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
#             )
#             try:
#                 with zipfile.ZipFile(archive_path, "r") as zf:
#                     with zf.open(member_name, "r") as src, open(tmp_path, "wb") as dst:
#                         while True:
#                             chunk = src.read(1024 * 1024)
#                             if not chunk:
#                                 break
#                             dst.write(chunk)
#                 os.replace(tmp_path, target_path)
#             except Exception:
#                 self._failure_stats["openvid_archive_extract_error"] += 1
#                 try:
#                     tmp_path.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 return None
#             if not self._is_cache_file_valid(target_path):
#                 self._failure_stats["openvid_archive_extracted_invalid"] += 1
#                 try:
#                     target_path.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 return None
#             self._failure_stats["openvid_archive_extract_hit"] += 1
#             return target_path
#         finally:
#             self._release_lock(lock_path)

#     def _cached_file_path(self, key: str, suffix=".mp4"):
#         file_name = hashlib.sha1(key.encode("utf-8")).hexdigest() + suffix
#         return self.local_video_cache_dir / file_name

#     def _build_http_session(self):
#         session = requests.Session()
#         try:
#             from urllib3.util.retry import Retry

#             retry = Retry(
#                 total=self.http_retry_total,
#                 connect=self.http_retry_total,
#                 read=self.http_retry_total,
#                 backoff_factor=0.3,
#                 status_forcelist=(429, 500, 502, 503, 504),
#                 allowed_methods=frozenset(["GET"]),
#                 raise_on_status=False,
#             )
#             adapter = HTTPAdapter(max_retries=retry)
#             session.mount("http://", adapter)
#             session.mount("https://", adapter)
#         except Exception:
#             pass
#         return session

#     def _acquire_lock(self, lock_path: Path, timeout_sec: float = 120.0):
#         deadline = time.time() + timeout_sec
#         while time.time() < deadline:
#             try:
#                 fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
#                 with os.fdopen(fd, "w", encoding="utf-8") as f:
#                     f.write(f"{os.getpid()} {time.time()}")
#                 return True
#             except FileExistsError:
#                 try:
#                     age = time.time() - lock_path.stat().st_mtime
#                     if age > 300:
#                         lock_path.unlink(missing_ok=True)
#                         continue
#                 except Exception:
#                     pass
#                 time.sleep(0.2)
#             except Exception:
#                 time.sleep(0.2)
#         return False

#     def _release_lock(self, lock_path: Path):
#         try:
#             lock_path.unlink(missing_ok=True)
#         except Exception:
#             pass

#     def _is_cache_file_valid(self, file_path: Path):
#         if not file_path.exists():
#             return False
#         try:
#             if file_path.stat().st_size < 4096:
#                 return False
#         except Exception:
#             return False
#         try:
#             import cv2

#             cap = cv2.VideoCapture(str(file_path))
#             if not cap.isOpened():
#                 cap.release()
#                 return False
#             total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             cap.release()
#             return total > 0
#         except Exception:
#             return False

#     def _download_url_to_file(self, url: str, target_path: Path, headers):
#         tmp_path = Path(f"{target_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}")
#         session = self._build_http_session()
#         try:
#             with session.get(url, stream=True, timeout=self.http_timeout_sec, headers=headers) as r:
#                 status = int(getattr(r, "status_code", 0) or 0)
#                 if status >= 400:
#                     return False, f"http_{status}"
#                 r.raise_for_status()
#                 bytes_written = 0
#                 with open(tmp_path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=1024 * 1024):
#                         if not chunk:
#                             continue
#                         f.write(chunk)
#                         bytes_written += len(chunk)
#             if bytes_written < 4096:
#                 try:
#                     tmp_path.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 return False, "download_too_small"
#             os.replace(tmp_path, target_path)
#             return True, "ok"
#         except requests.exceptions.Timeout:
#             return False, "timeout"
#         except requests.exceptions.HTTPError:
#             return False, "http_error"
#         except Exception:
#             return False, "download_exception"
#         finally:
#             try:
#                 tmp_path.unlink(missing_ok=True)
#             except Exception:
#                 pass
#             try:
#                 session.close()
#             except Exception:
#                 pass

#     def _materialize_video(self, video_spec):
#         kind = video_spec["kind"]
#         value = video_spec["value"]
#         if kind == "path":
#             if isinstance(value, str) and (
#                 value.startswith("http://") or value.startswith("https://")
#             ):
#                 return self._materialize_video({"kind": "url", "value": value})
#             if self.is_openvid:
#                 local_openvid = self._resolve_openvid_local_path(str(value))
#                 if local_openvid is not None:
#                     if not self._is_cache_file_valid(local_openvid):
#                         self._failure_stats["openvid_local_invalid"] += 1
#                         return None
#                     return str(local_openvid)
#             p = Path(value)
#             if p.exists():
#                 if not self._is_cache_file_valid(p):
#                     self._failure_stats["path_invalid"] += 1
#                     return None
#                 return str(p)
#             if self.is_openvid:
#                 normalized = self._normalize_rel_path(str(value))
#                 if normalized.lower().endswith(".mp4"):
#                     if (self.openvid_video_root is None) and (self.openvid_archive_root is None):
#                         self._failure_stats["openvid_filename_without_local_root"] += 1
#                     else:
#                         self._failure_stats["openvid_local_missing"] += 1
#                     return None
#             candidate_urls = self._candidate_urls_from_missing_path(str(value))
#             if self.quick_fail:
#                 candidate_urls = candidate_urls[: self.url_fallback_limit]
#             for url in candidate_urls:
#                 url_video = self._materialize_video({"kind": "url", "value": url})
#                 if url_video is not None:
#                     self._failure_stats["path_to_url_fallback_hit"] += 1
#                     return url_video
#             self._failure_stats["path_missing"] += 1
#             return None
#         if kind == "url":
#             local_path = self._cached_file_path(value, suffix=".mp4")
#             lock_path = Path(f"{local_path}.lock")
#             if local_path.exists() and self._is_cache_file_valid(local_path):
#                 return str(local_path)
#             if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
#                 self._failure_stats["lock_timeout"] += 1
#                 if local_path.exists() and self._is_cache_file_valid(local_path):
#                     return str(local_path)
#                 return None
#             try:
#                 headers = None
#                 hf_token = os.environ.get("HF_TOKEN", "").strip()
#                 if hf_token and "huggingface.co" in value:
#                     headers = {"Authorization": f"Bearer {hf_token}"}
#                 if "huggingface.co" in value and (not hf_token) and (not self._warned_hf_token):
#                     print("[WanVideoDataset] warning: huggingface url without HF_TOKEN")
#                     self._warned_hf_token = True
#                 if local_path.exists() and not self._is_cache_file_valid(local_path):
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                 if local_path.exists() and self._is_cache_file_valid(local_path):
#                     return str(local_path)
#                 ok, reason = self._download_url_to_file(value, local_path, headers=headers)
#                 if not ok:
#                     self._failure_stats[reason] += 1
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 if not self._is_cache_file_valid(local_path):
#                     self._failure_stats["cache_invalid_after_download"] += 1
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 return str(local_path)
#             finally:
#                 self._release_lock(lock_path)
#         if kind == "bytes":
#             if not isinstance(value, (bytes, bytearray)):
#                 self._failure_stats["bytes_invalid_type"] += 1
#                 return None
#             key = hashlib.sha1(value).hexdigest()
#             local_path = self._cached_file_path(key, suffix=".mp4")
#             lock_path = Path(f"{local_path}.lock")
#             if local_path.exists() and self._is_cache_file_valid(local_path):
#                 return str(local_path)
#             if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
#                 self._failure_stats["lock_timeout"] += 1
#                 if local_path.exists() and self._is_cache_file_valid(local_path):
#                     return str(local_path)
#                 return None
#             try:
#                 if local_path.exists() and not self._is_cache_file_valid(local_path):
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                 try:
#                     tmp_path = Path(
#                         f"{local_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
#                     )
#                     with open(tmp_path, "wb") as f:
#                         f.write(bytes(value))
#                     os.replace(tmp_path, local_path)
#                 except Exception:
#                     self._failure_stats["bytes_write_failed"] += 1
#                     try:
#                         tmp_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 if not self._is_cache_file_valid(local_path):
#                     self._failure_stats["bytes_cache_invalid"] += 1
#                     try:
#                         local_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass
#                     return None
#                 return str(local_path)
#             finally:
#                 self._release_lock(lock_path)
#         return None

#     def _load_video_frames(self, video_path):
#         import cv2

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return None, None, None
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
#         if total <= 0:
#             cap.release()
#             return None, None, None
#         if total < self.frame_num:
#             cap.release()
#             return None, None, None
#         duration = (total / fps) if fps > 0 else None
#         if duration is not None and duration < self.min_duration_sec:
#             cap.release()
#             return None, None, None
#         if duration is not None and duration > self.max_duration_sec:
#             cap.release()
#             return None, None, None

#         # indices = np.linspace(0, total - 1, self.frame_num, dtype=int)
#         frames = []
#         # for idx in indices:
#             # cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
#         # 训练取帧策略：
#         # 旧逻辑：全视频等距抽帧（np.linspace）
#         # 新逻辑：从开头连续读取前 frame_num 帧（更贴近“前缀片段学习”）
#         for _ in range(self.frame_num):
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#         cap.release()
#         if len(frames) < self.frame_num:
#             return None, None, None
#         return frames, fps, duration

#     def _process_frames(self, frames):
#         import cv2

#         h, w = frames[0].shape[:2]
#         area = h * w
#         if area > self.max_area:
#             scale = math.sqrt(self.max_area / area)
#             h = int(h * scale)
#             w = int(w * scale)
#         h = max(32, (h // 32) * 32)
#         w = max(32, (w // 32) * 32)
#         resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
#         tensor = torch.stack([torch.from_numpy(f) for f in resized]).float() / 127.5 - 1.0
#         tensor = tensor.permute(3, 0, 1, 2)
#         return tensor, resized[0]

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         n = len(self.samples)
#         default_trials = min(max(60, n // 4), n)
#         if self.quick_fail:
#             max_trials = min(default_trials, self.max_trials_cap, n)
#             if n >= 20:
#                 max_trials = max(20, max_trials)
#         else:
#             max_trials = default_trials
#         if max_trials <= 0:
#             raise RuntimeError("数据集为空，无法取样")
#         trial_stats = defaultdict(int)
#         for trial in range(max_trials):
#             if trial < 10:
#                 sample_idx = (idx + trial) % n
#             else:
#                 sample_idx = random.randint(0, n - 1)
#             sample = self.samples[sample_idx]
#             caption = sample["caption"]
#             token_ids = self.tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"]
#             if len(token_ids) > self.max_caption_tokens:
#                 trial_stats["caption_too_long"] += 1
#                 continue

#             video_path = self._materialize_video(sample["video_spec"])
#             if video_path is None:
#                 trial_stats["materialize_failed"] += 1
#                 if (trial + 1) % self.trial_log_interval == 0:
#                     trial_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(trial_stats.items())
#                     ) or "none"
#                     io_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(self._failure_stats.items())
#                     ) or "none"
#                     print(
#                         f"[WanVideoDataset] retry_progress idx={idx} tried={trial + 1}/{max_trials} "
#                         f"trial_stats=[{trial_detail}] io_stats=[{io_detail}]"
#                     )
#                 continue
#             frames, _, _ = self._load_video_frames(video_path)
#             if frames is None:
#                 trial_stats["decode_failed"] += 1
#                 try:
#                     p = Path(video_path)
#                     if self.local_video_cache_dir in p.parents:
#                         p.unlink(missing_ok=True)
#                 except Exception:
#                     pass
#                 if (trial + 1) % self.trial_log_interval == 0:
#                     trial_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(trial_stats.items())
#                     ) or "none"
#                     io_detail = ", ".join(
#                         f"{k}={v}" for k, v in sorted(self._failure_stats.items())
#                     ) or "none"
#                     print(
#                         f"[WanVideoDataset] retry_progress idx={idx} tried={trial + 1}/{max_trials} "
#                         f"trial_stats=[{trial_detail}] io_stats=[{io_detail}]"
#                     )
#                 continue
#             video_tensor, first_frame_np = self._process_frames(frames)
#             ref_image = Image.fromarray(first_frame_np)
#             mq_ref_image = ref_image
#             out_caption = caption
#             if random.random() < self.null_caption_prob:
#                 out_caption = ""
#             if random.random() < self.null_image_prob:
#                 mq_ref_image = None
#             if not self._printed_sample_info:
#                 print(
#                     f"[WanVideoDataset] sample_once "
#                     f"video_tensor={tuple(video_tensor.shape)} "
#                     f"caption_tokens={len(token_ids)} "
#                     f"mq_ref_is_none={mq_ref_image is None} "
#                     f"video_path={video_path}"
#                 )
#                 self._printed_sample_info = True
#             result = {
#                 "caption": out_caption,
#                 "video": video_tensor,
#                 "ref_image": ref_image,
#                 "mq_ref_image": mq_ref_image,
#                 "video_path": video_path,
#             }
#             self._last_good_sample = result
#             return result
#         if self._last_good_sample is not None:
#             return self._last_good_sample
#         trial_detail = ", ".join(f"{k}={v}" for k, v in sorted(trial_stats.items())) or "none"
#         io_detail = ", ".join(f"{k}={v}" for k, v in sorted(self._failure_stats.items())) or "none"
#         hf_token_ready = bool(os.environ.get("HF_TOKEN", "").strip())
#         raise RuntimeError(
#             f"样本解码失败，trials={max_trials}，dataset={self.dataset_name}，"
#             f"trial_stats=[{trial_detail}] io_stats=[{io_detail}] "
#             f"streaming={self.hf_streaming} hf_token={'set' if hf_token_ready else 'unset'} "
#             f"openvid_root={str(self.openvid_video_root) if self.openvid_video_root else 'unset'} "
#             f"openvid_archive_root={str(self.openvid_archive_root) if self.openvid_archive_root else 'unset'}，"
#             f"建议设置HF_TOKEN并优先使用--hf_no_streaming；OpenVid请配置OPENVID_VIDEO_ROOT或OPENVID_ARCHIVE_ROOT，"
#             f"或开启OPENVID_SNAPSHOT_DOWNLOAD=1自动从HF拉取归档"
#         )

#     @staticmethod
#     def collate_fn(batch):
#         return {k: [item[k] for item in batch] for k in batch[0]}




















# 下面这个是让mq的rms适配t5的情况：
import hashlib
import math
import os
import pickle
import random
import time
import zipfile
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from huggingface_hub import snapshot_download
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset
from transformers import AutoTokenizer


DEFAULT_STAGE_DATASET = {
    "stage1": "nkp37/OpenVid-1M",
    "stage2": "BestWishYsh/OpenS2V-5M",
}

DEFAULT_STAGE_TOTAL = {
    "nkp37/OpenVid-1M": 1_000_000,
    "BestWishYsh/OpenS2V-5M": 5_000_000,
}


class MetaQueryEncoderForWan(nn.Module):
    WAN_TEXT_DIM = 4096

    def __init__(
        self,
        qwen3vl_model_id: str,
        num_metaqueries: int = 256,
        connector_num_hidden_layers: int = 24,
        gradient_checkpointing: bool = False,
        train_input_embeddings: bool = True,
        connector_norm_init_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_metaqueries = num_metaqueries
        self.wan_text_dim = self.WAN_TEXT_DIM
        self.dtype = dtype
        self.device = torch.device(device)
        self.train_input_embeddings = bool(train_input_embeddings)
        self.connector_norm_init_scale = float(connector_norm_init_scale)
        self._printed_forward_stats = False

        from diffusers.models.normalization import RMSNorm
        from models.model import MLLMInContext, MLLMInContextConfig
        from models.transformer_encoder import Qwen2Encoder
        from transformers import Qwen2Config

        # 关键点：
        # 1) diffusion_model_id 设为 "none" -> 不会加载 Sana/SD 的扩散骨干；
        # 2) connector_out_dim_override 直接指定为 4096，与 Wan text_dim 对齐。
        config = MLLMInContextConfig(
            mllm_id=qwen3vl_model_id,
            diffusion_model_id="none",
            connector_out_dim_override=self.wan_text_dim,
            num_metaqueries=num_metaqueries,
            _gradient_checkpointing=gradient_checkpointing,
            connector_num_hidden_layers=connector_num_hidden_layers,
        )

        mllm_model = MLLMInContext(config).to(device=self.device, dtype=dtype)
        self.mllm_model = mllm_model
        self.tokenizer = mllm_model.get_tokenizer()
        self.tokenize = mllm_model.get_tokenize_fn()
        # 开启梯度检查点时关闭 KV cache，减少显存并避免反复告警。
        try:
            if hasattr(self.mllm_model.mllm_backbone, "config"):
                self.mllm_model.mllm_backbone.config.use_cache = False
            if hasattr(self.mllm_model.mllm_backbone, "generation_config"):
                self.mllm_model.mllm_backbone.generation_config.use_cache = False
        except Exception:
            pass
        print(
            f"[MetaQueryEncoderForWan] mllm_type={self.mllm_model.mllm_type} "
            f"transformer_loaded={self.mllm_model.transformer is not None}"
        )

        mllm_hidden_size = mllm_model.mllm_hidden_size
        print(
            f"[MetaQueryEncoderForWan] mllm_hidden={mllm_hidden_size} "
            f"target_wan_text_dim={self.wan_text_dim}"
        )
        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=mllm_hidden_size,
                intermediate_size=mllm_hidden_size * 4,
                num_hidden_layers=connector_num_hidden_layers,
                num_attention_heads=mllm_hidden_size // 64,
                num_key_value_heads=mllm_hidden_size // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            )
        )
        # 兼容自定义 Qwen2Encoder 的梯度检查点调用。
        if hasattr(encoder, "gradient_checkpointing"):
            encoder.gradient_checkpointing = bool(gradient_checkpointing)
        if gradient_checkpointing and not hasattr(encoder, "_gradient_checkpointing_func"):
            encoder._gradient_checkpointing_func = (
                lambda func, *gc_args: torch.utils.checkpoint.checkpoint(
                    func, *gc_args, use_reentrant=False
                )
            )
        norm = RMSNorm(self.wan_text_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            # 对齐 diffusion_model_id=none 的默认尺度（1.0）。
            # 历史上的 sqrt(5.5) 会把 MQ RMS 推高到约 2.34，容易与 Wan T5 条件尺度失配。
            norm.weight.fill_(float(self.connector_norm_init_scale))
        new_connector = nn.Sequential(
            encoder,
            nn.Linear(mllm_hidden_size, self.wan_text_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.wan_text_dim, self.wan_text_dim),
            norm,
        ).to(device=self.device, dtype=dtype)
        self.mllm_model.connector = new_connector
        self.mllm_model.connector_out_dim = self.wan_text_dim
        print(
            f"[MetaQueryEncoderForWan] connector_out={self.mllm_model.connector_out_dim} "
            f"num_metaqueries={self.num_metaqueries} "
            f"connector_norm_init_scale={self.connector_norm_init_scale}"
        )

        self.mllm_model.mllm_backbone.requires_grad_(False)
        self.mllm_model.connector.requires_grad_(True)
        self.mllm_model.mllm_backbone.get_input_embeddings().requires_grad_(self.train_input_embeddings)
        print(
            f"[MetaQueryEncoderForWan] train_connector=True "
            f"train_input_embeddings={self.train_input_embeddings}"
        )

        if hasattr(self.mllm_model, "transformer"):
            del self.mllm_model.transformer
            self.mllm_model.transformer = None
        torch.cuda.empty_cache()

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, captions, input_images=None):
        if input_images is not None:
            input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
                self.tokenizer, captions, input_images
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device, self.dtype)
                if self.mllm_model.mllm_type in ("qwenvl", "qwen3vl"):
                    pixel_values = pixel_values.squeeze(0)
            if image_sizes is not None:
                image_sizes = image_sizes.to(self.device)
        else:
            input_ids, attention_mask = self.tokenize(self.tokenizer, captions)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values = None
            image_sizes = None

        mq_features, _ = self.mllm_model.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        if not self._printed_forward_stats:
            print(
                f"[MetaQueryEncoderForWan] forward_once "
                f"input_ids={tuple(input_ids.shape)} "
                f"attention_mask={tuple(attention_mask.shape)} "
                f"mq_features={tuple(mq_features.shape)} "
                f"dtype={mq_features.dtype}"
            )
            self._printed_forward_stats = True
        return mq_features


def resolve_hf_stage_config(stage: str, dataset_name: str | None, subset_ratio: float):
    stage_name = stage.lower()
    ds = dataset_name or DEFAULT_STAGE_DATASET.get(stage_name, DEFAULT_STAGE_DATASET["stage1"])
    total_hint = DEFAULT_STAGE_TOTAL.get(ds, None)
    subset_size = int(total_hint * subset_ratio) if total_hint is not None else None
    return ds, subset_size, total_hint


class WanVideoDataset(Dataset):
    def __init__(
        self,
        frame_num: int = 81,
        max_area: int = 720 * 1280,
        null_caption_prob: float = 0.0,
        null_image_prob: float = 0.5,
        max_caption_tokens: int = 512,
        caption_tokenizer_path: str = "google/umt5-xxl",
        min_duration_sec: float = 0.5,
        max_duration_sec: float = 20.0,
        hf_stage: str = "stage1",
        hf_dataset_name: str | None = None,
        hf_split: str = "train",
        hf_subset_ratio: float = 0.01,
        hf_subset_size: int | None = None,
        hf_scan_factor: int = 30,
        hf_subset_cache_dir: str | None = None,
        hf_subset_use_cache: bool = True,
        hf_cache_dir: str | None = None,
        hf_streaming: bool = True,
        hf_shuffle_buffer: int = 10000,
        seed: int = 42,
        local_video_cache_dir: str | None = None,
        local_openvid_video_root: str | None = None,
        local_openvid_csv_path: str | None = None,
        local_openvid_limit: int | None = None,
        local_openvid_hd_video_root: str | None = None,
        local_openvid_hd_csv_path: str | None = None,
        local_openvid_hd_limit: int | None = None,
    ):
        self.frame_num = frame_num
        self.max_area = max_area
        self.null_caption_prob = null_caption_prob
        self.null_image_prob = null_image_prob
        self.max_caption_tokens = max_caption_tokens
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.hf_split = hf_split
        self.hf_cache_dir = hf_cache_dir
        self.hf_streaming = hf_streaming
        self.hf_shuffle_buffer = hf_shuffle_buffer
        self.seed = seed
        self.scan_factor = max(5, hf_scan_factor)
        self._printed_sample_info = False
        self._last_good_sample = None
        self._failure_stats = defaultdict(int)
        self._warned_hf_token = False
        self._subset_scanned = 0
        self._subset_accepted = 0
        self._subset_rejected_stats = defaultdict(int)
        self.quick_fail = os.environ.get("WAN_DATA_QUICK_FAIL", "1").strip().lower() not in (
            "0",
            "false",
            "off",
        )
        self.max_trials_cap = max(10, int(os.environ.get("WAN_DATA_MAX_TRIALS", "400")))
        self.trial_log_interval = max(
            1, int(os.environ.get("WAN_DATA_TRIAL_LOG_INTERVAL", "100"))
        )
        self.url_fallback_limit = max(
            1, int(os.environ.get("WAN_DATA_PATH_URL_FALLBACK_LIMIT", "2"))
        )
        self.http_retry_total = max(0, int(os.environ.get("WAN_DATA_HTTP_RETRY_TOTAL", "1")))
        self.http_timeout_sec = max(3, int(os.environ.get("WAN_DATA_HTTP_TIMEOUT_SEC", "12")))
        self.lock_timeout_sec = max(3, int(os.environ.get("WAN_DATA_LOCK_TIMEOUT_SEC", "20")))
        self.preclean_enabled = os.environ.get("WAN_DATA_PRECLEAN", "1").strip().lower() not in (
            "0",
            "false",
            "off",
        )
        self.preclean_log_interval = max(
            1, int(os.environ.get("WAN_DATA_PRECLEAN_LOG_INTERVAL", "200"))
        )
        self.preclean_scan_multiplier = max(
            1, int(os.environ.get("WAN_DATA_PRECLEAN_SCAN_MULTIPLIER", "1"))
        )
        self.preclean_scan_cap = max(
            1000, int(os.environ.get("WAN_DATA_PRECLEAN_SCAN_CAP", "200000"))
        )
        self.preclean_zero_accept_abort_scan = max(
            1000, int(os.environ.get("WAN_DATA_PRECLEAN_ZERO_ACCEPT_ABORT_SCAN", "20000"))
        )
        self.hf_subset_ratio = hf_subset_ratio
        self.hf_subset_size = hf_subset_size
        self.hf_subset_use_cache = hf_subset_use_cache
        self.hf_subset_cache_dir = Path(
            hf_subset_cache_dir or (Path(hf_cache_dir or ".hf_cache") / "subset_cache")
        )
        self.hf_subset_cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_video_cache_dir = Path(
            local_video_cache_dir or (Path(hf_cache_dir or ".hf_cache") / "video_cache")
        )
        self.local_video_cache_dir.mkdir(parents=True, exist_ok=True)
        if local_openvid_video_root is None:
            env_video_root = os.environ.get("OPENVID_LOCAL_VIDEO_ROOT", "").strip()
            local_openvid_video_root = env_video_root or None
        if local_openvid_csv_path is None:
            env_csv_path = os.environ.get("OPENVID_LOCAL_CSV_PATH", "").strip()
            local_openvid_csv_path = env_csv_path or None
        if local_openvid_limit is None:
            env_limit = os.environ.get("OPENVID_LOCAL_LIMIT", "").strip()
            if env_limit:
                try:
                    local_openvid_limit = int(env_limit)
                except Exception:
                    local_openvid_limit = None
        if local_openvid_hd_video_root is None:
            env_hd_video_root = os.environ.get("OPENVID_HD_LOCAL_VIDEO_ROOT", "").strip()
            local_openvid_hd_video_root = env_hd_video_root or None
        if local_openvid_hd_csv_path is None:
            env_hd_csv_path = os.environ.get("OPENVID_HD_LOCAL_CSV_PATH", "").strip()
            local_openvid_hd_csv_path = env_hd_csv_path or None
        if local_openvid_hd_limit is None:
            env_hd_limit = os.environ.get("OPENVID_HD_LOCAL_LIMIT", "").strip()
            if env_hd_limit:
                try:
                    local_openvid_hd_limit = int(env_hd_limit)
                except Exception:
                    local_openvid_hd_limit = None
        env_total_limit = os.environ.get("OPENVID_LOCAL_TOTAL_LIMIT", "").strip()
        local_openvid_total_limit = None
        if env_total_limit:
            try:
                local_openvid_total_limit = int(env_total_limit)
            except Exception:
                local_openvid_total_limit = None

        def _to_path(v):
            return Path(v).expanduser().resolve() if v else None

        def _to_limit(v):
            if v is None:
                return None
            try:
                iv = int(v)
            except Exception:
                return None
            return iv if iv > 0 else None

        self.local_openvid_video_root = _to_path(local_openvid_video_root)
        self.local_openvid_csv_path = _to_path(local_openvid_csv_path)
        self.local_openvid_limit = _to_limit(local_openvid_limit)
        self.local_openvid_hd_video_root = _to_path(local_openvid_hd_video_root)
        self.local_openvid_hd_csv_path = _to_path(local_openvid_hd_csv_path)
        self.local_openvid_hd_limit = _to_limit(local_openvid_hd_limit)
        self.local_openvid_total_limit = _to_limit(local_openvid_total_limit)

        self.local_openvid_sources = []
        if self.local_openvid_video_root is not None and self.local_openvid_csv_path is not None:
            self.local_openvid_sources.append(
                {
                    "name": "openvid",
                    "video_root": self.local_openvid_video_root,
                    "csv_path": self.local_openvid_csv_path,
                    "limit": self.local_openvid_limit,
                }
            )
        elif self.local_openvid_video_root is not None or self.local_openvid_csv_path is not None:
            print(
                "[WanVideoDataset] warning: openvid 普通源参数不完整，"
                "需同时提供 local_openvid_video_root 与 local_openvid_csv_path，已忽略该源"
            )

        if self.local_openvid_hd_video_root is not None and self.local_openvid_hd_csv_path is not None:
            self.local_openvid_sources.append(
                {
                    "name": "openvid_hd",
                    "video_root": self.local_openvid_hd_video_root,
                    "csv_path": self.local_openvid_hd_csv_path,
                    "limit": self.local_openvid_hd_limit,
                }
            )
        elif self.local_openvid_hd_video_root is not None or self.local_openvid_hd_csv_path is not None:
            print(
                "[WanVideoDataset] warning: openvid HD 源参数不完整，"
                "需同时提供 local_openvid_hd_video_root 与 local_openvid_hd_csv_path，已忽略该源"
            )

        self.local_openvid_enabled = len(self.local_openvid_sources) > 0
        tokenizer_local_only = os.environ.get("WAN_TOKENIZER_LOCAL_ONLY", "0").strip().lower() in (
            "1",
            "true",
            "on",
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                caption_tokenizer_path,
                local_files_only=tokenizer_local_only,
            )
        except Exception as e:
            if tokenizer_local_only:
                raise RuntimeError(
                    f"[WanVideoDataset] 本地加载 tokenizer 失败: {caption_tokenizer_path}. "
                    "请确认该路径可读，或关闭 WAN_TOKENIZER_LOCAL_ONLY。"
                ) from e
            # 兜底：网络异常时自动尝试仅本地缓存，避免 Determined 环境因外网抖动直接失败
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    caption_tokenizer_path,
                    local_files_only=True,
                )
                print(
                    f"[WanVideoDataset] tokenizer remote load failed, fallback local cache only: "
                    f"path={caption_tokenizer_path} err={e}"
                )
            except Exception as e2:
                raise RuntimeError(
                    f"[WanVideoDataset] tokenizer 加载失败: {caption_tokenizer_path}. "
                    "网络访问异常且本地缓存不可用。建议在 .sh 中设置 CAPTION_TOKENIZER_PATH 为本地目录，"
                    "并开启 TOKENIZER_LOCAL_ONLY=1。"
                ) from e2

        resolved_ds, stage_subset_size, total_hint = resolve_hf_stage_config(
            stage=hf_stage,
            dataset_name=hf_dataset_name,
            subset_ratio=hf_subset_ratio,
        )
        if self.local_openvid_enabled:
            self.dataset_name = "local/OpenVid-1M+HD"
            self.total_hint = None
            self.is_openvid = True
        else:
            self.dataset_name = resolved_ds
            self.total_hint = total_hint
            self.is_openvid = self.dataset_name.lower() == "nkp37/openvid-1m"
        self.openvid_record_streaming = os.environ.get("OPENVID_RECORD_STREAMING", "1").strip().lower() in (
            "1",
            "true",
            "on",
        )
        openvid_root = os.environ.get("OPENVID_VIDEO_ROOT", "").strip()
        if self.local_openvid_enabled:
            openvid_root = str(self.local_openvid_sources[0]["video_root"])
        self.openvid_video_root = Path(openvid_root) if openvid_root else None
        openvid_archive_root = os.environ.get("OPENVID_ARCHIVE_ROOT", "").strip()
        if self.local_openvid_enabled:
            openvid_archive_root = ""
        self.openvid_archive_root = Path(openvid_archive_root) if openvid_archive_root else None
        self.openvid_snapshot_download = os.environ.get("OPENVID_SNAPSHOT_DOWNLOAD", "0").strip().lower() in (
            "1",
            "true",
            "on",
        )
        if self.local_openvid_enabled:
            self.openvid_snapshot_download = False
        self.openvid_snapshot_dir = Path(
            os.environ.get("OPENVID_SNAPSHOT_DIR", str(Path(hf_cache_dir or ".hf_cache") / "openvid_repo"))
        )
        self.openvid_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.openvid_snapshot_patterns = [
            p.strip()
            for p in os.environ.get(
                "OPENVID_SNAPSHOT_PATTERNS",
                "Openvid_part*.zip,Openvid_part*.part*,OpenVidHD.csv,data/*",
            ).split(",")
            if p.strip()
        ]
        self.openvid_allow_http_guess = os.environ.get("OPENVID_ALLOW_HTTP_GUESS", "0").strip().lower() in (
            "1",
            "true",
            "on",
        )
        self._openvid_archive_index_built = False
        self._openvid_archive_map = {}
        self._openvid_archive_miss = set()
        self._openvid_autofallback_done = False
        self._openvid_archive_max_scan = max(
            0, int(os.environ.get("OPENVID_ARCHIVE_MAX_SCAN_FILES", "0"))
        )
        self.openvid_auto_join_parts = os.environ.get("OPENVID_AUTO_JOIN_PARTS", "1").strip().lower() in (
            "1",
            "true",
            "on",
        )
        self.openvid_joined_archive_dir = Path(
            os.environ.get("OPENVID_JOINED_ARCHIVE_DIR", str(self.local_video_cache_dir / "openvid_joined"))
        )
        self.openvid_joined_archive_dir.mkdir(parents=True, exist_ok=True)
        self.openvid_extracted_cache_dir = self.local_video_cache_dir / "openvid_extracted"
        self.openvid_extracted_cache_dir.mkdir(parents=True, exist_ok=True)
        self._subset_streaming = (
            (not self.local_openvid_enabled)
            and (self.hf_streaming or (self.is_openvid and self.openvid_record_streaming))
        )
        if (
            self.is_openvid
            and (not self.local_openvid_enabled)
            and self.openvid_record_streaming
            and self.openvid_snapshot_download
            and self.openvid_archive_root is None
        ):
            print("[WanVideoDataset] openvid_record_streaming=1, skip snapshot_download")
            self.openvid_snapshot_download = False
        if (
            self.is_openvid
            and (not self.local_openvid_enabled)
            and self.openvid_archive_root is None
            and self.openvid_snapshot_download
        ):
            self.openvid_archive_root = self._prepare_openvid_archive_root_from_hf()
        if self.local_openvid_enabled:
            per_source_limits = [src["limit"] for src in self.local_openvid_sources]
            if per_source_limits and all(v is not None for v in per_source_limits):
                self.target_subset_size = sum(int(v) for v in per_source_limits)
            else:
                self.target_subset_size = 0
        else:
            self.target_subset_size = hf_subset_size or stage_subset_size or 10000
        self.samples = self._build_subset()
        if len(self.samples) == 0:
            raise RuntimeError(f"数据集可用样本为0: {self.dataset_name}")
        print(
            f"[WanVideoDataset] dataset={self.dataset_name} split={self.hf_split} "
            f"target={self.target_subset_size} loaded={len(self.samples)} "
            f"scanned={self._subset_scanned} scan_factor={self.scan_factor} "
            f"streaming={self.hf_streaming} ratio={self.hf_subset_ratio} "
            f"subset_size_override={self.hf_subset_size} cache_dir={self.hf_cache_dir}"
        )
        if self.is_openvid:
            print(
                f"[WanVideoDataset] openvid_mode=1 "
                f"openvid_video_root={str(self.openvid_video_root) if self.openvid_video_root else 'unset'} "
                f"openvid_archive_root={str(self.openvid_archive_root) if self.openvid_archive_root else 'unset'} "
                f"snapshot_download={self.openvid_snapshot_download} "
                f"record_streaming={self.openvid_record_streaming} "
                f"auto_join_parts={self.openvid_auto_join_parts} "
                f"allow_http_guess={self.openvid_allow_http_guess}"
            )
        if self.local_openvid_enabled:
            for src in self.local_openvid_sources:
                print(
                    f"[WanVideoDataset] local_openvid_source "
                    f"name={src['name']} video_root={src['video_root']} "
                    f"csv_path={src['csv_path']} "
                    f"limit={src['limit'] if src['limit'] else 'all'}"
                )

    def _subset_cache_path(self):
        if self.local_openvid_enabled:
            source_parts = []
            for src in self.local_openvid_sources:
                source_parts.append(
                    f"{src['name']}:{src['video_root']}:{src['csv_path']}:limit={src['limit']}"
                )
            key = (
                f"local_openvid_multi|{'|'.join(source_parts)}|"
                f"f={self.frame_num}|min={self.min_duration_sec}|"
                f"max={self.max_duration_sec}|tok={self.max_caption_tokens}|seed={self.seed}|"
                f"preclean={self.preclean_enabled}"
            )
            name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pkl"
            return self.hf_subset_cache_dir / name
        key = (
            f"{self.dataset_name}|{self.hf_split}|{self.target_subset_size}|"
            f"{self.scan_factor}|{self.seed}|{self.hf_streaming}|"
            f"preclean={self.preclean_enabled}|f={self.frame_num}|min={self.min_duration_sec}|"
            f"max={self.max_duration_sec}|tok={self.max_caption_tokens}"
        )
        name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pkl"
        return self.hf_subset_cache_dir / name

    def _build_subset(self):
        cache_path = self._subset_cache_path()
        if self.local_openvid_enabled:
            return self._build_local_openvid_subset(cache_path)
        if self.hf_subset_use_cache and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                self._subset_scanned = int(payload.get("scanned", 0))
                samples = payload.get("samples", [])
                self._subset_accepted = len(samples)
                rejected_stats = payload.get("rejected_stats", {})
                self._subset_rejected_stats = defaultdict(
                    int, {str(k): int(v) for k, v in rejected_stats.items()}
                )
                print(f"[WanVideoDataset] subset_cache_hit={cache_path} loaded={len(samples)}")
                if len(samples) > 0:
                    return samples
            except Exception:
                pass
        ds = load_dataset(
            self.dataset_name,
            split=self.hf_split,
            streaming=self._subset_streaming,
            cache_dir=self.hf_cache_dir,
        )
        if self.hf_shuffle_buffer > 0 and hasattr(ds, "shuffle"):
            try:
                if self._subset_streaming:
                    ds = ds.shuffle(seed=self.seed, buffer_size=self.hf_shuffle_buffer)
                else:
                    ds = ds.shuffle(seed=self.seed)
            except TypeError:
                ds = ds.shuffle(seed=self.seed)

        samples = []
        rejected_stats = defaultdict(int)
        scan_multiplier = self.preclean_scan_multiplier if self.preclean_enabled else 1
        max_scan = self.target_subset_size * self.scan_factor * scan_multiplier
        max_scan = min(max_scan, self.preclean_scan_cap)
        scanned = 0
        for row in ds:
            scanned += 1
            parsed = self._extract_row(row)
            if parsed is not None:
                if self.preclean_enabled:
                    ok, reject_reason = self._preclean_sample(parsed)
                    if ok:
                        samples.append(parsed)
                    else:
                        rejected_stats[reject_reason] += 1
                else:
                    samples.append(parsed)
            else:
                rejected_stats["extract_failed"] += 1
            if scanned % self.preclean_log_interval == 0:
                rejected = sum(rejected_stats.values())
                top_reject = ", ".join(
                    f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:4]
                ) or "none"
                top_io = ", ".join(
                    f"{k}={v}" for k, v in sorted(self._failure_stats.items(), key=lambda x: -x[1])[:4]
                ) or "none"
                print(
                    f"[WanVideoDataset] preclean_progress scanned={scanned} "
                    f"accepted={len(samples)} rejected={rejected} top=[{top_reject}] io=[{top_io}] "
                    f"target={self.target_subset_size} max_scan={max_scan}"
                )
            if (
                self.preclean_enabled
                and len(samples) == 0
                and scanned >= self.preclean_zero_accept_abort_scan
            ):
                print(
                    f"[WanVideoDataset] preclean_early_abort scanned={scanned} accepted=0 "
                    f"reason=zero_accepted_until_threshold({self.preclean_zero_accept_abort_scan})"
                )
                break
            if len(samples) >= self.target_subset_size:
                break
            if scanned >= max_scan:
                break
        self._subset_scanned = scanned
        self._subset_accepted = len(samples)
        self._subset_rejected_stats = rejected_stats
        rejected = sum(rejected_stats.values())
        top_reject = ", ".join(
            f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:8]
        ) or "none"
        top_io = ", ".join(
            f"{k}={v}" for k, v in sorted(self._failure_stats.items(), key=lambda x: -x[1])[:8]
        ) or "none"
        print(
            f"[WanVideoDataset] preclean_done scanned={scanned} accepted={len(samples)} "
            f"rejected={rejected} top=[{top_reject}] io=[{top_io}] "
            f"preclean={self.preclean_enabled} target={self.target_subset_size} max_scan={max_scan}"
        )
        if self.hf_subset_use_cache:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "scanned": scanned,
                            "samples": samples,
                            "rejected_stats": dict(rejected_stats),
                        },
                        f,
                    )
                print(f"[WanVideoDataset] subset_cache_write={cache_path} saved={len(samples)}")
            except Exception:
                pass
        if len(samples) == 0:
            if (
                self.is_openvid
                and (not self._openvid_autofallback_done)
                and self.openvid_archive_root is None
                and self._failure_stats.get("openvid_filename_without_local_root", 0) > 0
                and self.openvid_snapshot_dir is not None
            ):
                self._openvid_autofallback_done = True
                print(
                    "[WanVideoDataset] openvid_record_streaming_detected_filename_only=1, "
                    "auto_fallback_to_snapshot_download=1"
                )
                self.openvid_snapshot_download = True
                self.openvid_archive_root = self._prepare_openvid_archive_root_from_hf()
                if self.openvid_archive_root is not None:
                    self._openvid_archive_index_built = False
                    self._openvid_archive_map = {}
                    self._openvid_archive_miss = set()
                    return self._build_subset()
            raise RuntimeError(
                f"预清洗后可用样本为0: dataset={self.dataset_name} scanned={scanned} "
                f"rejected={rejected} top=[{top_reject}] io=[{top_io}] "
                f"请检查OPENVID_SNAPSHOT_DIR中是否含Openvid_part*.zip/part*并开启OPENVID_AUTO_JOIN_PARTS=1"
            )
        return samples

    @staticmethod
    def _normalize_local_openvid_key(value):
        if value is None:
            return ""
        out = str(value).strip().replace("\\", "/")
        while out.startswith("./"):
            out = out[2:]
        out = out.lstrip("/")
        return out.lower()

    def _iter_local_openvid_files(self, video_root: Path):
        if video_root is None:
            return []
        if not video_root.exists():
            raise RuntimeError(
                f"local_openvid_video_root 不存在: {video_root}"
            )
        exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
        files = []
        for p in video_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in exts:
                files.append(p)
        files.sort()
        return files

    def _load_local_openvid_caption_index(self, csv_path: Path, source_name: str):
        if csv_path is None:
            raise RuntimeError("local_openvid_csv_path 未设置")
        if not csv_path.exists():
            raise RuntimeError(f"local_openvid_csv_path 不存在: {csv_path}")
        path_to_caption = {}
        name_to_caption = {}
        row_count = 0
        drop_no_video = 0
        drop_no_caption = 0
        selected_video_col = None
        selected_caption_col = None

        def _keep_longer(mapping, key, caption):
            old = mapping.get(key, None)
            if old is None or len(caption) > len(old):
                mapping[key] = caption

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [str(x) for x in (reader.fieldnames or [])]
            if not fieldnames:
                raise RuntimeError(f"CSV 无表头: {csv_path}")
            lowered = {name.lower(): name for name in fieldnames}
            for key in ("video", "video_path", "filename", "file", "path"):
                if key in lowered:
                    selected_video_col = lowered[key]
                    break
            for key in ("caption", "text", "description", "prompt", "summary"):
                if key in lowered:
                    selected_caption_col = lowered[key]
                    break
            if selected_video_col is None or selected_caption_col is None:
                raise RuntimeError(
                    "CSV 缺少 video/caption 列。"
                    f"当前列: {fieldnames}"
                )
            for row in reader:
                row_count += 1
                video_val = str(row.get(selected_video_col, "") or "").strip()
                caption_val = str(row.get(selected_caption_col, "") or "").strip()
                if not video_val:
                    drop_no_video += 1
                    continue
                if not caption_val:
                    drop_no_caption += 1
                    continue
                norm_path = self._normalize_local_openvid_key(video_val)
                if norm_path:
                    _keep_longer(path_to_caption, norm_path, caption_val)
                basename = self._normalize_local_openvid_key(Path(video_val).name)
                if basename:
                    _keep_longer(name_to_caption, basename, caption_val)
        print(
            f"[WanVideoDataset][local_openvid][{source_name}] csv_index_done rows={row_count} "
            f"video_col={selected_video_col} caption_col={selected_caption_col} "
            f"path_keys={len(path_to_caption)} name_keys={len(name_to_caption)} "
            f"drop_no_video={drop_no_video} drop_no_caption={drop_no_caption}"
        )
        return path_to_caption, name_to_caption

    def _lookup_local_openvid_caption(
        self,
        video_path: Path,
        video_root: Path,
        path_to_caption,
        name_to_caption,
    ):
        rel_path = str(video_path.relative_to(video_root)).replace("\\", "/")
        rel_key = self._normalize_local_openvid_key(rel_path)
        cap = path_to_caption.get(rel_key, None)
        if cap:
            return cap, rel_path, "rel_path"
        name_key = self._normalize_local_openvid_key(video_path.name)
        cap = name_to_caption.get(name_key, None)
        if cap:
            return cap, rel_path, "basename"
        return None, rel_path, "missing"

    def _build_local_openvid_subset(self, cache_path):
        if self.hf_subset_use_cache and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                self._subset_scanned = int(payload.get("scanned", 0))
                samples = payload.get("samples", [])
                self._subset_accepted = len(samples)
                rejected_stats = payload.get("rejected_stats", {})
                self._subset_rejected_stats = defaultdict(
                    int, {str(k): int(v) for k, v in rejected_stats.items()}
                )
                print(f"[WanVideoDataset] local_openvid_subset_cache_hit={cache_path} loaded={len(samples)}")
                if len(samples) > 0:
                    return samples
            except Exception:
                pass

        paired = []
        source_pair_stats = []
        max_missing_print = max(0, int(os.environ.get("WAN_LOCAL_MISSING_CAPTION_PRINT_MAX", "200")))
        for src in self.local_openvid_sources:
            src_name = str(src["name"])
            src_video_root = src["video_root"]
            src_csv_path = src["csv_path"]
            src_limit = src["limit"]
            video_files = self._iter_local_openvid_files(src_video_root)
            path_to_caption, name_to_caption = self._load_local_openvid_caption_index(
                src_csv_path, src_name
            )
            source_paired = []
            missing_caption = 0
            match_by_rel = 0
            match_by_name = 0
            for vf in video_files:
                caption, rel_path, matched_by = self._lookup_local_openvid_caption(
                    vf, src_video_root, path_to_caption, name_to_caption
                )
                if not caption:
                    missing_caption += 1
                    if missing_caption <= max_missing_print:
                        print(
                            f"[WanVideoDataset][local_openvid][{src_name}] "
                            f"missing_caption_skip video={rel_path}"
                        )
                    continue
                if matched_by == "rel_path":
                    match_by_rel += 1
                elif matched_by == "basename":
                    match_by_name += 1
                source_paired.append(
                    {
                        "caption": caption,
                        "video_spec": {"kind": "path", "value": str(vf)},
                        "raw": {
                            "video": rel_path,
                            "matched_by": matched_by,
                            "source_name": src_name,
                        },
                    }
                )
            if missing_caption > max_missing_print:
                print(
                    f"[WanVideoDataset][local_openvid][{src_name}] missing_caption_skip_more="
                    f"{missing_caption - max_missing_print}"
                )
            if src_limit and src_limit > 0 and len(source_paired) > src_limit:
                rng = random.Random(self.seed + (abs(hash(src_name)) % 10007))
                rng.shuffle(source_paired)
                source_paired = source_paired[:src_limit]
            source_pair_stats.append(
                {
                    "name": src_name,
                    "local_videos": len(video_files),
                    "paired": len(source_paired),
                    "missing_caption": missing_caption,
                    "matched_by_rel": match_by_rel,
                    "matched_by_name": match_by_name,
                    "limit": src_limit,
                }
            )
            paired.extend(source_paired)

        rng = random.Random(self.seed)
        rng.shuffle(paired)
        if len(paired) == 0:
            raise RuntimeError(
                "本地OpenVid(含HD)配对后样本数为0，请检查视频目录与CSV是否匹配。"
                f" sources={[(str(s['video_root']), str(s['csv_path'])) for s in self.local_openvid_sources]}"
            )
        for st in source_pair_stats:
            print(
                f"[WanVideoDataset][local_openvid][{st['name']}] pair_done "
                f"local_videos={st['local_videos']} paired={st['paired']} "
                f"missing_caption={st['missing_caption']} matched_by_rel={st['matched_by_rel']} "
                f"matched_by_name={st['matched_by_name']} "
                f"target_limit={st['limit'] if st['limit'] else 'all'}"
            )
        print(f"[WanVideoDataset][local_openvid] pair_merged_total={len(paired)}")
        if self.local_openvid_total_limit and len(paired) > self.local_openvid_total_limit:
            paired = paired[: self.local_openvid_total_limit]
            print(
                f"[WanVideoDataset][local_openvid] pair_merged_capped={len(paired)} "
                f"total_limit={self.local_openvid_total_limit}"
            )

        samples = []
        rejected_stats = defaultdict(int)
        scanned = 0
        for parsed in paired:
            scanned += 1
            if self.preclean_enabled:
                ok, reject_reason = self._preclean_sample(parsed)
                if ok:
                    samples.append(parsed)
                else:
                    rejected_stats[reject_reason] += 1
            else:
                samples.append(parsed)
            if scanned % self.preclean_log_interval == 0:
                rejected = sum(rejected_stats.values())
                top_reject = ", ".join(
                    f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:4]
                ) or "none"
                print(
                    f"[WanVideoDataset][local_openvid] preclean_progress scanned={scanned} "
                    f"accepted={len(samples)} rejected={rejected} top=[{top_reject}]"
                )
        self._subset_scanned = scanned
        self._subset_accepted = len(samples)
        self._subset_rejected_stats = rejected_stats
        rejected = sum(rejected_stats.values())
        top_reject = ", ".join(
            f"{k}={v}" for k, v in sorted(rejected_stats.items(), key=lambda x: -x[1])[:8]
        ) or "none"
        print(
            f"[WanVideoDataset][local_openvid] preclean_done scanned={scanned} "
            f"accepted={len(samples)} rejected={rejected} top=[{top_reject}]"
        )
        if self.hf_subset_use_cache:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "scanned": scanned,
                            "samples": samples,
                            "rejected_stats": dict(rejected_stats),
                        },
                        f,
                    )
                print(f"[WanVideoDataset] local_openvid_subset_cache_write={cache_path} saved={len(samples)}")
            except Exception:
                pass
        if len(samples) == 0:
            raise RuntimeError(
                "本地OpenVid(含HD)样本在预清洗后为0，请检查视频可解码性或放宽过滤阈值。"
            )
        return samples

    def _preclean_sample(self, sample):
        caption = sample["caption"]
        token_ids = self.tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"]
        if len(token_ids) > self.max_caption_tokens:
            return False, "caption_too_long"
        video_path = self._materialize_video(sample["video_spec"])
        if video_path is None:
            return False, "materialize_failed"
        if not self._probe_video_quick(video_path):
            try:
                p = Path(video_path)
                if self.local_video_cache_dir in p.parents:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
            return False, "probe_failed"
        return True, "ok"

    def _probe_video_quick(self, video_path):
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                return False
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if total <= 0 or total < self.frame_num:
                cap.release()
                return False
            duration = (total / fps) if fps > 0 else None
            if duration is not None and duration < self.min_duration_sec:
                cap.release()
                return False
            if duration is not None and duration > self.max_duration_sec:
                cap.release()
                return False
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return False
            return True
        except Exception:
            return False

    def _extract_row(self, row):
        caption = self._extract_caption(row)
        if caption is None:
            return None
        video_spec = self._extract_video_spec(row)
        if video_spec is None:
            return None
        return {
            "caption": caption,
            "video_spec": video_spec,
            "raw": row,
        }

    def _extract_caption(self, row):
        keys = (
            "caption",
            "text",
            "description",
            "prompt",
            "sentence",
            "summary",
        )
        for k in keys:
            v = row.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _extract_video_spec(self, row):
        keys = (
            "video",
            "mp4",
            "file",
            "video_path",
            "path",
            "url",
            "video_name",
            "filename",
        )
        candidates = []
        for k in keys:
            if k not in row:
                continue
            value = row[k]
            candidates.extend(self._collect_video_specs(value))
        if not candidates:
            return None
        return max(candidates, key=self._video_spec_priority)

    def _collect_video_specs(self, value, depth=0):
        if depth > 4:
            return []
        out = []
        spec = self._parse_video_value(value)
        if spec is not None:
            out.append(spec)
        if isinstance(value, dict):
            preferred_keys = (
                "bytes",
                "url",
                "download_url",
                "video_url",
                "href",
                "path",
                "file",
                "video",
            )
            for k in preferred_keys:
                if k in value:
                    out.extend(self._collect_video_specs(value.get(k), depth + 1))
            for k, v in value.items():
                if k in preferred_keys:
                    continue
                out.extend(self._collect_video_specs(v, depth + 1))
        elif isinstance(value, (list, tuple)):
            for v in value:
                out.extend(self._collect_video_specs(v, depth + 1))
        uniq = {}
        for item in out:
            kind = item.get("kind")
            val = item.get("value")
            if kind == "bytes":
                key = ("bytes", hashlib.sha1(bytes(val)).hexdigest())
            else:
                key = (kind, str(val))
            uniq[key] = item
        return list(uniq.values())

    def _video_spec_priority(self, spec):
        kind = spec.get("kind")
        value = spec.get("value")
        if kind == "bytes":
            return 300
        if kind == "url":
            return 200
        if kind == "path":
            try:
                p = Path(value)
                if p.exists():
                    return 100
            except Exception:
                pass
            return 10
        return 0

    def _parse_video_value(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if value.startswith("http://") or value.startswith("https://"):
                return {"kind": "url", "value": value}
            if value.startswith("hf://"):
                return {"kind": "url", "value": value}
            return {"kind": "path", "value": value}
        if isinstance(value, (bytes, bytearray)):
            return {"kind": "bytes", "value": value}
        return None

    def _normalize_rel_path(self, value: str):
        path = str(value).strip().replace("\\", "/")
        if path.startswith("hf://"):
            return path
        while path.startswith("./"):
            path = path[2:]
        return path.lstrip("/")

    def _candidate_urls_from_missing_path(self, value: str):
        path = self._normalize_rel_path(value)
        if path.startswith("http://") or path.startswith("https://"):
            return [path]
        if path.startswith("hf://datasets/"):
            return [
                path.replace(
                    f"hf://datasets/{self.dataset_name}/",
                    f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/",
                )
            ]
        if self.is_openvid and (not self.openvid_allow_http_guess):
            return []
        ds_name = self.dataset_name
        split = str(self.hf_split).strip("/")
        candidates = [
            f"https://huggingface.co/datasets/{ds_name}/resolve/main/{path}",
            f"https://huggingface.co/datasets/{ds_name}/resolve/main/data/{path}",
            f"https://huggingface.co/datasets/{ds_name}/resolve/main/videos/{path}",
            f"https://huggingface.co/datasets/{ds_name}/resolve/main/{split}/{path}",
            f"https://huggingface.co/datasets/{ds_name}/resolve/main/files/{path}",
        ]
        uniq = []
        seen = set()
        for u in candidates:
            if u in seen:
                continue
            seen.add(u)
            uniq.append(u)
        return uniq

    def _resolve_openvid_local_path(self, value: str):
        normalized = self._normalize_rel_path(value)
        if normalized.startswith("http://") or normalized.startswith("https://"):
            return None
        if normalized.startswith("hf://"):
            return None
        if self.openvid_video_root:
            p = self.openvid_video_root / normalized
            if p.exists():
                return p
            try:
                p2 = self.openvid_video_root / Path(normalized).name
                if p2.exists():
                    return p2
            except Exception:
                pass
        return self._extract_openvid_from_archives(normalized)

    def _prepare_openvid_archive_root_from_hf(self):
        try:
            print(
                f"[WanVideoDataset] openvid_snapshot_download_start "
                f"repo={self.dataset_name} local_dir={self.openvid_snapshot_dir} "
                f"patterns={self.openvid_snapshot_patterns}"
            )
            path = snapshot_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                local_dir=str(self.openvid_snapshot_dir),
                allow_patterns=self.openvid_snapshot_patterns,
            )
            print(f"[WanVideoDataset] openvid_snapshot_download_done local_dir={path}")
            return Path(path)
        except Exception:
            self._failure_stats["openvid_snapshot_download_error"] += 1
            return None

    def _iter_openvid_archive_files(self):
        if not self.openvid_archive_root or (not self.openvid_archive_root.exists()):
            return []
        archives = sorted(self.openvid_archive_root.rglob("*.zip"))
        if self.openvid_auto_join_parts:
            archives.extend(self._join_openvid_part_archives())
            archives = sorted(set(archives))
        if self._openvid_archive_max_scan > 0:
            archives = archives[: self._openvid_archive_max_scan]
        return archives

    def _join_openvid_part_archives(self):
        part_files = sorted(self.openvid_archive_root.rglob("*.part*"))
        groups = defaultdict(list)
        for part in part_files:
            name = part.name
            if ".part" not in name:
                continue
            prefix, suffix = name.split(".part", 1)
            if not suffix:
                continue
            groups[prefix].append((suffix, part))
        joined_archives = []
        for prefix, items in groups.items():
            items = sorted(items, key=lambda x: x[0])
            if len(items) < 2:
                continue
            out_zip = self.openvid_joined_archive_dir / f"{prefix}.zip"
            if not out_zip.exists():
                lock_path = Path(f"{out_zip}.lock")
                if not self._acquire_lock(lock_path, timeout_sec=max(self.lock_timeout_sec, 120)):
                    continue
                try:
                    if not out_zip.exists():
                        tmp_path = Path(
                            f"{out_zip}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
                        )
                        try:
                            with open(tmp_path, "wb") as wf:
                                for _, part_path in items:
                                    with open(part_path, "rb") as rf:
                                        while True:
                                            chunk = rf.read(1024 * 1024)
                                            if not chunk:
                                                break
                                            wf.write(chunk)
                            os.replace(tmp_path, out_zip)
                            print(
                                f"[WanVideoDataset] openvid_join_parts_done prefix={prefix} parts={len(items)} out={out_zip}"
                            )
                        except Exception:
                            self._failure_stats["openvid_join_parts_error"] += 1
                            try:
                                tmp_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                finally:
                    self._release_lock(lock_path)
            if out_zip.exists():
                joined_archives.append(out_zip)
        return joined_archives

    def _build_openvid_archive_index(self):
        if self._openvid_archive_index_built:
            return
        self._openvid_archive_index_built = True
        archives = self._iter_openvid_archive_files()
        if not archives:
            return
        indexed = 0
        for i, archive in enumerate(archives):
            try:
                with zipfile.ZipFile(archive, "r") as zf:
                    for name in zf.namelist():
                        if not name.lower().endswith(".mp4"):
                            continue
                        base = Path(name).name
                        if base not in self._openvid_archive_map:
                            self._openvid_archive_map[base] = (archive, name)
                            indexed += 1
            except Exception:
                self._failure_stats["openvid_archive_read_error"] += 1
            if (i + 1) % 5 == 0:
                print(
                    f"[WanVideoDataset] openvid_archive_index_progress scanned_archives={i + 1} "
                    f"indexed_videos={indexed}"
                )
        print(
            f"[WanVideoDataset] openvid_archive_index_done archives={len(archives)} "
            f"indexed_videos={indexed}"
        )

    def _extract_openvid_from_archives(self, normalized: str):
        if not self.openvid_archive_root:
            return None
        file_name = Path(normalized).name
        if (not file_name) or (not file_name.lower().endswith(".mp4")):
            return None
        if file_name in self._openvid_archive_miss:
            return None
        target_path = self.openvid_extracted_cache_dir / file_name
        if target_path.exists() and self._is_cache_file_valid(target_path):
            return target_path
        self._build_openvid_archive_index()
        arc = self._openvid_archive_map.get(file_name, None)
        if arc is None:
            self._openvid_archive_miss.add(file_name)
            self._failure_stats["openvid_archive_member_missing"] += 1
            return None
        archive_path, member_name = arc
        lock_path = Path(f"{target_path}.lock")
        if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
            self._failure_stats["lock_timeout"] += 1
            if target_path.exists() and self._is_cache_file_valid(target_path):
                return target_path
            return None
        try:
            if target_path.exists() and self._is_cache_file_valid(target_path):
                return target_path
            tmp_path = Path(
                f"{target_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
            )
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    with zf.open(member_name, "r") as src, open(tmp_path, "wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                os.replace(tmp_path, target_path)
            except Exception:
                self._failure_stats["openvid_archive_extract_error"] += 1
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return None
            if not self._is_cache_file_valid(target_path):
                self._failure_stats["openvid_archive_extracted_invalid"] += 1
                try:
                    target_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return None
            self._failure_stats["openvid_archive_extract_hit"] += 1
            return target_path
        finally:
            self._release_lock(lock_path)

    def _cached_file_path(self, key: str, suffix=".mp4"):
        file_name = hashlib.sha1(key.encode("utf-8")).hexdigest() + suffix
        return self.local_video_cache_dir / file_name

    def _build_http_session(self):
        session = requests.Session()
        try:
            from urllib3.util.retry import Retry

            retry = Retry(
                total=self.http_retry_total,
                connect=self.http_retry_total,
                read=self.http_retry_total,
                backoff_factor=0.3,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        except Exception:
            pass
        return session

    def _acquire_lock(self, lock_path: Path, timeout_sec: float = 120.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"{os.getpid()} {time.time()}")
                return True
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > 300:
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                time.sleep(0.2)
            except Exception:
                time.sleep(0.2)
        return False

    def _release_lock(self, lock_path: Path):
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _is_cache_file_valid(self, file_path: Path):
        if not file_path.exists():
            return False
        try:
            if file_path.stat().st_size < 4096:
                return False
        except Exception:
            return False
        try:
            import cv2

            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                cap.release()
                return False
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return total > 0
        except Exception:
            return False

    def _download_url_to_file(self, url: str, target_path: Path, headers):
        tmp_path = Path(f"{target_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}")
        session = self._build_http_session()
        try:
            with session.get(url, stream=True, timeout=self.http_timeout_sec, headers=headers) as r:
                status = int(getattr(r, "status_code", 0) or 0)
                if status >= 400:
                    return False, f"http_{status}"
                r.raise_for_status()
                bytes_written = 0
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_written += len(chunk)
            if bytes_written < 4096:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return False, "download_too_small"
            os.replace(tmp_path, target_path)
            return True, "ok"
        except requests.exceptions.Timeout:
            return False, "timeout"
        except requests.exceptions.HTTPError:
            return False, "http_error"
        except Exception:
            return False, "download_exception"
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                session.close()
            except Exception:
                pass

    def _materialize_video(self, video_spec):
        kind = video_spec["kind"]
        value = video_spec["value"]
        if kind == "path":
            if isinstance(value, str) and (
                value.startswith("http://") or value.startswith("https://")
            ):
                return self._materialize_video({"kind": "url", "value": value})
            if self.is_openvid:
                local_openvid = self._resolve_openvid_local_path(str(value))
                if local_openvid is not None:
                    if not self._is_cache_file_valid(local_openvid):
                        self._failure_stats["openvid_local_invalid"] += 1
                        return None
                    return str(local_openvid)
            p = Path(value)
            if p.exists():
                if not self._is_cache_file_valid(p):
                    self._failure_stats["path_invalid"] += 1
                    return None
                return str(p)
            if self.is_openvid:
                normalized = self._normalize_rel_path(str(value))
                if normalized.lower().endswith(".mp4"):
                    if (self.openvid_video_root is None) and (self.openvid_archive_root is None):
                        self._failure_stats["openvid_filename_without_local_root"] += 1
                    else:
                        self._failure_stats["openvid_local_missing"] += 1
                    return None
            candidate_urls = self._candidate_urls_from_missing_path(str(value))
            if self.quick_fail:
                candidate_urls = candidate_urls[: self.url_fallback_limit]
            for url in candidate_urls:
                url_video = self._materialize_video({"kind": "url", "value": url})
                if url_video is not None:
                    self._failure_stats["path_to_url_fallback_hit"] += 1
                    return url_video
            self._failure_stats["path_missing"] += 1
            return None
        if kind == "url":
            local_path = self._cached_file_path(value, suffix=".mp4")
            lock_path = Path(f"{local_path}.lock")
            if local_path.exists() and self._is_cache_file_valid(local_path):
                return str(local_path)
            if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
                self._failure_stats["lock_timeout"] += 1
                if local_path.exists() and self._is_cache_file_valid(local_path):
                    return str(local_path)
                return None
            try:
                headers = None
                hf_token = os.environ.get("HF_TOKEN", "").strip()
                if hf_token and "huggingface.co" in value:
                    headers = {"Authorization": f"Bearer {hf_token}"}
                if "huggingface.co" in value and (not hf_token) and (not self._warned_hf_token):
                    print("[WanVideoDataset] warning: huggingface url without HF_TOKEN")
                    self._warned_hf_token = True
                if local_path.exists() and not self._is_cache_file_valid(local_path):
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                if local_path.exists() and self._is_cache_file_valid(local_path):
                    return str(local_path)
                ok, reason = self._download_url_to_file(value, local_path, headers=headers)
                if not ok:
                    self._failure_stats[reason] += 1
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return None
                if not self._is_cache_file_valid(local_path):
                    self._failure_stats["cache_invalid_after_download"] += 1
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return None
                return str(local_path)
            finally:
                self._release_lock(lock_path)
        if kind == "bytes":
            if not isinstance(value, (bytes, bytearray)):
                self._failure_stats["bytes_invalid_type"] += 1
                return None
            key = hashlib.sha1(value).hexdigest()
            local_path = self._cached_file_path(key, suffix=".mp4")
            lock_path = Path(f"{local_path}.lock")
            if local_path.exists() and self._is_cache_file_valid(local_path):
                return str(local_path)
            if not self._acquire_lock(lock_path, timeout_sec=self.lock_timeout_sec):
                self._failure_stats["lock_timeout"] += 1
                if local_path.exists() and self._is_cache_file_valid(local_path):
                    return str(local_path)
                return None
            try:
                if local_path.exists() and not self._is_cache_file_valid(local_path):
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                try:
                    tmp_path = Path(
                        f"{local_path}.part.{os.getpid()}.{random.randint(0, 1_000_000_000)}"
                    )
                    with open(tmp_path, "wb") as f:
                        f.write(bytes(value))
                    os.replace(tmp_path, local_path)
                except Exception:
                    self._failure_stats["bytes_write_failed"] += 1
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return None
                if not self._is_cache_file_valid(local_path):
                    self._failure_stats["bytes_cache_invalid"] += 1
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return None
                return str(local_path)
            finally:
                self._release_lock(lock_path)
        return None

    def _load_video_frames(self, video_path):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0:
            cap.release()
            return None, None, None
        if total < self.frame_num:
            cap.release()
            return None, None, None
        duration = (total / fps) if fps > 0 else None
        if duration is not None and duration < self.min_duration_sec:
            cap.release()
            return None, None, None
        if duration is not None and duration > self.max_duration_sec:
            cap.release()
            return None, None, None

        # indices = np.linspace(0, total - 1, self.frame_num, dtype=int)
        frames = []
        # for idx in indices:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        # 训练取帧策略：
        # 旧逻辑：全视频等距抽帧（np.linspace）
        # 新逻辑：从开头连续读取前 frame_num 帧（更贴近“前缀片段学习”）
        for _ in range(self.frame_num):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) < self.frame_num:
            return None, None, None
        return frames, fps, duration

    def _process_frames(self, frames):
        import cv2

        h, w = frames[0].shape[:2]
        area = h * w
        if area > self.max_area:
            scale = math.sqrt(self.max_area / area)
            h = int(h * scale)
            w = int(w * scale)
        h = max(32, (h // 32) * 32)
        w = max(32, (w // 32) * 32)
        resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
        tensor = torch.stack([torch.from_numpy(f) for f in resized]).float() / 127.5 - 1.0
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor, resized[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n = len(self.samples)
        default_trials = min(max(60, n // 4), n)
        if self.quick_fail:
            max_trials = min(default_trials, self.max_trials_cap, n)
            # 启动阶段(last_good 为空)避免在坏样本上长时间重试导致分布式 rank 掉队。
            startup_trials = max(2, int(os.environ.get("WAN_DATA_STARTUP_MAX_TRIALS", "8")))
            if self._last_good_sample is None:
                max_trials = min(max_trials, startup_trials)
        else:
            max_trials = default_trials
        if max_trials <= 0:
            raise RuntimeError("数据集为空，无法取样")
        trial_stats = defaultdict(int)
        for trial in range(max_trials):
            if trial < 10:
                sample_idx = (idx + trial) % n
            else:
                sample_idx = random.randint(0, n - 1)
            sample = self.samples[sample_idx]
            caption = sample["caption"]
            token_ids = self.tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"]
            if len(token_ids) > self.max_caption_tokens:
                trial_stats["caption_too_long"] += 1
                continue

            video_path = self._materialize_video(sample["video_spec"])
            if video_path is None:
                trial_stats["materialize_failed"] += 1
                if (trial + 1) % self.trial_log_interval == 0:
                    trial_detail = ", ".join(
                        f"{k}={v}" for k, v in sorted(trial_stats.items())
                    ) or "none"
                    io_detail = ", ".join(
                        f"{k}={v}" for k, v in sorted(self._failure_stats.items())
                    ) or "none"
                    print(
                        f"[WanVideoDataset] retry_progress idx={idx} tried={trial + 1}/{max_trials} "
                        f"trial_stats=[{trial_detail}] io_stats=[{io_detail}]"
                    )
                continue
            frames, _, _ = self._load_video_frames(video_path)
            if frames is None:
                trial_stats["decode_failed"] += 1
                try:
                    p = Path(video_path)
                    if self.local_video_cache_dir in p.parents:
                        p.unlink(missing_ok=True)
                except Exception:
                    pass
                if (trial + 1) % self.trial_log_interval == 0:
                    trial_detail = ", ".join(
                        f"{k}={v}" for k, v in sorted(trial_stats.items())
                    ) or "none"
                    io_detail = ", ".join(
                        f"{k}={v}" for k, v in sorted(self._failure_stats.items())
                    ) or "none"
                    print(
                        f"[WanVideoDataset] retry_progress idx={idx} tried={trial + 1}/{max_trials} "
                        f"trial_stats=[{trial_detail}] io_stats=[{io_detail}]"
                    )
                continue
            video_tensor, first_frame_np = self._process_frames(frames)
            ref_image = Image.fromarray(first_frame_np)
            mq_ref_image = ref_image
            out_caption = caption
            if random.random() < self.null_caption_prob:
                out_caption = ""
            if random.random() < self.null_image_prob:
                mq_ref_image = None
            if not self._printed_sample_info:
                print(
                    f"[WanVideoDataset] sample_once "
                    f"video_tensor={tuple(video_tensor.shape)} "
                    f"caption_tokens={len(token_ids)} "
                    f"mq_ref_is_none={mq_ref_image is None} "
                    f"video_path={video_path}"
                )
                self._printed_sample_info = True
            result = {
                "caption": out_caption,
                "video": video_tensor,
                "ref_image": ref_image,
                "mq_ref_image": mq_ref_image,
                "video_path": video_path,
            }
            self._last_good_sample = result
            return result
        if self._last_good_sample is not None:
            return self._last_good_sample
        trial_detail = ", ".join(f"{k}={v}" for k, v in sorted(trial_stats.items())) or "none"
        io_detail = ", ".join(f"{k}={v}" for k, v in sorted(self._failure_stats.items())) or "none"
        hf_token_ready = bool(os.environ.get("HF_TOKEN", "").strip())
        raise RuntimeError(
            f"样本解码失败，trials={max_trials}，dataset={self.dataset_name}，"
            f"trial_stats=[{trial_detail}] io_stats=[{io_detail}] "
            f"streaming={self.hf_streaming} hf_token={'set' if hf_token_ready else 'unset'} "
            f"openvid_root={str(self.openvid_video_root) if self.openvid_video_root else 'unset'} "
            f"openvid_archive_root={str(self.openvid_archive_root) if self.openvid_archive_root else 'unset'}，"
            f"建议设置HF_TOKEN并优先使用--hf_no_streaming；OpenVid请配置OPENVID_VIDEO_ROOT或OPENVID_ARCHIVE_ROOT，"
            f"或开启OPENVID_SNAPSHOT_DOWNLOAD=1自动从HF拉取归档"
        )

    @staticmethod
    def collate_fn(batch):
        return {k: [item[k] for item in batch] for k in batch[0]}
