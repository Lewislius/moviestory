import json
from pathlib import Path
from typing import Any, Callable, Dict

import torch

from wan_lora_utils import (
    apply_lora_to_wan_model,
    infer_lora_rank_from_state,
    infer_lora_targets_from_state_keys,
)


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_tensor_state(src: Path) -> Dict[str, torch.Tensor]:
    if src.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(f"检测到 safetensors 权重但未能导入 safetensors: {src}") from e
        payload = load_file(str(src), device="cpu")
        return {k: v for k, v in payload.items() if torch.is_tensor(v)}

    payload = _safe_torch_load(src, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Wan trainable 文件格式异常: {src}")
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resolve_checkpoint_dir(path_or_dir: str | Path) -> Path:
    path = Path(path_or_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 路径不存在: {path}")
    return path if path.is_dir() else path.parent


def _pick_state_file(ckpt_dir: Path, stem: str) -> Path | None:
    sf = ckpt_dir / f"{stem}.safetensors"
    pt = ckpt_dir / f"{stem}.pt"
    if sf.exists():
        return sf
    if pt.exists():
        return pt
    return None


def _normalize_key(name: str) -> str:
    key = str(name)
    while "_fsdp_wrapped_module." in key:
        key = key.replace("_fsdp_wrapped_module.", "")
    return key


def _simplify_fsdp_module_name(name: str) -> str:
    key = str(name)
    while key.startswith("_fsdp_wrapped_module."):
        key = key[len("_fsdp_wrapped_module.") :]
    key = key.replace("._fsdp_wrapped_module", "")
    if key == "_fsdp_wrapped_module":
        key = ""
    return key


def _resolve_by_suffix(
    key: str,
    src: Dict[str, torch.Tensor],
    norm_src: Dict[str, torch.Tensor],
) -> torch.Tensor | None:
    exact = src.get(key, None)
    if exact is not None:
        return exact

    exact_norm = norm_src.get(_normalize_key(key), None)
    if exact_norm is not None:
        return exact_norm

    target = _normalize_key(key)
    matches = [v for k, v in norm_src.items() if k == target or k.endswith(f".{target}")]
    if len(matches) == 1:
        return matches[0]
    return None


def _has_fsdp_modules(model: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:
        return False
    return any(isinstance(m, FSDP) for m in model.modules())


def _remap_flat_state_by_local_template(
    model: torch.nn.Module,
    flat_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import LocalStateDictConfig, StateDictType

    cfg = LocalStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, cfg):
        template = model.state_dict()

    norm_src = {_normalize_key(k): v for k, v in flat_state.items()}
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in template.items():
        if not torch.is_tensor(value):
            continue
        src = _resolve_by_suffix(key, flat_state, norm_src)
        if src is None:
            continue
        if int(src.numel()) != int(value.numel()):
            continue
        remapped[key] = src.detach().cpu().contiguous().view_as(value)
    return remapped


def _inject_flat_params_into_fsdp_model(
    model: torch.nn.Module,
    flat_state: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    norm_src = {_normalize_key(k): v for k, v in flat_state.items()}
    stats = {
        "assigned": 0,
        "mismatched": 0,
        "scanned": 0,
        "key_not_found": 0,
    }

    for module_name, module in model.named_modules():
        if not isinstance(module, FSDP):
            continue
        stats["scanned"] += 1
        flat_param = getattr(module, "_flat_param", None)
        if flat_param is None and hasattr(module, "_handle") and getattr(module._handle, "flat_param", None) is not None:
            flat_param = module._handle.flat_param
        if flat_param is None or not torch.is_tensor(flat_param):
            continue

        simple_name = _simplify_fsdp_module_name(module_name)
        candidate_key = "_flat_param" if simple_name == "" else f"{simple_name}._flat_param"
        tensor = _resolve_by_suffix(candidate_key, flat_state, norm_src)
        if tensor is None:
            stats["key_not_found"] += 1
            continue
        if int(tensor.numel()) != int(flat_param.numel()):
            stats["mismatched"] += 1
            continue
        with torch.no_grad():
            flat_param.data.copy_(tensor.to(device=flat_param.device, dtype=flat_param.dtype).view_as(flat_param))
        stats["assigned"] += 1
    return stats


def _log(rank: int, message: str) -> None:
    print(f"[WanFinetune][rank{rank}] {message}")


def load_wan_finetune_weights(
    *,
    model: torch.nn.Module,
    checkpoint_path_or_dir: str | Path,
    record_metric: Callable[[str, Any], None],
    warn: Callable[[str], None],
    rank: int = 0,
    enable_fsdp_flat_inference: bool = False,
) -> Dict[str, Any]:
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_path_or_dir)
    picked = _pick_state_file(ckpt_dir, "wan_dit_trainable")
    lora_picked = _pick_state_file(ckpt_dir, "wan_dit_lora")

    if picked is None and lora_picked is None:
        record_metric("wan_finetune_loaded", 0)
        record_metric("wan_finetune_reason", "no_wan_trainable_file")
        return {
            "loaded": False,
            "reason": "no_wan_trainable_file",
            "checkpoint_dir": str(ckpt_dir),
        }

    training_args = _read_json_if_exists(ckpt_dir / "training_args.json")
    config = _read_json_if_exists(ckpt_dir / "config.json")
    training_args_excerpt = {
        key: training_args.get(key, None)
        for key in (
            "wan_train_mode",
            "wan_cond_name_pattern",
            "enable_wan_lora",
            "wan_lora_rank",
            "wan_lora_alpha",
            "wan_lora_dropout",
            "wan_lora_targets",
            "dit_fsdp",
            "use_sp",
            "world_size",
            "distributed_world_size",
            "nproc_per_node",
            "num_processes",
            "num_gpus",
        )
    }

    total_missing = 0
    total_unexpected = 0
    total_state_tensors = 0
    flat_key_count = 0
    load_mode = "none"
    loaded_files: list[str] = []
    fsdp_stats: Dict[str, int] = {}

    if lora_picked is not None:
        lora_state = _load_tensor_state(lora_picked)
        if not lora_state:
            warn("wan_dit_lora.* 为空，跳过 LoRA 加载")
        else:
            wan_lora_cfg = config.get("wan_lora", {}) if isinstance(config, dict) else {}
            rank_value = int(
                wan_lora_cfg.get(
                    "rank",
                    training_args_excerpt.get("wan_lora_rank", 0) or infer_lora_rank_from_state(lora_state, default=16),
                )
            )
            alpha_value = float(
                wan_lora_cfg.get(
                    "alpha",
                    training_args_excerpt.get("wan_lora_alpha", rank_value) or rank_value,
                )
            )
            dropout_value = float(
                wan_lora_cfg.get(
                    "dropout",
                    training_args_excerpt.get("wan_lora_dropout", 0.0) or 0.0,
                )
            )
            targets = wan_lora_cfg.get(
                "targets",
                training_args_excerpt.get("wan_lora_targets", ""),
            ) or infer_lora_targets_from_state_keys(lora_state.keys())

            matched = apply_lora_to_wan_model(
                model,
                rank=rank_value,
                alpha=alpha_value,
                dropout=dropout_value,
                target_types=targets,
            )
            if not matched:
                raise RuntimeError("检测到 wan_dit_lora.*，但推理图中未注入到任何 Wan Linear")

            missing_lora, unexpected_lora = model.load_state_dict(lora_state, strict=False)
            _log(
                rank,
                f"LoRA loaded: {lora_picked} tensors={len(lora_state)} "
                f"missing={len(missing_lora)} unexpected={len(unexpected_lora)}",
            )
            total_state_tensors += int(len(lora_state))
            total_missing += int(len(missing_lora))
            total_unexpected += int(len(unexpected_lora))
            loaded_files.append(str(lora_picked))
            record_metric("wan_lora_loaded", 1)
            record_metric("wan_lora_state_tensors", int(len(lora_state)))
            record_metric("wan_lora_missing", int(len(missing_lora)))
            record_metric("wan_lora_unexpected", int(len(unexpected_lora)))
            record_metric("wan_lora_path", str(lora_picked))

    if picked is not None:
        state = _load_tensor_state(picked)
        if not state:
            warn("wan_dit_trainable.* 为空，跳过加载")
        else:
            total_state_tensors += int(len(state))
            loaded_files.append(str(picked))
            flat_key_count = int(sum(1 for k in state.keys() if "_flat_param" in str(k)))
            record_metric("wan_finetune_flat_key_count", flat_key_count)

            has_fsdp = _has_fsdp_modules(model)
            if flat_key_count > 0 and has_fsdp and enable_fsdp_flat_inference:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import LocalStateDictConfig, StateDictType

                def _try_local_load(
                    local_state: Dict[str, torch.Tensor],
                    tag: str,
                ) -> tuple[list[str], list[str]] | None:
                    local_cfg = LocalStateDictConfig(offload_to_cpu=True)
                    try:
                        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, local_cfg):
                            return model.load_state_dict(local_state, strict=False)
                    except AssertionError as e:
                        # Some checkpoints are not a valid LOCAL_STATE_DICT for this FSDP
                        # graph; skip and continue to remap/inject fallback path.
                        _log(rank, f"{tag} skipped: {e}")
                        return None
                    except Exception as e:
                        _log(rank, f"{tag} failed: {type(e).__name__}: {e}")
                        return None

                missing, unexpected = model.load_state_dict(state, strict=False)
                load_mode = "fsdp_default_load"
                best_missing = len(missing)
                best_unexpected = len(unexpected)
                _log(
                    rank,
                    f"default load: tensors={len(state)} missing={best_missing} unexpected={best_unexpected}",
                )

                if best_unexpected > 0:
                    local_result = _try_local_load(state, "local load(raw)")
                    if local_result is not None:
                        missing_local, unexpected_local = local_result
                        _log(
                            rank,
                            f"local load(raw): missing={len(missing_local)} unexpected={len(unexpected_local)}",
                        )
                        if len(unexpected_local) < best_unexpected or (
                            len(unexpected_local) == best_unexpected and len(missing_local) < best_missing
                        ):
                            best_missing = len(missing_local)
                            best_unexpected = len(unexpected_local)
                            load_mode = "fsdp_local_state_load"

                if best_unexpected > 0:
                    remapped_state: Dict[str, torch.Tensor] = {}
                    try:
                        remapped_state = _remap_flat_state_by_local_template(model, state)
                    except Exception as e:
                        _log(rank, f"local remap build failed: {type(e).__name__}: {e}")
                    if remapped_state:
                        local_result = _try_local_load(remapped_state, "local load(remapped)")
                        if local_result is not None:
                            missing_remap, unexpected_remap = local_result
                            _log(
                                rank,
                                f"local load(remapped): tensors={len(remapped_state)} "
                                f"missing={len(missing_remap)} unexpected={len(unexpected_remap)}",
                            )
                            if len(unexpected_remap) <= best_unexpected:
                                best_missing = len(missing_remap)
                                best_unexpected = len(unexpected_remap)
                                load_mode = "fsdp_local_template_remap"

                if best_unexpected > 0:
                    fsdp_stats = _inject_flat_params_into_fsdp_model(model, state)
                    _log(
                        rank,
                        "flat inject: "
                        f"assigned={fsdp_stats.get('assigned', 0)} "
                        f"mismatched={fsdp_stats.get('mismatched', 0)} "
                        f"key_not_found={fsdp_stats.get('key_not_found', 0)} "
                        f"scanned={fsdp_stats.get('scanned', 0)}",
                    )
                    if fsdp_stats.get("assigned", 0) <= 0:
                        raise RuntimeError(
                            "检测到 FSDP flat 权重，但当前运行图未能完成 flat 参数注入。"
                            "请检查 world_size / FSDP wrap 是否与训练一致。"
                        )
                    if fsdp_stats.get("mismatched", 0) > 0:
                        warn(
                            f"flat 注入存在 numel mismatch({fsdp_stats['mismatched']})，"
                            "推理结果可能不可靠。"
                        )
                    if fsdp_stats.get("key_not_found", 0) > 0:
                        warn(
                            f"flat 注入存在 key_not_found({fsdp_stats['key_not_found']})，"
                            "请确认 FSDP 包装边界是否与训练一致。"
                        )
                    total_missing += 0
                    total_unexpected += 0
                    load_mode = "fsdp_flat_inject"
                else:
                    total_missing += int(best_missing)
                    total_unexpected += int(best_unexpected)
            else:
                if flat_key_count > 0:
                    warn(
                        "wan_dit_trainable.* 含 FSDP flat 参数键(_flat_param)。"
                        "当前推理图未启用 FSDP flat 直读，将按普通 state_dict 尝试加载。"
                    )
                missing, unexpected = model.load_state_dict(state, strict=False)
                _log(
                    rank,
                    f"portable load: tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}",
                )
                total_missing += int(len(missing))
                total_unexpected += int(len(unexpected))
                load_mode = "portable_state_dict"
                if unexpected:
                    warn(
                        f"Wan 微调权重出现 unexpected keys({len(unexpected)})，"
                        f"前8个: {unexpected[:8]}"
                    )

    if not loaded_files:
        record_metric("wan_finetune_loaded", 0)
        record_metric("wan_finetune_reason", "empty_state")
        return {
            "loaded": False,
            "reason": "empty_state",
            "checkpoint_dir": str(ckpt_dir),
        }

    wan_mode = str(training_args_excerpt.get("wan_train_mode", "")).strip().lower()
    if wan_mode == "full" and total_missing:
        warn(
            f"训练记录 wan_train_mode=full，但推理加载出现 missing keys({total_missing})。"
            "这通常表示 Wan 权重键名或 FSDP 图不兼容。"
        )
    if wan_mode == "cond_only" and total_missing:
        warn(
            f"训练记录 wan_train_mode=cond_only，但推理加载出现 missing keys({total_missing})。"
            "请检查 wan_cond_name_pattern 与当前 Wan 模型键名是否兼容。"
        )

    effective_ok = 1
    if total_unexpected:
        effective_ok = 0
    if wan_mode in {"full", "cond_only"} and total_missing:
        effective_ok = 0
    if load_mode == "fsdp_flat_inject" and (
        fsdp_stats.get("mismatched", 0) > 0 or fsdp_stats.get("key_not_found", 0) > 0
    ):
        effective_ok = 0

    record_metric("wan_finetune_loaded", 1)
    record_metric("wan_finetune_effective_loaded", int(effective_ok))
    record_metric("wan_finetune_state_tensors", int(total_state_tensors))
    record_metric("wan_finetune_missing", int(total_missing))
    record_metric("wan_finetune_unexpected", int(total_unexpected))
    record_metric("wan_finetune_path", " | ".join(loaded_files))
    record_metric("wan_finetune_load_mode", load_mode)
    record_metric("wan_finetune_checkpoint_dir", str(ckpt_dir))
    if fsdp_stats:
        record_metric("wan_finetune_fsdp_assigned", int(fsdp_stats.get("assigned", 0)))
        record_metric("wan_finetune_fsdp_mismatched", int(fsdp_stats.get("mismatched", 0)))
        record_metric("wan_finetune_fsdp_key_not_found", int(fsdp_stats.get("key_not_found", 0)))
        record_metric("wan_finetune_fsdp_scanned", int(fsdp_stats.get("scanned", 0)))

    return {
        "loaded": True,
        "checkpoint_dir": str(ckpt_dir),
        "picked": str(picked) if picked is not None else "",
        "lora_picked": str(lora_picked) if lora_picked is not None else "",
        "wan_train_mode": wan_mode,
        "flat_key_count": int(flat_key_count),
        "missing": int(total_missing),
        "unexpected": int(total_unexpected),
        "state_tensors": int(total_state_tensors),
        "effective_ok": int(effective_ok),
        "load_mode": load_mode,
        "fsdp_stats": dict(fsdp_stats),
    }
