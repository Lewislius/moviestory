"""
将 FSDP 训练保存的 wan_dit_trainable(_flat_param 键)转换为可普通推理加载的 portable 权重。

典型用途：
  - 训练使用了 --dit_fsdp 且 wan_train_mode=full；
  - 推理在非 FSDP 图上加载时出现 missing/unexpected（尤其 unexpected=_flat_param）。

示例：
  python convert_wan_fsdp_flat_to_portable.py \
    --checkpoint_dir /path/to/checkpoint-600 \
    --wan_checkpoint_dir /path/to/Wan2.2-TI2V-5B \
    --device 0 \
    --replace
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def _inject_flat_params_into_fsdp_model(model: torch.nn.Module, flat_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    当 NO_SHARD + use_orig_params=True 导致 load_state_dict(flat) 失配时，
    直接把 flat 张量按 FSDP 子模块写回 _flat_param，再导出 FULL_STATE_DICT。
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    norm_src = {_normalize_key(k): v for k, v in flat_state.items()}
    assigned = 0
    mismatched = 0
    scanned = 0
    key_not_found = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, FSDP):
            continue
        scanned += 1
        flat_param = getattr(module, "_flat_param", None)
        if flat_param is None and hasattr(module, "_handle") and getattr(module._handle, "flat_param", None) is not None:
            flat_param = module._handle.flat_param
        if flat_param is None or not torch.is_tensor(flat_param):
            continue

        simple_name = _simplify_fsdp_module_name(module_name)
        candidate_key = "_flat_param" if simple_name == "" else f"{simple_name}._flat_param"
        tensor = _resolve_by_suffix(candidate_key, flat_state, norm_src)
        if tensor is None:
            key_not_found += 1
            continue
        if int(tensor.numel()) != int(flat_param.numel()):
            print(
                f"[INJECT][WARN] numel mismatch for {candidate_key}: "
                f"ckpt={int(tensor.numel())} model={int(flat_param.numel())}"
            )
            mismatched += 1
            continue
        with torch.no_grad():
            flat_param.data.copy_(tensor.to(device=flat_param.device, dtype=flat_param.dtype).view_as(flat_param))
        assigned += 1
    print(f"[INJECT][SCAN] fsdp_modules={scanned} key_not_found={key_not_found}")
    return assigned, mismatched


def _remap_flat_state_by_local_template(
    model: torch.nn.Module,
    flat_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    基于 FSDP LOCAL_STATE_DICT 模板键做重映射：
    - 解决 raw checkpoint key（如 blocks.0._flat_param）与当前图 key 不完全一致的问题。
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import LocalStateDictConfig, StateDictType

    cfg = LocalStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, cfg):
        tmpl = model.state_dict()

    norm_src: Dict[str, torch.Tensor] = {_normalize_key(k): v for k, v in flat_state.items()}

    remapped: Dict[str, torch.Tensor] = {}
    mapped = 0
    for tk, tv in tmpl.items():
        if not torch.is_tensor(tv):
            continue
        src = _resolve_by_suffix(tk, flat_state, norm_src)
        if src is None:
            continue
        if int(src.numel()) != int(tv.numel()):
            continue
        remapped[tk] = src.detach().cpu().contiguous().view_as(tv)
        mapped += 1
    print(f"[REMAP] template_keys={sum(1 for _k, _v in tmpl.items() if torch.is_tensor(_v))} mapped={mapped}")
    return remapped


def _default_cond_keywords() -> list[str]:
    return [
        "cross_attn",
        "cross-attn",
        "crossattention",
        "cross_attention",
        "text_embedding",
        "time_projection",
        "modulation",
        "cross_attn_norm",
        "norm3",
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Wan FSDP-flat checkpoint to portable state dict")
    p.add_argument("--checkpoint_dir", type=str, required=True, help="包含 wan_dit_trainable.* 的 checkpoint 目录")
    p.add_argument("--wan_checkpoint_dir", type=str, required=True, help="Wan2.2 TI2V 基座 checkpoint 目录")
    p.add_argument("--device", type=int, default=0, help="CUDA 设备号")
    p.add_argument("--replace", action="store_true", help="转换完成后覆盖 checkpoint 内的 wan_dit_trainable.*")
    p.add_argument(
        "--wan_train_mode_override",
        type=str,
        default="",
        choices=["", "full", "cond_only", "frozen"],
        help="覆盖训练模式判断（默认从 training_args.json 读取）",
    )
    p.add_argument("--master_addr", type=str, default="127.0.0.1")
    p.add_argument("--master_port", type=str, default="29549")
    p.add_argument("--distributed", action="store_true", help="启用/兼容 torchrun 分布式转换")
    return p.parse_args()


def _init_dist_env(device: int, master_addr: str, master_port: str, force_distributed: bool) -> Tuple[int, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = bool(force_distributed or world_size > 1)

    if enabled:
        if not dist.is_available():
            raise RuntimeError("torch.distributed 不可用，无法执行 distributed 转换")
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", master_port)
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            run_device = local_rank if torch.cuda.is_available() else device
            if torch.cuda.is_available():
                torch.cuda.set_device(run_device)
                try:
                    dist.init_process_group(
                        backend=backend,
                        init_method="env://",
                        device_id=torch.device(f"cuda:{run_device}"),
                    )
                except TypeError:
                    dist.init_process_group(backend=backend, init_method="env://")
            else:
                dist.init_process_group(backend=backend, init_method="env://")
            world_size = int(dist.get_world_size())
            rank = int(dist.get_rank())
            local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank)))
        else:
            world_size = int(dist.get_world_size())
            rank = int(dist.get_rank())
        run_device = local_rank if torch.cuda.is_available() else device
    else:
        if dist.is_available() and not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", master_port)
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            run_device = device
            if torch.cuda.is_available():
                torch.cuda.set_device(run_device)
            dist.init_process_group(backend=backend, rank=0, world_size=1)
        else:
            run_device = device
            if torch.cuda.is_available():
                torch.cuda.set_device(run_device)
        world_size = 1
        rank = 0
        local_rank = 0

    return int(world_size), int(rank), int(local_rank), int(run_device)


def _pick_wan_state_file(ckpt_dir: Path) -> Path:
    sf = ckpt_dir / "wan_dit_trainable.safetensors"
    pt = ckpt_dir / "wan_dit_trainable.pt"
    if sf.exists():
        return sf
    if pt.exists():
        return pt
    raise FileNotFoundError(f"未找到 wan_dit_trainable.safetensors 或 wan_dit_trainable.pt: {ckpt_dir}")


def _load_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        payload = _safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"权重文件格式异常: {path}")
        state = {k: v for k, v in payload.items() if torch.is_tensor(v)}
    if not state:
        raise RuntimeError(f"权重文件为空: {path}")
    return state


def _resolve_train_mode(ckpt_dir: Path, mode_override: str) -> tuple[str, str]:
    if mode_override:
        return mode_override, ""
    targs = ckpt_dir / "training_args.json"
    if not targs.exists():
        return "full", ""
    try:
        payload = json.loads(targs.read_text(encoding="utf-8"))
        mode = str(payload.get("wan_train_mode", "full")).strip().lower()
        cond_pattern = str(payload.get("wan_cond_name_pattern", "")).strip()
        if mode not in {"full", "cond_only", "frozen"}:
            mode = "full"
        return mode, cond_pattern
    except Exception:
        return "full", ""


def _read_training_world_size_hint(ckpt_dir: Path) -> int:
    targs = ckpt_dir / "training_args.json"
    if not targs.exists():
        return 0
    try:
        payload = json.loads(targs.read_text(encoding="utf-8"))
    except Exception:
        return 0
    for key in ("world_size", "distributed_world_size", "nproc_per_node", "num_processes", "num_gpus"):
        val = payload.get(key, None)
        if val is None:
            continue
        try:
            ival = int(val)
            if ival > 0:
                return ival
        except Exception:
            continue
    return 0


def main():
    args = _parse_args()
    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir 不存在: {ckpt_dir}")

    in_file = _pick_wan_state_file(ckpt_dir)
    raw_state = _load_state(in_file)
    flat_count = sum(1 for k in raw_state.keys() if "_flat_param" in str(k))
    print(f"[LOAD] input={in_file} tensors={len(raw_state)} flat_keys={flat_count}")

    world_size, rank, local_rank, run_device = _init_dist_env(
        args.device, args.master_addr, args.master_port, bool(args.distributed)
    )
    print(
        f"[DIST] world_size={world_size} rank={rank} local_rank={local_rank} "
        f"run_device={run_device} distributed={int(world_size > 1)}"
    )
    ws_hint = _read_training_world_size_hint(ckpt_dir)
    if ws_hint > 0 and ws_hint != world_size and rank == 0:
        print(
            f"[DIST][WARN] 检测到训练配置可能使用 world_size={ws_hint}，"
            f"但当前转换 world_size={world_size}。FSDP flat checkpoint 在拓扑不一致时可能无法转换。"
        )

    try:
        from wan import WanTI2V
        from wan.configs import WAN_CONFIGS
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, LocalStateDictConfig, StateDictType

        config = WAN_CONFIGS["ti2v-5B"]
        wan = WanTI2V(
            config=config,
            checkpoint_dir=args.wan_checkpoint_dir,
            device_id=run_device,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=True,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=False,
        )

        missing, unexpected = wan.model.load_state_dict(raw_state, strict=False)
        print(f"[LOAD-DEFAULT][rank{rank}] missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print(f"[LOAD-DEFAULT][WARN][rank{rank}] unexpected preview: {unexpected[:8]}")

        if flat_count > 0 and unexpected:
            local_cfg = LocalStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wan.model, StateDictType.LOCAL_STATE_DICT, local_cfg):
                missing_l, unexpected_l = wan.model.load_state_dict(raw_state, strict=False)
            print(f"[LOAD-LOCAL][rank{rank}] missing={len(missing_l)} unexpected={len(unexpected_l)}")
            if unexpected_l:
                print(f"[LOAD-LOCAL][WARN][rank{rank}] unexpected preview: {unexpected_l[:8]}")
            if len(unexpected_l) < len(unexpected):
                missing, unexpected = missing_l, unexpected_l

        if flat_count > 0 and unexpected:
            # 先尝试按 LOCAL_STATE_DICT 模板重映射再加载（优先路径）
            remapped_state = _remap_flat_state_by_local_template(wan.model, raw_state)
            if remapped_state:
                missing2, unexpected2 = wan.model.load_state_dict(remapped_state, strict=False)
                print(
                    f"[LOAD-REMAP][rank{rank}] missing={len(missing2)} unexpected={len(unexpected2)} "
                    f"tensors={len(remapped_state)}"
                )
                if len(unexpected2) <= len(unexpected):
                    missing, unexpected = missing2, unexpected2

        if flat_count > 0 and unexpected:
            assigned, mismatched = _inject_flat_params_into_fsdp_model(wan.model, raw_state)
            print(f"[INJECT][rank{rank}] assigned={assigned} mismatched={mismatched} total_flat={flat_count}")
            if assigned <= 0:
                raise RuntimeError(
                    "检测到 FSDP flat 权重，但当前运行图未能完成 flat 参数注入。"
                    "建议在与训练一致的多卡拓扑下执行转换，或检查 checkpoint 是否损坏。"
                )

        with FSDP.state_dict_type(
            wan.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_state = wan.model.state_dict()

        if world_size > 1:
            dist.barrier()
        if rank != 0:
            print(f"[DONE][rank{rank}] non-main rank finished conversion collectives")
            return

        portable_state: Dict[str, torch.Tensor] = {}
        for name, tensor in full_state.items():
            if not torch.is_tensor(tensor):
                continue
            portable_state[_normalize_key(name)] = tensor.detach().cpu().contiguous()

        train_mode, cond_pattern = _resolve_train_mode(ckpt_dir, args.wan_train_mode_override)
        if train_mode == "frozen":
            portable_state = {}
        elif train_mode == "cond_only":
            if cond_pattern:
                kws = [k.strip().lower() for k in cond_pattern.split(",") if k.strip()]
            else:
                kws = _default_cond_keywords()
            portable_state = {
                k: v for k, v in portable_state.items()
                if any(kw in k.lower() for kw in kws)
            }
        print(
            f"[EXPORT] mode={train_mode} tensors={len(portable_state)} "
            f"params={sum(int(t.numel()) for t in portable_state.values()):,}"
        )
        if train_mode == "full" and len(portable_state) == 0:
            raise RuntimeError("train_mode=full 但导出结果为空，转换失败。")

        out_sf = ckpt_dir / "wan_dit_trainable_portable.safetensors"
        out_pt = ckpt_dir / "wan_dit_trainable_portable.pt"

        try:
            from safetensors.torch import save_file

            save_file(portable_state, str(out_sf))
            print(f"[SAVE] {out_sf}")
        except Exception as e:
            print(f"[SAVE][WARN] safetensors 保存失败，将仅写 pt: {e}")
        torch.save(portable_state, out_pt)
        print(f"[SAVE] {out_pt}")

        if args.replace:
            bak_sf = ckpt_dir / "wan_dit_trainable.fsdp_flat.bak.safetensors"
            bak_pt = ckpt_dir / "wan_dit_trainable.fsdp_flat.bak.pt"
            if (ckpt_dir / "wan_dit_trainable.safetensors").exists():
                if bak_sf.exists():
                    bak_sf.unlink()
                (ckpt_dir / "wan_dit_trainable.safetensors").rename(bak_sf)
                print(f"[BACKUP] {bak_sf}")
            if (ckpt_dir / "wan_dit_trainable.pt").exists():
                if bak_pt.exists():
                    bak_pt.unlink()
                (ckpt_dir / "wan_dit_trainable.pt").rename(bak_pt)
                print(f"[BACKUP] {bak_pt}")
            if out_sf.exists():
                out_sf.replace(ckpt_dir / "wan_dit_trainable.safetensors")
                print(f"[REPLACE] {ckpt_dir / 'wan_dit_trainable.safetensors'}")
            out_pt.replace(ckpt_dir / "wan_dit_trainable.pt")
            print(f"[REPLACE] {ckpt_dir / 'wan_dit_trainable.pt'}")

        print("[DONE] conversion finished")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
