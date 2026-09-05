#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import numbers
import os
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from PIL import Image
from tqdm import tqdm


# All checkpoints handled by this inference path were trained with FA2.  Some
# Determined workers also expose FA3, which Wan otherwise prefers implicitly.
# Pin the backend before importing any Wan module so identical jobs do not pick
# a different attention kernel according to the worker they land on.  Assign
# instead of setdefault: an inherited empty/FA3 value is incompatible with
# checkpoints trained with FA2.
os.environ["WAN_FLASH_ATTN_FORCE_VERSION"] = "2"
os.environ["WAN_FLASH_ATTN_FORCE_CONTIGUOUS"] = "1"


INFERENCE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = INFERENCE_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
HOME_ROOT = PROJECT_ROOT.parents[1]
WAN_ROOT = HOME_ROOT / "model" / "Wan2.2"
WAN_TRAIN_ROOT = WAN_ROOT / "scripts-metaquery-single" / "train"
METAQUERY_ROOT = HOME_ROOT / "model" / "Qwen3-VL-main" / "metaquery-main"
DEFAULT_CHECKPOINT = (
    CODE_ROOT
    / "checkpoint"
    / "three_router_mq256_conn24_legacysoft_openvid4000_steps500"
    / "checkpoint-final"
)

for path in (
    INFERENCE_ROOT,
    CODE_ROOT,
    WAN_TRAIN_ROOT,
    WAN_ROOT,
    METAQUERY_ROOT,
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from three_router_planner import (  # noqa: E402
    ThreeRouterConfig,
    build_three_router_encoder_class,
)


ROUTE_NAMES = ("role", "action", "global")
EXPECTED_MODALITIES = {
    "role": "reference_image",
    "action": "text",
    "global": "reference_image+text",
}
ROUTE_STATE_KEYS = {
    route: f"route_metaquery_embeddings.{route}" for route in ROUTE_NAMES
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict inference for the trained MovieStory three-router MQ + "
            "Wan2.2 TI2V checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint directory, or a run directory containing a latest pointer.",
    )
    parser.add_argument(
        "--wan_checkpoint_dir",
        default=str(WAN_ROOT / "Wan2.2-TI2V-5B"),
    )
    parser.add_argument(
        "--qwen3vl_model_id",
        default=str(HOME_ROOT / "model/Qwen3-VL-main/Qwen3-VL-2B-Thinking"),
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--ref_image", default="")
    parser.add_argument(
        "--output_path",
        default=str(CODE_ROOT / "inference_outputs/three_router_result.mp4"),
    )
    parser.add_argument("--verify_report_path", default="")
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--max_area", type=int, default=262144)
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument(
        "--guide_scale",
        type=float,
        default=1.0,
        help=(
            "CFG scale. The safe default is 1 (conditional prediction only); "
            "values >1 require an explicitly trained null branch."
        ),
    )
    parser.add_argument(
        "--cfg_uncond_mode",
        choices=("empty_mq", "negative_mq", "zero_mq"),
        default="empty_mq",
        help=(
            "Unconditional context for CFG>1. empty_mq matches joint-null "
            "training; negative_mq and zero_mq are explicit ablations."
        ),
    )
    parser.add_argument(
        "--first_frame_mode",
        choices=("preserved", "none"),
        default="preserved",
        help=(
            "preserved uses a clean timestep-zero reference latent slot and "
            "re-locks it after every solver step; none is a no-anchor ablation."
        ),
    )
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument(
        "--sample_solver",
        choices=("unipc", "dpm++"),
        default="unipc",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--encoder_device", type=int, default=0)
    parser.add_argument("--dit_device", type=int, default=1)
    parser.add_argument(
        "--runtime_audit",
        choices=("basic", "full"),
        default="full",
        help="full also runs image/text ablations through all three Qwen routes.",
    )
    parser.add_argument(
        "--audit_epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--audit_forward_retries",
        type=int,
        default=1,
        help=(
            "Number of synchronized Wan forward retries used to confirm an "
            "exact-zero conditioning audit before aborting."
        ),
    )
    parser.add_argument(
        "--audit_growth_limit",
        type=float,
        default=20.0,
        help="Abort when CFG amplification or step-to-step latent RMS growth exceeds this.",
    )
    parser.add_argument(
        "--max_noise_high_frequency_ratio",
        type=float,
        default=0.9,
        help="Fail decoded-video audit above this spatial high-frequency ratio.",
    )
    parser.add_argument(
        "--min_reference_frame_correlation",
        type=float,
        default=0.5,
        help="Minimum decoded first-frame correlation with the processed reference.",
    )
    parser.add_argument(
        "--max_reference_frame_mae",
        type=float,
        default=0.4,
        help="Maximum first-frame/reference MAE in normalized [-1,1] space.",
    )
    parser.add_argument(
        "--offload_model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--t5_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use T5 only to reproduce training-time MQ RMS matching.",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Validate checkpoint files/configuration without loading the models.",
    )
    parser.add_argument(
        "--parse_only",
        action="store_true",
        help="Parse arguments and print the resolved command configuration.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.guide_scale) or args.guide_scale < 1.0:
        parser.error("--guide_scale must be finite and >= 1.0")
    if (
        not math.isfinite(args.audit_growth_limit)
        or args.audit_growth_limit <= 1.0
    ):
        parser.error("--audit_growth_limit must be finite and > 1.0")
    if not math.isfinite(args.audit_epsilon) or args.audit_epsilon <= 0.0:
        parser.error("--audit_epsilon must be finite and > 0")
    if not 0 <= args.audit_forward_retries <= 3:
        parser.error("--audit_forward_retries must be within [0, 3]")
    if (
        not math.isfinite(args.max_noise_high_frequency_ratio)
        or not 0.0 < args.max_noise_high_frequency_ratio <= 2.0
    ):
        parser.error("--max_noise_high_frequency_ratio must be within (0, 2]")
    if (
        not math.isfinite(args.min_reference_frame_correlation)
        or not -1.0 <= args.min_reference_frame_correlation <= 1.0
    ):
        parser.error("--min_reference_frame_correlation must be within [-1, 1]")
    if (
        not math.isfinite(args.max_reference_frame_mae)
        or not 0.0 < args.max_reference_frame_mae <= 2.0
    ):
        parser.error("--max_reference_frame_mae must be within (0, 2]")
    return args


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Verification report contains non-finite float: {value}")
        return numeric_value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().item())
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {value}")
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot read JSON file {path}: {exc}") from exc
    payload = _jsonable(payload)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _resolve_checkpoint_dir(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_file():
        if path.name == "latest":
            target = path.read_text(encoding="utf-8").strip()
            if not target:
                raise RuntimeError(f"Empty latest pointer: {path}")
            path = (path.parent / target).resolve()
        else:
            path = path.parent
    if (path / "latest").is_file() and not (path / "config.json").is_file():
        target = (path / "latest").read_text(encoding="utf-8").strip()
        if not target:
            raise RuntimeError(f"Empty latest pointer: {path / 'latest'}")
        path = (path / target).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    return path


def _pick_tensor_file(
    checkpoint_dir: Path,
    stem: str,
    *,
    required: bool = True,
) -> Path | None:
    candidates = (
        checkpoint_dir / f"{stem}.safetensors",
        checkpoint_dir / f"{stem}.pt",
    )
    picked = next((path for path in candidates if path.is_file()), None)
    if picked is None and required:
        raise FileNotFoundError(
            f"Missing {stem}.safetensors or {stem}.pt in {checkpoint_dir}"
        )
    return picked


def _inspect_tensor_file(path: Path) -> Dict[str, tuple[int, ...]]:
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError(
                "safetensors is required to inspect the checkpoint. "
                "Activate the moviestory environment first."
            ) from exc
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {
                key: tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                for key in handle.keys()
            }
    state = _load_tensor_file(path)
    try:
        return {
            key: tuple(int(dim) for dim in tensor.shape)
            for key, tensor in state.items()
        }
    finally:
        del state
        gc.collect()


def _load_tensor_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(
                "safetensors is required to load this checkpoint"
            ) from exc
        state = load_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    if (
        not isinstance(state, dict)
        or not state
        or any(not isinstance(key, str) for key in state)
        or any(not torch.is_tensor(value) for value in state.values())
    ):
        raise RuntimeError(f"Invalid tensor state dictionary: {path}")
    return state


def _load_tensor_rows(
    path: Path,
    key: str,
    start_row: int,
) -> torch.Tensor:
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError(
                "safetensors is required to load frozen special-token rows"
            ) from exc
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise RuntimeError(f"Tensor {key} is absent from {path}")
            return handle.get_slice(key)[start_row:].clone()
    state = _load_tensor_file(path)
    try:
        if key not in state:
            raise RuntimeError(f"Tensor {key} is absent from {path}")
        return state[key][start_row:].clone()
    finally:
        del state
        gc.collect()


@dataclass(frozen=True)
class CheckpointBundle:
    directory: Path
    config: Dict[str, Any]
    router: Dict[str, Any]
    training_args: Dict[str, Any]
    trainer_state: Dict[str, Any]
    mq_state_path: Path
    wan_state_path: Path
    full_mq_path: Path
    full_embedding_key: str
    base_embedding_rows: int
    total_embedding_rows: int
    router_config: ThreeRouterConfig
    report: Dict[str, Any]


def validate_checkpoint_bundle(path_value: str | Path) -> CheckpointBundle:
    checkpoint_dir = _resolve_checkpoint_dir(path_value)
    required_json = (
        "config.json",
        "three_router_config.json",
        "training_args.json",
        "trainer_state.json",
        "metrics_tail.json",
    )
    missing = [
        name for name in required_json if not (checkpoint_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint is missing required metadata files: {missing}"
        )

    config = _read_json(checkpoint_dir / "config.json")
    router = _read_json(checkpoint_dir / "three_router_config.json")
    training_args = _read_json(checkpoint_dir / "training_args.json")
    trainer_state = _read_json(checkpoint_dir / "trainer_state.json")
    metrics_tail = _read_json(checkpoint_dir / "metrics_tail.json")
    router_config = ThreeRouterConfig(
        hidden_size=int(router.get("hidden_size", 0)),
        role_tokens=int(router.get("role_tokens", 0)),
        action_tokens=int(router.get("action_tokens", 0)),
        global_tokens=int(router.get("global_tokens", 0)),
    )

    exact_checks = {
        "router_enabled": router.get("enabled") is True,
        "routing_mode": router.get("routing_mode") == "isolated_modalities_v1",
        "route_modalities": router.get("route_modalities") == EXPECTED_MODALITIES,
        "joint_connector": router.get("joint_connector_forward") is True,
        "no_separate_connector": (
            router.get("separate_connector_forwards") is False
        ),
        "route_tables_trainable": (
            router.get("trainable_route_metaquery_parameters") is True
        ),
        "total_mq_tokens": router_config.total_tokens == 256,
        "connector_layers": int(config.get("connector_num_hidden_layers", 0))
        == 24,
        "wan_text_dim": int(config.get("wan_text_dim", 0)) == 4096,
        "dit_condition_mq_only": (
            training_args.get("dit_condition_mode") == "mq_only"
        ),
        "wan_train_mode_cond_only": config.get("wan_train_mode") == "cond_only",
        "wan_update_saved": config.get("has_wan_dit_trainable_pt") is True,
        "training_completed": int(trainer_state.get("global_step", 0)) > 0,
    }
    first_frame = router.get("wan_first_frame_conditioning", {})
    checkpoint_first_frame_mode = (
        first_frame.get("mode") if isinstance(first_frame, dict) else None
    )
    exact_checks.update(
        {
            "supported_first_frame_metadata": (
                isinstance(first_frame, dict)
                and first_frame.get("enabled") is True
                and checkpoint_first_frame_mode
                in {
                    "legacy_t2v_soft_anchor",
                    "clean_preserved_latent_slot",
                }
            ),
            "mq_norm_match_t5": (
                isinstance(router.get("mq_t5_rms"), dict)
                and router["mq_t5_rms"].get("match_enabled") is True
            ),
        }
    )
    failed = [name for name, passed in exact_checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Checkpoint metadata is incompatible with strict three-router "
            f"inference: {failed}"
        )

    mq_state_path = _pick_tensor_file(checkpoint_dir, "mq_encoder_trainable")
    wan_state_path = _pick_tensor_file(checkpoint_dir, "wan_dit_trainable")
    assert mq_state_path is not None
    assert wan_state_path is not None
    full_mq_path = next(
        (
            checkpoint_dir / name
            for name in ("model.safetensors", "mq_encoder_full.pt")
            if (checkpoint_dir / name).is_file()
        ),
        None,
    )
    if full_mq_path is None:
        raise FileNotFoundError(
            "Checkpoint lacks model.safetensors or mq_encoder_full.pt"
        )

    mq_shapes = _inspect_tensor_file(mq_state_path)
    wan_shapes = _inspect_tensor_file(wan_state_path)
    full_shapes = _inspect_tensor_file(full_mq_path)
    embedding_keys = [
        key
        for key in full_shapes
        if key.endswith("language_model.embed_tokens.weight")
    ]
    if len(embedding_keys) != 1:
        raise RuntimeError(
            "Expected one Qwen input embedding tensor in the full checkpoint, "
            f"found {embedding_keys}"
        )
    full_embedding_key = embedding_keys[0]
    base_embedding_rows = int(config.get("mllm_embed_rows_base", 0))
    total_embedding_rows = int(config.get("mllm_embed_rows_total", 0))
    expected_total_rows = base_embedding_rows + router_config.total_tokens + 2
    if (
        base_embedding_rows <= 0
        or total_embedding_rows != expected_total_rows
        or full_shapes[full_embedding_key]
        != (total_embedding_rows, router_config.hidden_size)
    ):
        raise RuntimeError(
            "Full-checkpoint Qwen embedding layout is incompatible: "
            f"base={base_embedding_rows}, total={total_embedding_rows}, "
            f"shape={full_shapes[full_embedding_key]}"
        )
    expected_route_shapes = {
        "role": (router_config.role_tokens, router_config.hidden_size),
        "action": (router_config.action_tokens, router_config.hidden_size),
        "global": (router_config.global_tokens, router_config.hidden_size),
    }
    for route_name, expected_shape in expected_route_shapes.items():
        key = ROUTE_STATE_KEYS[route_name]
        actual_shape = mq_shapes.get(key)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{key} shape mismatch: expected {expected_shape}, got "
                f"{actual_shape}"
            )
    connector_keys = [
        key for key in mq_shapes if key.startswith("mllm_model.connector.")
    ]
    if not connector_keys:
        raise RuntimeError("MQ trainable checkpoint has no Connector weights")
    expected_wan_count = int(
        config.get("extra_info", {}).get("wan_trainable_tensor_count", 0)
    )
    if expected_wan_count <= 0 or len(wan_shapes) != expected_wan_count:
        raise RuntimeError(
            "Wan trainable tensor count mismatch: "
            f"metadata={expected_wan_count}, file={len(wan_shapes)}"
        )
    if any("_flat_param" in key for key in wan_shapes):
        raise RuntimeError("Wan checkpoint contains non-portable FSDP flat keys")

    metric_records = metrics_tail.get("records")
    if not isinstance(metric_records, list) or not metric_records:
        raise RuntimeError("metrics_tail.json has no training records")
    final_metrics = max(
        (row for row in metric_records if isinstance(row, dict)),
        key=lambda row: int(row.get("train/step", -1)),
    )
    update_evidence = {
        "step": int(final_metrics.get("train/step", -1)),
        "all_route_updates_applied": float(
            final_metrics.get("train/router_all_updates_applied", 0.0)
        ),
        "joint_parameter_sample_l2_delta": float(
            final_metrics.get("train/param_sample_l2_delta", 0.0)
        ),
        "routes": {
            route: {
                "initial_delta_rms": float(
                    final_metrics.get(
                        f"train/router_{route}_initial_delta_rms",
                        0.0,
                    )
                ),
                "last_step_update_rms": float(
                    final_metrics.get(
                        f"train/router_{route}_step_update_rms",
                        0.0,
                    )
                ),
                "last_step_update_applied": float(
                    final_metrics.get(
                        f"train/router_{route}_update_applied",
                        0.0,
                    )
                ),
            }
            for route in ROUTE_NAMES
        },
    }
    update_failures = []
    if update_evidence["step"] != int(trainer_state["global_step"]):
        update_failures.append("metrics step does not match trainer global_step")
    if update_evidence["all_route_updates_applied"] != 1.0:
        update_failures.append("final all-route update flag is not 1")
    if update_evidence["joint_parameter_sample_l2_delta"] <= 0.0:
        update_failures.append("joint parameter sample has no cumulative delta")
    for route, values in update_evidence["routes"].items():
        if values["initial_delta_rms"] <= 0.0:
            update_failures.append(f"{route} has no cumulative update")
        if values["last_step_update_rms"] <= 0.0:
            update_failures.append(f"{route} has no final-step update")
        if values["last_step_update_applied"] != 1.0:
            update_failures.append(f"{route} final update was not applied")
    if update_failures:
        raise RuntimeError(
            "Checkpoint lacks proof that all trained routes were updated: "
            f"{update_failures}"
        )

    compatibility_warnings = []
    if checkpoint_first_frame_mode == "legacy_t2v_soft_anchor":
        compatibility_warnings.append(
            "Checkpoint was trained with the deprecated, flow-inconsistent "
            "legacy soft anchor. Inference will use the corrected preserved "
            "reference slot, but this cannot retroactively repair training."
        )
    report = {
        "status": (
            "pass_with_warning" if compatibility_warnings else "pass"
        ),
        "warnings": compatibility_warnings,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_step": int(trainer_state["global_step"]),
        "checks": exact_checks,
        "files": {
            "mq_trainable": mq_state_path,
            "mq_trainable_bytes": mq_state_path.stat().st_size,
            "wan_trainable": wan_state_path,
            "wan_trainable_bytes": wan_state_path.stat().st_size,
            "full_mq": full_mq_path,
            "full_mq_artifact_present": True,
        },
        "tensor_layout": {
            "mq_tensor_count": len(mq_shapes),
            "connector_tensor_count": len(connector_keys),
            "route_shapes": {
                route: mq_shapes[ROUTE_STATE_KEYS[route]]
                for route in ROUTE_NAMES
            },
            "wan_tensor_count": len(wan_shapes),
            "qwen_frozen_special_token_rows": (
                total_embedding_rows - base_embedding_rows
            ),
            "qwen_full_embedding_shape": full_shapes[full_embedding_key],
        },
        "training_update_evidence": update_evidence,
        "conditioning_contract": {
            "dit_context": "mq_only",
            "dit_context_tokens": router_config.total_tokens,
            "t5_usage": "RMS matching only; T5 tokens are not sent to Wan DiT",
            "checkpoint_training_first_frame_mode": checkpoint_first_frame_mode,
            "checkpoint_training_timestep_zero": first_frame.get(
                "timestep_zero"
            ),
            "route_modalities": EXPECTED_MODALITIES,
        },
    }
    return CheckpointBundle(
        directory=checkpoint_dir,
        config=config,
        router=router,
        training_args=training_args,
        trainer_state=trainer_state,
        mq_state_path=mq_state_path,
        wan_state_path=wan_state_path,
        full_mq_path=full_mq_path,
        full_embedding_key=full_embedding_key,
        base_embedding_rows=base_embedding_rows,
        total_embedding_rows=total_embedding_rows,
        router_config=router_config,
        report=report,
    )


def _tensor_rms(tensor: torch.Tensor) -> float:
    value = tensor.detach().float()
    if value.numel() == 0:
        raise ValueError("Cannot compute RMS of an empty tensor")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Tensor contains NaN/Inf")
    rms = float(value.square().mean().sqrt().item())
    if not math.isfinite(rms):
        raise FloatingPointError("Tensor RMS is non-finite")
    return rms


def _tensor_stats(tensor: torch.Tensor, *, label: str) -> Dict[str, Any]:
    value = tensor.detach().float()
    if value.numel() == 0:
        raise ValueError(f"{label} is empty")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} contains NaN/Inf")
    stats = {
        "shape": list(value.shape),
        "rms": _tensor_rms(value),
        "std": float(value.std(unbiased=False).item()),
        "absmax": float(value.abs().max().item()),
        "finite": True,
    }
    if any(
        not math.isfinite(float(stats[name]))
        for name in ("rms", "std", "absmax")
    ):
        raise FloatingPointError(f"{label} statistics are non-finite: {stats}")
    return stats


def _tensor_difference_stats(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    label: str,
) -> Dict[str, Any]:
    """Measure whether a conditioning ablation survives a processing stage."""
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(
            f"{label} shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    left_float = left.detach().float()
    right_float = right.detach().float().to(left_float.device)
    difference = left_float - right_float
    if not bool(torch.isfinite(difference).all()):
        raise FloatingPointError(f"{label} difference contains NaN/Inf")
    diff_rms = _tensor_rms(difference)
    reference_rms = _tensor_rms(left_float)
    max_abs = float(difference.abs().max().item())
    changed_fraction = float(
        difference.ne(0).to(dtype=torch.float32).mean().item()
    )
    return {
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "diff_rms": diff_rms,
        "relative_diff": diff_rms / (reference_rms + 1e-8),
        "max_abs_diff": max_abs,
        "changed_fraction": changed_fraction,
        "exact_equal": bool(torch.equal(left, right)),
    }


def _wan_attention_backend_report(attention_module: Any) -> Dict[str, Any]:
    """Describe the backend actually selected by Wan's direct FA call."""
    fa2 = bool(getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False))
    fa3 = bool(getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False))
    forced_version = getattr(
        attention_module, "_FLASH_ATTN_FORCE_VERSION", None
    )
    if forced_version == 2:
        if not fa2:
            raise RuntimeError(
                "Training-aligned FlashAttention 2 was requested but is not "
                "available in this runtime"
            )
        effective = "flash_attention_2"
    elif forced_version == 3:
        if not fa3:
            raise RuntimeError(
                "FlashAttention 3 was requested but is not available"
            )
        effective = "flash_attention_3"
    elif fa3:
        effective = "flash_attention_3_auto"
    elif fa2:
        effective = "flash_attention_2_auto"
    else:
        raise RuntimeError("Wan requires FlashAttention 2 or 3")
    return {
        "effective": effective,
        "forced_version": forced_version,
        "flash_attention_2_available": fa2,
        "flash_attention_3_available": fa3,
        "force_contiguous": bool(
            getattr(attention_module, "_FLASH_ATTN_FORCE_CONTIGUOUS", False)
        ),
        "training_aligned": forced_version == 2,
    }


def _configure_wan_flash_attention_2(
    attention_module: Any,
) -> Dict[str, Any]:
    """Force FA2 even when Wan attention was imported before this entrypoint.

    Wan reads its environment switches only once, at module import time.  A
    parent package can import ``wan.modules.attention`` before this inference
    module gets control, leaving the cached force-version value at ``None``.
    Updating both the process environment and the already-loaded module keeps
    the public launcher and the actual direct attention call in agreement.
    """
    if not bool(
        getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False)
    ):
        raise RuntimeError(
            "Training-aligned FlashAttention 2 is not available in this runtime"
        )

    previous_version = getattr(
        attention_module, "_FLASH_ATTN_FORCE_VERSION", None
    )
    previous_contiguous = bool(
        getattr(attention_module, "_FLASH_ATTN_FORCE_CONTIGUOUS", False)
    )
    os.environ["WAN_FLASH_ATTN_FORCE_VERSION"] = "2"
    os.environ["WAN_FLASH_ATTN_FORCE_CONTIGUOUS"] = "1"
    setattr(attention_module, "_FLASH_ATTN_FORCE_VERSION_STR", "2")
    setattr(attention_module, "_FLASH_ATTN_FORCE_VERSION", 2)
    setattr(attention_module, "_FLASH_ATTN_FORCE_CONTIGUOUS", True)

    report = _wan_attention_backend_report(attention_module)
    report.update(
        {
            "forced_version_before_runtime_config": previous_version,
            "force_contiguous_before_runtime_config": previous_contiguous,
            "runtime_force_applied": (
                previous_version != 2 or not previous_contiguous
            ),
        }
    )
    if not report["training_aligned"]:
        raise RuntimeError(
            "Unable to configure the Wan attention module for FlashAttention 2"
        )
    return report


def assess_decoded_video_quality(
    video: torch.Tensor,
    reference_tensor: torch.Tensor,
    *,
    max_high_frequency_ratio: float,
    min_reference_correlation: float,
    max_reference_mae: float,
) -> Dict[str, Any]:
    """Detect a broken reference frame and snow-like high-frequency output."""
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] < 2:
        raise ValueError(
            f"quality audit expects RGB [C,T,H,W] video with T>=2, got {tuple(video.shape)}"
        )
    if (
        reference_tensor.ndim != 4
        or reference_tensor.shape[0] != 3
        or reference_tensor.shape[1] < 1
        or reference_tensor.shape[2:] != video.shape[2:]
    ):
        raise ValueError(
            "reference tensor is incompatible with decoded video: "
            f"{tuple(reference_tensor.shape)} vs {tuple(video.shape)}"
        )
    decoded = video.detach().float()
    reference = reference_tensor[:, 0].to(
        device=decoded.device,
        dtype=torch.float32,
    )
    first_frame = decoded[:, 0]
    _tensor_stats(decoded, label="decoded video quality input")
    _tensor_stats(reference, label="decoded video quality reference")

    first_frame_mae = float((first_frame - reference).abs().mean().item())
    first_centered = first_frame - first_frame.mean()
    reference_centered = reference - reference.mean()
    correlation_denominator = (
        first_centered.square().sum().sqrt()
        * reference_centered.square().sum().sqrt()
    )
    first_frame_correlation = float(
        (
            (first_centered * reference_centered).sum()
            / correlation_denominator.clamp_min(1e-8)
        ).item()
    )

    # Exclude the preserved first frame so it cannot hide noisy generated slots.
    generated_frames = decoded[:, 1:].permute(1, 0, 2, 3)
    blurred = torch.nn.functional.avg_pool2d(
        generated_frames,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    high_frequency_rms = _tensor_rms(generated_frames - blurred)
    centered_generated_rms = _tensor_rms(
        generated_frames - generated_frames.mean()
    )
    high_frequency_ratio = high_frequency_rms / (
        centered_generated_rms + 1e-8
    )

    gray = generated_frames.mean(dim=1)
    if gray.shape[0] > 1:
        left = gray[:-1].reshape(gray.shape[0] - 1, -1)
        right = gray[1:].reshape(gray.shape[0] - 1, -1)
        left = left - left.mean(dim=1, keepdim=True)
        right = right - right.mean(dim=1, keepdim=True)
        temporal_correlations = (
            (left * right).sum(dim=1)
            / (
                left.square().sum(dim=1).sqrt()
                * right.square().sum(dim=1).sqrt()
            ).clamp_min(1e-8)
        )
        adjacent_frame_correlation = float(
            temporal_correlations.mean().item()
        )
    else:
        adjacent_frame_correlation = 1.0

    metrics = {
        "first_frame_reference_mae": first_frame_mae,
        "first_frame_reference_correlation": first_frame_correlation,
        "generated_spatial_high_frequency_rms": high_frequency_rms,
        "generated_spatial_high_frequency_ratio": high_frequency_ratio,
        "generated_adjacent_frame_correlation": adjacent_frame_correlation,
        "thresholds": {
            "max_reference_mae": float(max_reference_mae),
            "min_reference_correlation": float(min_reference_correlation),
            "max_high_frequency_ratio": float(max_high_frequency_ratio),
        },
    }
    if any(
        not math.isfinite(float(value))
        for key, value in metrics.items()
        if key != "thresholds"
    ):
        raise FloatingPointError(
            f"Decoded-video quality metrics are non-finite: {metrics}"
        )
    failures = []
    if first_frame_mae > max_reference_mae:
        failures.append(
            f"first-frame MAE {first_frame_mae:.6f} > {max_reference_mae:.6f}"
        )
    if first_frame_correlation < min_reference_correlation:
        failures.append(
            "first-frame correlation "
            f"{first_frame_correlation:.6f} < {min_reference_correlation:.6f}"
        )
    if high_frequency_ratio > max_high_frequency_ratio:
        failures.append(
            "generated high-frequency ratio "
            f"{high_frequency_ratio:.6f} > {max_high_frequency_ratio:.6f}"
        )
    metrics["status"] = "fail" if failures else "pass"
    metrics["failures"] = failures
    return metrics


def enforce_clean_reference_prefix_(
    latent: torch.Tensor,
    reference_latent: torch.Tensor,
) -> torch.Tensor:
    """In-place lock of the clean first latent slot used as I2V condition."""
    if latent.ndim != 4 or reference_latent.ndim != 4:
        raise ValueError(
            "latent/reference_latent must be [C,T,H,W], got "
            f"{tuple(latent.shape)} and {tuple(reference_latent.shape)}"
        )
    if (
        latent.shape[0] != reference_latent.shape[0]
        or latent.shape[2:] != reference_latent.shape[2:]
        or latent.shape[1] < 2
        or reference_latent.shape[1] < 1
    ):
        raise ValueError(
            "reference latent is incompatible with generated latent: "
            f"{tuple(reference_latent.shape)} vs {tuple(latent.shape)}"
        )
    latent[:, :1].copy_(
        reference_latent[:, :1].to(device=latent.device, dtype=latent.dtype)
    )
    return latent


def build_model_timestep_row(
    timestep_value: torch.Tensor,
    *,
    seq_len: int,
    tokens_per_latent_frame: int,
    preserve_first_frame: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build Wan token timesteps, assigning t=0 to the preserved prefix."""
    if seq_len <= 0 or tokens_per_latent_frame <= 0:
        raise ValueError("seq_len and tokens_per_latent_frame must be positive")
    row = (
        timestep_value.to(device=device, dtype=torch.float32)
        .reshape(1, 1)
        .expand(1, seq_len)
        .clone()
    )
    if preserve_first_frame:
        row[:, : min(seq_len, tokens_per_latent_frame)] = 0.0
    return row


def match_mq_rms(
    mq_features: torch.Tensor,
    t5_features: torch.Tensor,
    clip_min: float,
    clip_max: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    mq_rms = _tensor_rms(mq_features)
    t5_rms = _tensor_rms(t5_features)
    raw_scale = t5_rms / (mq_rms + 1e-8)
    scale = max(float(clip_min), min(float(clip_max), raw_scale))
    matched = mq_features * scale
    return matched, {
        "mq_rms_before": mq_rms,
        "t5_rms": t5_rms,
        "raw_scale": raw_scale,
        "applied_scale": scale,
        "mq_rms_after": _tensor_rms(matched),
    }


def _load_subset_strict(
    module: torch.nn.Module,
    state_path: Path,
    *,
    label: str,
    state: Dict[str, torch.Tensor] | None = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if state is None:
        state = _load_tensor_file(state_path)
    model_state = module.state_dict()
    missing_in_model = [key for key in state if key not in model_state]
    shape_mismatches = {
        key: {
            "checkpoint": tuple(state[key].shape),
            "model": tuple(model_state[key].shape),
        }
        for key in state
        if key in model_state and tuple(state[key].shape) != tuple(model_state[key].shape)
    }
    if missing_in_model or shape_mismatches:
        raise RuntimeError(
            f"{label} checkpoint does not match the model: "
            f"missing_in_model={missing_in_model[:8]}, "
            f"shape_mismatches={dict(list(shape_mismatches.items())[:4])}"
        )
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected {label} keys: {incompatible.unexpected_keys[:8]}"
        )
    return state, {
        "path": state_path,
        "loaded_tensor_count": len(state),
        "model_missing_keys_ignored": len(incompatible.missing_keys),
        "shape_mismatch_count": 0,
        "unexpected_key_count": 0,
    }


class ThreeRouterWanInference:
    def __init__(self, args: argparse.Namespace, bundle: CheckpointBundle) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for full inference")
        self.args = args
        self.bundle = bundle
        self.encoder_device = torch.device(f"cuda:{args.encoder_device}")
        self.dit_device = torch.device(f"cuda:{args.dit_device}")
        self.report: Dict[str, Any] = {
            "checkpoint_validation": bundle.report,
            "model_loading": {},
            "runtime": {},
        }
        self._validate_model_paths()
        self._load_wan()
        self._load_mq_encoder()
        self._load_wan_updates()

    def _validate_model_paths(self) -> None:
        wan_path = Path(self.args.wan_checkpoint_dir).expanduser().resolve()
        qwen_path = Path(self.args.qwen3vl_model_id).expanduser().resolve()
        if not wan_path.is_dir():
            raise FileNotFoundError(f"Wan checkpoint directory not found: {wan_path}")
        if not qwen_path.exists():
            raise FileNotFoundError(f"Qwen model path not found: {qwen_path}")
        trained_wan = str(self.bundle.training_args.get("wan_checkpoint_dir", ""))
        trained_qwen = str(self.bundle.config.get("qwen3vl_model_id", ""))
        mismatches = {}
        if trained_wan and Path(trained_wan).expanduser().resolve() != wan_path:
            mismatches["wan"] = {"trained": trained_wan, "inference": str(wan_path)}
        if trained_qwen and Path(trained_qwen).expanduser().resolve() != qwen_path:
            mismatches["qwen"] = {
                "trained": trained_qwen,
                "inference": str(qwen_path),
            }
        if mismatches:
            raise RuntimeError(
                "Base model paths differ from the checkpoint metadata: "
                f"{mismatches}"
            )

    def _load_wan(self) -> None:
        from wan import WanTI2V
        from wan.configs import WAN_CONFIGS
        from wan.modules import attention as wan_attention

        config = WAN_CONFIGS["ti2v-5B"]
        attention_backend = _configure_wan_flash_attention_2(wan_attention)
        self.wan = WanTI2V(
            config=config,
            checkpoint_dir=self.args.wan_checkpoint_dir,
            device_id=self.args.dit_device,
            rank=0,
            t5_cpu=bool(self.args.t5_cpu),
            init_on_cpu=True,
        )
        self.wan_config = config
        self.original_text_len = int(self.wan.model.text_len)
        if int(self.wan.model.text_dim) != 4096:
            raise RuntimeError(
                f"Wan text_dim must be 4096, got {self.wan.model.text_dim}"
            )
        self.report["model_loading"]["wan_base"] = {
            "path": self.args.wan_checkpoint_dir,
            "original_text_len": self.original_text_len,
            "text_dim": int(self.wan.model.text_dim),
            "initial_device": str(next(self.wan.model.parameters()).device),
            "attention_backend": attention_backend,
        }
        print(
            "[WAN-ATTN] "
            f"effective={attention_backend['effective']} "
            f"fa2={attention_backend['flash_attention_2_available']} "
            f"fa3={attention_backend['flash_attention_3_available']} "
            f"contiguous={attention_backend['force_contiguous']}"
        )

    def _load_mq_encoder(self) -> None:
        import train_connector_for_wan as connector_module

        encoder_class = build_three_router_encoder_class(
            connector_module.MetaQueryEncoderForWan,
            self.bundle.router_config,
            enabled=True,
        )
        encoder = encoder_class(
            qwen3vl_model_id=self.args.qwen3vl_model_id,
            num_metaqueries=self.bundle.router_config.total_tokens,
            connector_num_hidden_layers=24,
            gradient_checkpointing=False,
            train_input_embeddings=False,
            connector_norm_init_scale=float(
                self.bundle.config.get("mq_connector_norm_init_scale", 1.0)
            ),
            dtype=torch.bfloat16,
            device=str(self.encoder_device),
        )
        embedding = encoder.mllm_model.mllm_backbone.get_input_embeddings()
        embedding_weight = embedding.weight
        if int(encoder.mllm_model.num_embeddings) != self.bundle.base_embedding_rows:
            raise RuntimeError(
                "Fresh Qwen base embedding row count does not match training: "
                f"model={encoder.mllm_model.num_embeddings}, "
                f"checkpoint={self.bundle.base_embedding_rows}"
            )
        if int(embedding_weight.shape[0]) != self.bundle.total_embedding_rows:
            raise RuntimeError(
                "Fresh Qwen total embedding row count does not match training: "
                f"model={embedding_weight.shape[0]}, "
                f"checkpoint={self.bundle.total_embedding_rows}"
            )
        checkpoint_special_rows = _load_tensor_rows(
            self.bundle.full_mq_path,
            self.bundle.full_embedding_key,
            self.bundle.base_embedding_rows,
        )
        expected_special_shape = (
            self.bundle.router_config.total_tokens + 2,
            self.bundle.router_config.hidden_size,
        )
        if tuple(checkpoint_special_rows.shape) != expected_special_shape:
            raise RuntimeError(
                "Frozen Qwen special-token shape mismatch: "
                f"expected={expected_special_shape}, "
                f"checkpoint={tuple(checkpoint_special_rows.shape)}"
            )
        initial_special_rows = (
            embedding_weight[self.bundle.base_embedding_rows :]
            .detach()
            .float()
            .cpu()
            .clone()
        )
        with torch.no_grad():
            embedding_weight[self.bundle.base_embedding_rows :].copy_(
                checkpoint_special_rows.to(
                    device=embedding_weight.device,
                    dtype=embedding_weight.dtype,
                )
            )
        loaded_special_rows = (
            embedding_weight[self.bundle.base_embedding_rows :].detach()
        )
        expected_special_rows = checkpoint_special_rows.to(
            device=loaded_special_rows.device,
            dtype=loaded_special_rows.dtype,
        )
        if not torch.equal(loaded_special_rows, expected_special_rows):
            raise RuntimeError(
                "Frozen Qwen BOI/EOI/MetaQuery embedding rows were not "
                "restored exactly"
            )
        special_delta_rms = _tensor_rms(
            loaded_special_rows.float().cpu() - initial_special_rows
        )
        initial_routes = {
            route: encoder.route_metaquery_embeddings[route].detach().cpu().clone()
            for route in ROUTE_NAMES
        }
        state, load_report = _load_subset_strict(
            encoder,
            self.bundle.mq_state_path,
            label="MQ/Connector",
        )
        route_report = {}
        for route in ROUTE_NAMES:
            key = ROUTE_STATE_KEYS[route]
            loaded = encoder.route_metaquery_embeddings[route].detach()
            expected = state[key].to(device=loaded.device, dtype=loaded.dtype)
            if not torch.equal(loaded, expected):
                raise RuntimeError(f"Loaded route tensor does not equal checkpoint: {key}")
            delta = loaded.float().cpu() - initial_routes[route].float()
            delta_rms = _tensor_rms(delta)
            if delta_rms <= 0.0:
                raise RuntimeError(
                    f"{route} route table equals its fresh initialization; "
                    "the trained MQ update was not restored"
                )
            route_report[route] = {
                "shape": tuple(loaded.shape),
                "dtype": str(loaded.dtype),
                "checkpoint_exact_match": True,
                "delta_from_fresh_init_rms": delta_rms,
                "loaded_rms": _tensor_rms(loaded),
            }
        encoder.eval().requires_grad_(False)
        self.mq_encoder = encoder
        load_report["routes"] = route_report
        load_report["frozen_qwen_special_token_rows"] = {
            "source": self.bundle.full_mq_path,
            "embedding_key": self.bundle.full_embedding_key,
            "base_rows": self.bundle.base_embedding_rows,
            "loaded_rows": expected_special_shape[0],
            "shape": expected_special_shape,
            "checkpoint_exact_match": True,
            "delta_from_fresh_init_rms": special_delta_rms,
        }
        load_report["qwen_frozen"] = not any(
            parameter.requires_grad
            for parameter in encoder.mllm_model.mllm_backbone.parameters()
        )
        self.report["model_loading"]["mq_encoder"] = load_report
        del (
            state,
            initial_routes,
            initial_special_rows,
            checkpoint_special_rows,
            expected_special_rows,
        )
        gc.collect()

    def _load_wan_updates(self) -> None:
        base_state = self.wan.model.state_dict()
        state_preview = _load_tensor_file(self.bundle.wan_state_path)
        small_keys = [
            key
            for key, tensor in state_preview.items()
            if tensor.numel() <= 100_000 and key in base_state
        ][:24]
        before = {
            key: base_state[key].detach().float().cpu().clone()
            for key in small_keys
        }
        state, load_report = _load_subset_strict(
            self.wan.model,
            self.bundle.wan_state_path,
            label="Wan condition branch",
            state=state_preview,
        )
        post_state = self.wan.model.state_dict()
        checkpoint_match = True
        changed_keys = []
        max_delta = 0.0
        for key in small_keys:
            current_raw = post_state[key].detach().cpu()
            expected_raw = state[key].to(dtype=current_raw.dtype)
            checkpoint_match = checkpoint_match and torch.equal(
                current_raw,
                expected_raw,
            )
            current = current_raw.float()
            delta = float((current - before[key]).abs().max().item())
            max_delta = max(max_delta, delta)
            if delta > 0.0:
                changed_keys.append(key)
        if not checkpoint_match:
            raise RuntimeError("Wan condition weights do not equal the checkpoint")
        if not changed_keys:
            raise RuntimeError(
                "Sampled Wan condition tensors equal the base model; "
                "the trained Wan update was not restored"
            )
        self.wan.model.eval().requires_grad_(False)
        load_report.update(
            {
                "checkpoint_exact_match_on_sampled_tensors": True,
                "sampled_tensor_count": len(small_keys),
                "changed_vs_base_tensor_count": len(changed_keys),
                "changed_vs_base_preview": changed_keys[:8],
                "max_abs_delta_vs_base": max_delta,
            }
        )
        self.report["model_loading"]["wan_condition_update"] = load_report
        del state, before, base_state, post_state
        gc.collect()

    @staticmethod
    def _snapshot_routes(encoder: torch.nn.Module) -> Dict[str, torch.Tensor]:
        output = encoder.last_router_output
        if output is None:
            raise RuntimeError("Three-router output was not recorded")
        return {
            "role": output.role.detach().clone(),
            "action": output.action.detach().clone(),
            "global": output.global_route.detach().clone(),
        }

    def _encode_mq(
        self,
        prompt: str,
        image: Image.Image | None,
    ) -> torch.Tensor:
        images = [[image]] if image is not None else None
        with torch.inference_mode():
            features = self.mq_encoder([prompt], images)
        expected = (
            1,
            self.bundle.router_config.total_tokens,
            int(self.bundle.config["wan_text_dim"]),
        )
        if tuple(features.shape) != expected:
            raise RuntimeError(
                f"MQ output shape mismatch: expected {expected}, got "
                f"{tuple(features.shape)}"
            )
        if not bool(torch.isfinite(features).all()) or _tensor_rms(features) <= 0:
            raise RuntimeError("MQ output is zero or contains NaN/Inf")
        return features

    def _verify_positive_route_audit(self) -> None:
        audit = self.mq_encoder.last_route_input_audit
        expected = {
            "role": {
                "caption_nonempty": [False],
                "image_input_supplied": True,
                "pixel_values_present": True,
                "image_grid_present": True,
            },
            "action": {
                "caption_nonempty": [True],
                "image_input_supplied": False,
                "pixel_values_present": False,
                "image_grid_present": False,
            },
            "global": {
                "caption_nonempty": [True],
                "image_input_supplied": True,
                "pixel_values_present": True,
                "image_grid_present": True,
            },
        }
        failures = []
        for route, checks in expected.items():
            for key, value in checks.items():
                if audit.get(route, {}).get(key) != value:
                    failures.append(
                        f"{route}.{key}={audit.get(route, {}).get(key)!r}, "
                        f"expected={value!r}"
                    )
            expected_tokens = getattr(
                self.bundle.router_config, f"{route}_tokens"
            )
            output_shape = audit.get(route, {}).get("qwen_output_shape")
            if output_shape != (1, expected_tokens, self.bundle.router_config.hidden_size):
                failures.append(
                    f"{route}.qwen_output_shape={output_shape!r}"
                )
        connector_audit = self.mq_encoder.last_joint_connector_audit
        if connector_audit.get("call_count") != 1:
            failures.append(
                f"connector.call_count={connector_audit.get('call_count')}"
            )
        if connector_audit.get("input_shape") != (
            1,
            self.bundle.router_config.total_tokens,
            self.bundle.router_config.hidden_size,
        ):
            failures.append(
                f"connector.input_shape={connector_audit.get('input_shape')}"
            )
        if failures:
            raise RuntimeError(
                "Positive-condition three-router/Qwen audit failed: "
                + "; ".join(failures)
            )
        self.report["runtime"]["positive_route_audit"] = {
            "status": "pass",
            "routes": audit,
            "joint_connector": connector_audit,
        }

    def _route_ablation_stats(
        self,
        positive: Mapping[str, torch.Tensor],
        ablated: Mapping[str, torch.Tensor],
        unchanged_routes: set[str],
        tag: str,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        eps = float(self.args.audit_epsilon)
        for route in ROUTE_NAMES:
            diff = positive[route].float() - ablated[route].float()
            diff_rms = _tensor_rms(diff)
            base_rms = _tensor_rms(positive[route])
            ratio = diff_rms / (base_rms + 1e-8)
            should_be_unchanged = route in unchanged_routes
            passed = ratio <= eps if should_be_unchanged else diff_rms > eps
            result[route] = {
                "diff_rms": diff_rms,
                "relative_diff": ratio,
                "expected": "unchanged" if should_be_unchanged else "changed",
                "passed": passed,
            }
            if not passed:
                raise RuntimeError(
                    f"{tag} isolation check failed for {route}: {result[route]}"
                )
        return result

    def _encode_t5_texts(self, texts: Sequence[str]) -> list[torch.Tensor]:
        if not texts:
            raise ValueError("At least one text is required for T5 RMS matching")
        if self.args.t5_cpu:
            t5_device = torch.device("cpu")
        else:
            t5_device = self.dit_device
            self.wan.text_encoder.model.to(t5_device)
        with torch.inference_mode():
            values = self.wan.text_encoder(list(texts), t5_device)
        if len(values) != len(texts):
            raise RuntimeError(
                f"T5 returned {len(values)} sequences, expected {len(texts)}"
            )
        result = [
            value.to(device=self.dit_device, dtype=torch.bfloat16)
            for value in values
        ]
        for index, value in enumerate(result):
            _tensor_stats(value, label=f"T5 context {index}")
        if not self.args.t5_cpu and self.args.offload_model:
            self.wan.text_encoder.model.cpu()
            torch.cuda.empty_cache()
        return result

    def _build_contexts(
        self,
        prompt: str,
        negative_prompt: str,
        ref_image: Image.Image,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        mq_positive = self._encode_mq(prompt, ref_image)
        self._verify_positive_route_audit()
        positive_routes = self._snapshot_routes(self.mq_encoder)
        no_image_features = None

        if self.args.runtime_audit == "full":
            no_image_features = self._encode_mq(prompt, None)
            no_image_routes = self._snapshot_routes(self.mq_encoder)
            image_ablation = self._route_ablation_stats(
                positive_routes,
                no_image_routes,
                unchanged_routes={"action"},
                tag="image ablation",
            )
            no_text_features = self._encode_mq("", ref_image)
            no_text_routes = self._snapshot_routes(self.mq_encoder)
            text_ablation = self._route_ablation_stats(
                positive_routes,
                no_text_routes,
                unchanged_routes={"role"},
                tag="text ablation",
            )
            self.report["runtime"]["route_isolation_ablation"] = {
                "status": "pass",
                "image_removed": image_ablation,
                "text_removed": text_ablation,
            }

        use_cfg = self.args.guide_scale > 1.0
        uncond_mode = str(self.args.cfg_uncond_mode)
        uncond_prompt: str | None = None
        mq_unconditional: torch.Tensor | None = None
        t5_texts = [prompt]
        if use_cfg and uncond_mode != "zero_mq":
            if uncond_mode == "empty_mq":
                uncond_prompt = ""
            elif uncond_mode == "negative_mq":
                uncond_prompt = (
                    negative_prompt
                    if negative_prompt.strip()
                    else self.wan.sample_neg_prompt
                )
            else:
                raise ValueError(f"Unknown CFG unconditional mode: {uncond_mode}")
            mq_unconditional = self._encode_mq(uncond_prompt, None)
            t5_texts.append(uncond_prompt)

        t5_contexts = self._encode_t5_texts(t5_texts)
        t5_positive = t5_contexts[0]
        clip_min = float(
            self.bundle.config.get("mq_norm_match_clip_min", 0.03)
        )
        clip_max = float(
            self.bundle.config.get("mq_norm_match_clip_max", 4.0)
        )
        connector_image_ablation: Dict[str, Any] | None = None
        if no_image_features is not None:
            connector_image_ablation = _tensor_difference_stats(
                mq_positive[0],
                no_image_features[0],
                label="image ablation at Connector output before RMS matching",
            )
        mq_positive, positive_norm = match_mq_rms(
            mq_positive,
            t5_positive,
            clip_min,
            clip_max,
        )
        unconditional_norm: Dict[str, Any] | None = None
        if mq_unconditional is not None:
            mq_unconditional, unconditional_norm = match_mq_rms(
                mq_unconditional,
                t5_contexts[1],
                clip_min,
                clip_max,
            )
        no_image_norm: Dict[str, Any] | None = None
        if no_image_features is not None:
            no_image_features, no_image_norm = match_mq_rms(
                no_image_features,
                t5_positive,
                clip_min,
                clip_max,
            )
            no_image_features = no_image_features[0].to(
                device=self.dit_device,
                dtype=torch.bfloat16,
            )
        positive_context = mq_positive[0].to(
            device=self.dit_device,
            dtype=torch.bfloat16,
        )
        expected_shape = (
            self.bundle.router_config.total_tokens,
            int(self.bundle.config["wan_text_dim"]),
        )
        if tuple(positive_context.shape) != expected_shape:
            raise RuntimeError(
                f"Positive DiT context must be {expected_shape}, got "
                f"{tuple(positive_context.shape)}"
            )
        unconditional_context: torch.Tensor | None = None
        if use_cfg:
            if uncond_mode == "zero_mq":
                unconditional_context = torch.zeros_like(positive_context)
                unconditional_norm = {
                    "mode": "zero_mq",
                    "mq_rms_after": 0.0,
                }
            else:
                assert mq_unconditional is not None
                unconditional_context = mq_unconditional[0].to(
                    device=self.dit_device,
                    dtype=torch.bfloat16,
                )
            if tuple(unconditional_context.shape) != expected_shape:
                raise RuntimeError(
                    "Unconditional DiT context shape mismatch: expected "
                    f"{expected_shape}, got {tuple(unconditional_context.shape)}"
                )
            _tensor_stats(
                unconditional_context,
                label="CFG unconditional context",
            )
        if no_image_features is not None:
            if tuple(no_image_features.shape) != expected_shape:
                raise RuntimeError(
                    f"Image-ablated DiT context must be {expected_shape}, got "
                    f"{tuple(no_image_features.shape)}"
                )
            final_image_ablation = _tensor_difference_stats(
                positive_context,
                no_image_features,
                label="image ablation in final Wan DiT context",
            )
            image_context_diff = float(final_image_ablation["diff_rms"])
            if (
                connector_image_ablation is not None
                and float(connector_image_ablation["diff_rms"])
                <= self.args.audit_epsilon
            ):
                collapse_stage = "shared_connector"
            elif image_context_diff <= self.args.audit_epsilon:
                collapse_stage = "rms_match_or_bfloat16_quantization"
            else:
                collapse_stage = "none"
            self.report["runtime"]["dit_image_context_ablation"] = {
                "status": (
                    "pass"
                    if image_context_diff > self.args.audit_epsilon
                    else "fail"
                ),
                "diff_rms": image_context_diff,
                "relative_diff": final_image_ablation["relative_diff"],
                "dtype": str(positive_context.dtype),
                "collapse_stage": collapse_stage,
                "connector_output_before_rms": connector_image_ablation,
                "rms_match": {
                    "positive_scale": positive_norm["applied_scale"],
                    "image_ablated_scale": (
                        no_image_norm["applied_scale"]
                        if no_image_norm is not None
                        else None
                    ),
                },
                "final_dit_context": final_image_ablation,
            }
            if image_context_diff <= self.args.audit_epsilon:
                raise RuntimeError(
                    "Image conditioning collapsed before Wan DiT: "
                    f"stage={collapse_stage}, "
                    "the role/global Qwen routes changed but the final BF16 "
                    f"context diff_rms={image_context_diff:.8g}. See "
                    "runtime.dit_image_context_ablation in the verification report"
                )
        self.report["runtime"]["mq_t5_rms_match"] = {
            "positive": positive_norm,
            "unconditional": unconditional_norm,
            "clip": [clip_min, clip_max],
            "t5_tokens_sent_to_dit": 0,
            "mq_tokens_sent_to_dit": expected_shape[0],
        }
        self.report["runtime"]["cfg"] = {
            "enabled": use_cfg,
            "guide_scale": float(self.args.guide_scale),
            "unconditional_mode": uncond_mode if use_cfg else "not_computed",
            "unconditional_prompt": uncond_prompt,
            "negative_prompt_ignored": bool(
                negative_prompt.strip() and (not use_cfg or uncond_mode == "empty_mq")
            ),
            "training_aligned_default": (
                not use_cfg or uncond_mode == "empty_mq"
            ),
        }
        return (
            [positive_context],
            [unconditional_context] if unconditional_context is not None else None,
            no_image_features,
        )

    def _preprocess_reference(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, torch.Tensor]:
        import torchvision.transforms.functional as transform

        image = image.convert("RGB")
        width, height = image.size
        if self.args.size:
            output_width, output_height = map(int, self.args.size)
            scale = max(output_width / width, output_height / height)
            resized = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.LANCZOS,
            )
            left = (resized.width - output_width) // 2
            top = (resized.height - output_height) // 2
            image = resized.crop(
                (left, top, left + output_width, top + output_height)
            )
        else:
            area = width * height
            if area > self.args.max_area:
                scale = math.sqrt(float(self.args.max_area) / float(area))
                width = max(32, int(width * scale))
                height = max(32, int(height * scale))
            output_width = max(32, (width // 32) * 32)
            output_height = max(32, (height // 32) * 32)
            image = image.resize(
                (output_width, output_height),
                Image.Resampling.LANCZOS,
            )
        if image.width % 32 or image.height % 32:
            raise ValueError(
                f"Reference size must be 32-aligned, got {image.size}"
            )
        tensor = (
            transform.to_tensor(image)
            .sub_(0.5)
            .div_(0.5)
            .to(self.dit_device)
            .unsqueeze(1)
        )
        return image, tensor

    def _build_scheduler(self):
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        if self.args.sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.wan.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(
                self.args.sampling_steps,
                device=self.dit_device,
                shift=self.args.shift,
            )
            return scheduler, scheduler.timesteps
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=self.wan.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sigmas = get_sampling_sigmas(
            self.args.sampling_steps,
            self.args.shift,
        )
        timesteps, _ = retrieve_timesteps(
            scheduler,
            device=self.dit_device,
            sigmas=sigmas,
        )
        return scheduler, timesteps

    def _verify_wan_context_influence(
        self,
        pred_conditioned: torch.Tensor,
        latent_input: list[torch.Tensor],
        timestep: torch.Tensor,
        seq_len: int,
        context: list[torch.Tensor],
        no_image_context: torch.Tensor | None,
    ) -> torch.Tensor:
        retries = int(getattr(self.args, "audit_forward_retries", 1))
        attempts: list[Dict[str, Any]] = []
        verified_conditioned = pred_conditioned
        for attempt in range(retries + 1):
            pred_zero = self.wan.model(
                latent_input,
                t=timestep,
                context=[torch.zeros_like(context[0])],
                seq_len=seq_len,
            )[0]
            diff_rms = _tensor_rms(
                verified_conditioned.float() - pred_zero.float()
            )
            ratio = diff_rms / (_tensor_rms(verified_conditioned) + 1e-8)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "diff_rms": diff_rms,
                    "relative_diff": ratio,
                }
            )
            if diff_rms > self.args.audit_epsilon:
                break
            if attempt < retries:
                device = verified_conditioned.device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                # The original conditional result is not used after an exact
                # zero audit.  Recompute both sides of the comparison and, if
                # they pass, return this verified conditional prediction to
                # the sampler.
                verified_conditioned = self.wan.model(
                    latent_input,
                    t=timestep,
                    context=context,
                    seq_len=seq_len,
                )[0]
        result: Dict[str, Any] = {
            "mq_vs_zero_diff_rms": diff_rms,
            "mq_vs_zero_relative_diff": ratio,
            "attempt_count": len(attempts),
            "recovered_after_retry": (
                len(attempts) > 1 and diff_rms > self.args.audit_epsilon
            ),
            "attempts": attempts,
        }
        if diff_rms <= self.args.audit_epsilon:
            result["status"] = "fail"
            self.report["runtime"]["wan_context_influence"] = result
            raise RuntimeError(
                "Wan prediction is unchanged when all MQ/T5 context is "
                f"zeroed after {len(attempts)} attempt(s)"
            )
        if no_image_context is not None:
            pred_no_image = self.wan.model(
                latent_input,
                t=timestep,
                context=[no_image_context],
                seq_len=seq_len,
            )[0]
            image_diff = _tensor_rms(
                verified_conditioned.float() - pred_no_image.float()
            )
            result["image_condition_diff_rms"] = image_diff
            result["image_condition_relative_diff"] = image_diff / (
                _tensor_rms(verified_conditioned) + 1e-8
            )
            if image_diff <= self.args.audit_epsilon:
                result["image_condition_status"] = "fail"
                result["status"] = "fail"
                self.report["runtime"]["wan_context_influence"] = result
                raise RuntimeError(
                    "Wan prediction is unchanged after removing image "
                    "conditioning from the role/global MQ routes"
                )
            else:
                result["image_condition_status"] = "pass"
        result.setdefault("status", "pass")
        self.report["runtime"]["wan_context_influence"] = result
        return verified_conditioned

    def _encode_reference_latent(
        self,
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the Wan reference exactly as the strong-binding trainer does."""
        if reference_tensor.ndim != 4 or tuple(reference_tensor.shape[:2]) != (3, 1):
            raise ValueError(
                "Wan reference tensor must be [3,1,H,W], got "
                f"{tuple(reference_tensor.shape)}"
            )
        # StrongFirstFrameTrainingMixin ultimately delegates to the inherited
        # _encode_ref_image_to_latent(), whose VAE input is BF16.  Keeping the
        # same input dtype here avoids a subtle train/inference preprocessing
        # mismatch before the clean latent is locked into slot zero.
        vae_input = reference_tensor.to(
            device=self.dit_device,
            dtype=torch.bfloat16,
        )
        with torch.inference_mode():
            reference_latent = self.wan.vae.encode([vae_input])[0].float()
        _tensor_stats(reference_latent, label="reference latent")
        return reference_latent

    def _dit_context_token_count(self) -> int:
        """Return the context length used by this checkpoint's Wan forward."""
        return int(self.bundle.router_config.total_tokens)

    def _validate_dit_context_token_count(
        self,
        actual_tokens: int,
        configured_tokens: int,
    ) -> None:
        """Validate actual context length against Wan's padding capacity."""
        if actual_tokens != configured_tokens:
            raise RuntimeError(
                "Wan DiT context length does not match the inference contract: "
                f"context={actual_tokens}, configured={configured_tokens}"
            )

    def _extra_reference_prefix_slots(self) -> int:
        """Return reference slots prepended in addition to output latents."""
        return 0

    def _assess_output_quality(
        self,
        video: torch.Tensor,
        reference_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        return assess_decoded_video_quality(
            video,
            reference_tensor,
            max_high_frequency_ratio=float(
                self.args.max_noise_high_frequency_ratio
            ),
            min_reference_correlation=float(
                self.args.min_reference_frame_correlation
            ),
            max_reference_mae=float(self.args.max_reference_frame_mae),
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        ref_image: Image.Image,
    ) -> torch.Tensor:
        if not prompt.strip():
            raise ValueError("A non-empty --prompt is required")
        if self.args.frame_num <= 0 or (self.args.frame_num - 1) % 4 != 0:
            raise ValueError("--frame_num must be positive and have the form 4n+1")
        processed_ref, ref_tensor = self._preprocess_reference(ref_image)
        context, null_context, no_image_context = self._build_contexts(
            prompt,
            negative_prompt,
            processed_ref,
        )

        preserve_first_frame = self.args.first_frame_mode == "preserved"
        ref_latent: torch.Tensor | None = None
        if preserve_first_frame:
            ref_latent = self._encode_reference_latent(ref_tensor)
        target_latent_time = (
            (self.args.frame_num - 1) // self.wan_config.vae_stride[0] + 1
        )
        extra_prefix_slots = (
            self._extra_reference_prefix_slots() if preserve_first_frame else 0
        )
        if extra_prefix_slots not in (0, 1):
            raise RuntimeError(
                "This inference path supports zero or one extra reference "
                f"prefix slot, got {extra_prefix_slots}"
            )
        latent_time = target_latent_time + extra_prefix_slots
        latent_height = processed_ref.height // self.wan_config.vae_stride[1]
        latent_width = processed_ref.width // self.wan_config.vae_stride[2]
        target_shape = (
            self.wan.vae.model.z_dim,
            latent_time,
            latent_height,
            latent_width,
        )
        if ref_latent is not None and tuple(ref_latent.shape) != (
            target_shape[0],
            1,
            target_shape[2],
            target_shape[3],
        ):
            raise RuntimeError(
                "Reference VAE latent must be one slot compatible with the "
                f"generated latent, got {tuple(ref_latent.shape)} for {target_shape}"
            )
        patch_size = self.wan_config.patch_size
        tokens_per_latent_frame = math.ceil(
            latent_height
            * latent_width
            / (patch_size[1] * patch_size[2])
        )
        seq_len = latent_time * tokens_per_latent_frame
        generator = torch.Generator(device=self.dit_device)
        generator.manual_seed(
            self.args.seed
            if self.args.seed >= 0
            else random.randint(0, sys.maxsize)
        )
        latent = torch.randn(
            *target_shape,
            dtype=torch.float32,
            device=self.dit_device,
            generator=generator,
        )
        if ref_latent is not None:
            enforce_clean_reference_prefix_(latent, ref_latent)
        scheduler, timesteps = self._build_scheduler()

        trace_steps = {
            index
            for index in (
                0,
                1,
                5,
                10,
                len(timesteps) // 2,
                len(timesteps) - 1,
            )
            if 0 <= index < len(timesteps)
        }
        sampling_trace: list[Dict[str, Any]] = []
        previous_latent_rms = _tensor_rms(latent)
        original_text_len = int(self.wan.model.text_len)
        dit_context_tokens = self._dit_context_token_count()
        if dit_context_tokens <= 0:
            raise RuntimeError(
                f"Wan DiT context length must be positive, got {dit_context_tokens}"
            )
        if not context:
            raise RuntimeError("Wan DiT received no conditional context")
        self._validate_dit_context_token_count(
            int(context[0].shape[0]),
            dit_context_tokens,
        )
        self.wan.model.text_len = dit_context_tokens

        @contextmanager
        def no_sync_fallback():
            yield

        no_sync = getattr(self.wan.model, "no_sync", no_sync_fallback)
        try:
            self.wan.model.to(self.dit_device)
            torch.cuda.empty_cache()
            with (
                torch.amp.autocast("cuda", dtype=self.wan.param_dtype),
                torch.inference_mode(),
                no_sync(),
            ):
                for step_index, timestep_value in enumerate(tqdm(timesteps)):
                    timestep_float = float(timestep_value.item())
                    if not math.isfinite(timestep_float):
                        raise FloatingPointError(
                            f"Sampling timestep {step_index} is non-finite"
                        )
                    if ref_latent is not None:
                        enforce_clean_reference_prefix_(latent, ref_latent)
                    latent_before_stats = _tensor_stats(
                        latent,
                        label=f"latent before step {step_index}",
                    )
                    latent_input = [latent]
                    timestep = build_model_timestep_row(
                        timestep_value,
                        seq_len=seq_len,
                        tokens_per_latent_frame=tokens_per_latent_frame,
                        preserve_first_frame=preserve_first_frame,
                        device=self.dit_device,
                    )
                    pred_conditioned = self.wan.model(
                        latent_input,
                        t=timestep,
                        context=context,
                        seq_len=seq_len,
                    )[0]
                    pred_conditioned_stats = _tensor_stats(
                        pred_conditioned,
                        label=f"conditional prediction step {step_index}",
                    )
                    pred_unconditioned: torch.Tensor | None = None
                    pred_unconditioned_stats: Dict[str, Any] | None = None
                    if null_context is not None:
                        pred_unconditioned = self.wan.model(
                            latent_input,
                            t=timestep,
                            context=null_context,
                            seq_len=seq_len,
                        )[0]
                        pred_unconditioned_stats = _tensor_stats(
                            pred_unconditioned,
                            label=f"unconditional prediction step {step_index}",
                        )
                    if step_index == 0:
                        pred_conditioned = self._verify_wan_context_influence(
                            pred_conditioned,
                            latent_input,
                            timestep,
                            seq_len,
                            context,
                            no_image_context,
                        )
                        # A transient exact-zero audit causes a synchronized
                        # recomputation.  Keep trace/guidance statistics tied
                        # to the verified prediction that sampling will use.
                        pred_conditioned_stats = _tensor_stats(
                            pred_conditioned,
                            label=(
                                "verified conditional prediction step "
                                f"{step_index}"
                            ),
                        )
                    if pred_unconditioned is None:
                        prediction = pred_conditioned
                    else:
                        prediction = (
                            pred_unconditioned
                            + self.args.guide_scale
                            * (pred_conditioned - pred_unconditioned)
                        )
                    guided_stats = _tensor_stats(
                        prediction,
                        label=f"guided prediction step {step_index}",
                    )
                    branch_rms = max(
                        float(pred_conditioned_stats["rms"]),
                        float(
                            pred_unconditioned_stats["rms"]
                            if pred_unconditioned_stats is not None
                            else pred_conditioned_stats["rms"]
                        ),
                        1e-8,
                    )
                    cfg_amplification = float(guided_stats["rms"]) / branch_rms
                    if cfg_amplification > self.args.audit_growth_limit:
                        raise RuntimeError(
                            "CFG prediction amplification exceeded audit limit "
                            f"at step {step_index}: {cfg_amplification:.4f} > "
                            f"{self.args.audit_growth_limit:.4f}"
                        )
                    t_norm = max(
                        0.0,
                        min(
                            1.0,
                            timestep_float
                            / float(self.wan.num_train_timesteps),
                        ),
                    )
                    predicted_x0 = latent - t_norm * prediction.float()
                    predicted_x0_stats = _tensor_stats(
                        predicted_x0,
                        label=f"predicted x0 step {step_index}",
                    )
                    latent = scheduler.step(
                        prediction.unsqueeze(0),
                        timestep_value,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0].squeeze(0)
                    if ref_latent is not None:
                        enforce_clean_reference_prefix_(latent, ref_latent)
                    latent_after_stats = _tensor_stats(
                        latent,
                        label=f"latent after step {step_index}",
                    )
                    latent_growth = float(latent_after_stats["rms"]) / max(
                        previous_latent_rms,
                        1e-8,
                    )
                    if latent_growth > self.args.audit_growth_limit:
                        raise RuntimeError(
                            "Latent RMS growth exceeded audit limit at step "
                            f"{step_index}: {latent_growth:.4f} > "
                            f"{self.args.audit_growth_limit:.4f}"
                        )
                    previous_latent_rms = float(latent_after_stats["rms"])
                    if step_index in trace_steps:
                        prefix_error = 0.0
                        if ref_latent is not None:
                            prefix_error = float(
                                (
                                    latent[:, :1].float()
                                    - ref_latent[:, :1].float()
                                )
                                .abs()
                                .max()
                                .item()
                            )
                        sampling_trace.append(
                            {
                                "step": step_index,
                                "timestep": timestep_float,
                                "timestep_zero_prefix_tokens": (
                                    tokens_per_latent_frame
                                    if preserve_first_frame
                                    else 0
                                ),
                                "latent_before": latent_before_stats,
                                "pred_conditioned": pred_conditioned_stats,
                                "pred_unconditioned": pred_unconditioned_stats,
                                "guided_prediction": guided_stats,
                                "cfg_amplification": cfg_amplification,
                                "predicted_x0": predicted_x0_stats,
                                "latent_after": latent_after_stats,
                                "latent_growth": latent_growth,
                                "reference_prefix_max_abs_error": prefix_error,
                            }
                        )
                if self.args.offload_model:
                    self.wan.model.cpu()
                    torch.cuda.synchronize(self.dit_device)
                    torch.cuda.empty_cache()
                output_latent = (
                    latent[:, extra_prefix_slots:]
                    if extra_prefix_slots > 0
                    else latent
                )
                if int(output_latent.shape[1]) != target_latent_time:
                    raise RuntimeError(
                        "Output latent length does not match the requested video: "
                        f"expected={target_latent_time}, "
                        f"actual={int(output_latent.shape[1])}"
                    )
                video = self.wan.vae.decode([output_latent])[0]
        finally:
            self.wan.model.text_len = original_text_len
            if self.args.offload_model:
                self.wan.model.cpu()
                torch.cuda.empty_cache()

        video_stats = _tensor_stats(video, label="decoded video")
        expected_video_shape = (
            3,
            int(self.args.frame_num),
            int(processed_ref.height),
            int(processed_ref.width),
        )
        if tuple(video.shape) != expected_video_shape:
            raise RuntimeError(
                f"Decoded video shape mismatch: expected {expected_video_shape}, "
                f"got {tuple(video.shape)}"
            )
        quality_audit = self._assess_output_quality(video, ref_tensor)
        self.report["runtime"]["decoded_video_quality"] = quality_audit
        self.report["runtime"]["generation"] = {
            "status": "decoded_pending_container_validation",
            "output_size": [processed_ref.width, processed_ref.height],
            "frame_num": self.args.frame_num,
            "latent_shape": target_shape,
            "model_input_latent_shape": target_shape,
            "decoded_output_latent_shape": (
                target_shape[0],
                target_latent_time,
                target_shape[2],
                target_shape[3],
            ),
            "target_latent_slots": target_latent_time,
            "extra_reference_prefix_slots": extra_prefix_slots,
            "reference_prefix_excluded_from_decoded_output": bool(
                extra_prefix_slots
            ),
            "seq_len": seq_len,
            "dit_text_len_during_sampling": dit_context_tokens,
            "dit_text_len_restored": int(self.wan.model.text_len),
            "checkpoint_training_first_frame_mode": self.bundle.router[
                "wan_first_frame_conditioning"
            ].get("mode"),
            "inference_first_frame_mode": self.args.first_frame_mode,
            "first_frame_timestep_zero": preserve_first_frame,
            "reference_relocked_after_each_step": preserve_first_frame,
            "preserved_reference_latent_slots": int(preserve_first_frame),
            "sampling_trace": sampling_trace,
            "decoded_video": video_stats,
        }
        if quality_audit["status"] != "pass":
            self.report["runtime"]["generation"]["status"] = "fail"
            raise RuntimeError(
                "Decoded video failed reference/noise quality audit: "
                + "; ".join(quality_audit["failures"])
            )
        return video


def prepare_video_for_wan_writer(
    video: torch.Tensor,
    *,
    expected_frame_num: int,
    expected_size: tuple[int, int],
) -> torch.Tensor:
    """Validate [C,T,H,W] and add the batch dimension required by Wan."""
    if not torch.is_tensor(video) or video.ndim != 4:
        shape = tuple(video.shape) if torch.is_tensor(video) else type(video).__name__
        raise ValueError(f"Decoded video must be [C,T,H,W], got {shape}")
    expected_width, expected_height = expected_size
    expected_shape = (3, expected_frame_num, expected_height, expected_width)
    if tuple(video.shape) != expected_shape:
        raise ValueError(
            f"Decoded video shape must be {expected_shape}, got {tuple(video.shape)}"
        )
    _tensor_stats(video, label="video passed to writer")
    return video.unsqueeze(0)


def _probe_video_metadata(path: Path) -> Dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,nb_read_frames"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is required for strict video output validation"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}: {result.stderr.strip()}"
        )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(
            f"Expected exactly one video stream in {path}, got {len(streams)}"
        )
    stream = streams[0]
    frame_value = stream.get("nb_read_frames")
    if frame_value in (None, "N/A"):
        frame_value = stream.get("nb_frames")
    if frame_value in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report a frame count for {path}")
    measured_fps = math.nan
    rate_value = None
    for candidate in (
        stream.get("avg_frame_rate"),
        stream.get("r_frame_rate"),
    ):
        try:
            candidate_fps = float(Fraction(str(candidate)))
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(candidate_fps) and candidate_fps > 0.0:
            measured_fps = candidate_fps
            rate_value = candidate
            break
    if not math.isfinite(measured_fps):
        raise RuntimeError(
            f"Invalid ffprobe frame rate {rate_value!r} for {path}"
        )
    metadata = {
        "codec_name": str(stream.get("codec_name", "")),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_num": int(frame_value),
        "fps": measured_fps,
    }
    if any(
        not math.isfinite(float(metadata[name]))
        for name in ("width", "height", "frame_num", "fps")
    ):
        raise RuntimeError(f"Non-finite video metadata for {path}: {metadata}")
    return metadata


def validate_video_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_frame_num: int,
    expected_size: tuple[int, int],
    expected_fps: int,
) -> None:
    expected_width, expected_height = expected_size
    failures = []
    if int(metadata.get("width", -1)) != expected_width:
        failures.append(
            f"width={metadata.get('width')} expected={expected_width}"
        )
    if int(metadata.get("height", -1)) != expected_height:
        failures.append(
            f"height={metadata.get('height')} expected={expected_height}"
        )
    if int(metadata.get("frame_num", -1)) != expected_frame_num:
        failures.append(
            f"frame_num={metadata.get('frame_num')} expected={expected_frame_num}"
        )
    measured_fps = float(metadata.get("fps", math.nan))
    if not math.isfinite(measured_fps) or not math.isclose(
        measured_fps,
        float(expected_fps),
        rel_tol=0.0,
        abs_tol=1e-3,
    ):
        failures.append(f"fps={measured_fps} expected={expected_fps}")
    if failures:
        raise RuntimeError("Video metadata validation failed: " + "; ".join(failures))


def _save_video(
    video: torch.Tensor,
    output_path: Path,
    fps: int,
    *,
    expected_frame_num: int,
    expected_size: tuple[int, int],
) -> Dict[str, Any]:
    from wan.utils.utils import save_video

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    writer_input = prepare_video_for_wan_writer(
        video,
        expected_frame_num=expected_frame_num,
        expected_size=expected_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix or ".mp4",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        save_video(
            writer_input,
            save_file=str(temporary_path),
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if (
            not temporary_path.is_file()
            or temporary_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Video writer did not create a valid file: {temporary_path}"
            )
        metadata = _probe_video_metadata(temporary_path)
        validate_video_metadata(
            metadata,
            expected_frame_num=expected_frame_num,
            expected_size=expected_size,
            expected_fps=fps,
        )
        os.replace(temporary_path, output_path)
        return {
            **metadata,
            "bytes": output_path.stat().st_size,
            "status": "pass",
        }
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolved_args = vars(args).copy()
    if args.parse_only:
        print(
            json.dumps(
                {"status": "ok", "args": resolved_args},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    bundle = validate_checkpoint_bundle(args.checkpoint_dir)
    if args.check_only:
        print(
            json.dumps(
                _jsonable(bundle.report),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    ref_path = Path(args.ref_image).expanduser().resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not args.prompt.strip():
        raise ValueError("--prompt must be non-empty for full inference")
    report_path = (
        Path(args.verify_report_path).expanduser().resolve()
        if args.verify_report_path
        else Path(f"{args.output_path}.verify.json").expanduser().resolve()
    )
    output_path = Path(args.output_path).expanduser().resolve()
    if output_path == report_path:
        raise ValueError("--output_path and --verify_report_path must differ")
    pipeline: ThreeRouterWanInference | None = None
    try:
        pipeline = ThreeRouterWanInference(args, bundle)
        pipeline.report["input"] = {
            "reference_image": ref_path,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "frame_num": int(args.frame_num),
            "max_area": int(args.max_area),
            "seed": int(args.seed),
        }
        with Image.open(ref_path) as handle:
            reference = handle.convert("RGB")
        video = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            ref_image=reference,
        )
        expected_size = (
            int(pipeline.report["runtime"]["generation"]["output_size"][0]),
            int(pipeline.report["runtime"]["generation"]["output_size"][1]),
        )
        output_metadata = _save_video(
            video,
            output_path,
            args.fps,
            expected_frame_num=args.frame_num,
            expected_size=expected_size,
        )
        pipeline.report["runtime"]["generation"]["status"] = "pass"
        pipeline.report["output"] = {
            "status": "pass",
            "video": output_path,
            "verify_report": report_path,
            "metadata": output_metadata,
        }
        pipeline.report["status"] = "pass"
        print(f"[DONE] video={output_path}")
        print(f"[DONE] verify_report={report_path}")
    except Exception as exc:
        if pipeline is not None:
            pipeline.report["status"] = "fail"
            pipeline.report["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        raise
    finally:
        if pipeline is not None:
            _write_json(report_path, pipeline.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
