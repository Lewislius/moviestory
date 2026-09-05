#!/usr/bin/env python3
"""Training-aligned inference for the OpenVid4000 strong-binding checkpoint.

This entrypoint deliberately has no legacy soft-anchor or no-anchor execution
path.  It validates the complete training contract before loading Qwen or Wan,
then delegates model loading, three-route encoding, sampling audits, and video
writing to the shared inference implementation.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

import infer_3router_planner_wan as core


CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    CODE_ROOT
    / "checkpoint"
    / "three_router_mq256_conn24_strongbind_openvid4000_steps500"
)
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "inference_outputs"
    / "openvid4000_3router_strongbind"
    / "strongbind_result.mp4"
)


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def parse_args(argv: Sequence[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(raw_argv, "--checkpoint_dir"):
        raw_argv = ["--checkpoint_dir", str(DEFAULT_CHECKPOINT), *raw_argv]
    if not _has_option(raw_argv, "--output_path"):
        raw_argv = ["--output_path", str(DEFAULT_OUTPUT), *raw_argv]

    args = core.parse_args(raw_argv)
    if args.first_frame_mode != "preserved":
        raise ValueError(
            "Strong-binding inference requires --first_frame_mode preserved; "
            "the no-anchor path does not match this checkpoint"
        )
    if args.cfg_uncond_mode != "empty_mq":
        raise ValueError(
            "Strong-binding inference only accepts --cfg_uncond_mode empty_mq, "
            "matching joint-null training with caption='' and mq_ref_image=None"
        )
    return args


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_strong_binding_contract(bundle: Any) -> dict[str, Any]:
    """Reject checkpoints that differ from the flow-consistent train contract."""
    router = _as_mapping(bundle.router)
    training_args = _as_mapping(bundle.training_args)
    config = _as_mapping(bundle.config)
    trainer_state = _as_mapping(bundle.trainer_state)
    first_frame = _as_mapping(router.get("wan_first_frame_conditioning"))
    dropout = _as_mapping(router.get("conditioning_dropout"))
    joint_null = _as_mapping(dropout.get("joint_null_contract"))
    loss = _as_mapping(router.get("loss_contract"))

    joint_null_probability = float(training_args.get("joint_null_prob", -1.0))
    metadata_joint_null_probability = float(dropout.get("joint_null", -1.0))
    exact_checks = {
        "flow_consistent_format": (
            router.get("format")
            == "moviestory_three_router_planner_v6_flow_consistent"
        ),
        "strong_bind_enabled": (
            training_args.get("wan_first_frame_strong_bind") is True
            and training_args.get("moviestory_wan_first_frame_strong_bind") is True
        ),
        "wan_animate_slot": (
            training_args.get("enable_ti2v_first_frame_condition") is True
            and training_args.get("train_video_conditioning_mode")
            == "wan_animate_slot"
        ),
        "one_reference_slot_only": (
            int(training_args.get("train_animate_ref_frames", -1)) == 1
            and int(training_args.get("train_animate_temporal_frames", -1)) == 0
            and int(training_args.get("train_animate_conditional_frames", -1)) == 0
        ),
        "zero_timestep_and_prefix_loss_drop": (
            training_args.get("train_animate_preserve_timestep_zero") is True
            and training_args.get("train_animate_drop_prefix_loss") is True
        ),
        "no_soft_anchor": (
            training_args.get("train_ref_anchor_mode") == "none"
            and float(training_args.get("train_ref_anchor_alpha0", math.nan)) == 0.0
            and float(
                training_args.get("train_ref_anchor_warmup_ratio", math.nan)
            )
            == 0.0
        ),
        "clean_reference_metadata": (
            first_frame.get("enabled") is True
            and first_frame.get("mode") == "clean_preserved_latent_slot"
            and first_frame.get("source") == "ref_image"
            and int(first_frame.get("latent_slots", -1)) == 1
            and first_frame.get("timestep_zero") is True
            and first_frame.get("shares_video_random_timestep") is False
            and first_frame.get("remains_noisy_denoising_input") is False
        ),
        "flow_consistent_target_layout": (
            first_frame.get("original_first_target_slot_removed") is True
            and first_frame.get("original_first_target_slot_retained") is False
            and first_frame.get("excluded_from_denoising_loss") is True
        ),
        "metadata_has_no_soft_anchor": (
            first_frame.get("soft_anchor_mode") == "none"
            and float(first_frame.get("soft_anchor_alpha0", math.nan)) == 0.0
            and float(first_frame.get("soft_anchor_warmup_ratio", math.nan))
            == 0.0
        ),
        "joint_null_preserves_wan_reference": (
            joint_null.get("caption") == ""
            and "mq_ref_image" in joint_null
            and joint_null.get("mq_ref_image") is None
            and joint_null.get("wan_ref_image_preserved") is True
        ),
        "trained_cfg_null_branch": (
            0.0 < joint_null_probability <= 1.0
            and metadata_joint_null_probability == joint_null_probability
        ),
        "video_velocity_loss_only": (
            loss.get("name") == "video_ground_truth_velocity_mse_only"
            and loss.get("optimized_terms") == ["video_ground_truth_velocity_mse"]
            and loss.get("t5_alignment_loss") is False
            and loss.get("mq_image_preserve_loss") is False
            and loss.get("wan_function_distillation_loss") is False
            and loss.get("preserved_reference_prefix_in_loss") is False
        ),
        "frozen_qwen_input_embeddings": (
            training_args.get("train_mq_input_embeddings") is False
            and router.get("qwen_input_embedding_table_trainable") is False
        ),
        "checkpoint_step_consistent": (
            int(config.get("checkpoint_step", -1))
            == int(trainer_state.get("global_step", -2))
            and int(config.get("checkpoint_step", -1)) > 0
        ),
        "training_shape_contract_consistent": (
            int(config.get("frame_num", -1))
            == int(training_args.get("frame_num", -2))
            and int(config.get("max_area", -1))
            == int(training_args.get("max_area", -2))
        ),
    }
    failures = [name for name, passed in exact_checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Checkpoint is not compatible with strict strong-binding inference: "
            f"{failures}"
        )

    report = {
        "status": "pass",
        "checks": exact_checks,
        "training_contract": {
            "first_frame_mode": first_frame.get("mode"),
            "reference_source": first_frame.get("source"),
            "reference_latent_slots": int(first_frame["latent_slots"]),
            "reference_timestep": 0,
            "original_first_target_slot_removed": True,
            "reference_prefix_excluded_from_loss": True,
            "soft_anchor": "none",
            "joint_null_probability": joint_null_probability,
            "joint_null_wan_reference_preserved": True,
        },
        "inference_contract": {
            "initial_reference_prefix": "clean_vae_latent",
            "reference_prefix_relocked_before_model_forward": True,
            "reference_prefix_relocked_after_every_solver_step": True,
            "reference_prefix_token_timestep": 0,
            "cfg_unconditional_branch": "MQ('', None)",
            "cfg_wan_reference_preserved": True,
        },
    }
    bundle.report["strong_binding_validation"] = report
    bundle.report["conditioning_contract"].update(
        {
            "inference_first_frame_mode": "clean_preserved_latent_slot",
            "legacy_soft_anchor_accepted": False,
            "no_anchor_ablation_accepted": False,
            "cfg_unconditional_mode": "empty_mq",
            "cfg_wan_reference_preserved": True,
        }
    )
    return report


def validate_checkpoint_bundle(path_value: str | Path):
    bundle = core.validate_checkpoint_bundle(path_value)
    validate_strong_binding_contract(bundle)
    return bundle


class StrongBindThreeRouterWanInference(core.ThreeRouterWanInference):
    """Three-router Wan inference with an immutable strong-binding contract."""

    def __init__(self, args: Any, bundle: Any) -> None:
        if args.first_frame_mode != "preserved":
            raise ValueError("Strong binding cannot run without a preserved first frame")
        if args.cfg_uncond_mode != "empty_mq":
            raise ValueError("Only the jointly trained empty-MQ CFG branch is allowed")
        if "strong_binding_validation" not in bundle.report:
            validate_strong_binding_contract(bundle)
        super().__init__(args, bundle)
        self.report["runtime"]["strong_binding_contract"] = bundle.report[
            "strong_binding_validation"
        ]["inference_contract"]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.parse_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "entrypoint": "openvid4000_3router_strongbind",
                    "args": vars(args),
                },
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
                core._jsonable(bundle.report),
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

    output_path = Path(args.output_path).expanduser().resolve()
    report_path = (
        Path(args.verify_report_path).expanduser().resolve()
        if args.verify_report_path
        else Path(f"{output_path}.verify.json").resolve()
    )
    if output_path == report_path:
        raise ValueError("--output_path and --verify_report_path must differ")

    pipeline: StrongBindThreeRouterWanInference | None = None
    try:
        pipeline = StrongBindThreeRouterWanInference(args, bundle)
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
        expected_size = tuple(
            int(value)
            for value in pipeline.report["runtime"]["generation"]["output_size"]
        )
        output_metadata = core._save_video(
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
            core._write_json(report_path, pipeline.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
