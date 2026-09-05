import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from infer_3router_wan_4x48g_mode0 import ModeZeroThreeRouterWanInference
from infer_3router_wan_4x48g_mode1 import (
    DEFAULT_CHECKPOINT,
    ModeOneThreeRouterWanInference,
    parse_args,
    validate_checkpoint_bundle,
)
from three_router_planner import ThreeRouterConfig


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT = (
    CODE_ROOT
    / "checkpoint"
    / "three_router_mapped-mq-plus-t5_strongbind_openvid4000_4x48g_steps150"
    / "checkpoint-final"
)


class _VariableContextEncoder:
    def __init__(self, tokens: int):
        self.tokens = int(tokens)
        self.last_context_audit = {
            "mode": 1,
            "mq_tokens": 256,
            "t5_tokens": self.tokens - 256,
            "context_tokens": self.tokens,
            "hidden_size": 4,
            "mq_rms": 1.0,
            "t5_rms": 1.0,
            "mq_rms_match_scale": 1.0,
            "mapper_trainable": True,
        }

    def __call__(self, captions, images):
        del captions, images
        return torch.ones(1, self.tokens, 4)


def _pipeline(tokens: int) -> ModeOneThreeRouterWanInference:
    pipeline = object.__new__(ModeOneThreeRouterWanInference)
    pipeline.context_tokens = 768
    pipeline.bundle = SimpleNamespace(
        router_config=ThreeRouterConfig(),
        config={"wan_text_dim": 4},
    )
    pipeline.mq_encoder = _VariableContextEncoder(tokens)
    return pipeline


class ModeOneVariableContextTest(unittest.TestCase):
    def test_defaults_select_successful_mode_one_strongbind_checkpoint(self):
        args = parse_args(["--parse_only"])
        self.assertEqual(DEFAULT_CHECKPOINT, EXPECTED_CHECKPOINT)
        self.assertEqual(Path(args.checkpoint_dir), EXPECTED_CHECKPOINT)
        self.assertEqual(args.first_frame_mode, "preserved")
        self.assertEqual(args.cfg_uncond_mode, "empty_mq")
        self.assertEqual(args.frame_num, 49)

    def test_target_checkpoint_passes_complete_mode_one_contract(self):
        bundle = validate_checkpoint_bundle(EXPECTED_CHECKPOINT)
        self.assertEqual(bundle.report["status"], "pass")
        self.assertEqual(bundle.report["checkpoint_step"], 150)
        self.assertTrue(all(bundle.report["checks"].values()))
        contract = bundle.report["conditioning_contract"]
        self.assertEqual(
            contract["order"],
            "mapped_mq_then_frozen_t5_prompt_tokens",
        )
        self.assertEqual(contract["mq_tokens"], 256)
        self.assertEqual(contract["t5_token_capacity"], 512)
        self.assertEqual(contract["dit_text_len_capacity"], 768)
        self.assertEqual(contract["reference_timestep"], 0)
        self.assertTrue(contract["reference_prefix_retained_for_vae_decode"])

    def test_strongbind_reference_slot_is_retained_for_decode(self):
        for pipeline_class in (
            ModeZeroThreeRouterWanInference,
            ModeOneThreeRouterWanInference,
        ):
            with self.subTest(mode=pipeline_class.__name__):
                pipeline = object.__new__(pipeline_class)
                self.assertEqual(pipeline._extra_reference_prefix_slots(), 0)

    def test_actual_t5_length_is_variable_below_512_token_capacity(self):
        context, audit = _pipeline(288)._encode_mode_one_context("prompt", None)
        self.assertEqual(tuple(context.shape), (1, 288, 4))
        self.assertEqual(audit["mq_tokens"], 256)
        self.assertEqual(audit["t5_tokens"], 32)
        self.assertEqual(audit["context_tokens"], 288)

    def test_context_must_include_t5_and_fit_wan_768_capacity(self):
        for invalid_tokens in (256, 769):
            with self.subTest(tokens=invalid_tokens), self.assertRaises(
                RuntimeError
            ):
                _pipeline(invalid_tokens)._encode_mode_one_context(
                    "prompt", None
                )

    def test_launchers_pin_target_checkpoint_and_training_attention(self):
        inference_root = CODE_ROOT / "inference"
        launcher = (
            inference_root / "infer_openvid4000_3router_4x48g_mode1.sh"
        ).read_text(encoding="utf-8")
        cluster_config = (
            inference_root / "infer_openvid4000_3router_4x48g_mode1.yaml"
        ).read_text(encoding="utf-8")
        checkpoint_name = EXPECTED_CHECKPOINT.parent.name
        self.assertIn(checkpoint_name, launcher)
        self.assertIn(checkpoint_name, cluster_config)
        self.assertIn("export WAN_FLASH_ATTN_FORCE_VERSION=2", launcher)
        self.assertIn("export WAN_FLASH_ATTN_FORCE_CONTIGUOUS=1", launcher)


if __name__ == "__main__":
    unittest.main()
