import unittest
import importlib.util
from pathlib import Path

from infer_3router_wan_4x48g_mode0 import (
    DEFAULT_CHECKPOINT,
    parse_args,
    validate_checkpoint_bundle,
)
import infer_3router_planner_wan as core


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT = (
    CODE_ROOT
    / "checkpoint"
    / "three_router_mq-replaces-t5_strongbind_openvid4000_4x48g_steps150"
    / "checkpoint-final"
)


class ModeZeroStrongBindInferenceTest(unittest.TestCase):
    def test_shared_loader_resolves_the_real_wan_training_module(self):
        expected = (
            Path("/home/liuzhirui/model/Wan2.2")
            / "scripts-metaquery-single"
            / "train"
        )
        self.assertEqual(core.WAN_TRAIN_ROOT, expected)
        spec = importlib.util.find_spec("train_connector_for_wan")
        self.assertIsNotNone(spec)
        self.assertEqual(
            Path(spec.origin).resolve(),
            (expected / "train_connector_for_wan.py").resolve(),
        )

    def test_defaults_select_successful_mode_zero_checkpoint(self):
        args = parse_args(["--parse_only"])
        self.assertEqual(Path(args.checkpoint_dir), EXPECTED_CHECKPOINT)
        self.assertEqual(DEFAULT_CHECKPOINT, EXPECTED_CHECKPOINT)
        self.assertEqual(args.first_frame_mode, "preserved")
        self.assertEqual(args.cfg_uncond_mode, "empty_mq")
        self.assertEqual(args.frame_num, 49)

    def test_incompatible_reference_and_cfg_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires.*preserved"):
            parse_args(["--parse_only", "--first_frame_mode", "none"])
        with self.assertRaisesRegex(ValueError, "Only.*empty_mq"):
            parse_args(
                ["--parse_only", "--cfg_uncond_mode", "negative_mq"]
            )

    def test_successful_checkpoint_passes_complete_contract(self):
        bundle = validate_checkpoint_bundle(EXPECTED_CHECKPOINT)
        self.assertEqual(bundle.report["status"], "pass")
        self.assertEqual(bundle.report["checkpoint_step"], 150)
        self.assertTrue(all(bundle.report["checks"].values()))
        contract = bundle.report["conditioning_contract"]
        self.assertEqual(contract["order"], "processed_mq_replaces_t5")
        self.assertEqual(contract["mq_tokens"], 256)
        self.assertEqual(contract["t5_tokens_sent_to_dit"], 0)
        self.assertEqual(contract["reference_timestep"], 0)
        self.assertTrue(contract["reference_prefix_retained_for_vae_decode"])

    def test_launchers_pin_the_same_checkpoint_and_flash_attention(self):
        inference_root = CODE_ROOT / "inference"
        launcher = (
            inference_root
            / "infer_openvid4000_3router_4x48g_mode0.sh"
        ).read_text(encoding="utf-8")
        cluster_config = (
            inference_root
            / "infer_openvid4000_3router_4x48g_mode0.yaml"
        ).read_text(encoding="utf-8")
        checkpoint_name = EXPECTED_CHECKPOINT.parent.name
        self.assertIn(checkpoint_name, launcher)
        self.assertIn(checkpoint_name, cluster_config)
        self.assertIn("WAN_FLASH_ATTN_FORCE_VERSION=2", launcher)
        self.assertIn("WAN_FLASH_ATTN_FORCE_CONTIGUOUS=1", launcher)


if __name__ == "__main__":
    unittest.main()
