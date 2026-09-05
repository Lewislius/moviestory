import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from infer_3router_planner_wan_strongbind import (
    DEFAULT_CHECKPOINT,
    parse_args,
    validate_strong_binding_contract,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
TARGET_CHECKPOINT = (
    CODE_ROOT
    / "checkpoint"
    / "three_router_mq256_conn24_strongbind_openvid4000_steps500"
    / "checkpoint-final"
)


def _read_json(name):
    return json.loads((TARGET_CHECKPOINT / name).read_text(encoding="utf-8"))


def _target_bundle():
    return SimpleNamespace(
        router=_read_json("three_router_config.json"),
        training_args=_read_json("training_args.json"),
        config=_read_json("config.json"),
        trainer_state=_read_json("trainer_state.json"),
        report={"conditioning_contract": {}},
    )


class StrongBindInferenceTest(unittest.TestCase):
    def test_defaults_select_strong_checkpoint_and_preserved_empty_mq(self):
        args = parse_args(["--parse_only"])
        self.assertEqual(Path(args.checkpoint_dir), DEFAULT_CHECKPOINT)
        self.assertEqual(args.first_frame_mode, "preserved")
        self.assertEqual(args.cfg_uncond_mode, "empty_mq")
        self.assertEqual(args.frame_num, 49)

    def test_no_anchor_and_untrained_cfg_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires.*preserved"):
            parse_args(["--parse_only", "--first_frame_mode", "none"])
        with self.assertRaisesRegex(ValueError, "only accepts.*empty_mq"):
            parse_args(["--parse_only", "--cfg_uncond_mode", "negative_mq"])

    def test_target_checkpoint_metadata_passes_full_strong_contract(self):
        bundle = _target_bundle()
        report = validate_strong_binding_contract(bundle)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(
            bundle.report["conditioning_contract"][
                "cfg_wan_reference_preserved"
            ]
        )

    def test_legacy_first_frame_metadata_is_rejected(self):
        bundle = _target_bundle()
        bundle.router = copy.deepcopy(bundle.router)
        bundle.router["wan_first_frame_conditioning"]["mode"] = (
            "legacy_t2v_soft_anchor"
        )
        with self.assertRaisesRegex(RuntimeError, "clean_reference_metadata"):
            validate_strong_binding_contract(bundle)


if __name__ == "__main__":
    unittest.main()
