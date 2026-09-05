import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from infer_3router_planner_wan import (
    ThreeRouterWanInference,
    _save_video,
    _tensor_difference_stats,
    _tensor_rms,
    _configure_wan_flash_attention_2,
    _wan_attention_backend_report,
    _write_json,
    assess_decoded_video_quality,
    build_model_timestep_row,
    enforce_clean_reference_prefix_,
    match_mq_rms,
    parse_args,
    prepare_video_for_wan_writer,
    validate_video_metadata,
)
from infer_3router_wan_4x48g_mode0 import parse_args as parse_mode0_args
from infer_3router_wan_4x48g_mode1 import parse_args as parse_mode1_args


class ThreeRouterInferenceTest(unittest.TestCase):
    def test_parse_only_does_not_require_generation_inputs(self):
        args = parse_args(["--parse_only"])
        self.assertTrue(args.parse_only)
        self.assertEqual(args.runtime_audit, "full")
        self.assertEqual(args.guide_scale, 1.0)
        self.assertEqual(args.cfg_uncond_mode, "empty_mq")
        self.assertEqual(args.first_frame_mode, "preserved")
        self.assertEqual(args.audit_forward_retries, 1)
        self.assertEqual(args.encoder_device, 0)
        self.assertEqual(args.dit_device, 1)

    def test_mode_entrypoints_require_one_gpu(self):
        for mode_parse_args in (parse_mode0_args, parse_mode1_args):
            args = mode_parse_args(["--parse_only"])
            self.assertEqual(args.encoder_device, 0)
            self.assertEqual(args.dit_device, 0)
            with self.assertRaisesRegex(ValueError, "must be identical"):
                mode_parse_args(
                    [
                        "--parse_only",
                        "--encoder_device",
                        "0",
                        "--dit_device",
                        "1",
                    ]
                )

    def test_attention_backend_report_requires_training_aligned_fa2(self):
        report = _wan_attention_backend_report(
            SimpleNamespace(
                FLASH_ATTN_2_AVAILABLE=True,
                FLASH_ATTN_3_AVAILABLE=True,
                _FLASH_ATTN_FORCE_VERSION=2,
                _FLASH_ATTN_FORCE_CONTIGUOUS=True,
            )
        )
        self.assertEqual(report["effective"], "flash_attention_2")
        self.assertTrue(report["training_aligned"])
        self.assertTrue(report["force_contiguous"])

        with self.assertRaisesRegex(RuntimeError, "not available"):
            _wan_attention_backend_report(
                SimpleNamespace(
                    FLASH_ATTN_2_AVAILABLE=False,
                    FLASH_ATTN_3_AVAILABLE=True,
                    _FLASH_ATTN_FORCE_VERSION=2,
                )
            )

    def test_runtime_configuration_repairs_an_early_auto_import(self):
        attention_module = SimpleNamespace(
            FLASH_ATTN_2_AVAILABLE=True,
            FLASH_ATTN_3_AVAILABLE=True,
            _FLASH_ATTN_FORCE_VERSION=None,
            _FLASH_ATTN_FORCE_VERSION_STR="",
            _FLASH_ATTN_FORCE_CONTIGUOUS=False,
        )
        report = _configure_wan_flash_attention_2(attention_module)

        self.assertEqual(attention_module._FLASH_ATTN_FORCE_VERSION, 2)
        self.assertEqual(attention_module._FLASH_ATTN_FORCE_VERSION_STR, "2")
        self.assertTrue(attention_module._FLASH_ATTN_FORCE_CONTIGUOUS)
        self.assertEqual(report["effective"], "flash_attention_2")
        self.assertIsNone(report["forced_version_before_runtime_config"])
        self.assertTrue(report["runtime_force_applied"])
        self.assertTrue(report["training_aligned"])

    def test_preserved_prefix_uses_clean_latent_and_zero_timestep(self):
        latent = torch.randn(4, 3, 2, 2)
        reference = torch.full((4, 1, 2, 2), 7.0)
        enforce_clean_reference_prefix_(latent, reference)
        torch.testing.assert_close(latent[:, :1], reference)

        timestep = build_model_timestep_row(
            torch.tensor(800.0),
            seq_len=12,
            tokens_per_latent_frame=4,
            preserve_first_frame=True,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(timestep[:, :4], torch.zeros(1, 4))
        torch.testing.assert_close(timestep[:, 4:], torch.full((1, 8), 800.0))

    def test_reference_vae_input_matches_training_bfloat16_dtype(self):
        class CaptureVae:
            def __init__(self):
                self.input_dtype = None

            def encode(self, values):
                self.input_dtype = values[0].dtype
                return [torch.zeros(4, 1, 2, 2, dtype=torch.bfloat16)]

        pipeline = object.__new__(ThreeRouterWanInference)
        pipeline.dit_device = torch.device("cpu")
        pipeline.wan = SimpleNamespace(vae=CaptureVae())
        latent = pipeline._encode_reference_latent(torch.zeros(3, 1, 16, 16))

        self.assertEqual(pipeline.wan.vae.input_dtype, torch.bfloat16)
        self.assertEqual(latent.dtype, torch.float32)
        self.assertEqual(tuple(latent.shape), (4, 1, 2, 2))

    def test_writer_contract_adds_batch_dimension(self):
        video = torch.zeros(3, 5, 32, 48)
        writer_input = prepare_video_for_wan_writer(
            video,
            expected_frame_num=5,
            expected_size=(48, 32),
        )
        self.assertEqual(tuple(writer_input.shape), (1, 3, 5, 32, 48))
        with self.assertRaisesRegex(ValueError, "shape must be"):
            prepare_video_for_wan_writer(
                video,
                expected_frame_num=4,
                expected_size=(48, 32),
            )

    def test_save_video_writes_and_verifies_real_container_dimensions(self):
        captured_shape = None

        def fake_wan_save_video(tensor, save_file, fps, **kwargs):
            del kwargs
            nonlocal captured_shape
            captured_shape = tuple(tensor.shape)
            import imageio.v2 as imageio

            frames = (
                tensor[0]
                .permute(1, 2, 3, 0)
                .clamp(-1, 1)
                .add(1)
                .mul(127.5)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            writer = imageio.get_writer(
                save_file,
                fps=fps,
                codec="libx264",
            )
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

        wan_module = ModuleType("wan")
        utils_package = ModuleType("wan.utils")
        utils_module = ModuleType("wan.utils.utils")
        utils_module.save_video = fake_wan_save_video
        wan_module.utils = utils_package
        utils_package.utils = utils_module
        fake_modules = {
            "wan": wan_module,
            "wan.utils": utils_package,
            "wan.utils.utils": utils_module,
        }
        with TemporaryDirectory() as directory, patch.dict(
            sys.modules,
            fake_modules,
        ):
            output = Path(directory) / "dimension-contract.mp4"
            metadata = _save_video(
                torch.zeros(3, 5, 32, 48),
                output,
                24,
                expected_frame_num=5,
                expected_size=(48, 32),
            )
            self.assertEqual(captured_shape, (1, 3, 5, 32, 48))
            self.assertTrue(output.is_file())
            self.assertEqual(metadata["width"], 48)
            self.assertEqual(metadata["height"], 32)
            self.assertEqual(metadata["frame_num"], 5)
            self.assertEqual(metadata["status"], "pass")

    def test_video_metadata_mismatch_is_fatal(self):
        validate_video_metadata(
            {"width": 48, "height": 32, "frame_num": 5, "fps": 24.0},
            expected_frame_num=5,
            expected_size=(48, 32),
            expected_fps=24,
        )
        with self.assertRaisesRegex(RuntimeError, "frame_num=512"):
            validate_video_metadata(
                {"width": 512, "height": 64, "frame_num": 512, "fps": 24.0},
                expected_frame_num=49,
                expected_size=(512, 512),
                expected_fps=24,
            )

    def test_nonfinite_values_cannot_pass_audit_or_json(self):
        with self.assertRaises(FloatingPointError):
            _tensor_rms(torch.tensor([float("inf")]))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                _write_json(path, {"diff_rms": float("inf")})

    def test_condition_difference_stats_distinguish_collapse_from_change(self):
        context = torch.ones(2, 3, dtype=torch.bfloat16)
        collapsed = _tensor_difference_stats(
            context,
            context.clone(),
            label="collapsed context",
        )
        self.assertTrue(collapsed["exact_equal"])
        self.assertEqual(collapsed["diff_rms"], 0.0)
        self.assertEqual(collapsed["changed_fraction"], 0.0)

        ablated = context.clone()
        ablated[0, 0] = 0
        changed = _tensor_difference_stats(
            context,
            ablated,
            label="changed context",
        )
        self.assertFalse(changed["exact_equal"])
        self.assertGreater(changed["diff_rms"], 0.0)
        self.assertGreater(changed["changed_fraction"], 0.0)

    def test_decoded_video_quality_rejects_snow_noise(self):
        axis = torch.linspace(-1.0, 1.0, 32)
        reference_frame = (
            axis.view(1, 32).expand(32, 32)
            + axis.view(32, 1).expand(32, 32)
        ).clamp(-1, 1)
        reference = reference_frame.repeat(3, 1, 1).unsqueeze(1)

        smooth_video = reference.repeat(1, 5, 1, 1)
        smooth_audit = assess_decoded_video_quality(
            smooth_video,
            reference,
            max_high_frequency_ratio=0.9,
            min_reference_correlation=0.5,
            max_reference_mae=0.4,
        )
        self.assertEqual(smooth_audit["status"], "pass")

        generator = torch.Generator().manual_seed(7)
        snow_video = smooth_video.clone()
        snow_video[:, 1:] = torch.randn(
            snow_video[:, 1:].shape,
            generator=generator,
        )
        snow_audit = assess_decoded_video_quality(
            snow_video,
            reference,
            max_high_frequency_ratio=0.9,
            min_reference_correlation=0.5,
            max_reference_mae=0.4,
        )
        self.assertEqual(snow_audit["status"], "fail")
        self.assertGreater(
            snow_audit["generated_spatial_high_frequency_ratio"],
            0.9,
        )

    def test_mq_rms_matching_uses_t5_only_as_scale_reference(self):
        mq = torch.full((1, 4, 3), 2.0)
        t5 = torch.full((8, 3), 0.5)
        matched, report = match_mq_rms(mq, t5, 0.03, 4.0)
        torch.testing.assert_close(matched, torch.full_like(mq, 0.5))
        self.assertAlmostEqual(report["applied_scale"], 0.25)
        self.assertEqual(tuple(matched.shape), tuple(mq.shape))

    def test_mq_rms_scale_is_clipped(self):
        mq = torch.ones(1, 2, 2)
        t5 = torch.full((2, 2), 100.0)
        matched, report = match_mq_rms(mq, t5, 0.03, 4.0)
        self.assertAlmostEqual(report["applied_scale"], 4.0)
        torch.testing.assert_close(matched, torch.full_like(mq, 4.0))

    def test_first_step_zero_image_influence_is_fatal(self):
        class ProbeModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, _latent, *, t, context, seq_len):
                del t, seq_len
                self.calls += 1
                if self.calls == 1:
                    # The all-zero MQ probe must change the prediction.
                    return [torch.zeros(2, 2)]
                # The image-ablated probe can quantize to the same first-step
                # prediction when the reference latent is fully anchored.
                self.assert_distinct_image_context(context[0])
                return [torch.ones(2, 2)]

            @staticmethod
            def assert_distinct_image_context(context):
                if torch.equal(context, torch.ones_like(context)):
                    raise AssertionError("expected an image-ablated context")

        pipeline = object.__new__(ThreeRouterWanInference)
        pipeline.args = SimpleNamespace(audit_epsilon=1e-6)
        pipeline.wan = SimpleNamespace(model=ProbeModel())
        pipeline.report = {"runtime": {}}

        with self.assertRaisesRegex(
            RuntimeError,
            "unchanged after removing image conditioning",
        ):
            pipeline._verify_wan_context_influence(
                pred_conditioned=torch.ones(2, 2),
                latent_input=[torch.zeros(1)],
                timestep=torch.zeros(1),
                seq_len=1,
                context=[torch.ones(2, 2)],
                no_image_context=torch.full((2, 2), 0.5),
            )

        audit = pipeline.report["runtime"]["wan_context_influence"]
        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["image_condition_status"], "fail")
        self.assertGreater(audit["mq_vs_zero_diff_rms"], 0.0)

    def test_all_mq_context_still_has_to_influence_wan(self):
        class ContextIgnoringModel:
            def __call__(self, _latent, *, t, context, seq_len):
                del t, context, seq_len
                return [torch.ones(2, 2)]

        pipeline = object.__new__(ThreeRouterWanInference)
        pipeline.args = SimpleNamespace(audit_epsilon=1e-6)
        pipeline.wan = SimpleNamespace(model=ContextIgnoringModel())
        pipeline.report = {"runtime": {}}

        with self.assertRaisesRegex(
            RuntimeError,
            "all MQ/T5 context is zeroed",
        ):
            pipeline._verify_wan_context_influence(
                pred_conditioned=torch.ones(2, 2),
                latent_input=[torch.zeros(1)],
                timestep=torch.zeros(1),
                seq_len=1,
                context=[torch.ones(2, 2)],
                no_image_context=None,
            )
        self.assertEqual(
            pipeline.report["runtime"]["wan_context_influence"]["status"],
            "fail",
        )

    def test_transient_zero_difference_is_retried_and_replaced(self):
        class TransientModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, _latent, *, t, context, seq_len):
                del _latent, t, seq_len
                self.calls += 1
                if self.calls == 1:
                    # First zero-context probe aliases the supplied original
                    # prediction and triggers the guarded retry.
                    return [torch.ones(2, 2)]
                if bool(context[0].ne(0).any()):
                    return [torch.full((2, 2), 2.0)]
                return [torch.zeros(2, 2)]

        pipeline = object.__new__(ThreeRouterWanInference)
        pipeline.args = SimpleNamespace(
            audit_epsilon=1e-6,
            audit_forward_retries=1,
        )
        pipeline.wan = SimpleNamespace(model=TransientModel())
        pipeline.report = {"runtime": {}}

        verified = pipeline._verify_wan_context_influence(
            pred_conditioned=torch.ones(2, 2),
            latent_input=[torch.zeros(1)],
            timestep=torch.zeros(1),
            seq_len=1,
            context=[torch.ones(2, 2)],
            no_image_context=None,
        )

        torch.testing.assert_close(verified, torch.full((2, 2), 2.0))
        audit = pipeline.report["runtime"]["wan_context_influence"]
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["attempt_count"], 2)
        self.assertTrue(audit["recovered_after_retry"])
        self.assertEqual(audit["attempts"][0]["diff_rms"], 0.0)
        self.assertGreater(audit["attempts"][1]["diff_rms"], 0.0)


if __name__ == "__main__":
    unittest.main()
