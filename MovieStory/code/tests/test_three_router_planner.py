import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from three_router_planner import (
    StrongFirstFrameTrainingMixin,
    ThreeRouterConfig,
    ThreeRouterPlanner,
    bind_clean_reference_prefix,
    build_three_router_encoder_class,
    configure_wan_first_frame_strong_binding,
    remove_first_target_latent_slot,
    resolve_wan_reference_images,
    video_start_frame_to_reference_image,
)
from train_3router_planner_wan import (
    RouterParameterUpdateTracker,
    build_config,
    build_joint_null_dataset_class,
    build_router_wandb_config,
    configure_video_ground_truth_only_loss,
    configure_wandb_metrics,
    move_parameters_to_zero_weight_decay_group,
    parse_router_args,
    router_diagnostics_to_metrics,
)


class _DummyTokenizer:
    def convert_tokens_to_ids(self, token):
        if token.startswith("<img") and token.endswith(">"):
            return 100 + int(token[4:-1])
        raise KeyError(token)


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(256, hidden_size)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[30, 0] = 1.0
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        del attention_mask, image_grid_thw, kwargs
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        has_text = inputs_embeds[:, 0, 0] > 0.5
        has_image = pixel_values is not None
        self.calls.append(
            {
                "has_text": tuple(bool(value) for value in has_text.tolist()),
                "has_image": bool(has_image),
            }
        )
        hidden = inputs_embeds
        condition = has_text.float() * 100.0
        if has_image:
            condition = condition + 1000.0
        hidden = hidden + condition[:, None, None]
        return SimpleNamespace(logits=hidden)


class _RecordingConnector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.call_lengths = []

    def forward(self, tokens):
        self.call_lengths.append(int(tokens.shape[1]))
        return self.proj(tokens)


class _DummyRopeModel:
    def __init__(self):
        self.calls = []

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        self.calls.append(
            {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "attention_mask": attention_mask,
            }
        )
        positions = torch.full(
            (3, input_ids.shape[0], input_ids.shape[1]),
            7,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return positions, torch.zeros(input_ids.shape[0], 1)


class _DummyMetaQueryEncoder(nn.Module):
    def __init__(
        self,
        qwen3vl_model_id,
        num_metaqueries,
        connector_num_hidden_layers=1,
        gradient_checkpointing=False,
        train_input_embeddings=True,
        connector_norm_init_scale=1.0,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()
        del (
            qwen3vl_model_id,
            connector_num_hidden_layers,
            gradient_checkpointing,
            train_input_embeddings,
            connector_norm_init_scale,
            dtype,
            device,
        )
        self.num_metaqueries = num_metaqueries
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.tokenizer = _DummyTokenizer()
        self._printed_forward_stats = False
        hidden_size = 4
        self.mllm_model = nn.Module()
        self.mllm_model.mllm_hidden_size = hidden_size
        self.mllm_model.mllm_type = "qwen3vl"
        self.mllm_model.boi_token_id = 90
        self.mllm_model.eoi_token_id = 91
        self.mllm_model.mllm_backbone = _DummyBackbone(hidden_size)
        self.mllm_model.connector = _RecordingConnector(hidden_size)
        self.tokenize = self._tokenize

    @staticmethod
    def _tokenize(tokenizer, captions, input_images=None):
        del tokenizer
        if isinstance(captions, str):
            captions = [captions]
        batch = len(captions)
        rows = []
        for caption in captions:
            text_marker = 30 if caption else 31
            rows.append(
                [text_marker, 90]
                + list(range(100, 114))
                + [91]
            )
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        if input_images is None:
            return input_ids, attention_mask
        pixel_values = torch.ones(1, batch, 1)
        image_sizes = torch.ones(batch, 3, dtype=torch.long)
        return input_ids, attention_mask, pixel_values, image_sizes

    def get_trainable_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


class _DummyWanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None

    def forward(self, x, t=None, context=None, seq_len=None):
        del t, context, seq_len
        self.last_inputs = [sample.detach().clone() for sample in x]
        return x


class _DummyWandbRun:
    def __init__(self):
        self.metric_definitions = []

    def define_metric(self, name, **kwargs):
        self.metric_definitions.append((name, kwargs))


class _DummyStrongBindingBase:
    def __init__(self):
        self.args = SimpleNamespace(
            moviestory_wan_first_frame_strong_bind=True,
        )
        self.wan = SimpleNamespace(model=_DummyWanModel())
        self.video_latent = torch.arange(
            2 * 4 * 2 * 2,
            dtype=torch.float32,
        ).reshape(2, 4, 2, 2)
        self.reference_latent = torch.full((2, 1, 2, 2), 7.0)
        self.reference_sources = []

    def _encode_video(self, video_tensors):
        del video_tensors
        return [self.video_latent.clone()]

    def _encode_ref_image_to_latent(
        self,
        ref_img,
        latent_h,
        latent_w,
        z_channels,
    ):
        self.reference_sources.append(ref_img)
        self.asserted_ref_shape = (z_channels, 1, latent_h, latent_w)
        return self.reference_latent.clone()

    def _compute_loss(self, batch):
        target = self._encode_video(batch["video"])[0]
        reference = self._encode_ref_image_to_latent(
            batch["mq_ref_image"][0],
            latent_h=target.shape[2],
            latent_w=target.shape[3],
            z_channels=target.shape[0],
        )
        noisy_reference = torch.full_like(reference, 99.0)
        x_inputs = [torch.cat([noisy_reference, target], dim=1)]
        return self.wan.model(
            x_inputs,
            t=torch.ones(1),
            context=[],
            seq_len=1,
        )[0]


class _DummyStrongBindingTrainer(
    StrongFirstFrameTrainingMixin,
    _DummyStrongBindingBase,
):
    pass


class WanFirstFrameBindingTest(unittest.TestCase):
    def test_disabled_strong_binding_is_no_anchor_ablation(self):
        args = SimpleNamespace(
            enable_ti2v_first_frame_condition=False,
            train_video_conditioning_mode="wan_animate_slot",
            train_animate_ref_frames=1,
            train_animate_temporal_frames=2,
            train_animate_conditional_frames=3,
            train_animate_preserve_timestep_zero=True,
            train_animate_drop_prefix_loss=True,
            train_ref_anchor_mode="none",
            train_ref_anchor_alpha0=0.0,
            train_ref_anchor_warmup_ratio=0.0,
        )
        configure_wan_first_frame_strong_binding(args, enabled=False)
        self.assertFalse(args.moviestory_wan_first_frame_strong_bind)
        self.assertFalse(args.enable_ti2v_first_frame_condition)
        self.assertEqual(args.train_video_conditioning_mode, "legacy_t2v")
        self.assertEqual(args.train_animate_ref_frames, 0)
        self.assertEqual(args.train_animate_temporal_frames, 0)
        self.assertEqual(args.train_animate_conditional_frames, 0)
        self.assertFalse(args.train_animate_preserve_timestep_zero)
        self.assertFalse(args.train_animate_drop_prefix_loss)
        self.assertEqual(args.train_ref_anchor_mode, "none")
        self.assertEqual(args.train_ref_anchor_alpha0, 0.0)
        self.assertEqual(args.train_ref_anchor_warmup_ratio, 0.0)

    def test_configuration_forces_one_preserved_reference_slot(self):
        args = SimpleNamespace(
            train_video_conditioning_mode="legacy_t2v",
            train_ref_anchor_mode="animate_like",
        )
        configure_wan_first_frame_strong_binding(args, enabled=True)
        self.assertTrue(args.moviestory_wan_first_frame_strong_bind)
        self.assertTrue(args.enable_ti2v_first_frame_condition)
        self.assertEqual(args.train_video_conditioning_mode, "wan_animate_slot")
        self.assertEqual(args.train_animate_ref_frames, 1)
        self.assertEqual(args.train_animate_temporal_frames, 0)
        self.assertEqual(args.train_animate_conditional_frames, 0)
        self.assertTrue(args.train_animate_preserve_timestep_zero)
        self.assertTrue(args.train_animate_drop_prefix_loss)
        self.assertEqual(args.train_ref_anchor_mode, "none")

    def test_first_target_slot_is_removed_without_modifying_source(self):
        source = torch.arange(2 * 4 * 2 * 2).reshape(2, 4, 2, 2)
        source_before = source.clone()
        trimmed = remove_first_target_latent_slot([source])
        self.assertEqual(trimmed[0].shape, (2, 3, 2, 2))
        torch.testing.assert_close(trimmed[0], source[:, 1:])
        torch.testing.assert_close(source, source_before)

    def test_clean_reference_replaces_only_wan_prefix(self):
        noisy = torch.full((2, 4, 2, 2), 9.0)
        noisy[:, 1:] = 3.0
        noisy_before = noisy.clone()
        reference = torch.full((2, 1, 2, 2), 5.0)
        bound = bind_clean_reference_prefix([noisy], [reference])[0]
        torch.testing.assert_close(bound[:, :1], reference)
        torch.testing.assert_close(bound[:, 1:], noisy[:, 1:])
        torch.testing.assert_close(noisy, noisy_before)

    def test_mixin_uses_direct_ref_image_and_preserves_original_length(self):
        trainer = _DummyStrongBindingTrainer()
        output = trainer._compute_loss(
            {
                "caption": ["walk"],
                "video": ["video"],
                "mq_ref_image": ["dropped-mq-reference"],
                "ref_image": ["direct-reference"],
            }
        )
        self.assertEqual(trainer.reference_sources, ["direct-reference"])
        self.assertEqual(output.shape, trainer.video_latent.shape)
        torch.testing.assert_close(
            output[:, :1],
            trainer.reference_latent,
        )
        torch.testing.assert_close(
            output[:, 1:],
            trainer.video_latent[:, 1:],
        )
        self.assertEqual(len(trainer.wan.model._forward_pre_hooks), 0)
        self.assertFalse(trainer._moviestory_binding_active)

    def test_missing_reference_falls_back_to_video_start_frame(self):
        video = torch.tensor(
            [
                [[[-1.0, 1.0]], [[0.0, 0.0]]],
                [[[0.0, -1.0]], [[0.0, 0.0]]],
                [[[1.0, 0.0]], [[0.0, 0.0]]],
            ]
        )
        reference = video_start_frame_to_reference_image(video)
        self.assertEqual(reference.size, (2, 1))
        self.assertEqual(
            [reference.getpixel((0, 0)), reference.getpixel((1, 0))],
            [(0, 128, 255), (255, 0, 128)],
        )
        mid_gray = video_start_frame_to_reference_image(
            torch.zeros(3, 1, 1, 1)
        )
        self.assertEqual(mid_gray.getpixel((0, 0)), (128, 128, 128))
        resolved = resolve_wan_reference_images(
            {
                "video": [video],
                "ref_image": [None],
                "mq_ref_image": [None],
            }
        )
        self.assertEqual(
            [resolved[0].getpixel((0, 0)), resolved[0].getpixel((1, 0))],
            [reference.getpixel((0, 0)), reference.getpixel((1, 0))],
        )

        trainer = _DummyStrongBindingTrainer()
        trainer._compute_loss(
            {
                "caption": ["walk"],
                "video": [video],
            }
        )
        self.assertEqual(len(trainer.reference_sources), 1)
        self.assertEqual(
            [
                trainer.reference_sources[0].getpixel((0, 0)),
                trainer.reference_sources[0].getpixel((1, 0)),
            ],
            [reference.getpixel((0, 0)), reference.getpixel((1, 0))],
        )


class ThreeRouterPlannerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.config = ThreeRouterConfig(
            hidden_size=32,
            role_tokens=6,
            action_tokens=5,
            global_tokens=3,
        )
        self.planner = ThreeRouterPlanner(self.config)

    def test_shapes_and_layout(self):
        seed = torch.randn(2, 14, 32)
        output = self.planner(seed)
        self.assertEqual(output.tokens.shape, (2, 14, 32))
        self.assertEqual(output.role.shape, (2, 6, 32))
        self.assertEqual(output.action.shape, (2, 5, 32))
        self.assertEqual(output.global_route.shape, (2, 3, 32))

    def test_training_defaults_use_256_token_layout(self):
        router_args, base_argv = parse_router_args([])
        config = build_config(router_args)
        self.assertEqual(base_argv, [])
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.total_tokens, 256)
        self.assertEqual(
            config.route_slices,
            {
                "role": (0, 96),
                "action": (96, 192),
                "global": (192, 256),
            },
        )
        self.assertTrue(router_args.wan_first_frame_strong_bind)
        self.assertEqual(router_args.joint_null_prob, 0.1)

    def test_wandb_config_includes_router_layout(self):
        wandb_config = build_router_wandb_config(
            {"learning_rate": 1e-5},
            self.config,
            enabled=True,
            wan_first_frame_strong_bind=False,
        )
        self.assertEqual(wandb_config["learning_rate"], 1e-5)
        self.assertTrue(wandb_config["three_router_enabled"])
        self.assertEqual(wandb_config["router_role_tokens"], 6)
        self.assertEqual(wandb_config["router_action_tokens"], 5)
        self.assertEqual(wandb_config["router_global_tokens"], 3)
        self.assertEqual(wandb_config["router_total_tokens"], 14)
        self.assertFalse(wandb_config["wan_first_frame_strong_bind"])
        self.assertEqual(
            wandb_config["loss_contract"],
            "video_ground_truth_velocity_mse_only",
        )

    def test_joint_null_keeps_wan_reference_and_does_not_mutate_base_sample(self):
        class DummyDataset:
            def __init__(self):
                self.sample = {
                    "caption": "walk",
                    "mq_ref_image": "mq-image",
                    "ref_image": "wan-reference",
                    "video": "video",
                }

            def __getitem__(self, index):
                del index
                return self.sample

        dataset_class = build_joint_null_dataset_class(DummyDataset, 1.0)
        dataset = dataset_class()
        sample = dataset[0]
        self.assertEqual(sample["caption"], "")
        self.assertIsNone(sample["mq_ref_image"])
        self.assertEqual(sample["ref_image"], "wan-reference")
        self.assertTrue(sample["moviestory_joint_null"])
        self.assertEqual(dataset.sample["caption"], "walk")
        self.assertEqual(dataset.sample["mq_ref_image"], "mq-image")

    def test_loss_contract_disables_every_auxiliary_loss(self):
        args = SimpleNamespace()
        configure_video_ground_truth_only_loss(args)
        self.assertFalse(args.enable_t5_alignment)
        self.assertEqual(args.lambda_t5_align_l2, 0.0)
        self.assertEqual(args.lambda_t5_align_cos, 0.0)
        self.assertEqual(args.lambda_t5_align_stats, 0.0)
        self.assertFalse(args.enable_mq_image_preserve)
        self.assertEqual(args.lambda_mq_image_preserve, 0.0)
        self.assertFalse(args.enable_wan_func_distill)
        self.assertEqual(args.lambda_wan_func_distill, 0.0)
        self.assertEqual(
            args.moviestory_loss_contract,
            "video_ground_truth_velocity_mse_only",
        )

    def test_wandb_metrics_use_optimizer_step_and_loss_summaries(self):
        run = _DummyWandbRun()
        configure_wandb_metrics(run)
        definitions = dict(run.metric_definitions)
        self.assertEqual(
            definitions["train/*"]["step_metric"],
            "train/step",
        )
        self.assertEqual(
            definitions["train/loss_step"]["summary"],
            "min",
        )
        self.assertEqual(
            definitions["train/loss_ema"]["summary"],
            "min",
        )
        self.assertEqual(
            definitions["train/router_role_initial_delta_rms"]["summary"],
            "max",
        )

    def test_planner_is_parameter_free_identity_split(self):
        named_parameters = dict(self.planner.named_parameters())
        self.assertEqual(named_parameters, {})
        seed = torch.randn(1, 14, 32)
        output = self.planner(seed)
        torch.testing.assert_close(output.tokens, seed)
        self.assertEqual(
            set(output.diagnostics()),
            {
                "role_action_cosine",
                "role_global_cosine",
                "action_global_cosine",
                "role_rms",
                "action_rms",
                "global_rms",
            },
        )

    def test_gradient_reaches_seed(self):
        seed = torch.randn(2, 14, 32, requires_grad=True)
        output = self.planner(seed)
        output.tokens.square().mean().backward()
        self.assertIsNotNone(seed.grad)
        self.assertTrue(torch.isfinite(seed.grad).all())

    def test_rejects_invalid_layout(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.planner(torch.randn(1, 13, 32))

    def test_bfloat16_module_preserves_dtype(self):
        planner = ThreeRouterPlanner(self.config).to(dtype=torch.bfloat16)
        output = planner(torch.randn(1, 14, 32, dtype=torch.bfloat16))
        self.assertEqual(output.tokens.dtype, torch.bfloat16)

    def test_diagnostics_are_converted_to_batch_mean_metrics(self):
        diagnostics = {
            "role_action_cosine": torch.tensor([0.2, 0.4]),
            "role_global_cosine": torch.tensor([0.1, 0.3]),
            "action_global_cosine": torch.tensor([-0.2, 0.2]),
            "role_rms": torch.tensor([1.0, 2.0]),
            "action_rms": torch.tensor([2.0, 4.0]),
            "global_rms": torch.tensor([3.0, 5.0]),
        }
        metrics = router_diagnostics_to_metrics(diagnostics)
        self.assertAlmostEqual(metrics["train/router_role_action_cosine"], 0.3)
        self.assertAlmostEqual(metrics["train/router_role_rms"], 1.5)

    def test_diagnostic_metrics_ignore_non_finite_values(self):
        metrics = router_diagnostics_to_metrics(
            {
                "role_rms": torch.tensor([1.0, float("nan"), float("inf")]),
                "action_rms": torch.tensor([float("nan")]),
            }
        )
        self.assertEqual(metrics["train/router_role_rms"], 1.0)
        self.assertNotIn("train/router_action_rms", metrics)

    def test_route_metaqueries_get_zero_weight_decay_optimizer_group(self):
        regular = nn.Parameter(torch.ones(2))
        route = nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW(
            [regular, route],
            lr=1e-3,
            weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 1.0,
        )
        moved = move_parameters_to_zero_weight_decay_group(
            optimizer,
            scheduler,
            [route],
            group_name="route_metaquery_embeddings",
        )
        self.assertTrue(moved)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(len(scheduler.base_lrs), 2)
        route_group = next(
            group
            for group in optimizer.param_groups
            if group.get("name") == "route_metaquery_embeddings"
        )
        self.assertEqual(route_group["weight_decay"], 0.0)
        self.assertIs(route_group["params"][0], route)
        (regular.sum() + route.sum()).backward()
        optimizer.step()
        scheduler.step()

    def test_route_update_tracker_proves_each_optimizer_update(self):
        routes = nn.ParameterDict(
            {
                "role": nn.Parameter(torch.ones(3, dtype=torch.float32)),
                "action": nn.Parameter(torch.ones(2, dtype=torch.float32)),
                "global": nn.Parameter(torch.ones(1, dtype=torch.float32)),
            }
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "route_metaquery_embeddings",
                    "params": list(routes.parameters()),
                    "lr": 1e-5,
                    "weight_decay": 0.0,
                }
            ]
        )
        tracker = RouterParameterUpdateTracker(routes)
        optimizer.register_step_pre_hook(tracker.before_optimizer_step)
        optimizer.register_step_post_hook(tracker.after_optimizer_step)
        sum(parameter.sum() for parameter in routes.parameters()).backward()
        optimizer.step()
        for route_name in ("role", "action", "global"):
            prefix = f"train/router_{route_name}_"
            self.assertGreater(tracker.last_metrics[prefix + "step_grad_rms"], 0.0)
            self.assertGreater(
                tracker.last_metrics[prefix + "step_update_rms"],
                0.0,
            )
            self.assertEqual(
                tracker.last_metrics[prefix + "update_applied"],
                1.0,
            )
            self.assertGreater(
                tracker.last_metrics[prefix + "initial_delta_rms"],
                0.0,
            )
        self.assertEqual(
            tracker.last_metrics["train/router_all_updates_applied"],
            1.0,
        )

    def test_route_update_tracker_detects_bfloat16_stale_update(self):
        routes = nn.ParameterDict(
            {
                name: nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
                for name in ("role", "action", "global")
            }
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(routes.parameters()),
                    "lr": 1e-5,
                    "weight_decay": 0.0,
                }
            ]
        )
        tracker = RouterParameterUpdateTracker(
            routes,
            stale_update_patience=1,
        )
        optimizer.register_step_pre_hook(tracker.before_optimizer_step)
        optimizer.register_step_post_hook(tracker.after_optimizer_step)
        sum(parameter.sum() for parameter in routes.parameters()).backward()
        with self.assertRaisesRegex(RuntimeError, r"\[3-ROUTER\]\[STALE\]"):
            optimizer.step()

    def test_route_update_tracker_detects_disconnected_route(self):
        routes = nn.ParameterDict(
            {
                name: nn.Parameter(torch.ones(2, dtype=torch.float32))
                for name in ("role", "action", "global")
            }
        )
        optimizer = torch.optim.AdamW(
            [{"params": list(routes.parameters()), "lr": 1e-3}]
        )
        tracker = RouterParameterUpdateTracker(
            routes,
            stale_update_patience=1,
        )
        optimizer.register_step_pre_hook(tracker.before_optimizer_step)
        optimizer.register_step_post_hook(tracker.after_optimizer_step)
        routes["role"].sum().backward()
        with self.assertRaisesRegex(RuntimeError, r"\[3-ROUTER\]\[NO-GRAD\]"):
            optimizer.step()

    def test_adapter_isolates_qwen_modalities_then_uses_one_connector_sequence(self):
        config = ThreeRouterConfig(
            hidden_size=4,
            role_tokens=6,
            action_tokens=5,
            global_tokens=3,
        )
        encoder_class = build_three_router_encoder_class(
            _DummyMetaQueryEncoder,
            config,
        )
        encoder = encoder_class(
            qwen3vl_model_id="dummy",
            num_metaqueries=14,
            dtype=torch.float32,
            device="cpu",
        )
        for parameter in encoder.route_metaquery_embeddings.values():
            self.assertEqual(parameter.dtype, torch.float32)
        features = encoder(
            ["walk forward", "turn left"],
            [["image-a"], ["image-b"]],
        )
        self.assertEqual(features.shape, (2, 14, 4))
        self.assertEqual(
            encoder.mllm_model.connector.call_lengths,
            [14],
        )
        self.assertEqual(
            encoder.mllm_model.mllm_backbone.calls,
            [
                {"has_text": (False, False), "has_image": True},
                {"has_text": (True, True), "has_image": False},
                {"has_text": (True, True), "has_image": True},
            ],
        )
        self.assertEqual(
            encoder.last_route_input_audit["role"]["caption_nonempty"],
            [False, False],
        )
        self.assertTrue(
            encoder.last_route_input_audit["role"]["pixel_values_present"]
        )
        self.assertEqual(
            encoder.last_route_input_audit["action"]["caption_nonempty"],
            [True, True],
        )
        self.assertFalse(
            encoder.last_route_input_audit["action"]["image_input_supplied"]
        )
        self.assertEqual(
            encoder.last_route_input_audit["global"]["qwen_output_shape"],
            (2, 3, 4),
        )
        self.assertEqual(
            encoder.last_joint_connector_audit,
            {
                "call_count": 1,
                "input_shape": (2, 14, 4),
                "output_shape": (2, 14, 4),
            },
        )

    def test_qwen3vl_position_ids_preserve_image_rope_and_text_padding(self):
        encoder_class = build_three_router_encoder_class(
            _DummyMetaQueryEncoder,
            self.config,
        )
        rope_model = _DummyRopeModel()
        backbone = SimpleNamespace(model=rope_model)
        input_ids = torch.tensor([[10, 11, 12], [0, 20, 21]])
        attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        image_grid_thw = torch.ones(2, 3, dtype=torch.long)

        image_positions = encoder_class._qwen3vl_position_ids(
            backbone,
            input_ids,
            attention_mask,
            image_grid_thw,
        )
        torch.testing.assert_close(
            image_positions,
            torch.full((3, 2, 3), 7, dtype=torch.long),
        )
        self.assertEqual(len(rope_model.calls), 1)

        text_positions = encoder_class._qwen3vl_position_ids(
            SimpleNamespace(),
            input_ids,
            attention_mask,
            None,
        )
        expected_text = torch.tensor(
            [
                [[0, 1, 2], [0, 0, 1]],
                [[0, 1, 2], [0, 0, 1]],
                [[0, 1, 2], [0, 0, 1]],
            ]
        )
        torch.testing.assert_close(text_positions, expected_text)

    def test_joint_loss_updates_all_route_metaquery_parameters(self):
        config = ThreeRouterConfig(
            hidden_size=4,
            role_tokens=6,
            action_tokens=5,
            global_tokens=3,
        )
        encoder_class = build_three_router_encoder_class(
            _DummyMetaQueryEncoder,
            config,
        )
        encoder = encoder_class(
            qwen3vl_model_id="dummy",
            num_metaqueries=14,
            dtype=torch.float32,
            device="cpu",
        )
        features = encoder(
            ["walk forward"],
            [["image-a"]],
        )
        features.square().mean().backward()
        self.assertIsNone(
            encoder.mllm_model.mllm_backbone.embedding.weight.grad
        )
        role_grad = encoder.route_metaquery_embeddings["role"].grad
        action_grad = encoder.route_metaquery_embeddings["action"].grad
        global_grad = encoder.route_metaquery_embeddings["global"].grad
        self.assertIsNotNone(role_grad)
        self.assertGreater(float(role_grad.abs().sum()), 0.0)
        self.assertIsNotNone(action_grad)
        self.assertGreater(float(action_grad.abs().sum()), 0.0)
        self.assertIsNotNone(global_grad)
        self.assertGreater(float(global_grad.abs().sum()), 0.0)
        route_grad_rms = encoder.last_route_embedding_grad_rms
        self.assertGreater(float(route_grad_rms["role_mq_embedding_grad_rms"]), 0.0)
        self.assertGreater(
            float(route_grad_rms["action_mq_embedding_grad_rms"]),
            0.0,
        )
        self.assertGreater(
            float(route_grad_rms["global_mq_embedding_grad_rms"]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
