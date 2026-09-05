from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from native_i2v_3router import (  # noqa: E402
    ThreeRouterConfig,
    ThreeRouterPlanner,
    build_dual_mode_encoder_class,
    build_native_i2v_condition,
    native_flow_matching_pair,
)
from native_i2v_3router.distributed import GlobalBatchSampler  # noqa: E402
from native_i2v_3router.data import build_timeout_video_dataset_class  # noqa: E402
from train.train_metaquery_i2v_3router_4x48g import (  # noqa: E402
    build_optimizer,
    enforce_no_before_training_checkpoint,
    parse_args,
)


class CapturingVAE:
    def __init__(self) -> None:
        self.input = None

    def encode(self, videos):
        self.input = videos[0].detach().clone()
        _, frames, height, width = self.input.shape
        return [torch.zeros(16, (frames - 1) // 4 + 1, height // 8, width // 8)]


def test_training_defaults_are_mode_zero_cond_only_eight_gpu():
    args = parse_args([])
    assert args.conditioning_mode == 0
    assert args.wan_train_mode == "cond_only"
    assert args.expected_world_size == 8
    assert args.global_effective_batch == 8
    assert args.enable_wan_activation_checkpointing is True


def test_native_i2v_condition_matches_wan_generate_operations():
    pixels = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    image = Image.fromarray(pixels, mode="RGB")
    vae = CapturingVAE()
    condition = build_native_i2v_condition(
        vae=vae,
        first_frame=image,
        frame_num=5,
        latent_shape=(16, 2, 4, 6),
        vae_stride=(4, 8, 8),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    native_image = TF.to_tensor(image).sub_(0.5).div_(0.5)
    expected_first = F.interpolate(
        native_image[None].cpu(), size=(32, 48), mode="bicubic"
    ).transpose(0, 1)
    assert vae.input is not None
    assert vae.input.dtype == torch.float32
    assert tuple(vae.input.shape) == (3, 5, 32, 48)
    assert torch.equal(vae.input[:, :1], expected_first)
    assert int(torch.count_nonzero(vae.input[:, 1:])) == 0
    assert tuple(condition.mask.shape) == (4, 2, 4, 6)
    assert bool(torch.all(condition.mask[:, 0] == 1))
    assert bool(torch.all(condition.mask[:, 1] == 0))
    assert tuple(condition.y.shape) == (20, 2, 4, 6)
    assert torch.equal(condition.y[:4], condition.mask)


def test_native_flow_pair_is_velocity_target():
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[6.0]]]])
    noisy, target = native_flow_matching_pair(clean, noise, torch.tensor([0.25]))
    assert torch.equal(noisy, torch.tensor([[[[3.0]]]]))
    assert torch.equal(target, torch.tensor([[[[4.0]]]]))


def test_router_is_identity_split_in_fixed_order():
    config = ThreeRouterConfig(
        hidden_size=4, role_tokens=2, action_tokens=3, global_tokens=1
    )
    tokens = torch.arange(2 * 6 * 4).reshape(2, 6, 4).float()
    output = ThreeRouterPlanner(config)(tokens)
    assert output.tokens.data_ptr() == tokens.data_ptr()
    assert torch.equal(output.role, tokens[:, :2])
    assert torch.equal(output.action, tokens[:, 2:5])
    assert torch.equal(output.global_route, tokens[:, 5:])


class FakeMQEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.wan_text_dim = 8
        self.device = torch.device("cpu")

    def forward(self, captions, input_images=None):
        return torch.ones(len(captions), 3, 8)


def test_mode_one_context_is_mapped_mq_then_frozen_t5():
    encoder_class = build_dual_mode_encoder_class(
        FakeMQEncoder,
        conditioning_mode=1,
        mapper_bottleneck_size=4,
        mapper_rms_match=False,
    )
    encoder = encoder_class()
    encoder.bind_t5_provider(lambda captions: [torch.full((2, 8), 2.0) for _ in captions])
    context = encoder(["a", "b"], None)
    assert tuple(context.shape) == (2, 5, 8)
    assert torch.equal(context[:, 3:], torch.full((2, 2, 8), 2.0))
    assert encoder.last_context_audit["context_tokens"] == 5


def test_global_sampler_partitions_one_stream_without_overlap():
    samplers = [
        GlobalBatchSampler(
            16,
            rank=rank,
            world_size=4,
            global_batch_size=8,
            optimizer_steps=2,
            seed=7,
        )
        for rank in range(4)
    ]
    rows = [list(sampler) for sampler in samplers]
    assert [len(row) for row in rows] == [4, 4, 4, 4]
    assert len({int(value) for row in rows for value in row}) == 16
    draw_ids = sorted(value.global_draw_id for row in rows for value in row)
    assert draw_ids == list(range(16))


class FakeOptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mq_encoder = nn.Module()
        self.mq_encoder.mllm_model = nn.Module()
        self.mq_encoder.mllm_model.connector = nn.Sequential(nn.Linear(2, 2))
        self.mq_encoder.route_metaquery_embeddings = nn.ParameterDict(
            {"role": nn.Parameter(torch.ones(1, 2))}
        )
        self.mq_encoder.mapper_weight = nn.Parameter(torch.ones(2, 2))
        self.wan_weight = nn.Parameter(torch.ones(2, 2))

    def mq_trainable_parameters(self):
        return list(self.mq_encoder.parameters())

    def wan_trainable_parameters(self):
        return [self.wan_weight]


def test_optimizer_separates_connector_shards_from_replicated_mq():
    model = FakeOptimizerModel()
    optimizer, replicated, sharded = build_optimizer(model, parse_args([]))
    connector_ids = {
        id(parameter)
        for parameter in model.mq_encoder.mllm_model.connector.parameters()
    }
    replicated_ids = {id(parameter) for parameter in replicated}
    sharded_ids = {id(parameter) for parameter in sharded}
    assert connector_ids <= sharded_ids
    assert connector_ids.isdisjoint(replicated_ids)
    assert id(model.mq_encoder.route_metaquery_embeddings["role"]) in replicated_ids
    assert id(model.mq_encoder.mapper_weight) in replicated_ids
    assert id(model.wan_weight) in sharded_ids
    assert [group["name"] for group in optimizer.param_groups] == [
        "mq_connector_fsdp",
        "mq_replicated",
        "route_metaquery_embeddings",
        "wan_native_parameters",
    ]
    assert optimizer.defaults["foreach"] is False


def test_bounded_video_command_enforces_timeout():
    dataset_class = build_timeout_video_dataset_class(object)
    dataset = dataset_class.__new__(dataset_class)
    dataset._failure_stats = defaultdict(int)
    result = dataset._run_video_command(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        timeout=0.05,
        failure="unit_video",
    )
    assert result is None
    assert dataset._failure_stats["unit_video_timeout"] == 1


def test_launcher_enables_bounded_preclean_and_300_second_timeout():
    launcher = (
        CODE_ROOT / "train" / "train_openvid4000_3router_i2v_a14b_4x48g.sh"
    ).read_text(encoding="utf-8")
    assert 'WAN_DATA_PRECLEAN="${WAN_DATA_PRECLEAN:-1}"' in launcher
    assert 'WAN_DATA_PRECLEAN_LOG_INTERVAL="${WAN_DATA_PRECLEAN_LOG_INTERVAL:-25}"' in launcher
    assert 'WAN_DATA_PROBE_TIMEOUT_SEC="${WAN_DATA_PROBE_TIMEOUT_SEC:-8}"' in launcher
    assert 'WAN_DATA_DECODE_TIMEOUT_SEC="${WAN_DATA_DECODE_TIMEOUT_SEC:-30}"' in launcher
    assert 'WAN_DATA_MAX_TRIALS="${WAN_DATA_MAX_TRIALS:-4}"' in launcher
    assert 'WAN_I2V_DIST_TIMEOUT_SEC="${WAN_I2V_DIST_TIMEOUT_SEC:-300}"' in launcher
    assert (
        'WAN_KEEP_BEFORE_TRAINING_CHECKPOINT="${WAN_KEEP_BEFORE_TRAINING_CHECKPOINT:-0}"'
        in launcher
    )


def test_before_training_checkpoint_is_not_retained():
    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)
        before = output_dir / "checkpoint-before-training"
        before.mkdir()
        (before / "state.pt").write_bytes(b"unused")
        enforce_no_before_training_checkpoint(output_dir, rank=0)
        assert not before.exists()
