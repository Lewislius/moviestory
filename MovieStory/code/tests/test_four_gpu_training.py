import unittest
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from four_gpu_training.conditioning import build_dual_mode_encoder_class
from four_gpu_training.data import build_random_reference_dataset_class
from four_gpu_training.distributed import (
    GlobalBatchEquivalentSampler,
    clip_grad_norm_mixed_sharded_,
)
from four_gpu_training.random_reference import RandomReferenceTrainingMixin


class _DummyEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.wan_text_dim = 32
        self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, captions, input_images=None):
        del input_images
        batch = 1 if isinstance(captions, str) else len(captions)
        return self.weight * torch.ones(batch, 3, 32)


class _DummyDataset:
    def __init__(self, *args, **kwargs):
        del args
        self.seed = kwargs.get("seed", 42)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {
            "caption": "a prompt",
            "video": torch.zeros(3, 5, 32, 64),
            "ref_image": Image.new("RGB", (64, 32), "black"),
            "mq_ref_image": Image.new("RGB", (64, 32), "black"),
            "video_path": "/tmp/dummy-video.mp4",
        }


class _RecordingWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, x, **kwargs):
        del kwargs
        self.inputs = [value.detach().clone() for value in x]
        return x


class _DummyReferenceBase:
    def __init__(self):
        self.wan = SimpleNamespace(model=_RecordingWan())
        self.random_draws = []

    def _encode_ref_image_to_latent(self, ref_img, latent_h, latent_w, z_channels):
        self.seen_reference = ref_img
        return torch.full((z_channels, 1, latent_h, latent_w), 7.0)

    def _compute_loss(self, batch):
        self.random_draws.append(float(torch.rand(()).item()))
        self._encode_ref_image_to_latent(None, 2, 2, 2)
        self.wan.model([torch.zeros(2, 4, 2, 2)])
        return torch.tensor(0.0)


class _RandomReferenceTrainer(RandomReferenceTrainingMixin, _DummyReferenceBase):
    pass


class FourGPUTrainingTests(unittest.TestCase):
    def test_mode_zero_is_mq_only(self):
        encoder_class = build_dual_mode_encoder_class(
            _DummyEncoder,
            conditioning_mode=0,
        )
        encoder = encoder_class()
        output = encoder(["x", "y"])
        self.assertEqual(tuple(output.shape), (2, 3, 32))
        self.assertIsNone(encoder.mq_to_t5_mapper)
        self.assertEqual(encoder.last_context_audit["mode"], 0)

    def test_mode_one_maps_then_concatenates_t5(self):
        encoder_class = build_dual_mode_encoder_class(
            _DummyEncoder,
            conditioning_mode=1,
            mapper_bottleneck_size=8,
        )
        encoder = encoder_class()
        encoder.bind_t5_provider(
            lambda captions: [torch.full((2, 32), 2.0) for _ in captions]
        )
        output = encoder(["x", "y"])
        self.assertEqual(tuple(output.shape), (2, 5, 32))
        self.assertTrue(
            torch.equal(output[:, 3:], torch.full((2, 2, 32), 2.0))
        )
        output.mean().backward()
        mapper_grads = [
            parameter.grad
            for parameter in encoder.mq_to_t5_mapper.parameters()
        ]
        self.assertTrue(any(grad is not None for grad in mapper_grads))

    def test_sampler_preserves_one_global_batch_stream(self):
        world_size = 4
        global_batch = 8
        steps = 5
        samplers = [
            GlobalBatchEquivalentSampler(
                40,
                rank=rank,
                world_size=world_size,
                global_batch_size=global_batch,
                optimizer_steps=steps,
                seed=19,
            )
            for rank in range(world_size)
        ]
        rows = [list(sampler) for sampler in samplers]
        reconstructed = []
        for step in range(steps):
            for rank in range(world_size):
                reconstructed.extend(rows[rank][step * 2 : step * 2 + 2])
        expected = samplers[0]._global_stream().tolist()
        self.assertEqual(reconstructed, expected)
        self.assertEqual(len(set(reconstructed)), 40)
        reconstructed_draw_ids = []
        for step in range(steps):
            for rank in range(world_size):
                reconstructed_draw_ids.extend(
                    value.global_draw_id
                    for value in rows[rank][step * 2 : step * 2 + 2]
                )
        self.assertEqual(reconstructed_draw_ids, list(range(40)))

    def test_dataset_replaces_only_reference(self):
        dataset_class = build_random_reference_dataset_class(
            _DummyDataset,
            joint_null_prob=0.0,
        )
        dataset = dataset_class(
            seed=3,
            null_caption_prob=0.0,
            null_image_prob=0.0,
        )

        def fake_random_reference(self, video_path, *, key, height, width):
            del self, video_path, key
            return Image.new("RGB", (width, height), "red"), 7, 20

        dataset._random_reference = MethodType(fake_random_reference, dataset)
        sample = dataset[0]
        self.assertEqual(tuple(sample["video"].shape), (3, 5, 32, 64))
        self.assertEqual(sample["reference_frame_index"], 7)
        self.assertEqual(sample["reference_total_frames"], 20)
        self.assertEqual(sample["moviestory_global_draw_id"], 0)
        self.assertIsInstance(sample["moviestory_sample_seed"], int)
        self.assertEqual(sample["ref_image"].getpixel((0, 0)), (255, 0, 0))
        self.assertIs(sample["ref_image"], sample["mq_ref_image"])

    def test_random_reference_is_clean_prefix_without_target_trim(self):
        trainer = _RandomReferenceTrainer()
        reference = Image.new("RGB", (8, 8), "white")
        batch = {
            "video": [torch.zeros(3, 5, 8, 8)],
            "ref_image": [reference],
            "mq_ref_image": [None],
            "moviestory_sample_seed": [1234],
        }
        trainer._compute_loss(batch)
        self.assertEqual(trainer.seen_reference.mode, "RGB")
        self.assertEqual(trainer.seen_reference.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(tuple(trainer.wan.model.inputs[0].shape), (2, 4, 2, 2))
        self.assertTrue(
            torch.equal(
                trainer.wan.model.inputs[0][:, :1],
                torch.full((2, 1, 2, 2), 7.0),
            )
        )
        self.assertTrue(
            torch.equal(
                trainer.wan.model.inputs[0][:, 1:],
                torch.zeros(2, 3, 2, 2),
            )
        )

    def test_sample_rng_is_topology_neutral(self):
        reference = Image.new("RGB", (8, 8), "white")
        batch = {
            "video": [torch.zeros(3, 5, 8, 8)],
            "ref_image": [reference],
            "mq_ref_image": [reference],
            "moviestory_sample_seed": [987654],
        }
        first = _RandomReferenceTrainer()
        second = _RandomReferenceTrainer()
        first._compute_loss(batch)
        torch.manual_seed(77)
        _ = torch.rand(23)
        second._compute_loss(batch)
        self.assertEqual(first.random_draws, second.random_draws)

    def test_mixed_sharded_global_gradient_clip(self):
        replicated = nn.Parameter(torch.zeros(2))
        sharded = nn.Parameter(torch.zeros(2))
        replicated.grad = torch.tensor([3.0, 4.0])
        sharded.grad = torch.tensor([0.0, 12.0])
        norm = clip_grad_norm_mixed_sharded_(
            replicated_parameters=[replicated],
            sharded_parameters=[sharded],
            max_norm=6.5,
        )
        self.assertAlmostEqual(float(norm.item()), 13.0, places=5)
        self.assertTrue(
            torch.allclose(replicated.grad, torch.tensor([1.5, 2.0]), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(sharded.grad, torch.tensor([0.0, 6.0]), atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
