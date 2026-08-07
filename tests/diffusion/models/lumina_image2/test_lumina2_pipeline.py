# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Lumina-Image-2.0 request-batch forward path.

These exercise the request-level batching added to ``Lumina2Pipeline`` without
loading any weights: the pipeline is constructed with ``object.__new__`` and the
heavy seams (text encoder, VAE, transformer, scheduler) are mocked. They guard

* multi-request fusion collates per-request prompts into one batched
  ``encode_prompt`` call and splits the fused output back into one
  ``DiffusionOutput`` per request (in request order),
* the homogeneity guard rejects a batch whose requests disagree on the
  tensor-shaping / loop-critical params, instead of silently generating wrong
  images for all but the first request, and
* the image-to-image path initialises latents from the VAE-encoded image via
  ``scale_noise`` at the strength-derived first timestep and calls
  ``set_begin_index`` with the sliced offset, while the no-image path stays on
  the pure-noise flow.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import PIL.Image
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2 import Lumina2Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SchedulerConfig(dict):
    """dict that also allows attribute access (mimics diffusers' FrozenDict).

    ``forward`` reads both ``scheduler.config.num_train_timesteps`` (attribute)
    and ``scheduler.config.get(...)`` (mapping) — support both.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _make_lumina_sampling(**overrides):
    values = {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "guidance_scale_provided": False,
        "num_outputs_per_prompt": 0,
        "generator": None,
        "strength": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_request(request_id, prompt, **sampling_overrides):
    return SimpleNamespace(
        request_id=request_id,
        prompt=prompt,
        sampling_params=_make_lumina_sampling(**sampling_overrides),
    )


def _make_pipeline():
    pipeline = object.__new__(Lumina2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline._execution_device = torch.device("cpu")
    pipeline._guidance_scale = None
    pipeline._current_timestep = None
    pipeline._num_timesteps = None
    pipeline.default_sample_size = 128
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4))
    # Only num_train_timesteps and order are read from the scheduler in forward;
    # calculate_shift / retrieve_timesteps are monkeypatched away below.
    pipeline.scheduler = SimpleNamespace(config=_SchedulerConfig(num_train_timesteps=1000), order=1)
    return pipeline


@contextmanager
def _null_progress_bar(total=None):
    yield SimpleNamespace(update=lambda *a, **k: None)


def test_forward_batches_prompts_and_splits_outputs(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.calculate_shift",
        lambda *a, **k: 0.0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.retrieve_timesteps",
        lambda scheduler, num_inference_steps, device, **k: (torch.tensor([1.0]), 1),
    )

    pipeline = _make_pipeline()
    encode_calls = []
    prepare_latents_call = {}

    def _fake_encode_prompt(prompt, device, num_images_per_prompt=1, max_sequence_length=256):
        encode_calls.append({"prompt": prompt, "num_images_per_prompt": num_images_per_prompt})
        embeds = torch.zeros(len(prompt), 4, 8)
        mask = torch.ones(len(prompt), 4)
        return embeds, mask

    def _fake_prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
    ):
        prepare_latents_call.update(
            {"batch_size": batch_size, "generator": generator, "image": image, "timestep": timestep}
        )
        # Distinct per-request latent values so the split is checkable.
        return torch.arange(batch_size, dtype=torch.float32).view(batch_size, 1, 1, 1)

    pipeline.encode_prompt = _fake_encode_prompt
    pipeline.prepare_latents = _fake_prepare_latents
    pipeline.progress_bar = _null_progress_bar
    # No CFG for this case (guidance stays 1.0); identity noise + step keep the
    # mocked latents intact through the (single-step) denoise loop.
    pipeline.predict_noise_maybe_with_cfg = lambda **kwargs: kwargs["positive_kwargs"]["hidden_states"]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, latents, do_cfg, generator=None: latents

    gen_a = torch.Generator(device="cpu").manual_seed(1)
    gen_b = torch.Generator(device="cpu").manual_seed(2)
    batch = DiffusionRequestBatch(
        requests=[
            _make_request("lumina-a", "prompt-a", generator=gen_a),
            _make_request("lumina-b", {"prompt": "prompt-b", "negative_prompt": "neg-b"}, generator=gen_b),
        ]
    )

    outputs = pipeline.forward(batch, output_type="latent")

    # CFG is on (guidance defaults to 4.0), so there are two batched encode
    # calls: positive prompts, then per-request negative prompts. Request A is a
    # bare string (default negative ""); request B carries its own negative.
    assert len(encode_calls) == 2
    assert encode_calls[0]["prompt"] == ["prompt-a", "prompt-b"]
    assert encode_calls[1]["prompt"] == ["", "neg-b"]
    # Latents prepared for the full fused batch, with per-request generators,
    # and no image conditioning on the pure-noise path.
    assert prepare_latents_call["batch_size"] == 2
    assert prepare_latents_call["generator"] == [gen_a, gen_b]
    assert prepare_latents_call["image"] is None
    assert prepare_latents_call["timestep"] is None
    # Fused output split back into one DiffusionOutput per request, in order.
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].output, torch.tensor([[[[0.0]]]]))
    torch.testing.assert_close(outputs[1].output, torch.tensor([[[[1.0]]]]))


@pytest.mark.parametrize(
    "mismatch",
    [
        {"num_inference_steps": 20},
        {"height": 512},
        {"width": 768},
        {"guidance_scale": 7.0, "guidance_scale_provided": True},
        {"num_outputs_per_prompt": 2},
    ],
)
def test_resolve_batch_params_rejects_incompatible_requests(mismatch):
    pipeline = _make_pipeline()
    base = {"num_inference_steps": 30, "height": 1024, "width": 1024}
    batch = DiffusionRequestBatch(
        requests=[
            _make_request("a", "p", **base),
            _make_request("b", "p", **{**base, **mismatch}),
        ]
    )
    with pytest.raises(ValueError, match="must share"):
        pipeline._resolve_batch_params(
            batch,
            images=[None, None],
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            strength=0.6,
        )


def test_resolve_batch_params_accepts_matching_requests():
    pipeline = _make_pipeline()
    batch = DiffusionRequestBatch(
        requests=[
            _make_request("a", "p", num_inference_steps=30, height=1024, width=1024),
            _make_request("b", "p", num_inference_steps=30, height=1024, width=1024),
        ]
    )
    resolved = pipeline._resolve_batch_params(
        batch,
        images=[None, None],
        height=768,
        width=768,
        num_inference_steps=50,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        strength=0.6,
    )
    # Per-request values win over the pipeline defaults when provided; strength
    # is dropped from the contract because both requests are text-to-image.
    assert resolved == (1024, 1024, 30, 4.0, 1, None, False)


def test_resolve_batch_params_rejects_mixed_strength():
    pipeline = _make_pipeline()
    image = PIL.Image.new("RGB", (1024, 1024))
    batch = DiffusionRequestBatch(
        requests=[
            _make_request("a", {"prompt": "p", "multi_modal_data": {"image": image}}, strength=0.3),
            _make_request("b", {"prompt": "p", "multi_modal_data": {"image": image}}, strength=0.9),
        ]
    )
    with pytest.raises(ValueError, match="must share"):
        pipeline._resolve_batch_params(
            batch,
            images=[image, image],
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            strength=0.6,
        )


def test_resolve_batch_params_rejects_mixed_image_presence():
    pipeline = _make_pipeline()
    image = PIL.Image.new("RGB", (1024, 1024))
    batch = DiffusionRequestBatch(
        requests=[
            _make_request("a", {"prompt": "p", "multi_modal_data": {"image": image}}),
            _make_request("b", "p"),
        ]
    )
    with pytest.raises(ValueError, match="must share"):
        pipeline._resolve_batch_params(
            batch,
            images=[image, None],
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            strength=0.6,
        )


def test_resolve_batch_params_rejects_mismatched_derived_sizes():
    pipeline = _make_pipeline()
    tall = PIL.Image.new("RGB", (768, 1536))
    wide = PIL.Image.new("RGB", (1536, 768))
    batch = DiffusionRequestBatch(
        requests=[
            _make_request("a", {"prompt": "p", "multi_modal_data": {"image": tall}}, height=None, width=None),
            _make_request("b", {"prompt": "p", "multi_modal_data": {"image": wide}}, height=None, width=None),
        ]
    )
    with pytest.raises(ValueError, match="must share"):
        pipeline._resolve_batch_params(
            batch,
            images=[tall, wide],
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            strength=0.6,
        )


def test_get_timesteps_slices_from_strength():
    pipeline = _make_pipeline()
    sigmas = torch.linspace(1.0, 0.0, 11)[:-1]
    calls = {}

    def _set_begin_index(index):
        calls["begin_index"] = index

    pipeline.scheduler = SimpleNamespace(
        config=_SchedulerConfig(num_train_timesteps=1000),
        order=1,
        timesteps=sigmas,
        set_begin_index=_set_begin_index,
    )

    timesteps, remaining = pipeline.get_timesteps(num_inference_steps=10, strength=0.5)

    # strength=0.5 keeps the last 5 timesteps, and the scheduler is told to
    # start counting from the slice offset so scale_noise/step hit the right
    # sigma indices.
    assert remaining == 5
    torch.testing.assert_close(timesteps, sigmas[5:])
    assert calls["begin_index"] == 5


def test_get_timesteps_edge_strengths():
    pipeline = _make_pipeline()
    pipeline.scheduler = SimpleNamespace(
        config=_SchedulerConfig(num_train_timesteps=1000),
        order=1,
        timesteps=torch.arange(10, dtype=torch.float32),
        set_begin_index=lambda idx: None,
    )
    _, remaining_full = pipeline.get_timesteps(num_inference_steps=10, strength=1.0)
    _, remaining_none = pipeline.get_timesteps(num_inference_steps=10, strength=0.0)
    assert remaining_full == 10
    assert remaining_none == 0


def test_forward_i2i_scales_noise_at_strength_timestep(monkeypatch):
    """i2i request VAE-encodes the input image and scale_noises at t_start."""
    sigmas = torch.linspace(1.0, 0.0, 11)[:-1]
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.calculate_shift",
        lambda *a, **k: 0.0,
    )

    def _fake_retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
        scheduler.timesteps = sigmas
        return sigmas, len(sigmas)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.retrieve_timesteps",
        _fake_retrieve_timesteps,
    )

    pipeline = _make_pipeline()
    calls = {"encode": 0, "scale_noise": None, "begin_index": None, "preprocess_size": None}

    def _fake_preprocess(image, height=None, width=None):
        calls["preprocess_size"] = (height, width)
        return torch.zeros(1, 3, height, width)

    def _fake_vae_encode(pixel):
        calls["encode"] += 1
        b = pixel.shape[0]
        h, w = pixel.shape[-2], pixel.shape[-1]
        # 1024x1024 pixels compress to a 128x128 latent (vae_scale_factor * 2 = 16).
        latent_h, latent_w = h // 8, w // 8
        latent_dist = SimpleNamespace(sample=lambda generator=None: torch.ones(b, 4, latent_h, latent_w))
        return SimpleNamespace(latent_dist=latent_dist)

    def _fake_scale_noise(image_latents, timestep, noise):
        calls["scale_noise"] = {
            "image_latents": image_latents.clone(),
            "timestep": timestep.clone(),
            "noise_shape": tuple(noise.shape),
        }
        return image_latents + noise

    def _fake_set_begin_index(index):
        calls["begin_index"] = index

    pipeline.image_processor = SimpleNamespace(preprocess=_fake_preprocess)
    pipeline.vae = SimpleNamespace(
        encode=_fake_vae_encode,
        config=SimpleNamespace(shift_factor=0.1159, scaling_factor=0.3611),
    )
    pipeline.scheduler = SimpleNamespace(
        config=_SchedulerConfig(num_train_timesteps=1000),
        order=1,
        timesteps=sigmas,
        set_begin_index=_fake_set_begin_index,
        scale_noise=_fake_scale_noise,
    )

    def _fake_encode_prompt(prompt, device, num_images_per_prompt=1, max_sequence_length=256):
        embeds = torch.zeros(len(prompt), 4, 8)
        mask = torch.ones(len(prompt), 4)
        return embeds, mask

    pipeline.encode_prompt = _fake_encode_prompt
    pipeline.progress_bar = _null_progress_bar
    pipeline.predict_noise_maybe_with_cfg = lambda **kwargs: kwargs["positive_kwargs"]["hidden_states"]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, latents, do_cfg, generator=None: latents

    image = PIL.Image.new("RGB", (1024, 1024))
    gen = torch.Generator(device="cpu").manual_seed(1)
    batch = DiffusionRequestBatch(
        requests=[
            _make_request(
                "lumina-i2i",
                {"prompt": "edit-this", "multi_modal_data": {"image": image}},
                generator=gen,
                strength=0.5,
                num_inference_steps=10,
            )
        ]
    )

    pipeline.forward(batch, output_type="latent")

    assert calls["encode"] == 1
    assert calls["begin_index"] == 5
    assert calls["preprocess_size"] == (1024, 1024)
    scale_call = calls["scale_noise"]
    assert scale_call is not None
    # scale_noise's timestep is the first sigma of the strength-sliced tail.
    torch.testing.assert_close(scale_call["timestep"], sigmas[5:6])
    # image_latents = (encoded - shift_factor) * scaling_factor.
    expected_image_latents = (torch.ones(1, 4, 128, 128) - 0.1159) * 0.3611
    torch.testing.assert_close(scale_call["image_latents"], expected_image_latents)
    assert scale_call["noise_shape"] == (1, 4, 128, 128)


def test_forward_i2i_rejects_out_of_range_strength(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.calculate_shift",
        lambda *a, **k: 0.0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.retrieve_timesteps",
        lambda *a, **k: (torch.tensor([1.0]), 1),
    )

    pipeline = _make_pipeline()
    pipeline.encode_prompt = lambda *a, **k: (torch.zeros(1, 4, 8), torch.ones(1, 4))
    image = PIL.Image.new("RGB", (1024, 1024))
    batch = DiffusionRequestBatch(
        requests=[
            _make_request(
                "lumina-i2i",
                {"prompt": "p", "multi_modal_data": {"image": image}},
                strength=1.5,
            )
        ]
    )
    with pytest.raises(ValueError, match=r"strength should be in \[0.0, 1.0\]"):
        pipeline.forward(batch, output_type="latent")


def test_forward_t2i_warns_and_ignores_strength(monkeypatch, caplog):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.calculate_shift",
        lambda *a, **k: 0.0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2.retrieve_timesteps",
        lambda *a, **k: (torch.tensor([1.0]), 1),
    )

    pipeline = _make_pipeline()

    prepare_calls = {}

    def _fake_prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
    ):
        prepare_calls["image"] = image
        prepare_calls["timestep"] = timestep
        return torch.zeros(batch_size, 1, 1, 1)

    pipeline.encode_prompt = lambda prompt, *a, **k: (torch.zeros(len(prompt), 4, 8), torch.ones(len(prompt), 4))
    pipeline.prepare_latents = _fake_prepare_latents
    pipeline.progress_bar = _null_progress_bar
    pipeline.predict_noise_maybe_with_cfg = lambda **kwargs: kwargs["positive_kwargs"]["hidden_states"]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, latents, do_cfg, generator=None: latents

    batch = DiffusionRequestBatch(requests=[_make_request("lumina-a", "prompt-a", strength=0.4)])
    with caplog.at_level("WARNING"):
        outputs = pipeline.forward(batch, output_type="latent")
    # strength is silently dropped on the text-to-image path — the pure-noise
    # path runs and a warning surfaces to the log.
    assert prepare_calls["image"] is None
    assert prepare_calls["timestep"] is None
    assert len(outputs) == 1
    assert any("strength" in record.getMessage() for record in caplog.records)
