# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Lumina-Image-2.0 request-batch forward path.

These exercise the request-level batching added to ``Lumina2Pipeline`` without
loading any weights: the pipeline is constructed with ``object.__new__`` and the
heavy seams (text encoder, VAE, transformer, scheduler) are mocked. They guard

* multi-request fusion collates per-request prompts into one batched
  ``encode_prompt`` call and splits the fused output back into one
  ``DiffusionOutput`` per request (in request order), and
* the homogeneity guard rejects a batch whose requests disagree on the
  tensor-shaping / loop-critical params, instead of silently generating wrong
  images for all but the first request.
"""

from contextlib import contextmanager
from types import SimpleNamespace

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

    def _fake_prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        prepare_latents_call.update({"batch_size": batch_size, "generator": generator})
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
    # Latents prepared for the full fused batch, with per-request generators.
    assert prepare_latents_call["batch_size"] == 2
    assert prepare_latents_call["generator"] == [gen_a, gen_b]
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
    pipeline = object.__new__(Lumina2Pipeline)
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
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=4.0,
            num_images_per_prompt=1,
        )


def test_resolve_batch_params_accepts_matching_requests():
    pipeline = object.__new__(Lumina2Pipeline)
    batch = DiffusionRequestBatch(
        requests=[
            _make_request("a", "p", num_inference_steps=30, height=1024, width=1024),
            _make_request("b", "p", num_inference_steps=30, height=1024, width=1024),
        ]
    )
    resolved = pipeline._resolve_batch_params(
        batch,
        height=768,
        width=768,
        num_inference_steps=50,
        guidance_scale=4.0,
        num_images_per_prompt=1,
    )
    # Per-request values win over the pipeline defaults when provided.
    assert resolved == (1024, 1024, 30, 4.0, 1)
