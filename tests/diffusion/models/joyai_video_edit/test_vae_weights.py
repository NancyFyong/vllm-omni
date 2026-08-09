# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight checks for the JoyAI-Video-Edit VAE.

Skipped unless ``JOYAI_VIDEO_EDIT_MODEL_DIR`` points at a local checkout of the weights, so CI stays
green without them. These are the checks that need the actual checkpoint:

**The whitening oracle is the important one.** ``vae/config.json`` ships 64-long ``latents_mean`` /
``latents_std`` that the model authors fit over their training distribution using *their* encoder.
If those constants standardise this port's encoder output to roughly zero-mean/unit-variance, the
encode path — Stem resize, patchify channel order, spatial axis order, the per-conv temporal cache —
must agree with theirs. Nothing else available without a reference implementation pins the encoder
down this tightly, and unlike a reconstruction metric it is sensitive to *scale*, which is exactly
what the DiT downstream cares about.

Reconstruction quality is recorded as a much looser floor. This VAE compresses 121x by element count
(3x9x720x1248 pixels into 64x2x30x52 latents), roughly 2.5x harder than a typical 8x/4-channel image
VAE, and it is measured here at ~26 dB on static content — so a >30 dB gate is unreachable by
construction, not a sign of a defect. The useful signal in it is *uniformity across frames*, which is
what a broken cross-chunk cache would disturb.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from vllm_omni.diffusion.models.joyai_video_edit import joyai_video_edit_vae as vae_mod
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import VAE_LATENT_CHANNELS

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.gpu]

MODEL_DIR_ENV = "JOYAI_VIDEO_EDIT_MODEL_DIR"
# Both divisible by 24. Small enough to stay quick on CPU, large enough for real image statistics.
HEIGHT, WIDTH = 240, 432
NUM_FRAMES = 9
PAN_PIXELS_PER_FRAME = 2
# Natural photo, not a synthetic pattern: the shipped latent statistics were fit on real footage.
SOURCE_IMAGE = Path(__file__).resolve().parents[3] / "assets" / "hunyuan" / "hunyuan_image_ref.png"


def _vae_dir() -> Path:
    root = os.environ.get(MODEL_DIR_ENV)
    if not root:
        pytest.skip(f"set {MODEL_DIR_ENV} to the JoyAI-Video-Edit weights directory to run this")
    path = Path(root) / "vae"
    if not (path / "diffusion_pytorch_model.safetensors").is_file():
        pytest.skip(f"{path} does not contain the VAE checkpoint")
    return path


@pytest.fixture(scope="module")
def vae_config() -> dict:
    return json.loads((_vae_dir() / "config.json").read_text())


@pytest.fixture(scope="module")
def loaded_vae(vae_config) -> vae_mod.JoyAIVideoEditVAE:
    """Load with ``strict=True``.

    Both key lists being empty is itself a result worth having: it means every one of the 767M
    parameters landed in a correctly-named slot, so the module tree matches the checkpoint's.
    """
    init_kwargs = {k: v for k, v in vae_config.items() if not k.startswith("_")}
    vae = vae_mod.JoyAIVideoEditVAE(**init_kwargs).eval()
    vae.load_state_dict(load_file(_vae_dir() / "diffusion_pytorch_model.safetensors"), strict=True)
    device = "cuda" if torch.accelerator.is_available() else "cpu"
    return vae.to(device=device, dtype=torch.float32)


def _clip(device: str, pan: int = PAN_PIXELS_PER_FRAME) -> torch.Tensor:
    """A real photo panned across the frame, normalised to the ``[-1, 1]`` range the Head clamps to."""
    image = np.asarray(Image.open(SOURCE_IMAGE).convert("RGB"))
    frames = np.stack([image[:HEIGHT, pan * i : pan * i + WIDTH] for i in range(NUM_FRAMES)])
    clip = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].float()
    return clip.to(device=device) / 127.5 - 1.0


def _psnr(reference: torch.Tensor, other: torch.Tensor) -> float:
    """PSNR for a signal on ``[-1, 1]``, i.e. peak-to-peak 2."""
    return 10 * math.log10(4.0 / torch.mean((reference.float() - other.float()) ** 2).item())


def test_shipped_latent_statistics_whiten_this_encoders_output(loaded_vae, vae_config):
    """The strongest available check on the encode path.

    ``latents_mean``/``latents_std`` come from the authors' own encoder. Reproducing unit variance
    through them means this port's Stem, patchify channel order, spatial axis order and temporal cache
    all match — and it is a *scale*-sensitive check, unlike reconstruction quality, so a systematic
    factor (a skipped Stem, a transposed rearrange) cannot hide in it.
    """
    device = next(loaded_vae.parameters()).device
    mean = torch.tensor(vae_config["latents_mean"], device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae_config["latents_std"], device=device).view(1, -1, 1, 1, 1)
    assert mean.numel() == VAE_LATENT_CHANNELS

    with torch.no_grad():
        latent = loaded_vae.encode(_clip(device)).latent_dist.mode()
    normalized = (latent - mean) / std

    assert abs(normalized.mean().item()) < 0.15, f"normalized latent mean {normalized.mean().item():+.3f}"
    assert 0.8 < normalized.std().item() < 1.25, f"normalized latent std {normalized.std().item():.3f}"


def test_real_weights_produce_the_documented_latent_geometry(loaded_vae):
    """24x spatial / 8x temporal, verified against the real checkpoint rather than a toy config."""
    device = next(loaded_vae.parameters()).device
    with torch.no_grad():
        latent = loaded_vae.encode(_clip(device)).latent_dist.mode()
    expected_h, expected_w = vae_mod.latent_spatial_shape(HEIGHT, WIDTH)
    assert tuple(latent.shape) == (
        1,
        VAE_LATENT_CHANNELS,
        vae_mod.num_latent_frames(NUM_FRAMES),
        expected_h,
        expected_w,
    )


def test_reconstruction_is_uniform_across_the_chunk_boundary(loaded_vae):
    """No seam where latent frame 0 (frame 0 alone) meets latent frame 1 (frames 1-8).

    A static clip isolates the spatial path, so every frame should reconstruct about equally well.
    A cross-chunk cache that is threaded wrongly shows up as a step at the boundary rather than as a
    lower average, which an overall-PSNR assertion would absorb.
    """
    device = next(loaded_vae.parameters()).device
    clip = _clip(device, pan=0)
    with torch.no_grad():
        reconstructed = loaded_vae.decode(loaded_vae.encode(clip).latent_dist.mode()).sample

    per_frame = [_psnr(clip[:, :, i], reconstructed[:, :, i]) for i in range(NUM_FRAMES)]
    assert min(per_frame) > 22.0, f"per-frame PSNR {per_frame}"
    # ~26 dB is this VAE's spatial ceiling at 121x compression; the spread is the real assertion.
    assert max(per_frame) - min(per_frame) < 2.0, f"seam suspected, per-frame PSNR {per_frame}"
