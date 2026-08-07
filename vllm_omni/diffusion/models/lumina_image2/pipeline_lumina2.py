# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/lumina2/pipeline_lumina2.py

import inspect
import json
import math
import os
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.lumina_image2.lumina2_transformer import Lumina2Transformer2DModel
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_lumina2_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.output_type == "latent":
        return lambda x: x

    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        # VaeImageProcessor.postprocess denormalizes [-1, 1] -> [0, 1] and
        # converts to PIL (do_normalize defaults to True).
        return image_processor.postprocess(images)

    return post_process_func


class Lumina2Pipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Text-to-image and image-to-image pipeline for Lumina-Image-2.0 (Next-DiT).

    Gemma2 text encoder + AutoencoderKL VAE + Lumina2Transformer2DModel with a
    FlowMatchEulerDiscreteScheduler, ported from the diffusers ``Lumina2Pipeline``
    onto the vLLM-Omni diffusion runtime.

    A request carrying ``multi_modal_data["image"]`` takes the image-to-image
    path. Lumina-Image-2.0 ships no editing checkpoint, so image-to-image runs
    SDEdit-style on the same weights: the input image is VAE-encoded, re-noised
    at the sigma selected by ``strength``, and only the tail of the schedule is
    denoised. Requests without an image are unaffected.
    """

    # Component discovery for parallelism / CPU-offload placement.
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    # Request-level batching: multiple independent requests are fused into a
    # single denoise pass (see ``forward``). The Gemma text encoder pads every
    # prompt to a fixed ``max_sequence_length`` (default 256), so per-request
    # embeddings already share a sequence length and stack without ragged
    # padding.
    supports_request_batch: ClassVar[bool] = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        logger.info("Model path for initialization: %s", model)
        local_files_only = os.path.exists(model)
        logger.info("Local files only: %s", local_files_only)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        self.text_encoder = AutoModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=od_config.dtype,
            local_files_only=local_files_only,
        ).to(self._execution_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        self.tokenizer.padding_side = "right"

        self.vae = AutoencoderKL.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=od_config.dtype,
            local_files_only=local_files_only,
        ).to(self._execution_device)

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, Lumina2Transformer2DModel)
        self.transformer = Lumina2Transformer2DModel(quant_config=od_config.quantization_config, **transformer_kwargs)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = self.transformer.config.sample_size
        self.system_prompt = (
            "You are an assistant designed to generate superior images with the superior "
            "degree of image-text alignment based on textual prompts or user prompts."
        )

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    def predict_noise(self, **kwargs) -> torch.Tensor:
        """One transformer evaluation for the CFG mixin.

        Lumina-Image-2.0 negates the flow-matching velocity prediction (it treats
        t=0 as noise and t=1 as image). Negating here — before ``combine_cfg_noise``
        and ``cfg_normalize_function`` run — is algebraically identical to negating
        the final combined prediction: the CFG combination is affine and the
        norm-rescale ratio ``norm(cond)/norm(comb)`` is invariant to a global sign
        flip, so both the CFG and cond-only paths match the reference pipeline.
        """
        return -self.transformer(**kwargs)[0]

    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(
            text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        num_images_per_prompt: int = 1,
        system_prompt: str | None = None,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        prompt = [system_prompt + " <Prompt Start> " + p for p in prompt]

        prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
            prompt=prompt, device=device, max_sequence_length=max_sequence_length
        )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds, prompt_attention_mask

    def _collect_input_images(self, req: DiffusionRequestBatch) -> list[Image.Image | None]:
        """One input image per request, or ``None`` where the request has none.

        Requests are plain strings (text-to-image) or dicts that may carry
        ``multi_modal_data["image"]``. SDEdit conditions on exactly one image, so
        a multi-image request is rejected rather than silently truncated.
        """
        images: list[Image.Image | None] = []
        for prompt in req.prompts:
            raw = None if isinstance(prompt, str) else (prompt.get("multi_modal_data") or {}).get("image")
            if raw is None:
                images.append(None)
                continue
            if isinstance(raw, list):
                if len(raw) != 1:
                    raise ValueError(
                        "Lumina-Image-2.0 image-to-image conditions on exactly one input image, "
                        f"but a request supplied {len(raw)}."
                    )
                raw = raw[0]
            image = Image.open(raw) if isinstance(raw, str) else raw
            images.append(image.convert("RGB"))
        return images

    def _derive_i2i_size(self, image: Image.Image) -> tuple[int, int]:
        """Aspect-preserving output size at the model's native pixel budget.

        Used when an image-to-image request gives no explicit height/width: the
        input's aspect ratio is kept while the area is scaled to the resolution
        the transformer was trained at (``sample_size`` latent tokens per side),
        then floored to the patchify alignment.
        """
        src_width, src_height = image.size
        if src_width <= 0 or src_height <= 0:
            raise ValueError(f"Input image must have positive dimensions, got {src_width}x{src_height}.")

        target_pixels = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = math.sqrt(target_pixels / (src_width * src_height))
        return normalize_min_aligned_size(
            round(src_height * scale),
            round(src_width * scale),
            self.vae_scale_factor * 2,
        )

    def get_timesteps(self, num_inference_steps: int, strength: float) -> tuple[torch.Tensor, int]:
        """Slice the schedule down to the tail selected by ``strength``.

        ``set_begin_index`` is what makes the slice usable: both ``scale_noise``
        and ``step`` look up sigmas by step index, so the scheduler has to start
        counting at the offset the returned timesteps start at.
        """
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
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
        # VAE applies 8x compression but latent height/width must be divisible by
        # 2 to allow patchifying, hence the (vae_scale_factor * 2) factor.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if image is not None:
            return self._prepare_image_latents(image, shape, timestep, dtype, device, generator)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def _prepare_image_latents(
        self,
        image: torch.Tensor,
        shape: tuple[int, int, int, int],
        timestep: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        """VAE-encode the preprocessed images and re-noise them at ``timestep``.

        ``image`` holds one row per request; each row is expanded to that
        request's ``num_outputs_per_prompt`` latents with ``repeat_interleave``,
        matching the request-contiguous layout that ``encode_prompt`` and
        ``split_diffusion_output_by_request`` both assume.
        """
        batch_size = shape[0]
        image = image.to(device=device, dtype=dtype)
        num_input_images = image.shape[0]
        if num_input_images == 0 or batch_size % num_input_images != 0:
            raise ValueError(f"Cannot expand {num_input_images} input image(s) to an effective batch of {batch_size}.")
        outputs_per_image = batch_size // num_input_images

        if isinstance(generator, list):
            # Each request's first generator seeds its VAE sample, so the sample
            # is shared by that request's outputs and they diverge only through
            # the noise injected below.
            image_latents = torch.cat(
                [
                    self.vae.encode(image[idx : idx + 1]).latent_dist.sample(generator[idx * outputs_per_image])
                    for idx in range(num_input_images)
                ],
                dim=0,
            )
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(generator)

        # Exact inverse of the decode-side denormalization in ``forward``.
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = image_latents.repeat_interleave(outputs_per_image, dim=0)
        if tuple(image_latents.shape) != tuple(shape):
            raise ValueError(f"Encoded image latents have shape {tuple(image_latents.shape)}, expected {tuple(shape)}.")

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return self.scheduler.scale_noise(image_latents, timestep, noise)

    def _resolve_batch_params(
        self,
        req: DiffusionRequestBatch,
        *,
        images: list[Image.Image | None],
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        strength: float,
    ) -> tuple[int, int, int, float, int, float | None, bool]:
        """Resolve the shared sampling params for a fused request batch.

        Requests fused into one denoise pass must share the tensor-shaping and
        loop-critical params (``height``/``width``/``num_inference_steps``) and
        the CFG ``guidance_scale`` — otherwise a single batched latent tensor and
        a single scheduler timestep schedule cannot represent them. We resolve
        each value from the first request and reject any request that disagrees,
        rather than silently generating wrong images for the others.

        ``strength`` and whether a request carries an input image join that
        contract for the same reason: both pick which slice of the schedule is
        denoised. When an image-to-image request gives no explicit size, the size
        is derived from its input image, so batching two images of different
        aspect ratios is rejected here instead of silently resizing one of them.
        """
        sampling_list = req.sampling_params_list
        resolved = [
            self._resolve_request_params(
                sp,
                image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                strength=strength,
            )
            for sp, image in zip(sampling_list, images, strict=True)
        ]

        common = resolved[0]
        for idx, candidate in enumerate(resolved[1:], start=1):
            if candidate != common:
                raise ValueError(
                    "Batched Lumina-Image-2.0 requests must share height, width, num_inference_steps, "
                    "guidance_scale, num_outputs_per_prompt, strength, and input-image presence to be "
                    f"fused into one denoise pass; request 0 has {common} but request {idx} has {candidate}."
                )
        return common

    def _resolve_request_params(
        self,
        sp,
        image: Image.Image | None,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        strength: float,
    ) -> tuple[int, int, int, float, int, float | None, bool]:
        if image is not None and sp.height is None and sp.width is None:
            default_height, default_width = self._derive_i2i_size(image)
        else:
            default_height, default_width = height, width
        return (
            sp.height or default_height,
            sp.width or default_width,
            sp.num_inference_steps or num_inference_steps,
            sp.guidance_scale if sp.guidance_scale_provided else guidance_scale,
            sp.num_outputs_per_prompt if sp.num_outputs_per_prompt > 0 else num_images_per_prompt,
            # ``strength`` is meaningless without an input image, so it stays out
            # of the homogeneity contract for text-to-image requests.
            None if image is None else (strength if sp.strength is None else sp.strength),
            image is not None,
        )

    def forward(
        self,
        req: DiffusionRequestBatch,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        sigmas: list[float] | None = None,
        strength: float = 0.6,
        cfg_trunc_ratio: float = 1.0,
        cfg_normalization: bool = True,
        max_sequence_length: int = 256,
        output_type: str = "pil",
    ) -> list[DiffusionOutput]:
        # Collect one prompt / negative prompt per request. Requests may be a
        # plain string or a dict carrying its own negative prompt.
        prompts: list[str] = []
        negative_prompts: list[str] = []
        for item in req.prompts:
            if isinstance(item, str):
                prompts.append(item)
                negative_prompts.append(negative_prompt)
            else:
                prompts.append(item.get("prompt") or "")
                negative_prompts.append(item.get("negative_prompt") or negative_prompt)

        input_images = self._collect_input_images(req)
        (
            height,
            width,
            num_inference_steps,
            guidance_scale,
            num_images_per_prompt,
            strength,
            is_img2img,
        ) = self._resolve_batch_params(
            req,
            images=input_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            strength=strength,
        )
        if is_img2img:
            if not 0.0 <= strength <= 1.0:
                raise ValueError(f"The value of strength should be in [0.0, 1.0] but is {strength}")
        else:
            ignored_strengths = [sp.strength for sp in req.sampling_params_list if sp.strength is not None]
            if ignored_strengths:
                logger.warning(
                    "strength (%s) only applies to image-to-image generation and is ignored for this "
                    "text-to-image request.",
                    ignored_strengths[0],
                )
        # Per-request generators (one seed per request, expanded by
        # num_images_per_prompt) so batching is numerically request-local.
        generator = req.collate_request_generators(num_images_per_prompt, generator)

        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}."
            )

        device = self._execution_device
        batch_size = req.num_reqs

        self._guidance_scale = guidance_scale
        self._current_timestep = None

        # 3. Encode input prompts (batched: [B * num_images, seq, dim])
        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompts,
            device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if self.do_classifier_free_guidance:
            negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                negative_prompts,
                device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare timesteps
        latent_channels = self.transformer.config.in_channels
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # Upstream diffusers passes ``latents.shape[1]`` here, which for Lumina's
        # 4D (B, C, H, W) latents is the channel count and not a sequence length.
        # Reading it from the config is that same value, and it lets the
        # image-to-image path resolve the schedule before the latents exist.
        mu = calculate_shift(
            latent_channels,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        # 5. Prepare latents — pure noise for text-to-image, or the re-noised
        # input image for image-to-image.
        if is_img2img:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
            if num_inference_steps < 1:
                raise ValueError(
                    f"After adjusting num_inference_steps by strength {strength}, the number of pipeline "
                    f"steps is {num_inference_steps}, which is < 1 and not appropriate for this pipeline."
                )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            image_tensor = torch.cat(
                [self.image_processor.preprocess(image, height=height, width=width) for image in input_images],
                dim=0,
            )
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                latent_channels,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                image=image_tensor,
                timestep=latent_timestep,
            )
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                latent_channels,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # compute whether apply classifier-free truncation on this timestep
                do_classifier_free_truncation = (i + 1) / num_inference_steps > cfg_trunc_ratio
                # reverse the timestep since Lumina uses t=0 as noise and t=1 as image
                current_timestep = 1 - t / self.scheduler.config.num_train_timesteps
                current_timestep = current_timestep.expand(latents.shape[0])
                # CFG is disabled on late (truncated) timesteps regardless of scale.
                do_true_cfg = self.do_classifier_free_guidance and not do_classifier_free_truncation

                positive_kwargs = {
                    "hidden_states": latents,
                    "timestep": current_timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_attention_mask": prompt_attention_mask,
                    "return_dict": False,
                }
                negative_kwargs = None
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latents,
                        "timestep": current_timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_attention_mask": negative_prompt_attention_mask,
                        "return_dict": False,
                    }

                # predict_noise() negates the velocity; combine_cfg_noise() applies
                # the affine CFG mix and (when cfg_normalize) the norm-rescale that
                # matches Lumina's cfg_normalization.
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=cfg_normalization,
                )
                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg, generator=generator)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            result = DiffusionOutput(output=latents)
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            images = self.vae.decode(latents, return_dict=False)[0]
            result = DiffusionOutput(
                output=images, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
            )

        # Split the fused batch back into one DiffusionOutput per request.
        return split_diffusion_output_by_request(
            result,
            req,
            num_outputs_per_prompt=num_images_per_prompt,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = set()
        transformer_weights = (
            (name.replace("transformer.", "", 1), weight) for name, weight in weights if name.startswith("transformer.")
        )
        loaded_weights |= {f"transformer.{name}" for name in self.transformer.load_weights(transformer_weights)}
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        return loaded_weights
