# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/jd-opensource/JoyAI-Video-Edit
"""JoyAI-Video-Edit: instruction-driven video editing, offline fixed-length.

Upstream ships no ``__call__``: its ``Pipeline`` is a component container and the denoise loop lives in
``serving/joyomni_streaming.py::JoyOmniV2VStreamingSession._denoise_chunk``. This module reassembles
that loop as a request-mode pipeline. The rollout is **chunk-autoregressive** -- one latent frame at a
time, each conditioned on a bounded KV window over previously-generated clean chunks -- so
:meth:`JoyAIVideoEditPipeline.forward` is a loop over chunks rather than a loop over denoise steps.

Five things about the loop are load-bearing and none of them fail loudly:

**Three DiT forwards per chunk, not two.** Two denoise steps (the AR-DMD distillation's whole
schedule) plus a third pass at ``timestep=0`` over the *clean* result whose only purpose is to write
that chunk's KV into the window. Dropping the third produces plausible frames with no temporal
coherence between chunks, because every chunk then attends to an empty history.

**Eviction happens twice per chunk**, around that third forward, with different keep-sets:
before the store, keep only the history this chunk read; after it, keep only what the *next* chunk
will read. Porting one of the two leaks a chunk per step and runs out of memory around chunk 30 in
production while staying numerically correct.

**The source video is encoded through a sliding 9-frame window, not in one pass.** This is a layer
*above* the VAE's own internal chunking (``chunk_size: 48`` in ``vae/config.json``) -- both exist and
they are not the same mechanism. Latent frame ``k`` comes from encoding pixel frames
``[8(k-1) .. 8k]`` and keeping the last latent frame; frame 0 is encoded alone. A single whole-clip
encode would give a *different* (and untrained-against) conditioning signal, because each window
restarts the VAE's temporal cache. It is gated on ``causal`` upstream, which is ``True`` in the
shipped config.

**The condition encoder runs once per request.** The source video reaches the DiT as VAE latents
(``ref_video_latent``), never through the MLLM; re-encoding per chunk costs a 7B forward per latent
frame for an identical result.

**Decode is one whole-tensor pass.** Upstream's streaming decoder prepends a re-encoded *pseudo
latent* per chunk to keep a live stream continuous; that is an approximation of what a single decode
over chunk-aligned latents does exactly (measured bit-identical on chunk-aligned prefixes). Offline,
the exact path is both simpler and better, so the pseudo-latent machinery is deliberately absent.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_condition import (
    JoyAIVideoEditConditionEncoder,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    CHUNK_SIZE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    GLOBAL_SINK_CHUNK,
    KV_CACHE_PRE_ROPE,
    LOCAL_WINDOW_SIZE,
    MAX_CONDITION_TOKENS,
    NUM_INFERENCE_STEPS,
    NUM_TRAIN_TIMESTEPS,
    SCHEDULER_SHIFT,
    VAE_SPATIAL_COMPRESSION,
    VAE_TEMPORAL_COMPRESSION,
    dit_config,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_kv import (
    JoyKVWindow,
    cache_memory_ids_for_read,
    gather_window_temporal_ids,
    get_chunk_windows,
    split_window_temporal_ids,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_transformer import (
    JoyAIVideoEditTransformer3DModel,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_vae import (
    JoyAIVideoEditVAE,
    latent_spatial_shape,
    num_latent_frames,
    validate_pixel_shape,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_vae_compile import (
    channels_last_enabled,
    convert_conv3d_to_channels_last,
    prep_input_for,
    setup_vae_compile,
    vae_compile_enabled,
    warmup,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers.scheduling_flow_match_discrete import (
    FlowMatchDiscreteScheduler,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

logger = logging.getLogger(__name__)

#: Not a model property -- upstream carries no notion of frame rate. Used only to tag the output when
#: the request and the source video are both silent about it.
DEFAULT_JOYAI_VIDEO_EDIT_FPS = 24

DEFAULT_DIT_CHECKPOINT = "dit/joyai_video_edit_dit_0804.pth"
DEFAULT_VAE_SUBFOLDER = "vae"
DEFAULT_CONDITION_ENCODER_SUBFOLDER = "MiMo-VL-7B-RL-2508"

#: Checkpoint keys may be wrapped by a training harness. Stripped longest-first so ``model.module.``
#: does not leave a dangling ``module.``.
_CHECKPOINT_STATE_DICT_KEYS = ("module", "model", "state_dict", "ema_state_dict")
_CHECKPOINT_KEY_PREFIXES = ("model.module.", "module.model.", "transformer.", "module.", "model.")


# --- video I/O ----------------------------------------------------------------------------------
# Shared shape with `skyreels_v3/pipeline_skyreels_v3_v2v.py`, which is where these were factored
# from. The one deliberate divergence is `_normalize_frame_array` always returning `[T, H, W, 3]`
# (skyreels' returns a bare array that its own callers then subscript with `[0]`, indexing the first
# *frame* rather than unpacking a tuple).


def _autocast(device: torch.device | str, dtype: torch.dtype):
    """Upstream's ``_autocast_ctx`` (``joyomni_streaming.py:25``), same guards.

    Enabled only when the dtype is not fp32 and only on ``cuda``/``cpu``, because those are the two
    device types ``torch.autocast`` accepts here; anything else gets a no-op rather than a raise.

    Whether this changes a single bit is an *empirical* question, not an obvious one: with bf16 module
    weights fed bf16 activations, every op autocast would cast is already running in bf16, so the
    context can be a complete no-op. It is here for faithfulness to upstream's structure -- an op that
    later arrives holding fp32 (a norm kept in fp32, an fp32 scheduler state reaching a linear) runs
    bf16 under upstream and would run fp32 under a port without this, and that asymmetry is invisible
    until it is measured. See ``recipes/JD/JoyAI-Video-Edit.md`` for the measurement.
    """
    device_type = torch.device(device).type
    if device_type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32)


def _load_video_from_path_or_url(video: str) -> tuple[np.ndarray, float | None]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("JoyAI-Video-Edit requires PyAV (`av`) to load video paths or URLs.") from exc

    source: str | io.BytesIO
    if video.startswith("data:video"):
        try:
            _, payload = video.split(",", 1)
            source = io.BytesIO(base64.b64decode(payload))
        except ValueError as exc:
            raise ValueError("Invalid data URL video input.") from exc
    else:
        source = video

    with av.open(source) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise ValueError("Input video contains no decodable frames.")
    return np.stack(frames, axis=0), fps


def _normalize_frame_array(array: np.ndarray) -> np.ndarray:
    """Coerce any frame layout to contiguous ``uint8`` ``[T, H, W, 3]``."""
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 4 and array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 3, 0))
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Video frames must have shape [T, H, W, C], got {array.shape}.")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Video frames must have 1, 3, or 4 channels, got shape {array.shape}.")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    array = array.astype(np.float32)
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if min_value >= -1.0 and max_value <= 1.0:
        array = (array + 1.0) * 127.5 if min_value < 0.0 else array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).round().astype(np.uint8))


def _coerce_frames_to_uint8(frames: Any) -> tuple[np.ndarray, float | None]:
    """Normalize a supported video input into ``uint8`` RGB ``[T, H, W, 3]`` plus its source fps."""
    if isinstance(frames, str):
        return _load_video_from_path_or_url(frames)

    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError("JoyAI-Video-Edit supports a single input video per request.")
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(f"Unsupported video tensor shape {tuple(tensor.shape)}.")
        if tensor.shape[-1] not in (1, 3, 4) and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        return _normalize_frame_array(tensor.numpy()), None

    if isinstance(frames, np.ndarray):
        return _normalize_frame_array(frames), None

    if isinstance(frames, Sequence) and not isinstance(frames, (bytes, bytearray)):
        if not frames:
            raise ValueError("Input video contains no frames.")
        frame_arrays = []
        for frame in frames:
            if isinstance(frame, PIL.Image.Image):
                frame_arrays.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
            elif isinstance(frame, torch.Tensor):
                frame_tensor = frame.detach().cpu()
                if frame_tensor.ndim == 3 and frame_tensor.shape[0] in (1, 3, 4):
                    frame_tensor = frame_tensor.permute(1, 2, 0)
                frame_arrays.append(_normalize_frame_array(frame_tensor.numpy())[0])
            elif isinstance(frame, np.ndarray):
                frame_arrays.append(_normalize_frame_array(frame)[0])
            else:
                raise TypeError(f"Unsupported video frame type {frame.__class__}.")
        return np.stack(frame_arrays, axis=0), None

    raise TypeError(
        "JoyAI-Video-Edit video input must be a path/URL, numpy array, torch tensor, or a sequence "
        f"of frames, got {frames.__class__}."
    )


def _resize_video_uint8(frames: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize to the request geometry with PIL's bicubic filter, matching upstream exactly.

    ``joyomni_streaming.py::_resize_frame`` resizes every source frame through ``PIL.Image.BICUBIC``,
    and that specific filter is a fidelity requirement rather than a stylistic preference. PIL's
    bicubic is Catmull-Rom (``a = -0.5``) while ``F.interpolate(mode="bicubic")`` uses ``a = -0.75``,
    which is sharper; feeding the DiT conditioning pixels that differ from the ones upstream computes
    shifts the whole clip by a constant amount that is invisible as a defect -- the edit still looks
    plausible -- and measures as a *flat* few-dB agreement with upstream from frame 0 onward.
    ``antialias=True`` does not close the gap either, because torch ignores it when upscaling, which
    the showcase geometry (832x480 -> 1248x720) is. Nor does using PIL cost the aliasing protection
    that motivated antialias in the first place: PIL scales its filter support by the reduction
    factor, so the 1920x1080 -> 1248x720 downscale is still filtered rather than point-sampled.
    """
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    resampling = getattr(PIL.Image, "Resampling", PIL.Image).BICUBIC
    resized = [
        np.asarray(PIL.Image.fromarray(frame).resize((width, height), resampling), dtype=np.uint8) for frame in frames
    ]
    return np.stack(resized, axis=0)


def _video_uint8_to_vae_tensor(frames: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """``[T, H, W, 3]`` uint8 -> ``[1, 3, T, H, W]`` float on ``[-1, 1]``, the range the VAE expects."""
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float()
    return (tensor / 127.5 - 1.0).to(device)


def _resolve_input_video(prompt: OmniTextPrompt | dict[str, Any], extra_args: dict[str, Any]) -> Any:
    multi_modal_data = prompt.get("multi_modal_data") or {}
    for source, key in (
        (multi_modal_data, "video"),
        (multi_modal_data, "input_video"),
        (multi_modal_data, "cond_video"),
        (extra_args, "video_path"),
        (extra_args, "input_video"),
    ):
        if key in source and source[key] is not None:
            return source[key]
    raise ValueError(
        "JoyAI-Video-Edit requires a source video to edit. Use "
        "`multi_modal_data={'video': ...}` or pass `video_path` in extra_args / extra_body."
    )


# --- request geometry ---------------------------------------------------------------------------


def resolve_num_frames(available_frames: int, requested_frames: int | None) -> int:
    """Largest usable frame count: ``1 + 8n``, capped by what the source actually has.

    A request for a frame count the VAE cannot represent is rejected rather than rounded, because
    rounding a *requested* length silently returns a video of the wrong duration. An unconstrained
    request is trimmed down instead -- the source video's length is an input, not an instruction.

    ``num_frames <= 1`` counts as unconstrained. ``OmniDiffusionSamplingParams.num_frames`` defaults
    to ``1`` ("Default for image models"), so a caller who never mentions frame count is
    indistinguishable from one asking for a single frame -- and honouring the literal 1 would return a
    one-frame clip from every unparameterised video-edit request. The reinterpretation is logged
    because it is a reinterpretation; there is deliberately no way to request a 1-frame edit, which
    is image editing wearing a video model's cost.
    """
    if available_frames < 1:
        raise ValueError("The source video contains no frames.")

    if requested_frames is not None and requested_frames > 1:
        if requested_frames > available_frames:
            raise ValueError(
                f"`num_frames={requested_frames}` exceeds the {available_frames} frames the source "
                f"video provides; JoyAI-Video-Edit edits an existing clip and cannot extend it."
            )
        if requested_frames % VAE_TEMPORAL_COMPRESSION != 1:
            lower = 1 + VAE_TEMPORAL_COMPRESSION * ((requested_frames - 1) // VAE_TEMPORAL_COMPRESSION)
            raise ValueError(
                f"`num_frames` must be 1 + {VAE_TEMPORAL_COMPRESSION}n (got {requested_frames}); "
                f"the nearest valid values are {lower} and {lower + VAE_TEMPORAL_COMPRESSION}."
            )
        return requested_frames

    usable = 1 + VAE_TEMPORAL_COMPRESSION * ((available_frames - 1) // VAE_TEMPORAL_COMPRESSION)
    if requested_frames is not None and requested_frames == 1 and usable > 1:
        logger.info(
            "`num_frames` was left at its default of 1; editing the longest valid prefix of the "
            "source instead (%d of %d frames). Pass an explicit 1 + %dn value to choose a length.",
            usable,
            available_frames,
            VAE_TEMPORAL_COMPRESSION,
        )
    return usable


def resolve_generators(
    sampling_params: OmniDiffusionSamplingParams, device: torch.device | str
) -> tuple[torch.Generator | None, torch.Generator | None]:
    """Split one request seed into ``(noise_generator, encode_generator)`` -- two independent streams.

    Upstream runs the VAE source encode and the denoise rollout off *different* random streams: the
    posterior sample is drawn from the global RNG (``latent_dist.sample()`` with no generator,
    ``models/pipeline.py:315``) while per-chunk noise comes from a separately seeded
    ``torch.Generator`` (``joyomni_streaming.py:385``). Handing one generator to both -- the obvious
    thing to write, and what this port originally did -- makes the encode advance the Philox stream by
    one large draw per latent frame (15 on a 113-frame clip) before the rollout draws at all, so every
    chunk is denoised from noise upstream never used. Nothing about the result looks broken: it is a
    plausible edit of the same source, and it measured as a flat ~18 dB agreement with upstream from
    frame 0 to the end, which is why this is a named helper with a test rather than two lines inline.

    Both streams are seeded from the same value rather than letting the encode fall through to the
    global RNG as upstream does. Upstream can afford global RNG because a session owns its process; a
    serving engine cannot -- concurrent requests would interleave draws and our own output would stop
    being reproducible from its seed.
    """
    generator = sampling_params.generator
    if isinstance(generator, list):
        # A batch of generators cannot apply: `forward` rejects batch > 1 before reaching here.
        generator = generator[0] if generator else None
    if generator is None and sampling_params.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(sampling_params.seed))
    if generator is None:
        # Unseeded: both paths draw from the global RNG, which is upstream's behaviour for the encode
        # and is what the caller asked for by declining to pin a seed.
        return None, None
    return generator, torch.Generator(device=device).manual_seed(generator.initial_seed())


def resolve_resolution(source_height: int, source_width: int, height: int | None, width: int | None) -> tuple[int, int]:
    """Request resolution, defaulting to the shipped 720x1248 and rejecting non-multiples of 24.

    The divisibility check exists because the VAE's ``Stem`` downsamples by 3/2 *before* the encoder's
    16x: a width divisible by 16 but not 24 -- 1280 being the obvious one to reach for -- bypasses the
    ``Stem`` and yields a wrong-resolution latent with no error anywhere. Silently snapping to a
    nearby valid size would return a video whose dimensions the caller did not ask for, so this
    raises and names the constraint instead.
    """
    del source_height, source_width  # kept in the signature: bucket selection is a plausible future
    resolved_height = int(height or DEFAULT_HEIGHT)
    resolved_width = int(width or DEFAULT_WIDTH)
    bad = [
        f"{name}={size}"
        for name, size in (("height", resolved_height), ("width", resolved_width))
        if size % VAE_SPATIAL_COMPRESSION != 0
    ]
    if bad:
        raise ValueError(
            f"`height` and `width` must be divisible by {VAE_SPATIAL_COMPRESSION} (got {', '.join(bad)}). "
            f"Note this is stricter than the 16 the VAE's encoder alone implies -- the Stem adds a "
            f"factor of 3/2 in front of it. 1280 is the common trap; the reference default is "
            f"{DEFAULT_WIDTH}."
        )
    return resolved_height, resolved_width


# --- registry hooks -----------------------------------------------------------------------------


def get_joyai_video_edit_pre_process_func(od_config: OmniDiffusionConfig):
    """Normalize the source video and pin the request geometry before the worker sees it.

    Doing the decode here keeps a multi-hundred-megabyte ndarray out of the engine's request queue
    and, more importantly, turns an unusable geometry into an error at admission rather than after
    the 30 GiB DiT has been scheduled. :meth:`JoyAIVideoEditPipeline.forward` repeats every check, so
    the pipeline is equally usable without a pre-process pass.
    """
    del od_config

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        extra_args = request.sampling_params.extra_args or {}

        frames, source_fps = _coerce_frames_to_uint8(_resolve_input_video(prompt, extra_args))
        num_frames = resolve_num_frames(frames.shape[0], request.sampling_params.num_frames)
        frames = frames[:num_frames]

        height, width = resolve_resolution(
            frames.shape[1],
            frames.shape[2],
            request.sampling_params.height,
            request.sampling_params.width,
        )
        request.sampling_params.height = height
        request.sampling_params.width = width
        request.sampling_params.num_frames = num_frames

        prompt["multi_modal_data"] = dict(prompt.get("multi_modal_data") or {})
        prompt["multi_modal_data"]["video"] = _resize_video_uint8(frames, height=height, width=width)
        prompt.setdefault("additional_information", {})["source_video_fps"] = source_fps
        request.prompt = prompt
        return request

    return pre_process_func


def get_joyai_video_edit_post_process_func(od_config: OmniDiffusionConfig):
    """Convert the pipeline's ``(video, fps)`` into the standard video payload envelope."""
    del od_config

    def post_process_func(
        output: np.ndarray | torch.Tensor | tuple[np.ndarray | torch.Tensor, float],
        output_type: str = "np",
        sampling_params: OmniDiffusionSamplingParams | None = None,
    ):
        if sampling_params is not None and getattr(sampling_params, "output_type", None):
            output_type = sampling_params.output_type

        fps: float = DEFAULT_JOYAI_VIDEO_EDIT_FPS
        video = output
        if isinstance(output, tuple) and len(output) == 2:
            video, fps = output
        if output_type == "latent":
            return video

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu()
            if video.ndim == 5:
                video = video[0]
            if video.ndim == 4 and video.shape[0] in (1, 3, 4) and video.shape[-1] not in (1, 3, 4):
                video = video.permute(1, 2, 3, 0)
            video = video.numpy()

        if isinstance(video, np.ndarray):
            video = _normalize_frame_array(video)
            if output_type == "pil":
                video = [PIL.Image.fromarray(frame) for frame in video]
            elif output_type in {"pt", "tensor"}:
                video = torch.from_numpy(video).permute(0, 3, 1, 2)

        return {"payload": {"video": video}, "metadata": {"video": {"fps": fps}}}

    return post_process_func


# --- checkpoint ---------------------------------------------------------------------------------


def normalize_dit_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Unwrap a training-harness checkpoint into a flat, unprefixed state dict.

    The shipped ``.pth`` happens to be a bare ``OrderedDict`` of 894 tensors, but the wrappers
    handled here are what the same training code emits with checkpointing or DDP enabled, and the
    cost of tolerating them is one pass over the keys. A prefix left in place would surface as a
    ``strict=True`` failure listing 894 unexpected keys, which is loud but unhelpful.
    """
    for key in _CHECKPOINT_STATE_DICT_KEYS:
        if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break

    for prefix in _CHECKPOINT_KEY_PREFIXES:
        if all(name.startswith(prefix) for name in state_dict):
            return {name[len(prefix) :]: tensor for name, tensor in state_dict.items()}
    return dict(state_dict)


class JoyAIVideoEditPipeline(
    nn.Module,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Offline instruction-driven video editing, one latent frame per rollout step.

    ``CFGParallelMixin`` is deliberately absent rather than merely unused: this model was distilled
    without classifier-free guidance -- no negative prompt, no guidance scale, a single ``"cond"``
    cache scope -- so there is no second branch for CFG parallelism to place on another rank.
    """

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["condition_encoder.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    # Generic dummy warmup would have to invent a source video; there is no meaningful request
    # without one, and a synthetic clip that violates the 1 + 8n / 24-divisible geometry just fails.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        del prefix
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.target_dtype = getattr(od_config, "dtype", torch.bfloat16) or torch.bfloat16

        model_root = Path(od_config.model)
        if not model_root.is_dir():
            raise ValueError(
                f"JoyAI-Video-Edit needs `--model` to be a local directory (got {od_config.model!r}). "
                f"Run `examples/offline_inference/video_to_video/download_joyai_video_edit.py` first."
            )
        self.model_root = model_root
        settings = self._read_settings(model_root, od_config)

        self.chunk_size = int(settings.get("chunk_size", CHUNK_SIZE))
        self.local_window_size = int(settings.get("local_window_size", LOCAL_WINDOW_SIZE))
        self.global_sink_chunk = bool(settings.get("global_sink_chunk", GLOBAL_SINK_CHUNK))
        self.kv_cache_pre_rope = bool(settings.get("kv_cache_pre_rope", KV_CACHE_PRE_ROPE))
        self.num_inference_steps = int(settings.get("num_inference_steps", NUM_INFERENCE_STEPS))
        self.default_height = int(settings.get("default_height", DEFAULT_HEIGHT))
        self.default_width = int(settings.get("default_width", DEFAULT_WIDTH))
        self.max_condition_tokens = int(settings.get("max_condition_tokens", MAX_CONDITION_TOKENS))
        if self.chunk_size != CHUNK_SIZE:
            # The sliding source-encode window is `1 + chunk_size * 8` frames wide and the KV window
            # is sized in chunks, so a different chunk size is a different rollout the distilled
            # 2-step schedule was not trained for -- not a batching knob.
            raise ValueError(
                f"JoyAI-Video-Edit is distilled for `chunk_size={CHUNK_SIZE}`, got {self.chunk_size}. "
                f"Other values change the autoregressive rollout, not just its granularity."
            )

        # Geometry keys whose VAE compile has already been warmed; see `_warmup_vae_compile`.
        self._vae_warmed: set[tuple[int, int, int]] = set()
        self.vae = self._build_vae(model_root / settings.get("vae_path", DEFAULT_VAE_SUBFOLDER))
        self.transformer = self._build_transformer(model_root / settings.get("dit_checkpoint", DEFAULT_DIT_CHECKPOINT))
        self.condition_encoder = JoyAIVideoEditConditionEncoder.from_pretrained(
            str(model_root / settings.get("condition_encoder_path", DEFAULT_CONDITION_ENCODER_SUBFOLDER)),
            dtype=self.target_dtype,
            device=self.device,
            max_sequence_length=self.max_condition_tokens,
        )

        self.scheduler = FlowMatchDiscreteScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=float(settings.get("scheduler_shift", SCHEDULER_SHIFT)),
            reverse=True,
            solver="euler",
        )

        # Only these three: the default target list names `text_encoder.forward`/`tokenizer.forward`,
        # neither of which is an attribute of this pipeline.
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["vae.encode", "vae.decode", "diffuse"],
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    # -- construction helpers -------------------------------------------------------------------
    @staticmethod
    def _read_settings(model_root: Path, od_config: OmniDiffusionConfig) -> dict[str, Any]:
        """``model_index.json``, with ``od_config.model_config`` (i.e. ``--model-config``) on top."""
        index_path = model_root / "model_index.json"
        settings: dict[str, Any] = {}
        if index_path.is_file():
            settings = {k: v for k, v in json.loads(index_path.read_text()).items() if not k.startswith("_")}
        override = getattr(od_config, "model_config", None)
        if isinstance(override, dict):
            settings.update(override)
        return settings

    def _build_vae(self, vae_dir: Path) -> JoyAIVideoEditVAE:
        config_path = vae_dir / "config.json"
        weights_path = vae_dir / "diffusion_pytorch_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Expected a diffusers-format VAE in {vae_dir}.")
        config = {k: v for k, v in json.loads(config_path.read_text()).items() if not k.startswith("_")}
        vae = JoyAIVideoEditVAE(**config)
        vae.load_state_dict(load_file(str(weights_path)), strict=True)
        # `latents_mean`/`latents_std` are plain lists, not buffers, so `.to()` does not move them.
        self.latents_mean = torch.tensor(config["latents_mean"]).view(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(config["latents_std"]).view(1, -1, 1, 1, 1)
        vae = vae.to(device=self.device, dtype=self.target_dtype).eval().requires_grad_(False)
        # After `.to()`, so the `channels_last_3d` weight conversion applies to the device tensors the
        # convs will actually read; doing it before would be undone by the copy.
        if vae_compile_enabled():
            setup_vae_compile(vae)
        elif channels_last_enabled():
            convert_conv3d_to_channels_last(vae)
        return vae

    def _warmup_vae_compile(self, *, height: int, width: int, num_frames: int) -> None:
        """Pay the autotune cost here rather than inside the first chunk's latency.

        Keyed on geometry and remembered, because a served pipeline handles many requests and each
        warmed shape costs 8--13 s -- repeating that per request would cost far more than the compile
        saves. Encode is ``dynamic=False``, so it gets both window lengths `encode_source_windows`
        actually uses (1 for chunk 0, 9 thereafter). No ``latent_shapes`` are passed: ``_decode`` is not
        compiled, because its compiled steady state measured *slower* than eager (see
        `joyai_video_edit_vae_compile`), so warming it would spend a whole-clip decode to pre-select
        eager cuDNN algorithms.
        """
        if not vae_compile_enabled():
            return
        key = (height, width, num_frames)
        if key in self._vae_warmed:
            return
        self._vae_warmed.add(key)
        # Mirrors `encode_source_windows` exactly rather than hardcoding 1 and 9: a warmup on shapes
        # the rollout does not hit compiles a graph nothing calls and leaves the real call cold.
        pixel_shapes = [(1, height, width)]
        if num_frames > 1:
            pixel_shapes.append((self.chunk_size * VAE_TEMPORAL_COMPRESSION + 1, height, width))
        warmup(
            self.vae,
            device=self.device,
            dtype=self.target_dtype,
            pixel_shapes=pixel_shapes,
            latent_channels=self.vae.latent_channels,
        )

    def _build_transformer(self, checkpoint: Path) -> JoyAIVideoEditTransformer3DModel:
        """Bypass ``DiffusersPipelineLoader``: this is one 30 GiB ``.pth``, not a diffusers subfolder.

        ``mmap=True`` keeps ``torch.load`` from materialising all 30 GiB in host RAM, and
        ``assign=True`` hands those mapped tensors straight to the parameters instead of copying into
        the separately-allocated ones -- so the peak is one copy, not three. Constructing on ``meta``
        means the allocation that ``assign`` replaces never happened at all.
        """
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DiT checkpoint {checkpoint} does not exist.")
        model = JoyAIVideoEditTransformer3DModel(**dit_config(), device="meta", dtype=self.target_dtype)
        state_dict = normalize_dit_state_dict(torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu"))
        result = model.load_state_dict(state_dict, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:  # pragma: no cover - strict=True already raises
            raise RuntimeError(f"Unexpected DiT load result: {result}")
        del state_dict
        return model.to(device=self.device, dtype=self.target_dtype).eval().requires_grad_(False)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Report which parameters are materialized; every component was loaded in :meth:`__init__`.

        ``DiffusersPipelineLoader.load_weights`` calls this unconditionally -- *not* only when
        ``weights_sources`` is set -- and then requires the returned set to cover every
        ``named_parameters()`` entry. Returning an empty set therefore reads as "nothing loaded" and
        fails the load with all ~16.2B parameter names in the message.

        So rather than returning ``None`` to opt out of the check, return the names that actually hold
        data. The DiT is built on ``device="meta"`` and materialized by ``assign=True``, so a parameter
        left on meta is a real failure mode of this loading path, and this is the only place that
        catches it.
        """
        consumed = {name for name, _ in weights}
        if consumed:
            raise RuntimeError(
                f"JoyAI-Video-Edit loads its weights eagerly and declares no `weights_sources`, but "
                f"{len(consumed)} weights were handed to `load_weights`."
            )
        return {name for name, param in self.named_parameters() if not param.is_meta}

    # -- latents --------------------------------------------------------------------------------
    def normalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(device=latent.device, dtype=latent.dtype)
        std = self.latents_std.to(device=latent.device, dtype=latent.dtype)
        return (latent - mean) / std

    def denormalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(device=latent.device, dtype=latent.dtype)
        std = self.latents_std.to(device=latent.device, dtype=latent.dtype)
        return latent * std + mean

    def encode_source_windows(
        self,
        pixels: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Per-chunk source latents from a sliding 9-frame window over the source clip.

        Latent frame ``k`` is the last latent frame of an encode over pixel frames ``[8(k-1) .. 8k]``
        (9 frames); frame 0 is an encode over pixel frame 0 alone. Each window restarts the VAE's
        temporal cache, so this is *not* the same tensor a single whole-clip encode produces -- the
        DiT was conditioned on these window-local latents and it is the harder-to-notice of the two
        mistakes, since a whole-clip encode gives the right shape and plausible values.
        """
        total_frames = pixels.shape[2]
        validate_pixel_shape(total_frames, pixels.shape[3], pixels.shape[4])
        window_pixels = self.chunk_size * VAE_TEMPORAL_COMPRESSION
        latents = []
        # Upstream runs the whole sliding-window encode -- sampling and normalization included --
        # inside one `vae_dtype` autocast (`joyomni_streaming.py:1318` wrapping `_sample_vae_latents`),
        # so the context goes around the loop rather than around each `encode`.
        with _autocast(self.device, self.target_dtype):
            for k in range(num_latent_frames(total_frames)):
                window = (
                    pixels[:, :, :1]
                    if k == 0
                    else pixels[:, :, k * window_pixels - window_pixels : k * window_pixels + 1]
                )
                # `prep_input_for` matches the weights' memory format when the layout knob is on, and
                # is the identity otherwise -- upstream does the same at `joyomni_streaming.py:1314`.
                window = prep_input_for(self.vae, window.to(device=self.device, dtype=self.target_dtype))
                encoded = self.vae.encode(window).latent_dist.sample(generator=generator)
                latents.append(self.normalize_latents(encoded)[:, :, -1:])
        return torch.cat(latents, dim=2)

    # -- rollout --------------------------------------------------------------------------------
    def diffuse(
        self,
        source_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        *,
        num_inference_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Chunk-autoregressive rollout over ``source_latents``, returning normalized clean latents.

        One iteration per latent frame, and within it exactly three DiT forwards: two denoise steps
        and a KV-store pass over the resulting clean chunk. See the module docstring for why the
        third one and both eviction calls are not optional.
        """
        batch_size, _, total_latent_frames, latent_h, latent_w = source_latents.shape
        if batch_size != 1:
            raise ValueError(f"The rollout is single-request; got batch {batch_size}.")

        kv_window = JoyKVWindow()
        kv_window.reset()
        windows = get_chunk_windows(
            total_latent_frames=total_latent_frames,
            chunk_size=self.chunk_size,
            window_size=self.local_window_size,
            global_sink_chunk=self.global_sink_chunk,
        )
        noise_shape = (batch_size, self.transformer.in_channels, self.chunk_size, latent_h, latent_w)
        clean_chunks = []

        for window in self.progress_bar(windows):
            chunk_idx = window["chunk_idx"]
            selected_chunk_ids = window["selected_chunk_ids"]
            history_chunk_ids = selected_chunk_ids[:-1]
            active_chunk_id = selected_chunk_ids[-1]

            # Positions are renumbered contiguously per window, so `cached` are the history's
            # positions *in this window*, not the frames' original indices.
            window_ids = gather_window_temporal_ids(
                selected_chunk_ids,
                self.chunk_size,
                # Upstream computes the window as if the rollout ended here, which is what keeps the
                # final (possibly short) chunk from changing earlier chunks' position tables.
                chunk_idx + 1,
                self.device,
            )
            cached_ids, current_ids = split_window_temporal_ids(window_ids, self.chunk_size)
            current_temporal_ids = current_ids.unsqueeze(0).expand(batch_size, -1)
            cached_temporal_ids = None if cached_ids is None else cached_ids.unsqueeze(0).expand(batch_size, -1)
            read_ids = cache_memory_ids_for_read(history_chunk_ids, has_ref_image=False)

            source_chunk = source_latents[:, :, chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]
            latents = randn_tensor(noise_shape, generator=generator, device=self.device, dtype=self.target_dtype)

            # Re-called per chunk: `step()` advances `_step_index`, and every chunk restarts the
            # 2-step schedule from sigma 1.
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            for timestep in self.scheduler.timesteps:
                # `scheduler.step` returns fp32 and the running state is deliberately left there --
                # accumulating the Euler update in fp32 rather than round-tripping through bf16 each
                # step -- so the cast to the model dtype happens per use, not once.
                #
                # Upstream re-enters the autocast per timestep and closes it around `scheduler.step`
                # too (`joyomni_streaming.py:777-836`), so the context is inside this loop rather than
                # around it. The third forward below is deliberately **outside**: upstream's
                # `_store_clean_chunk_kv_cache` call sits after the `with` block ends
                # (`joyomni_streaming.py:834`), and a port that hoisted one context around all three
                # would run that pass under different rules than upstream does.
                with _autocast(self.device, self.target_dtype):
                    noise_pred = self.transformer(
                        hidden_states=latents.to(self.target_dtype),
                        timestep=timestep.repeat(batch_size),
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        ref_video_latent=source_chunk,
                        current_temporal_ids=current_temporal_ids,
                        cached_temporal_ids=cached_temporal_ids,
                        kv_window=kv_window,
                        kv_cache_mode="reuse",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=active_chunk_id,
                        kv_cache_selected_chunk_ids=read_ids,
                        kv_cache_pre_rope=self.kv_cache_pre_rope,
                    )[0]
                    latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

            clean_chunk = latents.to(dtype=self.target_dtype)
            clean_chunks.append(clean_chunk)

            # Eviction #1 of 2: the active slot is about to be overwritten with the *clean* chunk, so
            # only the history it read needs to survive.
            kv_window.evict_before_store(history_chunk_ids, has_ref_image=False)
            self.transformer(
                hidden_states=clean_chunk,
                timestep=torch.zeros((batch_size,), device=self.device, dtype=clean_chunk.dtype),
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                # No source chunk and no history: this pass exists only to produce keys and values
                # for the chunk's own clean tokens, and reading anything would put foreign tokens in
                # them.
                ref_video_latent=None,
                current_temporal_ids=current_temporal_ids,
                cached_temporal_ids=None,
                kv_window=kv_window,
                kv_cache_mode="store",
                kv_cache_scope="cond",
                kv_cache_chunk_id=active_chunk_id,
                kv_cache_selected_chunk_ids=[],
                kv_cache_pre_rope=self.kv_cache_pre_rope,
                skip_text_stream=True,
            )
            # Eviction #2 of 2: bounds residency at `local_window_size`. Without it the store above
            # grows the cache by one chunk every step.
            kv_window.evict_after_store(
                chunk_idx,
                chunk_size=self.chunk_size,
                window_size=self.local_window_size,
                global_sink_chunk=self.global_sink_chunk,
                has_ref_image=False,
            )

        kv_window.reset()
        return torch.cat(clean_chunks, dim=2)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents -> ``uint8`` ``[T, H, W, 3]``, in one whole-tensor decode.

        The VAE is chunk-causal, so a decode over the full chunk-aligned latent tensor is exact --
        it is the streaming pseudo-latent path that approximates *this*, not the reverse.
        """
        # Denormalization sits *outside* the autocast, matching upstream: it denormalizes at
        # `joyomni_streaming.py:1365` and only opens the context at :1383, around the `decode` call.
        latents = self.denormalize_latents(latents)
        with _autocast(self.device, self.target_dtype):
            # Matches upstream's `prep_input` at `joyomni_streaming.py:1381`; identity unless the
            # `channels_last_3d` layout is enabled.
            pixels = self.vae.decode(prep_input_for(self.vae, latents), return_dict=False)[0]
        pixels = pixels[0].float().clamp(-1.0, 1.0).permute(1, 2, 3, 0)
        return ((pixels + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

    # -- entry point ----------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            # Not merely unsupported: the DiT shares one rotary table and one KV scope across the
            # batch, and upstream indexes `current_temporal_ids[0]`, so samples 1..N would be
            # silently generated against sample 0's positions. `max_num_seqs: 1` in the deploy YAML
            # is the same guard one layer up, and is not sufficient on its own.
            raise ValueError(f"JoyAI-Video-Edit generates one video per request, got {len(request.prompts)} prompts.")

        raw_prompt = request.prompts[0]
        prompt_data = OmniTextPrompt(prompt=raw_prompt) if isinstance(raw_prompt, str) else raw_prompt
        instruction = prompt_data.get("prompt") or "" if not isinstance(raw_prompt, str) else raw_prompt
        if not instruction:
            raise ValueError("JoyAI-Video-Edit requires an edit instruction as the prompt.")

        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args or {}
        frames, source_fps = _coerce_frames_to_uint8(_resolve_input_video(prompt_data, extra_args))

        additional_info = prompt_data.get("additional_information") or {}
        if source_fps is None and isinstance(additional_info, dict):
            stored = additional_info.get("source_video_fps")
            if isinstance(stored, (int, float)) and not isinstance(stored, bool):
                source_fps = float(stored)
        # Frame rate is an I/O concern only -- no upstream source file has any notion of it, so an
        # explicit request wins, then whatever the source clip declared, then a plain default.
        fps = float(sampling_params.fps or extra_args.get("fps") or source_fps or DEFAULT_JOYAI_VIDEO_EDIT_FPS)

        num_frames = resolve_num_frames(frames.shape[0], sampling_params.num_frames)
        frames = frames[:num_frames]
        height, width = resolve_resolution(
            frames.shape[1], frames.shape[2], sampling_params.height, sampling_params.width
        )
        if frames.shape[1] != height or frames.shape[2] != width:
            frames = _resize_video_uint8(frames, height=height, width=width)
        self._warmup_vae_compile(height=height, width=width, num_frames=num_frames)

        num_inference_steps = int(sampling_params.num_inference_steps or self.num_inference_steps)
        if num_inference_steps != NUM_INFERENCE_STEPS:
            # AR-DMD distillation collapsed the trajectory into exactly two Euler steps; more of them
            # integrate a velocity field that no longer matches the shifted sigma grid.
            logger.warning(
                "JoyAI-Video-Edit was distilled for %d denoise steps; running %d will degrade output.",
                NUM_INFERENCE_STEPS,
                num_inference_steps,
            )

        noise_generator, encode_generator = resolve_generators(sampling_params, self.device)

        source_pixels = _video_uint8_to_vae_tensor(frames, self.device)
        source_latents = self.encode_source_windows(source_pixels, generator=encode_generator)
        expected = (num_latent_frames(num_frames), *latent_spatial_shape(height, width))
        if tuple(source_latents.shape[2:]) != expected:  # pragma: no cover - geometry is validated above
            raise RuntimeError(f"Source latents are {tuple(source_latents.shape[2:])}, expected {expected}.")

        # Once per request, on the first source frame: the per-chunk source signal reaches the DiT as
        # `ref_video_latent`, so re-encoding here would be a 7B forward per latent frame for an
        # identical result.
        prompt_embeds, prompt_embeds_mask = self.condition_encoder.encode(
            instruction,
            PIL.Image.fromarray(frames[0]),
            device=self.device,
            max_sequence_length=self.max_condition_tokens,
        )

        clean_latents = self.diffuse(
            source_latents,
            prompt_embeds.to(dtype=self.target_dtype),
            prompt_embeds_mask,
            num_inference_steps=num_inference_steps,
            generator=noise_generator,
        )
        video = self.decode_latents(clean_latents)

        # `setup_diffusion_pipeline_profiler` returns before defining `_stage_durations` when
        # profiling is off, so the property is only safe to touch when it is on.
        durations = self.stage_durations if self.enable_diffusion_pipeline_profiler else {}
        return DiffusionOutput(output=(video, fps), stage_durations=durations)
