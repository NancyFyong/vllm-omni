# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/jd-opensource/JoyAI-Video-Edit
"""MiMo-VL-7B condition encoder for JoyAI-Video-Edit.

The edit instruction is not encoded on its own: it is encoded *jointly with an anchor frame* of the
source video by an MLLM, and the last hidden states of that joint sequence become the DiT's text
stream. So this module produces one ``[1, <=1024, 4096]`` tensor per request, and the DiT concatenates
it into self-attention (there is no cross-attention).

Four things here are load-bearing and none of them fail loudly if ported wrong.

**The image-conditioned path always uses the ``multiple_images`` template.** Upstream's streaming
session asks for ``template_type="video"``, but ``Pipeline.encode_prompt`` does not forward
``template_type`` when ``images is not None`` -- it calls ``encode_prompt_multiple_images`` and lets the
default apply, and that function *raises* on anything but ``"multiple_images"``. Porting the caller's
apparent intent instead of its actual behaviour would swap in the video system preamble and change the
drop index, i.e. quietly re-condition the model on a prompt it was never distilled against.

**The drop index is derived from the ``image`` template and applied to a ``multiple_images`` prompt.**
That is only sound because the two share a byte-identical system preamble and because the caller
supplies the ``<|im_start|>user\\n`` turn that the ``image`` template bakes in. Measured with
MiMo-VL's tokenizer: the ``multiple_images`` preamble is 33 tokens, ``<|im_start|>`` is the 34th, so
dropping the ``image`` template's index of 34 lands exactly on the ``user`` token either way. Both
halves are recomputed at runtime from the caller's own tokenizer, never hardcoded -- see
:func:`resolve_drop_indices`.

**The system preamble contains a literal backslash-n.** ``"system\\n \\\\nDescribe the image..."`` -- the
second one is two characters, not a newline. It is presumably a bug in the upstream training script,
but it was trained in, so it is part of the contract. An editor or a careless retype that turns it
into a real newline retokenises the whole preamble and shifts the drop index.

**Truncation keeps the tail.** ``encode_prompt_multiple_images`` slices ``[-max:]`` while upstream's
*text-only* path slices ``[:max]``. Conflating them is a silent behaviour change, though on this path
it is latent rather than live: with the anchor frame area-normalised (below) a production request
lands at ~354 tokens against a 1024 budget, so truncation only engages on an unusually long
instruction. When it does, keeping the tail preserves the instruction and discards image padding,
which is the right trade -- keeping the head would do the opposite.

Two smaller notes. ``hidden_states[-1]`` is the post-final-norm state, equal to ``last_hidden_state``;
upstream indexes the tuple, so this does too. And ``AutoProcessor(use_fast=True)`` is deprecated in
transformers 5.x -- the parameter now only selects an image-processor backend and the default already
resolves to the same class, verified against this checkpoint, so it is omitted rather than translated.
"""

from __future__ import annotations

from typing import Any, Final

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    MAX_CONDITION_TOKENS,
    TEXT_STATES_DIM,
    VIT_FIXED_SIZE,
)

# --- prompt assembly ----------------------------------------------------------------------------

IM_START: Final = "<|im_start|>"
IM_END: Final = "<|im_end|>"
#: What upstream's streaming session writes into the user turn, and what the processor replaces with
#: however many ``<|image_pad|>`` tokens the anchor frame's ViT grid needs.
IMAGE_PLACEHOLDER: Final = "<image>\n"
VISION_PLACEHOLDER: Final = "<|vision_start|><|image_pad|><|vision_end|>"

TEMPLATE_IMAGE: Final = "image"
TEMPLATE_MULTIPLE_IMAGES: Final = "multiple_images"
TEMPLATE_VIDEO: Final = "video"

# The literal ``\\n`` after ``system\n`` is deliberate -- see the module docstring.
_SYSTEM_IMAGE: Final = (
    f"{IM_START}system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    f"quantity, text, spatial relationships of the objects and background:{IM_END}\n"
)
_SYSTEM_VIDEO: Final = (
    f"{IM_START}system\n \\nDescribe the video by detailing the following aspects:\n"
    "1. The main content and theme of the video.\n"
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n"
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n"
    "4. background environment, light, style and atmosphere.\n"
    f"5. camera angles, movements, and transitions used in the video:{IM_END}\n"
)
_ASSISTANT_TURN: Final = f"{IM_START}assistant\n"

#: All three of upstream's templates. Only ``multiple_images`` is ever formatted here; ``image`` is
#: kept because it is where the drop index comes from, and ``video`` because
#: :func:`resolve_drop_indices` reproduces upstream's full derivation rather than a subset of it.
PROMPT_TEMPLATES: Final[dict[str, str]] = {
    TEMPLATE_IMAGE: f"{_SYSTEM_IMAGE}{IM_START}user\n{{}}{IM_END}\n{_ASSISTANT_TURN}",
    TEMPLATE_MULTIPLE_IMAGES: f"{_SYSTEM_IMAGE}{{}}{_ASSISTANT_TURN}",
    TEMPLATE_VIDEO: f"{_SYSTEM_VIDEO}{IM_START}user\n{{}}{IM_END}\n{_ASSISTANT_TURN}",
}


def user_turn(prompt: str) -> str:
    """The user turn upstream's session builds, with the anchor frame's placeholder still symbolic."""
    return f"{IM_START}user\n{IMAGE_PLACEHOLDER}{prompt}{IM_END}\n"


def format_conditioning_text(prompt: str) -> str:
    """Assemble the full MLLM prompt for one edit instruction plus one anchor frame.

    ``str.format`` substitutes into the *template*, so braces inside ``prompt`` are never interpreted
    as fields -- an edit instruction like ``"make the {sky} blue"`` survives verbatim.
    """
    turn = user_turn(prompt).replace(IMAGE_PLACEHOLDER, VISION_PLACEHOLDER)
    return PROMPT_TEMPLATES[TEMPLATE_MULTIPLE_IMAGES].format(turn)


def resolve_drop_indices(tokenizer: Any) -> dict[str, int]:
    """How many leading tokens to discard from the MLLM's hidden states, per template.

    The system preamble is instruction scaffolding, not content, so its hidden states are dropped
    before the DiT sees them. The count depends on the tokenizer's merges, which is why upstream
    computes it at runtime and why this does too -- a hardcoded 34 would survive a vocabulary change
    and silently shift the text stream by a few tokens.

    ``image`` and ``multiple_images`` deliberately share one index; see the module docstring for why
    that is sound. ``video`` uses the *last* ``<|im_start|>`` because its preamble embeds numbered
    lines, so searching for the ``user`` token is not the same thing.
    """
    user_id = tokenizer.convert_tokens_to_ids("user")
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)

    image_prefix = tokenizer(PROMPT_TEMPLATES[TEMPLATE_IMAGE].split("{}")[0]).input_ids
    video_prefix = tokenizer(PROMPT_TEMPLATES[TEMPLATE_VIDEO].split("{}")[0]).input_ids

    if user_id not in image_prefix:
        raise ValueError("The `user` token does not appear in the tokenised image template prefix.")
    if im_start_id not in video_prefix:
        raise ValueError("The `<|im_start|>` token does not appear in the tokenised video template prefix.")

    image_index = image_prefix.index(user_id)
    return {
        TEMPLATE_IMAGE: image_index,
        TEMPLATE_MULTIPLE_IMAGES: image_index,
        TEMPLATE_VIDEO: max(i for i, token_id in enumerate(video_prefix) if token_id == im_start_id),
    }


# --- anchor frame -------------------------------------------------------------------------------


def vit_target_size(height: int, width: int, *, fixed_size: int = VIT_FIXED_SIZE) -> tuple[int, int]:
    """Area-preserving resize to roughly ``fixed_size ** 2`` pixels, aspect ratio untouched.

    The point is to make the number of ``<|image_pad|>`` tokens independent of the request's
    resolution, and it is what keeps the sequence inside the 1024-token budget at all. Measured on a
    720x1248 anchor frame: 336 image tokens resized, 1170 un-resized. Skipping this therefore does not
    raise -- the tail truncation just silently discards the *top* of the anchor frame (the grid is
    row-major, so the first rows go first) and the model conditions on a cropped source.
    """
    scale = ((fixed_size * fixed_size) / max(height * width, 1)) ** 0.5
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def resize_anchor_frame(frame: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
    """:func:`vit_target_size` applied to a PIL image or a ``[C, H, W]`` / ``[B, C, H, W]`` tensor.

    Bilinear without antialiasing, matching upstream on both branches. Tensors are interpolated in
    fp32 and cast back, because ``F.interpolate`` on a uint8 tensor is not supported.
    """
    if isinstance(frame, Image.Image):
        height, width = vit_target_size(frame.height, frame.width)
        return frame.resize((width, height), Image.Resampling.BILINEAR)

    if not isinstance(frame, torch.Tensor):
        raise TypeError(f"`frame` must be a PIL image or a tensor, got {type(frame).__name__}.")
    if frame.ndim not in (3, 4):
        raise ValueError(f"`frame` must be [C, H, W] or [B, C, H, W], got shape {tuple(frame.shape)}.")

    batched = frame if frame.ndim == 4 else frame.unsqueeze(0)
    height, width = vit_target_size(batched.shape[-2], batched.shape[-1])
    resized = nn.functional.interpolate(batched.float(), size=(height, width), mode="bilinear", align_corners=False).to(
        frame.dtype
    )
    return resized if frame.ndim == 4 else resized.squeeze(0)


def validate_text_encoder_width(hidden_size: int) -> None:
    """The MLLM's hidden width *is* the DiT's ``text_states_dim``; there is no adapter between them.

    MiMo-VL-7B is 4096 wide where plain Qwen2.5-VL-7B is 3584, so pointing this at the more obvious
    checkpoint is an easy mistake. It would fail anyway at the DiT's text projection, but with a bare
    shape mismatch 4096-vs-3584 rather than a sentence naming the cause.
    """
    if hidden_size != TEXT_STATES_DIM:
        raise ValueError(
            f"The condition encoder is {hidden_size} wide but the DiT's text stream expects "
            f"{TEXT_STATES_DIM}. JoyAI-Video-Edit needs MiMo-VL-7B-RL-2508; plain Qwen2.5-VL-7B is "
            f"3584 wide and will not fit."
        )


# --- encoder ------------------------------------------------------------------------------------


class JoyAIVideoEditConditionEncoder(nn.Module):
    """MiMo-VL wrapper producing the DiT's text stream from ``(instruction, anchor frame)``.

    Single-request by construction: the DiT shares one rotary table across the batch and its pipeline
    refuses ``len(prompts) > 1``, so accepting a list here would only create a way to build a batch
    that cannot be denoised.

    Call :meth:`encode` **once per request**, not once per chunk. The per-chunk source video reaches
    the DiT as ``ref_video_latent`` (VAE latents), never through the MLLM; re-encoding per chunk costs
    a 7B forward per latent frame -- a ~30x slowdown that no correctness test would catch.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        tokenizer: Any,
        processor: Any,
        *,
        max_sequence_length: int = MAX_CONDITION_TOKENS,
    ):
        super().__init__()
        if max_sequence_length <= 0:
            raise ValueError(f"`max_sequence_length` must be positive, got {max_sequence_length}.")
        self.text_encoder = text_encoder
        # Plain attributes on purpose: neither is an nn.Module, and component discovery walks children.
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.drop_indices = resolve_drop_indices(tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        max_sequence_length: int = MAX_CONDITION_TOKENS,
    ) -> JoyAIVideoEditConditionEncoder:
        """Load MiMo-VL-7B-RL-2508 from a local directory.

        ``local_files_only=True`` matches upstream and keeps a missing-weights mistake a fast local
        error instead of an attempted download from a serving process.
        """
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        validate_text_encoder_width(text_encoder.config.text_config.hidden_size)
        text_encoder = text_encoder.to(device).eval().requires_grad_(False)
        return cls(
            text_encoder,
            Qwen2Tokenizer.from_pretrained(path, local_files_only=True),
            AutoProcessor.from_pretrained(path, local_files_only=True),
            max_sequence_length=max_sequence_length,
        )

    @torch.no_grad()
    def encode(
        self,
        prompt: str,
        anchor_frame: Image.Image | torch.Tensor,
        *,
        device: torch.device | str | None = None,
        max_sequence_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one edit instruction against one anchor frame.

        Returns ``(prompt_embeds, prompt_embeds_mask)`` of shapes ``[1, L, 4096]`` and ``[1, L]`` with
        ``L <= max_sequence_length``. The mask is returned to match upstream's signature and is not
        read by the DiT (which attends to the whole concatenated text stream unmasked); with a single
        unpadded request it is all ones regardless.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a single string, got {type(prompt).__name__}.")
        limit = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        if limit <= 0:
            raise ValueError(f"`max_sequence_length` must be positive, got {limit}.")

        text = format_conditioning_text(prompt)
        image = resize_anchor_frame(anchor_frame)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")

        target = device if device is not None else next(self.text_encoder.parameters()).device
        moved = {k: v.to(target) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        hidden_states = self.text_encoder(**moved, output_hidden_states=True).hidden_states[-1]

        drop_idx = self.drop_indices[TEMPLATE_MULTIPLE_IMAGES]
        prompt_embeds = hidden_states[:, drop_idx:]
        prompt_embeds_mask = moved["attention_mask"][:, drop_idx:]

        # Tail, not head: the instruction follows the image tokens.
        if prompt_embeds.shape[1] > limit:
            prompt_embeds = prompt_embeds[:, -limit:, :]
            prompt_embeds_mask = prompt_embeds_mask[:, -limit:]
        return prompt_embeds, prompt_embeds_mask
