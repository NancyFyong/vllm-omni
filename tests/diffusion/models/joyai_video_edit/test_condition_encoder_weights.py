# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight checks for JoyAI-Video-Edit's MiMo-VL condition encoder.

Skipped unless ``JOYAI_VIDEO_EDIT_MODEL_DIR`` points at a local checkout, so CI stays green without the
16 GiB checkpoint. There is no numeric oracle for this stage -- the output is 4096-wide hidden states
that only mean something to the DiT -- so what the real weights add over ``test_condition_encoder.py``
is confirmation of the two facts that depend on MiMo-VL's *tokenizer and processor* rather than on the
port's own logic:

**The drop index against the real merges.** The CPU tests prove the index lands on the ``user`` token
under a stub tokenizer, which makes it a property of the templates. This confirms the same holds for
MiMo-VL's actual BPE merges -- the thing a stub cannot speak for, and the thing that decides whether
the DiT's text stream starts on content or mid-scaffolding.

**The token budget is a real constraint, and the resize is what satisfies it.** The processor decides
how many ``<|image_pad|>`` tokens a frame becomes, so only the real one can show that a 720x1248 anchor
frame needs 1170 image tokens un-resized against a 1024-token budget, and 336 resized. That is the
whole reason :func:`vit_target_size` exists, and it is invisible without the processor.

The tokenizer-only tests need no accelerator and no ViT, so they run on the ~11 MiB tokenizer files
alone; only the tests that instantiate the 7B model are gated on a device.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2Tokenizer

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_condition import (
    PROMPT_TEMPLATES,
    TEMPLATE_IMAGE,
    TEMPLATE_MULTIPLE_IMAGES,
    TEMPLATE_VIDEO,
    JoyAIVideoEditConditionEncoder,
    format_conditioning_text,
    resize_anchor_frame,
    resolve_drop_indices,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_CONDITION_TOKENS,
    TEXT_STATES_DIM,
)

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.gpu]

MODEL_DIR_ENV = "JOYAI_VIDEO_EDIT_MODEL_DIR"
ENCODER_SUBDIR = "MiMo-VL-7B-RL-2508"
PROMPT = "make the sky stormy and add falling snow"


def _encoder_path() -> Path:
    root = os.environ.get(MODEL_DIR_ENV)
    if not root:
        pytest.skip(f"set {MODEL_DIR_ENV} to the JoyAI-Video-Edit weights directory to run this")
    path = Path(root) / ENCODER_SUBDIR
    if not (path / "config.json").is_file():
        pytest.skip(f"{path} does not contain a MiMo-VL checkout")
    return path


@pytest.fixture(scope="module")
def tokenizer() -> Qwen2Tokenizer:
    """Tokenizer only -- a few MiB of vocab files, no model, no device."""
    return Qwen2Tokenizer.from_pretrained(_encoder_path(), local_files_only=True)


@pytest.fixture(scope="module")
def processor() -> AutoProcessor:
    """Processor only. It owns the image-token accounting but holds no weights."""
    return AutoProcessor.from_pretrained(_encoder_path(), local_files_only=True)


@pytest.fixture(scope="module")
def encoder() -> JoyAIVideoEditConditionEncoder:
    path = _encoder_path()
    if not torch.accelerator.is_available():
        pytest.skip("the 7B condition encoder needs an accelerator; a CPU forward is impractically slow")
    return JoyAIVideoEditConditionEncoder.from_pretrained(path, dtype=torch.bfloat16, device="cuda")


def _anchor_frame(height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH, *, seed: int = 0) -> Image.Image:
    """A frame with structure rather than a flat fill, so the ViT sees something to encode."""
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8, generator=generator)
    return Image.fromarray(pixels.numpy(), mode="RGB")


def _image_token_count(processor, tokenizer, frame: Image.Image) -> tuple[int, int]:
    """``(total tokens, image-pad tokens)`` for one formatted request."""
    inputs = processor(text=[format_conditioning_text(PROMPT)], images=[frame], padding=True, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return len(ids), sum(1 for token_id in ids if token_id == image_token_id)


# --- tokenizer-dependent facts -------------------------------------------------------------------


def test_the_real_tokenizer_produces_the_indices_the_cpu_tests_assume(tokenizer):
    """``{image: 34, multiple_images: 34, video: 91}`` with MiMo-VL's merges.

    Recorded as literals here and *only* here: the port computes them at runtime, and pinning them in
    the runtime path would silently misalign the text stream after any vocabulary change. Pinning them
    in a weight-gated test instead turns such a change into a named failure.
    """
    assert resolve_drop_indices(tokenizer) == {
        TEMPLATE_IMAGE: 34,
        TEMPLATE_MULTIPLE_IMAGES: 34,
        TEMPLATE_VIDEO: 91,
    }


def test_the_drop_index_lands_on_the_user_token_under_the_real_merges(tokenizer):
    """The same invariant the stub tokenizer proves, against the merges that actually ship.

    A BPE tokenizer is free to fuse ``user`` into a neighbouring token or to split the preamble
    differently from the stub; either would shift the boundary while leaving every shape intact. This
    is what makes borrowing the ``image`` template's index for the ``multiple_images`` path sound in
    production and not merely in principle.
    """
    drop_idx = resolve_drop_indices(tokenizer)[TEMPLATE_MULTIPLE_IMAGES]
    ids = tokenizer(format_conditioning_text(PROMPT)).input_ids

    assert tokenizer.decode(ids[drop_idx : drop_idx + 1]) == "user"
    # And the kept remainder starts at the turn header, with the instruction intact behind the image.
    kept = tokenizer.decode(ids[drop_idx:])
    assert kept.startswith("user\n<|vision_start|><|image_pad|><|vision_end|>")
    assert kept.endswith(f"{PROMPT}<|im_end|>\n<|im_start|>assistant\n")


def test_the_two_templates_share_a_preamble_token_count_not_just_a_prefix_string(tokenizer):
    """Why one index serves both: the ``multiple_images`` preamble is 33 tokens and ``user`` is the 34th.

    The ``image`` template supplies ``<|im_start|>user\\n`` itself; the ``multiple_images`` one has the
    caller supply it. The indices coincide only because the tokenised lengths line up, so state that
    directly rather than leaving it as a comment.
    """
    preamble = tokenizer(PROMPT_TEMPLATES[TEMPLATE_MULTIPLE_IMAGES].split("{}")[0]).input_ids
    assert len(preamble) == 33
    assert resolve_drop_indices(tokenizer)[TEMPLATE_MULTIPLE_IMAGES] == len(preamble) + 1


# --- processor-dependent facts -------------------------------------------------------------------


def test_the_resize_is_what_keeps_the_request_inside_the_token_budget(processor, tokenizer):
    """1170 image tokens un-resized vs 336 resized, against a 1024-token budget.

    This is the assertion that justifies :func:`vit_target_size`. Without the resize the image padding
    alone overruns the budget, and because truncation keeps the tail it is the *front* of the padding
    that goes -- i.e. the top rows of the anchor frame are silently cropped away and the model edits a
    partial source. Nothing raises; the output just tracks the source less well.
    """
    frame = _anchor_frame()
    raw_total, raw_image = _image_token_count(processor, tokenizer, frame)
    resized_total, resized_image = _image_token_count(processor, tokenizer, resize_anchor_frame(frame))

    assert raw_image > MAX_CONDITION_TOKENS, "un-resized padding no longer overruns the budget"
    assert resized_total < MAX_CONDITION_TOKENS, "the resized request should not need truncation at all"
    assert resized_image < raw_image / 3


def test_the_image_token_count_is_nearly_resolution_independent_after_the_resize(processor, tokenizer):
    """336 tokens from a 720x1248 frame, 324 from a 256x256 one -- within 4 %.

    Exact equality is not available: the ViT quantises to a patch grid, so a 28x48 grid (336 tokens)
    and a 36x36 one (324) both approximate the same area. What matters is that the *scale* no longer
    depends on the caller's resolution. A ``min(1, scale)`` variant of :func:`vit_target_size` -- which
    only ever shrinks -- would pass the budget test above and fail here by a factor of four, and would
    make a small source frame cost a quarter of the conditioning of a large one.
    """
    counts = {
        (height, width): _image_token_count(processor, tokenizer, resize_anchor_frame(_anchor_frame(height, width)))[1]
        for height, width in ((DEFAULT_HEIGHT, DEFAULT_WIDTH), (256, 256), (128, 128), (1080, 1920))
    }
    tokens = list(counts.values())
    assert max(tokens) / min(tokens) < 1.1, f"image-token cost varies with input resolution: {counts}"


# --- the loaded encoder --------------------------------------------------------------------------


def test_the_checkpoint_is_the_4096_wide_one_and_the_output_matches_the_dit(encoder):
    """MiMo-VL-7B-RL-2508 is 4096 wide where plain Qwen2.5-VL-7B is 3584.

    Confirms against the real config that the width guard admits the intended checkpoint -- the CPU
    test can only show it rejects the wrong number.
    """
    assert encoder.text_encoder.config.text_config.hidden_size == TEXT_STATES_DIM

    embeds, mask = encoder.encode(PROMPT, _anchor_frame())
    assert embeds.shape[0] == 1 and embeds.shape[2] == TEXT_STATES_DIM
    assert embeds.shape[1] <= MAX_CONDITION_TOKENS
    assert mask.shape == embeds.shape[:2]
    assert embeds.dtype == torch.bfloat16
    assert torch.isfinite(embeds).all()


def test_the_anchor_frame_changes_the_conditioning(encoder):
    """Two different frames, same resolution and same instruction -- the embeddings must differ.

    This is the one that catches a dropped or degenerate frame. ``resize_anchor_frame`` sits between the
    caller and the processor, and a version that returned a blank image, or that the processor ignored,
    would still yield a correctly-shaped 4096-wide tensor. Both frames resize to the same 28x48 grid, so
    the shapes match exactly and the comparison needs no alignment fudge.

    Note the direction the difference appears in: the LM is causal and the image precedes the
    instruction, so it is the instruction's hidden states that carry the frame, not the reverse.
    """
    baseline, _ = encoder.encode(PROMPT, _anchor_frame(seed=0))
    other_frame, _ = encoder.encode(PROMPT, _anchor_frame(seed=1))

    assert other_frame.shape == baseline.shape
    assert not torch.allclose(baseline, other_frame, rtol=1e-2, atol=1e-2)


def test_the_instruction_changes_the_conditioning(encoder):
    """Same frame, different edit request.

    Lengths differ because the instructions tokenise differently -- which is itself evidence the
    instruction is in the sequence rather than dropped -- so this compares the shared tail, where the
    instruction sits.
    """
    frame = _anchor_frame()
    baseline, _ = encoder.encode(PROMPT, frame)
    other_prompt, _ = encoder.encode("make it black and white", frame)

    shared = min(baseline.shape[1], other_prompt.shape[1])
    assert not torch.allclose(baseline[:, -shared:], other_prompt[:, -shared:], rtol=1e-2, atol=1e-2)


def test_encoding_is_deterministic_across_calls(encoder):
    """No sampling, no dropout, no state carried between requests.

    The encoder is built once per engine and reused for every request, so a leaked ``train()`` mode or
    a cache keyed on the wrong thing would show up as a request whose conditioning depends on what ran
    before it.
    """
    frame = _anchor_frame()
    first, first_mask = encoder.encode(PROMPT, frame)
    again, again_mask = encoder.encode(PROMPT, frame)
    assert torch.equal(first, again)
    assert torch.equal(first_mask, again_mask)
