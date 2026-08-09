# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JoyAI-Video-Edit's MiMo-VL condition encoder, on CPU with no weights.

The MLLM stage has no numeric oracle available -- its output is 4096-wide hidden states that only mean
something to the DiT -- so everything checkable is *structural*, and every structural mistake here
produces a correctly-shaped tensor that makes the model follow the instruction poorly. That reads as
"the model is bad", not as a bug, which is why these are pinned at this level of detail:

- the templates byte-exact, including the literal backslash-n that upstream trained in;
- the drop index derived at runtime, and -- the real invariant -- shown to land exactly on the ``user``
  token of the *formatted* prompt, which is what makes borrowing the ``image`` template's index for the
  ``multiple_images`` path sound;
- truncation from the tail, distinguished from head truncation (identical shapes);
- the anchor frame resized by *area*, which is what keeps the image from consuming the whole
  1024-token budget at production resolution.

The stub tokenizer is a deliberate design choice over the real one: the alignment invariant is a
property of the *templates*, not of MiMo-VL's merges, so it should be provable without 16 GiB of
weights or a network call. It is verified against the real tokenizer in the weighted test file.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_condition import (
    IM_START,
    IMAGE_PLACEHOLDER,
    PROMPT_TEMPLATES,
    TEMPLATE_IMAGE,
    TEMPLATE_MULTIPLE_IMAGES,
    TEMPLATE_VIDEO,
    VISION_PLACEHOLDER,
    JoyAIVideoEditConditionEncoder,
    format_conditioning_text,
    resize_anchor_frame,
    resolve_drop_indices,
    user_turn,
    validate_text_encoder_width,
    vit_target_size,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_CONDITION_TOKENS,
    TEXT_STATES_DIM,
    VIT_FIXED_SIZE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

STUB_WIDTH = 8

# Splits ChatML specials, newlines and whitespace-delimited words. `<` is excluded from the word class
# so `background:<|im_end|>` yields two tokens, as a BPE tokenizer would.
_TOKEN_RE = re.compile(r"<\|[^|>]*\|>|\n|[^\s<]+")


class _StubTokenizer:
    """Deterministic word-level tokenizer, enough to exercise :func:`resolve_drop_indices`.

    Only two properties matter for the invariant under test: the same substring always yields the same
    token count, and ``user`` / ``<|im_start|>`` are their own tokens. Both hold for MiMo-VL's
    tokenizer too, which is why the alignment carries over.
    """

    def __init__(self):
        vocab: dict[str, int] = {}
        for text in (*PROMPT_TEMPLATES.values(), "user", IM_START, VISION_PLACEHOLDER):
            for token in _TOKEN_RE.findall(text):
                vocab.setdefault(token, len(vocab))
        self._vocab = vocab
        self._inverse = {i: t for t, i in vocab.items()}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.setdefault(token, len(self._vocab))

    def id_to_token(self, token_id: int) -> str:
        return self._inverse[token_id]

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[self.convert_tokens_to_ids(t) for t in _TOKEN_RE.findall(text)])


class _StubProcessor:
    """Records what it was handed and returns a fixed-length batch of the requested size."""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.calls: list[dict] = []

    def __call__(self, *, text, images, padding, return_tensors):
        self.calls.append({"text": text, "images": images, "padding": padding})
        return {
            "input_ids": torch.zeros(1, self.seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, self.seq_len, dtype=torch.long),
        }


class _StubTextEncoder(nn.Module):
    """Returns two distinguishable hidden-state layers so "which layer" is observable."""

    def __init__(self, seq_len: int, width: int = STUB_WIDTH):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))  # gives the module a device
        self.last = torch.arange(seq_len * width, dtype=torch.float32).reshape(1, seq_len, width)

    def forward(self, *, output_hidden_states, **kwargs):
        assert output_hidden_states is True
        return SimpleNamespace(hidden_states=(-self.last, self.last))


def _encoder(seq_len: int, **kwargs) -> tuple[JoyAIVideoEditConditionEncoder, _StubTextEncoder, _StubProcessor]:
    text_encoder = _StubTextEncoder(seq_len)
    processor = _StubProcessor(seq_len)
    encoder = JoyAIVideoEditConditionEncoder(text_encoder, _StubTokenizer(), processor, **kwargs)
    return encoder, text_encoder, processor


# --- templates ----------------------------------------------------------------------------------


def test_the_system_preamble_carries_a_literal_backslash_n_not_a_newline():
    """Two characters, backslash and ``n``, after ``system\\n ``.

    Almost certainly an upstream bug in the training script -- but it was trained in, so it is the
    contract. A real newline there retokenises the preamble and shifts the drop index by a token or
    two, silently feeding the DiT a text stream that starts mid-instruction. Note that asserting
    ``"\\n" in template`` (a newline) passes on *both* strings and proves nothing; this asserts on the
    escaped form.
    """
    for name, template in PROMPT_TEMPLATES.items():
        assert "\\n" in template, f"{name} lost the literal backslash-n"
        assert template.startswith(f"{IM_START}system\n \\nDescribe the "), name


def test_templates_match_upstream_byte_for_byte():
    """Independently written literal, so the shared-preamble composition cannot drift unnoticed.

    ``image`` and ``multiple_images`` are built from one preamble constant in the port; if that
    refactor ever changes a character, the load still succeeds and only prompt-following degrades.
    """
    assert PROMPT_TEMPLATES[TEMPLATE_IMAGE] == (
        "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    assert PROMPT_TEMPLATES[TEMPLATE_VIDEO].startswith(
        "<|im_start|>system\n \\nDescribe the video by detailing the following aspects:\n"
        "1. The main content and theme of the video.\n"
    )
    assert PROMPT_TEMPLATES[TEMPLATE_VIDEO].endswith(
        "5. camera angles, movements, and transitions used in the video:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )


def test_the_multiple_images_template_omits_the_user_turn_that_the_image_template_bakes_in():
    """The two differ by exactly that, which is why the caller has to supply it.

    Upstream's ``encode_prompt`` never forwards ``template_type`` to the image-conditioned path, so
    this template is the live one for every V2V request even though the caller asks for ``"video"``.
    """
    assert PROMPT_TEMPLATES[TEMPLATE_MULTIPLE_IMAGES] == PROMPT_TEMPLATES[TEMPLATE_IMAGE].replace(
        f"{IM_START}user\n{{}}<|im_end|>\n", "{}"
    )
    assert f"{IM_START}user" not in PROMPT_TEMPLATES[TEMPLATE_MULTIPLE_IMAGES].split("{}")[0]


def test_format_conditioning_text_substitutes_the_vision_placeholder_in_place():
    """``<image>\\n`` -> ``<|vision_start|><|image_pad|><|vision_end|>``, with nothing between it and
    the instruction.

    The trailing newline belongs to the placeholder, not to the prompt: leaving it in puts a newline
    between the image and the instruction, which is a different token sequence than was trained.
    """
    text = format_conditioning_text("make it snow")
    assert IMAGE_PLACEHOLDER not in text
    assert f"{VISION_PLACEHOLDER}make it snow<|im_end|>\n" in text
    assert text.startswith(f"{IM_START}system\n")
    assert text.endswith(f"{IM_START}assistant\n")
    assert text.count(VISION_PLACEHOLDER) == 1


def test_braces_in_the_edit_instruction_are_not_interpreted_as_format_fields():
    """``str.format`` is applied to the template, so the instruction is never rescanned.

    An instruction like this would otherwise raise ``KeyError: 'sky'`` mid-request -- or worse, be
    silently consumed if it happened to contain ``{}``.
    """
    assert "make the {sky} blue" in format_conditioning_text("make the {sky} blue")
    assert "swap {} for {}" in format_conditioning_text("swap {} for {}")


def test_user_turn_is_a_complete_chatml_turn():
    assert user_turn("x") == f"{IM_START}user\n{IMAGE_PLACEHOLDER}x<|im_end|>\n"


# --- drop index ---------------------------------------------------------------------------------


def test_the_drop_index_lands_exactly_on_the_user_token_of_the_formatted_prompt():
    """The invariant that makes reusing the ``image`` index for ``multiple_images`` correct.

    The index is measured on the ``image`` template's prefix, where ``<|im_start|>user\\n`` is part of
    the template, and applied to a ``multiple_images`` prompt, where the *caller* supplies that turn.
    It only works because the counts coincide -- and it is exactly the sort of coincidence that a
    later "cleanup" of either half breaks while leaving both shapes intact.
    """
    tokenizer = _StubTokenizer()
    indices = resolve_drop_indices(tokenizer)
    ids = tokenizer(format_conditioning_text("make it snow")).input_ids
    assert tokenizer.id_to_token(ids[indices[TEMPLATE_MULTIPLE_IMAGES]]) == "user"

    # The alignment is load-bearing on the caller's user turn: drop the turn and the index misses.
    without_turn = PROMPT_TEMPLATES[TEMPLATE_MULTIPLE_IMAGES].format(f"{VISION_PLACEHOLDER}make it snow")
    bare_ids = tokenizer(without_turn).input_ids
    assert tokenizer.id_to_token(bare_ids[indices[TEMPLATE_MULTIPLE_IMAGES]]) != "user"


def test_image_and_multiple_images_share_one_index_while_video_derives_its_own():
    """And they are not interchangeable: ``video`` stops one token earlier.

    ``image`` searches for the ``user`` token; ``video`` takes the *last* ``<|im_start|>``, so it keeps
    ``<|im_start|>user\\n`` where ``image`` keeps only ``user\\n``. Deriving either one the other way
    shifts the text stream by a token.
    """
    tokenizer = _StubTokenizer()
    indices = resolve_drop_indices(tokenizer)
    assert indices[TEMPLATE_MULTIPLE_IMAGES] == indices[TEMPLATE_IMAGE]

    video_prefix = tokenizer(PROMPT_TEMPLATES[TEMPLATE_VIDEO].split("{}")[0]).input_ids
    assert tokenizer.id_to_token(video_prefix[indices[TEMPLATE_VIDEO]]) == IM_START
    image_prefix = tokenizer(PROMPT_TEMPLATES[TEMPLATE_IMAGE].split("{}")[0]).input_ids
    assert tokenizer.id_to_token(image_prefix[indices[TEMPLATE_IMAGE]]) == "user"


def test_a_tokenizer_that_merges_the_user_token_away_is_reported_rather_than_indexed():
    """``list.index`` would raise ``ValueError: 872 is not in list``, naming nothing useful.

    The realistic form of this is a tokenizer whose merges swallow ``user`` into a longer token, so the
    id exists in the vocabulary but never appears in the tokenised prefix.
    """

    class _MergesUserAway(_StubTokenizer):
        def __call__(self, text: str) -> SimpleNamespace:
            ids = super().__call__(text).input_ids
            user_id = self.convert_tokens_to_ids("user")
            return SimpleNamespace(input_ids=[i for i in ids if i != user_id])

    with pytest.raises(ValueError, match="`user` token"):
        resolve_drop_indices(_MergesUserAway())


# --- anchor frame -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "height,width",
    [(DEFAULT_HEIGHT, DEFAULT_WIDTH), (1080, 1920), (512, 512), (300, 400), (128, 128)],
)
def test_vit_target_size_preserves_area_and_aspect_ratio(height, width):
    """Area pinned near ``512**2``; aspect ratio unchanged.

    Rounding to integers is the only source of error, hence the loose tolerances -- they are still far
    tighter than a factor-of-two mistake such as resizing to 512 on the *long side* instead.
    """
    new_h, new_w = vit_target_size(height, width)
    assert abs(new_h * new_w - VIT_FIXED_SIZE**2) / VIT_FIXED_SIZE**2 < 0.02
    assert abs((new_h / new_w) - (height / width)) / (height / width) < 0.02


def test_vit_target_size_grows_a_small_frame_as_well_as_shrinking_a_large_one():
    """Area-preserving means both directions. A ``min(1, scale)`` "fix" would look harmless.

    The number of image tokens is what the 1024-token budget is spent on, so it must depend on
    ``VIT_FIXED_SIZE`` alone and not on the request's resolution -- upwards included.
    """
    assert vit_target_size(DEFAULT_HEIGHT, DEFAULT_WIDTH) < (DEFAULT_HEIGHT, DEFAULT_WIDTH)
    grown_h, grown_w = vit_target_size(128, 128)
    assert grown_h > 128 and grown_w > 128


def test_vit_target_size_never_returns_a_zero_side():
    """A wide-enough frame rounds its short side to 0 without the clamp, and the ViT then raises.

    Note the direction: a *small* extreme aspect ratio is scaled up (10000x3 grows), so the clamp only
    ever engages on a frame whose area already exceeds the target.
    """
    height, width = vit_target_size(1, 10_000_000)
    assert height == 1, "the short side must be clamped to 1, not rounded to 0"
    assert width > 1


def test_resize_anchor_frame_resizes_a_pil_image_to_the_computed_size():
    height, width = vit_target_size(DEFAULT_HEIGHT, DEFAULT_WIDTH)
    resized = resize_anchor_frame(Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT)))
    # PIL is (width, height); the helper returns (height, width). Swapping them is a real transposition.
    assert resized.size == (width, height)


@pytest.mark.parametrize("shape", [(3, DEFAULT_HEIGHT, DEFAULT_WIDTH), (2, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH)])
def test_resize_anchor_frame_preserves_a_tensors_rank_and_dtype(shape):
    """uint8 in, uint8 out, and a ``[C, H, W]`` input does not gain a batch dimension.

    ``F.interpolate`` rejects uint8 outright, so the fp32 detour is required; forgetting to cast back
    hands the processor a float tensor it normalises a second time.
    """
    frame = torch.randint(0, 255, shape, dtype=torch.uint8)
    resized = resize_anchor_frame(frame)
    height, width = vit_target_size(DEFAULT_HEIGHT, DEFAULT_WIDTH)
    assert resized.shape == (*shape[:-2], height, width)
    assert resized.dtype == torch.uint8


def test_resize_anchor_frame_rejects_shapes_and_types_it_cannot_interpret():
    with pytest.raises(ValueError, match=r"\[C, H, W\]"):
        resize_anchor_frame(torch.zeros(DEFAULT_HEIGHT, DEFAULT_WIDTH))
    with pytest.raises(TypeError, match="PIL image or a tensor"):
        resize_anchor_frame("not-a-frame")


# --- encoder wiring -----------------------------------------------------------------------------


def test_a_plain_qwen2_5_vl_checkpoint_is_refused_by_width():
    """3584 vs 4096. The DiT's text projection would raise a bare shape mismatch instead."""
    validate_text_encoder_width(TEXT_STATES_DIM)
    with pytest.raises(ValueError, match="MiMo-VL-7B-RL-2508"):
        validate_text_encoder_width(3584)


def test_the_preamble_is_dropped_and_the_last_hidden_layer_is_the_one_used():
    """Both halves in one assertion: ``hidden_states[-1][:, drop_idx:]`` and nothing else.

    The stub returns ``(-last, last)``, so taking ``hidden_states[0]`` -- or ``hidden_states`` from a
    model configured with fewer layers -- flips every sign while keeping the shape.
    """
    drop_idx = resolve_drop_indices(_StubTokenizer())[TEMPLATE_MULTIPLE_IMAGES]
    encoder, text_encoder, _ = _encoder(seq_len=drop_idx + 10)

    embeds, mask = encoder.encode("make it snow", Image.new("RGB", (64, 64)))
    assert torch.equal(embeds, text_encoder.last[:, drop_idx:])
    assert embeds.shape == (1, 10, STUB_WIDTH)
    assert mask.shape == (1, 10)


def test_truncation_keeps_the_tail_where_the_instruction_is():
    """Head truncation gives the same shape and drops the instruction.

    After the drop the sequence reads ``user\\n <image pads...> <instruction> <|im_end|> assistant``, so
    the image padding sits in front and ``[:1024]`` would keep padding and lose the edit request --
    shapes intact, prompt-following gone. With the anchor frame resized a production request only
    reaches ~354 tokens, so this is exercised with an oversized sequence: the direction matters for a
    long instruction, and it is what upstream's image-conditioned path does (its text-only path, which
    has no image padding to spend, keeps the head instead).
    """
    drop_idx = resolve_drop_indices(_StubTokenizer())[TEMPLATE_MULTIPLE_IMAGES]
    encoder, text_encoder, _ = _encoder(seq_len=drop_idx + MAX_CONDITION_TOKENS + 176)

    embeds, mask = encoder.encode("make it snow", Image.new("RGB", (64, 64)))
    assert embeds.shape == (1, MAX_CONDITION_TOKENS, STUB_WIDTH)
    assert mask.shape == (1, MAX_CONDITION_TOKENS)
    torch.testing.assert_close(embeds[0, -1], text_encoder.last[0, -1])
    # And explicitly not the head slice, which is the same shape.
    assert not torch.equal(embeds, text_encoder.last[:, drop_idx : drop_idx + MAX_CONDITION_TOKENS])


def test_the_processor_receives_the_formatted_prompt_and_the_resized_anchor_frame():
    """Skipping the resize is invisible downstream: the processor accepts any size.

    It would just spend the whole budget on image tokens. This pins that the frame handed over is the
    area-normalised one and the text is the fully-formatted template.
    """
    encoder, _, processor = _encoder(seq_len=64)
    encoder.encode("make it snow", Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT)))

    (call,) = processor.calls
    assert call["text"] == [format_conditioning_text("make it snow")]
    height, width = vit_target_size(DEFAULT_HEIGHT, DEFAULT_WIDTH)
    assert [image.size for image in call["images"]] == [(width, height)]


def test_a_list_wrapped_prompt_is_refused_rather_than_stringified():
    """Upstream's ``encode_prompt`` takes ``List[str]``; this takes one string.

    ``format`` on a list produces ``"['make it snow']"`` -- a valid prompt describing a Python literal,
    which conditions the model on brackets and quotes and raises nothing.
    """
    encoder, _, _ = _encoder(seq_len=64)
    with pytest.raises(TypeError, match="single string"):
        encoder.encode(["make it snow"], Image.new("RGB", (64, 64)))


def test_a_non_positive_token_budget_is_refused_rather_than_disabling_truncation():
    """``[-0:]`` is the whole tensor, so a zero budget silently means "no limit"."""
    with pytest.raises(ValueError, match="must be positive"):
        _encoder(seq_len=64, max_sequence_length=0)

    encoder, _, _ = _encoder(seq_len=64)
    with pytest.raises(ValueError, match="must be positive"):
        encoder.encode("make it snow", Image.new("RGB", (64, 64)), max_sequence_length=0)
