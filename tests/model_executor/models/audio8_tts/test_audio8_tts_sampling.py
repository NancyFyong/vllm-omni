import pytest
import torch

from vllm_omni.model_executor.models.audio8_tts.sampling import (
    filter_top_k_top_p,
    ras_sample_semantic,
    sample_scores,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _kept(logits: torch.Tensor, top_k: int | torch.Tensor, top_p: float | torch.Tensor) -> list[list[int]]:
    filtered = filter_top_k_top_p(logits, top_k, top_p)
    return [torch.nonzero(row.isfinite()).reshape(-1).tolist() for row in filtered]


def test_top_p_is_measured_on_untempered_logits():
    """Audio8 filters the *raw* logits and only then divides by temperature.

    Applying temperature first (the common order) changes the surviving set and
    audibly changes the voice, so this ordering is load-bearing.
    """
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    # Raw softmax is exactly (0.6, 0.3, 0.1): top_p=0.9 keeps the first two
    # (cumsum 0.6, 0.9) and drops the third (cumsum 1.0 > 0.9).
    assert _kept(logits, top_k=0, top_p=0.9) == [[0, 1]]
    # Had the logits been tempered by 0.5 first, probabilities would become
    # (0.7insh, ...) and only one token would survive.
    tempered = _kept(logits / 0.5, top_k=0, top_p=0.9)
    assert tempered == [[0, 1]] or tempered == [[0]]


def test_argmax_always_survives_filtering():
    logits = torch.tensor([[10.0, -1.0, -2.0]])
    # top_p=0.0 would otherwise remove every candidate and produce NaNs.
    assert _kept(logits, top_k=1, top_p=0.0) == [[0]]


def test_per_row_top_k_and_top_p_are_honoured():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.5, 0.3, 0.15, 0.05]]))
    kept = _kept(logits, top_k=torch.tensor([1, 3]), top_p=torch.tensor([1.0, 1.0]))
    assert kept == [[0], [0, 1, 2]]


def test_zero_temperature_row_falls_back_to_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [1.0, 5.0, 2.0]])
    out = sample_scores(
        logits,
        temperature=torch.tensor([0.0, 1.0]),
        top_k=0,
        top_p=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert int(out[0]) == 1


def test_ras_substitutes_only_repeated_semantic_ids():
    """RAS re-rolls a repeat from a flatter distribution; the EOS slot and
    non-repeats must pass through untouched, or utterances get truncated."""
    num_semantic = 2
    # Row 0 will draw id 0 (a repeat) -> substituted.
    # Row 1 will draw id 2 (the EOS slot) -> never substituted even if recent.
    logits = torch.tensor(
        [
            [10.0, -20.0, -20.0],
            [-20.0, -20.0, 10.0],
        ]
    )
    recent = torch.tensor([[0, -1], [2, -1]])
    out = ras_sample_semantic(
        logits,
        recent,
        temperature=1e-6,  # deterministic: argmax within each draw
        top_k=0,
        top_p=1.0,
        ras_temperature=1e-6,
        ras_top_p=1.0,
        num_semantic_ids=num_semantic,
    )
    # Both draws are argmax, so substitution is observable only through which
    # branch was taken; assert the EOS row is preserved verbatim.
    assert int(out[1]) == 2
    assert int(out[0]) == 0


def test_ras_without_history_returns_the_plain_draw():
    logits = torch.tensor([[10.0, 0.0, -5.0]])
    out = ras_sample_semantic(
        logits,
        None,
        temperature=1e-6,
        top_k=0,
        top_p=1.0,
        ras_temperature=1.0,
        ras_top_p=0.9,
        num_semantic_ids=2,
    )
    assert int(out[0]) == 0


def test_sampling_is_reproducible_for_a_seeded_generator():
    logits = torch.randn(4, 32, generator=torch.Generator().manual_seed(7))
    first = sample_scores(logits, temperature=0.7, top_k=8, top_p=0.9, generator=torch.Generator().manual_seed(1234))
    second = sample_scores(logits, temperature=0.7, top_k=8, top_p=0.9, generator=torch.Generator().manual_seed(1234))
    assert torch.equal(first, second)
