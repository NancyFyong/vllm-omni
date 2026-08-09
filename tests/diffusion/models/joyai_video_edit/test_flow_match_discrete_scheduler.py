# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for :class:`FlowMatchDiscreteScheduler`.

Every failure mode of this scheduler is silent. A wrong ``shift``, a sigma grid built from the
training timesteps instead of ``linspace(1, 0, n + 1)``, or a ``step_index`` that fails to advance all
produce correctly-shaped finite latents and a video that merely looks worse than it should -- which
reads as "the model is mediocre", not "the port is wrong". With only two denoise steps there is no
averaging-out: each step carries roughly half the sampling trajectory.

The numbers below are not copied from a run of this code. ``sigma_1`` is
``(shift * 0.5) / (1 + (shift - 1) * 0.5)`` evaluated by hand at ``shift = 5.159``, and the
total-displacement test is a closed-form property of the Euler sum that holds for any ``shift``.
"""

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    NUM_INFERENCE_STEPS,
    NUM_TRAIN_TIMESTEPS,
    SCHEDULER_SHIFT,
)
from vllm_omni.diffusion.models.schedulers import FlowMatchDiscreteScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

#: (5.159 * 0.5) / (1 + 4.159 * 0.5) = 2.5795 / 3.0795, to more digits than the assertion needs.
SIGMA_MIDPOINT = 0.8376359798668615


def _scheduler(**kwargs) -> FlowMatchDiscreteScheduler:
    return FlowMatchDiscreteScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=SCHEDULER_SHIFT, reverse=True, solver="euler", **kwargs
    )


# --- the distilled grid -------------------------------------------------------------------------


def test_two_step_sigmas_match_the_distilled_schedule():
    """The three numbers the AR-DMD distillation was trained against."""
    sched = _scheduler()
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    torch.testing.assert_close(
        sched.sigmas, torch.tensor([1.0, SIGMA_MIDPOINT, 0.0]), atol=1e-6, rtol=0, check_dtype=False
    )
    torch.testing.assert_close(
        sched.timesteps, torch.tensor([1000.0, SIGMA_MIDPOINT * 1000.0]), atol=1e-3, rtol=0, check_dtype=False
    )


def test_shift_actually_bends_the_grid():
    """Guards against the shift being registered but never applied: ``sd3_time_shift`` is the identity
    at ``shift = 1``, so a port that drops the call passes every shape and monotonicity check and
    differs only in these values."""
    unshifted = FlowMatchDiscreteScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=1.0)
    unshifted.set_timesteps(NUM_INFERENCE_STEPS)
    torch.testing.assert_close(unshifted.sigmas, torch.tensor([1.0, 0.5, 0.0]), atol=1e-6, rtol=0)
    assert SIGMA_MIDPOINT > 0.5 + 0.3, "shift 5.159 must push the midpoint far above the linear 0.5"


def test_grid_is_built_from_inference_steps_not_the_training_grid():
    """The reason this class exists alongside ``FlowMatchEulerDiscreteScheduler``.

    That scheduler slices a ``num_train_timesteps``-long grid; this one builds ``n + 1`` points
    directly. At two steps the two constructions disagree, so swapping one for the other -- an
    inviting simplification, since the names are nearly identical -- silently changes the schedule.
    """
    sched = _scheduler()
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    assert len(sched.sigmas) == NUM_INFERENCE_STEPS + 1

    euler = pytest.importorskip(
        "vllm_omni.diffusion.models.schedulers.scheduling_flow_match_euler_discrete"
    ).FlowMatchEulerDiscreteScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=SCHEDULER_SHIFT)
    euler.set_timesteps(NUM_INFERENCE_STEPS)
    assert not torch.allclose(
        euler.timesteps[:NUM_INFERENCE_STEPS].float().cpu(), sched.timesteps[:NUM_INFERENCE_STEPS].cpu(), atol=1e-2
    ), "the two schedulers agree at 2 steps; one of them is no longer building its own grid"


def test_set_shift_rebuilds_the_grid():
    sched = _scheduler()
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    before = sched.sigmas.clone()
    sched.set_shift(1.0)
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    assert not torch.allclose(before, sched.sigmas)


# --- the Euler step -----------------------------------------------------------------------------


def test_full_schedule_integrates_to_exactly_minus_v():
    """Closed-form oracle, independent of ``shift``.

    For a constant velocity ``v`` the Euler sum telescopes to ``x + v * (sigma_n - sigma_0)``, and the
    grid always runs 1 -> 0, so the total displacement is exactly ``-v``. This catches a
    ``_step_index`` that never advances (both steps would reuse ``dt`` of the first interval) as well
    as an off-by-one in ``self.sigmas[self.step_index + 1]``.
    """
    sched = _scheduler()
    sched.set_timesteps(NUM_INFERENCE_STEPS)

    x = torch.zeros(1, 4)
    v = torch.full((1, 4), 3.0)
    for timestep in sched.timesteps:
        x = sched.step(v, timestep, x, return_dict=False)[0]
    torch.testing.assert_close(x, -v, atol=1e-6, rtol=0)
    assert sched.step_index == NUM_INFERENCE_STEPS


def test_step_upcasts_to_fp32():
    """The pipeline relies on this: it keeps the running latent in whatever ``step`` returns and casts
    to bf16 per DiT call. If ``step`` stopped upcasting, the accumulation would silently drop to bf16
    and the ``.to(target_dtype)`` in the pipeline would become a no-op that looks correct."""
    sched = _scheduler()
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    out = sched.step(
        torch.ones(1, 4, dtype=torch.bfloat16), sched.timesteps[0], torch.zeros(1, 4, dtype=torch.bfloat16)
    ).prev_sample
    assert out.dtype is torch.float32


def test_step_does_not_mutate_the_sample_it_is_given():
    """The pipeline passes its running latent straight in, with no defensive ``.clone()``.

    That clone was there and was removed: ``step`` builds ``prev_sample`` with a non-in-place add, so
    the copy was a full-latent allocation per denoise step for nothing. This is the assertion that
    keeps it removable -- a future ``sample.add_()`` or ``sample.mul_()`` in ``step`` would corrupt the
    caller's tensor, and at two steps per chunk the resulting drift would look like a model quality
    problem rather than an aliasing bug.
    """
    sched = _scheduler()
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    sample = torch.ones(1, 4)
    before = sample.clone()
    sched.step(torch.full((1, 4), 2.0), sched.timesteps[0], sample, return_dict=False)
    torch.testing.assert_close(sample, before, atol=0, rtol=0)


def test_integer_timestep_is_rejected():
    """``for i, t in enumerate(scheduler.timesteps)`` then passing ``i`` is the classic misuse; it
    would index sigma 0 and 1 by coincidence at 2 steps and produce almost-right output."""
    sched = _scheduler()
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    with pytest.raises(ValueError, match="integer indices"):
        sched.step(torch.ones(1, 4), 0, torch.zeros(1, 4))
    with pytest.raises(ValueError, match="integer indices"):
        sched.step(torch.ones(1, 4), torch.tensor(0), torch.zeros(1, 4))


def test_unsupported_solver_is_rejected_at_construction():
    with pytest.raises(ValueError, match="not supported"):
        FlowMatchDiscreteScheduler(solver="dpm")
