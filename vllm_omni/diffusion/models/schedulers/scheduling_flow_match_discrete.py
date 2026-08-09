# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/jd-opensource/JoyAI-Video-Edit
"""Discrete flow-match scheduler used by JoyAI-Video-Edit.

This is *not* interchangeable with :class:`FlowMatchEulerDiscreteScheduler` in this package: that one
derives its sigmas from a ``num_train_timesteps``-long training grid, whereas this one builds them
directly from ``linspace(1, 0, num_inference_steps + 1)`` and then applies the SD3 time shift. With
the AR-DMD-distilled 2-step schedule the two disagree from the very first step, so the model needs
this exact construction.
"""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Euler flow-match scheduler over a shifted ``linspace(1, 0, n + 1)`` sigma grid.

    Args:
        num_train_timesteps: Scale mapping sigmas onto timesteps.
        shift: SD3 time-shift strength. JoyAI-Video-Edit was distilled at ``5.159``; changing it
            degrades output silently rather than raising.
        reverse: Whether sigmas run 1 -> 0 (the trained direction).
        solver: Only ``"euler"`` is supported.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
    ):
        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")

        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self._step_index = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_shift(self, shift: float) -> None:
        self.register_to_config(shift=shift)

    def sd3_time_shift(self, t: torch.Tensor) -> torch.Tensor:
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        self.num_inference_steps = num_inference_steps

        sigmas = self.sd3_time_shift(torch.linspace(1, 0, num_inference_steps + 1))
        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self._step_index = None

    def index_for_timestep(self, timestep: torch.Tensor, schedule_timesteps: torch.Tensor | None = None) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        # A repeated timestep means we are re-entering a step we already consumed; take the later one.
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep: torch.Tensor) -> None:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        self._step_index = self.index_for_timestep(timestep)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> FlowMatchDiscreteSchedulerOutput | tuple:
        """Take one Euler step. ``timestep`` must be a value from ``self.timesteps``, not its index."""
        if isinstance(timestep, int) or (torch.is_tensor(timestep) and not torch.is_floating_point(timestep)):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps is not supported. "
                "Pass one of the `scheduler.timesteps` values instead."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Integrate in fp32: at 2 steps each increment carries ~half the trajectory, so bf16
        # rounding here is visible in the output.
        sample = sample.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        prev_sample = sample + model_output.to(torch.float32) * dt

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)
