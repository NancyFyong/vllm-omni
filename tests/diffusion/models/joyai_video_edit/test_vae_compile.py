# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the optional VAE ``torch.compile`` path.

CPU, no weights, no compilation: these test the *bookkeeping* around the compile, which is where its
one shipped bug lived. `torch.compile` itself is not exercised -- there is nothing to assert about it
that inductor does not already guarantee, and compiling a 66-layer VAE is not an L1 cost.

The load-bearing test here is :func:`test_warmup_runs_encode_under_autocast`. Autocast state is part
of a compiled graph's guards, so warming a shape outside autocast does not satisfy a request that runs
inside one: the request recompiles and pays the exact cost the warmup exists to move. This port warmed
encode without autocast while ``encode_source_windows`` calls it inside one, and the result was a
compile that measured *four times slower end to end than eager* -- ``vae.encode`` 43.9 s over 15 chunks
against eager's 2.27 s -- with a fully warm TorchInductor cache and no autotuning in the timed region,
because every real shape recompiled. Nothing raised, no shape was wrong, and the isolated kernels were
in fact 2.3--3.3x faster than eager; only the end-to-end number showed it, and it pointed at the wrong
cause. A test that asserted "warmup calls encode" would have passed throughout.
"""

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit import joyai_video_edit_vae_compile as vc

# `cpu`, not `gpu`: every test here drives a stub VAE on CPU tensors and asserts on recorded call
# state, so none of them needs an accelerator. The hardware mark is what Buildkite collects on.
pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _RecordingVAE(torch.nn.Module):
    """Records the autocast state and memory format seen by each ``encode``/``decode`` call."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_channels = 64
        self.calls: list[dict] = []

    def _record(self, kind: str, x: torch.Tensor) -> None:
        self.calls.append(
            {
                "kind": kind,
                # Queried for *this* device type: bare `torch.is_autocast_enabled()` reports the CUDA
                # state and is therefore False throughout a CPU autocast block.
                "autocast": torch.is_autocast_enabled(x.device.type),
                "shape": tuple(x.shape),
                "channels_last_3d": x.is_contiguous(memory_format=torch.channels_last_3d),
            }
        )

    def encode(self, x: torch.Tensor):
        self._record("encode", x)
        return torch.zeros(1, self.latent_channels, 1, x.shape[3] // 24, x.shape[4] // 24)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        self._record("decode", z)
        return (torch.zeros(1, 3, 1, z.shape[3] * 24, z.shape[4] * 24),)


def _warm(vae: _RecordingVAE, **kwargs) -> tuple[int, int]:
    return vc.warmup(
        vae,
        device="cpu",
        dtype=torch.bfloat16,
        pixel_shapes=[(1, 48, 48), (9, 48, 48)],
        latent_shapes=[(2, 2, 2)],
        **kwargs,
    )


def test_warmup_runs_encode_under_autocast():
    """Encode must be warmed in the autocast state the pipeline calls it in, or it recompiles.

    Asserted for encode specifically, and separately from decode, because the two are warmed by
    different arguments and only encode regressed. Upstream's ``warmup_encode`` *default* is
    ``autocast=False``, which is what this port copied; its caller passes
    ``autocast=(vae_dtype != float32)`` at ``joyomni_streaming.py:258``, so the served behaviour is
    ``True`` for bf16 -- as upstream's own startup log records.
    """
    vae = _RecordingVAE()
    _warm(vae)
    encodes = [c for c in vae.calls if c["kind"] == "encode"]
    assert encodes, "warmup did not call encode at all"
    assert all(c["autocast"] for c in encodes), (
        f"encode warmed outside autocast: {[c['autocast'] for c in encodes]}; the compiled graph will "
        f"not be reused by encode_source_windows, which runs inside autocast"
    )


def test_warmup_runs_decode_under_autocast():
    """Same contract for decode, which ``decode_latents`` also calls inside an autocast block."""
    vae = _RecordingVAE()
    _warm(vae)
    decodes = [c for c in vae.calls if c["kind"] == "decode"]
    assert decodes and all(c["autocast"] for c in decodes)


def test_warmup_autocast_is_overridable_per_target():
    """The flags are independent: a caller can warm one target outside autocast without the other.

    Guards the wiring rather than a default -- a single shared flag would still pass both tests above
    while making the reference-image path (which upstream warms with ``autocast=False``) unexpressible.
    """
    vae = _RecordingVAE()
    _warm(vae, autocast_encode=False)
    assert not any(c["autocast"] for c in vae.calls if c["kind"] == "encode")
    assert all(c["autocast"] for c in vae.calls if c["kind"] == "decode")


def test_warmup_inputs_match_the_weight_layout():
    """Warmup inputs go through ``prep_input``, so they are ``channels_last_3d`` like the weights.

    A warmup on a contiguous input compiles a graph with a different input layout than the request's,
    which is the same guard-mismatch failure as the autocast one and equally silent.
    """
    vae = _RecordingVAE()
    _warm(vae)
    assert all(c["channels_last_3d"] for c in vae.calls), [c["shape"] for c in vae.calls]


def test_warmup_counts_shapes_and_survives_a_failure():
    """A shape that raises is reported, not propagated: a failed warmup must not fail the request."""

    class _Failing(_RecordingVAE):
        def encode(self, x: torch.Tensor):
            raise RuntimeError("no")

    ok_encode, ok_decode = _warm(_Failing())
    assert (ok_encode, ok_decode) == (0, 1)


def test_warmup_accepts_generators_for_shapes():
    """Shape arguments are materialised, so passing generators does not silently warm nothing.

    The counts are reported after the loops; an un-materialised generator would be exhausted by then
    and every shape would log as skipped while the warmup had in fact happened.
    """
    vae = _RecordingVAE()
    ok_encode, ok_decode = vc.warmup(
        vae,
        device="cpu",
        dtype=torch.bfloat16,
        pixel_shapes=((t, 48, 48) for t in (1, 9)),
        latent_shapes=((2, 2, 2) for _ in range(1)),
    )
    assert (ok_encode, ok_decode) == (2, 1)


def test_both_knobs_default_off(monkeypatch):
    """Compile and the layout conversion are opt-in.

    Measured on one H20: the layout alone shifts the whole-clip pixel checksum, and encode's compile
    costs 8-13 s per shape on first call against the ~0.1-0.17 s per 9-frame window it then saves --
    break-even around 100 windows, which a single offline request does not reach. Neither is a safe
    default, so the default is asserted rather than left to whichever env vars happen to be set.
    """
    monkeypatch.delenv(vc.COMPILE_ENV_VAR, raising=False)
    monkeypatch.delenv(vc.CHANNELS_LAST_ENV_VAR, raising=False)
    assert not vc.vae_compile_enabled()
    assert not vc.channels_last_enabled()


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("0", False), ("", False)])
def test_env_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(vc.COMPILE_ENV_VAR, value)
    assert vc.vae_compile_enabled() is expected


def test_only_encode_is_compiled():
    """``_decode`` and ``_encode_dynamic`` are both deliberately absent, for different measured reasons.

    ``_decode``: its compiled *steady state* is slower than eager on this hardware -- 0.692 s against
    0.633 s per whole-clip call in the same process -- on top of 34 s to compile, so no call count pays
    it back. Upstream compiles it because upstream streams: its decode runs per chunk over 1-2 latent
    frames, many calls at two static shapes, where this pipeline makes one call whose temporal extent is
    the clip length.

    ``_encode_dynamic``: upstream compiles the *same code object* as the static encode, and dynamo's
    recompile budget is per code object, so the two share one allowance of 8 -- which upstream's 49
    dynamic warmup shapes exhaust before the 6 static ones are warmed. Its own log shows the limit hit,
    after which the frame runs eager. Adding it here would not add a compile; it would remove one.

    Pinned as a test because "upstream compiles all three" is the obvious thing for a future reader to
    restore, and both restorations cost throughput while changing no output.
    """
    assert vc.COMPILE_DYNAMIC == {"_encode": False}


def test_setup_requires_the_targets_to_exist():
    """A renamed VAE internal must fail loudly, not silently compile nothing.

    ``setup_vae_compile`` walks names, so an upstream rename of ``_encode`` would otherwise leave the
    compile enabled, warmed, logged -- and absent.
    """

    class _NoEncode(torch.nn.Module):
        def _decode(self, z):  # pragma: no cover - never called
            return z

    with pytest.raises(RuntimeError, match="_encode"):
        vc.setup_vae_compile(_NoEncode())
