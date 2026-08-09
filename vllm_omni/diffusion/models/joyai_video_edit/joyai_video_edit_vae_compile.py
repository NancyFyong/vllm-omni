"""Optional ``torch.compile`` + ``channels_last_3d`` for the VAE, ported from upstream.

This is the port of ``deploy/xvideo/models/vae/vae_compile.py``, and it exists because profiling
named the VAE as the whole of this port's throughput gap to upstream. Upstream runs at ~1.79 s per
latent chunk against our ~2.19 s; of our 2.19 s, ``diffuse`` is 1.648 s -- *below* upstream's entire
per-chunk time -- while ``vae.encode`` + ``vae.decode`` add 0.316 s that upstream pays too but hides
behind its DiT thread, and compiles on top of that.

**Off by default, and only ``_encode`` is compiled at all.** Both halves of that are measurements, not
caution, and they point in opposite directions -- which is why upstream compiling both is not evidence
for doing so here. Per call at 720x1248 on one H20, same process, same clip:

===========  ==============  ================  ==================================================
target       eager           compiled steady   first call (the compile)
===========  ==============  ================  ==================================================
``_encode``  0.17--0.24 s    **0.072 s**       8--13 s *per shape*, and a warm cache does not help
``_decode``  **0.633 s**     0.692 s           34 s (229 s cold)
===========  ==============  ================  ==================================================

So compiled encode is 2.3--3.3x faster and dead stable at 0.072 s across every run, while compiled
decode is ~9% *slower than eager in its steady state* -- there is no call count at which compiling
``_decode`` pays back, so it is not compiled, and this is a deliberate divergence from upstream. That
divergence is a difference of regime rather than of hardware: upstream streams, so its decode runs once
per chunk over 1--2 latent frames with ``dynamic=False`` -- many calls, two static shapes -- whereas
``decode_latents`` here decodes the whole clip in a single call whose temporal extent *is* the clip
length. A static compile would then re-autotune per clip length and amortise it over one invocation,
and the ``dynamic=True`` compile that avoids that is the configuration measured slower above.

Encode's compile is a real win per call and still off by default, because the ~17 s for its two shapes
buys ~0.17 s per 9-frame window: break-even is around 100 windows, i.e. a clip of roughly 800 frames,
against the 15 windows of a 113-frame request. **A warm ``TORCHINDUCTOR_CACHE_DIR`` does not change
this** -- measured 12.0/11.7 s per shape cold against 13.0/13.4 s warm, because
``select_algorithm_autotune`` and 558 ``coordesc_tuning_bench`` calls re-run either way; only decode's
compile caches meaningfully (229 s to 34 s). The knob therefore pays for a long-lived server that
warms once and serves many requests, which is what upstream is, and not for a single offline request.
:func:`vae_compile_enabled` reads ``VLLM_OMNI_JOYAI_VAE_COMPILE``.

End to end on a 113-frame request, that arithmetic holds and is worth quoting because it is the number
a reader will otherwise expect the compile to improve. Measured back to back on one idle H20 (paired so
that node contention hits both arms equally -- an earlier unpaired attempt was thrown out when
``diffuse`` moved, which a VAE change cannot cause):

===============  ===========  ==============  ==============  ==============
run              wall         ``diffuse``     ``vae.encode``  ``vae.decode``
===============  ===========  ==============  ==============  ==============
eager            **32.57 s**  1.644 s/chunk   2.48 s total    2.13 s total
compile          52.12 s      1.646 s/chunk   21.60 s total   2.19 s total
===============  ===========  ==============  ==============  ==============

``diffuse`` is unchanged to 2 ms, so the arms are comparable, and the entire 19.55 s wall regression is
encode's 19.12 s -- the two warmup compiles, which land inside the timed region because
``_warmup_vae_compile`` runs there. Net of them the 15 calls take ~1.1 s against eager's 2.48 s, so the
steady-state win is exactly the one measured in isolation; it is simply 18x too small to repay the
compile at this clip length. The end-to-end and per-call measurements agree, which is the reason to
trust either.

**Enabling the compile also gives up bit-reproducibility.** Eager reproduces the whole-clip pixel
checksum exactly across three runs, on more than one GPU and under different node load (27,244,804,949); every
compiled run differs, including two runs of the *same* configuration (27,077,418,059 against
27,203,008,240). ``max-autotune`` benchmarks candidate kernels at compile time and machine conditions
decide which one wins, so output depends on what else the node was doing. That costs more than
tidiness: the checksum is what makes a numerics regression detectable at all, and under the compile
there is no fixed value to compare against.

Two further deliberate differences from upstream:

* ``_encode_dynamic`` (``dynamic=True``) is skipped, and not merely because the reference-image RV2V
  path it serves is deferred. Upstream's ``maybe_setup_encode_dynamic`` compiles *the same code object*
  as its static ``maybe_setup_encode`` -- both wrap ``vae._encode`` (``vae/vae.py:612``), differing only
  in the ``dynamic`` flag and where the wrapper is stored. Dynamo's recompile budget is per code object,
  so the two share one allowance of 8, and upstream's startup warms 49 dynamic shapes before 6 static
  ones. Its own log records the outcome: ``torch._dynamo hit config.recompile_limit (8)`` for
  ``_encode``, last guard ``(x.size()[4] % 3) != 0`` -- the Stem's stride-3 divisibility check at
  ``vae.py:482``, which cannot be proven under dynamic shapes and so specialises. A frame that exhausts
  its budget is skipped from then on, meaning it runs eager (verified directly: with the limit set to 4,
  seven static shapes exhaust it and a *fresh* wrapper over the same function then compiles nothing and
  returns the eager result). So at upstream's documented settings the encode compile is spent on the
  reference path and the shapes every request uses run uncompiled -- which is also why upstream's
  throughput advantage here is not evidence that its VAE compile works. Adding ``_encode_dynamic``
  would not add a compile; it would take the static one away. (If RV2V is ever enabled, note also that
  upstream warms this path outside autocast -- its log reads ``dynamic encode warmup done: 49/49 shapes
  autocast=False`` where every static shape reads ``autocast=True`` -- so it inherits the guard mismatch
  described in :func:`warmup` on top of the budget problem.)
* ``torch.accelerator.synchronize`` replaces ``torch.cuda.synchronize`` per this repo's banned-API
  rule.

The ``channels_last_3d`` weight conversion is separable from the compile and is applied by
:func:`convert_conv3d_to_channels_last`, because upstream applies both together and only one of them
could plausibly have been worth keeping on its own. Measured separately on case01, it was not: the
layout alone moved the whole-clip pixel checksum (27,607,984,473 against a twice-reproduced
27,244,804,949, about 1.2 grey levels in the mean) and bought no throughput that survives being
measured twice. The two measurements disagree in *sign* at the same 720x1248 and dtype -- 0.19 s per
encode call against eager's 0.151 s in the full pipeline, but 0.113 s against eager's 0.167 s in
isolation -- so the only claim the data supports is that it changes output and buys nothing reliable.
A knob like that has no defensible default but off, so
:envvar:`VLLM_OMNI_JOYAI_VAE_CHANNELS_LAST` exists to re-measure it on other hardware rather than to
be turned on. Compiling implies the layout, since the compiled kernels are autotuned against it -- so
**enabling the compile also accepts that numerics shift**, and it shifts them for the whole VAE rather
than only the compiled half: the conversion touches all 66 ``Conv3d`` weights, decode's included.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterable
from contextlib import nullcontext

import torch
from torch import nn

from vllm_omni.logger import init_logger

logger = init_logger(__name__)

COMPILE_ENV_VAR = "VLLM_OMNI_JOYAI_VAE_COMPILE"
CHANNELS_LAST_ENV_VAR = "VLLM_OMNI_JOYAI_VAE_CHANNELS_LAST"
COMPILE_MODE = "max-autotune-no-cudagraphs"
# Only `_encode`, and the omission of `_decode` is measured rather than cautious: see the module
# docstring. `dynamic=False` because encode sees many calls over exactly two shapes -- one per sliding
# window position -- so specialising on both is what makes the steady-state kernels worth having.
COMPILE_DYNAMIC = {"_encode": False}
# Set on a VAE once its Conv3d weights are `channels_last_3d`, so callers know whether to match the
# layout on their inputs. An attribute rather than a module global because it is a property of that
# VAE instance, and a served process may hold more than one.
LAYOUT_FLAG = "_vae_channels_last"


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def vae_compile_enabled() -> bool:
    """Whether to compile the VAE. Default off -- see the module docstring for why."""
    return _env_on(COMPILE_ENV_VAR)


def channels_last_enabled() -> bool:
    """Whether to apply the ``channels_last_3d`` layout *without* compiling.

    Separate from :func:`vae_compile_enabled` so the two can be measured apart -- upstream bundles
    them, so which one pays is not answerable from upstream's numbers. Compiling implies the layout.
    """
    return vae_compile_enabled() or _env_on(CHANNELS_LAST_ENV_VAR)


def _is_fx_tracing() -> bool:
    """Mirror upstream's probe, which tolerates the symbol having been renamed across versions."""
    symbolic_trace = torch.fx._symbolic_trace
    check = getattr(symbolic_trace, "is_fx_symbolic_tracing", None) or symbolic_trace.is_fx_tracing
    return bool(check())


def _fx_safe(compiled: Callable, fallback: Callable) -> Callable:
    """Route around the compiled callable while FX is tracing.

    Upstream carries this guard and it is kept: component discovery and any graph-capture pass walk
    the module with FX, and a Dynamo-compiled callable inside a symbolic trace raises rather than
    degrading. The eager original stays reachable, so tracing sees the real ops.
    """

    @functools.wraps(fallback)
    def wrapped(*args, **kwargs):
        return fallback(*args, **kwargs) if _is_fx_tracing() else compiled(*args, **kwargs)

    return wrapped


def _original(module: object, name: str) -> Callable:
    """Return ``module.name`` as it was before any compile wrapper, and remember it.

    Unwraps ``_torchdynamo_orig_callable`` so that calling setup twice compiles the eager function
    again rather than compiling a compiled wrapper.
    """
    stash = f"_vae_compile_original_{name}"
    cached = getattr(module, stash, None)
    if cached is not None:
        return cached
    original = getattr(module, name)
    while hasattr(original, "_torchdynamo_orig_callable"):
        original = original._torchdynamo_orig_callable
    setattr(module, stash, original)
    return original


def convert_conv3d_to_channels_last(vae: nn.Module) -> int:
    """Put every ``Conv3d`` weight in ``channels_last_3d``; returns how many were converted.

    Idempotent, and separate from the compile because it is cheap enough to apply unconditionally if
    it ever measures as a win on its own. Marks the VAE so callers can match the layout on their
    inputs -- see :func:`prep_input_for`.
    """
    count = 0
    for module in vae.modules():
        if isinstance(module, nn.Conv3d):
            module.weight.data = module.weight.data.to(memory_format=torch.channels_last_3d)
            count += 1
    setattr(vae, LAYOUT_FLAG, True)
    return count


def prep_input(x: torch.Tensor) -> torch.Tensor:
    """Match the weights' memory format, so the conv does not silently re-layout every call."""
    return x.to(memory_format=torch.channels_last_3d)


def prep_input_for(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``prep_input`` if this VAE's weights were converted, otherwise ``x`` untouched.

    Upstream calls ``prep_input`` on the real encode/decode inputs, not only in warmup, and that is
    load-bearing rather than tidiness: converting the *weights* alone leaves every conv reconciling
    two layouts per call, which costs time instead of saving it. Gating on the flag keeps the default
    path byte-for-byte the tensor it was before this module existed, so enabling the layout is the
    only thing that can shift conv algorithm selection.
    """
    return prep_input(x) if getattr(vae, LAYOUT_FLAG, False) else x


def setup_vae_compile(vae: nn.Module) -> bool:
    """Compile the targets in :data:`COMPILE_DYNAMIC` in place. Returns whether anything was done.

    Guarded by a ``_vae_compile_done`` mark on the module rather than a module-global id set:
    upstream's ``_configured: set[int]`` keys on ``id(vae)``, which is reused after a VAE is freed
    and would then skip setup on a *different* object that happened to land at the same address.
    """
    if getattr(vae, "_vae_compile_done", False):
        return False

    n_conv = convert_conv3d_to_channels_last(vae)
    compiled_targets = []
    for name, dynamic in COMPILE_DYNAMIC.items():
        if not hasattr(vae, name):
            raise RuntimeError(f"VAE has no {name}; cannot compile (upstream's vae_compile expects it)")
        original = _original(vae, name)
        compiled = torch.compile(original, mode=COMPILE_MODE, dynamic=dynamic)
        setattr(vae, name, _fx_safe(compiled, original))
        compiled_targets.append(f"{name}(dynamic={dynamic})")

    vae._vae_compile_done = True
    logger.info(
        "[joyai-vae-compile] %d Conv3d weights -> channels_last_3d; compiled %s (mode=%s)",
        n_conv,
        ", ".join(compiled_targets),
        COMPILE_MODE,
    )
    return True


def warmup(
    vae: nn.Module,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    pixel_shapes: Iterable[tuple[int, int, int]] = (),
    latent_shapes: Iterable[tuple[int, int, int]] = (),
    in_channels: int = 3,
    latent_channels: int = 64,
    autocast_encode: bool = True,
    autocast_decode: bool = True,
) -> tuple[int, int]:
    """Trigger compilation now, on the shapes the request will actually use.

    Without this the autotune cost lands inside the first chunk's latency instead of at load. Encode
    is compiled ``dynamic=False``, so every unseen shape recompiles and the caller must pass the exact
    ``(t, h, w)`` triples the rollout will hit. ``latent_shapes`` is accepted because :func:`warmup`
    outlives the current target list -- with ``_decode`` no longer compiled (see the module docstring)
    a decode warmup only pre-selects eager cuDNN algorithms, so the pipeline does not pass any. Returns
    ``(encode_ok, decode_ok)`` counts.

    **Both warmups run under autocast by default, and getting that wrong silently disables the whole
    mechanism.** Autocast state is part of the compiled graph's guards, so a warmup compiled outside
    autocast does not satisfy a request that runs inside it -- the request recompiles from scratch and
    pays the cost the warmup existed to move. This port had exactly that bug: encode was warmed
    without autocast while :meth:`encode_source_windows` calls it inside one, which measured as
    ``vae.encode`` taking 43.9 s over 15 chunks against eager's 2.27 s, even with a fully warm
    TorchInductor cache and no autotuning anywhere in the timed region. In isolation the compiled
    kernels are the *faster* ones (0.072 s against 0.17--0.24 s per 9-frame window); the regression was
    entirely an 8--13 s recompile landing on each of the two real shapes. Upstream's own
    ``warmup_encode`` *default* is the same trap (``autocast=False``) but its caller overrides it to
    ``autocast=(vae_dtype != float32)`` -- true for bf16 -- at ``joyomni_streaming.py:258``, which is
    why every static shape in upstream's startup log reads ``autocast=True``.
    """
    dev = torch.device(device)
    ok_encode = ok_decode = 0
    # Materialised because the counts are reported after the loops: a generator argument would be
    # exhausted by then and every shape would report as skipped.
    pixel_shapes = tuple(pixel_shapes)
    latent_shapes = tuple(latent_shapes)

    def autocast_ctx(enabled: bool):
        """The request's autocast state, reproduced -- see the note on guards in the docstring."""
        if not enabled or dev.type not in {"cuda", "cpu"}:
            return nullcontext()
        return torch.autocast(device_type=dev.type, dtype=dtype, enabled=True)

    for t, h, w in pixel_shapes:
        x = prep_input(torch.zeros(1, in_channels, t, h, w, device=dev, dtype=dtype))
        try:
            with torch.no_grad(), autocast_ctx(autocast_encode):
                vae.encode(x)
            ok_encode += 1
        except Exception as exc:  # noqa: BLE001 -- a failed warmup must not fail the request
            logger.warning("[joyai-vae-compile] encode warmup failed for (%d,%d,%d): %r", t, h, w, exc)

    for t, h, w in latent_shapes:
        z = prep_input(torch.zeros(1, latent_channels, t, h, w, device=dev, dtype=dtype))
        try:
            with torch.no_grad(), autocast_ctx(autocast_decode):
                vae.decode(z, return_dict=False)
            ok_decode += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("[joyai-vae-compile] decode warmup failed for (%d,%d,%d): %r", t, h, w, exc)

    # Gated on the device *matching* the accelerator, not merely on one existing: `synchronize` raises
    # `ValueError: cpu doesn't match the current accelerator cuda` for a CPU warmup on a GPU box, which
    # `is_available()` alone does not exclude.
    if dev.type != "cpu" and torch.accelerator.is_available():
        torch.accelerator.synchronize(dev)
    logger.info(
        "[joyai-vae-compile] warmup: encode %d/%d shapes, decode %d/%d shapes",
        ok_encode,
        len(pixel_shapes),
        ok_decode,
        len(latent_shapes),
    )
    return ok_encode, ok_decode
