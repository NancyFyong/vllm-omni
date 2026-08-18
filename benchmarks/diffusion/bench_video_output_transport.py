# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure host-memory cost of the two video output transport modes.

The diffusion worker hands large video tensors to the engine through POSIX
shared memory. Historically the reader always copied the payload out of the
segment and unlinked it, so a single video existed twice in host RAM at the
moment of hand-off. ``transport_mode="shared_memory"`` lets the reader map the
segment instead, removing that second copy.

This benchmark isolates the *reader* cost the way production does it:

* a producer subprocess packs the payload and stays alive (like a resident
  worker), so ``multiprocessing.resource_tracker`` does not unlink the segment;
* a fresh consumer subprocess maps or copies the payload and reports its own
  peak RSS.

Measuring the reader in a clean process is what makes the numbers trustworthy:
the producer's own peak (payload + segment) cannot leak into the measurement.

Usage::

    python benchmarks/diffusion/bench_video_output_transport.py
    python benchmarks/diffusion/bench_video_output_transport.py --size-mb 1024
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile

import torch

_MB = 1024 * 1024
_SENTINEL_VALUE = 42
_SENTINEL_INDEX = 12345
_READY = "PRODUCER_READY"
_RESULT = "RESULT"


def _peak_rss_mb() -> float:
    """Peak RSS of this process, in MiB (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_producer(handle_path: str, size_mb: int) -> None:
    """Pack a payload into shared memory, then stay alive until told to stop."""
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.ipc import pack_diffusion_output_shm

    payload = torch.full((size_mb * _MB,), _SENTINEL_VALUE, dtype=torch.uint8)
    payload[_SENTINEL_INDEX] = 7  # lets the consumer prove it read real data
    output = DiffusionOutput(output={"video": payload})
    pack_diffusion_output_shm(output)

    # After packing, the field holds a small metadata dict, not the payload.
    # That dict is all that crosses the process boundary (a few hundred bytes),
    # which is the whole point of the shared-memory hand-off.
    handle = output.output["video"]
    with open(handle_path, "w") as fh:
        json.dump(handle, fh)
    print(f"handle_bytes={len(json.dumps(handle))}", flush=True)

    print(_READY, flush=True)
    # Block until the driver closes our stdin. A resident owner keeps the
    # segment alive; if this process exited, resource_tracker would unlink it.
    sys.stdin.read()


def _run_consumer(handle_path: str, mode: str) -> None:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.ipc import borrowed_diffusion_output

    with open(handle_path) as fh:
        handle = json.load(fh)
    packed = DiffusionOutput(output={"video": handle})

    baseline_mb = _peak_rss_mb()

    # borrow=False reproduces the legacy copy-then-unlink reader exactly, so
    # both modes are measured through one call site.
    with borrowed_diffusion_output(packed, borrow=(mode == "shared_memory")) as unpacked:
        video = unpacked.output["video"]
        # Touch one byte per 4 KiB page so neither mode wins by leaving pages
        # unfaulted. Note: video.sum() would upcast uint8 to int64 and allocate
        # an 8x intermediate, swamping the very difference we are measuring.
        assert video[_SENTINEL_INDEX].item() == 7, "payload corrupted"
        assert video[0].item() == _SENTINEL_VALUE, "payload corrupted"
        touched = int(video[::4096].to(torch.int64).sum().item())
        assert touched > 0
        peak_mb = _peak_rss_mb()

    print(f"{_RESULT} {baseline_mb:.1f} {peak_mb:.1f} {peak_mb - baseline_mb:.1f}", flush=True)


def _measure(mode: str, size_mb: int, env: dict[str, str]) -> tuple[float, float, float, int]:
    handle_bytes = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        handle_path = os.path.join(tmpdir, "handle.pkl")
        producer = subprocess.Popen(
            [sys.executable, __file__, "--role=producer", f"--handle={handle_path}", f"--size-mb={size_mb}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            # Import logging shares stdout, so scan for the ready marker.
            while True:
                line = producer.stdout.readline()
                if not line:
                    raise RuntimeError("producer exited before packing the payload")
                if line.startswith("handle_bytes="):
                    handle_bytes = int(line.split("=", 1)[1])
                if line.strip() == _READY:
                    break

            completed = subprocess.run(
                [sys.executable, __file__, "--role=consumer", f"--handle={handle_path}", f"--mode={mode}"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        finally:
            producer.stdin.close()
            producer.wait(timeout=60)

    for line in completed.stdout.splitlines():
        if line.startswith(_RESULT):
            baseline, peak, delta = (float(x) for x in line.split()[1:4])
            return baseline, peak, delta, handle_bytes
    raise RuntimeError(f"consumer produced no result line:\n{completed.stdout}\n{completed.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["driver", "producer", "consumer"], default="driver")
    parser.add_argument("--handle", default="")
    parser.add_argument("--mode", choices=["copy", "shared_memory"], default="copy")
    parser.add_argument("--size-mb", type=int, default=512, help="payload size in MiB")
    args = parser.parse_args()

    if args.role == "producer":
        _run_producer(args.handle, args.size_mb)
        return 0
    if args.role == "consumer":
        _run_consumer(args.handle, args.mode)
        return 0

    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    # Subprocesses start in benchmarks/diffusion, so the repo root must be on
    # PYTHONPATH for `import vllm_omni` to resolve to this checkout.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    print(f"payload: {args.size_mb} MiB uint8 video tensor, reader measured in a fresh process\n")
    print(f"{'transport_mode':<16} {'RSS before':>14} {'RSS peak':>14} {'growth':>12}")
    print("-" * 60)

    results: dict[str, float] = {}
    handle_bytes = 0
    for mode in ("copy", "shared_memory"):
        baseline, peak, delta, handle_bytes = _measure(mode, args.size_mb, env)
        results[mode] = delta
        print(f"{mode:<16} {baseline:>10.1f} MiB {peak:>10.1f} MiB {delta:>8.1f} MiB")

    saved = results["copy"] - results["shared_memory"]
    print("-" * 60)
    print(f"\nmetadata crossing the process boundary: {handle_bytes} bytes")
    print(f"reader-side host memory avoided: {saved:.1f} MiB ({saved / args.size_mb:.2f}x payload)")
    print(
        "copy mode grows by ~1 payload because the reader allocates a private\n"
        "buffer while the segment is still mapped; shared_memory maps the pages\n"
        "the producer already wrote, so the payload is never duplicated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
