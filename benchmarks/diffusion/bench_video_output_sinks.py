# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare video output sinks from a *consumer* process on the same host.

Measures what a co-located consumer (an RL rollout worker) actually pays to
receive one video: bytes crossing the process boundary, its own peak RSS, and
whether the frames it gets back are the frames that were generated.

    python benchmarks/diffusion/bench_video_output_sinks.py --frames 48

The consumer runs in a separate process so the producer's own peak cannot
pollute the measurement.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Running this as a script puts the script's directory on sys.path, not the cwd,
# so the repo root has to be added explicitly for the editable checkout.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _frames(count: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((count, height, width, 3)) * 255).astype(np.uint8)


def _rss_mib() -> float:
    """Current RSS, not the peak.

    ru_maxrss is the high-water mark for the whole process, and importing torch
    already pushes that above any realistic video payload, which would report a
    zero difference between the sinks.
    """
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found")


def _run_consumer(mode: str, payload_path: str, expected_digest: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--role",
            "consumer",
            "--mode",
            mode,
            "--payload",
            payload_path,
            "--expected",
            expected_digest,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"consumer failed ({mode}):\n{proc.stdout}\n{proc.stderr}")


def _digest(array: np.ndarray) -> str:
    """Hash without copying: memoryview over a contiguous buffer."""
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def _consumer(mode: str, payload_path: str, expected_digest: str) -> None:
    # Import first, then take the baseline. The shared-memory path imports the
    # vllm_omni stack (and therefore torch), which costs far more than any video
    # payload and would otherwise be charged to the sink.
    if mode == "shared_memory":
        from vllm_omni.diffusion.ipc import borrowed_shm_array
    else:
        import av

    with open(payload_path) as handle_file:
        payload = json.load(handle_file)

    baseline = _rss_mib()

    if mode == "shared_memory":
        with borrowed_shm_array(payload["handle"]) as frames:
            # Hashing reads every byte, so both sinks are measured having really
            # consumed the whole video rather than just referencing it.
            exact = _digest(frames) == expected_digest
            rss = _rss_mib()
    else:
        video_bytes = base64.b64decode(payload["b64_json"])
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            with av.open(tmp_path) as container:
                stream = container.streams.video[0]
                decoded = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(stream)])
        finally:
            os.unlink(tmp_path)
        exact = _digest(decoded) == expected_digest
        rss = _rss_mib()

    print(
        "RESULT "
        + json.dumps(
            {
                "mode": mode,
                "baseline_mib": round(baseline, 1),
                "rss_mib": round(rss, 1),
                "delta_mib": round(rss - baseline, 1),
                "boundary_bytes": os.path.getsize(payload_path),
                "lossless": exact,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["driver", "consumer"], default="driver")
    parser.add_argument("--mode", choices=["base64", "shared_memory"], default="base64")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--payload")
    parser.add_argument("--expected")
    args = parser.parse_args()

    if args.role == "consumer":
        _consumer(args.mode, args.payload, args.expected)
        return

    frames = _frames(args.frames, args.height, args.width)
    payload_mib = frames.nbytes / 1024**2
    print(f"video: {frames.shape} uint8 = {payload_mib:.1f} MiB\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        expected_digest = _digest(frames)

        rows = []
        for mode in ("base64", "shared_memory"):
            payload_path = os.path.join(tmpdir, f"payload_{mode}.json")
            if mode == "shared_memory":
                from vllm_omni.diffusion.ipc import export_array_to_shm

                payload = {"handle": export_array_to_shm(frames)}
            else:
                from vllm_omni.entrypoints.openai.video_api_utils import encode_video_base64

                payload = {"b64_json": encode_video_base64(frames, fps=24)}
            with open(payload_path, "w") as out:
                json.dump(payload, out)
            rows.append(_run_consumer(mode, payload_path, expected_digest))

    print(f"{'sink':<15}{'boundary':>12}{'consumer RSS':>16}{'lossless':>10}")
    for row in rows:
        print(
            f"{row['mode']:<15}{row['boundary_bytes'] / 1024**2:>10.2f} MiB"
            f"{row['delta_mib']:>13.1f} MiB{str(row['lossless']):>10}"
        )
    print(f"\npayload = {payload_mib:.1f} MiB")


if __name__ == "__main__":
    main()
