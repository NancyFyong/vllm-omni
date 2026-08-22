# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WAN2.2 A/B for device-side uint8 video reduction.

Each round and flag state runs in a fresh subprocess so peak RSS is not polluted
by a previous allocator high-water mark. Payload size is inferred from the
returned tensor's dtype and shape; the same tensor crosses the worker boundary.
``--batch`` keeps several video payloads in flight at once and measures how the
caller-process host-memory saving scales.

    python benchmarks/diffusion/bench_device_postprocess.py --rounds 2
    python benchmarks/diffusion/bench_device_postprocess.py --batch 4 --steps 4
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("VLLM_OMNI_BENCH_OUT", "video_demo"))
MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PROMPT = (
    "A red fox walking through a snowy pine forest at sunrise, "
    "golden light through the trees, cinematic, highly detailed"
)
HEIGHT, WIDTH, FRAMES, STEPS, GUIDANCE, FPS, SEED = 480, 832, 81, 40, 4.0, 24, 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--child", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.rounds < 1 or args.batch < 1 or args.steps < 1:
        parser.error("--rounds, --batch and --steps must be positive")
    return args


def rss_mib() -> float:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


class RssSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak = 0.0

    def run(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, rss_mib())
            self.stop.wait(0.05)


def run_child(enable: bool, batch: int, steps: int) -> None:
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = Omni(
        model=MODEL,
        num_gpus=1,
        output_type="np",
        video_output_transport={"enable_device_postprocess": enable},
    )
    sampling = OmniDiffusionSamplingParams(
        output_type="np",
        seed=SEED,
        num_inference_steps=steps,
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        guidance_scale=GUIDANCE,
    )

    baseline = rss_mib()
    sampler = RssSampler()
    sampler.start()
    started = time.perf_counter()
    prompts = [{"prompt": PROMPT} for _ in range(batch)]
    outputs = engine.generate(prompts if batch > 1 else prompts[0], sampling)
    generate_s = time.perf_counter() - started
    sampler.stop.set()
    sampler.join()

    arrays: list[np.ndarray] = []
    for output in outputs:
        for video in output.images:
            array = video.numpy() if hasattr(video, "numpy") else np.asarray(video)
            if array.ndim == 5:
                arrays.extend(array[index] for index in range(array.shape[0]))
            else:
                arrays.append(array)
    if len(arrays) != batch:
        raise RuntimeError(f"expected {batch} videos, got {len(arrays)}")

    peak_rss = max(sampler.peak, rss_mib())
    engine.close()

    tag = "on" if enable else "off"
    np.save(OUTPUT_DIR / f"frames_{tag}.npy", arrays[0])

    from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes

    encode_times = []
    for _ in range(3):
        started = time.perf_counter()
        encoded = _encode_video_bytes(arrays[0], fps=FPS)
        encode_times.append((time.perf_counter() - started) * 1000)
    (OUTPUT_DIR / f"wan_{tag}.mp4").write_bytes(encoded)

    result = {
        "flag": enable,
        "batch": batch,
        "steps": steps,
        "dtype": str(arrays[0].dtype),
        "shape": list(arrays[0].shape),
        "videos": len(arrays),
        "payload_mib": round(sum(array.nbytes for array in arrays) / 1024**2, 2),
        "generate_s": round(generate_s, 2),
        "encode_ms": round(statistics.mean(encode_times), 1),
        "encode_range_ms": [round(min(encode_times), 1), round(max(encode_times), 1)],
        "peak_rss_mib": round(peak_rss, 1),
        "rss_delta_mib": round(peak_rss - baseline, 1),
        "mp4_bytes": len(encoded),
    }
    print("RESULT " + json.dumps(result), flush=True)


def run_driver(rounds: int, batch: int, steps: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for enable in (False, True):
        for round_index in range(rounds):
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    str(int(enable)),
                    "--batch",
                    str(batch),
                    "--steps",
                    str(steps),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            line = next((value for value in process.stdout.splitlines() if value.startswith("RESULT ")), None)
            if process.returncode or line is None:
                print(process.stdout[-4000:], process.stderr[-4000:])
                raise RuntimeError(f"child failed (enable={enable}, round={round_index})")
            rows.append({"round": round_index, **json.loads(line.removeprefix("RESULT "))})
            print(rows[-1], flush=True)

    with (OUTPUT_DIR / "results.json").open("w") as result_file:
        json.dump(rows, result_file, indent=2)
    if rounds > 1:
        print(f"\n{'metric':<16}{'flag off':>22}{'flag on':>22}")
        for key in ("payload_mib", "peak_rss_mib", "rss_delta_mib", "encode_ms", "generate_s"):
            cells = []
            for enable in (False, True):
                values = [row[key] for row in rows if row["flag"] is enable]
                cells.append(f"{statistics.mean(values):.1f} ({min(values):.1f}-{max(values):.1f})")
            print(f"{key:<16}{cells[0]:>22}{cells[1]:>22}")


def main() -> None:
    args = parse_args()
    if args.child is None:
        run_driver(args.rounds, args.batch, args.steps)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_child(bool(args.child), args.batch, args.steps)


if __name__ == "__main__":
    main()
