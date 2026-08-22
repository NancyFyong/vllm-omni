"""WAN2.2 A/B for the device-side uint8 reduction: real video, payload, memory, time.

Runs the engine twice with the same seed, once per flag state, and reports the Hop 1
payload, the consuming process peak RSS, MP4 encode time and generate time. Writes both
MP4s so the frames can be compared by eye.

    python benchmarks/diffusion/bench_device_postprocess.py
"""

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

OUT = os.environ.get("VLLM_OMNI_BENCH_OUT", "video_demo")
MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PROMPT = (
    "A red fox walking through a snowy pine forest at sunrise, "
    "golden light through the trees, cinematic, highly detailed"
)
# Native aspect from the repo preset for wan; 480p keeps two full runs affordable.
HEIGHT, WIDTH, FRAMES, STEPS, GUIDANCE, FPS = 480, 832, 81, 40, 4.0, 24
SEED = 1234


def rss_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def gpu_used_mib():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


class Sampler(threading.Thread):
    """Peak RSS/GPU while work runs. ru_maxrss is a process high-water mark
    already inflated by importing torch, so it shows no difference."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak_rss = 0.0
        self.peak_gpu = 0.0

    def run(self):
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, rss_mib())
            self.peak_gpu = max(self.peak_gpu, gpu_used_mib())
            self.stop.wait(0.05)


def child(enable):
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = Omni(
        model=MODEL, num_gpus=1, output_type="np", video_output_transport={"enable_device_postprocess": enable}
    )
    sp = OmniDiffusionSamplingParams(
        output_type="np",
        seed=SEED,
        num_inference_steps=STEPS,
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        guidance_scale=GUIDANCE,
    )

    base_rss, gpu_before = rss_mib(), gpu_used_mib()
    sampler = Sampler()
    sampler.start()
    t0 = time.perf_counter()
    outputs = engine.generate({"prompt": PROMPT}, sp)
    elapsed = time.perf_counter() - t0
    sampler.stop.set()
    sampler.join()

    video = outputs[0].images[0]
    arr = video.numpy() if hasattr(video, "numpy") else np.asarray(video)
    if arr.ndim == 5:
        # The server splits a batched payload before encoding; do the same.
        arr = arr[0]
    peak_rss = max(sampler.peak_rss, rss_mib())
    engine.close()

    tag = "on" if enable else "off"
    np.save(f"{OUT}/frames_{tag}.npy", arr)

    from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes

    times = []
    for _ in range(3):
        t1 = time.perf_counter()
        mp4 = _encode_video_bytes(arr, fps=FPS)
        times.append((time.perf_counter() - t1) * 1000)
    with open(f"{OUT}/wan_{tag}.mp4", "wb") as f:
        f.write(mp4)

    print(
        "RESULT "
        + json.dumps(
            {
                "flag": enable,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "payload_mib": arr.nbytes / 1024**2,
                "generate_s": round(elapsed, 2),
                "encode_ms": round(min(times), 1),
                "peak_rss_mib": round(peak_rss, 1),
                "rss_delta_mib": round(peak_rss - base_rss, 1),
                "gpu_peak_mib": sampler.peak_gpu,
                "gpu_before_mib": gpu_before,
                "mp4_bytes": len(mp4),
            }
        ),
        flush=True,
    )


def driver():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for enable in (False, True):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", str(int(enable))],
            capture_output=True,
            text=True,
            env=env,
        )
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
        if not line:
            print(p.stdout[-4000:], p.stderr[-4000:])
            raise SystemExit(f"child failed (enable={enable})")
        rows.append(json.loads(line[len("RESULT ") :]))
        print(rows[-1], flush=True)
    json.dump(rows, open(f"{OUT}/results.json", "w"), indent=2)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(bool(int(sys.argv[sys.argv.index("--child") + 1])))
    else:
        driver()
