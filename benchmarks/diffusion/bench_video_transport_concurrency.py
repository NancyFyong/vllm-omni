# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure device-side video reduction with concurrent HTTP clients.

Each (flag, concurrency) case starts a fresh server and samples aggregate VmRSS
across its process tree. This avoids carrying allocator state from one case into
the next. The synchronous video endpoint may serialize work; wall time and peak
RSS reveal whether requests actually overlap.

    python benchmarks/diffusion/bench_video_transport_concurrency.py
    python benchmarks/diffusion/bench_video_transport_concurrency.py --levels 1,2,4 --steps 4
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import requests


class RequestResult(TypedDict):
    bytes: int
    latency_s: float


class BenchmarkResult(TypedDict):
    flag: bool
    concurrency: int
    baseline_tree_rss_mib: float
    peak_tree_rss_mib: float
    growth_mib: float
    wall_s: float
    max_latency_s: float
    mp4_bytes: int


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("VLLM_OMNI_BENCH_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
PROMPT = "A red fox walking through a snowy pine forest at sunrise, cinematic"
HEIGHT, WIDTH, FRAMES, GUIDANCE, SEED = 480, 832, 81, 4.0, 1234
STARTUP_TIMEOUT_S = 900.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="1,2,4", help="Comma-separated concurrent client counts")
    parser.add_argument("--steps", type=int, default=4)
    return parser.parse_args()


def new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def tree_rss_mib(root_pid: int) -> float:
    """Sum VmRSS over the server and every descendant."""
    pids, total = [root_pid], 0.0
    seen: set[int] = set()
    while pids:
        pid = pids.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/status") as status:
                for line in status:
                    if line.startswith("VmRSS:"):
                        total += float(line.split()[1]) / 1024.0
                        break
            with open(f"/proc/{pid}/task/{pid}/children") as children:
                pids.extend(int(value) for value in children.read().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


class TreeSampler(threading.Thread):
    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid = pid
        self.stop = threading.Event()
        self.peak = 0.0

    def run(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, tree_rss_mib(self.pid))
            self.stop.wait(0.05)


def wait_healthy(port: int, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.time() + STARTUP_TIMEOUT_S
    with new_session() as session:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}; see {log_path}")
            try:
                if session.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    raise TimeoutError(f"server did not become healthy; see {log_path}")


def one_request(port: int, steps: int, timeout: float) -> RequestResult:
    form = {
        "model": MODEL,
        "prompt": PROMPT,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "num_frames": str(FRAMES),
        "num_inference_steps": str(steps),
        "guidance_scale": str(GUIDANCE),
        "seed": str(SEED),
    }
    fields = {key: (None, value) for key, value in form.items()}
    started = time.perf_counter()
    with new_session() as session:
        response = session.post(f"http://127.0.0.1:{port}/v1/videos/sync", files=fields, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"video request failed ({response.status_code}): {response.text[:500]}")
    return {"bytes": len(response.content), "latency_s": round(time.perf_counter() - started, 2)}


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_case(enable: bool, concurrency: int, steps: int, output_dir: Path) -> BenchmarkResult:
    port = free_port()
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        MODEL,
        "--omni",
        "--port",
        str(port),
        "--video-output-transport",
        json.dumps({"enable_device_postprocess": enable}),
    ]
    env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log_path = output_dir / f"server_{int(enable)}_c{concurrency}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            wait_healthy(port, proc, log_path)
            baseline = tree_rss_mib(proc.pid)
            sampler = TreeSampler(proc.pid)
            sampler.start()
            started = time.perf_counter()
            try:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(lambda _: one_request(port, steps, 1800), range(concurrency)))
            finally:
                sampler.stop.set()
                sampler.join()
            wall = time.perf_counter() - started
            peak = max(sampler.peak, baseline)
            row: BenchmarkResult = {
                "flag": enable,
                "concurrency": concurrency,
                "baseline_tree_rss_mib": round(baseline, 1),
                "peak_tree_rss_mib": round(peak, 1),
                "growth_mib": round(peak - baseline, 1),
                "wall_s": round(wall, 2),
                "max_latency_s": max(float(result["latency_s"]) for result in results),
                "mp4_bytes": int(results[0]["bytes"]),
            }
            print(row, flush=True)
            return row
        finally:
            stop_server(proc)


def main() -> None:
    args = parse_args()
    levels = [int(value) for value in args.levels.split(",")]
    if not levels or any(value < 1 for value in levels):
        raise ValueError("--levels must contain positive integers")

    output_dir = Path(os.environ.get("VLLM_OMNI_BENCH_OUT", "video_demo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkResult] = []
    for concurrency in levels:
        for enable in (False, True):
            rows.append(run_case(enable, concurrency, args.steps, output_dir))
    with (output_dir / "concurrency.json").open("w") as result_file:
        json.dump(rows, result_file, indent=2)

    print(f"\n{'concurrency':<13}{'off peak':>14}{'on peak':>14}{'difference':>14}{'off/on wall':>20}")
    for concurrency in levels:
        off = next(row for row in rows if row["concurrency"] == concurrency and not row["flag"])
        on = next(row for row in rows if row["concurrency"] == concurrency and row["flag"])
        saved = off["peak_tree_rss_mib"] - on["peak_tree_rss_mib"]
        wall = f"{off['wall_s']:.1f}/{on['wall_s']:.1f}s"
        print(
            f"{concurrency:<13}{off['peak_tree_rss_mib']:>11.1f} MiB"
            f"{on['peak_tree_rss_mib']:>11.1f} MiB{saved:>11.1f} MiB{wall:>20}"
        )


if __name__ == "__main__":
    main()
