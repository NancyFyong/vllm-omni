# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HuggingFace (diffusers) baseline for Lumina-Image-2.0 text-to-image.

Single-GPU reference run using diffusers ``Lumina2Pipeline``. Produces the
reference images (for quality comparison against vLLM-Omni) and the diffusers
speed baseline. One warmup pass, then timed per-prompt generation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/lumina_image2/huggingface/inference.py \
        --model-path Alpha-VLLM/Lumina-Image-2.0 \
        --prompts benchmarks/lumina_image2/prompts.json \
        --output-dir benchmarks/lumina_image2/outputs/diffusers
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import Lumina2Pipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Alpha-VLLM/Lumina-Image-2.0")
    p.add_argument("--prompts", default="benchmarks/lumina_image2/prompts.json")
    p.add_argument("--output-dir", default="benchmarks/lumina_image2/outputs/diffusers")
    p.add_argument("--output-file", default=None, help="JSON metrics path")
    args = p.parse_args()

    cfg = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    pipe = Lumina2Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    load_s = time.perf_counter() - t_load0
    mem_load = torch.cuda.max_memory_allocated() / (1024**3)

    def gen(prompt: str):
        g = torch.Generator("cuda").manual_seed(cfg["seed"])
        return pipe(
            prompt=prompt,
            negative_prompt=cfg["negative_prompt"],
            height=cfg["height"],
            width=cfg["width"],
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=cfg["guidance_scale"],
            generator=g,
        ).images[0]

    # Warmup (excluded from timing) using the first prompt.
    _ = gen(cfg["prompts"][0]["prompt"])
    torch.cuda.synchronize()

    results = []
    for item in cfg["prompts"]:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img = gen(item["prompt"])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        path = out_dir / f"{item['id']}.png"
        img.save(path)
        steps = cfg["num_inference_steps"]
        print(
            f"[diffusers] {item['id']:>10}  latency={dt * 1000:8.1f} ms  "
            f"per_step={dt / steps * 1000:6.1f} ms  peak={peak:.2f} GiB  -> {path}"
        )
        results.append(
            {"id": item["id"], "latency_ms": dt * 1000, "per_step_ms": dt / steps * 1000, "peak_gib": peak}
        )

    lat = [r["latency_ms"] for r in results]
    summary = {
        "backend": "diffusers",
        "model": args.model_path,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "steps": cfg["num_inference_steps"],
        "resolution": f"{cfg['width']}x{cfg['height']}",
        "load_s": load_s,
        "mem_after_load_gib": mem_load,
        "mean_latency_ms": sum(lat) / len(lat),
        "mean_per_step_ms": sum(lat) / len(lat) / cfg["num_inference_steps"],
        "per_prompt": results,
    }
    print("\n=== diffusers summary ===")
    print(json.dumps(summary, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(summary, indent=2))
    print("DIFFUSERS_BENCH_OK")


if __name__ == "__main__":
    main()
