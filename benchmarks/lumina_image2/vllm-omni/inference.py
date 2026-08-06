# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM-Omni offline inference for Lumina-Image-2.0 text-to-image.

Generates our images for the same prompts/seed/size/steps as the diffusers
baseline (for side-by-side quality comparison) plus per-prompt timing.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/lumina_image2/vllm-omni/inference.py \
        --prompts benchmarks/lumina_image2/prompts.json \
        --output-dir benchmarks/lumina_image2/outputs/vllm_omni
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_text_to_image_prompt, get_model_class_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Alpha-VLLM/Lumina-Image-2.0")
    p.add_argument("--prompts", default="benchmarks/lumina_image2/prompts.json")
    p.add_argument("--output-dir", default="benchmarks/lumina_image2/outputs/vllm_omni")
    p.add_argument("--output-file", default=None)
    args = p.parse_args()

    cfg = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    omni = Omni(model=args.model_path)
    load_s = time.perf_counter() - t0
    mcn = get_model_class_name(args.model_path)
    steps = cfg["num_inference_steps"]

    def gen(prompt: str):
        prompt_dict = build_text_to_image_prompt(
            model_class_name=mcn,
            prompt=prompt,
            negative_prompt=cfg["negative_prompt"],
            height=cfg["height"],
            width=cfg["width"],
        )
        sp = OmniDiffusionSamplingParams(
            height=cfg["height"],
            width=cfg["width"],
            seed=cfg["seed"],
            guidance_scale=cfg["guidance_scale"],
            num_inference_steps=steps,
        )
        outputs = omni.generate(prompt_dict, sampling_params_list=[sp])
        return extract_images_from_outputs(outputs)[0]

    # Warmup (excluded).
    _ = gen(cfg["prompts"][0]["prompt"])

    results = []
    for item in cfg["prompts"]:
        t1 = time.perf_counter()
        img = gen(item["prompt"])
        dt = time.perf_counter() - t1
        path = out_dir / f"{item['id']}.png"
        img.save(path)
        print(
            f"[vllm-omni] {item['id']:>10}  latency={dt * 1000:8.1f} ms  "
            f"per_step={dt / steps * 1000:6.1f} ms  -> {path}"
        )
        results.append({"id": item["id"], "latency_ms": dt * 1000, "per_step_ms": dt / steps * 1000})

    lat = [r["latency_ms"] for r in results]
    summary = {
        "backend": "vllm-omni-offline",
        "model": args.model_path,
        "steps": steps,
        "resolution": f"{cfg['width']}x{cfg['height']}",
        "load_s": load_s,
        "mean_latency_ms": sum(lat) / len(lat),
        "mean_per_step_ms": sum(lat) / len(lat) / steps,
        "per_prompt": results,
    }
    print("\n=== vllm-omni offline summary ===")
    print(json.dumps(summary, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(summary, indent=2))
    print("VLLM_OMNI_BENCH_OK")


if __name__ == "__main__":
    main()
