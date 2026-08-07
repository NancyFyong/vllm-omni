# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM-Omni offline inference for Lumina-Image-2.0 image-to-image (SDEdit).

Same served checkpoint as the t2i benchmark. Each prompt is re-run against the
same input image at multiple ``strength`` values (0.3 / 0.6 / 0.9 by default) so
the operator can eyeball how strength trades off preservation vs. divergence
while the same numbers pin i2i latency.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/lumina_image2/vllm-omni/inference_i2i.py \\
        --input-image benchmarks/lumina_image2/outputs/vllm_omni/landscape.png \\
        --prompts benchmarks/lumina_image2/prompts.json \\
        --output-dir benchmarks/lumina_image2/outputs/vllm_omni_i2i
"""

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_image_prompt, get_model_class_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Alpha-VLLM/Lumina-Image-2.0")
    p.add_argument("--prompts", default="benchmarks/lumina_image2/prompts.json")
    p.add_argument(
        "--input-image",
        required=True,
        help="Path to a single input image, reused across all prompts and strengths.",
    )
    p.add_argument("--output-dir", default="benchmarks/lumina_image2/outputs/vllm_omni_i2i")
    p.add_argument("--output-file", default=None)
    p.add_argument(
        "--strengths",
        default="0.3,0.6,0.9",
        help="Comma-separated strengths to sweep.",
    )
    args = p.parse_args()

    cfg = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strengths = [float(s) for s in args.strengths.split(",") if s.strip()]

    input_image = Image.open(args.input_image).convert("RGB")

    t0 = time.perf_counter()
    omni = Omni(model=args.model_path)
    load_s = time.perf_counter() - t0
    mcn = get_model_class_name(args.model_path)
    steps = cfg["num_inference_steps"]

    def gen(prompt: str, strength: float):
        prompt_dict = build_image_to_image_prompt(
            model_class_name=mcn,
            prompt=prompt,
            negative_prompt=cfg["negative_prompt"],
            input_image=input_image,
            height=cfg["height"],
            width=cfg["width"],
        )
        sp = OmniDiffusionSamplingParams(
            height=cfg["height"],
            width=cfg["width"],
            seed=cfg["seed"],
            guidance_scale=cfg["guidance_scale"],
            num_inference_steps=steps,
            strength=strength,
        )
        outputs = omni.generate(prompt_dict, sampling_params_list=[sp])
        return extract_images_from_outputs(outputs)[0]

    # Warmup (excluded).
    _ = gen(cfg["prompts"][0]["prompt"], strengths[0])

    results = []
    for strength in strengths:
        for item in cfg["prompts"]:
            t1 = time.perf_counter()
            img = gen(item["prompt"], strength)
            dt = time.perf_counter() - t1
            path = out_dir / f"{item['id']}_s{strength}.png"
            img.save(path)
            print(
                f"[vllm-omni i2i] {item['id']:>10} s={strength}  latency={dt * 1000:8.1f} ms  "
                f"per_step={dt / steps * 1000:6.1f} ms  -> {path}"
            )
            results.append(
                {
                    "id": item["id"],
                    "strength": strength,
                    "latency_ms": dt * 1000,
                    "per_step_ms": dt / steps * 1000,
                }
            )

    lat = [r["latency_ms"] for r in results]
    summary = {
        "backend": "vllm-omni-offline-i2i",
        "model": args.model_path,
        "steps": steps,
        "resolution": f"{cfg['width']}x{cfg['height']}",
        "strengths": strengths,
        "load_s": load_s,
        "mean_latency_ms": sum(lat) / len(lat),
        "mean_per_step_ms": sum(lat) / len(lat) / steps,
        "per_prompt": results,
    }
    print("\n=== vllm-omni offline i2i summary ===")
    print(json.dumps(summary, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(summary, indent=2))
    print("VLLM_OMNI_BENCH_OK")


if __name__ == "__main__":
    main()
