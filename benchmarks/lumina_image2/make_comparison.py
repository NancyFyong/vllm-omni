# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Assemble a side-by-side comparison grid: diffusers (left) vs vLLM-Omni (right),
one row per prompt. Output is a single PNG suitable for embedding in a PR.

Usage:
    python benchmarks/lumina_image2/make_comparison.py \
        --diffusers benchmarks/lumina_image2/outputs/diffusers \
        --vllm benchmarks/lumina_image2/outputs/vllm_omni \
        --prompts benchmarks/lumina_image2/prompts.json \
        --out benchmarks/lumina_image2/outputs/comparison.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--diffusers", required=True)
    p.add_argument("--vllm", required=True)
    p.add_argument("--prompts", default="benchmarks/lumina_image2/prompts.json")
    p.add_argument("--out", default="benchmarks/lumina_image2/outputs/comparison.png")
    p.add_argument("--thumb", type=int, default=512, help="Per-image side length")
    args = p.parse_args()

    cfg = json.loads(Path(args.prompts).read_text())
    items = cfg["prompts"]
    T = args.thumb
    header_h = 48
    label_h = 34
    pad = 8

    cols = 2
    rows = len(items)
    grid_w = cols * T + (cols + 1) * pad
    grid_h = header_h + rows * (T + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    hf = _font(26)
    lf = _font(20)

    # Column headers.
    headers = ["diffusers (Lumina2Pipeline)", "vLLM-Omni (this PR)"]
    for c, text in enumerate(headers):
        x = pad + c * (T + pad)
        draw.text((x + T // 2, header_h // 2), text, fill="black", font=hf, anchor="mm")

    for r, item in enumerate(items):
        y0 = header_h + pad + r * (T + label_h + pad)
        for c, base in enumerate((args.diffusers, args.vllm)):
            x = pad + c * (T + pad)
            img_path = Path(base) / f"{item['id']}.png"
            if img_path.exists():
                im = Image.open(img_path).convert("RGB").resize((T, T))
                canvas.paste(im, (x, y0))
            else:
                draw.rectangle([x, y0, x + T, y0 + T], outline="red")
                draw.text((x + T // 2, y0 + T // 2), "MISSING", fill="red", font=lf, anchor="mm")
        # Row label spanning both columns.
        draw.text(
            (grid_w // 2, y0 + T + label_h // 2),
            f"{item['id']}: {item['prompt'][:70]}",
            fill="black",
            font=lf,
            anchor="mm",
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"Saved comparison grid -> {args.out} ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()
