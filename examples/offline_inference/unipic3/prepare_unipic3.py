# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble a Skywork UniPic-3 checkpoint into a directory vllm-omni can load.

The teacher checkpoint (``Skywork/Unipic3``) already has the diffusers layout
expected by ``QwenImageEditPipeline``. The two distilled checkpoints do not:

* ``Skywork/Unipic3-DMD`` ships both ``transformer/`` and ``ema_transformer/``;
  the model card recommends inference from ``ema_transformer/``.
* ``Skywork/Unipic3-Consistency-Model`` ships **only** ``transformer/`` and
  ``ema_transformer/`` — no VAE, text encoder, scheduler, tokenizer, or
  processor — so it has to be composed with the teacher's non-transformer
  subfolders before vllm-omni's pipeline can load it.

This script assembles either variant into an output directory using symlinks,
so no weight bytes are copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Non-transformer subfolders that vllm-omni's ``QwenImageEditPipeline`` reads
# via ``from_pretrained(..., subfolder=...)``.
NON_TRANSFORMER_SUBFOLDERS = ("scheduler", "text_encoder", "vae", "tokenizer", "processor")
TOP_LEVEL_METADATA = ("model_index.json", "config.json")


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def _select_transformer(source: Path, use_ema: bool) -> Path:
    ema = source / "ema_transformer"
    plain = source / "transformer"
    if use_ema and ema.is_dir():
        return ema
    if plain.is_dir():
        return plain
    if ema.is_dir():
        return ema
    raise FileNotFoundError(f"No transformer/ or ema_transformer/ under {source}")


def _copy_or_link_model_index(source_index: Path, dst_index: Path) -> None:
    # ``model_index.json`` must live at the root, but it also references the
    # subfolder names — write a fresh copy so we can be sure the file exists as
    # a real file (some downstream tools open it with ``local_files_only``).
    with source_index.open() as f:
        data = json.load(f)
    with dst_index.open("w") as f:
        json.dump(data, f, indent=2)


def assemble_dmd(source: Path, output: Path, use_ema: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    transformer_src = _select_transformer(source, use_ema=use_ema)
    _link(transformer_src, output / "transformer")

    for sub in NON_TRANSFORMER_SUBFOLDERS:
        src = source / sub
        if not src.is_dir():
            raise FileNotFoundError(f"Missing '{sub}/' under {source}")
        _link(src, output / sub)

    for name in TOP_LEVEL_METADATA:
        src = source / name
        if src.is_file():
            if name == "model_index.json":
                _copy_or_link_model_index(src, output / name)
            else:
                _link(src, output / name)

    print(f"Assembled DMD variant at {output}")
    print(f"  transformer -> {transformer_src}")


def assemble_cm(source: Path, teacher: Path, output: Path, use_ema: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    transformer_src = _select_transformer(source, use_ema=use_ema)
    _link(transformer_src, output / "transformer")

    for sub in NON_TRANSFORMER_SUBFOLDERS:
        src = teacher / sub
        if not src.is_dir():
            raise FileNotFoundError(f"Missing '{sub}/' under teacher {teacher}")
        _link(src, output / sub)

    teacher_index = teacher / "model_index.json"
    if not teacher_index.is_file():
        raise FileNotFoundError(f"Missing model_index.json under teacher {teacher}")
    _copy_or_link_model_index(teacher_index, output / "model_index.json")

    # Prefer the CM's own transformer config.json so any tweaks it made to the
    # config land alongside the CM weights.
    cm_transformer_config = transformer_src / "config.json"
    if cm_transformer_config.is_file():
        shutil.copyfile(cm_transformer_config, output / "transformer_config.json")

    print(f"Assembled CM variant at {output}")
    print(f"  transformer -> {transformer_src}")
    print(f"  scheduler / vae / text_encoder / tokenizer / processor -> {teacher}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("dmd", "cm"))
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local snapshot of the distilled repo (Unipic3-DMD or Unipic3-Consistency-Model).",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="Local snapshot of Skywork/Unipic3. Required for --variant cm.",
    )
    parser.add_argument("--output", type=Path, required=True)
    ema = parser.add_mutually_exclusive_group()
    ema.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    ema.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.variant == "dmd":
        assemble_dmd(args.source, args.output, use_ema=args.use_ema)
    else:
        if args.teacher is None:
            print("--teacher is required for --variant cm", file=sys.stderr)
            return 2
        assemble_cm(args.source, args.teacher, args.output, use_ema=args.use_ema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
