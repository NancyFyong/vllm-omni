# Lumina-Image-2.0 benchmark assets

Image assets for the Lumina-Image-2.0 (Next-DiT) text-to-image PR
(`feat/lumina-image2`). Hosted on this branch so the PR description and
`benchmarks/lumina_image2/README.md` can embed them via raw URLs without
committing multi-MB binaries to the code branch.

All images: 1024×1024, 30 steps, guidance 4.0, seed 42, bf16, NVIDIA H20.

- `comparison.png` — side-by-side grid, diffusers `Lumina2Pipeline` (left) vs
  vLLM-Omni (right), one row per prompt (landscape / portrait / text / art).
- `diffusers/{id}.png` — diffusers baseline outputs.
- `vllm_omni/{id}.png` — vLLM-Omni outputs (same prompts/seed/size/steps).

Outputs are visually near-identical; both correctly render the in-image
"OPEN" chalkboard text in the `text` prompt.
