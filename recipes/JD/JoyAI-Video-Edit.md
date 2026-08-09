# JoyAI-Video-Edit

> Instruction-driven video editing, offline fixed-length (720x1248, 2 denoise steps)

## Summary

- Vendor: JD (Joy Future Academy)
- Model: [`jdopensource/JoyAI-Video-Edit`](https://huggingface.co/jdopensource/JoyAI-Video-Edit) — a
  16.2B dual-stream MMDiT with a causal video VAE, conditioned by
  [`XiaomiMiMo/MiMo-VL-7B-RL-2508`](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508)
- Task: Video-to-video editing — given a source clip and a text instruction, produce an edited clip
  that preserves the source's motion and layout
- Mode: Offline batch (`Omni.generate`). The model is natively streaming-capable; this integration
  covers fixed-length offline editing only
- Maintainer: Community

## When to use this recipe

Use this to run a single-GPU offline video edit. The model is AR-DMD-distilled to **two** denoise
steps and generates chunk-autoregressively — one latent frame per rollout step, each attending to a
bounded window over the chunks already generated — so cost is linear in clip length at flat memory. A
145-frame clip needs the same VRAM as a 17-frame one.

Do **not** reach for this recipe to edit a single image (see the image-edit models instead), to extend
a clip (this model cannot generate beyond the source's length), or expecting any of the diffusion
accelerations — none of them apply, see "Known limitations".

## References

- Upstream: <https://github.com/jd-opensource/JoyAI-Video-Edit>
- Example: [`examples/offline_inference/video_to_video/`](../../examples/offline_inference/video_to_video/)
- Deploy config: [`vllm_omni/deploy/joyai_video_edit.yaml`](../../vllm_omni/deploy/joyai_video_edit.yaml)

## Hardware Support

## GPU

### 1x H20 96GB

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: CUDA 13.0, torch 2.11.0+cu130
- vLLM version: from your current checkout
- vLLM-Omni version or commit: from your current checkout
- Extra dependency: `av` (PyAV), for reading and writing video

#### Command

```bash
# 1. Fetch the weights (~47 GiB across two Hugging Face repos) and synthesize model_index.json.
python examples/offline_inference/video_to_video/download_joyai_video_edit.py \
    --output-dir /path/to/models/joyai_video_edit

# 2. Run one edit. --model must be the local directory: the DiT ships as a raw .pth rather than a
#    diffusers subfolder, so a repo id will not resolve.
python examples/offline_inference/video_to_video/video_to_video.py \
    --model /path/to/models/joyai_video_edit \
    --video input.mp4 \
    --prompt "make it snow" \
    --num-frames 73 \
    --output edited.mp4
```

#### Verification

```bash
JOYAI_VIDEO_EDIT_MODEL_DIR=/path/to/models/joyai_video_edit \
    pytest -s -v tests/e2e/offline_inference/test_joyai_video_edit.py --run-level=advanced_model
```

Two tests, nine frames each: one asserts the payload envelope, geometry and dtype; one asserts that
two identical requests are served independently and reproducibly. About 3 minutes, most of it the two
weight loads. Both skip rather than fail when `JOYAI_VIDEO_EDIT_MODEL_DIR` is unset.

#### Fidelity against upstream

Checked by running upstream's own `deploy/xvideo` code on the same box, weights, seed and raw prompts
as this port, over the five clips in upstream's README, then comparing at two levels: **per frame** on
decoded pixels, and **tensor by tensor** at every seam of the pipeline. Two deviations were needed to
run upstream here at all and neither is on the denoise path: flash-attn-4 is sm100-only so upstream
fell back to its own SDPA path, and its JPEG websocket packer was bypassed to keep raw pixels. FP8 was
off on both sides. Upstream's fused `joyomni_ops` CUDA kernels **were** built and active in the
reference env, so this compares against upstream's real production numerics — the native-op
replacements in `joyai_video_edit_transformer.py` are measured against the fused kernels they replace,
not against a pure-torch fallback of upstream's own.

Decoded pixels alone cannot localise anything — every difference in the pipeline arrives at the frame
summed. So the primary instrument is `dump_ours.py` / `dump_upstream.py`, which tap the same named
tensors on both sides and diff them in dataflow order, and the numbers below are quoted as relative
L2 with cosine alongside. Three things that instrument needs in order not to lie:

- **Its own floor must be measured first**, per subsystem, or every reading is uninterpretable. Running
  the DiT's chunk-0 forward twice with identical kwargs inside one process gives **0.000e+00** — bf16
  matmul and flash attention are deterministic here, so the DiT's floor is zero, not "some small
  number". The MiMo-VL vision tower is likewise **0.000e+00** across the two transformers versions.
  The VAE's is **not** zero: upstream encoding the same window twice within one run differs by
  3.527e-03 on the *sampled* latent, because the posterior draw is not reproducible (see below). The
  deterministic posterior mean has no such floor and is the row to read.
- **Upstream's indices are not ours.** `JoyOmniRuntime.load` warms up 8 chunks in each of two
  orientations before the first real frame, so upstream's dump holds 37 `post_mean_*` where ours holds
  2. Pairing index-for-index reports rel 2.5 at cosine **-0.18** — which reads as catastrophic and is
  purely a pairing failure. `diff_dumps.py` therefore matches by content and prints the runner-up
  distance so an ambiguous pairing is flagged rather than trusted.
- **A per-seam error curve cannot separate "this block computes something else" from "this block
  faithfully amplifies what it was handed."** Substituting each block's inputs with upstream's own
  previous-block outputs makes every block start bit-identical, so the error it then shows is the error
  it *created*.

**Defects found and fixed.** Each produced entirely plausible edited video, and each is stated with the
tensor-level evidence that identified it, because pixel dB cannot apportion credit among them:

| defect | what it was | evidence |
|---|---|---|
| source encode shared the sampler's generator | one RNG stream where upstream has two | `post_mean_*`, guarded in `resolve_generators` |
| resize used torch's bicubic, not PIL's | different kernel, same name | guarded in `_resize_video_uint8` |
| rotary tables built in fp32 and cast once | upstream builds *and* consumes them in bf16 | 48x on `attn00_q`; `attn00_v` unchanged as control |
| `torch.autocast` contexts absent | upstream wraps the VAE and each denoise step | ~2x on every VAE row (table below) |

Both of the first two are guarded by tests in
`tests/diffusion/models/joyai_video_edit/test_pipeline_helpers.py`.

**The reference is not reproducible, and that sets the ceiling.** Before any pixel dB can be read, the
reference has to be measured against *itself*: two upstream runs, same seed, same box, same weights,
same code, same warmup. They are **not** bit-identical — on case01, 77.4% of pixels differ, by a mean
of 4.6 and a maximum of 255 uint8 levels. Upstream draws its VAE posterior `eps` from the *global* RNG,
which its own model construction plus a 16-chunk warmup has already advanced, so that draw cannot be
reproduced even by upstream. Nor is it only a matter of the starting offset. **Two different daemon
threads draw from that one global generator concurrently**: the source-window posterior sample at
`pipeline.py:315` (`.latent_dist.sample()`, no `generator=`) runs on `joyomni-vae-encode`, and the
pseudo-latent posterior sample at `joyomni_streaming.py:1415` runs on `joyomni-pseudo-encode`. Only the
DiT is serialised (`runtime.dit_lock`), so which of those two reaches the generator first varies with
thread scheduling, and the draw *order* changes run to run. Note that the per-chunk denoise noise is
**not** part of this — `joyomni_streaming.py:705` passes `generator=self.generator`, a dedicated seeded
one. So seeding the global RNG would not buy reproducibility either; the fix is a per-stage generator at
those two sites, which is a design change rather than a missing `manual_seed`. This port gets the
property for free by running the stages in series, which is the same choice that costs it the 18% (see
below): the overlap and the reproducibility are two views of one decision, not independent knobs.
Every figure below is therefore quoted as a **gap to that ceiling**; a raw
ours-vs-upstream dB means nothing on its own.

| | frame 0 | | | full clip | | | end of clip |
|---|---|---|---|---|---|---|---|
| | earlier | **now** | *ceiling* | earlier | **now** | *ceiling* | gap now |
| case01 | 18.44 | **25.90** | *28.27* | 18.77 | **23.70** | *27.55* | **1.69** |
| case02 | 26.79 | **38.45** | *45.20* | 16.70 | **20.98** | *21.30* | **−0.17** |
| case03 | 26.84 | **26.59** | *35.06* | 12.28 | **13.15** | *14.24* | **1.19** |
| case04 | 24.78 | **36.13** | *40.79* | 24.73 | **28.92** | *32.74* | **5.33** |
| case05 | 27.54 | **36.73** | *44.28* | 22.77 | **25.90** | *30.82* | **1.85** |

All dB, all against one fixed `upstream/` run; the ceiling column is `upstream/` against a second
upstream run of the same case. The last column is the mean gap over the final eight frames, and it is
the one to read for whether the rollout is stable — see below. Reproduce with
`ceiling_table.py ours ours_aligned`.

The full-clip ceiling varies enormously by case — 14.24 dB on case03 against 32.74 dB on case04 — and
that spread is the whole argument for quoting gaps. case03's raw **13.15 dB looks like a failure and is
not one**: it sits 1.10 dB under a ceiling of 14.24 dB, because case03 is the case whose rollout most
amplifies upstream's own unreproducible posterior draw. Read raw, case03 would have been the worst
result in the table; read against its ceiling it is the second best.

Frame 0 is the cleanest probe available — chunk 0 is a single latent frame with an empty KV window, so
it exercises the cold-start path with nothing accumulated behind it. The two columns then separate the
two failure modes an aggregate number conflates: flat-low from frame 0 means the runs were never doing
the same thing, while high-at-frame-0 decaying with index means chunk 0 agrees and the autoregressive
rollout is compounding a bf16-level difference it is fed back.

**case03 is the one case still unexplained**, and is called out rather than averaged away: its frame 0
did not improve (26.84 → 26.59 dB) and remains 8.47 dB under its ceiling, where the other four moved to
within 2.4–7.6 dB. Its `vs_source` distance is also the one that does not match upstream's — 11.74 dB
against upstream's 10.17 dB, meaning our edit on that clip is *weaker* than upstream's, not merely
different. Whatever that is, it is not any of the four defects fixed above.

Two notes on reading this table honestly. The earlier-snapshot column is a *snapshot*, not a per-fix
attribution — several fixes separate it from the current column and the pixel dB cannot apportion credit
among them, which is why each fix is justified at the tensor level above instead. And an earlier
version of this document carried a ladder of frame-0 numbers measured against a *different draw* of the
reference; once the reference was known to move, those rows were withdrawn rather than restated. The
`eps`-perturbation simulation they were paired with, though, turned out to be almost exactly right —
it predicted a 28.9 / 40.0 dB ceiling where the measured one is 28.27 / 40.79.

The rotary fix is worth its own note because it is invisible to any shape or dtype assertion: the
tables were dtype-correct and the port was self-consistent, but bf16 `cos`/`sin` carry ~3 decimal
digits, and the defect announced itself as a **median of 2 ULPs** on `q`. Not as a max — `max|ULP|`
was ~30000, which is sign flips on near-zero elements and says nothing. `attn00_v` is the control: it
does not pass through rotary and did not move.

Autocast is the one fix that was not obvious *a priori*: with bf16 weights fed bf16 activations, every
op autocast would cast is already running bf16, so the context can be a complete no-op. It had to be
measured. It was not a no-op — the VAE improved roughly 2x throughout, consistent with something
inside the VAE handing fp32 to a conv that ran fp32 here and bf16 upstream:

| | before | after |
|---|---|---|
| `post_mean_00` (deterministic posterior mean) | 3.055e-03 | **1.216e-03** |
| `post_std_00` | 1.806e-02 | 8.647e-03 |
| `src_latent_00` | 6.386e-03 | 3.312e-03 |
| `src_latent_01` | 5.867e-03 | 4.628e-03 |
| `dit_in_ref_video_latent` | 7.477e-03 | 5.966e-03 |

Note the shape of upstream's autocast topology, because it is asymmetric and load-bearing: the context
is entered **per timestep inside** the denoise loop and closes around `scheduler.step` too, while the
third forward — the KV store — sits **outside** it; VAE encode is wrapped around the whole
sliding-window loop including normalization, while decode wraps only `decode`, with denormalization
outside. A port that hoisted one context around all three forwards would run that pass under different
rules than upstream. `_autocast` mirrors it seam for seam.

**The DiT is clean.** With every DiT input overwritten by upstream's own — so the forward starts
bit-identical — the per-block created error is flat across all 40 blocks: worst image-stream block is
27 at 2.224e-03, worst text-stream block is 29 at 3.923e-04, cosine 1.000000 everywhere. No block is
defective; there is no op to fix. Separately, `sdpa_attention` at these shapes was confirmed
**bit-identical to upstream's flash-attn-4** (0.000e+00), with mem-efficient at 1.231e-03 and math at
1.364e-03 — so the native-ops substitution is exact on the attention path, not merely close.

**What remains open, and why it is not fixable from here.** The condition encoder is the dominant term:
`embeds` differs by 1.614e-01 at cosine 0.986991, and substituting upstream's `embeds` while holding
everything else fixed drops `dit_out` from 9.859e-02 to 3.473e-02 — so roughly 65% of the DiT's output
error enters through that one input. Attributing it: the vision tower is bit-identical, `vblock00` is
bit-identical, and `vblock01` *creates* 1.83e-05 from that identical input, which then amplifies
~1.28x per block to 3.9e-02 by block 31. Two candidate causes were tested and both closed negative —
`allow_bf16_reduced_precision_reduction=False` changes nothing at any stage (0.000e+00 against the
unpinned run), and both sides' unpinned SDPA default already equals pinned `FLASH_ATTENTION`
bit-exactly, with `EFFICIENT_ATTENTION` having no kernel at these shapes on either side and `math`
pinned identically on both sides being *worse* (2.323e-04). What is left lives inside the flash and
GEMM kernels themselves, cu128 under upstream's torch 2.9.1 against cu130 under our 2.11.0. That is
not reachable from Python without pinning torch, so it is recorded rather than fixed.

The VAE posterior draw is the other irreducible term, and it is the same mechanism as the ceiling above.
It is why `post_mean_*` rather than `src_latent_*` is the row that answers "did the video enter the VAE
correctly" — the mean is deterministic, so the unmatchable term is excluded by construction rather than
argued away.

**The rollout is stable, but not because the per-chunk curves are flat — they are not.** Ours decay
(case02 falls 31.97 → 17.42 dB over the clip), and an earlier version of this document wrongly claimed
otherwise. Decay is *expected*: each chunk conditions on the previous chunk's own output, so the rollout
amplifies whatever chunk 0 started with — and it amplifies upstream's unreproducible posterior draw
exactly as readily as a kernel difference. Upstream's own two runs decay hardest of all, by 26.29 dB on
case02 and 21–22 dB on case03/case05.

So decay alone is not evidence of a defect, and neither is a smaller decay evidence of health — a run
starting nearer the noise floor has less room left to fall, which is exactly our position. The
unconfounded test is the **end-of-clip gap** in the table above: over the last eight frames this port
sits **1.19–1.85 dB** under the ceiling on case01/03/05 and **inside** it on case02 (−0.17 dB). On four
of five cases the gap *narrows* across the rollout (case02 6.76 → −0.17, case03 8.47 → 1.19, case05
7.56 → 1.85, case01 2.37 → 1.69); only case04 widens, by 0.66 dB. The rollout therefore does not
systematically amplify our divergence faster than the reference amplifies its own — which is the check
that would have caught a wrong pre-RoPE offset in the KV window, and which a bare per-chunk curve,
having no scale to be judged against, cannot perform. `clean_latents` is the one tap that cannot be
checked at all — upstream's dump has no counterpart key.

#### Why neither run reproduces the published demos

Worth stating separately, because it looks like a port defect and is not one: **neither this port nor
upstream's own code, run here on upstream's own weights, reproduces the videos in upstream's README from
the prompts printed next to them.** Those prompts are not the prompts that produced those videos.
`static/index.html:225` ships the WebUI's prompt-enhancement checkbox **checked**, and
`DEPLOYMENT.md:102` says what happens without an endpoint configured: *"If these variables are not set,
prompt enhancement falls back to the raw user prompt."* So the documented path rewrites every
instruction through `deploy/xvideo/serving/pe.py` — an "elite AI Video-to-Video Prompt Architect"
(`DEFAULT_MODEL = "Qwen2.5-VL-72B"`) conditioned on a downscaled source frame — while any comparison
that drives the pipeline directly, as every measurement above does, silently uses the short string.

Driving upstream's `SYSTEM_PROMPT` and `V2V_TEMPLATE` unmodified against a local vision-language
endpoint expands the five README instructions by 16-26x (44 chars to 1125, 88 to 1512), into
per-subject garment lists grounded in the actual footage. Re-running the five clips on those prompts,
scored as PSNR against each clip's own source at a common 832x480 — *lower is a stronger edit*, and both
sides reach 832x480 by the same kind of reduction from 1248x720:

| case | published | ours, raw prompt | upstream, raw prompt | ours, enhanced prompt |
| --- | --- | --- | --- | --- |
| case01 | 12.94 | 17.96 | 17.90 | **13.02** |
| case02 | 10.13 | 8.80 | 8.81 | 8.68 |
| case03 | 10.97 | 11.75 | 10.17 | 11.36 |
| case04 | 19.24 | 20.32 | 20.40 | 20.22 |
| case05 | 17.63 | 17.07 | 17.19 | **17.62** |

Two readings, and the second is the one that matters. First, **ours and upstream on the same raw prompt
agree within 0.13 dB on four of five clips** — an end-to-end check that is independent of the
tensor-level work above. Second, case01 is the only clip whose published result sits far from both raw
runs, and enhancing the prompt closes 5.02 dB of that distance to **0.08**; mean absolute distance to
published falls from 1.75 dB to 0.58 dB across all five. Frame by frame the raw-prompt run restyles only
the room while leaving the diners in modern clothes, and the enhanced-prompt run puts them in wigs,
velvet and lace as the published clip does. Where published was already in line with the raw runs
(case02, case04) enhancement barely moves anything, which is the behaviour that makes the first result
believable rather than a knob tuned until it agreed.

This establishes the mechanism, not the exact published prompts: a locally-served 7B stood in for the
72B architect, and its reply needed a `<think>` block stripped before use. That stripping is itself
evidence about upstream's intent — `_sanitize_enhanced` (`pe.py:137`) removes markdown links and URLs
and has no notion of `<think>`, so upstream expects a non-reasoning model returning the paragraph alone.

The practical consequence for this recipe: prompt enhancement is a property of upstream's **serving
layer**, not of the model, and this port has no equivalent — it edits according to the prompt it is
handed. Short instructions therefore under-edit relative to the demos, and that is the prompt's doing,
not the pipeline's.

**Running PE through upstream's own server then showed the enhancement often cannot arrive intact, and
the reason is upstream's truncation rather than the prompt.** The condition encoder tokenizes with
`max_length=max_sequence_length + drop_idx` and `truncation=True` (`pipeline.py:90-92`), and the
tokenizer's `truncation_side` is HF's default `right` — so what survives is the **head** of the prompt
and the tail is discarded. For the `video` template the streaming path uses (`joyomni_streaming.py:579`)
the prefix is 94 tokens and `drop_idx` is 91, so `max_length` is 1115 and 1024 content tokens survive.
Nothing bounds the reply against that: `_sanitize_enhanced` (`pe.py:137`) strips markdown links and URLs
and neither shortens the reply nor warns. Measured by `pe_prompt_truncation.py` against the real
tokenizer, on the server's own recorded `prompts.json`:

| served reply | chars | tokens | dropped | what the DiT is conditioned on |
| --- | --- | --- | --- | --- |
| case01, three attempts | 6505 / 10635 / 6588 | 1363 / 2189 / 1385 | **347 / 1173 / 369** | reasoning only, cut mid-sentence |
| case04 | 4962 | 1005 | 0 | the whole reply, with 11 tokens to spare |
| case02 | 3805 | 767 | 0 | the whole enhanced prompt |
| case03 | 3116 | 642 | 0 | the whole enhanced prompt |
| raw prompt, for scale | 88 | 16 | 0 | the instruction, whole |

So the loss is length-dependent, and that is what makes it more than a curiosity: **the one clip whose
published result the raw prompt could not reach is the one clip whose enhanced prompt is too long to
survive.** case01 needed 5.02 dB of enhancement in the table above, and all three of its served replies
lose their tail — where the instruction lives. Its `<think>` block alone is 1055 tokens against 331 for
the edit instruction that follows, so the surviving window opens `<think>` and never reaches `</think>`:
the DiT sees the model deliberating ("*…the table setting should stay? Wait, no—wait…*") and never sees
"replace the interior with stone castle walls". case02 and case03 got shorter replies and were unharmed,
and case04 cleared the budget by 11 tokens — the threshold is not comfortably far from what this endpoint
routinely produces, so which clips are damaged is close to arbitrary.

**Surviving the truncation is not the same as arriving usefully, and this is the part that compounds.**
`_sanitize_enhanced` removes markdown links and URLs and leaves the reasoning block untouched, so what
the DiT is conditioned on is 65-87% deliberation in every one of the six served replies (4848 / 9213 /
4978 / 2490 / 2161 / 3654 chars of `<think>` against totals of 6505 / 10635 / 6588 / 3805 / 3116 / 4962).
Even the untruncated ones therefore spend most of their 1024-token budget on the model debating with
itself, with the actual instruction — 1315-1622 chars — at the tail. That ordering is what makes the two
faults multiply rather than merely coexist: reasoning comes *first* and is the bulk of the reply, so
right-truncation does not remove an arbitrary tail, it deterministically keeps the deliberation and drops
the answer. A `truncation_side='left'` would have been wrong for a template whose system prompt leads,
but stripping `<think>…</think>` in `_sanitize_enhanced` would have cost nothing and fixed both.

Two faults are stacked here and only one is unconditional. The right-truncation is upstream's and holds
for any endpoint: any reply over 1024 content tokens loses its tail silently. That the lost tail is *the
whole instruction* is the endpoint's contribution — the only vision-language model available locally is
the `MiMo-VL-7B-RL-2508` that ships with this very model, and it is a reasoning model, whereas upstream's
`DEFAULT_MODEL = "Qwen2.5-VL-72B"` would answer in a paragraph and emit no `<think>` for the sanitizer to
miss. The stripped replies used for the *ours,
enhanced prompt* column above ran 1125-1512 chars, roughly 320 tokens, and so were never truncated —
which is why that column moved case01 to within 0.08 dB of published while the served path cannot. This
does not show the published demos were made with a truncated prompt; it shows that the deployment
upstream documents, on the endpoint its own weights provide, cannot deliver a long enhanced prompt to the
model at all.

**Running all five clips through that path confirms it, and the decisive evidence is not a dB number.**
Each clip was driven over `/ws` with `use_pe: true` by `pe_serve_client.py`, so the server did its own
enhancement; `edit_strength.py` then scored every arm against its source at a common 832x480:

| case | published | ours (raw) | upstream (raw) | ours (PE, intact) | **upstream server (PE)** |
| --- | --- | --- | --- | --- | --- |
| case01 | 12.94 | 17.96 | 17.90 | 13.02 | **17.72** |
| case02 | 10.13 | 8.80 | 8.81 | 8.68 | 9.32 |
| case03 | 10.97 | 11.75 | 10.17 | 11.36 | 11.87 |
| case04 | 19.24 | 20.32 | 20.40 | 20.22 | 19.96 |
| case05 | 17.63 | 17.07 | 17.19 | 17.62 | 16.64 |

On case01, enhancement is worth 5.02 dB when it arrives intact (17.96 → 13.02, landing 0.08 dB from
published). Through upstream's own server it is worth **0.24 dB — about 5% of its effect.** Truncation
does not weaken the enhancement, it erases it. The encode asymmetry runs against that reading rather than
for it: case01's served artifact is the recorder's own encode where the other four are CRF-8 from
`/download_last`, and a lossier encode deviates further from its source, pushing case01's dB *down*
toward looking edited — so 17.72 is generous to the served run.

The frames say the same thing far more specifically than the scalar can, and this is the part worth
trusting. The served output *does* change the interior — stone walls, a wrought-iron chandelier, warm
candlelight — and leaves the people entirely modern: the man stays in a polo shirt, the women in modern
clothes, nobody in period dress or a wig, where both the intact-prompt run and the published demo
transform all three people. The instruction was "transform the people, hairstyles, **and interior**", and
the served run did only the last of the three. That is exactly what the cut predicts. The surviving window
ends mid-sentence on interior description — "*wrought-iron chandeliers emitting warm candlelight, walls
lined with heraldic tapestries … ambient hues of deep burgundy and olive green*" — and the first sentence
*dropped* is the answer itself: "*Replace the people's casual modern clothing with 18th-19th century
British aristocratic attire: the man with a tailored tailcoat and powdered wig …*". So what survived the
truncation is what the output changed, and what was dropped names precisely what it failed to change.
Because that argument runs from prompt text to visible content, it does not rest on decoded-pixel dB at
all; the table above only corroborates it.

**case02-05 gain nothing from the served enhancement, and case03 is visibly worse for it.** On dB they
scatter 0.3-1.0 dB either side of their raw runs in both directions, which on its own would read as noise.
The frames are more specific. case03 asks to "make all dogs white, add colorful hats, and turn the
sunglasses hot pink"; the raw-prompt run performs all three and lands closest to published, while the
served run leaves several dogs brown and renders *every* hat pink instead of colorful — it followed the
instruction less completely, not more. Its prompt was never truncated (642 tokens, well inside the
budget), so truncation cannot explain it; what remains is the dilution described above, 69% of that
reply being `<think>` deliberation competing for the same 1024 tokens as three concrete edits. Treat that
as consistent with the dilution rather than as proof of it: this is one clip at one seed, and the served
arm differs from our raw arm in implementation as well as in prompt, so the two causes are not separated.

Either way the direction is settled for these four, and the reason they were never the interesting cases
is in the raw column: read it against published — 8.80 vs 10.13, 11.75 vs 10.97, 20.32 vs 19.24,
17.07 vs 17.63 — and all four already match or exceed published edit strength on the raw README prompt.
There was no gap for enhancement to close. So the generalisation the data supports is narrower and
stronger than "enhancement fixes the gap": **case01 is the only clip whose raw prompt underdelivers, and
it is the only clip whose enhanced prompt is too long to survive.** A prediction that case02-05 would move
toward their enhanced runs was made before these runs and did not hold; it was a bad prediction, because
it assumed enhancement should help where nothing was missing.

One further obstacle to taking "just run it upstream's way" as the baseline: **upstream's documented
default does not start on sm90.** `run_server.sh:36` defaults `JOYOMNI_FP8_IMG=1`, and that is not a
fast path that degrades — `dit.py:321` → `fp8_linear.py:54` raises `AssertionError: joyomni_ops fp8 ops
not available; cannot build Fp8Linear` on the first block of the first forward, because the vendored
extension is not built here. The failure mode is worse than a failed request: the startup warmup catches
it (`full-pipeline warmup skipped/failed: RuntimeError("async streaming worker failed: ...")`) and the
process then never binds its port — observed sleeping with 828 threads and no listener 15 minutes later.
`JOYOMNI_FP8_IMG=0` is required, so this port's lack of FP8 GEMMs costs nothing that upstream actually
delivers on this hardware. (FA4 is the same story and is already handled: it logs
`flash-attn-4 cute backend requires sm100; this GPU is sm90` and falls back to SDPA, which measured
identical to 0.000e+00.) The launch used here is recorded in `launch_upstream_server.sh` alongside the
measurement scripts.

Two further obstacles turned up in the served path itself, both costing a whole session rather than a
request, and neither visible from the documentation:

- **A session opened while the server is busy dies, and its output is not lost.** Three of five sessions
  ended `1011 (internal error) keepalive ping timeout` — `uvicorn.run`
  (`serve_joyomni_streaming.py:1910`) takes the websockets defaults, ping every 20 s and close on a 20 s
  pong timeout, and exposes no override. A session that gets a warm, uncontended server does *not* hit
  this: the clean run completed in 226 s and returned 105 of 113 frames at 1248x720 through
  `/download_last`. The ones that died were opened while the startup warmup or a previous session's work
  was still running. When it does happen, the frames still reach disk — the recorder writes them at
  session close — but `/download_last` answers **404**, because it publishes only after an explicit
  `finalize_recording` that the dead socket never carries. That combination reads as "the run produced
  nothing" while the mp4 sits in `--record-dir`, so `pe_serve_client.py` salvages from there.
- **The startup warmup is ~90 s of work that the log appears to place after startup completes.** Reading
  it wrongly is what makes the first failure baffling. `--preload` is already the default
  (`BooleanOptionalAction, default=True`, line 1891), so the runtime *is* built during `lifespan`:
  an 89 s DiT load, the VAE compile warmups, then two full-pipeline warmups logging 43.6 s and 46.4 s.
  But those progress lines are plain `print()`, block-buffered when stdout is redirected, while
  `INFO: Application startup complete` goes to unbuffered stderr — so in a `2>&1` log the warmup appears
  *below* the line that says startup finished. The dynamo warning's timestamp (23:11:52, above
  `Application startup complete`) is what settles the real order.

The final chunk is also not emitted, on every clip: 105 frames back for 113 sent, 153 for 161, 177 for
185, 200 for 209, 153 for 161. Four of the five are short by exactly 8, i.e. `1 + 8n` of `1 + 8(n+1)` —
the server holds the last chunk rather than flushing it. case04's 9 is one frame more than that, so a
whole chunk is the floor rather than the rule, and the extra frame is unaccounted for. For a live stream
this is invisible; for a fixed clip it silently shortens the result by a chunk or more, which also means
any metric comparing served output against a reference has to truncate to the shorter clip rather than
assume alignment.

#### Notes

- Memory usage: ~47 GiB resident for the weights (30 GiB DiT + 15 GiB condition encoder + 1.4 GiB
  VAE); ~53 GiB peak at 720x1248. Wants an 80 GiB-class card.
- Throughput at 720x1248 on one H20: **~2.11 s per latent chunk**, measured over 108 chunks (the five
  upstream showcase clips, 829 frames total, 227.9 s of rollout — per-clip spread was 2.08–2.14 s, so
  the cost per chunk is flat in clip length, as bounded KV inference predicts). A 73-frame clip
  (10 chunks) is therefore about 21 s of rollout plus ~11 s of weight loading.
- **Upstream is ~18% faster per chunk (1.79 s), and the difference is not compute.** Profiled on
  case01 with `enable_diffusion_pipeline_profiler=True`: of 2.194 s/chunk, `diffuse` is **1.648 s**,
  `vae.encode` 0.173 s, `vae.decode` 0.143 s, and 0.230 s is per-request work (the condition encoder
  runs once per session) plus pre/post-processing. Our denoise alone is therefore *below* upstream's
  entire per-chunk time, so nothing here computes more slowly — upstream simply overlaps five daemon
  threads with only the DiT serialised (`runtime.dit_lock`), which hides its VAE behind its denoise.
  It also `torch.compile`s the VAE — but on this box that compile does not survive its own startup, so
  the overlap is the whole of the advantage (see the recompile-budget bullet below). The VAE's
  0.316 s/chunk accounts for 79% of the
  0.401 s/chunk gap. That throughput is what buys this port's reproducibility: running the stages in
  series is why our runs are bit-identical and upstream's are not (see the ceiling
  above). Recovering it — VAE `torch.compile`, or moving encode/decode onto worker threads — is a
  deliberate trade against that property, not a free win.
- **`channels_last_3d` on the VAE must stay off, and it is not a judgement call.** Upstream converts
  all 66 `Conv3d` weights to that layout before compiling, so it was the obvious thing to copy. Doing
  it here changes the output — whole-clip pixel checksum 27,607,984,473 against 27,244,804,949, a shift
  of about 1.2 grey levels in the mean. Two independent eager runs, on different GPUs under different
  load, produced bit-identical checksums, so that difference is the layout and not run-to-run noise;
  that determinism is also what makes checksum comparison usable as a regression guard at all. What it
  buys is *nothing that survives being measured twice*: at the same 720x1248 and dtype it was slower in
  the full pipeline (0.19 s per encode call against eager's 0.151 s, 2.212 s/chunk against 2.174) and
  faster in isolation (0.113 s against 0.167 s). The two disagree in sign, so the honest claim is that
  it moves numerics for no reliable throughput change — which has no defensible default but off.
- **VAE `torch.compile`: a real 2.3–3.3× win on `encode`, a measured loss on `decode`, and off by
  default for a third reason that is neither of those.** Isolated per call at 720x1248 on one H20, same
  process:

  | target | eager | compiled steady | first call (the compile) |
  |---|---|---|---|
  | `_encode` | 0.17–0.24 s | **0.072 s** | 8–13 s *per shape* |
  | `_decode` | **0.633 s** | 0.692 s | 34 s (229 s cold) |

  So `_decode` is **not** compiled here, diverging from upstream deliberately: its compiled steady
  state is ~9% *slower than eager*, so no call count pays back the 34 s. That is a difference of regime,
  not of hardware — upstream streams, so its decode runs once per chunk over 1–2 latent frames at two
  static shapes, where `decode_latents` makes one call whose temporal extent *is* the clip length.
  Encode's win is real and stable (0.072 s in every run), but its ~17 s for two shapes buys ~0.17 s per
  9-frame window: break-even is ~100 windows, about an 800-frame clip, against the 15 windows of a
  113-frame request. **A warm `TORCHINDUCTOR_CACHE_DIR` does not change that** — 12.0/11.7 s per shape
  cold against 13.0/13.4 s warm, because `select_algorithm_autotune` and 558 `coordesc_tuning_bench`
  calls re-run either way; only decode's compile caches meaningfully. `VLLM_OMNI_JOYAI_VAE_COMPILE=1`
  is therefore for a long-lived server that warms once and serves many requests — which is what
  upstream is — and not for one offline request.

  End to end on the 113-frame case01, that arithmetic holds, and it is worth quoting because it is the
  number a reader will expect the compile to *improve*. Run back to back on one idle H20 so node
  contention hits both arms equally — an earlier unpaired attempt was discarded when `diffuse` moved to
  3.575 s/chunk, which a VAE-only change cannot cause:

  | run | wall | `diffuse` | `vae.encode` | `vae.decode` | pixel checksum |
  |---|---|---|---|---|---|
  | eager | **32.57 s** | 1.644 s/chunk | 2.48 s | 2.13 s | 27,244,804,949 |
  | compile | 52.12 s | 1.646 s/chunk | 21.60 s | 2.19 s | 27,203,008,240 |

  `diffuse` is unchanged to 2 ms, so the arms are comparable, and the whole 19.55 s regression is
  encode's 19.12 s — the two warmup compiles, which land inside the timed region because
  `_warmup_vae_compile` runs there. Net of them the 15 calls take ~1.1 s against eager's 2.48 s, so the
  steady-state win is exactly the isolated one; it is simply 18× too small to repay the compile at this
  clip length. Per-call and end-to-end measurements agreeing is the reason to trust either.
- **Enabling the compile also costs bit-reproducibility, which is a bigger loss than the throughput.**
  Eager reproduces the whole-clip checksum exactly across three runs, on more than one GPU and under
  different node load (27,244,804,949); every compiled run differs, *including two runs of the same
  configuration* (27,077,418,059 against 27,203,008,240). `max-autotune` benchmarks candidate kernels at
  compile time
  and machine conditions pick the winner, so the output depends on what else the node was doing. The
  checksum is what makes a numerics regression detectable here at all, and under the compile there is no
  fixed value left to compare against.
- **Upstream's encode compile is spent before it serves a single request, and that resolves why its
  advantage is pure overlap.** `vae_compile.py` wraps the *same code object* twice —
  `maybe_setup_encode` (`dynamic=False`) and `maybe_setup_encode_dynamic` (`dynamic=True`) both compile
  `vae._encode` (`vae/vae.py:612`), differing only in the flag and where the wrapper is stored. Dynamo's
  recompile budget is per code object, so they share one allowance of 8 while startup warms **49**
  dynamic reference-image shapes before 6 static ones. Upstream's own log records the result:
  `torch._dynamo hit config.recompile_limit (8)` for `_encode`, last guard `(x.size()[4] % 3) != 0` —
  the `Stem`'s stride-3 divisibility check at `vae.py:482`, which cannot be proven under dynamic shapes
  and so specialises per width. An exhausted frame is skipped thereafter, i.e. runs eager; verified
  directly rather than taken from the docs — with the limit set to 4, seven shapes through one wrapper
  exhaust it and a fresh wrapper over the same function then compiles nothing and returns the eager
  result. The limit was hit on **two independent startups** of the same server, so it is deterministic
  rather than a scheduling artefact. So at upstream's documented settings the encode compile is consumed
  by a path no ordinary
  request uses and the shapes every request *does* use run uncompiled. This is also the real reason
  `_encode_dynamic` is skipped in this port: adding it would not add a compile, it would take the static
  one away.
- **The bug that made the above measurable at all is worth stating, because its symptom pointed the
  wrong way.** Compile first measured as a 4× end-to-end *regression* (`vae.encode` 43.9 s over 15
  chunks against eager's 2.27 s) with a fully warm inductor cache and zero `AUTOTUNE` blocks in the
  timed region — which reads exactly like "the autotuned kernels are simply worse on sm90". They are
  not: this port warmed encode *outside* autocast while `encode_source_windows` calls it inside one,
  and autocast state is part of a compiled graph's guards, so the warmed graph could never be reused
  and each real shape recompiled inside the request. Upstream's own `warmup_encode` default has the
  same trap (`autocast=False`) and its caller overrides it to `autocast=(vae_dtype != float32)` at
  `joyomni_streaming.py:258`, which is why every static shape in upstream's startup log reads
  `autocast=True`. Only isolating the VAE per call separated a bad-kernel story from a recompile story;
  `tests/diffusion/models/joyai_video_edit/test_vae_compile.py` now pins the autocast contract.
- Key flags: `num_inference_steps=2` and `max_num_seqs=1` are correctness constraints, not tuning
  knobs — see below.
- Geometry: frames must be `1 + 8n`; height and width must be divisible by **24**. The 24 is stricter
  than the 16 the VAE encoder alone implies, because a `Stem` layer with stride 3 sits in front of it.
  `1280` is the trap — divisible by 16, not by 3 — which is why the reference width is `1248`. Both
  constraints raise an error naming the nearest valid values rather than snapping silently.

#### Known limitations

- **Batch size 1 only.** The DiT shares one rotary position table and one KV cache scope across the
  batch, so a fused pair would generate the second video against the first one's positions.
  `max_num_seqs: 1` in the deploy YAML keeps requests serialised (passing a list of requests works —
  they just run one at a time), and the pipeline raises if that cap is ever lifted.
- **`num_inference_steps` is fixed at 2.** The AR-DMD distillation collapsed the sampling trajectory
  into exactly two Euler steps on a shift-5.159 sigma grid; raising it integrates a velocity field
  that no longer matches the grid. The pipeline warns instead of silently degrading.
- **No parallelism.** Tensor, sequence, and HSDP parallelism are not wired up. CFG parallelism is
  structurally inapplicable: the model was distilled without classifier-free guidance, so there is no
  negative branch to place on a second rank.
- **No Cache-DiT or TeaCache.** Residual caching needs roughly ten or more steps to find
  step-to-step redundancy, and there are two.
- **No CPU offload.**
- **No streaming.** Deferred, not unsupported upstream — the chunk-autoregressive rollout is what
  makes streaming possible, and `JoyKVWindow` is the seam a later streaming phase would re-back.
- **No negative prompt**, for the same reason CFG parallelism does not apply.
- A `num_frames` of 0 or 1 both mean "edit the longest valid prefix of the source"; see the example
  README for why, and note there is consequently no way to request a 1-frame edit of a longer clip.
