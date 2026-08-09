# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen geometry for JoyAI-Video-Edit.

Every value here was derived from the shipped checkpoint
(``dit/joyai_video_edit_dit_0804.pth``: a flat ``OrderedDict`` of 894 bf16 tensors,
16,263,675,968 parameters) rather than from upstream's ``Transformer3DModel.__init__`` defaults,
which describe a different, smaller model. Tests import from here so a silent drift in any of these
shows up as a failed assertion rather than a shape error 40 layers deep.
"""

from __future__ import annotations

from typing import Final

# --- DiT geometry -------------------------------------------------------------------------------
# ``img_in.weight`` is (4096, 64, 1, 1, 1) -> a Conv3d with a 1x1x1 kernel, i.e. the DiT performs no
# patchification at all. All spatial reduction is the VAE's (24x), which is why ``proj_out`` emits
# ``out_channels * prod(patch_size) == 64``.
PATCH_SIZE: Final[list[int]] = [1, 1, 1]
IN_CHANNELS: Final = 64
OUT_CHANNELS: Final = 64
HIDDEN_SIZE: Final = 4096
HEADS_NUM: Final = 32
HEAD_DIM: Final = HIDDEN_SIZE // HEADS_NUM  # 128
MLP_WIDTH_RATIO: Final = 4.0
MM_DOUBLE_BLOCKS_DEPTH: Final = 40
# MiMo-VL-7B-RL-2508 has hidden_size 4096 (wider than plain Qwen2.5-VL-7B's 3584).
TEXT_STATES_DIM: Final = 4096

TOTAL_DIT_PARAMS: Final = 16_263_675_968

# --- Normalisation / modulation ----------------------------------------------------------------
NORM_EPS: Final = 1e-6
# AdaLN-single: ``condition_embedder.time_proj`` emits 6 * hidden, and each block owns a learned
# ``modulate_table`` (1, 6, hidden) offset on it. Order: shift1, scale1, gate1, shift2, scale2, gate2.
NUM_MODULATION_CHUNKS: Final = 6
TIME_FREQ_DIM: Final = 256

# --- RoPE ---------------------------------------------------------------------------------------
# Interleaved (GPT-J) 3D RoPE over (t, h, w); must sum to HEAD_DIM.
ROPE_DIM_LIST: Final[list[int]] = [16, 56, 56]
ROPE_THETA: Final = 256

# A second rotation composed on top of the 3D one, keyed by which stream a token came from. This is
# the *only* thing distinguishing the noisy target chunk from the clean source chunk: both are given
# identical (t, h, w) positions, so dropping it makes the edit condition invisible to the model.
SOURCE_ID_ROPE_DIM: Final = 128
SOURCE_ID_ROPE_THETA: Final = 256.0
SOURCE_ID_TARGET: Final = 0.0
SOURCE_ID_EDIT_CONDITION: Final = 1.0
SOURCE_ID_EXTRA_REF_IMAGE: Final = 2.0

SELF_ATTN_MODE_REF_IMAGE_CACHE: Final = "ref_image_cache"

# --- Autoregressive rollout / KV window ---------------------------------------------------------
CHUNK_SIZE: Final = 1
# ``global_sink_chunk`` + window 3 selects chunk 0 plus the two most recent -> at most 3 resident
# chunks, which is what bounds memory and position range.
LOCAL_WINDOW_SIZE: Final = 3
GLOBAL_SINK_CHUNK: Final = True
# ``True`` in the shipped deployment config. It declares a chunk-autoregressive *rollout*, not a
# causal attention mask -- attention is unmasked at every call site. Its real consumers are the
# pipeline (it gates the sliding-window VAE encode and the per-chunk `chunk_size`) and the streaming
# session, which refuses a non-causal config outright.
CAUSAL: Final = True
KV_CACHE_PRE_ROPE: Final = True
# Sentinel chunk id for the static reference-image KV entry (stored post-RoPE).
KV_CACHE_ID_REF_IMAGE: Final = -1

# --- Sampling -----------------------------------------------------------------------------------
# AR-DMD distilled: 2 steps is the trained operating point, not a speed knob. No CFG.
NUM_INFERENCE_STEPS: Final = 2
SCHEDULER_SHIFT: Final = 5.159
NUM_TRAIN_TIMESTEPS: Final = 1000

# --- VAE ----------------------------------------------------------------------------------------
VAE_LATENT_CHANNELS: Final = 64
# Stem stride 3 x three 2x downsample blocks = 24x spatial; 8x temporal.
VAE_SPATIAL_COMPRESSION: Final = 24
VAE_TEMPORAL_COMPRESSION: Final = 8

# --- Conditioning -------------------------------------------------------------------------------
MAX_CONDITION_TOKENS: Final = 1024
# Anchor frame is resized area-preserving to roughly this many pixels per side before the ViT.
VIT_FIXED_SIZE: Final = 512

# --- Request defaults ---------------------------------------------------------------------------
# Width must stay divisible by the VAE Stem's stride of 3; 1280 is *not* (1280 % 3 == 2), which is
# why upstream's server default is 1248.
DEFAULT_HEIGHT: Final = 720
DEFAULT_WIDTH: Final = 1248


def dit_config() -> dict[str, object]:
    """Exact keyword arguments for ``JoyAIVideoEditTransformer3DModel``."""
    return {
        "patch_size": list(PATCH_SIZE),
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "hidden_size": HIDDEN_SIZE,
        "heads_num": HEADS_NUM,
        "text_states_dim": TEXT_STATES_DIM,
        "mlp_width_ratio": MLP_WIDTH_RATIO,
        "mm_double_blocks_depth": MM_DOUBLE_BLOCKS_DEPTH,
        "rope_dim_list": list(ROPE_DIM_LIST),
        "theta": ROPE_THETA,
        "chunk_size": CHUNK_SIZE,
        "local_window_size": LOCAL_WINDOW_SIZE,
        "global_sink_chunk": GLOBAL_SINK_CHUNK,
        "causal": CAUSAL,
        "source_id_rope_dim": SOURCE_ID_ROPE_DIM,
        "source_id_rope_theta": SOURCE_ID_ROPE_THETA,
    }
