# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video/audio muxing utilities using PyAV (no ffmpeg binary dependency)."""

from __future__ import annotations

import functools
import io
from fractions import Fraction
from typing import Any, cast

import av
import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_VIDEO_CODEC = "h264"

# Fast-preset options per encoder. FFmpeg rejects options that belong to another
# encoder family (passing NVENC's "preset=p1" to libx264 fails avcodec_open2),
# so these cannot be shared and must follow whichever codec actually runs.
_FAST_CODEC_OPTIONS: dict[str, dict[str, str]] = {
    "h264": {"preset": "ultrafast", "threads": "0"},
    "libx264": {"preset": "ultrafast", "threads": "0"},
    "hevc": {"preset": "ultrafast", "threads": "0"},
    "libx265": {"preset": "ultrafast", "threads": "0"},
    "h264_nvenc": {"preset": "p1", "tune": "ull"},
    "hevc_nvenc": {"preset": "p1", "tune": "ull"},
}

# Extra options for latency-sensitive streaming. NVENC's "ull" tune above already
# implies ultra-low latency, so only the software encoders need this.
_LOW_LATENCY_OPTIONS: dict[str, dict[str, str]] = {
    "h264": {"tune": "zerolatency"},
    "libx264": {"tune": "zerolatency"},
    "hevc": {"tune": "zerolatency"},
    "libx265": {"tune": "zerolatency"},
}


@functools.cache
def _encoder_is_usable(codec: str) -> bool:
    """Whether this FFmpeg build and this machine can actually open ``codec``.

    ``add_stream`` accepts an unusable encoder and only fails later on the first
    ``encode()``, so the encoder has to be opened to find out. Hardware encoders
    are the usual mismatch: a build can expose ``h264_nvenc`` on a GPU that has
    no NVENC engine (Hopper data-center parts ship none).
    """
    try:
        ctx = av.codec.CodecContext.create(codec, "w")
        ctx.width, ctx.height, ctx.pix_fmt = 64, 64, "yuv420p"
        ctx.open()
    except Exception:
        return False
    return True


def default_video_codec_options(codec: str, *, low_latency: bool = False) -> dict[str, str]:
    """Fast-preset encoder options matching ``codec``."""
    options = dict(_FAST_CODEC_OPTIONS.get(codec, {}))
    if low_latency:
        options.update(_LOW_LATENCY_OPTIONS.get(codec, {}))
    return options


def resolve_encoder_settings(
    codec: str | None,
    codec_options: dict[str, str] | None = None,
    *,
    low_latency: bool = False,
    fallback: str = DEFAULT_VIDEO_CODEC,
) -> tuple[str, dict[str, str]]:
    """Pick an encoder that can run here plus the options that match it.

    ``codec_options`` are honoured only when the requested encoder is the one
    that ends up running. If it is unavailable and we fall back, its options are
    dropped in favour of the fallback's own defaults, because FFmpeg refuses
    options from a different encoder family and would fail the whole encode.

    Empty ``codec_options`` means "use the fast defaults for this codec".
    """
    requested = codec or fallback
    if requested != fallback and not _encoder_is_usable(requested):
        logger.warning(
            "Video encoder %r cannot be opened on this host; falling back to %r and its default options.",
            requested,
            fallback,
        )
        return fallback, default_video_codec_options(fallback, low_latency=low_latency)
    if codec_options:
        return requested, dict(codec_options)
    return requested, default_video_codec_options(requested, low_latency=low_latency)


class FragmentedMP4Muxer:
    """Incrementally mux video frames into one fragmented MP4 byte stream."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float = 25.0,
        video_codec: str = "h264",
        crf: str = "18",
        video_codec_options: dict[str, str] | None = None,
    ) -> None:
        self._buf = io.BytesIO()
        self._closed = False
        self._container = av.open(
            self._buf,
            mode="w",
            format="mp4",
            options={"movflags": "+frag_every_frame+empty_moov+default_base_moof"},
        )

        self._stream: av.VideoStream = cast(
            av.VideoStream,
            self._container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)),
        )
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"

        options: dict[str, object] = {"crf": str(crf)}
        if video_codec_options:
            options.update(video_codec_options)
        self._stream.options = options

        try:
            self._stream.codec_context.max_b_frames = 0
        except AttributeError:
            pass

    def mux_video_frames(self, video_frames: np.ndarray) -> bytes:
        """Mux a batch of ``uint8`` RGB frames and return newly written MP4 bytes."""
        if self._closed:
            raise RuntimeError("Cannot mux frames after FragmentedMP4Muxer.close().")
        if video_frames.ndim != 4 or video_frames.shape[-1] != 3:
            raise ValueError("video_frames must have shape (T, H, W, 3).")
        if video_frames.dtype != np.uint8:
            raise ValueError("video_frames must be uint8.")
        if video_frames.shape[1] != self._stream.height or video_frames.shape[2] != self._stream.width:
            raise ValueError("All fragmented MP4 chunks in a session must use the same frame size.")

        for frame_data in video_frames:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        return self._read_new_bytes()

    def close(self) -> bytes:
        """Flush delayed encoder packets, close the container, and return final bytes."""
        if self._closed:
            return b""
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._closed = True
        return self._read_new_bytes()

    def _read_new_bytes(self) -> bytes:
        """Return newly muxed bytes in the current video container,
        then clear the buffer to prepare for the next chunk."""
        chunk = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate()
        return chunk


def finalize_streaming_video_bytes(
    video_bytes: bytes,
    *,
    input_format: str,
    fps: float = 25.0,
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Convert streamed video bytes into a progressive MP4 for local playback."""
    if not video_bytes:
        return video_bytes

    normalized_format = input_format.lower()
    if normalized_format == "m4s":
        demux_format = "mp4"
    else:
        raise ValueError(f"Unsupported streaming video format: {input_format}")

    try:
        with cast(Any, av.open(io.BytesIO(video_bytes), format=demux_format)) as container:
            stream = container.streams.video[0]
            frame_arrays = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    except Exception:
        return video_bytes

    if not frame_arrays:
        return video_bytes

    frames_u8 = np.ascontiguousarray(np.stack(frame_arrays, axis=0), dtype=np.uint8)
    return mux_video_audio_bytes(
        frames_u8,
        None,
        fps=float(fps),
        video_codec_options=video_codec_options,
    )


def mux_video_audio_bytes(
    video_frames: np.ndarray,
    audio_waveform: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    audio_sample_rate: int = 44100,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Mux video frames and optional audio waveform into MP4 bytes.

    Args:
        video_frames: uint8 array of shape ``(T, H, W, 3)`` (RGB).
        audio_waveform: float32 array – mono ``(N,)`` or ``(N, C)`` / ``(C, N)``.
        fps: Video frame rate.
        audio_sample_rate: Audio sample rate in Hz.
        video_codec: Video codec name.
        audio_codec: Audio codec name.
        crf: Constant rate factor for the video encoder.

    Returns:
        Raw MP4 bytes ready to be written to disk or streamed.
    """
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")

    v_stream = cast(av.VideoStream, container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)))
    v_stream.width = video_frames.shape[2]
    v_stream.height = video_frames.shape[1]
    v_stream.pix_fmt = "yuv420p"

    options: dict[str, object] = {"crf": str(crf)}
    if video_codec_options:
        options.update(video_codec_options)
    v_stream.options = options

    a_stream: av.AudioStream | None = None
    samples: np.ndarray | None = None
    layout: str | None = None
    if audio_waveform is not None:
        samples = audio_waveform.astype(np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
            samples = np.ascontiguousarray(samples.T)
        num_channels = samples.shape[0]
        layout = "stereo" if num_channels >= 2 else "mono"
        a_stream = cast(av.AudioStream, container.add_stream(audio_codec, rate=audio_sample_rate))
        a_stream.layout = layout

    for frame_data in video_frames:
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in v_stream.encode(frame):
            container.mux(packet)
    for packet in v_stream.encode():
        container.mux(packet)

    if a_stream is not None and audio_waveform is not None:
        if samples is None or layout is None:
            raise ValueError("Audio samples were not prepared for muxing.")
        audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
        audio_frame.sample_rate = audio_sample_rate
        # AAC has a one-frame encoder delay. Mark the input waveform as
        # starting at t=0 so the MP4 muxer writes the corresponding negative
        # priming timestamp instead of exposing the delay as leading silence.
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, audio_sample_rate)
        for packet in a_stream.encode(audio_frame):
            container.mux(packet)
        for packet in a_stream.encode():
            container.mux(packet)

    container.close()
    return buf.getvalue()
