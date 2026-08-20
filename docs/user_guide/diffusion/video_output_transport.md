# Video Output Transport

Video payloads are orders of magnitude larger than image or audio payloads: a
raw float32 tensor of 81–121 frames at 720p reaches hundreds of megabytes to
several gigabytes per request. `VideoOutputTransportConfig` controls how that
payload is reduced and how it reaches the client.

All of it is opt-in. The defaults reproduce the previous behaviour exactly.

## Configuration

`video_output_transport` lives on the diffusion config, so it can be set
programmatically, in a deploy YAML, or as a CLI JSON string.

```python
from vllm_omni.entrypoints.omni import Omni

llm = Omni(
    model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    output_type="np",
    video_output_transport={
        "enable_device_postprocess": True,
        "transport_mode": "url",
        "output_format": "mp4",
    },
)
```

```bash
vllm-omni serve <model> \
  --video-output-transport '{"enable_device_postprocess": true, "transport_mode": "url"}'
```

| Field | Default | Effect |
| --- | --- | --- |
| `enable_device_postprocess` | `False` | Run denormalize/clamp/layout/uint8 on the device before the D2H copy. |
| `transport_mode` | `"base64"` | `"base64"` returns an inline payload; `"url"` stores the artifact and returns a URL. |
| `output_format` | `"mp4"` | Container for encoded artifacts: `mp4` or `webm`. |
| `video_codec` | `None` | Encoder name. `None` means the container's default (h264 for mp4, VP9 for webm). |
| `video_codec_options` | `{}` | Encoder options. Empty means the fast presets for whichever codec runs. |
| `shm_threshold_bytes` | `1000000` | Tensors above this size cross the worker→engine hop through shared memory. |

## Device-side postprocessing

With `enable_device_postprocess: True`, supported pipelines convert the decoded
video to uint8 frames on the GPU, so the worker→engine copy carries 4x less
data. Output is bit-for-bit identical to the CPU path.

Supported: WAN 2.2, HunyuanVideo 1.5, LTX-2, MiniMax-H3, Cosmos3,
LingBot-Video.

The reduction is skipped, falling back to the float path, when any of these
hold:

- the request asks for `latent` or `pil` output rather than `np`
- frame interpolation is enabled, which needs the float tensor
- Cosmos3 guardrails are enabled, because the safety check inspects the float
  video
- the request produces an image or an action sequence rather than a video

## Artifact delivery

`transport_mode: "url"` keeps the encoded video out of the JSON body, avoiding
the ~33% inflation base64 adds, and lets the client fetch the bytes with a
normal chunked download.

By default artifacts are served by this server at
`/v1/videos/artifacts/{key}`, so no extra infrastructure is required. To let a
static server, CDN, or object store serve them instead, publish the storage
directory and point the server at it:

```bash
export VLLM_OMNI_SERVER_STORAGE__PUBLIC_BASE_URL="https://cdn.example.com/videos"
```

Artifact handles then become that URL, and `/v1/videos/{id}/content` redirects
to it instead of streaming bytes through this process.

### Object storage

Storage backends are pluggable. An out-of-tree backend registers a factory and
returns a `UrlStorageHandle`, which the video routes redirect to:

```python
from vllm_omni.entrypoints.openai.storage import (
    StorageBaseManager,
    UrlStorageHandle,
    register_storage_backend,
)

class MyObjectStore(StorageBaseManager):
    async def open(self, storage_key):
        return UrlStorageHandle(url=f"https://my-bucket/{storage_key}")
    ...

register_storage_backend("my-object-store", lambda config: MyObjectStore())
```

No object-store client ships in-tree.

## Encoders

`video_codec` accepts any encoder PyAV exposes. A hardware encoder that the
host cannot open falls back to the container's software default with a warning,
so the same configuration is safe across mixed hardware:

```python
video_output_transport={"video_codec": "h264_nvenc"}
```

Encoder options are not portable between encoder families, so when a fallback
happens the requested codec's options are dropped in favour of the fallback's
own defaults.

## Zero-copy for co-located consumers

A consumer in another process on the same host can read a worker's video output
directly out of shared memory instead of copying it:

```python
from vllm_omni.diffusion.ipc import borrowed_diffusion_output

with borrowed_diffusion_output(packed_output) as output:
    video = output.output["video"]  # a view over the shared segment
    frames = video.clone()         # anything outliving the block must be copied
```

!!! warning

    Leaving the `with` block invalidates every borrowed tensor. Reading one
    afterwards is a use-after-free, the same contract as `free()`, Arrow, or
    CUDA IPC.

This is a library API for co-located frameworks; the engine's own output path
copies as before.
