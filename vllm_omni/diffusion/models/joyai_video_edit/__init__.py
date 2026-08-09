# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyAI-Video-Edit: instruction-driven video editing (JD Open Source).

Submodules are importable directly and are deliberately *not* re-exported here: the pipeline pulls in
the 30 GiB DiT's module tree plus the MiMo-VL condition encoder, and the registry resolves it lazily
by name, so an eager re-export would make merely importing this package expensive.
"""
