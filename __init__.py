"""Self Gradient Forcing+ for ComfyUI (unofficial integration of https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus)."""

import logging

try:
    from .sgfplus_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[SGF+] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
