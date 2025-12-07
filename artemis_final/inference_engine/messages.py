

"""
Message and image helpers for OpenAI-style (LLM + VLM) requests.

This module is intentionally self-contained and can be reused by both
LLM and VLM test suites. It provides:

- Lightweight helpers to normalise / detect different image types:
  * file paths
  * HTTP URLs
  * data URLs
  * base64 strings
  * bytes
  * PIL.Image.Image
  * numpy.ndarray

- A `build_messages` function that produces OpenAI-compatible
  multimodal `messages` payloads, suitable for `client.chat.completions.create`.

All functions are pure and side-effect free, except for reading image
files from disk when a file path is provided.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any, Dict, List, Optional, Union

# Optional imports: PIL and numpy are only needed if you pass those types in.
try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


ImageLike = Union[str, bytes, "Image.Image", "np.ndarray"]


# ---------------------------------------------------------------------------
# Low-level detection helpers
# ---------------------------------------------------------------------------


def _is_data_url(s: str) -> bool:
    """Return True if the string looks like a data:image/... URL."""
    return isinstance(s, str) and s.startswith("data:image/")


def _is_http_url(s: str) -> bool:
    """Return True if the string looks like an HTTP/HTTPS URL."""
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def _looks_b64(s: str) -> bool:
    """
    Heuristic check for base64-like strings.

    This is intentionally permissive – we only use it when the caller
    has clearly passed a string that is *not* a path and not a URL.
    """
    if not isinstance(s, str):
        return False
    s2 = s.strip().replace("\n", "")
    if not s2:
        return False
    if len(s2) % 4 != 0:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=]+", s2) is not None


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------


def _to_png_bytes_from_any(img: ImageLike) -> bytes:
    """
    Convert a variety of Python image representations into PNG bytes.

    Supported inputs:
    - Bytes / bytearray -> returned as bytes (assumed already image bytes).
    - File path (str)   -> read from disk.
    - PIL.Image.Image   -> encoded as PNG.
    - numpy.ndarray     -> encoded as PNG (via PIL).

    URLs / data URLs / base64 strings should be handled at a higher level and
    are *not* passed to this function.
    """
    # File path on disk
    if isinstance(img, str):
        # At this stage we assume non-URL / non-data-URL / non-b64 strings are paths.
        with open(img, "rb") as f:
            return f.read()

    # Raw bytes
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)

    # PIL image
    if Image is not None and isinstance(img, Image.Image):  # type: ignore[attr-defined]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # numpy array
    if np is not None and isinstance(img, np.ndarray):  # type: ignore[attr-defined]
        if Image is None:
            raise ValueError("PIL is required to convert numpy.ndarray to PNG bytes.")
        pil_img = Image.fromarray(img)  # type: ignore[call-arg]
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    raise ValueError(
        "Unsupported image type. Use URL/data URL/base64 string, "
        "file path, bytes, PIL.Image.Image, or numpy.ndarray."
    )


def _image_to_part(img: ImageLike) -> Dict[str, Any]:
    """
    Convert an `ImageLike` object into an OpenAI-style image content part:

        {"type": "image_url", "image_url": {"url": "..."}}

    The `url` may be:
    - http(s) URL as-is
    - data URL as-is
    - constructed from base64 PNG bytes for:
      * bare base64 strings
      * paths
      * PIL / numpy / bytes
    """
    # String cases: URL, data URL, or bare base64
    if isinstance(img, str):
        if _is_http_url(img) or _is_data_url(img):
            return {"type": "image_url", "image_url": {"url": img}}
        if _looks_b64(img):
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}"},
            }

    # Anything else: convert to PNG bytes then to data URL
    png_bytes = _to_png_bytes_from_any(img)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


# ---------------------------------------------------------------------------
# High-level message builder
# ---------------------------------------------------------------------------


def build_messages(
    prompt: Optional[str] = None,
    images: Optional[Union[Dict[str, Any], List[ImageLike], ImageLike]] = None,
    content_parts: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-style multimodal messages.

    This is a thin, reusable abstraction over the "messages" structure used by
    `client.chat.completions.create`. It supports:

    1. Arbitrary pre-built content:
       - If `content_parts` is provided, they are used as the `user` content
         verbatim. Example:
           content_parts = [
               {"type": "text", "text": "Question:"},
               {"type": "image_url", "image_url": {"url": "..."}},
           ]

    2. Prompt + images:
       - If `content_parts` is not provided, we construct a single `user` message
         where:
           - `prompt` becomes a text part (if not None).
           - `images` become one or more image parts.
         Both are combined into a single `content` list.

    3. Optional system message:
       - If `system` is provided, a `{"role": "system", "content": [...]}` message
         is prepended, where content is a single text part.

    Parameters
    ----------
    prompt:
        Optional user text prompt.
    images:
        Either:
          - None
          - A single ImageLike (path/URL/bytes/PIL/numpy)
          - A list of ImageLike
          - An already-formed image content dict with `{"type": "image_url", ...}`.
    content_parts:
        If provided, overrides `prompt` and `images` and is used as the `user`
        content as-is.
    system:
        Optional system prompt string.

    Returns
    -------
    List[Dict[str, Any]]
        A list of message dicts suitable for OpenAI-style chat APIs.
    """
    messages: List[Dict[str, Any]] = []

    # Optional system message
    if system:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system}],
            }
        )

    # If caller provides content_parts directly, use it as-is
    if content_parts is not None:
        messages.append({"role": "user", "content": content_parts})
        return messages

    # Otherwise, we combine prompt + images into one user message
    parts: List[Dict[str, Any]] = []

    if prompt is not None:
        parts.append({"type": "text", "text": prompt})

    if images is not None:
        # If caller already built an image_url dict, just append it
        if isinstance(images, dict) and images.get("type") == "image_url":
            parts.append(images)
        else:
            # Normalise to a list
            if not isinstance(images, list):
                images = [images]  # type: ignore[list-item]
            for img in images:  # type: ignore[assignment]
                parts.append(_image_to_part(img))

    if not parts:
        raise ValueError("Provide at least one of: prompt, images, or content_parts.")

    messages.append({"role": "user", "content": parts})
    return messages