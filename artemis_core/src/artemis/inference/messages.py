import base64
import io
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

ImageLike = Union[str, bytes, "Image.Image", "np.ndarray"]

def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "data:image/"))

def _image_to_base64_url(img: ImageLike) -> str:
    if isinstance(img, str):
        if _is_url(img):
            return img
        # Assume file path
        with open(img, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            return f"data:image/png;base64,{b64}" # Default to png header for simplicity or detect mime

    if isinstance(img, bytes):
        b64 = base64.b64encode(img).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    if Image is not None and isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    if np is not None and isinstance(img, np.ndarray):
        if Image is None:
            raise ValueError("PIL required for numpy conversion")
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    raise ValueError(f"Unsupported image type: {type(img)}")

def build_messages(
    prompt: Optional[str] = None,
    images: Optional[Union[List[ImageLike], ImageLike]] = None,
    system: Optional[str] = None
) -> List[Dict[str, Any]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})

    if images is not None:
        if not isinstance(images, list):
            images = [images]
        
        for img in images:
            url = _image_to_base64_url(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
            
    if content:
        messages.append({"role": "user", "content": content})
        
    return messages
