import base64

from io import BytesIO
from typing import Optional
from PIL import Image
from IPython.display import display



def to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def ensure_image_bytes(image_bytes: Optional[bytes]) -> bytes:
    """Guarantee bytes exist (fallback to 1x1 PNG if sample has no image)."""
    if image_bytes is None:
        img = Image.new("RGB", (1, 1), (255, 255, 255))
        buf = BytesIO(); img.save(buf, format="PNG")
        return buf.getvalue()
    return image_bytes

def hf_first_image_bytes(sample) -> Optional[bytes]:
    imgs = sample.get("images") or []
    if not imgs:
        return None
    b = imgs[0].get("bytes")
    if isinstance(b, list):
        b = bytes(b)  # some HF datasets store ints
    return b

def hf_user_and_ref(sample):
    # VisionArena-Chat: single-turn → [ [user_msg], [ref_ans] ]
    conv = sample["conversation"]
    user_text = conv[0][0]["content"]
    ref_answer = conv[1][0]["content"]
    return user_text, ref_answer

def show_image(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes))
    display(img)

