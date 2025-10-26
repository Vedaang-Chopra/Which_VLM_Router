import os, io, base64, time, uuid, requests
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

BACKEND = os.getenv("OCR_BACKEND", "paddle")
if BACKEND == "paddle":
    from backends.paddle_backend import OCRRunner
elif BACKEND == "tesseract":
    from backends.tess_backend import OCRRunner
else:
    raise RuntimeError("Unknown OCR_BACKEND: " + BACKEND)

API_KEY = os.getenv("API_KEY", "changeme")
MODEL_NAME = os.getenv("MODEL_NAME", "ocr-1")

app = FastAPI(title="OpenAI-compatible OCR")
runner = OCRRunner()

def _fetch_image(image_part: Dict[str, Any]) -> Image.Image:
    if "image_url" in image_part and image_part["image_url"]:
        url = image_part["image_url"]["url"]
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if "image" in image_part and "b64_json" in image_part["image"]:
        b = base64.b64decode(image_part["image"]["b64_json"])
        return Image.open(io.BytesIO(b)).convert("RGB")
    raise ValueError("No image found in message content")

def _extract_images_and_prompt(messages: List[Dict[str, Any]]):
    imgs, txts = [], []
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, str):  # some clients still send plain text
            txts.append(content)
            continue
        for part in content:
            if part.get("type") in ("input_text","text"):
                txts.append(part.get("text", ""))
            elif part.get("type") in ("input_image","image_url","image"):
                imgs.append(_fetch_image(part))
    return imgs, "\n".join([t for t in txts if t]).strip()

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

@app.middleware("http")
async def check_api_key(request, call_next):
    if request.url.path.startswith("/v1/"):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.split(" ",1)[1] != API_KEY:
            return JSONResponse({"error": {"message":"Unauthorized"}}, status_code=401)
    return await call_next(request)

@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    t0 = time.time()
    try:
        images, prompt = _extract_images_and_prompt(payload.messages)
        if not images:
            text = "No image provided. Please attach an image part."
        else:
            text = runner.ocr(images[0])  # single-image; extend as needed
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        elapsed = time.time() - t0
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model or MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(text.split()),
                "total_tokens": len(text.split())
            },
            "system_fingerprint": "ocrsrv-001",
            "latency_ms": int(elapsed*1000)
        }
    except Exception as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=400)

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "you"}]
    }
PY
