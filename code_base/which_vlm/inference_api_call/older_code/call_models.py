# pip install openai pyyaml pillow numpy

import base64, io, re, time, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from openai import OpenAI  # official SDK

# ---------- helpers for images ----------

def _is_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:image/")

def _is_http_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))

def _looks_b64(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s2 = s.strip().replace("\n", "")
    return len(s2) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/=]+", s2 or "") is not None

def _to_png_bytes_from_any(img: Union[str, bytes, "PIL.Image.Image", "np.ndarray"]) -> Optional[bytes]:
    # URLs / data URLs / base64 strings -> handled elsewhere (return None)
    if isinstance(img, str) and (_is_http_url(img) or _is_data_url(img) or _looks_b64(img)):
        return None
    if isinstance(img, str):
        with open(img, "rb") as f:
            return f.read()
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass
    try:
        import numpy as np
        from PIL import Image
        if "np" in globals() or "numpy" in globals():
            pass
        if isinstance(img, np.ndarray):
            im = Image.fromarray(img)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass
    raise ValueError("Unsupported image type (use URL/data URL/base64, path, bytes, PIL, or numpy).")

def _image_to_part(img: Union[str, bytes, "PIL.Image.Image", "np.ndarray"]) -> Dict[str, Any]:
    if isinstance(img, str):
        if _is_http_url(img) or _is_data_url(img):
            return {"type": "image_url", "image_url": {"url": img}}
        if _looks_b64(img):
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
    png_bytes = _to_png_bytes_from_any(img)
    if png_bytes is None:
        raise RuntimeError("Unexpected None PNG bytes")
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

def build_messages(
    prompt: Optional[str] = None,
    images: Optional[Union[Dict[str, Any], List[Union[str, bytes, "PIL.Image.Image", "np.ndarray"]]]] = None,
    content_parts: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-style multimodal messages.
    - If content_parts provided, they become the 'user' content as-is.
    - Otherwise we combine prompt + images into a single user message.
    - Optional system message is supported.
    """
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": [{"type":"text","text": system}]})

    if content_parts is not None:
        msgs.append({"role": "user", "content": content_parts})
        return msgs

    parts: List[Dict[str, Any]] = []
    if prompt is not None:
        parts.append({"type": "text", "text": prompt})
    if images is not None:
        if isinstance(images, dict) and images.get("type") == "image_url":
            parts.append(images)
        else:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                parts.append(_image_to_part(img))
    if not parts:
        raise ValueError("Provide a prompt, images, or content_parts.")
    msgs.append({"role": "user", "content": parts})
    return msgs

# ---------- data structures ----------

@dataclass
class ModelEndpoint:
    name: str
    base_url: str
    api_key: str
    model_id: str
    pricing: Dict[str, float]      # {prompt_per_1k, completion_per_1k}
    extra_params: Dict[str, Any]   # passed through to the API

# ---------- the runner (OpenAI SDK) ----------

class OpenAIStyleVLMRunner:
    def __init__(self, models: List[ModelEndpoint], request_timeout_s: int = 120, max_workers: int = 4):
        self.models: Dict[str, ModelEndpoint] = {m.name: m for m in models}
        # One OpenAI client per endpoint (base_url); api_key can be "EMPTY"
        self.clients: Dict[str, OpenAI] = {
            m.name: OpenAI(api_key=m.api_key, base_url=m.base_url) for m in models
        }
        self.request_timeout_s = request_timeout_s
        self.max_workers = max_workers

    @classmethod
    def from_yaml(cls, path: str, **kwargs) -> "OpenAIStyleVLMRunner":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        models = [
            ModelEndpoint(
                name=m["name"],
                base_url=m["base_url"].rstrip("/"),
                api_key=m.get("api_key", "EMPTY"),
                model_id=m["model_id"],
                pricing=m.get("pricing", {}),
                extra_params=m.get("extra_params", {}),
            )
            for m in cfg["models"]
        ]
        return cls(models, **kwargs)

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "OpenAIStyleVLMRunner":
        with open(path, "r") as f:
            cfg = json.load(f)
        models = [
            ModelEndpoint(
                name=m["name"],
                base_url=m["base_url"].rstrip("/"),
                api_key=m.get("api_key", "EMPTY"),
                model_id=m["model_id"],
                pricing=m.get("pricing", {}),
                extra_params=m.get("extra_params", {}),
            )
            for m in cfg["models"]
        ]
        return cls(models, **kwargs)

    # ----- single call -----

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sends an OpenAI Chat Completions request to one model.
        Returns: dict with response_text, raw, latency_ms, usage, est_cost, conf_proxy
        """
        if model_name not in self.models:
            raise KeyError(f"Unknown model: {model_name}")
        m = self.models[model_name]
        client = self.clients[model_name]

        payload = dict(
            model=m.model_id,
            messages=messages,
            temperature=kwargs.pop("temperature", 0.2),
            top_p=kwargs.pop("top_p", 1.0),
            max_tokens=kwargs.pop("max_tokens", 512),
            **m.extra_params,
            **kwargs,
        )

        t0 = time.perf_counter()
        resp = client.chat.completions.create(**payload)
        dt_ms = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        response_text = getattr(choice.message, "content", "") or ""

        usage = getattr(resp, "usage", None)
        est_cost = self._estimate_cost(usage, m.pricing)

        # If server supplies logprobs, you could compute a proxy; otherwise None
        conf_proxy = None

        return {
            "ok": True,
            "model": model_name,
            "model_id": m.model_id,
            "response_text": response_text,
            "raw": resp.model_dump() if hasattr(resp, "model_dump") else resp,  # SDK v1 returns pydantic model
            "latency_ms": dt_ms,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage or {}),
            "est_cost": est_cost,
            "conf_proxy": conf_proxy,
            "request": payload,
        }

    # ----- parallel fan-out (threads, not async) -----

    def fanout(
        self,
        messages: List[Dict[str, Any]],
        model_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calls multiple models in parallel using ThreadPoolExecutor.
        """
        if model_names is None:
            model_names = list(self.models.keys())

        results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {
                ex.submit(self.chat, mn, messages, **kwargs): mn for mn in model_names
            }
            for fut in as_completed(futs):
                mn = futs[fut]
                try:
                    results[mn] = fut.result()
                except Exception as e:
                    results[mn] = {"ok": False, "model": mn, "error": str(e)}
        return results

    # ----- convenience wrapper -----

    def run_all(
        self,
        prompt: Optional[str] = None,
        images: Optional[Union[Dict[str, Any], List[Union[str, bytes, "PIL.Image.Image", "np.ndarray"]]]] = None,
        content_parts: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        msgs = build_messages(prompt=prompt, images=images, content_parts=content_parts, system=system)
        return self.fanout(messages=msgs, model_names=model_names, **kwargs)

    # ----- cost helper -----

    @staticmethod
    def _estimate_cost(usage: Optional[Any], pricing: Dict[str, float]) -> float:
        if not usage:
            return 0.0
        try:
            pt = getattr(usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0)
            ct = getattr(usage, "completion_tokens", 0) or usage.get("completion_tokens", 0)
        except Exception:
            pt = ct = 0
        return (pt / 1000.0) * pricing.get("prompt_per_1k", 0.0) + (ct / 1000.0) * pricing.get("completion_per_1k", 0.0)
