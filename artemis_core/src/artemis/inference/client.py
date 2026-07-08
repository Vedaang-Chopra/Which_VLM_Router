import time
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI

from .models import ModelEndpoint, load_endpoints_from_config
from .messages import build_messages, ImageLike

class VLMClient:
    def __init__(self, endpoints: List[ModelEndpoint], max_workers: int = 4):
        self.endpoints = endpoints
        self.max_workers = max_workers
        self.endpoint_map = defaultdict(list)
        
        # Initialize specialized clients
        self.clients = {}
        for ep in endpoints:
            self.endpoint_map[ep.name].append(ep)
            self.clients[ep.base_url] = OpenAI(api_key=ep.api_key, base_url=ep.base_url)

    def _get_client_and_endpoint(self, model_name: str):
        eps = self.endpoint_map.get(model_name)
        if not eps:
            raise ValueError(f"Unknown model: {model_name}")
        # Random for simple "load balancing"
        ep = random.choice(eps)
        return self.clients[ep.base_url], ep

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        try:
            client, ep = self._get_client_and_endpoint(model_name)
            
            # DeepSeek OCR fallback for completions endpoint if needed
            is_deepseek_ocr = "deepseek" in model_name.lower() and "ocr" in model_name.lower()
            
            start = time.perf_counter()
            
            if is_deepseek_ocr:
                # Convert messages to prompt for legacy completion endpoint
                prompt = ""
                for m in messages:
                    content = m.get('content', '')
                    if isinstance(content, list):
                        # Naive text extraction
                        text_parts = [c['text'] for c in content if c['type'] == 'text']
                        content = " ".join(text_parts)
                    prompt += f"{m['role']}: {content}\n"
                
                resp = client.completions.create(
                    model=ep.model_id,
                    prompt=prompt,
                    **{**ep.extra_params, **kwargs}
                )
                text = resp.choices[0].text
            else:
                resp = client.chat.completions.create(
                    model=ep.model_id,
                    messages=messages,
                    **{**ep.extra_params, **kwargs}
                )
                text = resp.choices[0].message.content

            lat = (time.perf_counter() - start) * 1000
            
            return {
                "ok": True,
                "model": model_name,
                "response_text": text,
                "latency_ms": lat,
                "usage": resp.usage.model_dump() if resp.usage else {},
            }
            
        except Exception as e:
            return {
                "ok": False,
                "model": model_name,
                "error": str(e)
            }

    def generate(
        self, 
        prompt: str, 
        image: Optional[ImageLike] = None, 
        model: str = "gemma_3_27b",
        **kwargs
    ) -> Dict[str, Any]:
        """Simple convenience wrapper."""
        msgs = build_messages(prompt=prompt, images=image)
        return self.chat(model, msgs, **kwargs)

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]], # [{'prompt':..., 'image':..., 'model':...}]
    ) -> List[Dict[str, Any]]:
        """Run batch types in parallel."""
        results = [None] * len(inputs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as exc:
            futures = {}
            for i, inp in enumerate(inputs):
                model = inp.get('model', 'gemma_3_27b')
                msgs = build_messages(prompt=inp.get('prompt'), images=inp.get('image'))
                futures[exc.submit(self.chat, model, msgs)] = i
                
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results
