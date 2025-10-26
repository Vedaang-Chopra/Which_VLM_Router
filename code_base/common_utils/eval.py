from openai import OpenAI
from typing import Any, Dict, Iterable, List, Optional
from common_utils.image_utils import to_b64, ensure_image_bytes, hf_first_image_bytes, hf_user_and_ref
import time
from dataclasses import dataclass

class EvalClient:
    """Thin wrapper around vLLM/OpenAI Chat Completions for eval only."""
    def __init__(self, base_url: str, api_key: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        ).model_dump()




class EvalMessageBuilder:
    """Builds the multimodal message for eval model."""
    @staticmethod
    def build(user_text: str, img_b64: str) -> List[Dict[str, Any]]:
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }]






@dataclass
class EvalOutput:
    conversation_id: Any
    dataset_model: Any
    user_question: str
    reference_answer: str
    model_answer: str
    latency_sec: float
    images : List

class EvalRunner:
    """Orchestrates: sample → build messages → call eval → return EvalOutput."""
    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client

    def run_on_sample(self, sample: Dict[str, Any]) -> EvalOutput:
        img = ensure_image_bytes(hf_first_image_bytes(sample))
        img_b64 = to_b64(img)
        user_q, ref = hf_user_and_ref(sample)

        messages = EvalMessageBuilder.build(user_q, img_b64)

        t0 = time.perf_counter()
        resp = self.client.chat(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        latency = time.perf_counter() - t0

        model_answer = resp["choices"][0]["message"]["content"]

        return EvalOutput(
            conversation_id=sample.get("conversation_id"),
            dataset_model=sample.get("model"),
            user_question=user_q,
            reference_answer=ref,
            model_answer=model_answer,
            latency_sec=round(latency, 4),
            images = sample.get("images") or []
        )





# test_client= EvalClient(base_url="http://localhost:8000/v1", api_key="EMPTY")
