from typing import Any, Dict, List
from openai import OpenAI
from dataclasses import dataclass
from common_utils.image_utils import to_b64, ensure_image_bytes

class JudgeClient:
    """Thin wrapper around vLLM/OpenAI Chat Completions for judging."""
    def __init__(self, base_url: str, api_key: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        ).model_dump()


from typing import Any, Dict, List

JUDGE_PROMPT = r"""
You are a strict multimodal judge. You will be given:
- TASK_CATEGORY  (e.g., ocr, captioning, entity_recognition, chart_qa, table_extraction, ui_screenshot, diagram, math_reasoning, general_vqa, etc.)
- USER_QUESTION
- IMAGE
- REFERENCE_ANSWER  (original baseline to match unless clearly wrong)
- MODEL_ANSWER      (answer to evaluate)

Your job:
A) First verify REFERENCE_ANSWER against USER_QUESTION and IMAGE. Treat it as the baseline to match unless it is clearly contradicted by the IMAGE.
B) Judge MODEL_ANSWER against USER_QUESTION, IMAGE, and REFERENCE_ANSWER for three qualities:
   1) Correctness – factual consistency with the IMAGE and the USER_QUESTION; no contradictions.
   2) Completeness – covers the key facts/details present in the REFERENCE_ANSWER (and the question intent).
   3) Relevance – avoids off-topic or unsupported details (not visible in the IMAGE or unrelated to the question).

Category-specific priorities (THINK and adapt the importance; do not use a single fixed formula):
- ocr / document / table_extraction / entity_recognition:
  * Priority: Completeness of required fields and exact textual fidelity (numbers, dates, units).
  * Penalize missing fields or wrong strings heavily; minor styling differences are irrelevant.
  * Small bonuses for correctly normalizing formats (e.g., $1,234.00 → 1234.00) if faithful to IMAGE.
- captioning / scene_description:
  * Priority: Correctness of salient objects, attributes, relations; avoid hallucinations.
  * Completeness = coverage of *important* salient elements (not every pixel).
  * Relevance: penalize invented objects/attributes not supported by IMAGE.
- chart_qa / diagram / math_reasoning / ui_screenshot (functional understanding):
  * Priority: Correctness of values, units, relations, and step logic tied to the IMAGE.
  * Completeness = includes the necessary elements (labels, axes, UI field names) to answer the question fully.
  * Relevance: no speculative UI functions or chart trends not present.
- general_vqa:
  * Balance all three; penalize confident but unsupported claims.

Scoring (0–100), category-aware:
- Reason about the three qualities using the category priorities above; choose a score that reflects overall quality for this CATEGORY.
- Use score bands as guidance (not a formula):
  * 95–100: Perfect or near-perfect for this CATEGORY; no contradictions; no missing key facts; may include small, IMAGE-verified improvements.
  * 80–94: Minor issues (small omission OR slight imprecision) but solid overall.
  * 60–79: Noticeable issues (several omissions OR one clear error), still partially useful.
  * 30–59: Major issues; limited usefulness.
  * 0–29: Mostly wrong/irrelevant/refusal.
- Guardrails:
  * If any contradiction with IMAGE on a core fact OR key facts are missing → cap at 99.
  * If MODEL_ANSWER is empty/irrelevant/refusal → 0–10.
  * If MODEL_ANSWER adds new details that are clearly visible and correct, you MAY place it at the top of its band or into 95–100 (still capped at 100).

Output:
Respond ONLY with valid JSON exactly like:
{"score": <int>, "justification": "<one short sentence citing the main reason(s)>"}
Do not include any other text, markdown, or explanations.
""".strip()



class JudgeMessageBuilder:
    @staticmethod
    def build(user_question: str, reference_answer: str, model_answer: str, task_category: str, img_b64: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": [
                {"type": "text", "text":
                f"TASK_CATEGORY:\n{task_category}\n\n"
                f"USER_QUESTION:\n{user_question}\n\n"
                f"REFERENCE_ANSWER:\n{reference_answer}\n\n"
                f"MODEL_ANSWER:\n{model_answer}\n\n"
                "Judge MODEL_ANSWER for this category with respect to USER_QUESTION, IMAGE, and REFERENCE_ANSWER, "
                "and return only the JSON with a 0–100 score and a one-sentence justification."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
            },
        ]



import json, time
from typing import Optional, Tuple

class LLMJudge:
    def __init__(self, client, model: str, *, max_retries: int = 3, backoff: float = 0.6, max_tokens: int = 128):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_tokens = max_tokens

    def judge(self, user_q: str, reference: str, model_answer: str, task_category: str, img_b64: str) -> Tuple[Optional[int], str]:
        messages = JudgeMessageBuilder.build(user_q, reference, model_answer, task_category, img_b64)
        last_err = None
        for i in range(self.max_retries):
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    top_p=1,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = resp["choices"][0]["message"]["content"]
                obj = json.loads(content)
                score = int(obj.get("score"))
                just = obj.get("justification", "")
                return score, just
            except Exception as e:
                last_err = e
                time.sleep(self.backoff * (i + 1))
        return None, f"JudgeError: {last_err}" if last_err else "JudgeError"



@dataclass
class JudgedOutput:
    # from eval
    conversation_id: Any
    dataset_model: Any
    category: str
    user_question: str
    reference_answer: str
    model_answer: str
    latency_sec: float
    # judge
    judge_model: str
    judge_score: Optional[int]
    judge_justification: str




from typing import Dict, List, Tuple

class CategoryJudgeRunner:
    def __init__(self, judge: LLMJudge, *, id_key: str = "conversation_id", verbose: bool = True):
        self.judge = judge
        self.id_key = id_key
        self.verbose = verbose

    def _build_rec_index(self, buckets: Dict[str, List[dict]]) -> Dict[str, dict]:
        idx = {}
        for recs in buckets.values():
            for r in recs:
                rid = r.get(self.id_key)
                if rid is not None:
                    idx[rid] = r
        return idx

    def _first_img_b64(self, rec: dict) -> str:
        imgs = rec.get("images") or []
        if imgs:
            b = imgs[0].get("bytes")
            if isinstance(b, list):
                b = bytes(b)
        else:
            b = None
        b = ensure_image_bytes(b)
        return to_b64(b)

    def run(
        self,
        results_by_cat: Dict[str, List],   # List[EvalOutput]
        buckets: Dict[str, List[dict]],
        *,
        judge_model_name: str,
    ) -> Tuple[Dict[str, List[JudgedOutput]], List[JudgedOutput]]:
        """Returns (judged_by_cat, judged_all_flat)."""
        if self.verbose:
            print("[JUDGE] Starting judging pass...")

        rec_index = self._build_rec_index(buckets)
        judged_by_cat: Dict[str, List[JudgedOutput]] = {}
        judged_all: List[JudgedOutput] = []

        for cat, eval_outputs in results_by_cat.items():
            if self.verbose:
                print(f"[JUDGE] Category '{cat}' → {len(eval_outputs)} item(s)")
            cat_list: List[JudgedOutput] = []
            for i, out in enumerate(eval_outputs, 1):
                rec = rec_index.get(out.conversation_id)
                if rec is None:
                    if self.verbose:
                        print(f"  - [{cat}] {i}/{len(eval_outputs)} conv_id={out.conversation_id}  WARN: raw record not found; skipping")
                    continue
                img_b64 = self._first_img_b64(rec)

                score, just = self.judge.judge(
                    user_q=out.user_question,
                    reference=out.reference_answer,
                    model_answer=out.model_answer,
                    task_category=cat,
                    img_b64=img_b64,
                )

                j = JudgedOutput(
                    conversation_id=out.conversation_id,
                    dataset_model=out.dataset_model,
                    category=cat,
                    user_question=out.user_question,
                    reference_answer=out.reference_answer,
                    model_answer=out.model_answer,
                    latency_sec=out.latency_sec,
                    judge_model=judge_model_name,
                    judge_score=score,
                    judge_justification=just or "",
                )
                cat_list.append(j)
                judged_all.append(j)

                if self.verbose:
                    print(f"  - [{cat}] {i}/{len(eval_outputs)} conv_id={out.conversation_id} score={score} just={str(just)[:90]}")

            judged_by_cat[cat] = cat_list

        if self.verbose:
            total = sum(len(v) for v in judged_by_cat.values())
            print(f"[JUDGE] Done. Judged {total} items across {len(judged_by_cat)} categories.")
        return judged_by_cat, judged_all
