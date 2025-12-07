import json
import random
import re
import threading
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("VLM_JUDGE")

# The detailed VLM Judge Prompt
VLM_JUDGE_PROMPT = """You are an expert multimodal evaluator. Your job is to score and rank multiple candidate answers
to a question about an image.

You will be given:
- The question
- A ground truth reference answer (optional but highly recommended)
- An image
- Four candidate answers labeled A, B, C, and D (the order is randomized)

Your tasks:

1. Evaluate each answer on these criteria:
   - Correctness relative to the image and question
   - Alignment with the ground truth (if provided)
   - Completeness (does it cover all important details?)
   - Visual grounding (avoid hallucinations)
   - Clarity and precision

2. For EACH of A, B, C, D, assign a score from **0 to 10**:
   - 10 = fully correct, complete, and grounded
   - 7  = mostly correct, minor issues
   - 4  = partially correct or missing major details
   - 1  = mostly wrong or hallucinated
   - 0  = entirely incorrect

3. Produce a ranking from best to worst.
   - Ties are allowed
   - Format the ranking as a list of groups:
     Example: [["B"], ["A"], ["C","D"]]

4. Output ONLY valid JSON in this format:

{
  "scores": {
    "A": <0-10>,
    "B": <0-10>,
    "C": <0-10>,
    "D": <0-10>
  },
  "ranking": [
    [...],
    [...]
  ]
}

Do NOT include any explanations outside the JSON."""


class VLMJudge:
    """
    Listwise VLM Judge using any VLM model (Llama Scout, Molmo, etc).
    
    Evaluates multiple model answers for a single (image, question) sample.
    Sends the image + all candidate answers + ground truth to the VLM.
    Returns per-model scores (0-10) and ranking groups.
    """
    
    def __init__(
        self,
        runner: Any,  # OpenAIStyleRunner
        model_names: List[str],  # List of model names for load balancing
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ):
        self.runner = runner
        self.model_names = model_names
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._model_cycle = 0
        self._lock = threading.Lock()

    def _get_next_model(self) -> str:
        """Thread-safe round-robin selection of model."""
        with self._lock:
            if not self.model_names:
                raise ValueError("No model names configured for VLMJudge")
            model = self.model_names[self._model_cycle % len(self.model_names)]
            self._model_cycle += 1
            return model

    def evaluate_listwise(
        self,
        image_url: Optional[str],
        question: str,
        ground_truth: str,
        answers_dict: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Evaluate multiple model answers for one sample.
        
        Args:
            image_url: Base64 data URL or http URL for the image
            question: The question text
            ground_truth: The reference answer
            answers_dict: {model_name: answer_text} for all models
            
        Returns:
            {
                "per_model": {
                    model_name: {"score": float, "rank_group": int},
                    ...
                },
                "raw_json": <parsed response>,
                "error": <if failed>
            }
        """
        if len(answers_dict) < 2:
            return {"error": "need_at_least_2_answers"}
        
        # 1. Randomize order and assign letters
        model_names = list(answers_dict.keys())
        random.shuffle(model_names)
        
        candidates_text = ""
        letter_map = {}  # letter -> model_name
        for idx, model_name in enumerate(model_names):
            letter = chr(65 + idx)  # A, B, C, D...
            letter_map[letter] = model_name
            answer = answers_dict[model_name]
            candidates_text += f"Answer {letter}:\n{answer}\n\n"
        
        # 2. Build combined prompt
        full_prompt = f"""{VLM_JUDGE_PROMPT}

---

Question: {question}

Ground Truth: {ground_truth}

Candidate Answers:
{candidates_text}"""
        
        # 3. Build messages - image FIRST in content array
        if image_url:
            messages = [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": full_prompt},
                ]}
            ]
        else:
            messages = [
                {"role": "user", "content": full_prompt}
            ]
        
        # 4. Call API (deterministic generation)
        model_name = self._get_next_model()
        result = self.runner.chat(
            model_name=model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        
        if not result.get('ok'):
            return {"error": result.get('error_message', 'API call failed')}
        
        raw_content = result.get('response_text', '')
        
        # 5. Parse JSON with robust cleanup
        data = None
        scores = {}
        ranking = []
        
        try:
            cleaned = raw_content.replace("```json", "").replace("```", "").strip()
            # Find JSON boundaries
            if "{" in cleaned:
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                if end > start:
                    cleaned = cleaned[start:end]
            data = json.loads(cleaned)
            scores = data.get("scores", {})
            ranking = data.get("ranking", [])
        except json.JSONDecodeError as e:
            # Try regex extraction as fallback
            logger.warning(f"JSON parse failed, trying regex. Raw: {raw_content[:200]}...")
            
            # Extract scores with regex: "A": 8 or "A": "8"
            score_pattern = r'["\']([A-Z])["\']\s*:\s*([\d.]+)'
            score_matches = re.findall(score_pattern, raw_content)
            if score_matches:
                scores = {letter: float(score) for letter, score in score_matches}
                logger.info(f"Extracted scores via regex: {scores}")
            else:
                logger.warning(f"Could not parse VLM output. Raw: {raw_content[:300]}")
                return {"error": "json_parse_error", "raw_content": raw_content[:500]}
        
        # 6. Map back to model names
        per_model_result = {}
        for letter, score in scores.items():
            model_name = letter_map.get(letter.upper())
            if model_name:
                try:
                    per_model_result[model_name] = {
                        "score": float(score),
                        "rank_group": 99  # Default, will be updated from ranking
                    }
                except (TypeError, ValueError):
                    per_model_result[model_name] = {"score": 0.0, "rank_group": 99}
        
        # Parse ranking groups
        current_rank = 1
        for group in ranking:
            if isinstance(group, list):
                for letter in group:
                    model_name = letter_map.get(letter.upper() if isinstance(letter, str) else letter)
                    if model_name and model_name in per_model_result:
                        per_model_result[model_name]["rank_group"] = current_rank
                current_rank += 1
        
        return {
            "per_model": per_model_result,
            "raw_json": data
        }

    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        model_answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Convenience wrapper matching existing interface.
        Extracts fields from sample dict and calls evaluate_listwise.
        """
        result = self.evaluate_listwise(
            image_url=sample.get('image_url'),
            question=sample.get('prompt_text', ''),
            ground_truth=sample.get('ground_truth', ''),
            answers_dict=model_answers,
        )
        
        # Rename fields to match DB column names
        if 'per_model' in result:
            for model_name in result['per_model']:
                scores = result['per_model'][model_name]
                result['per_model'][model_name] = {
                    'judge_vlm_score': scores.get('score'),
                    'judge_vlm_rank_group': scores.get('rank_group'),
                }
        
        return result


# Backward compatibility alias
MolmoJudge = VLMJudge
