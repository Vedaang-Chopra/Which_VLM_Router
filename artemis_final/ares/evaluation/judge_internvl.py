import json
import random
from typing import Dict, List, Any, Optional

class InternVLJudge:
    """
    Evaluator using InternVL3-78B (or similar) as a listwise ranker.
    Uses inference_engine.runners.OpenAIStyleRunner for robust API calls.
    """
    def __init__(
        self,
        runner: Any, # OpenAIStyleRunner
        model_name: str = "internvl_judge_model", 
        temperature: float = 0.0,
    ):
        self.runner = runner
        self.model_name = model_name
        self.temperature = temperature

    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        model_answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample with multiple model answers.
        """
        
        # 1. Prepare candidates
        model_names = list(model_answers.keys())
        random.shuffle(model_names)
        
        candiates_text = ""
        letter_map = {} 
        for idx, model_name in enumerate(model_names):
            letter = chr(65 + idx) # A, B, C...
            letter_map[letter] = model_name
            answer = model_answers[model_name]
            candiates_text += f"Answer {letter}:\n{answer}\n\n"

        # 2. Build Prompt
        question = sample.get('prompt_text', '')
        ground_truth = sample.get('ground_truth', '')
        
        system_prompt = (
            "You are an expert AI assistant taking the role of an impartial judge. "
            "You will be given a question, (optionally an image), a ground truth answer, and several candidate answers. "
            "Your task is to evaluate each candidate answer based on correctness, completeness, and clarity.\n"
            "Score each answer from 0 to 10.\n"
            "Also rank the answers into groups (Best to Worst).\n"
            "Return the result ONLY as a JSON object."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Ground Truth: {ground_truth}\n\n"
            f"Candidate Answers:\n{candiates_text}\n"
            "Please evaluate the answers. Return a JSON object with the following format:\n"
            "{\n"
            '  "scores": { "A": <score>, "B": <score>, ... },\n'
            '  "ranking": [["A"], ["B", "C"], ["D"]] \n'
            "}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Handle image if present (assuming runner supports standard content blocks)
        if 'image_url' in sample and sample['image_url']:
             messages[1]["content"] = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": sample['image_url']}}
             ]
        elif 'image' in sample: # raw image or bytes
             # If the runner supports passing raw images, we'd do it here. 
             # For now assume image_url or text-only if local-ish
             pass

        # 3. Call API via Runner
        # The runner handles retries and errors
        result = self.runner.chat(
            model_name=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
            response_format={"type": "json_object"} # forward this if supported
        )
        
        if not result['ok']:
            return {"error": result.get('error_message', 'Unknown error')}

        raw_content = result['response_text']

        # 4. Parse JSON
        try:
            cleaned = raw_content.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            scores = data.get("scores", {})
            ranking = data.get("ranking", [])
        except json.JSONDecodeError:
            return {"error": "json_parse_error", "raw_content": raw_content}

        # 5. Map back
        per_model_result = {}
        for letter, score in scores.items():
            model_name = letter_map.get(letter)
            if model_name:
                per_model_result[model_name] = {
                    "judge_internvl_score": float(score),
                    "judge_internvl_rank_group": 99
                }
        
        current_rank = 1
        for group in ranking:
            for letter in group:
                model_name = letter_map.get(letter)
                if model_name and model_name in per_model_result:
                    per_model_result[model_name]["judge_internvl_rank_group"] = current_rank
            current_rank += 1

        return {
            "per_model": per_model_result,
            "raw_json": data
        }
