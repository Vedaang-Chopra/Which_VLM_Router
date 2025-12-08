
import re
from typing import Any, Dict, List, Optional, Tuple, Union
# from rouge_score import rouge_scorer

# scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)



# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

class Scorer:
    """Compute various scoring metrics for VLM outputs."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        text = text.lower().strip()
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def exact_match(pred: str, gt: str) -> float:
        """Exact string match (case-insensitive, stripped)."""
        if not pred or not gt:
            return 0.0
        return 1.0 if pred.lower().strip() == gt.lower().strip() else 0.0
    
    @staticmethod
    def exact_match_normalized(pred: str, gt: str) -> float:
        """Exact match after full normalization."""
        pred_norm = Scorer.normalize_text(pred)
        gt_norm = Scorer.normalize_text(gt)
        return 1.0 if pred_norm == gt_norm else 0.0
    
    @staticmethod
    def contains_match(pred: str, gt: str) -> float:
        """Check if normalized GT is contained in normalized prediction."""
        if not pred or not gt:
            return 0.0
        pred_norm = Scorer.normalize_text(pred)
        gt_norm = Scorer.normalize_text(gt)
        return 1.0 if gt_norm in pred_norm else 0.0
    
    @staticmethod
    def gt_in_response(pred: str, gt: str) -> float:
        """Check if GT appears in response (less strict)."""
        if not pred or not gt:
            return 0.0
        return 1.0 if gt.lower().strip() in pred.lower() else 0.0
    
    @staticmethod
    def token_f1(pred: str, gt: str) -> float:
        """Compute token-level F1 score."""
        if not pred or not gt:
            return 0.0
            
        pred_tokens = set(Scorer.normalize_text(pred).split())
        gt_tokens = set(Scorer.normalize_text(gt).split())
        
        if not pred_tokens or not gt_tokens:
            return 0.0
            
        common = pred_tokens & gt_tokens
        if not common:
            return 0.0
            
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def numeric_match(pred: str, gt: str, tolerance: float = 0.01) -> Optional[float]:
        """Match numeric answers with tolerance."""
        def extract_number(s: str) -> Optional[float]:
            if not s:
                return None
            # Try to find numbers in the string
            matches = re.findall(r'-?\d+\.?\d*', s.replace(',', ''))
            if matches:
                try:
                    return float(matches[0])
                except:
                    return None
            return None
        
        pred_num = extract_number(pred)
        gt_num = extract_number(gt)
        
        if pred_num is None or gt_num is None:
            return None
            
        if gt_num == 0:
            return 1.0 if pred_num == 0 else 0.0
            
        rel_diff = abs(pred_num - gt_num) / abs(gt_num)
        return 1.0 if rel_diff <= tolerance else 0.0
    
    @staticmethod
    def extract_mc_letter(text: str) -> Optional[str]:
        """Extract multiple choice letter (A, B, C, D) from text."""
        if not text:
            return None
            
        patterns = [
            r"(?:answer|choice)[\s:]*([A-D])",  # "Answer: A" or "choice A"
            r"^\s*\(?([A-D])\)?[\s\.\):]",      # "(A)" or "A." at start
            r"\(([A-D])\)",                      # "(A)" anywhere
            r"^([A-D])$",                        # Just the letter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.strip(), re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Last resort: first capital letter A-D in the response
        for char in text[:50]:
            if char.upper() in 'ABCD':
                return char.upper()
                
        return None
    
    @staticmethod
    def mc_letter_match(pred: str, gt: str) -> Optional[float]:
        """Match multiple choice letter answers."""
        pred_letter = Scorer.extract_mc_letter(pred)
        gt_letter = Scorer.extract_mc_letter(gt)
        
        if pred_letter is None or gt_letter is None:
            return None
            
        return 1.0 if pred_letter == gt_letter else 0.0
    
    @staticmethod
    def is_refusal(response: str) -> bool:
        """Detect if model refused to answer."""
        if not response:
            return False
        response_lower = response.lower()
        refusal_phrases = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i don't have", "i do not have",
            "sorry, i", "i apologize",
            "as an ai", "as a language model",
            "i'm not able", "i am not able",
        ]
        return any(phrase in response_lower for phrase in refusal_phrases)
    
    @classmethod
    def compute_all_scores(
        cls, 
        pred: str, 
        gt: str, 
        gt_type: str = "exact"
    ) -> Dict[str, Any]:
        """Compute all relevant scores for a prediction."""
        scores = {
            'score_exact_match': cls.exact_match(pred, gt),
            'score_exact_match_normalized': cls.exact_match_normalized(pred, gt),
            'score_contains_gt': cls.contains_match(pred, gt),
            'score_gt_in_response': cls.gt_in_response(pred, gt),
            'score_f1': cls.token_f1(pred, gt),
            'score_numeric_match': None,
            'score_mc_letter_match': None,
            'pred_answer_letter': None,
            'gt_answer_letter': None,
            'is_refusal': cls.is_refusal(pred),
        }
        
        # Task-specific scores
        if gt_type == 'numeric':
            scores['score_numeric_match'] = cls.numeric_match(pred, gt)
            
        if gt_type == 'mc':
            scores['score_mc_letter_match'] = cls.mc_letter_match(pred, gt)
            scores['pred_answer_letter'] = cls.extract_mc_letter(pred)
            scores['gt_answer_letter'] = cls.extract_mc_letter(gt)
        
        # Determine if correct based on GT type
        if gt_type == 'numeric' and scores['score_numeric_match'] is not None:
            scores['is_correct'] = scores['score_numeric_match'] >= 0.99
        elif gt_type == 'mc' and scores['score_mc_letter_match'] is not None:
            scores['is_correct'] = scores['score_mc_letter_match'] >= 0.99
        elif gt_type == 'freeform':
            scores['is_correct'] = scores['score_f1'] >= 0.5
        else:
            scores['is_correct'] = scores['score_exact_match_normalized'] >= 0.99 or scores['score_contains_gt'] >= 0.99
            
        return scores



import json
from typing import Callable, List, Dict, Any, Tuple, Optional


class SemanticF1Evaluator:
    """
    LLM/VLM-as-a-judge evaluator for free-form answers.

    Strategy:
      - Extract atomic statements from model answer and ground truth(s)
      - Match them to compute recall (which ground-truth facts were covered?)
      - Check consistency of each answer statement vs. ground truth to compute precision
      - Combine into F1

    You must provide `chat_fn`, a function:
        chat_fn(messages: List[Dict[str, str]], max_tokens: int) -> str

    where `messages` is OpenAI-style:
        [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    and the function returns the model's text response.
    """

    def __init__(
        self,
        chat_fn: Callable[[List[Dict[str, str]], int], str],
    ):
        self.chat_fn = chat_fn

    # ---------- Helper: JSON-safe parsing ----------
        # ---------- New helpers for QA context ----------

    @staticmethod
    def _build_answer_text(question: str, answer: str) -> str:
        return f"Question: {question}\nModel answer: {answer}"

    @staticmethod
    def _build_reference_text(question: str, reference: str) -> str:
        return f"Question: {question}\nCorrect answer: {reference}"


    @staticmethod
    def _extract_json_block(raw: str) -> str:
        """
        Try to pull out the JSON block from a noisy LLM response.
        """
        start = raw.find("{")
        bracket_start = raw.find("[")
        # Prefer { if both present at valid positions
        if start != -1 and (bracket_start == -1 or start < bracket_start):
            open_char, close_char = "{", "}"
        else:
            open_char, close_char = "[", "]"
            start = bracket_start

        if start == -1:
            return raw  # fallback

        end = raw.rfind(close_char)
        if end == -1:
            return raw
        return raw[start : end + 1]

    # ---------- Step 1: Atomic statement extraction ----------

    def extract_atomic_statements(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Use judge LLM to extract atomic factual statements from arbitrary text.
        Returns a list of strings.
        """
        system_prompt = (
            "You are an evaluation assistant. "
            "Given some text (typically a question with an answer, or references), "
            "extract a list of atomic factual statements.\n\n"
            "Rules:\n"
            "- Each statement must describe a semantic fact about the world or the task, "
            "  not about characters, spelling, formatting, or punctuation.\n"
            "- Do NOT create statements like 'The text contains the letter N' or "
            "  'The text contains a period'.\n"
            "- If the answer is short (e.g., a single word), convert it into a full factual "
            "  statement using the question context (e.g., 'The capital of France is Paris.').\n\n"
            "Return ONLY valid JSON: a list of strings."
        )
        user_prompt = (
            "Text:\n```"
            + text
            + "```\n\nReturn a JSON list of the atomic factual statements."
        )

        raw = self.chat_fn(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )

        try:
            json_str = self._extract_json_block(raw)
            statements = json.loads(json_str)
        except Exception as e:
            print("JSON parse error in extract_atomic_statements:", e)
            print("Raw output:", raw)
            statements = []

        clean = [s.strip() for s in statements if isinstance(s, str) and s.strip()]
        return clean

    @staticmethod
    def _combine_references(references: List[str]) -> str:
        """
        Combine multiple reference answers / transcripts into one block.
        """
        if len(references) == 1:
            return references[0]
        return "\n".join(f"[REF {i+1}] {r}" for i, r in enumerate(references))

    # ---------- Step 2: Matching for recall ----------

    def match_statements_for_recall(
        self,
        gen_statements: List[str],
        gt_statements: List[str],
        max_tokens: int = 1024,
    ) -> List[Tuple[int, int]]:
        """
        Ask the judge LLM to match generated statements to ground-truth statements.

        Returns a list of (gen_index, gt_index) pairs representing true positive matches.
        """
        system_prompt = (
            "You are a strict evaluator. "
            "You are given two lists of atomic factual statements:\n"
            "- List A: statements from a model's answer\n"
            "- List B: statements from the reference answer(s)\n\n"
            "Your task is to decide which statements from A convey the same factual content as statements in B. "
            "Match only if they agree on the core fact (entities, attributes, relations). "
            "Return ONLY JSON with a key 'matches', whose value is a list of [a_index, b_index] integer pairs."
        )

        payload = {
            "A_gen_statements": gen_statements,
            "B_reference_statements": gt_statements,
        }

        raw = self.chat_fn(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Here are the two lists as JSON:\n```"
                    + json.dumps(payload, ensure_ascii=False)
                    + "```",
                },
            ],
            max_tokens=max_tokens,
        )

        try:
            json_str = self._extract_json_block(raw)
            data = json.loads(json_str)
            matches = data.get("matches", [])
        except Exception as e:
            print("JSON parse error in match_statements_for_recall:", e)
            print("Raw output:", raw)
            matches = []

        normalized = []
        for pair in matches:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and isinstance(pair[0], int)
                and isinstance(pair[1], int)
            ):
                if 0 <= pair[0] < len(gen_statements) and 0 <= pair[1] < len(gt_statements):
                    normalized.append((pair[0], pair[1]))

        return normalized

    @staticmethod
    def compute_recall(matches: List[Tuple[int, int]], num_gt: int) -> float:
        """
        Recall = (# unique ground-truth statements matched) / (total ground-truth statements)
        """
        if num_gt == 0:
            return 0.0
        tp = len(set(gt_idx for _, gt_idx in matches))
        return tp / num_gt

    # ---------- Step 3: Consistency check for precision ----------

    def check_consistency_batch(
        self,
        gen_statements: List[str],
        references: List[str],
        max_tokens: int = 1024,
    ) -> List[str]:
        """
        For each generated statement, label it as:
          - 'consistent'
          - 'inconsistent'
          - 'unknown'

        using the combined reference text.
        """
        combined_refs = self._combine_references(references)

        system_prompt = (
            "You are an evaluation assistant. "
            "You are given:\n"
            "1) Reference answer(s) describing the correct information.\n"
            "2) A list of atomic statements from a model's answer.\n\n"
            "For each statement, decide if it is:\n"
            "- 'consistent'   : clearly supported or implied by the references.\n"
            "- 'inconsistent' : contradicted or clearly not true given the references.\n"
            "- 'unknown'      : not contradicted, but not supported (missing information).\n\n"
            "Return ONLY JSON with key 'labels' whose value is a list of these strings, "
            "in the same order as the input statements."
        )

        payload = {
            "reference_text": combined_refs,
            "statements": gen_statements,
        }

        raw = self.chat_fn(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Here is the data as JSON:\n```"
                    + json.dumps(payload, ensure_ascii=False)
                    + "```",
                },
            ],
            max_tokens=max_tokens,
        )

        try:
            json_str = self._extract_json_block(raw)
            data = json.loads(json_str)
            labels = data.get("labels", [])
        except Exception as e:
            print("JSON parse error in check_consistency_batch:", e)
            print("Raw output:", raw)
            labels = []

        labels = [str(lbl).strip().lower() for lbl in labels]
        # normalize length
        if len(labels) < len(gen_statements):
            labels += ["unknown"] * (len(gen_statements) - len(labels))
        elif len(labels) > len(gen_statements):
            labels = labels[: len(gen_statements)]

        return labels

    @staticmethod
    def compute_precision(labels: List[str]) -> float:
        """
        Precision = TP / (TP + FP)
        TP = 'consistent'
        FP = 'inconsistent'
        'unknown' is ignored.
        """
        tp = sum(1 for l in labels if l == "consistent")
        fp = sum(1 for l in labels if l == "inconsistent")
        denom = tp + fp
        if denom == 0:
            return 0.0
        return tp / denom

    @staticmethod
    def compute_f1(precision: float, recall: float) -> float:
        if precision == 0.0 and recall == 0.0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # ---------- Single-answer API ----------

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        references: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a single model answer against one or more references for a QA-style task.
        `question` is included so the judge can turn short labels into full facts.
        """

        # ---- Step 1: atomic statements with question context ----
        # model side
        answer_text = self._build_answer_text(question, answer)
        gen_statements = self.extract_atomic_statements(answer_text)

        # reference side
        ref_texts = [self._build_reference_text(question, r) for r in references]
        combined_refs = self._combine_references(ref_texts)
        gt_statements = self.extract_atomic_statements(combined_refs)

        # ---- Step 2: recall ----
        matches = self.match_statements_for_recall(gen_statements, gt_statements)
        recall = self.compute_recall(matches, num_gt=len(gt_statements))

        # ---- Step 3: precision (also include question in consistency check) ----
        # Use reference *answers* (already semantically tied via question text)
        labels = self.check_consistency_batch(gen_statements, [combined_refs])
        precision = self.compute_precision(labels)

        # ---- Step 4: F1 ----
        f1 = self.compute_f1(precision, recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gen_statements": gen_statements,
            "gt_statements": gt_statements,
            "matches": matches,
            "labels": labels,
        }


    # ---------- Multi-model helpers ----------

    def evaluate_example_multi_model(
        self,
        question: str,
        model_answers: Dict[str, str],
        references: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models' answers for a single example.
        model_answers: {model_name: answer_text}
        returns: {model_name: metrics_dict_from_evaluate_answer}
        """
        results = {}
        for model_name, answer in model_answers.items():
            results[model_name] = self.evaluate_answer(question, answer, references)
        return results
        

    def evaluate_dataset_multi_model(
        self,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Each item:
        {
            "id": ...,
            "question": "...",
            "references": [...],
            "model_answers": {model_name: answer, ...}
        }
        """
        per_example = []
        model_sums = {}

        for item in dataset:
            ex_id = item.get("id")
            question = item["question"]
            references = item["references"]
            model_answers = item["model_answers"]

            per_model = self.evaluate_example_multi_model(
                question=question,
                model_answers=model_answers,
                references=references,
            )

            per_example.append(
                {
                    "id": ex_id,
                    "per_model": per_model,
                }
            )

            for model_name, metrics in per_model.items():
                if model_name not in model_sums:
                    model_sums[model_name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0}
                model_sums[model_name]["precision"] += metrics["precision"]
                model_sums[model_name]["recall"] += metrics["recall"]
                model_sums[model_name]["f1"] += metrics["f1"]
                model_sums[model_name]["count"] += 1

        per_model_avg = {
            m: {
                "precision": v["precision"] / v["count"],
                "recall": v["recall"] / v["count"],
                "f1": v["f1"] / v["count"],
                "count": v["count"],
            }
            for m, v in model_sums.items()
        }

        return {
            "per_example": per_example,
            "per_model_average": per_model_avg,
        }






GLIDER_PROMPT_TEMPLATE = """Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format. 
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
{data}

Pass Criteria:
{pass_criteria}

Rubric:
{rubric}

Your output must in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
"""


import re
from typing import Tuple, List

GLIDER_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
GLIDER_HIGHLIGHT_RE = re.compile(r"<highlight>\s*(.*?)\s*</highlight>", re.DOTALL | re.IGNORECASE)
GLIDER_SCORE_RE     = re.compile(r"<score>\s*(.*?)\s*</score>", re.DOTALL | re.IGNORECASE)

def parse_glider_output(text: str) -> Tuple[str, List[str], int]:
    """
    Parse GLIDER output into (reasoning, highlight_list, score_int).
    Be forgiving if the format is slightly off.
    """
    reasoning_match = GLIDER_REASONING_RE.search(text or "")
    highlight_match = GLIDER_HIGHLIGHT_RE.search(text or "")
    score_match     = GLIDER_SCORE_RE.search(text or "")

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    highlight_raw = highlight_match.group(1).strip() if highlight_match else ""
    score_raw = score_match.group(1).strip() if score_match else ""

    # crude highlight parsing: split by newline or comma
    if "[" in highlight_raw and "]" in highlight_raw:
        # try to remove brackets
        highlight_raw = highlight_raw.strip("[]")
    highlight_items = [h.strip().strip('"').strip("'") for h in re.split(r"[\n,]", highlight_raw) if h.strip()]

    try:
        score = int(re.findall(r"-?\d+", score_raw)[0])
    except Exception:
        score = 0  # fallback

    return reasoning, highlight_items, score



from typing import Dict, Any, Optional

class GliderEvaluator:
    """
    GLIDER-based evaluator for QA / TextVQA / Molmo-style tasks.

    Instead of computing semantic F1 via atomic statements, this class:
      - Builds a structured data block (question, GT, model answer, optional context)
      - Applies a pass criteria + rubric
      - Calls GLIDER and parses <reasoning>, <highlight>, <score>
    """

    def __init__(
        self,
        chat_fn,
        base_prompt: str = GLIDER_PROMPT_TEMPLATE,
        model_name: str = "PatronusAI/glider",
        max_tokens: int = 1024,
    ):
        self.chat_fn = chat_fn
        self.base_prompt = base_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens

    # ---- 1. Build the "data" block that GLIDER will evaluate ----

    def build_data_block(
        self,
        *,
        question: str,
        model_answer: str,
        ground_truth: str,
        sample_id: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> str:
        """
        Create the <USER INPUT> / <MODEL OUTPUT> style block recommended by GLIDER.
        You can customize this to include OCR, transcripts, etc.
        """
        parts = []

        if sample_id is not None:
            parts.append(f"[EXAMPLE ID]: {sample_id}")

        parts.append("\n[QUESTION]:")
        parts.append(question.strip())

        parts.append("\n[GROUND TRUTH ANSWER]:")
        parts.append(ground_truth.strip())

        if extra_context:
            parts.append("\n[ADDITIONAL CONTEXT]:")
            parts.append(extra_context.strip())

        parts.append("\n[MODEL OUTPUT]:")
        parts.append(model_answer.strip())

        block = "\n".join(parts)
        return f"<USER INPUT>\n{block}\n</USER INPUT>"

    # ---- 2. Default pass criteria & rubric for "answer correctness" ----

    @staticmethod
    def default_pass_criteria() -> str:
        return (
            "- The model's answer should state the correct answer as given in the ground truth.\n"
            "- The answer must not contradict the ground truth.\n"
            "- The answer should not introduce unsupported hallucinated facts.\n"
            "- The answer should be concise and focused on the question.\n"
        )

    @staticmethod
    def default_rubric() -> str:
        return (
            "Score 5: The answer is exactly correct, fully consistent with the ground truth, "
            "and does not contain hallucinated or irrelevant information.\n"
            "Score 4: The answer is semantically correct and consistent with the ground truth, "
            "with only minor formatting or wording differences.\n"
            "Score 3: The answer is partially correct or somewhat vague but mostly aligned with the ground truth.\n"
            "Score 1-2: The answer is weakly related to the ground truth, contains noticeable errors or hallucinations.\n"
            "Score 0: The answer is incorrect, contradictory to the ground truth, or completely unrelated.\n"
        )

    # ---- 3. Build final prompt ----

    def build_prompt(
        self,
        data: str,
        pass_criteria: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> str:
        if pass_criteria is None:
            pass_criteria = self.default_pass_criteria()
        if rubric is None:
            rubric = self.default_rubric()
        return self.base_prompt.format(
            data=data,
            pass_criteria=pass_criteria,
            rubric=rubric,
        )

    # ---- 4. Main API: evaluate a single example ----

    def evaluate(
        self,
        *,
        question: str,
        model_answer: str,
        ground_truth: str,
        sample_id: Optional[str] = None,
        extra_context: Optional[str] = None,
        pass_criteria: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate one QA example. Returns:
          {
            "score": int,
            "reasoning": str,
            "highlight": List[str],
            "raw_output": str,
          }
        """
        data_block = self.build_data_block(
            question=question,
            model_answer=model_answer,
            ground_truth=ground_truth,
            sample_id=sample_id,
            extra_context=extra_context,
        )

        prompt = self.build_prompt(
            data=data_block,
            pass_criteria=pass_criteria,
            rubric=rubric,
        )

        messages = [{"role": "user", "content": prompt}]
        raw_out = self.chat_fn(
            messages=messages,
            max_tokens=self.max_tokens,
            model_name=self.model_name,
        )

        reasoning, highlight, score = parse_glider_output(raw_out)

        return {
            "score": score,
            "reasoning": reasoning,
            "highlight": highlight,
            "raw_output": raw_out,
        }


class Evaluator:
    """
    Main evaluator facade for Ares.
    Wraps Scorer to provide simple evaluation interface for sample records.
    """
    
    def evaluate_single(self, sample: Any) -> Dict[str, Any]:
        """
        Run basic scoring on a sample record.
        
        Args:
            sample: SampleRecord object or dictionary containing:
                   - response_parsed (or response_raw)
                   - ground_truth
                   - ground_truth_type (optional, default='exact')
                   
        Returns:
            Dictionary of scores (e.g. {'score_exact_match': 1.0, ...})
        """
        # Determine prediction text
        pred = ""
        if hasattr(sample, 'response_parsed') and sample.response_parsed:
            pred = sample.response_parsed
        elif hasattr(sample, 'response_raw') and sample.response_raw:
            pred = sample.response_raw
        elif isinstance(sample, dict):
            pred = sample.get('response_parsed') or sample.get('response_raw') or ""
            
        # Determine ground truth and type
        gt = ""
        gt_type = "exact"
        
        if hasattr(sample, 'ground_truth'):
            gt = sample.ground_truth
            if hasattr(sample, 'ground_truth_type') and sample.ground_truth_type:
                gt_type = sample.ground_truth_type
        elif isinstance(sample, dict):
            gt = sample.get('ground_truth', "")
            gt_type = sample.get('ground_truth_type', "exact")
            
        return Scorer.compute_all_scores(pred, gt, gt_type)
