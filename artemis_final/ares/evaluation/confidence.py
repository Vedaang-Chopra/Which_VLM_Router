"""
Confidence estimation from VLM responses.

This module provides functions to estimate confidence scores from 
VLM response logprobs, with a heuristic fallback when logprobs are unavailable.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


SPECIAL_TOKENS = {
    "|im_start|",  # Note: angle brackets removed for compatibility
    "|im_end|",
    "|assistant|",
    "|user|",
}


def is_special_token(token: str) -> bool:
    """
    Return True if the token should be ignored for confidence computation.

    - Matches tokens containing special markers
    - Pure whitespace tokens are also ignored.
    """
    if not token:
        return True
    # Check if any special token pattern is contained
    for special in SPECIAL_TOKENS:
        if special in token:
            return True
    if token.strip() == "":
        return True
    return False


def extract_token_logprobs_from_choice(choice: Any) -> List[Dict[str, float]]:
    """
    Extract a list of {"token": str, "logprob": float} for the first choice's
    generated tokens, from either object-style or dict-style OpenAI responses.

    Returns an empty list if logprobs/content are missing.
    """
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None and isinstance(choice, dict):
        logprobs = choice.get("logprobs")

    if logprobs is None:
        return []

    # vLLM OpenAI-compatible: logprobs.content is a list of token entries
    content = getattr(logprobs, "content", None)
    if content is None and isinstance(logprobs, dict):
        content = logprobs.get("content")

    if not content:
        return []

    tokens: List[Dict[str, float]] = []
    for entry in content:
        if entry is None:
            continue

        # Handle both object-style and dict-style
        token = getattr(entry, "token", None)
        if token is None and isinstance(entry, dict):
            token = entry.get("token")

        logprob = getattr(entry, "logprob", None)
        if logprob is None and isinstance(entry, dict):
            logprob = entry.get("logprob")

        if token is None or logprob is None:
            continue

        try:
            tokens.append({"token": str(token), "logprob": float(logprob)})
        except (TypeError, ValueError):
            continue

    return tokens


def compute_confidence_scores(tokens: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Given a list of {"token": str, "logprob": float}, filter out special tokens
    and compute several confidence metrics:

    - mean_token_prob: mean of P(y_t) over non-special tokens
    - max_token_prob: max P(y_t) over non-special tokens
    - avg_log_prob: mean of log p(y_t) over non-special tokens
    - num_tokens: number of non-special tokens

    If there are no usable tokens, returns zeros.
    """
    filtered = [t for t in tokens if not is_special_token(t["token"])]

    if not filtered:
        return {
            "mean_token_prob": 0.0,
            "max_token_prob": 0.0,
            "avg_log_prob": 0.0,
            "num_tokens": 0,
        }

    probs: List[float] = []
    for t in filtered:
        lp = t["logprob"]
        try:
            probs.append(math.exp(lp))
        except OverflowError:
            continue

    if not probs:
        return {
            "mean_token_prob": 0.0,
            "max_token_prob": 0.0,
            "avg_log_prob": 0.0,
            "num_tokens": 0,
        }

    mean_token_prob = sum(probs) / len(probs)
    max_token_prob = max(probs)
    avg_log_prob = sum(t["logprob"] for t in filtered) / len(filtered)

    return {
        "mean_token_prob": mean_token_prob,
        "max_token_prob": max_token_prob,
        "avg_log_prob": avg_log_prob,
        "num_tokens": len(filtered),
    }


def estimate_confidence_from_logprobs(logprobs_data: Optional[Dict]) -> Tuple[float, Dict]:
    """
    Estimate confidence from logprobs data returned by the VLM.
    
    Parameters
    ----------
    logprobs_data : dict or None
        The logprobs dict from the response, with 'content' key containing tokens.
        
    Returns
    -------
    tuple
        (confidence_score: float, confidence_json: dict)
    """
    if not logprobs_data:
        return 0.0, {"source": "no_logprobs", "reason": "logprobs_not_available"}
    
    content = logprobs_data.get("content", [])
    if not content:
        return 0.0, {"source": "no_logprobs", "reason": "empty_content"}
    
    # Build tokens list
    tokens = []
    for entry in content:
        if entry is None:
            continue
        token = entry.get("token") if isinstance(entry, dict) else getattr(entry, "token", None)
        logprob = entry.get("logprob") if isinstance(entry, dict) else getattr(entry, "logprob", None)
        
        if token is not None and logprob is not None:
            try:
                tokens.append({"token": str(token), "logprob": float(logprob)})
            except (TypeError, ValueError):
                continue
    
    if not tokens:
        return 0.0, {"source": "no_logprobs", "reason": "no_valid_tokens"}
    
    scores = compute_confidence_scores(tokens)
    confidence = scores["mean_token_prob"]
    
    return confidence, {
        "source": "logprobs",
        **scores,
    }


def estimate_confidence_heuristic(response_text: str) -> Tuple[float, Dict]:
    """
    Heuristic-based confidence estimation when logprobs are not available.
    
    Parameters
    ----------
    response_text : str
        The text response from the model.
        
    Returns
    -------
    tuple
        (confidence_score: float, confidence_json: dict)
    """
    if not response_text:
        return 0.0, {"source": "heuristic", "reason": "empty_response"}
    
    uncertainty_phrases = [
        "i think",
        "maybe",
        "possibly",
        "might be",
        "could be",
        "not sure",
        "uncertain",
        "probably",
        "perhaps",
        "i believe",
    ]
    lower_text = response_text.lower()
    has_uncertainty = any(phrase in lower_text for phrase in uncertainty_phrases)

    confidence = 0.7
    if has_uncertainty:
        confidence -= 0.2
    if len(response_text.split()) < 10 and response_text:
        confidence += 0.1

    confidence = max(0.0, min(1.0, confidence))
    return confidence, {
        "source": "heuristic",
        "has_uncertainty": has_uncertainty,
        "token_count": len(response_text.split()),
    }


def estimate_confidence(
    response: Dict[str, Any],
    prefer_logprobs: bool = True
) -> Tuple[float, Dict]:
    """
    Main interface for confidence estimation.
    
    Tries logprobs first if available, falls back to heuristic otherwise.
    
    Parameters
    ----------
    response : dict
        The response dict from the VLM, containing 'logprobs' and 'response_text'.
    prefer_logprobs : bool
        If True, prefer logprobs-based estimation when available.
        
    Returns
    -------
    tuple
        (confidence_score: float, confidence_json: dict)
    """
    if prefer_logprobs:
        logprobs_data = response.get("logprobs")
        if logprobs_data:
            conf, conf_json = estimate_confidence_from_logprobs(logprobs_data)
            if conf > 0:
                return conf, conf_json
    
    # Fallback to heuristic
    response_text = response.get("response_text", "")
    return estimate_confidence_heuristic(response_text)
