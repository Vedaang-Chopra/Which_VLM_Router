
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import re
import pandas as pd


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Extract features from images and text for routing."""
    
    @staticmethod
    def extract_image_features(image: Image.Image, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract basic image features."""
        features = {
            'img_width': image.width,
            'img_height': image.height,
            'img_aspect_ratio': round(image.width / image.height, 3) if image.height > 0 else None,
            'img_file_size_bytes': None,
        }
        
        if image_path and Path(image_path).exists():
            features['img_file_size_bytes'] = Path(image_path).stat().st_size
            
        return features
    
    @staticmethod
    def extract_text_features(prompt: str) -> Dict[str, Any]:
        """Extract text/prompt features."""
        prompt_lower = prompt.lower()
        
        # Detect question type
        question_type = None
        if prompt_lower.startswith('what'):
            question_type = 'what'
        elif prompt_lower.startswith('how'):
            question_type = 'how'
        elif prompt_lower.startswith('why'):
            question_type = 'why'
        elif prompt_lower.startswith('where'):
            question_type = 'where'
        elif prompt_lower.startswith('when'):
            question_type = 'when'
        elif prompt_lower.startswith('who'):
            question_type = 'who'
        elif prompt_lower.startswith('which'):
            question_type = 'which'
        elif prompt_lower.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ')):
            question_type = 'yes_no'
        elif prompt_lower.startswith('how many'):
            question_type = 'counting'
            
        # Detect MC options
        has_mc = bool(re.search(r'\([A-D]\)', prompt) or re.search(r'[A-D][\.\)]', prompt))
        
        return {
            'txt_prompt_length_chars': len(prompt),
            'txt_prompt_length_words': len(prompt.split()),
            'txt_question_type': question_type,
            'txt_has_mc_options': has_mc,
        }
    
    @staticmethod
    def compute_image_hash(image: Image.Image) -> str:
        """Compute SHA256 hash of image bytes."""
        import io
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return hashlib.sha256(buf.getvalue()).hexdigest()[:16]



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_routing_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute routing labels: for each sample, which model is the "winner"?
    
    Winner = model with highest score at lowest cost (latency).
    """
    routing_records = []
    
    for sample_id in df['sample_id'].unique():
        sample_df = df[df['sample_id'] == sample_id]
        
        # Get best score
        best_score = sample_df['is_correct'].max()
        
        # Among correct models, pick fastest
        correct_models = sample_df[sample_df['is_correct'] == best_score]
        if len(correct_models) > 0:
            winner = correct_models.loc[correct_models['latency_ms'].idxmin()]
        else:
            # If none correct, pick highest F1
            winner = sample_df.loc[sample_df['score_f1'].idxmax()]
        
        routing_records.append({
            'sample_id': sample_id,
            'winner_model': winner['model_name'],
            'winner_score': winner['is_correct'],
            'winner_latency_ms': winner['latency_ms'],
            'n_models_correct': int(sample_df['is_correct'].sum()),
            'source_config': winner['source_config'],
            'router_task': winner['router_task'],
        })
    
    return pd.DataFrame(routing_records)


def analyze_model_strengths(df):
    """
    For each router_task, compute accuracy per model and identify
    the best model and its accuracy.
    """
    # Group: accuracy per (task, model)
    acc = (
        df.groupby(["router_task", "model_name"])["is_correct"]
        .mean()
        .reset_index(name="accuracy")
    )

    # Pivot: rows = task, cols = model_name, values = accuracy
    pivot = acc.pivot(
        index="router_task",
        columns="model_name",
        values="accuracy",
    )

    # Only look at numeric columns when computing argmax / max
    numeric_cols = pivot.select_dtypes(include="number").columns

    best_model = pivot[numeric_cols].idxmax(axis=1)
    best_accuracy = pivot[numeric_cols].max(axis=1)

    # Attach back to a copy so we don't mutate original unexpectedly
    result = pivot.copy()
    result["best_model"] = best_model
    result["best_accuracy"] = best_accuracy

    # Optional: make router_task a column again
    return result.reset_index()

