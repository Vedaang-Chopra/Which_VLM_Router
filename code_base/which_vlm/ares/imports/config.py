
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# All available Cauldron configs
ALL_CAULDRON_CONFIGS = [
    'ai2d', 'aokvqa', 'chart2text', 'chartqa', 'clevr', 'clevr_math', 
    'cocoqa', 'datikz', 'diagram_image_to_text', 'docvqa', 'dvqa', 
    'figureqa', 'finqa', 'geomverse', 'hateful_memes', 'hitab', 'iam', 
    'iconqa', 'infographic_vqa', 'intergps', 'localized_narratives', 
    'mapqa', 'mimic_cgd', 'multihiertt', 'nlvr2', 'ocrvqa', 'okvqa', 
    'plotqa', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 
    'robut_wtq', 'scienceqa', 'screen2words', 'spot_the_diff', 'st_vqa', 
    'tabmwp', 'tallyqa', 'tat_qa', 'textcaps', 'textvqa', 'tqa', 
    'vistext', 'visual7w', 'visualmrc', 'vqarad', 'vqav2', 'vsr', 'websight'
]


# Task category mapping for routing
CONFIG_TO_TASK = {
    # OCR / Document Understanding
    'docvqa': 'document_ocr',
    'infographic_vqa': 'document_ocr',
    'ocrvqa': 'document_ocr',
    'st_vqa': 'scene_text_ocr',
    'textvqa': 'scene_text_ocr',
    'iam': 'handwriting_ocr',
    'rendered_text': 'rendered_text_ocr',
    
    # Chart / Graph Understanding
    'chartqa': 'chart_reasoning',
    'chart2text': 'chart_captioning',
    'plotqa': 'chart_reasoning',
    'figureqa': 'chart_reasoning',
    'dvqa': 'chart_reasoning',
    'vistext': 'chart_captioning',
    
    # Table Understanding
    'hitab': 'table_reasoning',
    'tabmwp': 'table_math',
    'finqa': 'table_math',
    'tat_qa': 'table_reasoning',
    'multihiertt': 'table_reasoning',
    'robut_sqa': 'table_reasoning',
    'robut_wikisql': 'table_reasoning',
    'robut_wtq': 'table_reasoning',
    
    # Diagram / Science
    'ai2d': 'diagram_reasoning',
    'diagram_image_to_text': 'diagram_captioning',
    'scienceqa': 'science_reasoning',
    'geomverse': 'geometry_reasoning',
    'intergps': 'geometry_reasoning',
    
    # General VQA
    'vqav2': 'general_vqa',
    'okvqa': 'knowledge_vqa',
    'aokvqa': 'knowledge_vqa',
    'cocoqa': 'general_vqa',
    'visual7w': 'general_vqa',
    'vqarad': 'medical_vqa',
    
    # Counting / Spatial
    'tallyqa': 'counting',
    'clevr': 'spatial_reasoning',
    'clevr_math': 'visual_math',
    'nlvr2': 'spatial_reasoning',
    'vsr': 'spatial_reasoning',
    
    # Captioning / Description
    'textcaps': 'image_captioning',
    'localized_narratives': 'dense_captioning',
    'screen2words': 'ui_captioning',
    'visualmrc': 'visual_mrc',
    
    # Specialized
    'iconqa': 'icon_reasoning',
    'raven': 'abstract_reasoning',
    'hateful_memes': 'meme_classification',
    'spot_the_diff': 'difference_detection',
    'mapqa': 'map_reasoning',
    'mimic_cgd': 'medical_report',
    'datikz': 'code_generation',
    'tqa': 'textbook_qa',
    'websight': 'web_understanding',
}

# Ground truth type by task
TASK_GT_TYPE = {
    'document_ocr': 'exact',
    'scene_text_ocr': 'exact',
    'handwriting_ocr': 'exact',
    'rendered_text_ocr': 'exact',
    'chart_reasoning': 'exact',
    'chart_captioning': 'freeform',
    'table_reasoning': 'exact',
    'table_math': 'numeric',
    'diagram_reasoning': 'mc',
    'diagram_captioning': 'freeform',
    'science_reasoning': 'mc',
    'geometry_reasoning': 'exact',
    'general_vqa': 'exact',
    'knowledge_vqa': 'exact',
    'medical_vqa': 'exact',
    'counting': 'numeric',
    'spatial_reasoning': 'exact',
    'visual_math': 'numeric',
    'image_captioning': 'freeform',
    'dense_captioning': 'freeform',
    'ui_captioning': 'freeform',
    'visual_mrc': 'exact',
    'icon_reasoning': 'mc',
    'abstract_reasoning': 'mc',
    'meme_classification': 'exact',
    'difference_detection': 'freeform',
    'map_reasoning': 'exact',
    'medical_report': 'freeform',
    'code_generation': 'freeform',
    'textbook_qa': 'exact',
    'web_understanding': 'freeform',
}






# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    run_id: str = field(default_factory=lambda: f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    save_images: bool = True
    output_dir: Path = Path("./experiment_data")
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['output_dir'] = str(d['output_dir'])
        return d


@dataclass 
class SampleRecord:
    """Complete record for one (sample, model) pair."""
    
    # === IDENTITY ===
    sample_id: str
    run_id: str
    timestamp_utc: str
    
    # === INPUT DATA ===
    image_path: Optional[str]
    image_bytes_hash: Optional[str]
    prompt_raw: str
    prompt_formatted: Optional[str]
    system_prompt: Optional[str]
    source_dataset: str
    source_config: str
    router_task: str
    ground_truth: str
    ground_truth_type: str
    mc_options: Optional[List[str]]
    source_index: int
    
    # === INPUT FEATURES ===
    img_width: Optional[int]
    img_height: Optional[int]
    img_aspect_ratio: Optional[float]
    img_file_size_bytes: Optional[int]
    txt_prompt_length_chars: int
    txt_prompt_length_words: int
    txt_question_type: Optional[str]
    txt_has_mc_options: bool
    
    # === MODEL INFO ===
    model_name: str
    model_id: str
    
    # === OUTPUT DATA ===
    response_raw: Optional[str]
    response_parsed: Optional[str]
    response_length_chars: int
    response_length_tokens: Optional[int]
    stop_reason: Optional[str]
    error_message: Optional[str]
    is_refusal: bool
    ok: bool
    
    # === QUALITY SCORES ===
    score_exact_match: float
    score_exact_match_normalized: float
    score_contains_gt: float
    score_gt_in_response: float
    score_f1: float
    score_numeric_match: Optional[float]
    score_mc_letter_match: Optional[float]
    is_correct: bool
    pred_answer_letter: Optional[str]
    gt_answer_letter: Optional[str]
    
    # === COST METRICS ===
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_ms: float
    estimated_cost_usd: float
    
    # === INFERENCE CONFIG ===
    inference_temperature: float
    inference_max_tokens: int
    inference_top_p: float
    
    # === SEMANTIC F1 RESULTS ===
    semantic_f1_precision: Optional[float] = None
    semantic_f1_recall: Optional[float] = None
    semantic_f1_f1: Optional[float] = None
    semantic_f1_gen_statements: Optional[List[str]] = None
    semantic_f1_gt_statements: Optional[List[str]] = None
    semantic_f1_matches: Optional[List[Tuple[str, str]]] = None
    semantic_f1_labels: Optional[List[str]] = None
    
    # === GLIDER EVALUATION RESULTS ===
    glider_score: Optional[float] = None
    glider_reasoning: Optional[str] = None
    glider_highlight: Optional[str] = None
    glider_raw_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
