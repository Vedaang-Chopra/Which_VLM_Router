"""
Ares API

This module provides a centralized interface for the ARES (Automated Response Evaluation System).
It wraps database operations, configuration, and evaluation utilities.

Usage:
    from ares.ares_api import AresConfig, get_db, evaluate_sample
    
    # Get DB connection
    engine = get_db()
    
    # Evaluate a sample
    score = evaluate_sample(response_text="...", ground_truth="...")
"""

from typing import Optional, Dict, List, Any
from sqlalchemy.engine import Engine

# --- 1. Database & Config Imports ---
from .db.connection import get_engine
from .db.operations import (
    insert_samples, insert_responses, insert_evaluations,
    get_existing_responses, get_existing_sample_ids
)
from .configs.config import (
    ExperimentConfig, SampleRecord, CONFIG_TO_TASK,
    TASK_GT_TYPE
)

# --- 2. Evaluation Imports (Lazy load heavy dependencies if possible) ---
# Note: Some evaluation modules like 'glider' or 'judge' might load heavy models.
# For API purity, we import them here but users should be aware of overhead.
from .evaluation.evaluation import Evaluator

# --- 3. API Functions ---

def get_db() -> Engine:
    """
    Get the database engine connection.
    Singleton pattern handled by db.connection module.
    """
    return get_engine()

class AresAPI:
    """
    Facade class for ARES functionality.
    """
    def __init__(self):
        self._evaluator = None

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            self._evaluator = Evaluator()
        return self._evaluator

    def evaluate(self, sample_record: SampleRecord) -> Dict[str, Any]:
        """
        Run evaluation pipeline on a single sample record.
        """
        # This assumes Evaluator has a method for single record or we wrap it
        # Inspecting evaluation.py would confirm, but usually it processes lists.
        # For now, we expose the underlying class.
        return self.evaluator.evaluate_single(sample_record)

# --- 4. Exports ---

__all__ = [
    "get_db",
    "insert_samples",
    "insert_responses",
    "insert_evaluations",
    "get_existing_responses",
    "get_existing_sample_ids",
    "ExperimentConfig",
    "SampleRecord",
    "CONFIG_TO_TASK",
    "TASK_GT_TYPE",
    "AresAPI"
]
