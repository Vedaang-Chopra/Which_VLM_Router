"""
ARES (Automated Response Evaluation System)

Ares is a framework for evaluating VLM responses and performing data analysis.

Usage:
    from ares import get_db, insert_samples, AresAPI

See public_api.py for more details.
"""

from .public_api import (
    get_db,
    insert_samples,
    insert_responses,
    insert_evaluations,
    get_existing_responses,
    get_existing_sample_ids,
    ExperimentConfig,
    SampleRecord,
    AresAPI
)

__all__ = [
    "get_db",
    "insert_samples",
    "insert_responses",
    "insert_evaluations",
    "get_existing_responses",
    "get_existing_sample_ids",
    "ExperimentConfig",
    "SampleRecord",
    "AresAPI"
]
