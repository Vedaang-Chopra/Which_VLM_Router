"""
Error tracking module for the Artemis Router.

Tracks routing errors (misroutes, low confidence decisions) for:
1. Debugging and analysis
2. Targeted retraining
3. System monitoring
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class RoutingError:
    """A single routing error record."""
    request_id: str
    router_mode: str
    chosen_model: str
    best_model: Optional[str] = None
    confidence: Optional[float] = None
    rewards: Optional[Dict[str, float]] = None
    error_type: str = "unknown"
    severity: str = "medium"
    expected_latency_ms: Optional[float] = None
    actual_latency_ms: Optional[float] = None
    expected_cost_usd: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    expected_accuracy: Optional[float] = None
    actual_accuracy: Optional[float] = None
    task_type: Optional[str] = None
    source_dataset: Optional[str] = None
    prompt_snippet: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    sample_id: Optional[int] = None


class ErrorTracker:
    """
    Tracks routing errors to PostgreSQL for analysis and retraining.
    
    Usage:
        tracker = ErrorTracker(db_url)
        tracker.log_error(RoutingError(...))
        errors = tracker.get_errors_for_retraining(limit=1000)
    """
    
    def __init__(self, db_url: str):
        """
        Initialize error tracker.
        
        Args:
            db_url: PostgreSQL connection string
        """
        self.db_url = db_url
        self.engine: Engine = create_engine(db_url, pool_size=3, max_overflow=5)
        self._ensure_table()
    
    def _ensure_table(self):
        """Ensure the routing_errors table exists."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT to_regclass('public.routing_errors')"
                ))
                if result.scalar() is None:
                    logger.warning("routing_errors table not found. Run migration_errors.sql first.")
        except Exception as e:
            logger.error(f"Failed to check routing_errors table: {e}")
    
    def log_error(self, error: RoutingError) -> Optional[int]:
        """
        Log a routing error to the database.
        
        Args:
            error: RoutingError dataclass
            
        Returns:
            The database ID of the inserted error, or None on failure
        """
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    INSERT INTO routing_errors 
                    (request_id, router_mode, chosen_model, best_model, confidence,
                     rewards, error_type, severity, expected_latency_ms, actual_latency_ms,
                     expected_cost_usd, actual_cost_usd, expected_accuracy, actual_accuracy,
                     task_type, source_dataset, prompt_snippet, meta, sample_id)
                    VALUES 
                    (:request_id, :router_mode, :chosen_model, :best_model, :confidence,
                     :rewards, :error_type, :severity, :expected_latency_ms, :actual_latency_ms,
                     :expected_cost_usd, :actual_cost_usd, :expected_accuracy, :actual_accuracy,
                     :task_type, :source_dataset, :prompt_snippet, :meta, :sample_id)
                    RETURNING id
                """), {
                    "request_id": error.request_id,
                    "router_mode": error.router_mode,
                    "chosen_model": error.chosen_model,
                    "best_model": error.best_model,
                    "confidence": error.confidence,
                    "rewards": json.dumps(error.rewards) if error.rewards else None,
                    "error_type": error.error_type,
                    "severity": error.severity,
                    "expected_latency_ms": error.expected_latency_ms,
                    "actual_latency_ms": error.actual_latency_ms,
                    "expected_cost_usd": error.expected_cost_usd,
                    "actual_cost_usd": error.actual_cost_usd,
                    "expected_accuracy": error.expected_accuracy,
                    "actual_accuracy": error.actual_accuracy,
                    "task_type": error.task_type,
                    "source_dataset": error.source_dataset,
                    "prompt_snippet": error.prompt_snippet[:200] if error.prompt_snippet else None,
                    "meta": json.dumps(error.meta) if error.meta else None,
                    "sample_id": error.sample_id,
                })
                error_id = result.scalar()
                logger.debug(f"Logged routing error {error.request_id} -> id={error_id}")
                return error_id
        except Exception as e:
            logger.error(f"Failed to log routing error: {e}")
            return None
    
    def log_misroute(
        self,
        request_id: str,
        router_mode: str,
        chosen_model: str,
        best_model: str,
        confidence: float,
        rewards: Dict[str, float],
        task_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Optional[int]:
        """
        Convenience method to log a misroute error.
        
        A misroute is when chosen_model != best_model.
        """
        error = RoutingError(
            request_id=request_id,
            router_mode=router_mode,
            chosen_model=chosen_model,
            best_model=best_model,
            confidence=confidence,
            rewards=rewards,
            error_type="misroute",
            severity="medium",
            task_type=task_type,
            prompt_snippet=prompt,
        )
        return self.log_error(error)
    
    def log_low_confidence(
        self,
        request_id: str,
        router_mode: str,
        chosen_model: str,
        confidence: float,
        rewards: Dict[str, float],
        fallback_triggered: bool = False,
        task_type: Optional[str] = None,
    ) -> Optional[int]:
        """
        Convenience method to log a low-confidence routing decision.
        """
        error = RoutingError(
            request_id=request_id,
            router_mode=router_mode,
            chosen_model=chosen_model,
            confidence=confidence,
            rewards=rewards,
            error_type="low_confidence",
            severity="low" if fallback_triggered else "medium",
            task_type=task_type,
            meta={"fallback_triggered": fallback_triggered},
        )
        return self.log_error(error)
    
    def get_errors_for_retraining(
        self,
        limit: int = 1000,
        error_types: Optional[List[str]] = None,
        min_severity: str = "low",
    ) -> List[Dict[str, Any]]:
        """
        Get routing errors that haven't been used for retraining yet.
        
        Args:
            limit: Maximum number of errors to return
            error_types: Filter by error types (default: all)
            min_severity: Minimum severity to include
            
        Returns:
            List of error records as dicts
        """
        severity_order = ["low", "medium", "high", "critical"]
        min_sev_idx = severity_order.index(min_severity)
        allowed_severities = severity_order[min_sev_idx:]
        
        query = """
            SELECT 
                id, request_id, router_mode, chosen_model, best_model,
                confidence, rewards, error_type, task_type, source_dataset,
                prompt_snippet, created_at
            FROM routing_errors
            WHERE used_for_retraining = FALSE
              AND severity = ANY(:severities)
        """
        
        if error_types:
            query += " AND error_type = ANY(:error_types)"
        
        query += " ORDER BY created_at DESC LIMIT :limit"
        
        params = {
            "severities": allowed_severities,
            "limit": limit,
        }
        if error_types:
            params["error_types"] = error_types
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                return [dict(row._mapping) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get errors for retraining: {e}")
            return []
    
    def mark_as_retrained(self, error_ids: List[int]) -> int:
        """
        Mark errors as used for retraining.
        
        Args:
            error_ids: List of error IDs to mark
            
        Returns:
            Number of rows updated
        """
        if not error_ids:
            return 0
        
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    UPDATE routing_errors
                    SET used_for_retraining = TRUE,
                        retrained_at = NOW()
                    WHERE id = ANY(:ids)
                """), {"ids": error_ids})
                return result.rowcount
        except Exception as e:
            logger.error(f"Failed to mark errors as retrained: {e}")
            return 0
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics for recent errors.
        
        Args:
            hours: Look back this many hours
            
        Returns:
            Dict with summary statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_errors,
                COUNT(*) FILTER (WHERE error_type = 'misroute') as misroutes,
                COUNT(*) FILTER (WHERE error_type = 'low_confidence') as low_confidence,
                AVG(confidence) as avg_confidence,
                COUNT(DISTINCT chosen_model) as unique_models
            FROM routing_errors
            WHERE created_at > NOW() - INTERVAL '%s hours'
        """ % hours
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return {}
        except Exception as e:
            logger.error(f"Failed to get error summary: {e}")
            return {}


def create_error_tracker(db_url: str) -> ErrorTracker:
    """Factory function to create an error tracker."""
    return ErrorTracker(db_url)
