"""
DataCollector: Handles logging of samples, responses, and feedback to Postgres.
Uses the new schema defined in ares/db/migration_collected.sql.
"""
import logging
import json
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from common.config_loader import GlobalConfig

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Handles data collection for the VLM Router system.
    Logs samples, responses, and feedback to Postgres.
    """
    
    def __init__(self, cfg: GlobalConfig):
        """
        Initialize the DataCollector.
        
        Args:
            cfg: GlobalConfig from common.config_loader
        """
        self.cfg = cfg
        self.db_url = cfg.db.url
        self.engine: Engine = create_engine(self.db_url, pool_size=5, max_overflow=10)
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure required tables exist."""
        # The tables should be created by the migration in Docker.
        # For safety, we try to create them here too.
        try:
            with self.engine.begin() as conn:
                # Check if main table exists
                result = conn.execute(text(
                    "SELECT to_regclass('public.vlm_samples_collected')"
                ))
                if result.scalar() is None:
                    logger.info("Tables not found. Attempting to create...")
                    # Read and execute migration
                    from pathlib import Path
                    migration_path = Path(__file__).parent.parent / "ares" / "db" / "migration_collected.sql"
                    if migration_path.exists():
                        with open(migration_path) as f:
                            conn.execute(text(f.read()))
                        logger.info("Tables created successfully.")
                    else:
                        logger.warning(f"Migration file not found: {migration_path}")
        except Exception as e:
            logger.error(f"Failed to ensure tables: {e}")

    def log_sample_start(self,
                         request_id: str,
                         router_mode: str,
                         input_messages: List[Dict[str, Any]],
                         router_decision: Dict[str, Any],
                         lb_decision: Dict[str, Any],
                         meta: Optional[Dict[str, Any]] = None) -> int:
        """
        Log a new sample (request) to the database.
        
        Returns:
            The database ID of the inserted sample.
        """
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO vlm_samples_collected 
                (request_id, router_mode, input_messages, chosen_model, router_decision, lb_decision, meta)
                VALUES (:rid, :mode, :msgs, :model, :rd, :lbd, :meta)
                RETURNING id
            """), {
                "rid": request_id,
                "mode": router_mode,
                "msgs": json.dumps(input_messages),
                "model": lb_decision.get("final_model", "unknown"),
                "rd": json.dumps(router_decision),
                "lbd": json.dumps(lb_decision),
                "meta": json.dumps(meta) if meta else None
            })
            sample_id = result.scalar()
            logger.debug(f"Logged sample {request_id} -> id={sample_id}")
            return sample_id

    def log_model_response(self,
                           sample_id: int,
                           model_name: str,
                           raw_response: Dict[str, Any],
                           normalized_output: Optional[Dict[str, Any]] = None,
                           latency_ms: Optional[int] = None,
                           cost_cents: Optional[float] = None,
                           score: Optional[float] = None,
                           error: Optional[str] = None) -> int:
        """
        Log a model response to the database.
        
        Returns:
            The database ID of the inserted response.
        """
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO vlm_responses_collected
                (sample_id, model_name, raw_response, normalized_output, latency_ms, cost_cents, score, error)
                VALUES (:sid, :model, :raw, :norm, :lat, :cost, :score, :err)
                RETURNING id
            """), {
                "sid": sample_id,
                "model": model_name,
                "raw": json.dumps(raw_response),
                "norm": json.dumps(normalized_output) if normalized_output else None,
                "lat": latency_ms,
                "cost": cost_cents,
                "score": score,
                "err": error
            })
            response_id = result.scalar()
            logger.debug(f"Logged response for sample {sample_id} -> id={response_id}")
            return response_id

    def log_feedback(self, sample_id: int, feedback_params: Dict[str, Any]) -> int:
        """
        Log feedback for a sample.
        
        Returns:
            The database ID of the inserted feedback.
        """
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO vlm_feedback (sample_id, feedback_params)
                VALUES (:sid, :params)
                RETURNING id
            """), {
                "sid": sample_id,
                "params": json.dumps(feedback_params)
            })
            feedback_id = result.scalar()
            logger.debug(f"Logged feedback for sample {sample_id} -> id={feedback_id}")
            return feedback_id

    def get_sample_id_by_request_id(self, request_id: str) -> Optional[int]:
        """Look up sample ID by request_id."""
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id FROM vlm_samples_collected WHERE request_id = :rid"
            ), {"rid": request_id})
            row = result.fetchone()
            return row[0] if row else None

    def fetch_training_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch collected samples with feedback for retraining.
        
        Returns:
            List of dicts suitable for building training datasets.
        """
        query = """
            SELECT 
                s.id, s.request_id, s.router_mode, s.input_messages, s.chosen_model,
                s.router_decision, s.lb_decision,
                r.model_name, r.latency_ms, r.score as response_score,
                f.feedback_params
            FROM vlm_samples_collected s
            LEFT JOIN vlm_responses_collected r ON s.id = r.sample_id
            LEFT JOIN vlm_feedback f ON s.id = f.sample_id
            WHERE f.id IS NOT NULL  -- Only samples with feedback
            ORDER BY s.created_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return [dict(row._mapping) for row in result.fetchall()]
