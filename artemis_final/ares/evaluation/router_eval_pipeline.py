"""
Router Evaluation Pipeline

Complete evaluation pipeline that:
- Loads samples and model responses from PostgreSQL
- Computes static ground-truth-based metrics (Scorer)
- Computes confidence scores (estimate_confidence)
- Runs VLM Judge (listwise ranking with image) - Llama Scout or similar
- Runs Glider text evaluator (optional)
- Writes all raw metrics back to SQL
- Supports CLI with filtering by split, tasks, models
"""

import pandas as pd
import logging
import json
import time
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import threading
from tqdm.auto import tqdm

# Ares imports
from ares.evaluation.evaluation import Scorer, GliderEvaluator, parse_glider_output
from ares.evaluation.judge_molmo import VLMJudge
from ares.evaluation.confidence import estimate_confidence
from ares.db.operations import insert_evaluations

# Inference Engine
from inference_engine.runners import OpenAIStyleRunner

logger = logging.getLogger("EVAL_PIPELINE")


class ProgressTracker:
    """Disk-based progress tracker for resumability."""
    
    def __init__(self, tracker_path: str = "eval_progress.json"):
        self.tracker_path = Path(tracker_path)
        self.data = self._load()
        self._lock = threading.Lock()
    
    def _load(self) -> Dict[str, Dict]:
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, 'r') as f:
                    return json.load(f)
            except:
                return {"completed_samples": {}, "stats": {}}
        return {"completed_samples": {}, "stats": {}}
    
    def _save(self):
        with open(self.tracker_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_completed_samples(self, source_config: str) -> Set[str]:
        return set(self.data["completed_samples"].get(source_config, []))
    
    def mark_completed(self, source_config: str, sample_ids: List[str]):
        with self._lock:
            if source_config not in self.data["completed_samples"]:
                self.data["completed_samples"][source_config] = []
            existing = set(self.data["completed_samples"][source_config])
            existing.update(sample_ids)
            self.data["completed_samples"][source_config] = list(existing)
            self._save()
    
    def update_stats(self, source_config: str, glider: int, molmo: int, errors: int):
        with self._lock:
            if source_config not in self.data["stats"]:
                self.data["stats"][source_config] = {"glider": 0, "vlm_judge": 0, "errors": 0}
            self.data["stats"][source_config]["glider"] += glider
            self.data["stats"][source_config]["vlm_judge"] += molmo
            self.data["stats"][source_config]["errors"] += errors
            self._save()
    
    def get_summary(self) -> Dict:
        total_samples = sum(len(v) for v in self.data["completed_samples"].values())
        total_glider = sum(s.get("glider", 0) for s in self.data["stats"].values())
        total_vlm_judge = sum(s.get("vlm_judge", 0) for s in self.data["stats"].values())
        return {"total_samples": total_samples, "total_glider": total_glider, "total_vlm_judge": total_vlm_judge}
    
    def reset(self, source_config: str = None):
        with self._lock:
            if source_config:
                self.data["completed_samples"].pop(source_config, None)
                self.data["stats"].pop(source_config, None)
            else:
                self.data = {"completed_samples": {}, "stats": {}}
            self._save()


class RouterEvalPipeline:
    """
    Complete evaluation pipeline supporting:
    - Static GT metrics via Scorer
    - Confidence estimation
    - VLM Judge (listwise with image) - Llama Scout or similar
    - Glider text evaluator
    - Parallel processing across 4 GPUs
    """
    
    def __init__(
        self,
        engine,
        runner: OpenAIStyleRunner,
        glider_model_names: List[str],
        vlm_judge_model_names: List[str],
        tracker_path: str = "eval_progress.json",
        use_glider: bool = True,
        use_vlm_judge: bool = True,
    ):
        self.engine = engine
        self.runner = runner
        self.glider_model_names = glider_model_names
        self.vlm_judge_model_names = vlm_judge_model_names
        self.use_glider = use_glider
        self.use_vlm_judge = use_vlm_judge
        
        self.tracker = ProgressTracker(tracker_path)
        self.scorer = Scorer()
        
        # Glider load balancing
        self._glider_cycle = itertools.cycle(glider_model_names) if glider_model_names else None
        self._glider_lock = threading.Lock()
        
        def load_balanced_glider_chat_fn(messages, max_tokens=1024):
            with self._glider_lock:
                model_name = next(self._glider_cycle)
            res = self.runner.chat(
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0
            )
            return res.get('response_text', '')

        self.glider_evaluator = GliderEvaluator(chat_fn=load_balanced_glider_chat_fn) if glider_model_names else None
        self.vlm_judge = VLMJudge(runner=self.runner, model_names=self.vlm_judge_model_names) if vlm_judge_model_names else None

    # =========================================================================
    # Data Loading Functions (from spec)
    # =========================================================================
    
    def load_joined_df(
        self,
        split: Optional[str] = None,
        router_tasks: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load vlm_samples + vlm_responses + vlm_images into a single DataFrame.
        Each row is one (sample_id, model_name) response with all sample-level fields.
        """
        conditions = ["r.ok = true"]
        
        if split:
            conditions.append(f"s.data_split = '{split}'")
        if router_tasks:
            tasks_str = ", ".join([f"'{t}'" for t in router_tasks])
            conditions.append(f"s.router_task IN ({tasks_str})")
        if models:
            models_str = ", ".join([f"'{m}'" for m in models])
            conditions.append(f"r.model_name IN ({models_str})")
        if sample_ids:
            ids_tuple = tuple(sample_ids)
            ids_str = str(ids_tuple) if len(ids_tuple) > 1 else f"('{ids_tuple[0]}')"
            conditions.append(f"r.sample_id IN {ids_str}")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT 
            r.sample_id, r.model_name, r.response_raw, r.response_parsed,
            r.confidence_score, r.confidence_source,
            r.score_exact_match, r.score_exact_match_normalized, r.score_f1,
            r.score_contains_gt, r.score_gt_in_response,
            r.score_numeric_match, r.score_mc_letter_match,
            r.is_correct, r.is_refusal,
            s.prompt_text, s.ground_truth, s.ground_truth_type, 
            s.source_config, s.router_task, s.data_split,
            s.image_id,
            i.image_bytes,
            e.glider_score, e.glider_reasoning,
            e.judge_molmo_score, e.judge_molmo_rank_group
        FROM vlm_responses r
        JOIN vlm_samples s ON r.sample_id = s.sample_id
        LEFT JOIN vlm_images i ON s.image_id = i.image_id
        LEFT JOIN vlm_evaluations e ON r.sample_id = e.sample_id AND r.model_name = e.model_name
        WHERE {where_clause}
        ORDER BY r.sample_id, r.model_name
        """
        return pd.read_sql(query, self.engine)

    def fetch_source_configs(self, split: Optional[str] = None) -> List[str]:
        query = "SELECT DISTINCT source_config FROM vlm_samples"
        if split:
            query += f" WHERE data_split = '{split}'"
        query += " ORDER BY source_config"
        with self.engine.connect() as conn:
            return [row[0] for row in conn.execute(text(query))]

    def fetch_sample_ids_for_config(self, source_config: str) -> List[str]:
        query = f"SELECT DISTINCT sample_id FROM vlm_samples WHERE source_config = '{source_config}'"
        with self.engine.connect() as conn:
            return [row[0] for row in conn.execute(text(query))]

    # =========================================================================
    # Static Metrics (from spec)
    # =========================================================================
    
    def run_static_metrics(self, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        """
        Recompute static GT-based metrics using Scorer if missing or forced.
        Updates: score_exact_match, score_f1, score_numeric_match, etc.
        """
        updates = []
        for idx, row in df.iterrows():
            # Check if we need to compute
            needs_compute = force or pd.isna(row.get('score_exact_match'))
            
            if needs_compute:
                gt_type = row.get('ground_truth_type', 'exact') or 'exact'
                scores = self.scorer.compute_all_scores(
                    pred=row.get('response_raw') or "",
                    gt=row.get('ground_truth') or "",
                    gt_type=gt_type
                )
                # Update DataFrame in place
                for key, value in scores.items():
                    if key in df.columns:
                        df.at[idx, key] = value
                updates.append(row['sample_id'])
        
        if updates:
            logger.info(f"Computed static metrics for {len(updates)} responses")
        return df

    # =========================================================================
    # Confidence Metrics (from spec)
    # =========================================================================
    
    def run_confidence(self, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        """
        Compute confidence scores using estimate_confidence if missing or forced.
        """
        updates = 0
        for idx, row in df.iterrows():
            needs_compute = force or pd.isna(row.get('confidence_score'))
            
            if needs_compute:
                # Build response dict for confidence estimation
                response_dict = {
                    'response_text': row.get('response_raw') or "",
                    'response_parsed': row.get('response_parsed') or "",
                    # logprobs would be here if available
                }
                
                try:
                    conf_score, conf_details = estimate_confidence(response_dict)
                    df.at[idx, 'confidence_score'] = conf_score
                    if 'confidence_source' in df.columns:
                        df.at[idx, 'confidence_source'] = conf_details.get('source', 'heuristic')
                    updates += 1
                except Exception as e:
                    logger.warning(f"Confidence estimation failed for {row['sample_id']}: {e}")
        
        if updates:
            logger.info(f"Computed confidence for {updates} responses")
        return df

    # =========================================================================
    # Judge Evaluators
    # =========================================================================
    
    def _bytes_to_data_url(self, image_bytes: bytes) -> Optional[str]:
        """Convert image bytes to PNG base64 data URL for vLLM compatibility."""
        if image_bytes is None:
            return None
        try:
            # Open image and convert to PNG for vLLM compatibility
            img = Image.open(BytesIO(image_bytes))
            buffer = BytesIO()
            img.convert('RGB').save(buffer, format='PNG')
            png_bytes = buffer.getvalue()
            b64 = base64.b64encode(png_bytes).decode('utf-8')
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            return None

    def run_glider(self, df: pd.DataFrame, force: bool = False) -> List[Dict]:
        """Run Glider text evaluator for each (sample, model) response."""
        if not self.glider_evaluator:
            return []
        
        to_process = [row for _, row in df.iterrows() 
                      if force or pd.isna(row.get('glider_score'))]
        if not to_process:
            return []
        
        results = []
        results_lock = threading.Lock()
        
        def process_one(row):
            try:
                data_block = self.glider_evaluator.build_data_block(
                    question=row['prompt_text'],
                    model_answer=row['response_raw'] or "",
                    ground_truth=row['ground_truth'],
                    sample_id=row['sample_id']
                )
                prompt = self.glider_evaluator.build_prompt(data_block)
                res = self.glider_evaluator.chat_fn([{"role": "user", "content": prompt}])
                reasoning, highlights, score = parse_glider_output(res)
                return {
                    "sample_id": row["sample_id"],
                    "model_name": row["model_name"],
                    "glider_score": score,
                    "glider_reasoning": reasoning,
                    "glider_highlight": json.dumps(highlights) if highlights else None,
                    "glider_raw_output": res
                }
            except Exception as e:
                logger.error(f"Glider error for {row['sample_id']}/{row['model_name']}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=min(64, max(1, len(to_process)))) as exc:
            futures = [exc.submit(process_one, r) for r in to_process]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    with results_lock:
                        results.append(res)
        return results

    def run_vlm_judge(self, df: pd.DataFrame, force: bool = False) -> List[Dict]:
        """
        Run VLM Judge VLM Judge for each sample.
        Sends image + all model answers for listwise ranking.
        """
        if not self.vlm_judge:
            return []
        
        grouped = df.groupby('sample_id')
        samples_to_run = []
        
        for sid, grp in grouped:
            # Check if we need to run for this sample
            needs_run = force or grp['judge_molmo_score'].isna().any()
            if needs_run and len(grp) >= 2:  # Need at least 2 answers for ranking
                samples_to_run.append((sid, grp))
        
        if not samples_to_run:
            return []
        
        results = []
        results_lock = threading.Lock()
        
        def process_sample(sid, group):
            try:
                first = group.iloc[0]
                
                # Get image from DB and convert to data URL
                image_bytes = first.get('image_bytes')
                if image_bytes is None:
                    logger.warning(f"No image_bytes for {sid} - image_id: {first.get('image_id')}")
                    image_url = None
                else:
                    image_url = self._bytes_to_data_url(image_bytes)
                    logger.debug(f"Image for {sid}: {len(image_bytes)} bytes")
                
                # Collect all model answers
                answers_dict = {
                    row['model_name']: (row['response_raw'] or "") 
                    for _, row in group.iterrows()
                }
                
                # Call VLM Judge judge
                eval_res = self.vlm_judge.evaluate_listwise(
                    image_url=image_url,
                    question=first['prompt_text'],
                    ground_truth=first['ground_truth'],
                    answers_dict=answers_dict,
                )
                
                if 'error' in eval_res:
                    logger.warning(f"VLM Judge error for {sid}: {eval_res['error']}")
                    return []
                
                # Format results for DB
                per_model = eval_res.get('per_model', {})
                raw_json = eval_res.get('raw_json', {})
                
                return [{
                    "sample_id": sid,
                    "model_name": mname,
                    "judge_molmo_score": scores.get('score'),
                    "judge_molmo_rank_group": scores.get('rank_group'),
                    "judge_molmo_raw": json.dumps(raw_json)
                } for mname, scores in per_model.items()]
                
            except Exception as e:
                logger.error(f"VLM Judge error for {sid}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=min(64, max(1, len(samples_to_run)))) as exc:
            futures = [exc.submit(process_sample, sid, grp) for sid, grp in samples_to_run]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    with results_lock:
                        results.extend(res)
        return results

    # =========================================================================
    # DB Write-back
    # =========================================================================
    
    def write_back_to_db(self, records: List[Dict]):
        """Write evaluation records back to vlm_evaluations table."""
        if not records:
            return
        try:
            insert_evaluations(records, self.engine)
        except Exception as e:
            logger.error(f"DB write failed: {e}")

    # =========================================================================
    # Main Pipeline Execution
    # =========================================================================
    
    def process_batch_parallel(self, df_batch: pd.DataFrame, force: bool) -> tuple:
        """Process a batch with PARALLEL Glider and VLM Judge evaluation."""
        glider_results = []
        vlm_judge_results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            if self.use_glider:
                futures.append(('glider', executor.submit(self.run_glider, df_batch, force)))
            if self.use_vlm_judge:
                futures.append(('molmo', executor.submit(self.run_vlm_judge, df_batch, force)))
            
            for name, future in futures:
                result = future.result()
                if name == 'glider':
                    glider_results = result
                else:
                    vlm_judge_results = result
        
        return glider_results, vlm_judge_results

    def process_source_config(self, source_config: str, batch_size: int, force: bool, pbar: tqdm):
        """Process all samples for one source_config."""
        all_sample_ids = self.fetch_sample_ids_for_config(source_config)
        
        if not force:
            completed = self.tracker.get_completed_samples(source_config)
            sample_ids = [sid for sid in all_sample_ids if sid not in completed]
            skipped = len(all_sample_ids) - len(sample_ids)
            if skipped > 0:
                pbar.update(skipped)
        else:
            sample_ids = all_sample_ids
        
        glider_count = 0
        vlm_judge_count = 0
        errors = 0
        
        for i in range(0, len(sample_ids), batch_size):
            batch_ids = sample_ids[i:i+batch_size]
            
            # Load data with images
            df_batch = self.load_joined_df(sample_ids=batch_ids)
            if df_batch.empty:
                continue
            
            try:
                # Run static metrics and confidence (fast, CPU-only)
                df_batch = self.run_static_metrics(df_batch, force=force)
                df_batch = self.run_confidence(df_batch, force=force)
                
                # Run judge evaluators in parallel (GPU)
                glider_res, vlm_judge_res = self.process_batch_parallel(df_batch, force)
                
                # Write results to DB
                self.write_back_to_db(glider_res)
                self.write_back_to_db(vlm_judge_res)
                
                glider_count += len(glider_res)
                vlm_judge_count += len(vlm_judge_res)
                self.tracker.mark_completed(source_config, batch_ids)
                
            except Exception as e:
                errors += 1
                logger.error(f"Batch error: {e}")
            
            pbar.update(len(batch_ids))
            pbar.set_postfix({'G': glider_count, 'M': vlm_judge_count, 'E': errors})
        
        self.tracker.update_stats(source_config, glider_count, vlm_judge_count, errors)

    def evaluate_all(
        self,
        batch_size: int = 50,
        force: bool = False,
        max_parallel_configs: int = 2,
        split: Optional[str] = None,
    ):
        """
        Main entry point for batch evaluation.
        
        Args:
            batch_size: Number of samples per batch
            force: If True, recompute all metrics even if present
            max_parallel_configs: Number of source_configs to process in parallel
            split: Optional filter for data_split (train/val/test)
        """
        configs = self.fetch_source_configs(split=split)
        summary = self.tracker.get_summary()
        
        logger.info(f"=" * 60)
        logger.info(f"EVALUATION PIPELINE")
        logger.info(f"=" * 60)
        logger.info(f"Source configs: {len(configs)}")
        logger.info(f"Previous progress: {summary['total_samples']} samples")
        logger.info(f"Use Glider: {self.use_glider}, Use VLM Judge: {self.use_vlm_judge}")
        logger.info(f"Force recompute: {force}")
        logger.info(f"Split filter: {split or 'ALL'}")
        logger.info(f"=" * 60)
        
        start = time.time()
        
        def process_config(source_config: str):
            all_ids = self.fetch_sample_ids_for_config(source_config)
            with tqdm(total=len(all_ids), desc=f"[{source_config[:25]}]", unit="s", leave=True) as pbar:
                self.process_source_config(source_config, batch_size, force, pbar)
            return source_config
        
        with ThreadPoolExecutor(max_workers=max_parallel_configs) as exc:
            futures = {exc.submit(process_config, cfg): cfg for cfg in configs}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Config failed: {e}")
        
        elapsed = time.time() - start
        final = self.tracker.get_summary()
        logger.info(f"=" * 60)
        logger.info(f"COMPLETE in {elapsed/60:.1f} min")
        logger.info(f"Total: {final['total_samples']} samples, {final['total_glider']} Glider, {final['total_vlm_judge']} VLM Judge")
        logger.info(f"=" * 60)
    
    def reset_progress(self, source_config: str = None):
        """Reset progress tracker."""
        self.tracker.reset(source_config)
        logger.info(f"Progress reset: {source_config or 'ALL'}")
