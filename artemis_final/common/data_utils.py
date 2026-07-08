import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_data_dir() -> Path:
    """Returns the directory containing sample data."""
    # Robustly resolve the project root relative to this file
    return Path(__file__).resolve().parents[3] / "examples" / "data"

def load_model_profiles(parquet_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads aggregated model profiles from cached parquet data.
    """
    if parquet_path is None:
        parquet_path = get_data_dir() / "router_profiles_with_utility.parquet"
        
    path = Path(parquet_path)
    if not path.exists():
        logger.warning(f"Profile data not found at {path}. Returning empty DF.")
        return pd.DataFrame()
        
    df = pd.read_parquet(path)
    
    if 'model_name' not in df.columns:
        return pd.DataFrame()
        
    # Normalize accuracy column
    acc_col = 'is_correct'
    if 'is_correct' not in df.columns:
        if 'utility_accuracy' in df.columns:
            acc_col = 'utility_accuracy'
        elif 'score_exact_match_normalized' in df.columns:
            acc_col = 'score_exact_match_normalized'
            
    agg_dict = {
        'latency_ms': 'mean',
        'estimated_cost_usd': 'mean',
    }
    if acc_col in df.columns:
        agg_dict[acc_col] = 'mean'
        
    profiles = df.groupby('model_name').agg(agg_dict).reset_index()
    
    rename_map = {
        'latency_ms': 'avg_latency_ms',
        'estimated_cost_usd': 'avg_cost_usd',
    }
    if acc_col in profiles.columns:
        rename_map[acc_col] = 'avg_accuracy'
        
    profiles.rename(columns=rename_map, inplace=True)
    return profiles

def compute_oracle_best_model(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the oracle best model for each sample in eval_df.
    Selects model with highest accuracy, using cost as a tie-breaker.
    """
    if eval_df.empty:
        return eval_df
        
    acc_col = 'is_correct'
    if 'is_correct' not in eval_df.columns:
        if 'utility_accuracy' in eval_df.columns:
            acc_col = 'utility_accuracy'
        elif 'score_exact_match_normalized' in eval_df.columns:
            acc_col = 'score_exact_match_normalized'
    
    if acc_col not in eval_df.columns:
        logger.warning("No accuracy column found to compute oracle.")
        return eval_df

    # Validate required columns exist
    if 'estimated_cost_usd' not in eval_df.columns:
        logger.warning("Missing 'estimated_cost_usd' column for oracle computation.")
        return eval_df

    # Sort to prioritize accuracy (desc) then cost (asc)
    sorted_df = eval_df.sort_values(
        by=['sample_id', acc_col, 'estimated_cost_usd'], 
        ascending=[True, False, True]
    )
    
    best_models = sorted_df.drop_duplicates(subset=['sample_id'], keep='first')[['sample_id', 'model_name', acc_col]]
    best_models.rename(columns={'model_name': 'oracle_best_model', acc_col: 'oracle_score'}, inplace=True)
    
    return best_models

def load_eval_split(split: str = "test", limit: Optional[int] = None) -> pd.DataFrame:
    """Load evaluation data filtered by split."""
    parquet_path = get_data_dir() / "router_profiles_with_utility.parquet"
    if not parquet_path.exists():
        logger.warning(f"Data not found at {parquet_path}")
        return pd.DataFrame()
        
    df = pd.read_parquet(parquet_path)
    
    if "data_split" in df.columns:
        df = df[df["data_split"] == split]
        
    if limit:
        unique_samples = df['sample_id'].unique()[:limit]
        df = df[df['sample_id'].isin(unique_samples)]
        
    return df
