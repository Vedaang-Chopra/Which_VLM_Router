import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_data_dir() -> Path:
    # Assuming standard structure: root/examples/data/
    # This file is in root/artemis_final/common/
    # So root is ../..
    # This might differ if installed as package, but for repo structure:
    return Path(__file__).parent.parent.parent / "examples" / "data"

def load_model_profiles(parquet_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads aggregated model profiles from cached parquet data.
    """
    if parquet_path is None:
        parquet_path = get_data_dir() / "router_profiles_with_utility.parquet"
        
    path = Path(parquet_path)
    if not path.exists():
        # Fallback or error
        logger.warning(f"Profile data not found at {path}. Returning empty DF.")
        return pd.DataFrame()
        
    df = pd.read_parquet(path)
    
    # Check for required columns for aggregation
    # We want one row per model with avg_latency, avg_cost, avg_accuracy
    if 'model_name' not in df.columns:
        return pd.DataFrame()
        
    # Handle accuracy column variations
    acc_col = 'is_correct'
    if 'is_correct' not in df.columns:
        if 'utility_accuracy' in df.columns:
            acc_col = 'utility_accuracy'
        elif 'score_exact_match_normalized' in df.columns:
            acc_col = 'score_exact_match_normalized'
            
    # Aggregate
    agg_dict = {
        'latency_ms': 'mean',
        'estimated_cost_usd': 'mean',
    }
    if acc_col in df.columns:
        agg_dict[acc_col] = 'mean'
        
    profiles = df.groupby('model_name').agg(agg_dict).reset_index()
    
    # Rename
    rename_map = {
        'latency_ms': 'avg_latency_ms',
        'estimated_cost_usd': 'avg_cost_usd',
    }
    if acc_col in profiles.columns:
        rename_map[acc_col] = 'avg_accuracy'
        
    profiles.rename(columns=rename_map, inplace=True)
    return profiles

def compute_oracle_best_model(eval_df: pd.DataFrame, profiles_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Computes the oracle best model for each sample in eval_df.
    
    Logic:
    1. Group by sample_id
    2. For each sample, find row with max accuracy (primary) and min cost (secondary tie-breaker)
    3. Add 'oracle_best_model' column to the returned DF.
    """
    if eval_df.empty:
        return eval_df
        
    # If already aggregated (one row per sample), we might need to go back to source 
    # But usually eval_df passed here is the raw "long" format with multiple models per sample.
    
    # Identify accuracy column
    acc_col = 'is_correct'
    if 'is_correct' not in eval_df.columns:
        if 'utility_accuracy' in eval_df.columns:
            acc_col = 'utility_accuracy'
        elif 'score_exact_match_normalized' in eval_df.columns:
            acc_col = 'score_exact_match_normalized'
    
    if acc_col not in eval_df.columns:
        logger.warning("No accuracy column found to compute oracle.")
        return eval_df

    # Sort by sample_id, accuracy (desc), cost (asc)
    # This puts the best model for each sample first
    sorted_df = eval_df.sort_values(
        by=['sample_id', acc_col, 'estimated_cost_usd'], 
        ascending=[True, False, True]
    )
    
    # Drop duplicates to keep top 1
    best_models = sorted_df.drop_duplicates(subset=['sample_id'], keep='first')[['sample_id', 'model_name', acc_col]]
    best_models.rename(columns={'model_name': 'oracle_best_model', acc_col: 'oracle_score'}, inplace=True)
    
    # Merge back if input was condensed, or just return the map.
    # But often we want to augment the original data. 
    # If the input `eval_df` has multiple rows per sample, this function might just return the bests.
    # Or strict 'wide' format? 
    # Let's assume we return the dataframe of best models (one row per sample) with metadata.
    
    return best_models

def load_eval_split(split: str = "test", limit: Optional[int] = None) -> pd.DataFrame:
    """
    Loads evaluation data, optionally filtered by split and limit.
    """
    parquet_path = get_data_dir() / "router_profiles_with_utility.parquet"
    if not parquet_path.exists():
        logger.warning(f"Data not found at {parquet_path}")
        return pd.DataFrame()
        
    df = pd.read_parquet(parquet_path)
    
    if "data_split" in df.columns:
        df = df[df["data_split"] == split]
        
    if limit:
        # Get n unique samples
        unique_samples = df['sample_id'].unique()[:limit]
        df = df[df['sample_id'].isin(unique_samples)]
        
    return df
