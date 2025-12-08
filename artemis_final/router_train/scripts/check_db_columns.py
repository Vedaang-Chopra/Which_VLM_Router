import sys
import os
from pathlib import Path
from sqlalchemy import text
import pandas as pd

# Add paths
current_dir = Path.cwd()
artemis_dir = current_dir.parent.parent # artemis_final
sys.path.append(str(artemis_dir))
sys.path.append(str(artemis_dir / "router_train"))

from router_train.config import DBConfig
from router_train.db_utils import get_engine

def check_columns():
    try:
        config = DBConfig.from_env()
        engine = get_engine(config)
        
        print("\n--- vlm_evaluations Columns ---")
        df_eval = pd.read_sql(text("SELECT * FROM vlm_evaluations LIMIT 0"), engine)
        for c in df_eval.columns:
            print(f" - {c}")
            
        print("\n--- vlm_responses Columns ---")
        df_resp = pd.read_sql(text("SELECT * FROM vlm_responses LIMIT 0"), engine)
        for c in df_resp.columns:
            print(f" - {c}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_columns()
