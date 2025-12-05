"""Database configuration."""

DB_USER = "vlmrouter"
DB_PASS = "vlmrouter"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "vlmrouter"

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

TABLE_NAME = "vlm_router_evaluations"

# Model configuration
MODEL_PREFIXES = ['m1', 'm2', 'm3', 'm4', 'm5']
MODEL_NAMES = ['deepseek_ocr', 'qwen2_5_vl_3b', 'qwen2_5_vl_7b', 'qwen3_vl_8b_thinking', 'gemma_3_27b']
MODEL_PREFIX_TO_NAME = dict(zip(MODEL_PREFIXES, MODEL_NAMES))
MODEL_NAME_TO_PREFIX = dict(zip(MODEL_NAMES, MODEL_PREFIXES))