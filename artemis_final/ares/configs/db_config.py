"""Database configuration."""

import os

# Connection modes: 'local' or 'ngrok'
DB_MODE = os.environ.get('DB_MODE', 'local')  # Change to 'local' for localhost

# Local connection (Docker on localhost)
LOCAL_DB_URL = "postgresql+psycopg2://vlmrouter:vlmrouter@localhost:5432/vlmrouter"

# Ngrok connection (remote PACE setup)
# Format: postgresql://user:password@host:port/database
NGROK_DB_URL = "postgresql+psycopg2://vlmrouter:vlmrouter@4.tcp.ngrok.io:16035/vlmrouter"

# Active connection URL
DB_URL = os.environ.get('DATABASE_URL') or (NGROK_DB_URL if DB_MODE == 'ngrok' else LOCAL_DB_URL)

# Individual components (for reference)
DB_USER = "vlmrouter"
DB_PASS = "vlmrouter"
DB_HOST = "0.tcp.us.ngrok.io" if DB_MODE == 'ngrok' else "localhost"
DB_PORT = "19423" if DB_MODE == 'ngrok' else "5432"
DB_NAME = "vlmrouter"


# Table names (normalized schema)
TABLES = {
    'samples': 'vlm_samples',
    'images': 'vlm_images', 
    'responses': 'vlm_responses',
    'evaluations': 'vlm_evaluations',
}

# Legacy - for backward compatibility
TABLE_NAME = "vlm_responses"  # Old monolithic table name (alias)
TABLE_NAME_SAMPLE = "vlm_responses"

# Model configuration
MODEL_PREFIXES = ['m1', 'm2', 'm3', 'm4', 'm5']
MODEL_NAMES = ['deepseek_ocr', 'qwen2_5_vl_3b', 'qwen2_5_vl_7b', 'qwen3_vl_8b_thinking', 'gemma_3_27b']
MODEL_PREFIX_TO_NAME = dict(zip(MODEL_PREFIXES, MODEL_NAMES))
MODEL_NAME_TO_PREFIX = dict(zip(MODEL_NAMES, MODEL_PREFIXES))