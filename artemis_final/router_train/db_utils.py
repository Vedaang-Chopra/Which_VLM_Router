"""
Database utilities for loading profiling data from PostgreSQL and local SQLite.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import DBConfig

logger = logging.getLogger(__name__)


def get_engine(db_config: DBConfig) -> Engine:
    """
    Create SQLAlchemy engine from database configuration.

    Args:
        db_config: Database configuration object

    Returns:
        SQLAlchemy engine instance
    """
    connection_string = db_config.get_connection_string()
    engine = create_engine(
        connection_string,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,
        max_overflow=10,
    )
    logger.info(f"Created database engine for {db_config.host}:{db_config.port}/{db_config.name}")
    return engine


def load_db_config_from_env() -> DBConfig:
    """
    Load database configuration from environment variables.

    Environment variables:
        - DB_USER: Database username
        - DB_PASS: Database password
        - DB_HOST: Database host
        - DB_PORT: Database port
        - DB_NAME: Database name

    Returns:
        DBConfig instance
    """
    return DBConfig.from_env()


def test_connection(db_config: DBConfig) -> bool:
    """
    Test database connection.

    Args:
        db_config: Database configuration

    Returns:
        True if connection successful, False otherwise
    """
    try:
        engine = get_engine(db_config)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_table_info(db_config: DBConfig) -> dict:
    """
    Get information about database tables.

    Args:
        db_config: Database configuration

    Returns:
        Dictionary with table names as keys and row counts as values
    """
    engine = get_engine(db_config)
    tables = ["vlm_sample", "vlm_responses", "vlm_evaluation", "vlm_images"]

    table_info = {}
    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.fetchone()[0]
                table_info[table] = count
                logger.info(f"Table {table}: {count} rows")
            except Exception as e:
                logger.warning(f"Could not get info for table {table}: {e}")
                table_info[table] = None

    return table_info


def load_profiles_real_schema(
    db_config: DBConfig,
    limit: Optional[int] = None,
    data_split: Optional[str] = None,
    limit_per_split: bool = False,
) -> pd.DataFrame:
    """
    Load profiling data using the REAL SQL schema.

    Joins vlm_samples, vlm_responses, vlm_evaluations, and vlm_images tables
    with proper column aliasing.

    Args:
        db_config: Database configuration
        limit: Optional limit on number of rows to load (for testing)
        data_split: Optional filter by data_split (e.g., "train", "val", "test")
        limit_per_split: If True and no data_split is provided, apply `limit`
            per data_split partition rather than overall.

    Returns:
        DataFrame with columns:
            - sample_id: Unique sample identifier
            - source_config: Source configuration
            - source_dataset: Source dataset name
            - router_task: Task category
            - data_split: Data split (train/val/test)
            - prompt_raw: Raw prompt text (aliased from prompt_text)
            - txt_prompt_length_chars: Prompt character count
            - txt_prompt_length_words: Prompt word count
            - img_width: Image width in pixels
            - img_height: Image height in pixels
            - img_aspect_ratio: Image aspect ratio
            - model_name: Model identifier
            - model_prefix: Model prefix (optional)
            - latency_ms: Response latency in milliseconds
            - cost_usd: Cost in USD (aliased from estimated_cost_usd)
            - confidence_score: Model confidence score
            - glider_score: GLIDER evaluation score (0-5)
    """
    engine = get_engine(db_config)

    where_clauses = []
    if data_split is not None:
        where_clauses.append(f"s.data_split = '{data_split}'")

    where_clause = ""
    if where_clauses:
        where_clause = " WHERE " + " AND ".join(where_clauses)

    extra_select = ""
    if limit_per_split and limit and data_split is None:
        extra_select = """
        , ROW_NUMBER() OVER (PARTITION BY s.data_split, s.router_task ORDER BY s.sample_id) AS split_row_num
        """

    base_query = f"""
    SELECT
        -- sample-level
        s.sample_id,
        s.source_config,
        s.source_dataset,
        s.router_task,
        s.data_split,
        s.prompt_text                 AS prompt_raw,
        s.txt_prompt_length_chars,
        s.txt_prompt_length_words,

        -- image metadata
        i.img_width,
        i.img_height,
        i.img_aspect_ratio,

        -- response-level (per model)
        r.model_name,
        r.model_prefix,
        r.latency_ms,
        r.estimated_cost_usd          AS cost_usd,
        r.confidence_score,

        -- evaluation-level (per model)
        ev.glider_score
        {extra_select}

    FROM vlm_samples s
    JOIN vlm_responses r
      ON s.sample_id = r.sample_id
    JOIN vlm_evaluations ev
      ON s.sample_id = ev.sample_id
     AND r.model_name = ev.model_name
    LEFT JOIN vlm_images i
      ON s.image_id = i.image_id
    {where_clause}
    """

    if limit_per_split and limit and data_split is None:
        query = f"""
    SELECT *
    FROM (
    {base_query}
    ) _split_limited
    WHERE split_row_num <= {limit}
    """
    else:
        query = base_query
        if limit is not None and limit > 0:
            query += f"\n    LIMIT {limit}"

    logger.info(f"Loading profiling data from database (real schema)...")
    if data_split:
        logger.info(f"  Filtering by data_split: {data_split}")
    if limit:
        logger.info(f"  Limit: {limit}")
    if limit_per_split and limit and data_split is None:
        logger.info("  Applying the limit per data_split/router_task partition")

    try:
        # Execute query
        df = pd.read_sql(text(query), engine)
        df = df.loc[:, df.columns != "split_row_num"]

        logger.info(f"Loaded {len(df)} profile records")
        logger.info(f"  Unique samples: {df['sample_id'].nunique()}")
        logger.info(f"  Unique models: {df['model_name'].nunique()}")
        logger.info(f"  Source datasets: {df['source_dataset'].unique().tolist()}")
        logger.info(f"  Router tasks: {df['router_task'].unique().tolist()}")
        if 'data_split' in df.columns:
            logger.info(f"  Data splits: {df['data_split'].value_counts().to_dict()}")

        # Data quality checks
        _validate_profiles_real_schema(df)

        return df

    except Exception as e:
        logger.error(f"Failed to load profiles from database: {e}")
        raise


def _validate_profiles_real_schema(df: pd.DataFrame) -> None:
    """
    Validate loaded profiles data using real schema.

    Args:
        df: Profiles dataframe
    """
    total_rows = len(df)

    # Check for missing critical fields
    critical_fields = ["sample_id", "prompt_raw", "model_name", "source_dataset", "router_task"]
    for field in critical_fields:
        if field not in df.columns:
            logger.warning(f"Missing column: {field}")
            continue
        null_count = df[field].isnull().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count}/{total_rows} ({100*null_count/total_rows:.1f}%) null values in {field}")

    # Check for glider_score (main accuracy signal)
    if "glider_score" in df.columns:
        null_count = df["glider_score"].isnull().sum()
        logger.info(f"  glider_score: {null_count}/{total_rows} ({100*null_count/total_rows:.1f}%) missing")

        # Check value range (should be 0-5)
        valid_data = df["glider_score"].dropna()
        if len(valid_data) > 0:
            min_val = valid_data.min()
            max_val = valid_data.max()
            logger.info(f"  glider_score range: [{min_val:.2f}, {max_val:.2f}]")
            if min_val < 0 or max_val > 5:
                logger.warning(f"Found glider_score values outside expected [0, 5] range")

    # Check for missing image data
    if "img_width" in df.columns and "img_height" in df.columns:
        img_null = df["img_width"].isnull() | df["img_height"].isnull()
        img_null_count = img_null.sum()
        if img_null_count > 0:
            logger.info(f"  Image metadata: {img_null_count}/{total_rows} ({100*img_null_count/total_rows:.1f}%) missing")

    # Check for zero/missing costs or latencies
    if "cost_usd" in df.columns:
        null_count = df["cost_usd"].isnull().sum()
        zero_cost = (df["cost_usd"] == 0).sum()
        logger.info(f"  cost_usd: {null_count} null, {zero_cost} zero")

    if "latency_ms" in df.columns:
        null_count = df["latency_ms"].isnull().sum()
        zero_latency = (df["latency_ms"] == 0).sum()
        logger.info(f"  latency_ms: {null_count} null, {zero_latency} zero")

    # Check confidence score
    if "confidence_score" in df.columns:
        null_count = df["confidence_score"].isnull().sum()
        logger.info(f"  confidence_score: {null_count}/{total_rows} ({100*null_count/total_rows:.1f}%) missing")


def get_sqlite_engine(db_path: Union[str, Path]) -> Engine:
    """
    Create SQLAlchemy engine for local SQLite database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLAlchemy engine instance
    """
    db_path = Path(db_path)
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)
    logger.info(f"Created SQLite engine for {db_path}")
    return engine


def save_to_sqlite(
    df: pd.DataFrame,
    db_path: Union[str, Path],
    table_name: str = "vlm_profiles",
    if_exists: str = "replace",
) -> None:
    """
    Save dataframe to SQLite database.

    Args:
        df: Dataframe to save
        db_path: Path to SQLite database file
        table_name: Name of table to create
        if_exists: What to do if table exists ('replace', 'append', 'fail')
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_sqlite_engine(db_path)

    logger.info(f"Saving {len(df)} rows to {db_path}:{table_name}")
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    logger.info(f"✓ Saved to {db_path}")


def load_from_sqlite(
    db_path: Union[str, Path],
    table_name: str = "vlm_profiles",
    data_split: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load dataframe from local SQLite database.

    Args:
        db_path: Path to SQLite database file
        table_name: Name of table to load
        data_split: Optional filter by data_split (e.g., "train", "val", "test")
        limit: Optional limit on number of rows to load

    Returns:
        DataFrame with profiling data
    """
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    engine = get_sqlite_engine(db_path)

    # Build query
    query = f"SELECT * FROM {table_name}"

    where_clauses = []
    if data_split is not None:
        where_clauses.append(f"data_split = '{data_split}'")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    if limit is not None and limit > 0:
        query += f" LIMIT {limit}"

    logger.info(f"Loading from {db_path}:{table_name}")
    if data_split:
        logger.info(f"  Filtering by data_split: {data_split}")
    if limit:
        logger.info(f"  Limit: {limit}")

    df = pd.read_sql(query, engine)

    logger.info(f"Loaded {len(df)} rows")
    if 'sample_id' in df.columns:
        logger.info(f"  Unique samples: {df['sample_id'].nunique()}")
    if 'model_name' in df.columns:
        logger.info(f"  Unique models: {df['model_name'].nunique()}")
    if 'data_split' in df.columns:
        logger.info(f"  Data splits: {df['data_split'].value_counts().to_dict()}")

    return df
