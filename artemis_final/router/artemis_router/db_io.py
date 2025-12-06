"""
Database I/O operations for the Artemis Router.

Handles:
- Reading test/validation samples from SQL
- Writing router decision logs to SQL
"""

import io
import os
import time
import json
from typing import List, Optional
from pathlib import Path

import pandas as pd
from PIL import Image
from sqlalchemy.engine import Engine
from sqlalchemy import text

from .config import DataConfig
from .schemas import Sample, LogRecord


def _row_to_sample(row: pd.Series, cfg: DataConfig) -> Sample:
    """
    Convert a DataFrame row to a Sample object.

    Args:
        row: DataFrame row from samples table
        cfg: Data configuration

    Returns:
        Sample object
    """
    # Load image
    img = None
    image_uri = None

    # Try loading from image_png bytes (if available)
    if "image_png" in row.index:
        img_bytes = row["image_png"]
        if isinstance(img_bytes, (bytes, bytearray, memoryview)):
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image_uri = f"bytes:{row.get(cfg.id_column, 'unknown')}"
            except Exception as e:
                print(f"[WARN] Failed to decode image_png for {row.get(cfg.id_column)}: {e}")

    # Try loading from image_path (if available and image_png failed)
    if img is None and cfg.image_path_column in row.index:
        image_path = row[cfg.image_path_column]
        if image_path and not pd.isna(image_path):
            full_path = os.path.join(cfg.image_root_dir, image_path)
            image_uri = full_path
            try:
                img = Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to load image from {full_path}: {e}")

    # Build metadata dict
    metadata = {
        "split": row.get(cfg.split_column),
    }

    # Include image metadata if available
    if "img_width" in row.index:
        metadata["img_width"] = row["img_width"]
    if "img_height" in row.index:
        metadata["img_height"] = row["img_height"]
    if "img_aspect_ratio" in row.index:
        metadata["img_aspect_ratio"] = row["img_aspect_ratio"]

    # Include other useful metadata
    for col in ["router_task", "source_dataset", "txt_question_type"]:
        if col in row.index:
            metadata[col] = row[col]

    return Sample(
        sample_id=str(row[cfg.id_column]),
        source="db",
        text=str(row.get(cfg.text_column, "")),
        image=img,
        image_uri=image_uri,
        metadata=metadata,
        label=row.get(cfg.label_column) if cfg.label_column in row.index else None,
    )


def load_samples_from_db(
    engine: Engine,
    cfg: DataConfig,
    split: str,
    limit: Optional[int] = None
) -> List[Sample]:
    """
    Load samples from database for a given split.

    Args:
        engine: SQLAlchemy engine
        cfg: Data configuration
        split: Dataset split (e.g., "test", "val")
        limit: Maximum number of samples to load

    Returns:
        List of Sample objects
    """
    query = f"SELECT * FROM {cfg.samples_table} WHERE {cfg.split_column} = :split"
    params = {"split": split}

    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    df = pd.read_sql(query, engine, params=params)
    samples = [_row_to_sample(row, cfg) for _, row in df.iterrows()]
    return samples


def load_sample_from_db(
    engine: Engine,
    cfg: DataConfig,
    sample_id: str,
    split: Optional[str] = None
) -> Sample:
    """
    Load a single sample from database by ID.

    Args:
        engine: SQLAlchemy engine
        cfg: Data configuration
        sample_id: Sample identifier
        split: Optional split filter

    Returns:
        Sample object

    Raises:
        ValueError: If sample not found
    """
    if split:
        query = f"""
            SELECT * FROM {cfg.samples_table}
            WHERE {cfg.id_column} = :id AND {cfg.split_column} = :split
            LIMIT 1
        """
        params = {"id": sample_id, "split": split}
    else:
        query = f"""
            SELECT * FROM {cfg.samples_table}
            WHERE {cfg.id_column} = :id
            LIMIT 1
        """
        params = {"id": sample_id}

    df = pd.read_sql(query, engine, params=params)

    if df.empty:
        raise ValueError(f"No sample found with id={sample_id}, split={split}")

    return _row_to_sample(df.iloc[0], cfg)


def log_to_sql(engine: Engine, cfg: DataConfig, record: LogRecord):
    """
    Insert a LogRecord into the router logs table.

    Args:
        engine: SQLAlchemy engine
        cfg: Data configuration
        record: Log record to insert
    """
    sql = text(f"""
        INSERT INTO {cfg.logs_table}
        (timestamp, sample_id, source, split, text, image_uri, label,
         router_chosen_model, router_probs, router_inference_ms, extra_metadata)
        VALUES (:timestamp, :sample_id, :source, :split, :text, :image_uri, :label,
                :router_chosen_model, :router_probs, :router_inference_ms, :extra_metadata)
    """)

    params = {
        "timestamp": record.timestamp,
        "sample_id": record.sample_id,
        "source": record.source,
        "split": record.split,
        "text": record.text,
        "image_uri": record.image_uri,
        "label": record.label,
        "router_chosen_model": record.router_chosen_model,
        "router_probs": json.dumps(record.router_probs),
        "router_inference_ms": record.router_inference_ms,
        "extra_metadata": json.dumps(record.extra_metadata),
    }

    with engine.begin() as conn:
        conn.execute(sql, params)


def batch_log_to_sql(engine: Engine, cfg: DataConfig, records: List[LogRecord]):
    """
    Insert multiple LogRecords in a single transaction.

    Args:
        engine: SQLAlchemy engine
        cfg: Data configuration
        records: List of log records to insert
    """
    if not records:
        return

    sql = text(f"""
        INSERT INTO {cfg.logs_table}
        (timestamp, sample_id, source, split, text, image_uri, label,
         router_chosen_model, router_probs, router_inference_ms, extra_metadata)
        VALUES (:timestamp, :sample_id, :source, :split, :text, :image_uri, :label,
                :router_chosen_model, :router_probs, :router_inference_ms, :extra_metadata)
    """)

    params_list = [
        {
            "timestamp": r.timestamp,
            "sample_id": r.sample_id,
            "source": r.source,
            "split": r.split,
            "text": r.text,
            "image_uri": r.image_uri,
            "label": r.label,
            "router_chosen_model": r.router_chosen_model,
            "router_probs": json.dumps(r.router_probs),
            "router_inference_ms": r.router_inference_ms,
            "extra_metadata": json.dumps(r.extra_metadata),
        }
        for r in records
    ]

    with engine.begin() as conn:
        conn.execute(sql, params_list)
