"""Database CRUD operations."""

import json
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ares.db.connection import get_engine


def batch_insert_records(
    table_name: str,
    records: List[Dict[str, Any]],
    engine: Engine = None,
    batch_size: int = 100
) -> int:
    """
    Batch insert records with ON CONFLICT DO UPDATE.
    
    Args:
        table_name: Target table
        records: List of record dicts
        engine: SQLAlchemy engine (uses default if None)
        batch_size: Records per batch
        
    Returns:
        Number of records inserted
    """
    if not records:
        return 0
    
    if engine is None:
        engine = get_engine()
    
    columns = list(records[0].keys())
    col_str = ", ".join(columns)
    val_placeholders = ", ".join([f":{col}" for col in columns])
    
    # Build UPDATE clause for conflict
    update_cols = [c for c in columns if c != 'sample_id']
    update_str = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
    
    insert_sql = text(f"""
        INSERT INTO {table_name} ({col_str})
        VALUES ({val_placeholders})
        ON CONFLICT (sample_id) DO UPDATE SET {update_str}, updated_at = NOW()
    """)
    
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            for record in batch:
                # Process special types
                processed = {}
                for k, v in record.items():
                    if isinstance(v, (dict, list)):
                        processed[k] = json.dumps(v)
                    else:
                        processed[k] = v
                conn.execute(insert_sql, processed)
            total += len(batch)
    
    return total


def update_record(
    table_name: str,
    sample_id: str,
    updates: Dict[str, Any],
    engine: Engine = None
) -> bool:
    """Update specific columns for a sample."""
    if not updates:
        return False
    
    if engine is None:
        engine = get_engine()
    
    set_parts = [f"{k} = :{k}" for k in updates.keys()]
    set_clause = ", ".join(set_parts)
    
    sql = text(f"""
        UPDATE {table_name}
        SET {set_clause}, updated_at = NOW()
        WHERE sample_id = :sample_id
    """)
    
    params = {**updates, 'sample_id': sample_id}
    for k, v in params.items():
        if isinstance(v, (dict, list)):
            params[k] = json.dumps(v)
    
    with engine.begin() as conn:
        conn.execute(sql, params)
    
    return True


def batch_update_records(
    table_name: str,
    records: List[Dict[str, Any]],
    key_column: str = 'sample_id',
    engine: Engine = None
) -> int:
    """Batch update multiple records."""
    if engine is None:
        engine = get_engine()
    
    count = 0
    for record in records:
        sample_id = record.pop(key_column, None)
        if sample_id and record:
            update_record(table_name, sample_id, record, engine)
            count += 1
    
    return count


def get_row_count(table_name: str, where_clause: str = None, engine: Engine = None) -> int:
    """Get row count with optional filter."""
    if engine is None:
        engine = get_engine()
    
    sql = f"SELECT COUNT(*) FROM {table_name}"
    if where_clause:
        sql += f" WHERE {where_clause}"
    
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return result.scalar()