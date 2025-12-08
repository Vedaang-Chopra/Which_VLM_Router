"""
Data validation utilities for router training.

Handles missing data, validates data quality, and ensures training data is clean.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails critically."""
    pass


def validate_profiles_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    allow_missing_images: bool = True,
    allow_missing_confidence: bool = True,
    max_missing_cost_pct: float = 10.0,
    max_missing_latency_pct: float = 10.0,
    max_missing_glider_pct: float = 5.0,
) -> Dict[str, any]:
    """
    Validate profiles dataframe and report data quality issues.

    Args:
        df: Profiles dataframe from load_profiles_real_schema()
        required_columns: List of required columns (default: auto-detect)
        allow_missing_images: Allow missing image metadata
        allow_missing_confidence: Allow missing confidence scores
        max_missing_cost_pct: Max % of missing cost_usd values allowed
        max_missing_latency_pct: Max % of missing latency_ms values allowed
        max_missing_glider_pct: Max % of missing glider_score values allowed

    Returns:
        Dictionary with validation results and statistics

    Raises:
        DataValidationError: If critical validation fails
    """
    if required_columns is None:
        required_columns = [
            'sample_id',
            'prompt_text',
            'model_name',
            'source_dataset',
            'router_task',
        ]

    validation_report = {
        'total_rows': len(df),
        'validation_passed': True,
        'errors': [],
        'warnings': [],
        'statistics': {},
    }

    # Check required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        validation_report['validation_passed'] = False
        validation_report['errors'].append(f"Missing required columns: {missing_cols}")
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    total_rows = len(df)

    # Check for null values in required columns
    for col in required_columns:
        null_count = df[col].isnull().sum()
        null_pct = 100 * null_count / total_rows
        if null_count > 0:
            validation_report['warnings'].append(
                f"Column '{col}': {null_count}/{total_rows} ({null_pct:.2f}%) null values"
            )
            logger.warning(f"Column '{col}': {null_count} null values ({null_pct:.2f}%)")

    # Check critical metrics
    critical_checks = [
        ('estimated_cost_usd', max_missing_cost_pct, False),
        ('latency_ms', max_missing_latency_pct, False),
        ('glider_score', max_missing_glider_pct, False),
    ]

    for col, max_missing_pct, is_optional in critical_checks:
        if col not in df.columns:
            if not is_optional:
                validation_report['validation_passed'] = False
                validation_report['errors'].append(f"Missing critical column: {col}")
            continue

        null_count = df[col].isnull().sum()
        null_pct = 100 * null_count / total_rows

        validation_report['statistics'][f'{col}_missing_pct'] = null_pct

        if null_pct > max_missing_pct:
            validation_report['validation_passed'] = False
            validation_report['errors'].append(
                f"Column '{col}': {null_pct:.2f}% missing (max allowed: {max_missing_pct}%)"
            )
            logger.error(f"Column '{col}': {null_pct:.2f}% missing (threshold: {max_missing_pct}%)")
        elif null_pct > 0:
            validation_report['warnings'].append(
                f"Column '{col}': {null_pct:.2f}% missing"
            )
            logger.warning(f"Column '{col}': {null_pct:.2f}% missing")

    # Check image metadata
    if 'img_width' in df.columns and 'img_height' in df.columns:
        img_null = df['img_width'].isnull() | df['img_height'].isnull()
        img_null_count = img_null.sum()
        img_null_pct = 100 * img_null_count / total_rows

        validation_report['statistics']['img_metadata_missing_pct'] = img_null_pct

        if img_null_pct > 0:
            if not allow_missing_images and img_null_pct > 50:
                validation_report['validation_passed'] = False
                validation_report['errors'].append(
                    f"Image metadata missing for {img_null_pct:.2f}% of rows"
                )
            else:
                validation_report['warnings'].append(
                    f"Image metadata missing for {img_null_pct:.2f}% of rows"
                )
                logger.info(f"Image metadata missing: {img_null_count}/{total_rows} ({img_null_pct:.2f}%)")

    # Check confidence score
    if 'confidence_score' in df.columns:
        conf_null_count = df['confidence_score'].isnull().sum()
        conf_null_pct = 100 * conf_null_count / total_rows

        validation_report['statistics']['confidence_missing_pct'] = conf_null_pct

        if conf_null_pct > 0 and not allow_missing_confidence:
            validation_report['warnings'].append(
                f"Confidence scores missing for {conf_null_pct:.2f}% of rows"
            )
            logger.info(f"Confidence scores missing: {conf_null_count}/{total_rows} ({conf_null_pct:.2f}%)")

    # Check data ranges
    if 'glider_score' in df.columns:
        valid_glider = df['glider_score'].dropna()
        if len(valid_glider) > 0:
            min_glider = valid_glider.min()
            max_glider = valid_glider.max()
            if min_glider < 0 or max_glider > 5:
                validation_report['warnings'].append(
                    f"glider_score out of range [0, 5]: [{min_glider:.2f}, {max_glider:.2f}]"
                )
                logger.warning(f"glider_score range: [{min_glider:.2f}, {max_glider:.2f}] (expected [0, 5])")

    if 'estimated_cost_usd' in df.columns:
        zero_cost = (df['estimated_cost_usd'] == 0).sum()
        if zero_cost > 0:
            validation_report['warnings'].append(f"Found {zero_cost} rows with zero cost")
            logger.warning(f"Found {zero_cost} rows with zero cost")

    if 'latency_ms' in df.columns:
        zero_latency = (df['latency_ms'] == 0).sum()
        if zero_latency > 0:
            validation_report['warnings'].append(f"Found {zero_latency} rows with zero latency")
            logger.warning(f"Found {zero_latency} rows with zero latency")

    # Check for duplicate (sample_id, model_name) pairs
    duplicates = df.duplicated(subset=['sample_id', 'model_name']).sum()
    if duplicates > 0:
        validation_report['validation_passed'] = False
        validation_report['errors'].append(f"Found {duplicates} duplicate (sample_id, model_name) pairs")
        logger.error(f"Found {duplicates} duplicate (sample_id, model_name) pairs")

    # Data distribution stats
    validation_report['statistics']['unique_samples'] = df['sample_id'].nunique()
    validation_report['statistics']['unique_models'] = df['model_name'].nunique()
    if 'data_split' in df.columns:
        validation_report['statistics']['data_split_counts'] = df['data_split'].value_counts().to_dict()

    # Log summary
    if validation_report['validation_passed']:
        logger.info("✓ Data validation PASSED")
    else:
        logger.error("✗ Data validation FAILED")
        for error in validation_report['errors']:
            logger.error(f"  ERROR: {error}")

    if validation_report['warnings']:
        logger.warning(f"Data validation has {len(validation_report['warnings'])} warnings:")
        for warning in validation_report['warnings'][:5]:  # Show first 5
            logger.warning(f"  WARNING: {warning}")

    return validation_report


def clean_profiles_dataframe(
    df: pd.DataFrame,
    fill_missing_cost: bool = True,
    fill_missing_latency: bool = True,
    fill_missing_confidence: bool = True,
    drop_missing_glider: bool = True,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean profiles dataframe by handling missing values.

    Args:
        df: Profiles dataframe
        fill_missing_cost: Fill missing costs with median per model
        fill_missing_latency: Fill missing latency with median per model
        fill_missing_confidence: Fill missing confidence with 0.5
        drop_missing_glider: Drop rows with missing glider_score
        drop_duplicates: Drop duplicate (sample_id, model_name) pairs

    Returns:
        Tuple of (cleaned_df, cleaning_stats)
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    cleaning_stats = {
        'initial_rows': initial_rows,
        'dropped_rows': 0,
        'filled_cost': 0,
        'filled_latency': 0,
        'filled_confidence': 0,
    }

    # Drop duplicates
    if drop_duplicates and 'sample_id' in df_clean.columns and 'model_name' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['sample_id', 'model_name'], keep='first')
        dropped = before - len(df_clean)
        if dropped > 0:
            cleaning_stats['dropped_duplicates'] = dropped
            logger.info(f"Dropped {dropped} duplicate (sample_id, model_name) pairs")

    # Drop rows with missing glider_score (critical for training)
    if drop_missing_glider and 'glider_score' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=['glider_score'])
        dropped = before - len(df_clean)
        if dropped > 0:
            cleaning_stats['dropped_missing_glider'] = dropped
            logger.info(f"Dropped {dropped} rows with missing glider_score")

    # Fill missing estimated_cost_usd with median per model
    if fill_missing_cost and 'estimated_cost_usd' in df_clean.columns:
        missing_cost = df_clean['estimated_cost_usd'].isnull().sum()
        if missing_cost > 0:
            # Fill with median per model, or global median if model has no data
            model_median_cost = df_clean.groupby('model_name')['estimated_cost_usd'].transform(
                lambda x: x.fillna(x.median())
            )
            global_median = df_clean['estimated_cost_usd'].median()
            df_clean['estimated_cost_usd'] = model_median_cost.fillna(global_median)
            cleaning_stats['filled_cost'] = missing_cost
            logger.info(f"Filled {missing_cost} missing estimated_cost_usd values with model medians")

    # Fill missing latency_ms with median per model
    if fill_missing_latency and 'latency_ms' in df_clean.columns:
        missing_latency = df_clean['latency_ms'].isnull().sum()
        if missing_latency > 0:
            model_median_latency = df_clean.groupby('model_name')['latency_ms'].transform(
                lambda x: x.fillna(x.median())
            )
            global_median = df_clean['latency_ms'].median()
            df_clean['latency_ms'] = model_median_latency.fillna(global_median)
            cleaning_stats['filled_latency'] = missing_latency
            logger.info(f"Filled {missing_latency} missing latency_ms values with model medians")

    # Fill missing confidence_score with neutral value
    if fill_missing_confidence and 'confidence_score' in df_clean.columns:
        missing_conf = df_clean['confidence_score'].isnull().sum()
        if missing_conf > 0:
            df_clean['confidence_score'] = df_clean['confidence_score'].fillna(0.5)
            cleaning_stats['filled_confidence'] = missing_conf
            logger.info(f"Filled {missing_confidence} missing confidence_score values with 0.5")

    # Fill missing image dimensions with zeros (will be ignored in text-only samples)
    if 'img_width' in df_clean.columns:
        df_clean['img_width'] = df_clean['img_width'].fillna(0)
    if 'img_height' in df_clean.columns:
        df_clean['img_height'] = df_clean['img_height'].fillna(0)
    if 'img_aspect_ratio' in df_clean.columns:
        df_clean['img_aspect_ratio'] = df_clean['img_aspect_ratio'].fillna(1.0)

    cleaning_stats['final_rows'] = len(df_clean)
    cleaning_stats['dropped_rows'] = initial_rows - len(df_clean)

    logger.info(f"✓ Data cleaning complete: {initial_rows} → {len(df_clean)} rows ({cleaning_stats['dropped_rows']} dropped)")

    return df_clean, cleaning_stats


def validate_train_val_test_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, any]:
    """
    Validate train/val/test split has no leakage and balanced distribution.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe

    Returns:
        Dictionary with split statistics

    Raises:
        DataValidationError: If data leakage detected
    """
    split_stats = {
        'train_samples': train_df['sample_id'].nunique(),
        'val_samples': val_df['sample_id'].nunique(),
        'test_samples': test_df['sample_id'].nunique(),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
    }

    # Check for sample_id leakage
    train_samples = set(train_df['sample_id'].unique())
    val_samples = set(val_df['sample_id'].unique())
    test_samples = set(test_df['sample_id'].unique())

    train_val_overlap = train_samples & val_samples
    train_test_overlap = train_samples & test_samples
    val_test_overlap = val_samples & test_samples

    if train_val_overlap:
        raise DataValidationError(f"Data leakage: {len(train_val_overlap)} samples in both train and val")
    if train_test_overlap:
        raise DataValidationError(f"Data leakage: {len(train_test_overlap)} samples in both train and test")
    if val_test_overlap:
        raise DataValidationError(f"Data leakage: {len(val_test_overlap)} samples in both val and test")

    logger.info("✓ No data leakage detected between splits")

    # Check model coverage (all splits should have same models)
    train_models = set(train_df['model_name'].unique())
    val_models = set(val_df['model_name'].unique())
    test_models = set(test_df['model_name'].unique())

    all_models = train_models | val_models | test_models

    if train_models != all_models:
        missing = all_models - train_models
        logger.warning(f"Training set missing models: {missing}")
        split_stats['train_missing_models'] = list(missing)

    if val_models != all_models:
        missing = all_models - val_models
        logger.warning(f"Validation set missing models: {missing}")
        split_stats['val_missing_models'] = list(missing)

    if test_models != all_models:
        missing = all_models - test_models
        logger.warning(f"Test set missing models: {missing}")
        split_stats['test_missing_models'] = list(missing)

    # Split proportions
    total_samples = split_stats['train_samples'] + split_stats['val_samples'] + split_stats['test_samples']
    split_stats['train_pct'] = 100 * split_stats['train_samples'] / total_samples
    split_stats['val_pct'] = 100 * split_stats['val_samples'] / total_samples
    split_stats['test_pct'] = 100 * split_stats['test_samples'] / total_samples

    logger.info(f"Split proportions: Train={split_stats['train_pct']:.1f}%, Val={split_stats['val_pct']:.1f}%, Test={split_stats['test_pct']:.1f}%")

    return split_stats
