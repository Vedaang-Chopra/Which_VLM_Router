# performance_utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Glider + basic helpers
# ---------------------------------------------------------------------

def normalize_glider(
    raw: pd.Series | np.ndarray,
    low: float = 1.0,
    high: float = 5.0,
) -> pd.Series:
    """Scale raw Glider scores into the 0–1 range.

    Parameters
    ----------
    raw : Series or array
        Raw Glider scores per sample.
    low, high : float
        Expected min/max bounds of the Glider metric.

    Returns
    -------
    pandas.Series
        Normalized scores suitable for downstream perf weighting.

    Usage
    -----
    Use this before combining Glider with correctness/F1 so all signals share
    a comparable range.
    """
    raw_s = pd.Series(raw, copy=False).astype(float)
    return (raw_s - low) / max(high - low, 1e-8)


# ---------------------------------------------------------------------
# 2. Sample-level score: sample_score(s, m)
# ---------------------------------------------------------------------

def compute_sample_score_from_columns(
    is_correct: pd.Series,
    acc_soft: pd.Series,
    glider_norm: pd.Series,
    w_g_correct: float = 0.2,
    w_near: float = 0.4,
    w_g_wrong: float = 0.1,
) -> pd.Series:
    """Compute a blended score for an individual (sample, model) pair.

    Inputs are the boolean correctness flag, a soft accuracy metric (F1, EM,
    etc.), and the normalized Glider score. Correct answers get a base score of
    1 plus a small Glider bump, while incorrect answers fall back to the soft
    accuracy + Glider mix. The coefficients let you emphasize near misses vs.
    confident wins.

    Returns a Series that can be appended to the dataframe as
    `<model_name>__sample_score` and later feed into routing analyses.
    """
    is_corr_float = pd.Series(is_correct, copy=False).astype(float)
    acc_soft = pd.Series(acc_soft, copy=False).fillna(0.0).astype(float)
    g_norm = pd.Series(glider_norm, copy=False).fillna(0.0).astype(float)

    correct_mask = is_corr_float >= 0.5

    score_correct = 1.0 + w_g_correct * g_norm
    score_wrong = w_near * acc_soft + w_g_wrong * g_norm

    sample_score = pd.Series(0.0, index=is_corr_float.index, dtype=float)
    sample_score.loc[correct_mask] = score_correct[correct_mask]
    sample_score.loc[~correct_mask] = score_wrong[~correct_mask]

    return sample_score


def add_sample_scores_for_models(
    df_proc: pd.DataFrame,
    model_names: Sequence[str],
    glider_low: float = 1.0,
    glider_high: float = 5.0,
    w_g_correct: float = 0.2,
    w_near: float = 0.4,
    w_g_wrong: float = 0.1,
    suffix_is_correct: str = "__is_correct",
    suffix_score_soft: str = "__score_f1",
    suffix_glider: str = "__glider_score",
    out_suffix: str = "__sample_score",
) -> pd.DataFrame:
    """Attach `<model_name>__sample_score` columns for every model.

    The helper looks up the correctness, soft-score, and Glider columns for
    each model, calls :func:`compute_sample_score_from_columns`, and returns a
    dataframe copy with the new score columns. Use this once after loading your
    dataset so the rest of the notebook can treat sample scores as first-class
    fields.
    """
    df_proc = df_proc.copy()

    for name in model_names:
        col_corr = f"{name}{suffix_is_correct}"
        col_soft = f"{name}{suffix_score_soft}"
        col_glid = f"{name}{suffix_glider}"

        is_corr = df_proc.get(col_corr, pd.Series(0.0, index=df_proc.index))
        acc_soft = df_proc.get(col_soft, pd.Series(0.0, index=df_proc.index))
        glider_raw = df_proc.get(col_glid, pd.Series(glider_low, index=df_proc.index))

        g_norm = normalize_glider(glider_raw, low=glider_low, high=glider_high)

        df_proc[f"{name}{out_suffix}"] = compute_sample_score_from_columns(
            is_correct=is_corr,
            acc_soft=acc_soft,
            glider_norm=g_norm,
            w_g_correct=w_g_correct,
            w_near=w_near,
            w_g_wrong=w_g_wrong,
        )

    return df_proc


# ---------------------------------------------------------------------
# 3. Priors: global_prior(m), task_prior(task, m)
# ---------------------------------------------------------------------

@dataclass
class PriorConfig:
    router_task_col: str = "router_task"
    model_col: str = "model_name"
    perf_col: str = "sample_score"     # or "is_correct", "score_f1"
    alpha: float = 50.0                # smoothing strength


def compute_global_prior(
    df_val: pd.DataFrame,
    cfg: PriorConfig,
) -> pd.DataFrame:
    """Aggregate mean performance per model for a validation dataframe.

    Parameters follow :class:`PriorConfig`. Feed a *long-format* dataframe with
    columns `[router_task, model_name, sample_score]` to obtain a small table
    mapping every model to its global average performance. Use this table when
    constructing hierarchical performance weights.
    """
    gp = (
        df_val
        .groupby(cfg.model_col)[cfg.perf_col]
        .mean()
        .rename("global_prior")
        .reset_index()
    )
    return gp


def compute_task_prior(
    df_val: pd.DataFrame,
    global_prior_df: pd.DataFrame,
    cfg: PriorConfig,
) -> pd.DataFrame:
    """Compute smoothed task-level priors per model.

    Takes the long-format dataframe plus the global prior from
    :func:`compute_global_prior` and returns
    `[router_task, model_name, task_prior, n_task]`, where `task_prior` blends
    task-specific means with the global fallback via `alpha`. Pass this into
    :func:`build_perf_matrix_hierarchical` to incorporate task context.
    """
    # Raw task-level means
    task_prior_raw = (
        df_val
        .groupby([cfg.router_task_col, cfg.model_col])[cfg.perf_col]
        .mean()
        .rename("task_prior_raw")
        .reset_index()
    )

    # Counts for smoothing
    counts = (
        df_val
        .groupby([cfg.router_task_col, cfg.model_col])[cfg.perf_col]
        .count()
        .rename("n_task")
        .reset_index()
    )

    tp = task_prior_raw.merge(
        counts, on=[cfg.router_task_col, cfg.model_col], how="left"
    )
    tp = tp.merge(global_prior_df, on=cfg.model_col, how="left")

    alpha = cfg.alpha
    tp["task_prior"] = (
        (tp["n_task"] * tp["task_prior_raw"] + alpha * tp["global_prior"])
        / (tp["n_task"] + alpha)
    )

    return tp[[cfg.router_task_col, cfg.model_col, "task_prior", "n_task"]]


def merge_priors_into_df(
    df_proc: pd.DataFrame,
    model_names: Sequence[str],
    prior_cfg: PriorConfig,
    global_prior_df: pd.DataFrame,
    task_prior_df: pd.DataFrame,
    model_col_in_df: str = "model_name",
) -> pd.DataFrame:
    """Add prior columns directly to a long-format dataframe.

    Use this when you want explicit `global_prior`/`task_prior` columns beside
    every (sample, model) record—e.g., for debugging or exporting analysis
    tables. Wide-format notebooks can skip this and stay with matrices.
    """
    df = df_proc.copy()

    # Ensure model_name column exists
    if model_col_in_df not in df.columns:
        raise ValueError(
            f"{model_col_in_df} must exist in df_proc (long format) to merge priors."
        )

    df = df.merge(global_prior_df, on=prior_cfg.model_col, how="left")
    df = df.merge(
        task_prior_df[[prior_cfg.router_task_col, prior_cfg.model_col, "task_prior"]],
        on=[prior_cfg.router_task_col, prior_cfg.model_col],
        how="left",
    )

    return df


# ---------------------------------------------------------------------
# 4. perf(s, m) = w_sample * sample_score + w_task * task_prior + w_global * global_prior
# ---------------------------------------------------------------------

@dataclass
class PerfWeightsHier:
    w_sample: float = 0.7
    w_task: float = 0.2
    w_global: float = 0.1


def build_perf_matrix_hierarchical(
    df_proc: pd.DataFrame,
    model_names: Sequence[str],
    router_task_col: str,
    sample_score_suffix: str = "__sample_score",
    task_prior_df: Optional[pd.DataFrame] = None,
    global_prior_df: Optional[pd.DataFrame] = None,
    prior_cfg: Optional[PriorConfig] = None,
    weights: PerfWeightsHier = PerfWeightsHier(),
) -> np.ndarray:
    """Assemble a dense matrix `[N, M]` of hierarchical performance scores.

    Inputs are the wide dataframe with `<model>__sample_score`, the task/global
    prior tables, and weighting coefficients. The output is ready for routing
    logic (`np.argmax` per row). If you omit priors, only the sample score term
    remains, making this function a superset of linear scoring.
    """
    N = len(df_proc)
    M = len(model_names)
    perf_mat = np.zeros((N, M), dtype=float)

    # Build dicts for quick lookup if priors provided
    model_to_global = {}
    task_model_to_prior = {}

    if prior_cfg is not None and global_prior_df is not None:
        model_to_global = dict(
            zip(global_prior_df[prior_cfg.model_col], global_prior_df["global_prior"])
        )

    if prior_cfg is not None and task_prior_df is not None:
        for _, row in task_prior_df.iterrows():
            key = (row[prior_cfg.router_task_col], row[prior_cfg.model_col])
            task_model_to_prior[key] = row["task_prior"]

    tasks = df_proc[router_task_col].values

    for j, name in enumerate(model_names):
        col_sample = f"{name}{sample_score_suffix}"
        sample_score = (
            df_proc.get(col_sample, pd.Series(0.0, index=df_proc.index))
            .fillna(0.0)
            .astype(float)
        )

        # Broadcast priors to [N]
        if prior_cfg is not None and global_prior_df is not None:
            g_prior_val = model_to_global.get(name, 0.0)
            g_prior = np.full(N, g_prior_val, dtype=float)
        else:
            g_prior = np.zeros(N, dtype=float)

        if prior_cfg is not None and task_prior_df is not None:
            t_prior = np.zeros(N, dtype=float)
            for i, task in enumerate(tasks):
                t_prior[i] = task_model_to_prior.get((task, name), 0.0)
        else:
            t_prior = np.zeros(N, dtype=float)

        perf = (
            weights.w_sample * sample_score.values
            + weights.w_task * t_prior
            + weights.w_global * g_prior
        )
        perf_mat[:, j] = perf

    return perf_mat


# ---------------------------------------------------------------------
# 5. Linear performance for sweeps: w_corr, w_f1, w_glid (your current configs)
# ---------------------------------------------------------------------

@dataclass
class PerfConfigLinear:
    name: str
    w_corr: float
    w_f1: float
    w_glid: float


def build_perf_mat_linear(
    df_proc: pd.DataFrame,
    model_names: Sequence[str],
    config: PerfConfigLinear,
    glider_low: float = 1.0,
    glider_high: float = 5.0,
    suffix_corr: str = "__is_correct",
    suffix_f1: str = "__score_f1",
    suffix_glider: str = "__glider_score",
    suffix_valid: str = "__valid_mask",
) -> np.ndarray:
    """Construct the classic linear performance matrix `[N, M]`.

    Pulls `<model>__is_correct`, `<model>__score_f1`, `<model>__glider_score`,
    and `<model>__valid_mask` columns to compute the weighted sum defined by
    :class:`PerfConfigLinear`. Invalid rows are zeroed out so they never win in
    argmax routing. Use this for baseline sweeps that ignore priors.
    """
    N = len(df_proc)
    M = len(model_names)
    perf_mat = np.zeros((N, M), dtype=float)

    for j, name in enumerate(model_names):
        col_corr = f"{name}{suffix_corr}"
        col_f1 = f"{name}{suffix_f1}"
        col_glider = f"{name}{suffix_glider}"
        col_valid = f"{name}{suffix_valid}"

        corr = (
            df_proc.get(col_corr, pd.Series(0.0, index=df_proc.index))
            .fillna(0.0)
            .astype(float)
        )
        f1 = (
            df_proc.get(col_f1, pd.Series(0.0, index=df_proc.index))
            .fillna(0.0)
            .astype(float)
        )
        glider_raw = (
            df_proc.get(col_glider, pd.Series(0.0, index=df_proc.index))
            .fillna(glider_low)
            .astype(float)
        )
        valid = (
            df_proc.get(col_valid, pd.Series(True, index=df_proc.index))
            .astype(bool)
        )

        g_norm = normalize_glider(glider_raw, low=glider_low, high=glider_high)

        w_corr = config.w_corr
        w_f1 = config.w_f1
        w_glid = config.w_glid

        perf = w_corr * corr + w_f1 * f1 + w_glid * g_norm
        perf = perf.where(valid, 0.0)

        perf_mat[:, j] = perf.values

    return perf_mat


def sweep_perf_configs_linear(
    df_proc: pd.DataFrame,
    model_names: Sequence[str],
    cost_mat: np.ndarray,
    valid_mat: np.ndarray,
    perf_configs: Sequence[PerfConfigLinear],
    lambda_cost: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate many linear configs against a fixed cost matrix.

    Returns two dataframes:
      * `perf_stats_df` → one row per config with the mean performance/cost.
      * `routing_df`    → stacked routing fractions for each (config, model).

    Use this helper in notebooks to replace manual loops when sweeping weights.
    """
    N = len(df_proc)
    idx_to_name = {i: n for i, n in enumerate(model_names)}

    perf_config_stats: List[Dict] = []
    routing_rows: List[Dict] = []

    HUGE_NEG = -1e12

    for cfg in perf_configs:
        cfg_name = cfg.name

        perf_mat_cfg = build_perf_mat_linear(df_proc, model_names, cfg)

        util = perf_mat_cfg - lambda_cost * cost_mat
        util = np.where(valid_mat, util, HUGE_NEG)

        best_idx = np.argmax(util, axis=1)

        chosen_perf = perf_mat_cfg[np.arange(N), best_idx]
        chosen_cost = cost_mat[np.arange(N), best_idx]

        perf_config_stats.append(
            {
                "config": cfg_name,
                "w_corr": cfg.w_corr,
                "w_f1": cfg.w_f1,
                "w_glid": cfg.w_glid,
                "mean_perf": float(chosen_perf.mean()),
                "mean_cost": float(chosen_cost.mean()),
            }
        )

        counts = (
            pd.Series(best_idx)
            .map(idx_to_name)
            .value_counts()
            .reindex(model_names, fill_value=0)
        )
        frac = counts / float(N)

        for model_name, v in frac.items():
            routing_rows.append(
                {
                    "perf_config": cfg_name,
                    "model": model_name,
                    "fraction": float(v),
                }
            )

    perf_stats_df = pd.DataFrame(perf_config_stats)
    routing_df = pd.DataFrame(routing_rows)

    return perf_stats_df, routing_df


def normalize_cost_matrix(cost_mat: np.ndarray) -> np.ndarray:
    """Scale cost values per matrix into [0, 1] via percentile min-max.

    Handy when you want cost penalties with robust bounds (ignoring outliers)
    before multiplying by `lambda_cost`.
    """
    c_min = np.percentile(cost_mat, 2)
    c_max = np.percentile(cost_mat, 98)
    cost_norm = (cost_mat - c_min) / (c_max - c_min + 1e-8)
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    return cost_norm


# ---------------------------------------------------------------------
# 6. Notebook helpers for modular preprocessing
# ---------------------------------------------------------------------

def compute_valid_mask_for_model(
    df: pd.DataFrame,
    prefix: str,
    response_suffix: str = "response_raw",
    ok_suffix: str = "ok",
    error_suffix: str = "error_message",
) -> pd.Series:
    """Return a boolean Series indicating whether a model produced a usable response.

    Inputs: dataframe, the model's prefix, and optional column suffix overrides.
    Outputs: `<model>__valid_mask`-ready Series you can assign back into the
    dataframe. Use this to ensure routing never selects models that errored out
    or failed to return text.
    """
    idx = df.index
    col_resp = f"{prefix}{response_suffix}"
    col_ok = f"{prefix}{ok_suffix}"
    col_err = f"{prefix}{error_suffix}"

    resp = df.get(col_resp, pd.Series(np.nan, index=idx))
    resp_str = resp.astype(str)
    has_text = resp.notna() & (resp_str.str.strip() != "")

    ok_series = df.get(col_ok, pd.Series(True, index=idx)).fillna(True).astype(bool)

    if col_err in df.columns:
        err = df[col_err].astype("object")
        no_error = err.isna() | (err.astype(str).str.strip() == "")
    else:
        no_error = pd.Series(True, index=idx)

    valid_mask = has_text & ok_series & no_error
    return valid_mask.astype(bool)


def add_valid_mask_columns(
    df: pd.DataFrame,
    model_specs: Sequence[Dict],
    response_suffix: str = "response_raw",
    ok_suffix: str = "ok",
    error_suffix: str = "error_message",
    out_suffix: str = "__valid_mask",
) -> pd.DataFrame:
    """Vectorized helper to append `<model>__valid_mask` columns for all specs.

    Feed the raw dataframe plus your MODEL_SPECS list; receive a copy with
    boolean mask columns ready for downstream perf/cost computations.
    """
    df_out = df.copy()
    for spec in model_specs:
        mask = compute_valid_mask_for_model(
            df_out,
            prefix=spec["prefix"],
            response_suffix=response_suffix,
            ok_suffix=ok_suffix,
            error_suffix=error_suffix,
        )
        df_out[f"{spec['name']}{out_suffix}"] = mask
    return df_out


def add_linear_perf_columns(
    df: pd.DataFrame,
    model_names: Sequence[str],
    config: PerfConfigLinear,
    suffix_perf: str = "__perf",
    suffix_corr: str = "__is_correct",
    suffix_f1: str = "__score_f1",
    suffix_glider: str = "__glider_score",
    suffix_valid: str = "__valid_mask",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Attach `<model>__perf` columns and get the underlying matrix in one call.

    Useful when notebooks need both the dataframe view (for plotting/groupby)
    and the numpy view (for routing argmax). Returns the mutated dataframe plus
    the `[N, M]` matrix used to populate it.
    """
    perf_mat = build_perf_mat_linear(
        df_proc=df,
        model_names=model_names,
        config=config,
        suffix_corr=suffix_corr,
        suffix_f1=suffix_f1,
        suffix_glider=suffix_glider,
        suffix_valid=suffix_valid,
    )

    df_out = df.copy()
    for idx, name in enumerate(model_names):
        df_out[f"{name}{suffix_perf}"] = perf_mat[:, idx]

    return df_out, perf_mat


def build_valid_matrix(
    df: pd.DataFrame,
    model_names: Sequence[str],
    suffix_valid: str = "__valid_mask",
) -> np.ndarray:
    """Stack the `<model>__valid_mask` columns into a `[N, M]` ndarray.

    Feed this into routing utilities to zero out invalid entries when computing
    utilities.
    """
    mats = [
        df.get(f"{name}{suffix_valid}", pd.Series(False, index=df.index)).astype(bool).values
        for name in model_names
    ]
    return np.stack(mats, axis=1)


def build_long_perf_dataframe(
    df: pd.DataFrame,
    model_names: Sequence[str],
    router_task_col: str,
    value_suffix: str = "__sample_score",
    model_col: str = "model_name",
    value_name: str = "sample_score",
) -> pd.DataFrame:
    """Convert wide sample-score columns into a long dataframe for prior tooling.

    Required columns: the router-task identifier and each `<model>__sample_score`.
    The output has `[router_task, model_name, sample_score]` and plugs directly
    into :func:`compute_global_prior` / :func:`compute_task_prior`.
    """
    required_cols = [router_task_col] + [f"{name}{value_suffix}" for name in model_names]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for long perf dataframe: {missing}")

    wide = df[required_cols].copy()
    melted = wide.melt(
        id_vars=router_task_col,
        var_name="_model_col",
        value_name=value_name,
    )
    melted[model_col] = melted["_model_col"].str.replace(value_suffix, "", regex=False)
    melted = melted.drop(columns="_model_col")
    return melted[[router_task_col, model_col, value_name]]
