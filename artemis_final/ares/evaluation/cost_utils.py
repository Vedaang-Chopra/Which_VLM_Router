from typing import Dict, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------
# 1. True Dollar Cost per Row (your original function)
# -----------------------------------------------------------
def compute_true_cost(row, model_name, prefix, pricing):
    """
    Compute true USD cost = (in_tokens * prompt_cost) + (out_tokens * completion_cost).
    Returns None if this model did not run or tokens missing.

    Parameters
    ----------
    row : pandas.Series
        A single dataframe row with token columns.
    model_name : str
        Name of the model (key into `pricing`).
    prefix : str
        Prefix for the token columns, e.g. "gemma_" → "gemma_input_tokens".
    pricing : dict
        { model_name: { "prompt_per_1k": float, "completion_per_1k": float } }
    """
    in_tok_col  = prefix + "input_tokens"
    out_tok_col = prefix + "output_tokens"

    in_tok  = row.get(in_tok_col, None)
    out_tok = row.get(out_tok_col, None)

    # Model did not run / tokens missing
    if pd.isna(in_tok) or pd.isna(out_tok):
        return None

    prompt_cost_per_1k     = pricing[model_name]["prompt_per_1k"]
    completion_cost_per_1k = pricing[model_name]["completion_per_1k"]

    cost = (in_tok / 1000.0)  * prompt_cost_per_1k \
         + (out_tok / 1000.0) * completion_cost_per_1k

    return float(cost)


# -----------------------------------------------------------
# 2. Aggregate Cost per Model (e.g., mean cost over dataset)
# -----------------------------------------------------------
def compute_model_avg_cost(df, model_name, prefix, pricing, agg="mean"):
    """
    Compute an aggregate cost per model over a dataframe.

    Returns a single scalar (mean or median) cost in USD.
    """
    costs = []

    for _, row in df.iterrows():
        c = compute_true_cost(row, model_name, prefix, pricing)
        if c is not None:
            costs.append(c)

    if not costs:
        return None

    costs = np.array(costs, dtype=float)
    if agg == "mean":
        return float(costs.mean())
    elif agg == "median":
        return float(np.median(costs))
    else:
        raise ValueError(f"Unknown agg='{agg}', use 'mean' or 'median'.")


# -----------------------------------------------------------
# 3. Normalize Model Costs to [0, 1]
# -----------------------------------------------------------
def normalize_costs(model_costs):
    """
    Normalize a dict of {model_name: cost} to [0, 1].

    Returns
    -------
    dict {model_name: normalized_cost}
      cheapest model → 0.0
      most expensive → 1.0
    """
    # Filter out None
    filtered = {m: c for m, c in model_costs.items() if c is not None}
    if not filtered:
        return {m: None for m in model_costs}

    values = np.array(list(filtered.values()), dtype=float)
    min_c = float(values.min())
    max_c = float(values.max())

    if max_c == min_c:
        # All models same cost → all zero
        return {m: 0.0 if c is not None else None for m, c in model_costs.items()}

    norm = {
        m: (c - min_c) / (max_c - min_c)
        for m, c in model_costs.items()
        if c is not None
    }

    # Keep None for models without cost
    for m, c in model_costs.items():
        if c is None:
            norm[m] = None

    return norm


# -----------------------------------------------------------
# 4. Combine Performance & Cost into a Single Score
# -----------------------------------------------------------
def combine_perf_and_cost(perf, cost_norm, lambda_cost=1.0, mode="linear"):
    """
    Combine performance and normalized cost into a single routing score.

    Parameters
    ----------
    perf : float
        Performance score for (sample, model). Typically ~[0,1].
    cost_norm : float or None
        Normalized cost in [0,1]. If None, cost is ignored.
    lambda_cost : float
        Strength of cost penalty.
    mode : str
        "linear"  → score = perf - lambda_cost * cost_norm
        "exp"     → score = perf * exp(-lambda_cost * cost_norm)
        "ratio"   → score = perf / (1.0 + lambda_cost * cost_norm)

    Returns
    -------
    float
        Final combined score (higher is better).
    """
    if cost_norm is None:
        # No cost available → pure performance
        return float(perf)

    if mode == "linear":
        # Baseline: linear tradeoff between perf and cost
        return float(perf - lambda_cost * cost_norm)

    elif mode == "exp":
        # Non-linear option: keep perf, but penalize cost exponentially
        penalty = np.exp(-lambda_cost * cost_norm)
        return float(perf * penalty)

    elif mode == "ratio":
        # Another non-linear option: performance per "cost unit"
        return float(perf / (1.0 + lambda_cost * cost_norm))

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'linear', 'exp', or 'ratio'.")


# -----------------------------------------------------------
# 5. Notebook helpers
# -----------------------------------------------------------
def add_cost_columns_for_models(
    df: pd.DataFrame,
    model_specs: Sequence[Dict],
    pricing: Dict[str, Dict[str, float]],
    cost_suffix: str = "__cost",
    cost_norm_suffix: str = "__cost_norm",
    valid_suffix: str = "__valid_mask",
) -> pd.DataFrame:
    """Add `<model_name>__cost` (USD) and `<model_name>__cost_norm` columns."""
    df_out = df.copy()

    for spec in model_specs:
        name = spec["name"]
        prefix = spec["prefix"]
        valid_col = f"{name}{valid_suffix}"

        costs = [
            compute_true_cost(row, name, prefix, pricing)
            for _, row in df_out.iterrows()
        ]
        cost_series = pd.Series(costs, index=df_out.index, dtype="float64")

        valid_mask = df_out.get(valid_col, pd.Series(True, index=df_out.index)).astype(bool)
        cost_series = cost_series.where(valid_mask, np.nan)

        valid_costs = cost_series[valid_mask & cost_series.notna()]
        if valid_costs.empty:
            cost_norm = pd.Series(0.0, index=df_out.index)
        else:
            c_min = float(valid_costs.min())
            c_max = float(valid_costs.max())
            if np.isclose(c_max, c_min):
                cost_norm = pd.Series(0.0, index=df_out.index)
            else:
                cost_norm = (cost_series - c_min) / (c_max - c_min)
            cost_norm = cost_norm.fillna(1.0)

        df_out[f"{name}{cost_suffix}"] = cost_series
        df_out[f"{name}{cost_norm_suffix}"] = cost_norm

    return df_out


# def build_cost_matrix(
#     df: pd.DataFrame,
#     model_names: Sequence[str],
#     cost_suffix: str = "__cost",
#     invalid_fill: float = 1e6,
# ) -> np.ndarray:
#     """Return `[N, M]` matrix with cost per sample/model."""
#     mats = []
#     for name in model_names:
#         col = f"{name}{cost_suffix}"
#         cost = df.get(col, pd.Series(np.nan, index=df.index)).astype(float)
#         mats.append(cost.fillna(invalid_fill).values)
#     return np.stack(mats, axis=1)

def build_cost_matrix(df, model_names, cost_suffix="__cost", invalid_is_zero=True, invalid_fill=1e6):
    """
    Build cost matrix [N x M].

    If invalid_is_zero = True (recommended):
        If a model's valid_mask is False → set cost = 0.

    Otherwise fallback to invalid_fill (old behaviour).
    """
    cost_mat = np.zeros((len(df), len(model_names)))

    for j, m in enumerate(model_names):
        col_cost = f"{m}{cost_suffix}"
        col_valid = f"{m}__valid_mask"

        if col_cost in df.columns:
            raw_cost = df[col_cost].to_numpy()
            valid_mask = df[col_valid].to_numpy() if col_valid in df.columns else np.isfinite(raw_cost)

            if invalid_is_zero:
                # Correct behavior: invalid models contribute ZERO cost
                fixed_cost = np.where(valid_mask, raw_cost, 0.0)
            else:
                fixed_cost = np.where(valid_mask & np.isfinite(raw_cost), raw_cost, invalid_fill)

            cost_mat[:, j] = fixed_cost

        else:
            # Entire column missing → also cost = 0
            cost_mat[:, j] = 0.0 if invalid_is_zero else invalid_fill

    return cost_mat


# def compute_utility_matrix(
#     perf: np.ndarray,
#     cost: np.ndarray,
#     scheme: str = "linear",
#     lambda_cost: float = 10000.0,
#     delta: float = 0.02,
#     eps: float = 1e-8,
# ) -> np.ndarray:
#     """Return utility matrix for the requested cost/performance trade-off scheme."""
#     perf = np.asarray(perf, dtype=float)
#     cost = np.asarray(cost, dtype=float)

#     if scheme == "perf_only":
#         return perf.copy()

#     if scheme == "linear":
#         return perf - lambda_cost * cost

#     if scheme == "eff_ratio":
#         return perf / (cost + eps)

#     if scheme == "lexicographic":
#         max_perf = perf.max(axis=1, keepdims=True)
#         is_close = perf >= (max_perf - delta)
#         util = 100.0 * perf - cost
#         util[~is_close] -= 1000.0
#         return util

#     raise ValueError(f"Unknown scheme: {scheme}")



import numpy as np
import pandas as pd


def compute_utility_matrix(
    perf: np.ndarray,
    cost: np.ndarray,
    scheme: str = "linear",
    lambda_cost: float = 10000.0,
    delta: float = 0.02,
    perf_target: float | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Return utility matrix for the requested cost/performance trade-off scheme.

    Parameters
    ----------
    perf : np.ndarray
        Shape (n_samples, n_models). Higher is better (e.g., accuracy, score).
    cost : np.ndarray
        Shape (n_samples, n_models). Higher is worse (e.g., USD, tokens).
    scheme : str
        One of:
        - "perf_only"            : performance only, ignore cost
        - "linear"               : perf - lambda_cost * cost
        - "eff_ratio"            : perf / (cost + eps)
        - "lexicographic"        : perf-first, cost as tiebreak (delta controls closeness)
        - "min_cost_satisficing" : require perf >= perf_target (or delta), then pick cheapest
        - "quad_cost"            : perf - lambda_cost * cost**2
        - "log_cost"             : perf - lambda_cost * log1p(cost)
    lambda_cost : float
        Cost weight for schemes that use it.
    delta : float
        For "lexicographic": closeness tolerance in performance.
        For "min_cost_satisficing" (if perf_target is None): used as the perf threshold.
    perf_target : float or None
        Optional explicit performance threshold for "min_cost_satisficing".
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    util : np.ndarray
        Utility matrix, same shape as `perf`.
    """
    perf = np.asarray(perf, dtype=float)
    cost = np.asarray(cost, dtype=float)

    if scheme == "perf_only":
        # Upper bound: routing only by performance.
        return perf.copy()

    if scheme == "linear":
        # Classic linear tradeoff.
        return perf - lambda_cost * cost

    if scheme == "eff_ratio":
        # Bang-for-buck: higher score per unit of cost.
        return perf / (cost + eps)

    if scheme == "lexicographic":
        # Performance is king. Among models within delta of max perf,
        # prefer higher perf and lower cost.
        max_perf = perf.max(axis=1, keepdims=True)
        is_close = perf >= (max_perf - delta)

        util = 100.0 * perf - cost
        # Harshly penalize anything not within delta of the best perf.
        util[~is_close] -= 1000.0
        return util

    if scheme == "min_cost_satisficing":
        # Require performance >= threshold, then pick the cheapest model.
        # Use explicit perf_target if given, else reuse delta as threshold.
        threshold = perf_target if perf_target is not None else delta

        meets_target = perf >= threshold
        util = -cost.copy()            # lower cost => higher utility
        util[~meets_target] = -1e12    # nuke models that don't meet target
        return util

    if scheme == "quad_cost":
        # Stronger penalty on expensive models.
        return perf - lambda_cost * (cost ** 2)

    if scheme == "log_cost":
        # Softer penalty across large cost ranges.
        return perf - lambda_cost * np.log1p(cost)

    raise ValueError(f"Unknown scheme: {scheme}")
