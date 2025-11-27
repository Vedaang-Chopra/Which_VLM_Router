# results_writer.py
from __future__ import annotations
import os, json, csv, datetime, base64
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, List, Optional

try:
    import pandas as pd  # optional; falls back to csv module if missing
except Exception:
    pd = None


def _nowstamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _rowify(x: Any) -> Dict[str, Any]:
    if is_dataclass(x):
        d = asdict(x)
    elif isinstance(x, dict):
        d = dict(x)
    elif hasattr(x, "__dict__"):
        d = vars(x).copy()
    else:
        d = {"value": x}
    # Drop images if present (avoids bytes in results)
    d.pop("images", None)
    return d


def _json_default(x: Any) -> Any:
    if isinstance(x, bytes):
        return base64.b64encode(x).decode("ascii")
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, set):
        return list(x)
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return tolist()
    if hasattr(x, "__dict__"):
        return x.__dict__
    return str(x)


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def _csv_write(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        cols = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def _flatten_by_cat(by_cat: Optional[Dict[str, List[Any]]]) -> List[Any]:
    if not by_cat:
        return []
    out: List[Any] = []
    for _, lst in by_cat.items():
        out.extend(lst)
    return out


class ResultsWriter:
    def __init__(self, output_dir: str = "outputs", prefix: Optional[str] = None, timestamp: bool = True):
        self.output_dir = output_dir
        self.prefix = prefix or "run"
        self.tag = _nowstamp() if timestamp else ""
        os.makedirs(self.output_dir, exist_ok=True)

    def _path(self, stem: str, ext: str) -> str:
        name = f"{self.prefix}_{stem}"
        if self.tag:
            name += f"_{self.tag}"
        return os.path.join(self.output_dir, f"{name}.{ext}")

    # ---------- Eval (CSV/JSONL, optional) ----------
    def save_eval(self, all_results: List[Any]) -> Dict[str, str]:
        rows = [_rowify(r) for r in all_results]
        p_csv = self._path("eval", "csv")
        p_jsonl = self._path("eval", "jsonl")
        _csv_write(p_csv, rows)
        _jsonl_write(p_jsonl, rows)
        return {"csv": p_csv, "jsonl": p_jsonl}

    # ---------- Judge (CSV/JSONL, optional) ----------
    def save_judged(self, judged_all: List[Any]) -> Dict[str, str]:
        rows = [_rowify(r) for r in judged_all]
        p_csv = self._path("judged", "csv")
        p_jsonl = self._path("judged", "jsonl")
        _csv_write(p_csv, rows)
        _jsonl_write(p_jsonl, rows)
        return {"csv": p_csv, "jsonl": p_jsonl}

    # ---------- Buckets & summaries (optional) ----------
    def save_buckets(self, buckets: Dict[str, List[dict]]) -> Dict[str, str]:
        clean = {
            cat: [{k: v for k, v in rec.items() if k != "images"} for rec in recs]
            for cat, recs in buckets.items()
        }

        p_full = self._path("buckets_full_nobytes", "json")
        with open(p_full, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False)

        summary = {
            "counts": {c: len(recs) for c, recs in clean.items()},
            "preview_ids": {c: [r.get("conversation_id") for r in recs[:5]] for c, recs in clean.items()},
        }
        p_sum = self._path("buckets_summary", "json")
        with open(p_sum, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return {"json_full": p_full, "json_summary": p_sum}

    # ---------- Single JSON with eval + judged (across all buckets) ----------
    def save_one_json(
        self,
        *,
        # pass either the flat arrays...
        all_results: Optional[List[Any]] = None,
        judged_all: Optional[List[Any]] = None,
        # ...or pass per-category dicts; we'll flatten them
        eval_by_cat: Optional[Dict[str, List[Any]]] = None,
        judged_by_cat: Optional[Dict[str, List[Any]]] = None,
        stem: str = "combined",
    ) -> str:
        """
        Write ONE JSON file that includes eval + judged results from all buckets.
        You can provide either flat arrays (all_results/judged_all) OR per-category dicts
        (eval_by_cat/judged_by_cat). If both are provided, flat arrays take precedence.
        """
        if all_results is None and eval_by_cat is not None:
            all_results = _flatten_by_cat(eval_by_cat)
        if judged_all is None and judged_by_cat is not None:
            judged_all = _flatten_by_cat(judged_by_cat)

        # Build per-category counts if dicts were provided
        per_category_counts: Dict[str, Dict[str, int]] = {}
        if eval_by_cat or judged_by_cat:
            cats = set()
            if eval_by_cat:
                cats |= set(eval_by_cat.keys())
            if judged_by_cat:
                cats |= set(judged_by_cat.keys())
            for c in sorted(cats):
                per_category_counts[c] = {
                    "eval": len(eval_by_cat.get(c, [])) if eval_by_cat else 0,
                    "judged": len(judged_by_cat.get(c, [])) if judged_by_cat else 0,
                }

        payload: Dict[str, Any] = {
            "meta": {"prefix": self.prefix, "timestamp": self.tag},
            "per_category_counts": per_category_counts or None,
            "eval": [_rowify(r) for r in (all_results or [])],
            "judged": [_rowify(r) for r in (judged_all or [])],
        }

        path = self._path(stem, "json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=_json_default)
        return path

    # ---------- Convenience multi-file (optional) ----------
    def save_all(
        self,
        *,
        all_results: Optional[List[Any]] = None,
        judged_all: Optional[List[Any]] = None,
        buckets: Optional[Dict[str, List[dict]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        if all_results is not None:
            out["eval"] = self.save_eval(all_results)
        if judged_all is not None:
            out["judged"] = self.save_judged(judged_all)
        if buckets is not None:
            out["buckets"] = self.save_buckets(buckets)
        return out
