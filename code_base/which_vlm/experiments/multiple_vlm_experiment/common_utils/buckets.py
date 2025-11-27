

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datasets import load_dataset
import time
from collections import defaultdict


# --- your sampler (unchanged) ---
def sample_by_category(
    it,
    category_key="categories",
    id_key=None,                 # if provided, used to dedupe (e.g., "conversation_id")
    target_categories=None,      # if provided, limit to these category names
    k=10,
    exclusive=False,
):
    """
    Returns dict: {category: [records]}
    - Non-exclusive: a record can appear in multiple category lists.
    - Exclusive: each record appears in at most one category (the 'neediest' bucket).
    """
    buckets = defaultdict(list)
    counts = defaultdict(int)
    used_ids = set()

    def all_filled():
        cats = target_categories or list(counts.keys())
        return all(counts.get(c, 0) >= k for c in cats)

    for rec in it:
        cats_dict = rec.get(category_key, {}) or {}
        cats = [c for c, v in cats_dict.items() if v]
        if not cats:
            continue

        if target_categories is not None:
            cats = [c for c in cats if c in target_categories]
            if not cats:
                continue

        rec_id = rec.get(id_key) if id_key else None
        if exclusive:
            if rec_id is not None and rec_id in used_ids:
                continue
            neediest = None
            neediest_count = float("inf")
            for c in cats:
                if counts[c] < k and counts[c] < neediest_count:
                    neediest = c
                    neediest_count = counts[c]
            if neediest is not None:
                buckets[neediest].append(rec)
                counts[neediest] += 1
                if rec_id is not None:
                    used_ids.add(rec_id)
        else:
            for c in cats:
                if counts[c] < k:
                    buckets[c].append(rec)
                    counts[c] += 1

        if all_filled():
            break

    return dict(buckets)




# --- upgraded runner with debug + separate bucketing ---
class CategoryEvalRunner:
    """
    1) build_buckets(): build category buckets with debug summaries.
    2) run()/run_with_buckets(): evaluate k items per category and collect results.
    """
    def __init__(
        self,
        eval_runner,                    # your EvalRunner (must expose run_on_sample(sample) -> EvalOutput)
        *,
        k: int = 10,
        id_key: str = "conversation_id",
        exclusive: bool = True,
        verbose: bool = True,
        preview_per_cat: int = 2,       # how many IDs/text previews to print per cat
        text_preview_chars: int = 80,   # trim user question in debug prints
    ):
        self.eval_runner = eval_runner
        self.k = k
        self.id_key = id_key
        self.exclusive = exclusive
        self.verbose = verbose
        self.preview_per_cat = preview_per_cat
        self.text_preview_chars = text_preview_chars

    # ---------- Bucketing (separate function) ----------
    def build_buckets(
        self,
        iterable: Iterable[dict],
        target_categories: List[str],
        *,
        category_key: str = "categories",
    ) -> Dict[str, List[dict]]:
        if self.verbose:
            print("[BUCKETS] Building buckets...")
            print(f"  - target_categories: {target_categories}")
            print(f"  - k per category   : {self.k}")
            print(f"  - exclusive        : {self.exclusive}")
            print(f"  - id_key           : {self.id_key}")
            t0 = time.perf_counter()

        buckets = sample_by_category(
            it=iterable,
            category_key=category_key,
            id_key=self.id_key,
            target_categories=target_categories,
            k=self.k,
            exclusive=self.exclusive,
        )

        if self.verbose:
            dt = time.perf_counter() - t0
            print(f"[BUCKETS] Done in {dt:.2f}s")
            self._debug_bucket_counts(buckets, target_categories)
            self._debug_preview(buckets)

        return buckets

    # ---------- Main run (keeps original return signature) ----------
    def run(
        self,
        iterable: Iterable[dict],
        target_categories: List[str],
        *,
        category_key: str = "categories",
    ) -> Tuple[Dict[str, List[Any]], List[Any]]:
        buckets = self.build_buckets(
            iterable=iterable,
            target_categories=target_categories,
            category_key=category_key,
        )
        return self._run_buckets(buckets)

    # ---------- Variant that also returns the buckets ----------
    def run_with_buckets(
        self,
        iterable: Iterable[dict],
        target_categories: List[str],
        *,
        category_key: str = "categories",
    ) -> Tuple[Dict[str, List[Any]], List[Any], Dict[str, List[dict]]]:
        buckets = self.build_buckets(
            iterable=iterable,
            target_categories=target_categories,
            category_key=category_key,
        )
        results_by_cat, all_results = self._run_buckets(buckets)
        return results_by_cat, all_results, buckets

    # ---------- Internal helpers ----------
    def _run_buckets(
        self, buckets: Dict[str, List[dict]]
    ) -> Tuple[Dict[str, List[Any]], List[Any]]:
        results_by_cat: Dict[str, List[Any]] = {}
        all_results: List[Any] = []

        if self.verbose:
            print("[EVAL] Starting per-category evaluation...")

        for cat, recs in buckets.items():
            if self.verbose:
                print(f"[EVAL] Category '{cat}' → {len(recs)} item(s)")
            cat_results = []
            for i, rec in enumerate(recs, 1):
                conv_id = rec.get(self.id_key)
                try:
                    t0 = time.perf_counter()
                    out = self.eval_runner.run_on_sample(rec)  # -> EvalOutput
                    dt = time.perf_counter() - t0
                    cat_results.append(out)
                    all_results.append(out)
                    if self.verbose:
                        uq = getattr(out, "user_question", "")
                        uq_short = (uq[:self.text_preview_chars] + "…") if len(uq) > self.text_preview_chars else uq
                        print(f"  - [{cat}] {i}/{len(recs)} conv_id={conv_id} "
                              f"latency={out.latency_sec:.3f}s (wall {dt:.3f}s)  Q: {uq_short}")
                except Exception as e:
                    print(f"  - [{cat}] {i}/{len(recs)} conv_id={conv_id}  ERROR: {e}")
            results_by_cat[cat] = cat_results

        if self.verbose:
            total = sum(len(v) for v in results_by_cat.values())
            print(f"[EVAL] Done. Collected {total} results across {len(results_by_cat)} categories.")
        return results_by_cat, all_results

    def _debug_bucket_counts(self, buckets: Dict[str, List[dict]], targets: List[str]) -> None:
        print("[BUCKETS] Counts:")
        for c in targets:
            n = len(buckets.get(c, []))
            mark = "✓" if n >= self.k else "!"
            print(f"  - {c:<16} {n:>3}/{self.k} {mark}")
        # warn about extra cats (present in data but not requested)
        extras = [c for c in buckets.keys() if c not in targets]
        if extras:
            print(f"[BUCKETS] Note: extra categories present (not requested): {extras}")

    def _debug_preview(self, buckets: Dict[str, List[dict]]) -> None:
        print("[BUCKETS] Preview (first few IDs per category):")
        for c, recs in buckets.items():
            ids = [r.get(self.id_key) for r in recs[: self.preview_per_cat]]
            print(f"  - {c:<16} {ids}")
