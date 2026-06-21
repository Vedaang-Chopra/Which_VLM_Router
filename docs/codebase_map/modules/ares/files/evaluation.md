# evaluation.py
>
> Module: ares
> Layer: Core
> Path: artemis_final/ares/evaluation/evaluation.py

## Purpose

Ground-truth-based evaluation (Scorer) and optional Glider text-only fast evaluator.

## Classes

### Scorer

Ground truth scoring. Methods: `score(response, ground_truth) -> Dict[metric, value]`. Computes accuracy, F1, precision, recall.

### GliderEvaluator

Text-only fast evaluator. May load heavy models at init. Methods: `evaluate(text_response) -> score`.

### parse_glider_output

Parses Glider's text output into structured scores.

## Imports

Internal: None
External: `re`, `tqdm`, optional: glider library

## Known Issues

GliderEvaluator may load heavy models at init time (evaluation.py:32 note). Not all methods are fully implemented for all task types.
