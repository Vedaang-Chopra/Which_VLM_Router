# schemas.py
>
> Module: router
> Layer: Schema
> Path: artemis_final/router/core/schemas.py

## Purpose

Unified data structures for all sample sources (DB, HTTP, synthetic) and routing decisions. These types flow between router, load balancer, and evaluation.

## Classes

### Sample

Unified sample for all input sources. Fields: sample_id, source, text, image (PIL), image_uri, metadata, label.

### RouterDecision

Router's output for one sample. Fields: sample_id, chosen_model, probs (model→float), raw_logits, model_order, inference_ms.

### InferenceResult

Bundles a Sample with its RouterDecision.

### LogRecord

Flat record for SQL logging of routing decisions. Fields: timestamp, sample_id, source, text, image_uri, split, label, router_chosen_model, router_probs, router_inference_ms, extra_metadata.

## Imports

Internal: None (this is the base schema layer)
External: PIL.Image, dataclasses, typing

## Known Issues

None. This is a pure-schema file.
