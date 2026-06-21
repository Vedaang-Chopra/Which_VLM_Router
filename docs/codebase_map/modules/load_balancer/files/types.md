# types.py
>
> Module: load_balancer
> Layer: Schema
> Path: artemis_final/load_balancer/core/types.py

## Purpose

Core data types for load balancing: RouterOutput, SchedulingContext, SchedulingDecision, BudgetExhaustedError, TrafficSimulationResult.

## Classes

### RouterOutput

Input from the router. Fields: sample_id, task_type, router_probs (model→float), preferred_model, max_prob.

### SchedulingContext

Request context. Fields: arrival_ts_ms, load_profile, metadata.

### SchedulingDecision

Output from the scheduler. Fields: chosen_model, is_overloaded, est_latency_ms, est_cost_usd, queue_delay_ms, total_latency_ms, sla_violated, routing_method, modified_probs.

### BudgetExhaustedError

Exception raised when no model meets constraints.

### TrafficSimulationResult

Result of traffic simulation. Fields: arrival_rate, duration_s, total_requests, decisions, avg_latency_ms, p50/p95/p99_latency_ms, max_queue_length, avg_queue_delay_ms, throughput_rps, sla_violation_rate, avg_cost_usd, total_cost_usd, model_usage.

## Imports

Internal: dataclasses
External: typing

## Known Issues

None. Pure schema file.
