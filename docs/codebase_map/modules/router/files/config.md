# config.py
>
> Module: router
> Layer: Schema
> Path: artemis_final/router/core/config.py

## Purpose

Configuration dataclasses for traffic simulation, SQL/W&B logging, and load balancer interface.

## Classes

### TrafficConfig

Traffic simulation parameters. Fields: default_rps, default_duration_seconds, synthetic_image_shape, synthetic_text_length, arrival_pattern, sample_selection_strategy.

### LoggingConfig

SQL and W&B logging settings. Fields: sql_enabled, db_url, logs_table, wandb_enabled, wandb_project, log_router_probs, log_metadata, log_latency, batch_size, flush_interval_seconds.

### LBConfig

Load balancer interface configuration. Fields: enabled, protocol (http/grpc/kafka), endpoint, timeout_seconds, retry_attempts, fallback_to_router_choice, headers, api_key, health_check_interval_seconds.

## Imports

Internal: None
External: dataclasses, typing

## Known Issues

None.
