# Module: common
>
> Status: COMPLETE
> Directory: artemis_final/common/
> Entry point: config_loader.py::load_global_config()
> Last updated: 2026-06-20

## Purpose

Shared configuration loading and utility functions used across all ARTEMIS modules.

## Public API

| Function | Signature | Purpose |
|---|---|---|
| `load_global_config` | `load_global_config(config_path?) -> GlobalConfig` | Load and return GlobalConfig singleton |
| `get_base_dir` | `get_base_dir() -> Path` | Get base directory for relative paths |

## Internal Structure

| File | Responsibility |
|---|---|
| `config_loader.py` | GlobalConfig dataclass; load_global_config() from YAML |
| `utils.py` | Shared utility functions |
| `types.py` | Shared type definitions |
| `db.py` | Database connection helpers |
| `__init__.py` | Package init with public re-exports |

## Dependencies

External: `yaml`, `dataclasses`, `pathlib`

## Known Issues

None. This module is fully complete and used by all other modules.

## What an Agent Must Know

- `GlobalConfig` contains nested dataclasses for `router`, `load_balancer`, and `inference_engine`.
- Config is loaded once and cached. For hot-reloading, call `load_global_config()` again with the new path.
- All modules import from `common` — changing GlobalConfig structure affects the entire project.
