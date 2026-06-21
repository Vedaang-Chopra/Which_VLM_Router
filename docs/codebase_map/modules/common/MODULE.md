# Codebase Map: common
>
> Directory: artemis_final/common/
> Entry point: config_loader.py::load_global_config()
> Status: COMPLETE

## Responsibility

Shared configuration loading and utility functions used by all ARTEMIS modules. Provides the `GlobalConfig` dataclass that every other module imports for its settings.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `config_loader.py` | Schema | `GlobalConfig` dataclass; `load_global_config(path?)` singleton from YAML; `get_base_dir()` |
| `utils.py` | Utility | Shared utility functions |
| `types.py` | Schema | Type aliases: `RouterRequest`, `RouterDecision`, `LBDecision` |
| `db.py` | Utility | Database connection helpers |
| `__init__.py` | Runner | Package init with public re-exports |

## Change Guide

- **To add a new config field**: add to `GlobalConfig` dataclass in `config_loader.py`; all modules importing `load_global_config()` will see it
- **To change config file location**: pass `config_path` to `load_global_config(path)`
- **To add a new type alias**: add to `types.py`

## Dependencies

External: `yaml`, `dataclasses`, `pathlib`
