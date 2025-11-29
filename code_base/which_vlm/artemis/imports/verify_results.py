#!/usr/bin/env python3
"""
Utility script to verify evaluation results and resume interrupted runs.

Usage:
    python verify_results.py --run-id exp_20250127_123456
    python verify_results.py --latest
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "inference_vlm.yaml"


def find_latest_run(base_dir: Path) -> Path:
    """Find the most recent run directory."""
    run_dirs = sorted(base_dir.glob("exp_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    return run_dirs[-1]


def load_expected_model_names(config_path: Path) -> List[str]:
    """Return model names declared in a YAML config, if any."""
    if yaml is None:
        print("⚠️  PyYAML is not installed. Skipping model coverage checks.")
        return []

    try:
        with config_path.expanduser().open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"⚠️  Model config not found at {config_path}. Skipping model coverage checks.")
        return []
    except Exception as exc:
        print(f"⚠️  Could not read model config {config_path}: {exc}. Skipping model coverage checks.")
        return []

    models = cfg.get("models")
    if not isinstance(models, list):
        print(f"⚠️  No 'models' list found in {config_path}. Skipping model coverage checks.")
        return []

    names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    if not names:
        print(f"⚠️  Config {config_path} does not define any model names. Skipping model coverage checks.")
    return names


def verify_run(
    run_dir: Path,
    verbose: bool = True,
    expected_models: Sequence[str] | None = None,
) -> Dict:
    """Verify results from a run directory."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Verifying results in: {run_dir}")
        print(f"{'='*80}\n")

    # Check directory exists
    if not run_dir.exists():
        print(f"❌ Directory not found: {run_dir}")
        return

    # Find all parquet files
    parquet_files = list(run_dir.glob("*.parquet"))
    config_files = [f for f in parquet_files if f.name != "all_results.parquet"]
    combined_file = run_dir / "all_results.parquet"

    if verbose:
        print(f"📁 Found {len(config_files)} config files")
        print(f"📁 Combined file exists: {combined_file.exists()}")

    # Load and verify each config
    results = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "configs": {},
        "total_records": 0,
        "models": set(),
        "missing_models": {},
        "unexpected_models": {},
    }
    expected_model_set = set(expected_models or [])

    for config_file in sorted(config_files):
        config_name = config_file.stem
        try:
            df = pd.read_parquet(config_file)
            model_count = df["model_name"].nunique() if "model_name" in df.columns else 0
            sample_count = df["sample_id"].nunique() if "sample_id" in df.columns else 0
            results["configs"][config_name] = {
                "records": len(df),
                "models": model_count,
                "samples": sample_count,
                "missing_models": [],
                "unexpected_models": [],
            }
            results["total_records"] += len(df)
            if "model_name" in df.columns:
                results["models"].update(df["model_name"].unique())

            if verbose:
                print(f"✅ {config_name:30s} {len(df):6d} records, "
                      f"{sample_count:4d} samples, "
                      f"{model_count:2d} models")

            if expected_model_set and "model_name" in df.columns:
                actual_models = set(df["model_name"].dropna().unique().tolist())
                missing = sorted(expected_model_set - actual_models)
                unexpected = sorted(actual_models - expected_model_set)

                if missing:
                    results["configs"][config_name]["missing_models"] = missing
                    results["missing_models"][config_name] = missing
                    if verbose:
                        print(f"   ⚠️ Missing models: {', '.join(missing)}")
                if unexpected:
                    results["configs"][config_name]["unexpected_models"] = unexpected
                    results["unexpected_models"][config_name] = unexpected
                    if verbose:
                        print(f"   ⚠️ Unexpected models: {', '.join(unexpected)}")

        except Exception as e:
            print(f"❌ {config_name:30s} ERROR: {e}")
            results["configs"][config_name] = {"error": str(e)}

    results["models"] = sorted(list(results["models"]))

    # Verify combined file
    if combined_file.exists():
        try:
            df_combined = pd.read_parquet(combined_file)
            if verbose:
                print(f"\n📊 Combined file: {len(df_combined):,} records")

            if len(df_combined) != results["total_records"]:
                print(f"⚠️  Warning: Combined file has {len(df_combined)} records, "
                      f"but sum of configs is {results['total_records']}")
        except Exception as e:
            print(f"❌ Error reading combined file: {e}")

    # Check summary.json
    summary_file = run_dir / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            if verbose:
                print(f"\n📝 Summary JSON exists")
                print(f"   Overall accuracy: {summary.get('overall_accuracy', 'N/A'):.4f}")
                print(f"   Total samples: {summary.get('total_samples', 'N/A'):,}")
        except Exception as e:
            print(f"❌ Error reading summary.json: {e}")
    else:
        if verbose:
            print(f"\n⚠️  No summary.json found")

    if verbose:
        print(f"\n{'='*80}")
        print(f"Summary:")
        print(f"  Configs processed: {len(results['configs'])}")
        print(f"  Total records: {results['total_records']:,}")
        print(f"  Models: {', '.join(results['models'])}")
        print(f"{'='*80}\n")

    if expected_model_set and verbose:
        if results["missing_models"]:
            print("\n⚠️  Configs with missing model outputs detected:")
            for cfg, missing in sorted(results["missing_models"].items()):
                print(f"   - {cfg}: {', '.join(missing)}")
        else:
            print("\n✅ All configs include the expected models.")

    return results


def find_missing_configs(run_dir: Path, all_configs: list) -> list:
    """Find configs that haven't been processed yet."""
    processed = [f.stem for f in run_dir.glob("*.parquet") if f.name != "all_results.parquet"]
    missing = [c for c in all_configs if c not in processed]
    return missing


def main():
    default_model_config = DEFAULT_MODEL_CONFIG

    parser = argparse.ArgumentParser(description="Verify evaluation results")
    parser.add_argument("--run-id", help="Specific run ID to verify (e.g., exp_20250127_123456)")
    parser.add_argument("--latest", action="store_true", help="Verify latest run")
    parser.add_argument("--base-dir", default="./experiment_data/runs", help="Base directory for runs")
    parser.add_argument("--check-missing", action="store_true", help="Check for missing configs")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--model-config",
        default=str(default_model_config),
        help="Path to YAML file listing expected model names (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Disable per-config model coverage checks",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Determine which run to verify
    if args.run_id:
        run_dir = base_dir / args.run_id
    elif args.latest:
        run_dir = find_latest_run(base_dir)
    else:
        # Default to latest
        run_dir = find_latest_run(base_dir)

    expected_models: Sequence[str] = []
    if not args.skip_model_check:
        model_config_path = Path(args.model_config)
        expected_models = load_expected_model_names(model_config_path)

    # Verify results
    results = verify_run(
        run_dir,
        verbose=not args.quiet,
        expected_models=expected_models,
    )

    # Check for missing configs if requested
    if args.check_missing:
        try:
            # Try to import config
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import ALL_CAULDRON_CONFIGS

            missing = find_missing_configs(run_dir, ALL_CAULDRON_CONFIGS)
            if missing:
                print(f"\n⚠️  Missing {len(missing)} configs:")
                for config in missing:
                    print(f"   - {config}")
                print(f"\nTo resume, add to notebook:")
                print(f"configs_to_process = {missing}")
            else:
                print(f"\n✅ All configs processed!")
        except ImportError:
            print("\n⚠️  Could not import config to check missing configs")


if __name__ == "__main__":
    main()
