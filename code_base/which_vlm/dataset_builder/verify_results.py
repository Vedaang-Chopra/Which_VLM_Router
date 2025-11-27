#!/usr/bin/env python3
"""
Utility script to verify evaluation results and resume interrupted runs.

Usage:
    python verify_results.py --run-id exp_20250127_123456
    python verify_results.py --latest
"""

import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


def find_latest_run(base_dir: Path) -> Path:
    """Find the most recent run directory."""
    run_dirs = sorted(base_dir.glob("exp_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    return run_dirs[-1]


def verify_run(run_dir: Path, verbose: bool = True):
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
    }

    for config_file in sorted(config_files):
        config_name = config_file.stem
        try:
            df = pd.read_parquet(config_file)
            results["configs"][config_name] = {
                "records": len(df),
                "models": df["model_name"].nunique() if "model_name" in df.columns else 0,
                "samples": df["sample_id"].nunique() if "sample_id" in df.columns else 0,
            }
            results["total_records"] += len(df)
            if "model_name" in df.columns:
                results["models"].update(df["model_name"].unique())

            if verbose:
                print(f"✅ {config_name:30s} {len(df):6d} records, "
                      f"{df['sample_id'].nunique():4d} samples, "
                      f"{df['model_name'].nunique():2d} models")
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

    return results


def find_missing_configs(run_dir: Path, all_configs: list) -> list:
    """Find configs that haven't been processed yet."""
    processed = [f.stem for f in run_dir.glob("*.parquet") if f.name != "all_results.parquet"]
    missing = [c for c in all_configs if c not in processed]
    return missing


def main():
    parser = argparse.ArgumentParser(description="Verify evaluation results")
    parser.add_argument("--run-id", help="Specific run ID to verify (e.g., exp_20250127_123456)")
    parser.add_argument("--latest", action="store_true", help="Verify latest run")
    parser.add_argument("--base-dir", default="./experiment_data/runs", help="Base directory for runs")
    parser.add_argument("--check-missing", action="store_true", help="Check for missing configs")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

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

    # Verify results
    results = verify_run(run_dir, verbose=not args.quiet)

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
