#!/usr/bin/env python3
"""
Setup script for Artemis Router.

This script helps you get started with the router by:
1. Checking prerequisites
2. Validating configuration
3. Testing database connectivity
4. Running a quick smoke test

Usage:
    python setup_router.py --config router_config.yaml
"""

import argparse
import sys
from pathlib import Path

try:
    from artemis_router.config import load_config, validate_config
    from artemis_router.router_engine import RouterEngine
    from artemis_router.traffic_simulator import make_synthetic_sample
except ImportError as e:
    print(f"Error importing artemis_router: {e}")
    print("\nMake sure you're in the router directory or have it in your PYTHONPATH:")
    print("  export PYTHONPATH=$PYTHONPATH:/path/to/router")
    sys.exit(1)


def check_imports():
    """Check that all required packages are installed."""
    print("Checking imports...")

    required = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "sqlalchemy": "SQLAlchemy",
        "pandas": "Pandas",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "numpy": "NumPy",
    }

    optional = {
        "wandb": "Weights & Biases (for logging)",
        "requests": "Requests (for HTTP LB)",
    }

    missing = []

    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    print("\nOptional packages:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - not installed (optional)")

    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install torch transformers sqlalchemy pandas pillow pyyaml numpy")
        return False

    print("\n✅ All required packages installed")
    return True


def check_config(config_path):
    """Load and validate configuration."""
    print(f"\nLoading configuration from: {config_path}")

    try:
        cfg = load_config(config_path)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None

    print("\nValidating configuration...")
    issues = validate_config(cfg)

    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return None

    print("✅ Configuration is valid")

    print("\nConfiguration summary:")
    print(f"  Device: {cfg.router.device}")
    print(f"  Dtype: {cfg.router.dtype}")
    print(f"  Models: {len(cfg.router.model_name_order)}")
    print(f"  Checkpoint: {cfg.router.checkpoint_path}")
    print(f"  SQL logging: {cfg.logging.sql_enabled}")
    print(f"  W&B logging: {cfg.logging.wandb_enabled}")

    return cfg


def test_database(cfg):
    """Test database connectivity."""
    print("\nTesting database connection...")

    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(cfg.data.db_url)

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        print("✅ Database connection successful")

        # Check if tables exist
        with engine.connect() as conn:
            # Check samples table
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {cfg.data.samples_table} LIMIT 1"))
                count = result.fetchone()[0]
                print(f"✅ Samples table exists ({count} rows)")
            except Exception as e:
                print(f"⚠️  Samples table issue: {e}")

            # Check logs table
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {cfg.data.logs_table} LIMIT 1"))
                count = result.fetchone()[0]
                print(f"✅ Logs table exists ({count} rows)")
            except Exception as e:
                print(f"⚠️  Logs table not found - run sql/router_logs_schema.sql to create it")

        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_router(cfg):
    """Test router initialization and inference."""
    print("\nTesting router initialization...")

    try:
        # Temporarily disable logging for smoke test
        cfg.logging.sql_enabled = False
        cfg.logging.wandb_enabled = False
        cfg.lb.enabled = False

        engine = RouterEngine(cfg)
        print("✅ Router engine initialized")

        print("\nRunning smoke test with synthetic sample...")
        sample = make_synthetic_sample(0, cfg.traffic)
        result = engine.route_sample(sample)

        print(f"✅ Inference successful")
        print(f"  Chosen model: {result.router_decision.chosen_model}")
        print(f"  Latency: {result.router_decision.inference_ms:.2f}ms")
        print(f"  Probabilities:")
        for model, prob in sorted(result.router_decision.probs.items(), key=lambda x: x[1], reverse=True):
            print(f"    {model:30s}: {prob:.4f}")

        return True

    except Exception as e:
        print(f"❌ Router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup and test Artemis Router")
    parser.add_argument("--config", type=str, default="router_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip-db", action="store_true",
                       help="Skip database connectivity test")
    parser.add_argument("--skip-router", action="store_true",
                       help="Skip router initialization test")

    args = parser.parse_args()

    print("="*60)
    print("Artemis Router Setup")
    print("="*60)

    # Check imports
    if not check_imports():
        sys.exit(1)

    # Check config
    cfg = check_config(args.config)
    if cfg is None:
        print("\n❌ Setup failed - fix configuration issues and try again")
        sys.exit(1)

    # Test database
    if not args.skip_db:
        if not test_database(cfg):
            print("\n⚠️  Database test failed - router may not work correctly")
            print("Fix database issues or use --skip-db to skip this test")

    # Test router
    if not args.skip_router:
        if not test_router(cfg):
            print("\n❌ Router test failed - check configuration and checkpoint")
            sys.exit(1)

    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run notebooks/01_router_unit_test.ipynb for comprehensive testing")
    print("  2. Run notebooks/02_traffic_simulation.ipynb for performance analysis")
    print("  3. Start building your application!")
    print("\nFor help, see README.md")


if __name__ == "__main__":
    main()
