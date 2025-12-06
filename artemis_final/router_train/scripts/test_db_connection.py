#!/usr/bin/env python3
"""
Test database connection and verify schema.

Usage:
    python scripts/test_db_connection.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, DBConfig
from db_utils import get_table_info, test_connection


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test database connection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="Database host",
    )

    parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Database port",
    )

    parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Database name",
    )

    parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="Database user",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Load config
    config = Config.default()

    # Override with CLI args
    if args.db_host:
        config.db.host = args.db_host
    if args.db_port:
        config.db.port = args.db_port
    if args.db_name:
        config.db.name = args.db_name
    if args.db_user:
        config.db.user = args.db_user

    logger.info("=" * 80)
    logger.info("Database Connection Test")
    logger.info("=" * 80)
    logger.info(f"Host: {config.db.host}")
    logger.info(f"Port: {config.db.port}")
    logger.info(f"Database: {config.db.name}")
    logger.info(f"User: {config.db.user}")
    logger.info("=" * 80)

    # Test connection
    logger.info("\nTesting connection...")
    if not test_connection(config.db):
        logger.error("✗ Connection failed!")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check PostgreSQL is running: pg_isready")
        logger.error("2. Verify credentials")
        logger.error("3. Check network connectivity")
        logger.error("4. Ensure database exists")
        return 1

    logger.info("✓ Connection successful!")

    # Get table info
    logger.info("\nQuerying table information...")
    table_info = get_table_info(config.db)

    logger.info("\n" + "=" * 80)
    logger.info("Table Information")
    logger.info("=" * 80)

    all_tables_exist = True
    expected_tables = ["vlm_sample", "vlm_responses", "vlm_evaluation", "vlm_images"]

    for table in expected_tables:
        count = table_info.get(table)
        if count is None:
            logger.error(f"✗ {table:20s} - NOT FOUND")
            all_tables_exist = False
        elif count == 0:
            logger.warning(f"⚠ {table:20s} - {count:,} rows (empty)")
        else:
            logger.info(f"✓ {table:20s} - {count:,} rows")

    logger.info("=" * 80)

    if not all_tables_exist:
        logger.error("\n✗ Some required tables are missing!")
        logger.error("\nRequired schema:")
        logger.error("  - vlm_sample (sample_id, prompt_raw, source_dataset, router_task)")
        logger.error("  - vlm_responses (sample_id, model_name, response_text, cost_usd, latency_ms)")
        logger.error("  - vlm_evaluation (sample_id, model_name, exact_match, f1, vqa_acc, critic_score, hallucination_score)")
        logger.error("  - vlm_images (sample_id, img_width, img_height)")
        return 1

    # Check for empty tables
    empty_tables = [t for t, c in table_info.items() if c == 0]
    if empty_tables:
        logger.warning(f"\n⚠ Warning: {len(empty_tables)} table(s) are empty: {empty_tables}")
        logger.warning("The dataset builder will fail if tables have no data.")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Database connection test PASSED!")
    logger.info("You can now run: python scripts/run_build_dataset.py")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
