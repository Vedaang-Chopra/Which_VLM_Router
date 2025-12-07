#!/usr/bin/env python
"""
CLI Entrypoint for Router Evaluation Pipeline

Usage:
    python -m ares.evaluation \
        --db-url postgresql+psycopg2://user:pass@localhost/dbname \
        --split val \
        --router-tasks "chart_qa,general_qa" \
        --models "model_a,model_b" \
        --use-molmo \
        --use-glider \
        --batch-size 50 \
        --force
"""

import argparse
import logging
import os
import sys

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit('/ares/', 1)[0])

from sqlalchemy import create_engine

from ares.evaluation.router_eval_pipeline import RouterEvalPipeline
from inference_engine.runners import OpenAIStyleRunner
from inference_engine.config import ModelEndpoint


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)-15s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    # Quiet noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VLM evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on validation set with both judges
  python -m ares.evaluation --split val --use-molmo --use-glider
  
  # Force recompute Molmo scores only
  python -m ares.evaluation --use-molmo --force-molmo
  
  # Filter by specific tasks and models
  python -m ares.evaluation --router-tasks chart_qa,doc_qa --models qwen,llava
        """
    )
    
    # Database
    parser.add_argument(
        '--db-url',
        default=os.environ.get('DATABASE_URL', 'postgresql+psycopg2://localhost/vlm_router'),
        help='PostgreSQL connection URL (default: $DATABASE_URL or localhost)'
    )
    
    # Filtering
    parser.add_argument('--split', choices=['train', 'val', 'test'], 
                        help='Filter by data split')
    parser.add_argument('--router-tasks', type=str,
                        help='Comma-separated router tasks to filter')
    parser.add_argument('--models', type=str,
                        help='Comma-separated model names to filter')
    
    # Judge selection
    parser.add_argument('--use-molmo', action='store_true', default=True,
                        help='Use Molmo VLM judge (default: True)')
    parser.add_argument('--no-molmo', action='store_true',
                        help='Disable Molmo judge')
    parser.add_argument('--use-glider', action='store_true', default=True,
                        help='Use Glider text evaluator (default: True)')
    parser.add_argument('--no-glider', action='store_true',
                        help='Disable Glider evaluator')
    
    # Force flags
    parser.add_argument('--force', action='store_true',
                        help='Force recompute all metrics')
    parser.add_argument('--force-static', action='store_true',
                        help='Force recompute static metrics only')
    parser.add_argument('--force-confidence', action='store_true',
                        help='Force recompute confidence scores only')
    parser.add_argument('--force-molmo', action='store_true',
                        help='Force recompute Molmo scores only')
    parser.add_argument('--force-glider', action='store_true',
                        help='Force recompute Glider scores only')
    
    # Endpoint configuration
    parser.add_argument('--glider-urls', type=str,
                        default='http://localhost:8805/v1,http://localhost:8807/v1',
                        help='Comma-separated Glider endpoint URLs')
    parser.add_argument('--vlm-judge-urls', type=str,
                        default='http://localhost:8806/v1,http://localhost:8808/v1',
                        help='Comma-separated VLM Judge endpoint URLs')
    
    # Processing
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Samples per batch (default: 50)')
    parser.add_argument('--max-parallel-configs', type=int, default=2,
                        help='Source configs to process in parallel (default: 2)')
    parser.add_argument('--max-workers', type=int, default=64,
                        help='Max concurrent API requests (default: 64)')
    parser.add_argument('--timeout', type=int, default=180,
                        help='API request timeout in seconds (default: 180)')
    
    # Other
    parser.add_argument('--tracker-path', type=str, default='eval_progress.json',
                        help='Path to progress tracker file')
    parser.add_argument('--reset-progress', action='store_true',
                        help='Reset progress tracker before running')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging')
    
    return parser.parse_args()


def build_endpoints(urls: str, name_prefix: str, model_id: str) -> list:
    """Build ModelEndpoint list from comma-separated URLs."""
    endpoints = []
    for i, url in enumerate(urls.split(',')):
        url = url.strip()
        if url:
            endpoints.append(ModelEndpoint(
                name=f"{name_prefix}-{i+1}",
                model_id=model_id,
                base_url=url,
                api_key="EMPTY",
                pricing={},
                extra_params={},
            ))
    return endpoints


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("CLI")
    
    # Handle flag conflicts
    use_vlm_judge = args.use_vlm_judge and not args.no_molmo
    use_glider = args.use_glider and not args.no_glider
    force = args.force
    
    logger.info("Initializing evaluation pipeline...")
    
    # Build endpoints
    glider_endpoints = build_endpoints(
        args.glider_urls, "glider", "PatronusAI/glider"
    ) if use_glider else []
    
    vlm_judge_endpoints = build_endpoints(
        args.vlm_judge_urls, "vlm-judge", "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8"
    ) if use_vlm_judge else []
    
    all_endpoints = glider_endpoints + vlm_judge_endpoints
    
    if not all_endpoints:
        logger.error("No endpoints configured. Use --use-molmo or --use-glider")
        sys.exit(1)
    
    # Create runner
    runner = OpenAIStyleRunner(
        models=all_endpoints,
        request_timeout_s=args.timeout,
        max_workers=args.max_workers,
    )
    
    # Create DB engine
    engine = create_engine(args.db_url)
    
    # Create pipeline
    pipeline = RouterEvalPipeline(
        engine=engine,
        runner=runner,
        glider_model_names=[e.name for e in glider_endpoints],
        vlm_judge_model_names=[e.name for e in vlm_judge_endpoints],
        tracker_path=args.tracker_path,
        use_glider=use_glider,
        use_vlm_judge=use_vlm_judge,
    )
    
    # Reset progress if requested
    if args.reset_progress:
        pipeline.reset_progress()
    
    # Parse filters
    router_tasks = args.router_tasks.split(',') if args.router_tasks else None
    models = args.models.split(',') if args.models else None
    
    # Log config
    logger.info(f"DB: {args.db_url}")
    logger.info(f"Split: {args.split or 'ALL'}")
    logger.info(f"Router tasks: {router_tasks or 'ALL'}")
    logger.info(f"Models: {models or 'ALL'}")
    logger.info(f"Glider endpoints: {len(glider_endpoints)}")
    logger.info(f"VLM Judge endpoints: {len(vlm_judge_endpoints)}")
    logger.info(f"Force: {force}")
    
    # Run pipeline
    pipeline.evaluate_all(
        batch_size=args.batch_size,
        force=force,
        max_parallel_configs=args.max_parallel_configs,
        split=args.split,
    )
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
