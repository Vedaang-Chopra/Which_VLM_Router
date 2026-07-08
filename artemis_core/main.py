#!/usr/bin/env python3
"""
Artemis VLM Router - CLI Entrypoint.
"""
import argparse
import logging
import sys
import uuid
import time
from pathlib import Path
from PIL import Image

# Add src to path if running directly
sys.path.append(str(Path(__file__).parent / "src"))

from artemis.common.config_loader import load_config
from artemis.common.utils import set_seed, setup_logging
from artemis.router import RewardRouter
from artemis.load_balancer import LoadBalancer
from artemis.load_balancer.types import RouterOutput, SchedulingContext, ModelCapacityConfig
from artemis.inference import VLMClient, load_endpoints_from_config

logger = logging.getLogger("artemis")

def main():
    parser = argparse.ArgumentParser(description="Artemis VLM Router CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=str, required=False, help="Path to image file")
    parser.add_argument("--mode", type=str, default="balanced", choices=["accuracy", "fast", "cheap", "balanced"])
    parser.add_argument("--config", type=str, help="Path to artemis.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    args = parser.parse_args()

    # 1. Setup Environment
    set_seed(args.seed)
    setup_logging(logging.INFO, args.log_file)
    
    try:
        # 2. Load Configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # 3. Initialize Components
        logger.info("Initializing Router...")
        router_ckpt = str(Path(config.router.checkpoint_path).expanduser())
        
        if not Path(router_ckpt).exists():
            # Strict error handling: fail if model is missing
            raise FileNotFoundError(f"Router checkpoint not found at: {router_ckpt}")
        
        router = RewardRouter(router_ckpt, device=config.router.device)

        logger.info("Initializing Load Balancer...")
        # Map config to LB strict types
        lb_model_configs = {}
        for m in config.models:
             lb_model_configs[m['name']] = ModelCapacityConfig(
                 min_replicas=m.get('min_replicas', 1),
                 max_replicas=m.get('max_replicas', 5),
                 sla_ms=m.get('sla_ms', 2000.0)
             )
             
        lb = LoadBalancer(
            model_configs=lb_model_configs,
            mode=args.mode
        )

        logger.info("Initializing Inference Engine...")
        endpoints = load_endpoints_from_config(config.models)
        client = VLMClient(endpoints)

        # 4. Execute Pipeline
        logger.info(f"Processing Request: '{args.prompt}'")
        
        # Step A: Router
        pil_img = None
        if args.image:
            img_path = Path(args.image)
            if not img_path.exists():
                 raise FileNotFoundError(f"Image not found at: {img_path}")
            pil_img = Image.open(img_path)
            
        route_result = router.route(args.prompt, pil_img, mode=args.mode)
        logger.info(f"Router Preference: {route_result['chosen_model']} (Scores: {route_result.get('scores')})")
        
        # Step B: Load Balancer
        router_out = RouterOutput(
            sample_id=str(uuid.uuid4()),
            task_type="vlm",
            router_probs=route_result['scores'],
            preferred_model=route_result['chosen_model']
        )
        
        lb_ctx = SchedulingContext(
            arrival_ts_ms=time.time() * 1000,
            metadata={"mode": args.mode, "seed": args.seed}
        )
        
        decision = lb.schedule(router_out, lb_ctx)
        final_model = decision.chosen_model
        logger.info(f"Load Balancer Decision: {final_model} (Estimated Latency: {decision.total_latency_ms:.1f}ms)")
        
        # Step C: Inference
        logger.info(f"Calling Model: {final_model}...")
        result = client.generate(args.prompt, args.image, model=final_model)
        
        # Output
        print("\n" + "="*50)
        print(f"FINAL RESPONSE ({final_model})")
        print("="*50)
        print(result.get("response_text", "Error: No response"))
        print("-" * 50)
        print(f"Latency: {result.get('latency_ms', 0):.2f} ms")
        print("="*50 + "\n")

    except FileNotFoundError as e:
        logger.error(f"Configuration/File Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected Runtime Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
