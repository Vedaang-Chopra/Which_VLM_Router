#!/usr/bin/env python3
"""
Artemis VLM Router - CLI Entrypoint

This script provides a command-line interface to the system.
It utilizes the centralized system_api to initialize services and process requests.

Usage:
    python main.py --prompt "Describe this image" --image "data/sample.jpg" --mode balanced
"""
import argparse
import sys
import logging
import json
import time
from pathlib import Path

# Ensure repo root is in path
sys.path.append(str(Path(__file__).parent.absolute()))

from system_api.pipeline import init_system, handle_chat_completion
from system_api.schemas import ChatCompletionRequest, ChatMessage

# Configure Logging for CLI
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger("artemis_cli")

def parse_args():
    parser = argparse.ArgumentParser(description="Artemis VLM Router CLI")
    
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the VLM.")
    parser.add_argument("--image", type=str, required=False, help="Path to the image file (optional).")
    parser.add_argument("--mode", type=str, default="balanced", 
                        choices=["accuracy", "cheap", "fast", "balanced"],
                        help="Routing mode preference.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--dry-run", action="store_true", help="Initialize only, do not run inference.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    print(">>> Initializing Artemis System...")
    try:
        services = init_system()
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        sys.exit(1)
        
    if args.dry_run:
        print("✅ System initialized successfully (Dry Run). Exiting.")
        sys.exit(0)

    # Construct Request
    messages = [
        ChatMessage(role="user", content=args.prompt)
    ]
    # Note: Currently the system_api expects image handling logic in the request or prompt construction.
    # The existing handle_chat_completion extracts text from messages. 
    # For now, we follow the pattern of the API. Integrating image passing might need a schema update
    # or just passing it via the inference call if the router supports multimodal inputs in the prompt.
    # However, the current handle_chat_completion doesn't explicitly parse image fields from ChatCompletionRequest
    # (it has a TODO). We will just proceed with text-based prompting or assume the user puts the image path in prompt?
    # No, the requirement says "demonstrate full flow".
    # Let's check handle_chat_completion again. It has `image_path = None` hardcoded?
    # Ah, I should fix that in the request construction here or acknowledge the limitation.
    # Re-reading `pipeline.py`: line 81 `image_path = None # Logic to extract image not implemented yet`.
    # Wait, that was in the OLD main.py? 
    # Let's check `system_api/pipeline.py` which I read earlier.
    # Yes, line 137 in `pipeline.py` calls `inf_svc.call_model(..., image_path=image_path, ...)` but `image_path` variable is not extracted from request!
    # I should probably pass it if I can.
    # Since I cannot edit `system_api/pipeline.py` (it wasn't in the plan to fix it, just use it),
    # I will modify THIS script to manually inject the image path if the pipeline allows, 
    # OR I will just note that image path support is limited.
    # WAIT! Usage of `inference_service` supports `image_path`.
    # `handle_chat_completion` inside `pipeline.py` does NOT seem to extract it. 
    # This is a gap in the existing code.
    # However, I can't rewrite the whole `system_api` right now without scope creep.
    # But wait! I can just import `InferenceService` and call it directly if I want to show it working?
    # NO, the goal is to use the "pipeline".
    #
    # Actually, looking at `system_api/pipeline.py` again:
    # It does NOT properly parse the image from the request.
    # BUT, I can pass a special marker in the content? Or I can subclass/modify the internal logic?
    #
    # BETTER APPROACH:
    # Since I am rewriting `main.py` which calls `handle_chat_completion`, and `handle_chat_completion` is defined in `system_api/pipeline.py`...
    # I should check if I can modify `handle_chat_completion` to accept image path?
    # I didn't verify `system_api/pipeline.py` for modification in the plan, but it's crucial for "End-to-End Image Inference".
    # If `pipeline.py` is broken for images, I should fix it.
    # I will check `pipeline.py` content again.
    # Line 86-94 extracts prompt.
    # It does NOT extract image.
    # This is a bug/missing feature.
    # I will stick to the plan: "Reuse as much as possible". 
    # BUT "demonstrate FULL end-to-end flow... run image inference".
    # If the pipeline doesn't support it, I should fix the pipeline OR
    # implement the logic manually in `main.py` using the individual services (like the notebook).
    #
    # The user asked for "ONE main Python script... that initializes... exposes ONE high-level function".
    # So I can implement `run_single_request` in `main.py` that does the logic correctly (mimicking the notebook),
    # instead of relying on the broken `handle_chat_completion`.
    # This aligns with the requirement "Uses the same steps as the notebook".
    # So I will NOT use `handle_chat_completion` blindly. I will write `run_single_request` that does the steps.
    
    print(f"Processing Request: Prompt='{args.prompt}', Image='{args.image}', Mode='{args.mode}'")
    
    # Run the custom single request logic (fixing the image path gap)
    result = run_single_request(services, args.image, args.prompt, args.mode)
    
    # ---------------------------------------------------------
    # Pretty Print Output
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print(f"🚀 FINAL RESPONSE ({result['chosen_model']})")
    print("="*40)
    print(result['response'])
    print("-" * 40)
    print(f"Metadata:")
    print(f" - Latency: {result.get('latency', 'N/A')}")
    print(f" - Router Mode: {args.mode}")
    print("="*40 + "\n")

def run_single_request(services, image_path, prompt, mode):
    """
    Orchestrate the request: Router -> Load Balancer -> Inference
    This mimics system_api.pipeline.handle_chat_completion but adds proper image support.
    """
    router = services['router']
    lb = services['lb']
    inference = services['inference']
    collector = services['collector']
    
    request_id = str(uuid.uuid4())
    
    # 1. Router
    try:
        router_result = router.predict(prompt, mode=mode)
        chosen_model = router_result['chosen_model']
        router_probs = router_result.get('rewards', {})
    except Exception as e:
        logger.error(f"Router error: {e}")
        chosen_model = "qwen2_5_vl_7b" # Fallback
        router_probs = {}
        
    # 2. Load Balancer
    lb_decision = lb.schedule(
        sample_id=request_id,
        task_type="vlm",
        router_probs=router_probs,
        preferred_model=chosen_model
    )
    final_model = lb_decision['chosen_model']
    
    # 3. Inference
    start_time = time.time()
    try:
        # Note: If image_path is None, the system treats it as text-only LLM usually.
        inf_result = inference.call_model(
            model_name=final_model,
            prompt=prompt,
            image_path=image_path, # Passed correctly here!
            temperature=0.7,
            max_tokens=512
        )
        content = inf_result.get('text') or inf_result.get('content') or ""
    except Exception as e:
        logger.error(f"Inference error: {e}")
        content = f"Error during inference: {e}"
        inf_result = {"error": str(e)}
        
    latency_ms = (time.time() - start_time) * 1000
    
    # 4. Collection (Optional but good)
    # Using collector.log_model_response logic partially here if needed, 
    # but for CLI we might skip detailed logging to keep it simple or use the background task approach.
    # To be safe and simple, we skip async logging in CLI for now to avoid hanging.
    
    return {
        "chosen_model": final_model,
        "response": content,
        "latency": f"{latency_ms:.2f} ms",
        "metadata": lb_decision
    }

if __name__ == "__main__":
    main()
