

def return_model_specs():
        # Define your model blocks and their prefixes exactly as in your columns
    MODEL_SPECS = [
        {"id": 0, "name": "deepseek_ocr",         "prefix": "deepseek_ocr__"},
        {"id": 1, "name": "qwen2_5_vl_3b",       "prefix": "qwen2_5_vl_3b__"},
        {"id": 2, "name": "qwen2_5_vl_7b",       "prefix": "qwen2_5_vl_7b__"},
        {"id": 3, "name": "qwen3_vl_8b_thinking","prefix": "qwen3_vl_8b_thinking__"},
        {"id": 4, "name": "gemma_3_27b",         "prefix": "gemma_3_27b__"},
    ]

    print("Configured models:")
    for m in MODEL_SPECS:
        print(f"  id={m['id']} name={m['name']} prefix={m['prefix']}")

    return MODEL_SPECS



def return_model_pricing():
    MODEL_PRICING = {
    "deepseek_ocr": {
        "prompt_per_1k": 0.00003,
        "completion_per_1k": 0.0001,
    },
    "qwen2_5_vl_3b": {
        "prompt_per_1k": 0.0001,
        "completion_per_1k": 0.0001,
    },
    "qwen2_5_vl_7b": {
        "prompt_per_1k": 0.0002,
        "completion_per_1k": 0.0002,
    },
    "qwen3_vl_8b_thinking": {
        "prompt_per_1k": 0.00018,
        "completion_per_1k": 0.0021,
    },
    "gemma_3_27b": {
        "prompt_per_1k": 0.00009,
        "completion_per_1k": 0.00016,
    },
    }
    # MODEL_PRICING = {
    # "deepseek_ocr": {"prompt_per_1k": 15, "completion_per_1k": 150},
    # "qwen2_5_vl_3b": {"prompt_per_1k": 10, "completion_per_1k": 100},
    # "qwen2_5_vl_7b": {"prompt_per_1k": 20, "completion_per_1k": 200},
    # "qwen3_vl_8b_thinking": {"prompt_per_1k": 50, "completion_per_1k": 500},
    # "gemma_3_27b":{"prompt_per_1k": 30, "completion_per_1k": 300},
    # }

    print("Configured model pricing (USD per 1K tokens):")
    for model_name, pricing in MODEL_PRICING.items():
        print(f"  {model_name}: prompt=${pricing['prompt_per_1k']}, "
              f"completion=${pricing['completion_per_1k']}")
    return MODEL_PRICING

