import json
from pathlib import Path

nb_path = Path("/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Which_VLM_Router/artemis_final/notebooks/load_balancer/02_queues_and_latency_under_load.ipynb")

with open(nb_path, 'r') as f:
    nb = json.load(f)

# The corrected code for setup_environment
new_code = """
def setup_environment(replicas=1):
    stats = StatsRegistry()
    task = "vqa"
    
    model_stats = {
        "fast_model": (100.0, 0.9, 0.002),
        "medium_model": (300.0, 0.92, 0.005),
        "slow_model": (600.0, 0.95, 0.01),
    }

    for m, (lat, acc, cost) in model_stats.items():
        stats.update_latency(task, m, lat)
        stats.update_accuracy(task, m, acc)
        stats.update_cost(task, m, cost)

    configs = {}
    for m, (lat, acc, cost) in model_stats.items():
        configs[m] = ModelCapacityConfig(
            model_name=m,
            base_latency_ms=lat,
            min_replicas=replicas,
            max_replicas=replicas,
            max_qps_per_replica=20.0,
            cost_per_request_usd=cost
        )
    
    # Loose SLA so we can observe queues building up without just rejecting
    sla = {"default": 5000.0} 
    
    lb = ArtemisLoadBalancer(
        model_configs=configs,
        stats_registry=stats,
        latency_sla_ms=sla,
        scheduling_mode="capacity_aware",
        simulation_only=True
    )
    
    return lb

print("Environment setup function defined.")
"""

# Find the cell
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def setup_environment(replicas=1):" in source and "ModelCapacityConfig(max_rps=" in source:
            # Replace the source
            # Split into lines and keep newlines
            new_lines = [line + "\n" for line in new_code.strip().split("\n")]
            # Remove last newline to be clean? Notebooks usually keep them.
            # Actually, split keeps them if I don't use splitlines.
            
            # Better way to match format:
            lines = new_code.strip().split('\n')
            final_source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
            
            cell['source'] = final_source
            found = True
            break

if found:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully.")
else:
    print("Could not find the target cell to patch.")
