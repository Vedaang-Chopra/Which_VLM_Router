
import json
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fix_01(nb_path):
    path = Path(nb_path)
    if not path.exists(): return
    
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Fix SchedulingContext arrival_time
                if 'arrival_time=time.time()' in line:
                    line = line.replace('arrival_time=time.time()', 'arrival_ts_ms=time.time() * 1000')
                    modified = True
                
                # Fix RouterOutput missing args (specific to 01 if needed, but 01 uses manual args in the user snippet)
                # User's error trace for 01 showed RouterOutput was created successfully in cell 5, but let's check cell 5 in 01.
                # In 01 view:
                # router_output = RouterOutput(..., sample_id="...", task_type="...")
                # It HAS them in cell 5 lines 209, 210.
                # So 01 purely needs SchedulingContext fix.
                
                new_source.append(line)
            cell['source'] = new_source

    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        logging.info(f"Fixed 01: {path.name}")
    else:
        logging.info(f"01: No changes needed for {path.name}")

def fix_router_output_args(nb_path):
    # Applies to 02, 03, 04
    path = Path(nb_path)
    if not path.exists(): return
    
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    modified = False
    
    # We want to insert sample_id="sim_req", task_type="vqa" into RouterOutput calls
    # Strategy: Find 'max_prob=...' and append comma + new args replacing the newline
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            new_lines = []
            for i, line in enumerate(lines):
                # Detect instantiation ending of RouterOutput or max_prob line
                # Notebook 02: max_prob=0.6\n
                # Notebook 03: max_prob=0.8\n
                
                if 'max_prob=' in line and 'RouterOutput' not in line: 
                    # check if next line is closing paren or if this line is closing paren
                    # e.g. "        max_prob=0.6\n"
                    # We want to change it to "        max_prob=0.6,\n        sample_id='sim_req', task_type='vqa'\n"
                    
                    # Be careful not to double add
                    if 'sample_id=' not in line and 'sample_id=' not in "".join(lines[i:i+5]):
                        # remove \n, add comma, args, restore \n
                        stripped = line.rstrip()
                        if not stripped.endswith(','):
                            stripped += ','
                        
                        # Use same indentation
                        indent = line[:len(line) - len(line.lstrip())]
                        
                        # Handle simple replacement
                        # if using f-string in sample_id, it might be tricky.
                        # For 02/03/04 simulation loops, we often want 'sample_id' to be dynamic?
                        # In 02: 
                        # router_out = RouterOutput(..., max_prob=0.6)
                        # inside run_simulation function. But correct sample_id is only available inside loop?
                        # RouterOutput is created BEFORE the loop in 02!
                        # So we can just give it a generic ID for the router output object itself.
                        # The SchedulingDecision uses 'sample_id' from SchedulingContext.
                        # RouterOutput.sample_id implies the sample the router saw.
                        
                        updated_line = stripped + f"\n{indent}sample_id='sim_req', task_type='vqa',\n"
                        new_lines.append(updated_line)
                        modified = True
                        continue
                
                new_lines.append(line)
            cell['source'] = new_lines

    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        logging.info(f"Fixed RouterOutput in {path.name}")
    else:
        logging.info(f"No RouterOutput changes for {path.name}")

def main():
    base = Path("artemis_final/notebooks/load_balancer")
    
    fix_01(base / "01_single_request_walkthrough.ipynb")
    
    files = [
        "02_queues_and_latency_under_load.ipynb",
        "03_modes_comparison.ipynb",
        "04_cost_budget_and_sla.ipynb"
    ]
    for fname in files:
        fix_router_output_args(base / fname)

if __name__ == "__main__":
    main()
