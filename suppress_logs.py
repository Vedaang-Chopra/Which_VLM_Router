import json

nb_path = 'artemis_final/notebooks/ares/01_parallel_inference_to_db.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Code to suppress specific log spam
suppress_code = """
# Suppress specific connection errors
import logging
class NoGemmaMetricsFilter(logging.Filter):
    def filter(self, record):
        return "gemma_3_27b" not in record.getMessage()

logging.getLogger("ares.metrics.metrics_client").addFilter(NoGemmaMetricsFilter())
"""

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "logging.basicConfig" in source:
            # Append suppression code after basicConfig
            cell['source'].append(suppress_code)
            updated = True
            print("Added log suppression.")
            break

if updated:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook saved.")
else:
    print("Could not find logging setup cell.")
