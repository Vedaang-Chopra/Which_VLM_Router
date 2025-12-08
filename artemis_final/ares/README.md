# ARES (Automated Response Evaluation System)

**ARES** is the evaluation and data management subsystem of Artemis. It handles dataset loading, response storage, and automated scoring using judges (Glider, Semantic F1, etc.).

## Features
- **Database Management**: Schema definitions and connection handling (`ares.db`).
- **Evaluation**: Automated scoring metrics (Exact Match, F1, Glider) (`ares.evaluation`).
- **Data Ingestion**: Tools to import Cauldron datasets (`ares.data`).
- **Metrics**: Aggregation and reporting (`ares.metrics`).

## Usage

### Public API
All core functionality is exposed via `ares.public_api`:

```python
from ares import get_db, AresAPI, SampleRecord

# 1. Database Access
engine = get_db()

# 2. Evaluation
api = AresAPI()

# Create a sample record (or use dict)
sample = {
    "response_parsed": "The answer is 42.",
    "ground_truth": "42",
    "ground_truth_type": "exact"
}

# Run evaluation
scores = api.evaluate(sample)
print(scores['score_exact_match']) # 1.0
```

### Database Operations
```python
from ares import insert_samples, get_existing_responses

# Insert new samples
insert_samples(samples_list)

# Query responses
responses = get_existing_responses(run_id="exp_001")
```

## Directory Structure
```
ares/
├── public_api.py         # Main entry point
├── db/                   # Database schemas and operations
├── evaluation/           # Scoring logic (Scorer, Glider, SemanticF1)
├── configs/              # Configuration (Tasks, Prompts)
├── metrics/              # Aggregation utilities
└── README.md
```
