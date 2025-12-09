# Ares

Ares (Automated Response Evaluation System) handles dataset management, response storage, and automated evaluation for the Artemis project.

## Overview

The module provides infrastructure for:
1.  **Data Ingestion**: Loading and processing datasets (e.g., from Cauldron).
2.  **Evaluation**: scoring model outputs using various metrics (Exact Match, F1, Glider).
3.  **Persistence**: Storing experiment results and samples in the database.

## Architecture

- **Public API** (`public_api.py`): Entry point for evaluation and database interactions.
- **Database** (`db/`): Schema definitions and ORM (SQLModel) classes.
- **Evaluation** (`evaluation/`): Implementation of judges and scorers.

## Usage

### Database Access

Retrieve the database engine or session.

```python
from artemis_final.ares.public_api import get_db

engine = get_db()
```

### Evaluation

Use the `AresAPI` to score model responses.

```python
from artemis_final.ares.public_api import AresAPI

api = AresAPI()
sample = {
    "response_parsed": "Paris",
    "ground_truth": "Paris",
    "ground_truth_type": "exact"
}
scores = api.evaluate(sample)
# result: {'score_exact_match': 1.0, ...}
```
