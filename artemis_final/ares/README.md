# ARES: Automated Response Evaluation System

**The Data Engine for Artemis**

ARES (Automated Response Evaluation System) is the data processing and evaluation engine for the Artemis VLM Router. It handles dataset management, parallel inference, and rigorous evaluation of VLM outputs.

## 🚀 Quick Start

### Python API

The `ares_api` module provides a central access point for database and evaluation operations.

```python
from ares.ares_api import get_db, evaluate_sample, SampleRecord

# 1. Access Database
engine = get_db()
print(f"Connected to DB: {engine.url}")

# 2. Evaluate a Sample (Conceptual)
# (Assuming you have a SampleRecord object)
# score = evaluate_sample(sample_record)
```

### Setup

```bash
cd artemis_final/ares
pip install -r requirements.txt
```

---

## 📂 Directory Layout

The `ares` module is organized into specialized submodules:

- **`ares_api.py`**: **Main Entry Point**. Facade for common operations.
- **`db/`**: Database operations and schema.
  - `schema.sql`: PostgreSQL schema definition.
  - `operations.py`: CRUD operations for samples/responses/evaluations.
- **`evaluation/`**: Scoring and Judging logic.
  - `evaluation.py`: Main evaluation pipeline.
  - `judge_molmo.py`: VLM-as-a-judge implementation.
- **`parallel/`**: High-performance parallel inference tools.
- **`configs/`**: Configuration definitions (`ExperimentConfig`, `SampleRecord`).

---

## 🏗️ Core Architecture

ARES consists of three main components:

### 1. Database (PostgreSQL)
A standardized schema for storing VLM inputs, outputs, and scores.
- **`vlm_samples`**: The questions and ground truth.
- **`vlm_responses`**: What the models answered.
- **`vlm_evaluations`**: How good the answers were.

See [db/README.md](db/README.md) for full schema details.

### 2. Parallel Inference
Tools to run inference across thousands of samples efficiently, managing:
- Rate limiting
- Failed request retries
- Batch processing

### 3. Evaluation Pipeline
A multi-stage evaluation system:
1. **Rule-Based**: Exact match, F1 score, numeric match.
2. **Model-Based**: LLM-as-a-judge (Glider, Molmo) for complex reasoning tasks.

---

## 📊 Data Flow

1. **Ingest**: Datasets are loaded from "Cauldron" format into `vlm_samples`.
2. **Inference**: `parallel` module queries VLMs and saves to `vlm_responses`.
3. **Evaluate**: `evaluation` module scores responses and saves to `vlm_evaluations`.
4. **Train**: Router trains on this rich dataset of (Question, Image) -> (Best Model Reward).

---

## 🔧 Configuration

ARES uses `configs/config.py` to define:
- **Task Mappings**: Mapping datasets (e.g., `docvqa`) to router tasks (`document_ocr`).
- **Data Types**: Defining if a task needs `exact` match or `freeform` evaluation.

Example Task Mapping:
```python
CONFIG_TO_TASK = {
    'docvqa': 'document_ocr',
    'ai2d': 'diagram_reasoning',
    # ...
}
```

---

## ✅ Usage Guide

### Running Database Migrations
To set up the database, run the SQL scripts in `db/`:
```bash
psql -d vlm_router_db -f db/schema.sql
```

### Running Evaluation
(Typically run via notebooks or pipeline scripts)

See `notebooks/` for end-to-end examples of data processing.

