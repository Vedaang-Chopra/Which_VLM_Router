# Artemis VLM Router - Complete Implementation Walkthrough

**A step-by-step guide from zero to production deployment**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Part 1: Environment Setup](#part-1-environment-setup)
4. [Part 2: Database Setup](#part-2-database-setup)
5. [Part 3: Data Collection & Evaluation](#part-3-data-collection--evaluation)
6. [Part 4: Router Training](#part-4-router-training)
7. [Part 5: Router Inference](#part-5-router-inference)
8. [Part 6: Load Balancer Setup](#part-6-load-balancer-setup)
9. [Part 7: End-to-End Integration](#part-7-end-to-end-integration)
10. [Part 8: Production Deployment](#part-8-production-deployment)
11. [Part 9: Monitoring & Feedback Loop](#part-9-monitoring--feedback-loop)

---

## System Overview

Artemis is a **VLM Router System** that intelligently routes vision-language queries to the best model based on:
- **Accuracy**: Maximize prediction quality
- **Cost**: Minimize API costs
- **Latency**: Minimize response time
- **Balance**: Multi-objective optimization

### High-Level Flow

```
User Query → Router → Load Balancer → VLM Model → Response
     ↓                                              ↓
  Logging ←──────── Feedback Loop ←─────────────────┘
     ↓
  Retraining
```

---

## Prerequisites

### Required Knowledge

- Python 3.9+
- SQL basics (PostgreSQL)
- Machine Learning fundamentals
- Docker basics (optional but recommended)

### Required Software

```bash
# Check Python version
python --version  # Should be 3.9+

# Check Docker (optional)
docker --version
docker-compose --version

# Check PostgreSQL (if not using Docker)
psql --version
```

### Hardware Requirements

- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU, GPU (NVIDIA/Apple Silicon)
- **Storage**: 50GB free space

---

## Part 1: Environment Setup

### Step 1.1: Clone Repository

```bash
# Navigate to your projects directory
cd ~/projects

# Clone the repository
git clone https://github.com/yourusername/Which_VLM_Router.git
cd Which_VLM_Router/artemis_final
```

### Step 1.2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Verify activation
which python  # Should point to venv/bin/python
```

### Step 1.3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install artemis_final as editable package
cd ..  # Go to Which_VLM_Router directory
pip install -e .
cd artemis_final
```

### Step 1.4: Verify Installation

```bash
# Test imports
python -c "from artemis_final.router.artemis_router import RewardRouterInference; print('✓ Router OK')"
python -c "from artemis_final.load_balancer import ArtemisLoadBalancer; print('✓ LB OK')"
python -c "from artemis_final.inference_engine import WhichVLMClient; print('✓ Inference OK')"

# If all print ✓, you're good to go!
```

**Troubleshooting**: If imports fail, ensure you ran `pip install -e .` from the `Which_VLM_Router` directory.

---

## Part 2: Database Setup

### Step 2.1: Start PostgreSQL with Docker

```bash
cd artemis_final

# Start PostgreSQL
docker-compose up -d postgres

# Wait for database to be ready (10-20 seconds)
sleep 15

# Check if it's running
docker ps | grep postgres
```

### Step 2.2: Run Migrations

```bash
# Run main schema
docker exec -i $(docker ps -q -f name=postgres) psql -U artemis -d artemis < ares/db/schema.sql

# Run migrations
docker exec -i $(docker ps -q -f name=postgres) psql -U artemis -d artemis < ares/db/migration_collected.sql
docker exec -i $(docker ps -q -f name=postgres) psql -U artemis -d artemis < ares/db/migration_add_eval_cols.sql
docker exec -i $(docker ps -q -f name=postgres) psql -U artemis -d artemis < ares/db/migration_add_confidence_cols.sql
docker exec -i $(docker ps -q -f name=postgres) psql -U artemis -d artemis < ares/db/migration_add_molmo_cols.sql

echo "✓ All migrations completed"
```

### Step 2.3: Verify Database

```bash
# Connect to database
docker exec -it $(docker ps -q -f name=postgres) psql -U artemis -d artemis

# In psql prompt, run:
\dt  # List all tables
# Should see: vlm_samples, vlm_images, vlm_responses, vlm_evaluations

\d vlm_samples  # Describe vlm_samples table

\q  # Quit psql
```

**Expected Output**:
```
List of relations
Schema | Name              | Type  | Owner
-------|-------------------|-------|-------
public | vlm_samples       | table | artemis
public | vlm_images        | table | artemis
public | vlm_responses     | table | artemis
public | vlm_evaluations   | table | artemis
```

### Step 2.4: Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export DATABASE_URL="postgresql+psycopg2://artemis:artemis@localhost:5432/artemis"

# Reload shell config
source ~/.bashrc  # or source ~/.zshrc

# Verify
echo $DATABASE_URL
```

---

## Part 3: Data Collection & Evaluation

### Step 3.1: Understanding the Data Pipeline

```
1. Load Datasets    → vlm_samples table (prompts, ground truth)
2. Run VLM Inference → vlm_responses table (model outputs)
3. Evaluate Responses → vlm_evaluations table (scores)
4. Compute Rewards   → Training data for router
```

### Step 3.2: Load Sample Datasets

**Option A: Use Existing Data** (if available)

```bash
# Check if data already exists
docker exec -it $(docker ps -q -f name=postgres) psql -U artemis -d artemis -c "SELECT COUNT(*) FROM vlm_samples;"

# If count > 0, you have data! Skip to Step 3.4
```

**Option B: Load New Datasets**

```bash
# Launch Jupyter
jupyter notebook

# Open notebook:
# ares/notebooks/00_pushing_individual_dataset.ipynb

# Follow the notebook to load datasets from Hugging Face or local parquet files
```

**Key Steps in Notebook**:
1. Configure dataset paths
2. Load parquet files
3. Extract images, prompts, ground truth
4. Insert into `vlm_samples` and `vlm_images` tables
5. Verify row counts

### Step 3.3: Run VLM Inference (Optional - Data Already Collected)

**Note**: This step requires running VLM models locally or having API access. For this walkthrough, we assume you have pre-collected data.

```bash
# Open notebook:
# ares/notebooks/01_parallel_inference_to_db.ipynb

# This notebook:
# 1. Loads samples from vlm_samples
# 2. Calls each of 5 VLM models
# 3. Stores responses in vlm_responses
# 4. Tracks GPU metrics, latency, cost
```

**If you don't have VLM access**: Skip to Step 3.4 and use the pre-populated database.

### Step 3.4: Compute Evaluation Scores

```bash
# Open notebook:
# ares/notebooks/03_cost_utility_computation.ipynb

# This notebook computes:
# - sample_score: Combined accuracy metric
# - perf_hier: Hierarchical performance (best > better > good > ok)
# - cost_norm: Normalized cost
# - utility: Final utility score
```

**Key Outputs**:
- Updates `vlm_responses` table with computed scores
- Creates reward values for router training
- Generates visualizations of model performance

### Step 3.5: Explore the Data (Recommended)

```bash
# Open notebooks for EDA:
# 1. ares/notebooks/03_eda_samples_responses.ipynb
#    - Dataset overview
#    - Model performance comparison
#    - GPU metrics analysis

# 2. ares/notebooks/04_eda_evaluations.ipynb
#    - Evaluation score distributions
#    - Task × Model performance heatmaps
#    - Correlation analysis
```

**What to Look For**:
- Do models show clear specialization? (e.g., DeepSeek OCR better for text extraction)
- Are there tasks where all models struggle?
- Is there cost/latency/accuracy trade-off?

---

## Part 4: Router Training

### Step 4.1: Prepare Local Database Cache

**Why**: Training iterates over data many times. A local SQLite cache is 10-100x faster than PostgreSQL.

```bash
# Open notebook:
# router_train/notebooks/00_prepare_local_database.ipynb

# This notebook:
# 1. Queries PostgreSQL for all training data
# 2. Saves to local SQLite: router_train/data/vlm_router_cache.db
# 3. Generates metadata: router_train/data/vlm_router_cache_metadata.json
```

**Expected Output**:
```
✓ Loaded 43,900 samples
✓ 5 models: deepseek_ocr, qwen2_5_vl_3b, qwen2_5_vl_7b, qwen3_vl_8b_thinking, gemma_3_27b
✓ 4 modes: accuracy, cheap, fast, balanced
✓ Saved to vlm_router_cache.db (size: 234 MB)
```

### Step 4.2: Train Reward Router (⭐ RECOMMENDED)

```bash
# Open notebook:
# router_train/notebooks/02_reward_router_sql_to_training.ipynb

# This is the MAIN training notebook
```

**Training Workflow**:

1. **Load Data** from SQLite cache
   ```python
   df = load_from_sqlite('data/vlm_router_cache.db')
   # Shape: (219,500, 50) - 43,900 samples × 5 models
   ```

2. **Compute Rewards** for each (sample, model, mode) tuple
   ```python
   # Accuracy mode: reward = accuracy^2 * hierarchical_score
   # Cheap mode: reward = accuracy * hierarchical_score - w * cost^e
   # Fast mode: reward = accuracy * hierarchical_score - w * latency^e
   # Balanced mode: Multi-objective with Pareto weighting
   ```

3. **Create Dataset**
   ```python
   # Input: Text (prompt + metadata)
   # Target: Reward (scalar per model)
   # Total examples: 219,500 × 4 modes = 878,000
   ```

4. **Train Model**
   ```python
   # Architecture:
   # DistilBERT → 768-dim embedding
   #   + Model Embedding (32-dim)
   #   + Mode Embedding (16-dim)
   #   → MLP (512-dim hidden, 2 layers)
   #   → Reward prediction (1 scalar)

   # Loss: MSE (predicted_reward, actual_reward)
   # Optimizer: AdamW, lr=2e-5
   # Epochs: 5
   ```

5. **Evaluate**
   ```python
   # Metrics:
   # - Pearson correlation (predicted vs actual reward)
   # - Routing accuracy (does argmax(predicted) match best model?)
   # - Oracle comparison (how much better is oracle?)
   ```

**Expected Results**:
```
Epoch 1/5: Loss=0.0245, Val Pearson=0.1876
Epoch 2/5: Loss=0.0198, Val Pearson=0.2103
Epoch 3/5: Loss=0.0176, Val Pearson=0.2251
Epoch 4/5: Loss=0.0165, Val Pearson=0.2348
Epoch 5/5: Loss=0.0159, Val Pearson=0.2393

✓ Best model saved: checkpoints/best_reward_router.pt
```

**Interpretation**:
- **Val Pearson = 0.24**: Moderate positive correlation (good for routing)
- **Routing Accuracy ~35%**: Router picks best model 35% of the time
- **Oracle Accuracy ~62%**: Best possible model picks correct 62% of the time
- **Random Baseline ~20%**: Random selection gets 20% (1/5 models)

### Step 4.3: (Optional) Train Alternative Routers

**Pairwise Ranking Router**:
```bash
# Open: router_train/notebooks/03_pairwise_ranking_router.ipynb
# Approach: Learn which model is better for each sample (pairwise preference)
# Loss: Margin ranking loss
# Pro: Simpler objective
# Con: Doesn't capture magnitude of differences
```

**Classical CE/KL Router**:
```bash
# Open: router_train/notebooks/04_classical_ce_kl_router.ipynb
# Approach: Softmax classification over models
# Loss: Cross-entropy + KL divergence for soft labels
# Pro: Standard classification setup
# Con: Assumes one "correct" model per sample
```

**Recommendation**: Stick with Reward Router for flexibility.

### Step 4.4: Verify Checkpoint

```bash
# Check checkpoint exists
ls -lh checkpoints/best_reward_router.pt
# Should show ~180 MB file

# Inspect checkpoint
python -c "
import torch
ckpt = torch.load('checkpoints/best_reward_router.pt', map_location='cpu')
print('Keys:', ckpt.keys())
print('Model params:', sum(p.numel() for p in ckpt['model_state_dict'].values()) / 1e6, 'M')
"

# Expected output:
# Keys: dict_keys(['model_state_dict', 'config', 'training_metadata'])
# Model params: 66.8 M
```

---

## Part 5: Router Inference

### Step 5.1: Understand Router Architecture

```
Input: "What is shown in this diagram?" + metadata
  ↓
1. Format text:
   "[ROUTER] Task: diagram_reasoning. Dataset: ai2d. Question: What is shown in this diagram?"
  ↓
2. Tokenize and encode with DistilBERT
   → 768-dim text embedding
  ↓
3. Add learned embeddings:
   + Model embedding for each of 5 models (32-dim each)
   + Mode embedding for chosen mode (16-dim)
  ↓
4. MLP forward pass (2 layers, 512-dim hidden)
  ↓
5. Output: 5 reward predictions (one per model)
   [0.23, 0.68, 0.85, 0.76, 0.92]
  ↓
6. argmax → Choose model with highest reward
   → "gemma_3_27b" (reward=0.92)
```

### Step 5.2: Test Router Inference (Python)

```python
from artemis_final.router.artemis_router import RewardRouterInference

# Initialize router
router = RewardRouterInference(
    checkpoint_path='checkpoints/best_reward_router.pt',
    device='cpu'  # or 'cuda' or 'mps'
)

# Single routing decision
result = router.route(
    prompt="What is the capital of France?",
    mode="balanced",
    metadata={'router_task': 'qa', 'source_dataset': 'test'}
)

print(f"Chosen model: {result['chosen_model']}")
print(f"Rewards: {result['rewards']}")
print(f"Inference time: {result['inference_ms']:.2f}ms")
```

**Expected Output**:
```
Chosen model: qwen2_5_vl_7b
Rewards: {
    'deepseek_ocr': 0.234,
    'qwen2_5_vl_3b': 0.679,
    'qwen2_5_vl_7b': 0.851,
    'qwen3_vl_8b_thinking': 0.763,
    'gemma_3_27b': 0.825
}
Inference time: 15.23ms
```

### Step 5.3: Test All Routing Modes

```python
# Test all 4 modes
modes = ['accuracy', 'cheap', 'fast', 'balanced']
prompt = "Describe this chart showing GDP growth."

for mode in modes:
    result = router.route(
        prompt=prompt,
        mode=mode,
        metadata={'router_task': 'chartqa', 'source_dataset': 'test'}
    )
    print(f"{mode:12} → {result['chosen_model']}")
```

**Expected Behavior**:
```
accuracy     → gemma_3_27b              (largest model)
cheap        → qwen2_5_vl_3b            (smallest model)
fast         → qwen2_5_vl_3b            (fastest model)
balanced     → qwen2_5_vl_7b            (good middle ground)
```

### Step 5.4: Run Router Unit Tests

```bash
# Open notebook:
# router/notebooks/02_router_unit_tests.ipynb

# Tests:
# 1. ✓ Basic routing works
# 2. ✓ All 4 modes work
# 3. ✓ Batch routing works
# 4. ✓ Invalid mode raises error
# 5. ✓ Deterministic (same input → same output)
# 6. ✓ Stats tracking works

# Expected: 6/6 tests pass
```

### Step 5.5: Compare Router Architectures

```bash
# Open notebook:
# router/notebooks/01_understanding_router_architectures.ipynb

# This notebook loads all 3 router checkpoints and compares:
# - Reward Router (text-only, reward prediction)
# - Pairwise Router (text-only, pairwise ranking)
# - Classical Router (text-only, classification)

# Insights:
# - Routers make VERY different decisions (0% agreement!)
# - Reward router is most flexible (supports 4 modes)
# - Pairwise router is simplest
# - Classical router has highest accuracy on validation
```

### Step 5.6: Run Real-World Experiments

```bash
# Open notebook:
# router/notebooks/03_experiments_and_load_testing.ipynb

# This notebook:
# 1. Loads real samples from database (with images)
# 2. Runs router on each sample
# 3. Analyzes routing decisions by task type
# 4. Visualizes sample-specific routing

# Key finding: Router learns task specialization
# - OCR tasks → DeepSeek OCR
# - Math tasks → Qwen3 Thinking
# - Diagram tasks → Gemma 27B
```

---

## Part 6: Load Balancer Setup

### Step 6.1: Understand Load Balancer Purpose

**Why do we need a load balancer?**

Router gives us "Which model is best?" but doesn't consider:
- Is that model currently overloaded?
- Will using that model violate our SLA (latency budget)?
- Is there a cheaper model that's "good enough" and available?

**Load Balancer solves this** by:
1. Tracking model capacity (queue length, active requests)
2. Predicting expected latency (queue time + inference time)
3. Choosing best model that meets SLA

### Step 6.2: Configure Load Balancer

**Edit**: `load_balancer/load_balancer_config.yaml`

```yaml
global:
  latency_sla_ms: 2000  # Target: respond within 2 seconds
  max_accuracy_drop: 0.05  # Allow max 5% accuracy drop
  default_scheduling_mode: capacity_aware

models:
  deepseek_ocr:
    base_latency_ms: 300  # Average inference time
    min_replicas: 1
    max_replicas: 5  # Can scale up to 5 instances
    sla_ms: 1500  # Per-model SLA
    max_qps_per_replica: 3.0  # Throughput limit
    cost_per_request_usd: 0.0001
    autoscale:
      enable: true
      scale_up_latency_factor: 0.8  # Scale up if latency > 0.8 * sla_ms
      scale_down_util_threshold: 0.3  # Scale down if utilization < 30%

  # ... (configure all 5 models)
```

**Key Parameters**:
- `base_latency_ms`: Get from ARES evaluation results
- `max_qps_per_replica`: Formula: `1000 / base_latency_ms`
- `cost_per_request_usd`: From API pricing or compute costs

### Step 6.3: Test Load Balancer (Python)

```python
from artemis_final.load_balancer import ArtemisLoadBalancer

# Initialize
lb = ArtemisLoadBalancer(
    config_path='load_balancer/load_balancer_config.yaml',
    scheduling_mode='capacity_aware'
)

# Simulate router decision
router_probs = {
    'deepseek_ocr': 0.23,
    'qwen2_5_vl_3b': 0.68,
    'qwen2_5_vl_7b': 0.85,
    'qwen3_vl_8b_thinking': 0.76,
    'gemma_3_27b': 0.92
}

# Schedule request
decision = lb.schedule(
    sample_id='req_001',
    task_type='vlm',
    router_probs=router_probs,
    preferred_model='gemma_3_27b'  # Router's choice
)

print(f"LB chose: {decision['chosen_model']}")
print(f"Expected latency: {decision['expected_latency_ms']}ms")
print(f"Reason: {decision['scheduling_reason']}")
```

**Possible Outputs**:

**Scenario 1: Accept router choice**
```
LB chose: gemma_3_27b
Expected latency: 2000ms
Reason: router_choice_accepted
```

**Scenario 2: Override due to capacity**
```
LB chose: qwen2_5_vl_7b
Expected latency: 850ms
Reason: preferred_model_overloaded (gemma_3_27b queue=5)
```

**Scenario 3: Choose cheapest**
```
LB chose: qwen2_5_vl_3b
Expected latency: 400ms
Reason: cost_minimizing (saves $0.0004 per request)
```

### Step 6.4: Run Load Balancer Stress Test

```bash
# Open notebook:
# load_balancer/notebooks/02_load_balancer_stress_test.ipynb

# This notebook simulates 750 requests across 4 load profiles:
# 1. Low load (5 RPS for 30s)
# 2. Medium load (20 RPS for 60s)
# 3. High load (50 RPS for 60s)
# 4. Bursty load (10 RPS with spikes to 100 RPS)

# Compares 3 scheduling modes:
# - router_only: Trust router (98% SLA violations!)
# - capacity_aware: Consider queue lengths (94% violations)
# - cost_minimizing: Minimize cost (93% violations)

# Key insight: SLA violations are high because base latencies
# exceed SLA target. Need faster models or relaxed SLA.
```

### Step 6.5: Explore End-to-End Pipeline

```bash
# Open notebook:
# load_balancer/notebooks/00_pipeline_tutorial.ipynb

# This notebook shows the COMPLETE flow:
# 1. Load sample from SQL (with image)
# 2. Router predicts best model
# 3. Load balancer schedules request
# 4. Inference engine calls VLM
# 5. Response parsed and logged
# 6. Confidence score computed

# Expected output: Full pipeline in ~2-3 seconds
```

---

## Part 7: End-to-End Integration

### Step 7.1: Understand System API

The `system_api` module ties everything together:

```
FastAPI Server (system_api/main.py)
  ↓
/v1/chat/completions endpoint
  ↓
1. Extract prompt from request
2. Call RouterService → get routing decision
3. Call LoadBalancerService → get scheduling decision
4. Call InferenceService → execute VLM
5. Log to DataCollector
6. Return response
```

### Step 7.2: Start API Server

```bash
# Terminal 1: Start database
docker-compose up postgres

# Terminal 2: Start API server
uvicorn artemis_final.system_api.main:app --reload --host 0.0.0.0 --port 8000

# You should see:
# INFO: Uvicorn running on http://0.0.0.0:8000
# INFO: All services initialized successfully.
```

### Step 7.3: Test API with curl

```bash
# Health check
curl http://localhost:8000/health

# Expected: {"status": "ok"}

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ],
    "router_mode": "balanced",
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Expected response:
{
  "id": "uuid-here",
  "object": "chat.completion",
  "model": "qwen2_5_vl_7b",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "2 + 2 equals 4."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

### Step 7.4: Test API with Python

```python
import requests

# Make request
response = requests.post(
    'http://localhost:8000/v1/chat/completions',
    json={
        'messages': [
            {'role': 'user', 'content': 'Describe this chart.'}
        ],
        'router_mode': 'accuracy',  # Force best model
        'temperature': 0.7,
        'max_tokens': 200
    }
)

result = response.json()
print(f"Model used: {result['model']}")
print(f"Response: {result['choices'][0]['message']['content']}")
```

### Step 7.5: Submit Feedback

```bash
# Get sample_id from previous response
SAMPLE_ID="uuid-from-response"

# Submit feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{
    \"sample_id\": \"$SAMPLE_ID\",
    \"params\": {
      \"score\": 0.95,
      \"text\": \"Excellent response, very accurate\"
    }
  }"

# Expected: {"status": "received"}
```

---

## Part 8: Production Deployment

### Step 8.1: Use Docker Compose (Full Stack)

```bash
# Edit docker-compose.yml to add your VLM endpoints
# (Or use mock inference for testing)

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# Expected:
# NAME                STATUS
# artemis-postgres    Up (healthy)
# artemis-router      Up

# View logs
docker-compose logs -f artemis-router
```

### Step 8.2: Production Configuration

**Update `configs/artemis.yaml`**:

```yaml
db:
  url: "postgresql+psycopg2://artemis:artemis@postgres:5432/artemis"

router:
  checkpoint_path: "checkpoints/best_reward_router.pt"
  device: "cuda"  # ← Use GPU in production!

load_balancer:
  config_file: "load_balancer/load_balancer_config.yaml"

# ... rest of config
```

**Set environment variables**:

```bash
# .env file
DATABASE_URL=postgresql+psycopg2://artemis:artemis@postgres:5432/artemis
ROUTER_DEVICE=cuda
WANDB_API_KEY=your_key_here
WANDB_PROJECT=artemis-production
```

### Step 8.3: Performance Tuning

**Enable GPU**:
```yaml
# configs/artemis.yaml
router:
  device: "cuda:0"  # Use first GPU
```

**Connection Pooling**:
```python
# common/db.py (already configured)
engine = create_engine(
    db_url,
    pool_size=20,  # Max 20 connections
    max_overflow=40,  # Allow 40 overflow
    pool_pre_ping=True  # Verify connections
)
```

**Batch Inference** (for high throughput):
```python
# Use router.route_batch() instead of route()
results = router.route_batch(
    prompts=['Q1', 'Q2', 'Q3'],
    modes=['balanced', 'fast', 'accuracy']
)
```

### Step 8.4: Monitoring Setup

**Enable W&B Logging**:

```yaml
# router/router_config_reward.yaml
logging:
  wandb_enabled: true
  wandb_project: "artemis-production"
  wandb_run_name: "router-prod-v1"
```

**Metrics to Track**:
- Router latency (P50, P95, P99)
- Model selection distribution
- SLA violations
- Cost per request
- Accuracy (if ground truth available)

**Grafana Dashboards** (optional):
1. Set up Prometheus metrics endpoint
2. Configure Grafana to scrape metrics
3. Create dashboards for:
   - Request rate
   - Latency percentiles
   - Model utilization
   - Error rates

### Step 8.5: Scaling

**Horizontal Scaling** (Multiple API instances):

```yaml
# docker-compose.yml
services:
  artemis-router:
    deploy:
      replicas: 3  # Run 3 instances
    # ... rest of config

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - artemis-router
```

**Load Balancer Autoscaling**:

Already configured in `load_balancer_config.yaml`:
```yaml
models:
  qwen2_5_vl_7b:
    autoscale:
      enable: true
      scale_up_latency_factor: 0.8
      scale_down_util_threshold: 0.3
```

---

## Part 9: Monitoring & Feedback Loop

### Step 9.1: Data Collection

The system automatically collects:
- Every request (prompt, routing decision, LB decision)
- Every response (model output, latency, tokens)
- User feedback (scores, corrections)

**Stored in**:
- `vlm_samples_collected` - Live traffic samples
- `vlm_responses_collected` - Live responses
- `vlm_feedback` - User feedback

### Step 9.2: Trigger Retraining

**Manual Retraining**:

```bash
curl -X POST http://localhost:8000/admin/retrain

# Expected: {"status": "retraining_started"}

# Check logs for progress
docker-compose logs -f artemis-router
```

**Automatic Retraining** (schedule with cron):

```bash
# Add to crontab (retrain weekly on Sunday at 2 AM)
0 2 * * 0 curl -X POST http://localhost:8000/admin/retrain
```

**Retraining Workflow**:
1. Query new samples from `vlm_samples_collected`
2. Compute rewards
3. Fine-tune router (1 epoch)
4. Save new checkpoint
5. Hot-reload router (no downtime!)

### Step 9.3: Monitor Routing Decisions

**Query routing logs**:

```sql
-- Connect to database
docker exec -it $(docker ps -q -f name=postgres) psql -U artemis -d artemis

-- View recent routing decisions
SELECT
    sample_id,
    router_choice,
    lb_choice,
    router_mode,
    inference_latency_ms,
    created_at
FROM router_live_logs
ORDER BY created_at DESC
LIMIT 10;

-- Model selection distribution
SELECT
    lb_choice,
    COUNT(*) as count,
    AVG(inference_latency_ms) as avg_latency
FROM router_live_logs
GROUP BY lb_choice
ORDER BY count DESC;
```

### Step 9.4: Analyze Performance Drift

```python
# Check if router performance is degrading
import pandas as pd
from artemis_final.common.db import get_session_factory

Session = get_session_factory(DATABASE_URL)

with Session() as session:
    # Query last 7 days of routing decisions
    query = """
    SELECT
        DATE(created_at) as date,
        router_choice,
        COUNT(*) as count,
        AVG(user_score) as avg_score
    FROM router_live_logs
    WHERE created_at > NOW() - INTERVAL '7 days'
      AND user_score IS NOT NULL
    GROUP BY DATE(created_at), router_choice
    ORDER BY date DESC;
    """

    df = pd.read_sql(query, session.bind)
    print(df)

# If avg_score is dropping → time to retrain!
```

### Step 9.5: A/B Testing

**Test new router version**:

```python
# Deploy two versions
# Version A: Current production router
# Version B: New retrained router

# Route 90% to A, 10% to B
import random

if random.random() < 0.1:
    router = router_b  # New version
else:
    router = router_a  # Current version

# Track which version performed better
# After 1 week, compare metrics and decide
```

---

## 🎯 Success Checklist

### Environment Setup ✓
- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Package installed (`pip install -e .`)
- [ ] Imports working

### Database Setup ✓
- [ ] PostgreSQL running (Docker or native)
- [ ] Schema created (4 tables)
- [ ] Migrations applied (5 files)
- [ ] Can connect from Python

### Data Pipeline ✓
- [ ] Datasets loaded to `vlm_samples`
- [ ] VLM responses in `vlm_responses`
- [ ] Evaluation scores computed
- [ ] Rewards calculated

### Router Training ✓
- [ ] Local cache created
- [ ] Reward router trained
- [ ] Checkpoint saved (180 MB)
- [ ] Validation metrics look good (Pearson ~0.24)

### Router Inference ✓
- [ ] Router loads checkpoint successfully
- [ ] Single routing works
- [ ] Batch routing works
- [ ] All 4 modes work
- [ ] Unit tests pass (6/6)

### Load Balancer ✓
- [ ] Configuration file created
- [ ] Load balancer initializes
- [ ] Scheduling decisions make sense
- [ ] Stress test completed

### API Integration ✓
- [ ] API server starts
- [ ] Health check responds
- [ ] Chat completions endpoint works
- [ ] Feedback endpoint works

### Production Deployment ✓
- [ ] Docker Compose works
- [ ] GPU enabled (if available)
- [ ] Monitoring configured
- [ ] Performance tuned

### Feedback Loop ✓
- [ ] Data collection working
- [ ] Retraining endpoint works
- [ ] Metrics tracking set up
- [ ] A/B testing plan defined

---

## 🐛 Common Issues & Solutions

### Issue: "Module not found"

**Solution**: Install as editable package
```bash
cd /path/to/Which_VLM_Router
pip install -e .
```

---

### Issue: "Database connection failed"

**Solution**: Check Docker and connection string
```bash
docker ps | grep postgres  # Ensure running
echo $DATABASE_URL  # Verify env var
psql $DATABASE_URL -c "SELECT 1"  # Test connection
```

---

### Issue: "Router latency > 100ms"

**Solution**: Use GPU
```yaml
# configs/artemis.yaml
router:
  device: "cuda"  # or "mps" for Apple Silicon
```

**Benchmarks**:
- CPU: 20-50ms
- GPU: 5-15ms
- MPS: 10-25ms

---

### Issue: "Load balancer always violates SLA"

**Solution**: Check if `base_latency_ms` in config matches reality
```yaml
models:
  qwen2_5_vl_7b:
    base_latency_ms: 800  # ← Ensure this is accurate
    sla_ms: 2500  # ← And this is achievable
```

If VLM inference takes 3000ms but SLA is 2000ms, violations are unavoidable.

---

### Issue: "Router makes illogical decisions"

**Solution**: Verify model/mode order matches training
```python
# Check training config
from artemis_final.router_train.config import RouterModelConfig
config = RouterModelConfig()
print("Training model order:", config.model_names)

# Check inference config
import yaml
with open('router/router_config_reward.yaml') as f:
    inf_config = yaml.safe_load(f)
    print("Inference model order:", inf_config['router']['model_name_order'])

# These MUST match exactly!
```

---

## 📚 Next Steps

### Learning More

1. **Read the Papers**:
   - Router architecture inspiration: [RouteLLM](https://arxiv.org/abs/2406.18665)
   - Reward modeling: [InstructGPT](https://arxiv.org/abs/2203.02155)
   - Load balancing: [VLM Serving Systems](https://arxiv.org/abs/2312.xxxxx)

2. **Explore Notebooks**:
   - All 18 notebooks are documented and functional
   - Start with `load_balancer/notebooks/00_pipeline_tutorial.ipynb` for overview

3. **Experiment**:
   - Try different reward functions
   - Add new routing modes
   - Test with your own VLM models

### Production Enhancements

1. **Add Authentication**:
   - API keys for `/v1/chat/completions`
   - Rate limiting per user

2. **Improve Observability**:
   - Structured logging (JSON)
   - Distributed tracing (Jaeger)
   - Real-time dashboards (Grafana)

3. **Optimize Performance**:
   - Model quantization (INT8)
   - ONNX export for faster inference
   - Request batching

4. **Expand Capabilities**:
   - Multi-modal router (vision + text)
   - Online learning (continuous retraining)
   - Multi-region deployment

---

## 🎓 Congratulations!

You've successfully:
- ✅ Set up the entire Artemis VLM Router system
- ✅ Trained a reward-based router
- ✅ Deployed an end-to-end pipeline
- ✅ Integrated router, load balancer, and inference engine
- ✅ Set up monitoring and feedback loops

You now have a **production-ready VLM routing system** that can:
- Route queries to the best model based on accuracy, cost, or latency
- Handle SLA constraints with load balancing
- Continuously improve through feedback and retraining

**Enjoy building with Artemis!** 🚀
