# Architecture

## System Overview

```mermaid
graph TD
    User([User Request<br/>text + image]) --> Router

    subgraph Router ["1. Router (artemis_final/router/)"]
        Enc[DistilBERT Encoder<br/>frozen, 768-dim]
        ME[Model Embeddings<br/>5 × 32-dim]
        MLE[Mode Embedding<br/>4 modes]
        MLP[MLP Head<br/>2 layers, 512-dim]
        Enc --> MLP
        ME --> MLP
        MLE --> MLP
    end

    Router --> LB

    subgraph LB ["2. Load Balancer (artemis_final/load_balancer/)"]
        Sched[ArtemisLoadBalancer<br/>capacity-aware scheduler]
        Stats[StatsRegistry<br/>per-task latency/cost]
        SLA[SlaMonitor<br/>latency SLA tracking]
        Sched --> Stats
        Sched --> SLA
    end

    LB --> IE

    subgraph IE ["3. Inference Engine (artemis_final/inference_engine/)"]
        Client[WhichVLMClient<br/>OpenAI-compatible]
    end

    IE --> VLM1[qwen2_5_vl_3b]
    IE --> VLM2[qwen2_5_vl_7b]
    IE --> VLM3[qwen3_vl_8b_thinking]
    IE --> VLM4[gemma_3_27b]
    IE --> VLM5[deepseek_ocr]

    VLM1 --> R1[Response]
    VLM2 --> R2[Response]
    VLM3 --> R3[Response]
    VLM4 --> R4[Response]
    VLM5 --> R5[Response]

    R1 --> Out([User])
    R2 --> Out
    R3 --> Out
    R4 --> Out
    R5 --> Out

    subgraph ARES ["4. ARES (artemis_final/ares/)"]
        Eval[RouterEvalPipeline<br/>Scorer + VLMJudge + Glider]
        DB[(PostgreSQL<br/>samples/responses<br/>/evaluations)]
        Eval --> DB
    end

    DB -. "periodic retraining" .-> Router
```

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as System API
    participant R as Router
    participant LB as Load Balancer
    participant IE as Inference Engine
    participant V as VLM Backend
    participant DB as PostgreSQL
    participant Eval as ARES

    U->>API: POST /v1/chat/completions<br/>(prompt, mode, image)

    rect rgb(240, 248, 255)
        Note over API,R: 1. Routing
        API->>R: route_request(prompt, mode, metadata)
        R->>R: encode text (DistilBERT)<br/>add model + mode embeddings
        R->>R: MLP forward → rewards per VLM
        R-->>API: {chosen_model, rewards, mode}
    end

    rect rgb(255, 250, 240)
        Note over API,LB: 2. Load Balancing
        API->>LB: schedule(router_output, context)
        LB->>LB: check SLA + queue capacity
        alt model overloaded
            LB->>LB: override to cheapest feasible model
        end
        LB-->>API: {final_model, queue_wait, total_latency}
    end

    rect rgb(240, 255, 240)
        Note over API,IE: 3. Inference
        API->>IE: run_image(prompt, image, model)
        IE->>V: POST /v1/chat/completions
        V-->>IE: response, usage, latency
        IE-->>API: {text, cost, latency}
    end

    API-->>U: response

    rect rgb(255, 240, 245)
        Note over API,DB: 4. Evaluation (async)
        API->>DB: log_sample, log_response
        DB-->>Eval: samples available
        Eval->>Eval: run Scorer + VLMJudge
        Eval->>DB: write evaluations
    end
```

## Module Dependency Graph

```mermaid
graph LR
    subgraph Training ["Training Pipeline"]
        DB1[("PostgreSQL")]
        RT[Router Train<br/>router_train/]
        RT --> DB1
    end

    subgraph Inference ["Inference Pipeline"]
        SR[System API<br/>system_api/]
        R[Router<br/>router/]
        LB[Load Balancer<br/>load_balancer/]
        IE[Inference Engine<br/>inference_engine/]
        V[VLM Backends]
    end

    subgraph Evaluation ["Evaluation"]
        Eval[ARES<br/>ares/]
        Eval --> DB1
    end

    SR --> R
    R --> LB
    LB --> IE
    IE --> V
    V --> Eval
    Eval -. "retrain" .-> RT
    RT -. "checkpoint" .-> R
```

## Component Descriptions

### Router (`artemis_final/router/`)

The router is the core intelligence of ARTEMIS. At inference time it:

1. Encodes the input text with a frozen DistilBERT encoder
2. Concatenates model-type and mode embeddings
3. Runs the combined vector through an MLP head to predict reward scores for each VLM
4. Returns the VLM with the highest predicted reward for the given mode

Three architectures are supported:

- **RewardRouterInference** (recommended) — predicts scalar rewards via MSE loss
- **PairwiseRouterInference** — learns a ranking via margin-based loss
- **ClassicalRouterInference** — classifies which model is best via cross-entropy

### Load Balancer (`artemis_final/load_balancer/`)

After the router selects a model, the load balancer applies operational constraints:

- **SLA verification**: Checks whether the selected model's current queue would violate the latency SLA
- **Queue management**: Tracks in-flight requests per model
- **Override logic**: If constraints are violated, redirects to the next-best feasible model

The scheduler runs in `capacity_aware` mode by default and logs all decisions to its internal `SlaMonitor`.

### Inference Engine (`artemis_final/inference_engine/`)

A thin OpenAI-compatible client that:

- Calls VLM backends via the `/v1/chat/completions` endpoint
- Extracts token usage, latency, and cost from responses
- Runs both LLM (text-only) and VLM (image + text) inference through a unified `WhichVLMClient` interface

### ARES (`artemis_final/ares/`)

The evaluation and data pipeline. Key components:

- **Scorer** — evaluates responses against ground truth (accuracy, F1, etc.)
- **VLMJudge** (Molmo) — provides listwise ranking of model responses with images
- **GliderEvaluator** — optional text-only evaluator for faster scoring
- **DB operations** — reads/writes all tables in PostgreSQL

The `RouterEvalPipeline` orchestrates parallel evaluation across all samples and models, then writes results back to the database for retraining.

### Router Training (`artemis_final/router_train/`)

The training pipeline:

1. Loads profiling data from PostgreSQL (samples, responses, evaluations)
2. Computes multi-objective reward functions per (sample, model, mode) tuple
3. Builds a PyTorch dataset and trains the router with the selected loss function
4. Evaluates against oracle and baseline routers
5. Exports a `.pt` checkpoint to `checkpoints/`

Training is driven by Jupyter notebooks in `notebooks/router_train/`.

## Cross-Module Data Contracts

| Contract | Defined In | Used By | Fields |
|---|---|---|---|
| `RouterOutput` | `load_balancer/core/types.py` | LB, System API | sample_id, task_type, router_probs, preferred_model |
| `SchedulingDecision` | `load_balancer/core/types.py` | System API, inference | chosen_model, is_overloaded, est_latency_ms, est_cost_usd |
| `Sample` | `router/core/schemas.py` | Router, ARES, DB | sample_id, source, text, image, label |
| `RouterDecision` | `router/core/schemas.py` | Router, LB | chosen_model, probs, raw_logits, model_order |
| `GlobalConfig` | `common/config_loader.py` | All modules | db.url, router, load_balancer, inference_engine config |
