# Phase 1A — File Discovery Plan

PROJECT_ROOT: /Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Which_VLM_Router

## Your only job in this phase: build a plan. Read nothing yet. Write one file.

---

## Step 1: Get the full file tree

Run this exact command:

```bash
find /Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Which_VLM_Router \
  -type f | sort
```

---

## Step 2: Apply skip rules — remove these from your list

Remove any file whose path contains any of these directory segments:
```
.git/
__pycache__/
.venv/
venv/
env/
Lib/
site-packages/
dist-packages/
node_modules/
.cache/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
wandb/
mlruns/
runs/
execution_results/
outputs/
results/
logs/
data/raw/
data/interim/
data/processed/
datasets/raw/
.ipynb_checkpoints/
```

Also remove any file with these extensions:
```
.pt .pth .pkl .ckpt .safetensors .onnx .bin
.npy .npz .h5 .hdf5 .arrow .parquet
.pyc .pyo .pyd
.log
.jpg .jpeg .png .gif .bmp .tiff .ico
(exception: keep any image inside docs/ or assets/ — these may be architecture diagrams)
```

Also remove these specific filenames regardless of location:
```
poetry.lock
package-lock.json
yarn.lock
Pipfile.lock
.DS_Store
.gitignore
.gitattributes
```

---

## Step 3: Categorize what remains

Put every surviving file into exactly one category:

**CORE** — read in full
- All `.py` files
- All `.md` files
- All `.ipynb` files (read cell source only, not outputs)

**CONFIG** — read in full if under 50KB, summarize only if larger
- `*.yaml`, `*.yml`, `*.toml`, `*.cfg`, `*.ini`
- `requirements*.txt`, `setup.py`, `setup.cfg`, `pyproject.toml`
- `.env.example` (not `.env` — skip `.env` entirely, it has secrets)
- `*.json` config files under 50KB (check size first with `wc -c`)

**SKIP** — do not read, just note existence
- `*.csv`, `*.tsv`, `*.jsonl` — note path and approximate size only
- Any `*.json` over 50KB — note path and size only
- `.env` files — note existence, do not read (may contain secrets)

---

## Step 4: Write this one file immediately

Write `docs/meta/FILE_PLAN.json`:

```json
{
  "generated_at": "<ISO datetime>",
  "project_root": "<path>",
  "total_files_found": 0,
  "total_files_to_read": 0,
  "total_files_skipped": 0,
  "read_queue": [
    {
      "path": "relative/path/file.py",
      "category": "CORE",
      "estimated_size_kb": 12,
      "batch": 1
    }
  ],
  "skipped": [
    {
      "path": "relative/path/train.csv",
      "reason": "dataset file",
      "size_kb": 450000
    }
  ],
  "modules_detected": [
    "router",
    "encoder",
    "load_balancer"
  ]
}
```

**Assign batch numbers:** Group files into batches of 10. Assign batch 1 to the most structurally important files first:
- Priority 1 (batch 1–2): `README.md`, `AGENTS.md`, `CONVENTIONS.md`, all `__init__.py`, all `runners.py` / `interfaces.py` / `api.py`
- Priority 2 (next batches): All other `.py` files, ordered by module
- Priority 3 (last batches): Config files, notebooks, other `.md` files

---

## Step 5: Print a summary to confirm

After writing `FILE_PLAN.json`, print:
```
DISCOVERY COMPLETE
Total files found: N
Files to read: N (across M batches of 10)
Files skipped: N
Modules detected: [list]
Largest files to read: [top 5 by size]
```

Stop here. Do not read any files yet.
