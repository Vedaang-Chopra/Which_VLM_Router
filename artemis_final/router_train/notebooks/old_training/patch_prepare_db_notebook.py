import json
import os

NOTEBOOK_PATH = "/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Which_VLM_Router/artemis_final/notebooks/router_train/00_prepare_local_database.ipynb"

def patch_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # We want to replace the cell that calls `load_profiles_real_schema`
    # or the cell we previously patched.
    # I will look for "FETCHING DATA FROM POSTGRESQL" which is in the print statement.
    # This is persistent across my edits.
    
    target_snippet_1 = "df_all = load_profiles_real_schema("
    target_snippet_2 = "FETCHING DATA FROM POSTGRESQL"
    
    cell_index = -1
    
    for i, cell in enumerate(nb['cells']):
        source_str = "".join(cell.get('source', []))
        if target_snippet_1 in source_str or target_snippet_2 in source_str:
            cell_index = i
            break
            
    if cell_index == -1:
        print("Could not find the target cell to patch.")
        return

    print(f"Found target cell at index {cell_index}")
    
    # New content with Corrected Columns based on Schema
    # r.response_raw instead of response_text
    # ev.semantic_f1_f1 instead of semantic_f1_score
    
    new_source = [
        "# Load all data with filtering for non-empty responses/evals\n",
        "from sqlalchemy import text\n",
        "from db_utils import get_engine\n",
        "\n",
        "print(f\"\\n{'='*80}\")\n",
        "print(\"FETCHING DATA FROM POSTGRESQL (Filtered by Molmo Score)\")\n",
        "print(f\"{'='*80}\\n\")\n",
        "\n",
        "engine = get_engine(db_config)\n",
        "\n",
        "# Construct query to fetch samples that have valid response AND evaluation\n",
        "query = \"\"\"\n",
        "SELECT\n",
        "    -- sample-level\n",
        "    s.sample_id,\n",
        "    s.source_config,\n",
        "    s.source_dataset,\n",
        "    s.router_task,\n",
        "    s.data_split,\n",
        "    s.prompt_text                 AS prompt_raw,\n",
        "    s.txt_prompt_length_chars,\n",
        "    s.txt_prompt_length_words,\n",
        "\n",
        "    -- image metadata\n",
        "    i.img_width,\n",
        "    i.img_height,\n",
        "    i.img_aspect_ratio,\n",
        "\n",
        "    -- response-level (per model)\n",
        "    r.model_name,\n",
        "    r.model_prefix,\n",
        "    r.latency_ms,\n",
        "    r.estimated_cost_usd          AS cost_usd,\n",
        "    r.confidence_score,\n",
        "    r.response_raw,               -- Corrected: response_raw\n",
        "    r.total_tokens,\n",
        "\n",
        "    -- evaluation-level (per model)\n",
        "    ev.glider_score,\n",
        "    ev.judge_molmo_score,\n",
        "    ev.judge_molmo_rank_group,\n",
        "    ev.judge_molmo_raw\n",
        "\n",
        "FROM vlm_samples s\n",
        "JOIN vlm_responses r\n",
        "  ON s.sample_id = r.sample_id\n",
        "JOIN vlm_evaluations ev\n",
        "  ON s.sample_id = ev.sample_id\n",
        " AND r.model_name = ev.model_name\n",
        "LEFT JOIN vlm_images i\n",
        "  ON s.image_id = i.image_id\n",
        "\n",
        "WHERE \n",
        "    -- Ensure response is present and non-empty\n",
        "    r.response_raw IS NOT NULL \n",
        "    AND length(r.response_raw) > 0\n",
        "    -- Ensure evaluation score exists (Molmo)\n",
        "    AND ev.judge_molmo_score IS NOT NULL\n",
        "\"\"\"\n",
        "\n",
        "if LIMIT and LIMIT > 0:\n",
        "    query += f\" LIMIT {LIMIT}\"\n",
        "\n",
        "print(\"[Executing Query]\")\n",
        "# Execute\n",
        "try:\n",
        "    df_all = pd.read_sql(text(query), engine)\n",
        "    \n",
        "    print(f\"\\n[Data Loaded]\")\n",
        "    print(f\"  Total rows: {len(df_all):,}\")\n",
        "    if 'sample_id' in df_all.columns:\n",
        "        print(f\"  Unique samples: {df_all['sample_id'].nunique():,}\")\n",
        "    if 'model_name' in df_all.columns:\n",
        "        print(f\"  Unique models: {df_all['model_name'].nunique()}\")\n",
        "    \n",
        "    if 'data_split' in df_all.columns:\n",
        "        print(f\"\\n  Data split distribution:\")\n",
        "        for split, count in df_all['data_split'].value_counts().items():\n",
        "            pct = 100 * count / len(df_all)\n",
        "            print(f\"    {split:6s}: {count:7,} ({pct:5.2f}%)\")\n",
        "            \n",
        "except Exception as e:\n",
        "    print(f\"Error fetching data: {e}\")\n",
        "    # Fallback to empty df to prevent notebook crash\n",
        "    df_all = pd.DataFrame()\n"

    ]
    
    nb['cells'][cell_index]['source'] = new_source
    
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook patched successfully with Molmo columns.")

if __name__ == "__main__":
    patch_notebook()
