# LiteratureAgent Pass Pipeline

Patched controller script:

`<repo_root>\literature_agent_full_end_to_end_v21_3_english_sanitizer.py`

If you have a local copy elsewhere, use that path in the `$script` variable below.

Set common paths (edit these to match your workstation):

```powershell
$script = "$PWD\literature_agent_full_end_to_end_v21_3_english_sanitizer.py"
$base = "$PWD\..\..\data\Perovskite_database_content_all_data.csv"
$ontology = "$PWD\..\..\config\perovskite_ontology_library_v19.json"
$work = "$PWD\..\..\lit_outputs"
$integration = "$PWD\..\..\artifacts\literature_dataset_update"
$modelScript = "$PWD\..\..\models\pce_then_stability_same_approach.py"
$modelOut = "$PWD\..\..\artifacts\pce_then_stability_lit_updated"
$knowledgeGraphOut = "$PWD\..\..\artifacts\literature_knowledge_graph"
```

Build or refresh the evidence-grounded claim-centered scientific knowledge
graph from existing records, summaries, consolidated structured extraction,
figure evidence, and raw evidence snippets:

```powershell
python $script `
  --pipeline_stage knowledge_graph `
  --base_csv $base `
  --ontology_path $ontology `
  --work_dir $work `
  --knowledge_graph_out_dir $knowledgeGraphOut
```

Fresh Google Drive extract batch:

```powershell
python $script `
  --pipeline_stage extract_batch `
  --base_csv $base `
  --ontology_path $ontology `
  --work_dir $work `
  --integration_out_dir $integration `
  --from_scratch --full_literature_run --run_mode initial --max_papers 50 `
  --single_paper_source drive `
  --vision_enable 0 --inline_vision 0 --llm_cache_enable 1 --no_require_doi
```

Resume extract batch:

```powershell
python $script `
  --pipeline_stage extract_batch `
  --base_csv $base `
  --ontology_path $ontology `
  --work_dir $work `
  --run_mode expand --full_literature_run --max_papers 50 `
  --vision_enable 0 --inline_vision 0 --llm_cache_enable 1 --no_require_doi
```

Sanitize summaries:

```powershell
python $script `
  --pipeline_stage sanitize_summaries `
  --base_csv $base `
  --work_dir $work
```

Vision pass:

```powershell
python $script `
  --pipeline_stage vision_pass `
  --base_csv $base `
  --work_dir $work `
  --vision_max_figures_per_paper 2 `
  --vision_update_mode append
```

Integrate dry run:

```powershell
python $script `
  --pipeline_stage integrate_and_model `
  --base_csv $base `
  --work_dir $work `
  --integration_out_dir $integration `
  --integration_dry_run --no_require_doi
```

Integrate and run model:

```powershell
python $script `
  --pipeline_stage integrate_and_model `
  --base_csv $base `
  --work_dir $work `
  --integration_out_dir $integration `
  --model_script $modelScript `
  --model_out_dir $modelOut `
  --no_require_doi --run_model
```

Timing logs:

```powershell
# Per-paper and batch timing events are written here during extraction:
Get-Content "$work\timing_logs\paper_timing.jsonl" -Tail 20

# Controller-level elapsed time is stored here after any stage:
Get-Content "$work\full_end_to_end_v20_run_summary.json"
```
