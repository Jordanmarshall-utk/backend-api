# Before/After Literature-Mining Model Comparison

## Purpose

This workflow measures whether adding LiteratureAgent records changes:

- PCE predictive performance
- stability regression performance
- stability classification performance
- the number of usable rows for each target
- physics-informed harsh-condition target coverage and performance

The comparison uses the same current model script for both datasets:

```text
Before: original Perovskite Database
After:  original database plus accepted LiteratureAgent records
```

## Interpretation

A useful result can appear in two ways:

1. **Coverage improvement**
   More rows become available for a target, especially sparse stability or
   physical harsh-condition targets.

2. **Metric improvement**
   Matched OOF or grouped-holdout metrics improve after adding literature-mined
   records.

Metric changes are indicative, but they are not definitive causal proof that
literature mining improved generalization because the updated dataset also
changes the cross-validation population. The strongest follow-up evaluation is
a frozen external holdout that is never used for either training run.

Do not compare runs that use different model code, backend, CV settings,
hyperparameters, physical assumptions, or feature rules. The comparison utility
writes `configuration_comparison.csv` and warns about mismatches.

Each new model run writes:

```text
run_environment.json
dataset_profile.json
```

`run_environment.json` records the model backend and SHA-256 hash of the model
script, allowing the report to verify that both runs used identical model code.

## Commands

Set paths:

```powershell
$model = "<repo_root>\models\pce_then_stability_same_approach.py"
$compare = "<repo_root>\scripts\compare_model_runs.py"
$beforeCsv = "<repo_root_or_data_dir>\Perovskite_database_content_all_data.csv"
$afterCsv = "<repo_root>\artifacts\literature_dataset_update\updated_perovskite_database_with_literature_agent.csv"
$beforeOut = "<repo_root>\artifacts\pce_then_stability_before_literature"
$afterOut = "<repo_root>\artifacts\pce_then_stability_after_literature"
$comparisonOut = "<repo_root>\artifacts\pce_stability_before_after_comparison"
```

Run the original database:

```powershell
python $model --csv $beforeCsv --out $beforeOut
```

Run the literature-updated database:

```powershell
python $model --csv $afterCsv --out $afterOut
```

Generate comparison tables and plots:

```powershell
python $compare `
  --before $beforeOut `
  --after $afterOut `
  --out $comparisonOut `
  --before-label "Original Perovskite Database" `
  --after-label "LiteratureAgent-updated database"
```

## Primary Outputs

```text
README.txt
tables/
    configuration_comparison.csv
    aligned_model_comparison.csv
    metric_comparison_long.csv
    unmatched_models.csv
plots/
    overview/
        target_support_before_after.png
        signed_metric_improvement.png
    pce/
        pce_before_after_metrics.png
    stability_regression/
        stability_regression_r2_before_after.png
        stability_regression_rmse_before_after.png
    stability_classification/
        stability_classification_auc_before_after.png
        stability_classification_balanced_accuracy_before_after.png
```

Use `plots/overview/target_support_before_after.png` to show whether literature
mining increased usable training data. Use the matched metric plots to show
whether that added data improved or worsened predictive performance.

## Optional Full Plot Gallery

To gather every plot from a model-result folder into one categorized gallery
without moving the original files:

```powershell
python "<repo_root>\scripts\organize_model_plots.py" `
  --run-dir "<repo_root>\artifacts\pce_then_stability_after_literature"
```

This creates:

```text
plot_gallery/
    before_after_comparison/
    pce/
    stability_regression/
    stability_classification/
    physical_stability/
    other/
    plot_gallery_index.csv
    README.txt
```
