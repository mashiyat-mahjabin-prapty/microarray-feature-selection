# Interpretable Biomarker Discovery from Microarray Gene-Expression Data

This repository contains the code and curated results for an undergraduate
thesis on reproducible biomarker discovery and cancer classification using
high-dimensional microarray gene-expression data. The study evaluates 21
binary and multiclass classification tasks with a common two-stage
feature-selection workflow.

## Study overview

The workflow:

1. preprocesses each dataset within the machine-learning pipeline;
2. ranks probes using cross-validated XGBoost feature importance;
3. aggregates fold-wise ranks and retains the 500 highest-ranked probes;
4. refines the candidate set using SVM-RFECV with balanced accuracy;
5. derives a stable consensus biomarker panel;
6. assesses the selected panel with repeated stratified cross-validation; and
7. supports interpretation with SHAP attribution and functional enrichment.

The main implementation is
[`scripts/feature_selection_pipeline.py`](scripts/feature_selection_pipeline.py).

> **Interpretation of performance estimates:** feature discovery is performed
> on the full cohort, after which the selected panel is assessed using repeated
> stratified cross-validation. The reported values therefore describe
> post-selection panel performance; they are not claimed to be a nested-CV
> estimate of the complete biomarker-discovery procedure.

## Datasets

The analysis covers the following 21 tasks:

| Task type | Datasets |
|---|---|
| Binary | Adenocarcinoma, BrainTumor, BreastCancer, ColonTumor, Gastric, Leukemia, Lung, Lymphoma, Myeloma, OvarianCancer, Prostate |
| Multiclass | BrainCancer, Crohns, EndometrialCancer, Glioma, Leukemia (3 classes), Leukemia (4 classes), LungCancer, Lymphoma, MLL, SRBCT |

Each processed matrix is a CSV file with samples in rows, probes in columns, and
the target in a column named `CLASS`.

The source Adenocarcinoma matrix contains repeated biological feature
identifiers. Pandas disambiguates repeated column names with suffixes such as
`.1` and `.2`; the included result tables use those disambiguated names. Keep
the matrix unchanged when reproducing the archived analysis.

Generate the portable dataset manifest before running SHAP:

```bash
python scripts/make_dataset_manifest.py
```

This writes `outputs/dataset_manifest.csv` with repository-relative paths.

## Repository structure

```text
microarray_biomarker_selection/
├── Binary/                    # Eleven binary-classification matrices
├── Multiclass/                # Ten multiclass matrices
├── outputs/
│   └── model_eval/            # Curated per-dataset results
├── scripts/                   # Pipeline, interpretation, and figure scripts
├── requirements.txt
└── README.md
```

## Installation

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the biomarker pipeline

From the repository root, this example reproduces the final settings for the
binary Adenocarcinoma task:

```bash
python scripts/feature_selection_pipeline.py --csv Binary/Adenocarcinoma/Adenocarcinoma.csv --out-dir outputs/model_eval --xgb-cv-splits 5 --top-n 500 --consensus-threshold 0.50 --rfecv-cv-splits 5 --rfecv-step 0.10 --rfecv-scoring balanced_accuracy --min-features-to-select 5 --ml-cv-splits 5 --ml-repeats 5 --seed 42 --n-jobs -1
```

Windows users can run all 21 tasks with:

```powershell
.\scripts\run_pipeline_all.ps1
```

If Python is not available as `python`, provide the executable explicitly:

```powershell
.\scripts\run_pipeline_all.ps1 -PythonCommand "<PLEASE INSERT PATH TO PYTHON.EXE>"
```

The pipeline automatically reduces the number of stratified folds when the
smallest class contains fewer than five samples. In the archived analysis,
EndometrialCancer uses three folds, the four-class Leukemia task uses four, and
the remaining tasks use five.

The curated result folder for each task contains:

| File | Description |
|---|---|
| `classes.csv` | Encoded class labels |
| `config.json` | Run configuration and effective CV settings |
| `xgb_top_features.csv` | The 500 candidate probes passed to RFECV |
| `rfecv_cv_results.csv` | RFECV performance across panel sizes |
| `rfecv_selected_features_by_fold.csv` | Features selected in each RFECV fold |
| `rfecv_feature_consensus_all.csv` | Fold-wise selection frequencies |
| `rfecv_feature_consensus_threshold.csv` | Consensus features meeting the threshold |
| `final_selected_features.csv` | Final selected biomarker panel |
| `ml_fold_metrics.csv` | Fold-level post-selection assessment |
| `ml_summary_metrics.csv` | Summary performance statistics |

A fresh run also writes complete fold-level XGBoost rankings and consensus
tables. Those larger intermediate files can be placed in the accompanying
results archive instead of GitHub.

## SHAP interpretation

The SHAP script uses XGBoost contribution values for post-hoc interpretation;
it does not use SHAP values for model-performance estimation.

After generating the dataset manifest, run one task with:

```bash
python scripts/shap_analysis.py --dataset-name Binary_Adenocarcinoma --formats png
```

Windows users can run all tasks with:

```powershell
.\scripts\shap_all_run.ps1
```

The defaults read selection results from `outputs/model_eval` and write SHAP
outputs to `outputs/shap`. Use `--label-file` to provide an optional
probe-to-gene mapping. Unmapped features retain their probe identifiers. Add
`--collapse-duplicate-genes` only when the intended analysis is to average
probes mapped to the same gene before model fitting.

## Recreating figures

Create the LR-versus-stacking performance figure with:

```bash
python scripts/draw_main_lr_stacking_boxplots.py --formats png pdf
```

Create the workflow figure with:

```bash
python scripts/draw_workflow_figure.py --formats png pdf svg
```

The literature-comparison plot requires its manually curated input table:

```bash
python scripts/plot_literature_comparison.py --csv outputs/literature_comparison_tidy.csv --out-prefix outputs/figures/literature_comparison_combined --formats png pdf
```

Prefer vector PDF or SVG files for manuscript submission and PNG files for
GitHub previews. Place large TIFF collections in the results archive.

## Reproducibility

- The base random seed is fixed at `42`.
- Fold-specific XGBoost seeds are deterministically derived from the base seed.
- Random states are fixed for logistic regression, SVM, random forest,
  XGBoost, and the stacking classifier.
- Equal-ranked features are resolved with deterministic secondary sorting.
- Preprocessing is fitted within each evaluation fold.
- Adaptive fold counts and run parameters are recorded in `config.json`.

The same inputs and package versions should be used for the closest
reproduction. Hardware, operating-system, and library differences can still
produce small floating-point differences even with fixed random seeds.

## Data and results availability

Before publishing the repository, confirm that every source dataset permits
redistribution. If redistribution is not permitted, remove the corresponding
matrix from the public repository and provide its accession number, source URL,
and preprocessing instructions.

The recommended public-release layout is:

- GitHub: code, documentation, compact CSV/JSON results, and final PNG/PDF
  figures;
- a versioned data repository: processed matrices, full intermediate results,
  SHAP outputs, and large TIFF files.

Replace the placeholders below before publication:

- **Processed datasets and complete results:** `[PLEASE INSERT ARCHIVE DOI]`
- **Archived software release:** `[PLEASE INSERT SOFTWARE DOI]`

## Citation

If you use this repository, cite the associated manuscript and archived
software/data release:

`[PLEASE INSERT THE FINAL MANUSCRIPT CITATION]`

## Contact

Questions and reproducibility reports can be submitted through this
repository's GitHub Issues page.
