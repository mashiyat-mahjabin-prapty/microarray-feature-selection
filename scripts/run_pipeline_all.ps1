param(
  [string]$PythonCommand = "python",
  [string]$OutputDirectory
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "feature_selection_pipeline.py"

if (-not $OutputDirectory) {
  $OutputDirectory = Join-Path $repoRoot "outputs\model_eval"
}

$datasets = @(
  "Binary\Adenocarcinoma\Adenocarcinoma.csv",
  "Binary\BrainTumor\BrainTumor.csv",
  "Binary\BreastCancer\BreastCancer.csv",
  "Binary\ColonTumor\ColonTumor.csv",
  "Binary\Gastric\Gastric.csv",
  "Binary\Leukemia\Leukemia.csv",
  "Binary\Lung\Lung.csv",
  "Binary\Lymphoma\Lymphoma.csv",
  "Binary\Myeloma\Myeloma.csv",
  "Binary\OvarianCancer\OvarianCancer.csv",
  "Binary\Prostate\Prostate.csv",
  "Multiclass\BrainCancer\BrainCancer.csv",
  "Multiclass\Crohns\Crohns.csv",
  "Multiclass\EndometrialCancer\EndometrialCancer.csv",
  "Multiclass\Glioma\Glioma.csv",
  "Multiclass\Leukemia_3\Leukemia_3.csv",
  "Multiclass\Leukemia_4\Leukemia_4.csv",
  "Multiclass\LungCancer\LungCancer.csv",
  "Multiclass\Lymphoma\Lymphoma.csv",
  "Multiclass\MLL\MLL.csv",
  "Multiclass\SRBCT\SRBCT.csv"
)

foreach ($relativeCsv in $datasets) {
  $csv = Join-Path $repoRoot $relativeCsv
  Write-Host ""
  Write-Host "Running $relativeCsv" -ForegroundColor Cyan

  & $PythonCommand $script `
    --csv $csv `
    --out-dir $OutputDirectory `
    --xgb-cv-splits 5 `
    --top-n 500 `
    --consensus-threshold 0.50 `
    --rfecv-cv-splits 5 `
    --rfecv-step 0.10 `
    --rfecv-scoring balanced_accuracy `
    --min-features-to-select 5 `
    --ml-cv-splits 5 `
    --ml-repeats 5 `
    --seed 42 `
    --n-jobs -1

  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: $relativeCsv" -ForegroundColor Red
  } else {
    Write-Host "DONE: $relativeCsv" -ForegroundColor Green
  }
}
