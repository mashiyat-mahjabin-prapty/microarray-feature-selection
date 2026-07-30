param(
  [string]$PythonCommand = "python",
  [string]$ResultsDirectory,
  [string]$OutputDirectory
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "shap_analysis.py"
$manifest = Join-Path $repoRoot "outputs\dataset_manifest.csv"

if (-not $ResultsDirectory) {
  $ResultsDirectory = Join-Path $repoRoot "outputs\model_eval"
}
if (-not $OutputDirectory) {
  $OutputDirectory = Join-Path $repoRoot "outputs\shap"
}
if (-not (Test-Path -LiteralPath $manifest)) {
  & $PythonCommand (Join-Path $PSScriptRoot "make_dataset_manifest.py")
  if ($LASTEXITCODE -ne 0) {
    throw "Could not generate the dataset manifest."
  }
}

$datasets = @(
  "Binary_Adenocarcinoma",
  "Binary_BrainTumor",
  "Binary_BreastCancer",
  "Binary_ColonTumor",
  "Binary_Gastric",
  "Binary_Leukemia",
  "Binary_Lung",
  "Binary_Lymphoma",
  "Binary_Myeloma",
  "Binary_OvarianCancer",
  "Binary_Prostate",
  "Multiclass_BrainCancer",
  "Multiclass_Crohns",
  "Multiclass_EndometrialCancer",
  "Multiclass_Glioma",
  "Multiclass_Leukemia_3",
  "Multiclass_Leukemia_4",
  "Multiclass_LungCancer",
  "Multiclass_Lymphoma",
  "Multiclass_MLL",
  "Multiclass_SRBCT"
)

foreach ($dataset in $datasets) {
  Write-Host ""
  Write-Host "Running SHAP for $dataset" -ForegroundColor Cyan

  $maxDisplay = if ($dataset -eq "Binary_Leukemia") { 5 } else { 10 }

  & $PythonCommand $script `
    --dataset-name $dataset `
    --results-dir $ResultsDirectory `
    --manifest $manifest `
    --out-dir $OutputDirectory `
    --max-display $maxDisplay `
    --formats tif eps png `
    --seed 42 `
    --n-jobs 1

  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: $dataset" -ForegroundColor Red
  } else {
    Write-Host "DONE: $dataset" -ForegroundColor Green
  }
}
