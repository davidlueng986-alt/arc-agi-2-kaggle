# Download the official ARC Prize 2026 (ARC-AGI-2) files with the Kaggle CLI.
# Auth: a KGAT token in %USERPROFILE%\.kaggle\access_token, or KAGGLE_API_TOKEN.
# The JSON files land in ..\data and are gitignored. Do not commit them.
# See official\HANDBOOK.md for why, and official\data-manifest.json for the
# 2026-10-14 hashes.

param(
    [switch]$IncludeModel
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$data = Join-Path $root 'data'
New-Item -ItemType Directory -Force -Path $data | Out-Null

kaggle competitions download -c arc-prize-2026-arc-agi-2 -p $data --force
$zip = Join-Path $data 'arc-prize-2026-arc-agi-2.zip'
if (-not (Test-Path $zip)) {
    throw "Download did not produce $zip"
}
Expand-Archive -Path $zip -DestinationPath $data -Force
Remove-Item $zip

$expected = @(
    'arc-agi_training_challenges.json',
    'arc-agi_training_solutions.json',
    'arc-agi_evaluation_challenges.json',
    'arc-agi_evaluation_solutions.json',
    'arc-agi_test_challenges.json',
    'sample_submission.json'
)
foreach ($name in $expected) {
    $path = Join-Path $data $name
    if (-not (Test-Path $path)) {
        throw "Missing $name after unzip"
    }
    Write-Host ("{0}`t{1} bytes" -f $name, (Get-Item $path).Length)
}

if ($IncludeModel) {
    python -c "import kagglehub; p = kagglehub.model_download('sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'); print('Model at:', p)"
}

Write-Host "Done. Compare hashes with official\data-manifest.json"
