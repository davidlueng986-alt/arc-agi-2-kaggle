# Download ARC Prize 2026 (ARC-AGI-2) competition data + model weights
# Requires: kaggle CLI authenticated (kaggle.com/settings -> API token)

$root = Split-Path -Parent $PSScriptRoot

# 1. Competition data
kaggle competitions download -c arc-prize-2026-arc-agi-2 -p "$root\data" --force
Expand-Archive -Path "$root\data\arc-prize-2026-arc-agi-2.zip" -DestinationPath "$root\data" -Force

# 2. Base model (Qwen3-4B SFT, ~8GB) for local reproduction
python -c "import kagglehub; p = kagglehub.model_download('sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'); print('Model at:', p)"

Write-Host "Done. Data in $root\data, model in %USERPROFILE%\.cache\kagglehub"
