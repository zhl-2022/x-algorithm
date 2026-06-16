param(
    [switch]$ConfirmTrain,
    [int]$MaxSteps = 30,
    [string]$Dataset = "data/stage3_dpo_preferences.jsonl",
    [string]$OutDir = "outputs/stage3_qwen3_17b_dpo",
    [string]$StartAdapter = ""
)

$ErrorActionPreference = "Stop"

if (-not $ConfirmTrain) {
    throw "This script starts A800 DPO training. Re-run with -ConfirmTrain when you are ready."
}
if ($MaxSteps -lt 1 -or $MaxSteps -gt 200) {
    throw "MaxSteps must be between 1 and 200 for this learning script."
}
foreach ($Value in @($Dataset, $OutDir, $StartAdapter)) {
    if ($Value -and $Value -match "[\s']") {
        throw "This wrapper only accepts values without whitespace or single quotes: $Value"
    }
}

$EnvArgs = "-e CONFIRM_STAGE3_TRAIN=1 -e MAX_STEPS=$MaxSteps -e DATASET=$Dataset -e OUT_DIR=$OutDir"
if ($StartAdapter) {
    $EnvArgs = "$EnvArgs -e START_ADAPTER=$StartAdapter"
}

$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && docker run --rm --gpus device=0 $EnvArgs -v /root/zhl/qwen-qlora-lab:/workspace/qwen-qlora-lab qwen-qlora-swift:latest bash /workspace/qwen-qlora-lab/scripts/stage3_train_dpo_inside.sh"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
