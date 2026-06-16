param(
    [switch]$ConfirmTrain,
    [int]$MaxSteps = 20
)

$ErrorActionPreference = "Stop"

if (-not $ConfirmTrain) {
    throw "This script starts A800 GRPO training. Re-run with -ConfirmTrain when you are ready."
}
if ($MaxSteps -lt 1 -or $MaxSteps -gt 100) {
    throw "MaxSteps must be between 1 and 100 for this learning script."
}

$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && docker run --rm --gpus device=0 -e CONFIRM_STAGE3_TRAIN=1 -e MAX_STEPS=$MaxSteps -v /root/zhl/qwen-qlora-lab:/workspace/qwen-qlora-lab qwen-qlora-swift:latest bash /workspace/qwen-qlora-lab/scripts/stage3_train_grpo_inside.sh"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
