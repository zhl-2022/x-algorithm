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

$DriverMounts = @(
    "--runtime=runc",
    "--device=/dev/nvidia0",
    "--device=/dev/nvidiactl",
    "--device=/dev/nvidia-uvm",
    "--device=/dev/nvidia-uvm-tools",
    "-v /usr/lib64/libcuda.so.565.57.01:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro",
    "-v /usr/lib64/libcuda.so.565.57.01:/usr/lib/x86_64-linux-gnu/libcuda.so:ro",
    "-v /usr/lib64/libnvidia-ml.so.565.57.01:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro",
    "-v /usr/lib64/libnvidia-ml.so.565.57.01:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so:ro",
    "-v /usr/lib64/libnvidia-ptxjitcompiler.so.565.57.01:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro"
) -join " "

$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && docker run --rm $DriverMounts -e CONFIRM_STAGE3_TRAIN=1 -e MAX_STEPS=$MaxSteps -v /root/zhl/qwen-qlora-lab:/workspace/qwen-qlora-lab qwen-qlora-swift:latest bash /workspace/qwen-qlora-lab/scripts/stage3_train_grpo_inside.sh"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
