param(
    [switch]$ConfirmEval,
    [int]$MaxNewTokens = 256
)

$ErrorActionPreference = "Stop"

if (-not $ConfirmEval) {
    throw "This script starts A800 fixed-prompt evaluation. Re-run with -ConfirmEval when you are ready."
}
if ($MaxNewTokens -lt 32 -or $MaxNewTokens -gt 1024) {
    throw "MaxNewTokens must be between 32 and 1024 for this learning script."
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

$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && docker run --rm $DriverMounts -e CONFIRM_STAGE3_EVAL=1 -e MAX_NEW_TOKENS=$MaxNewTokens -v /root/zhl/qwen-qlora-lab:/workspace/qwen-qlora-lab qwen-qlora-swift:latest bash /workspace/qwen-qlora-lab/scripts/stage3_run_alignment_eval_inside.sh"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
