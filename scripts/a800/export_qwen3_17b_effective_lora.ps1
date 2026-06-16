param(
    [switch]$ConfirmExport
)

$ErrorActionPreference = "Stop"
if ($ConfirmExport) {
    $RemoteCommand = "CONFIRM_EXPORT=1 bash /root/zhl/qwen-qlora-lab/scripts/stage2_export_effective_lora.sh"
} else {
    $RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/stage2_export_effective_lora.sh"
}

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
