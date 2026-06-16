param()

$ErrorActionPreference = "Stop"
$RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/stage3_readiness_check.sh"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
