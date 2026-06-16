param()

$ErrorActionPreference = "Stop"
$RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/stage3_prepare_alignment_assets.sh"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
