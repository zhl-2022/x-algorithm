$ErrorActionPreference = "Stop"
$RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/stop_qwen3_17b_deploy.sh"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
