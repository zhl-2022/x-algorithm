param(
    [int]$Port = 19080
)

$ErrorActionPreference = "Stop"
$RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/deploy_qwen3_17b_base.sh $Port"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
