param()

$ErrorActionPreference = "Stop"
$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && python3 scripts/stage3_reward_plugin.py"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
