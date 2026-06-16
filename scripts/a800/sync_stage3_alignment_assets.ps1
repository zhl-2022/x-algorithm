param()

$ErrorActionPreference = "Stop"

$SnapshotRoot = Resolve-Path "docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small"
$Generator = Join-Path $SnapshotRoot "scripts/stage3_make_alignment_data.py"

python $Generator | Write-Host

$Archive = Join-Path $env:TEMP "stage3_alignment_assets.tar"
if (Test-Path $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}

$Items = @(
    "data/stage3_dpo_preferences.jsonl",
    "data/stage3_grpo_prompts.jsonl",
    "data/stage3_alignment_eval_cases.jsonl",
    "scripts/stage3_make_alignment_data.py",
    "scripts/stage3_reward_plugin.py",
    "scripts/stage3_prepare_alignment_assets.sh",
    "scripts/stage3_readiness_check.sh",
    "scripts/stage3_train_dpo_inside.sh",
    "scripts/stage3_train_grpo_inside.sh"
)

& tar -C $SnapshotRoot -cf $Archive @Items

$RemoteTar = "/tmp/stage3_alignment_assets.tar"
scp $Archive "srv4:$RemoteTar"
ssh srv4 "scp -o StrictHostKeyChecking=no $RemoteTar root@10.100.1.3:$RemoteTar"
ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 'mkdir -p /root/zhl/qwen-qlora-lab && tar -C /root/zhl/qwen-qlora-lab -xf $RemoteTar && chmod +x /root/zhl/qwen-qlora-lab/scripts/stage3_*.sh /root/zhl/qwen-qlora-lab/scripts/stage3_*.py'"

Write-Host "Stage 3 alignment assets synced to /root/zhl/qwen-qlora-lab"
