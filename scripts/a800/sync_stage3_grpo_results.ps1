param()

$ErrorActionPreference = "Stop"

$SnapshotRoot = Resolve-Path "docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small"
$RemoteTar = "/tmp/stage3_grpo_results_safe.tar"
$LocalTar = Join-Path $env:TEMP "stage3_grpo_results_safe.tar"

if (Test-Path $LocalTar) {
    Remove-Item -LiteralPath $LocalTar -Force
}

$RemoteCommand = 'cd /root/zhl/qwen-qlora-lab && rm -f /tmp/stage3_grpo_results_safe.tar && { find logs -maxdepth 1 -type f \( -name "stage3_grpo*" -o -name "stage3_reward_selftest.log" \) 2>/dev/null; find outputs/stage3_qwen3_17b_grpo -type f \( -name "args.json" -o -name "logging.jsonl" -o -name "trainer_state.json" -o -name "adapter_config.json" -o -name "additional_config.json" -o -name "README.md" -o -path "*/images/*.png" \) 2>/dev/null; } | tar -cf /tmp/stage3_grpo_results_safe.tar -T -'

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
ssh srv4 "scp -o StrictHostKeyChecking=no root@10.100.1.3:$RemoteTar $RemoteTar"
scp "srv4:$RemoteTar" $LocalTar

$ArchiveEntries = & tar -tf $LocalTar
$UnsafeEntries = $ArchiveEntries | Where-Object {
    $_ -match '(^|/)(adapter_model\.safetensors|.*\.(safetensors|bin|pt|gguf|onnx))$'
}

if ($UnsafeEntries) {
    $UnsafeList = $UnsafeEntries -join [Environment]::NewLine
    throw "Refusing to extract archive because it contains model weights or binary artifacts:$([Environment]::NewLine)$UnsafeList"
}

& tar -xf $LocalTar -C $SnapshotRoot

Write-Host "Stage 3 GRPO safe results synced into $SnapshotRoot"
