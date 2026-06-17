param()

$ErrorActionPreference = "Stop"

$SnapshotRoot = Resolve-Path "docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small"
$RemoteTar = "/tmp/stage3_alignment_eval_results_safe.tar"
$LocalTar = Join-Path $env:TEMP "stage3_alignment_eval_results_safe.tar"

if (Test-Path $LocalTar) {
    Remove-Item -LiteralPath $LocalTar -Force
}

$RemoteCommand = 'cd /root/zhl/qwen-qlora-lab && rm -f /tmp/stage3_alignment_eval_results_safe.tar && { find data -maxdepth 1 -type f -name "stage3_alignment_infer_prompts.jsonl" 2>/dev/null; find logs -maxdepth 1 -type f -name "stage3_alignment_eval*" 2>/dev/null; } | tar -cf /tmp/stage3_alignment_eval_results_safe.tar -T -'

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

Write-Host "Stage 3 alignment evaluation results synced into $SnapshotRoot"
