param(
    [int]$Port = 19082,
    [string]$Model = "qwen3-1.7b-qlora-effective",
    [string]$InputPath = "data/stage2_sampling_prompts.jsonl",
    [string]$OutputPath = "logs/stage2_sampling_grid.jsonl",
    [string]$MarkdownPath = "logs/stage2_sampling_grid.md"
)

$ErrorActionPreference = "Stop"
foreach ($Value in @($Model, $InputPath, $OutputPath, $MarkdownPath)) {
    if ($Value -match "[\s']") {
        throw "This wrapper only accepts values without whitespace or single quotes: $Value"
    }
}

$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && python3 scripts/stage2_sampling_grid.py --port $Port --model $Model --input $InputPath --output $OutputPath --markdown $MarkdownPath"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
