param(
    [int]$Port = 19082,
    [string]$Model = "qwen3-1.7b-qlora-effective",
    [string]$InputPath = "data/stage2_eval_cases.jsonl",
    [string]$OutputPath = "logs/stage2_effective_lora_infer.jsonl",
    [string]$MarkdownPath = "logs/stage2_effective_lora_infer.md",
    [double]$Temperature = 0.2,
    [double]$TopP = 0.8
)

$ErrorActionPreference = "Stop"
foreach ($Value in @($Model, $InputPath, $OutputPath, $MarkdownPath)) {
    if ($Value -match "[\s']") {
        throw "This wrapper only accepts values without whitespace or single quotes: $Value"
    }
}

$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && python3 scripts/stage2_openai_batch.py --port $Port --model $Model --input $InputPath --output $OutputPath --markdown $MarkdownPath --temperature $Temperature --top_p $TopP"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
