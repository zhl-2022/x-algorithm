param(
    [string]$Cases = "data/stage2_eval_cases.jsonl",
    [string]$Pred = "logs/stage2_effective_lora_infer.jsonl",
    [string]$Report = "logs/stage2_effective_lora_eval.md",
    [string]$Csv = "logs/stage2_effective_lora_eval.csv"
)

$ErrorActionPreference = "Stop"
foreach ($Value in @($Cases, $Pred, $Report, $Csv)) {
    if ($Value -match "[\s']") {
        throw "This wrapper only accepts values without whitespace or single quotes: $Value"
    }
}

$RemoteCommand = "cd /root/zhl/qwen-qlora-lab && python3 scripts/stage2_keyword_eval.py --cases $Cases --pred $Pred --report $Report --csv $Csv"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
