param(
    [int]$Port = 19081,
    [string]$Prompt = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Prompt)) {
    $PromptB64 = "6K+355So566A5rSB5Lit5paH6Kej6YeK77yaQTgwMCDlhbHkuqvorq3nu4PjgII="
} else {
    $PromptBytes = [System.Text.Encoding]::UTF8.GetBytes($Prompt)
    $PromptB64 = [Convert]::ToBase64String($PromptBytes)
}
$RemoteCommand = "PROMPT_B64=$PromptB64 bash /root/zhl/qwen-qlora-lab/scripts/call_qwen3_17b_lora.sh $Port"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
