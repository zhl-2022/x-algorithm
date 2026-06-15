param(
    [int]$Port = 19080,
    [string]$Prompt = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Prompt)) {
    $PromptB64 = "6K+35Y+q5Zue5aSN77yaQTgwMCBRd2VuMy0xLjdCIOacjeWKoeW3sui/numAmuOAgi9ub190aGluaw=="
} else {
    $PromptBytes = [System.Text.Encoding]::UTF8.GetBytes($Prompt)
    $PromptB64 = [Convert]::ToBase64String($PromptBytes)
}
$RemoteCommand = "PROMPT_B64=$PromptB64 bash /root/zhl/qwen-qlora-lab/scripts/call_qwen3_17b_base.sh $Port"

ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
