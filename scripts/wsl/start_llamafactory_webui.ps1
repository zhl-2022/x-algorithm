param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 7861
)

$ErrorActionPreference = "Stop"

$bashCommand = @"
set -e
cd /home/zhl/finetune-webui
source /home/zhl/finetune-webui/llamafactory-env/bin/activate
export GRADIO_SERVER_NAME=$HostName
export GRADIO_SERVER_PORT=$Port
echo "LLaMA-Factory WebUI: http://${HostName}:$Port"
llamafactory-cli webui
"@

$tempScript = New-TemporaryFile
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

try {
    [System.IO.File]::WriteAllText($tempScript.FullName, $bashCommand, $utf8NoBom)
    $windowsScript = $tempScript.FullName -replace "\\", "/"
    $wslScript = (wsl -- wslpath -a $windowsScript).Trim()
    wsl -- bash $wslScript
}
finally {
    Remove-Item -LiteralPath $tempScript.FullName -Force -ErrorAction SilentlyContinue
}
