param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

$bashCommand = @"
set -e
cd /home/zhl/finetune-webui
source /home/zhl/finetune-webui/swift-env/bin/activate
echo "ms-swift WebUI: http://${HostName}:$Port"
swift web-ui --lang zh --server_name $HostName --server_port $Port
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
