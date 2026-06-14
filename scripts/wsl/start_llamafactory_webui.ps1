param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 7861
)

$ErrorActionPreference = "Stop"

function Test-WslPortInUse {
    param([int]$CandidatePort)

    $checkCommand = "ss -ltn | awk '{print `$4}' | grep -Eq '(^|:)$($CandidatePort)$'"
    wsl -- bash -lc $checkCommand | Out-Null
    return $LASTEXITCODE -eq 0
}

function Resolve-WslPort {
    param(
        [int]$PreferredPort,
        [int]$StartPort,
        [int]$EndPort
    )

    if ($PreferredPort -gt 0 -and -not (Test-WslPortInUse -CandidatePort $PreferredPort)) {
        return $PreferredPort
    }

    if ($PreferredPort -gt 0) {
        Write-Warning "WSL port $PreferredPort is already in use; searching $StartPort-$EndPort."
    }

    for ($candidate = $StartPort; $candidate -le $EndPort; $candidate++) {
        if (-not (Test-WslPortInUse -CandidatePort $candidate)) {
            return $candidate
        }
    }

    throw "No free WSL port found in range $StartPort-$EndPort."
}

$Port = Resolve-WslPort -PreferredPort $Port -StartPort 7861 -EndPort 7899

$bashCommand = @"
set -e
cd /home/zhl/finetune-webui
source /home/zhl/finetune-webui/llamafactory-env/bin/activate
export NO_PROXY="127.0.0.1,localhost,::1,`$NO_PROXY"
export no_proxy="127.0.0.1,localhost,::1,`$no_proxy"
export BROWSER=/bin/true
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
