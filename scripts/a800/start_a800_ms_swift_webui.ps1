param(
    [int]$LocalPort = 7861,
    [int]$RemotePort = 17860,
    [switch]$NoBrowser,
    [switch]$KeepRemoteOnFailure
)

$ErrorActionPreference = "Stop"

function Test-LocalTcpPort {
    param([int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(300, $false)) {
            $client.EndConnect($iar)
            return $true
        }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/start_ms_swift_webui.sh $RemotePort"
ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"

if (Test-LocalTcpPort -Port $LocalPort) {
    Write-Host "Local port $LocalPort is already open. Reusing http://127.0.0.1:$LocalPort"
} else {
    $TunnelSpec = "127.0.0.1:${LocalPort}:10.100.1.3:${RemotePort}"
    $Process = Start-Process -FilePath "ssh" -ArgumentList @("-o", "ExitOnForwardFailure=yes", "-N", "-L", $TunnelSpec, "srv4") -WindowStyle Hidden -PassThru
    $PidFile = Join-Path $env:TEMP "a800_ms_swift_webui_${LocalPort}.pid"
    Set-Content -Path $PidFile -Value $Process.Id -Encoding ascii
    Write-Host "Started SSH tunnel pid=$($Process.Id), local http://127.0.0.1:$LocalPort -> A800 :$RemotePort"
}

$Url = "http://127.0.0.1:$LocalPort"
for ($i = 0; $i -lt 60; $i++) {
    try {
        $curl = & curl.exe -fsS --max-time 5 $Url 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE"
        }
        Write-Host "ms-swift WebUI is ready: $Url"
        if (-not $NoBrowser) {
            Start-Process $Url
        }
        exit 0
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $KeepRemoteOnFailure) {
    Write-Host "Local tunnel failed; stopping remote WebUI container to avoid leftovers."
    powershell -NoProfile -File "$PSScriptRoot\stop_a800_ms_swift_webui.ps1" -LocalPort $LocalPort -RemotePort $RemotePort
}

throw "Local tunnel did not become ready: $Url"
