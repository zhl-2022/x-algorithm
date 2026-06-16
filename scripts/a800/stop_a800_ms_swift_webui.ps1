param(
    [int]$LocalPort = 7861,
    [int]$RemotePort = 17860,
    [switch]$KeepRemote
)

$ErrorActionPreference = "Stop"

$PidFile = Join-Path $env:TEMP "a800_ms_swift_webui_${LocalPort}.pid"
if (Test-Path -LiteralPath $PidFile) {
    $TunnelPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue
    if ($TunnelPid) {
        $Process = Get-Process -Id ([int]$TunnelPid) -ErrorAction SilentlyContinue
        if ($Process) {
            Stop-Process -Id $Process.Id -Force
            Write-Host "Stopped SSH tunnel pid=$($Process.Id)"
        }
    }
    Remove-Item -LiteralPath $PidFile -Force
}

$Pattern = "127.0.0.1:${LocalPort}:10.100.1.3:${RemotePort}"
$TunnelProcesses = Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'" |
    Where-Object { $_.CommandLine -like "*$Pattern*" }
foreach ($Tunnel in $TunnelProcesses) {
    Stop-Process -Id $Tunnel.ProcessId -Force
    Write-Host "Stopped SSH tunnel pid=$($Tunnel.ProcessId)"
}

if (-not $KeepRemote) {
    $RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/stop_ms_swift_webui.sh"
    ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
}
