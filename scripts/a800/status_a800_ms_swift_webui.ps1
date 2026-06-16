param(
    [int]$LocalPort = 7861,
    [int]$RemotePort = 17860
)

$ErrorActionPreference = "Stop"

Write-Host "Local tunnel processes:"
$Pattern = "127.0.0.1:${LocalPort}:10.100.1.3:${RemotePort}"
Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'" |
    Where-Object { $_.CommandLine -like "*$Pattern*" } |
    Select-Object ProcessId, CommandLine

Write-Host ""
Write-Host "Remote WebUI status:"
$RemoteCommand = "bash /root/zhl/qwen-qlora-lab/scripts/status_ms_swift_webui.sh $RemotePort"
ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 '$RemoteCommand'"
