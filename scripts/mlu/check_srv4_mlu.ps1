param(
    [string]$Target = "srv4",
    [string]$Image = "cambricon-base/pytorch:v25.12.0-torch2.9.1-torchmlu1.30.2-ubuntu22.04-py310",
    [string]$Devices = "2,3"
)

$ErrorActionPreference = "Stop"

$workflow = Join-Path $HOME "Documents\WindowsPowerShell\CodexServerWorkflow\Invoke-KlbServer.ps1"
if (-not (Test-Path $workflow)) {
    throw "Server workflow script not found: $workflow"
}

function Invoke-Remote {
    param([Parameter(Mandatory = $true)][string]$Command)
    powershell -NoProfile -File $workflow -Target $Target -Command $Command
}

Write-Host "== srv4 connectivity =="
Invoke-Remote "hostname && whoami && date"

Write-Host "`n== cnmon snapshot =="
Invoke-Remote "cnmon | head -n 80"

Write-Host "`n== xalgorithm containers =="
Invoke-Remote "docker ps -a --filter name=xalgorithm-mlu --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' || true"

Write-Host "`n== recommended image check =="
$pythonCheck = "import torch, torch_mlu; print(torch.__version__); print(torch_mlu.__version__); print(torch.mlu.is_available()); print(torch.mlu.device_count())"
$dockerCommand = "docker run --rm --privileged -e MLU_VISIBLE_DEVICES=$Devices $Image python -c '$pythonCheck'"
Invoke-Remote $dockerCommand
