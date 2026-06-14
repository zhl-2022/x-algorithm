param()

$ErrorActionPreference = "Stop"

$bashCommand = @'
set -e

echo "== WSL =="
whoami
cat /etc/os-release | sed -n '1,3p'
command -v uv || true

echo
echo "== GPU passthrough =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "nvidia-smi not found"
fi

echo
echo "== ms-swift env =="
source /home/zhl/finetune-webui/swift-env/bin/activate
python -c "import torch, swift, transformers, gradio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('swift', getattr(swift, '__version__', 'unknown')); print('transformers', transformers.__version__); print('gradio', gradio.__version__)"
swift web-ui --help | sed -n '1,20p'
deactivate

echo
echo "== LLaMA-Factory env =="
source /home/zhl/finetune-webui/llamafactory-env/bin/activate
python -c "import torch, torchaudio, transformers, gradio, llamafactory; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__); print('transformers', transformers.__version__); print('gradio', gradio.__version__); print('llamafactory', getattr(llamafactory, '__version__', 'unknown'))"
llamafactory-cli env | sed -n '1,40p'
'@

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
