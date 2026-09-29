$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating Python environment failed.' }
}
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $ProjectPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Installing pip failed.' }
& $ProjectPython -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw 'Installing CUDA PyTorch failed.' }
& $ProjectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Installing project dependencies failed.' }
& $ProjectPython -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is required for training'"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
if (-not (Test-Path 'models\bert-base-chinese\model.safetensors')) {
    & $ProjectPython prepare_model.py
    if ($LASTEXITCODE -ne 0) { throw 'Downloading Chinese BERT failed.' }
}
