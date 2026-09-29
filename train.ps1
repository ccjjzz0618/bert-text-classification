$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
& '.\.venv\Scripts\python.exe' -u train.py @args
if ($LASTEXITCODE -ne 0) { throw 'Training failed; see the error above.' }
