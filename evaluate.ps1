$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
& '.\.venv\Scripts\python.exe' evaluate.py @args
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; see the error above.' }
