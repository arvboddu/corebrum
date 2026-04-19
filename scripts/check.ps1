$ErrorActionPreference = "Stop"

Write-Host "Checking Python sources..."
python -m compileall main.py engine

Write-Host "Checking JSON manifests..."
Get-Content package.json -Raw | ConvertFrom-Json | Out-Null
Get-Content corebrum-extension\manifest.json -Raw | ConvertFrom-Json | Out-Null

Write-Host "Checks passed."
