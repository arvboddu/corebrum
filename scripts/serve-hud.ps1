$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$port = 8080

if ($args.Length -gt 0 -and $args[0]) {
  $port = [int]$args[0]
}

Set-Location $repoRoot

Write-Host "Serving Corebrum HUD from $repoRoot on http://127.0.0.1:$port/"
python -m http.server $port --bind 127.0.0.1
