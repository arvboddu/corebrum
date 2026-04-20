#!/usr/bin/env pwsh
# Corebrum Ollama Setup Script
# Sets OLLAMA_ORIGINS permanently and pulls model if needed

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Corebrum Ollama Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Step 1: Set OLLAMA_ORIGINS permanently at User level
Write-Host "`n[1/3] Setting OLLAMA_ORIGINS environment variable..." -ForegroundColor Yellow
$currentValue = [Environment]::GetEnvironmentVariable("OLLAMA_ORIGINS", "User")
if ($currentValue -ne "*") {
    [Environment]::SetEnvironmentVariable("OLLAMA_ORIGINS", "*", "User")
    Write-Host "  -> Set OLLAMA_ORIGINS=*" -ForegroundColor Green
    Write-Host "  -> Restart your shell for changes to take effect" -ForegroundColor Yellow
} else {
    Write-Host "  -> OLLAMA_ORIGINS already set to '*'" -ForegroundColor Green
}

# Set for current session
$env:OLLAMA_ORIGINS = "*"

# Step 2: Check if llama3.2:3b model is available
Write-Host "`n[2/3] Checking for llama3.2:3b model..." -ForegroundColor Yellow
$models = ollama list 2>$null | Out-String

if ($models -match "llama3\.2:3b") {
    Write-Host "  -> Model llama3.2:3b is already available" -ForegroundColor Green
} else {
    Write-Host "  -> Model NOT found. Pulling llama3.2:3b..." -ForegroundColor Yellow
    Write-Host "  -> This may take several minutes..." -ForegroundColor Cyan
    
    try {
        ollama pull llama3.2:3b
        Write-Host "  -> Successfully pulled llama3.2:3b" -ForegroundColor Green
    } catch {
        Write-Host "  -> ERROR: Failed to pull model: $_" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Verify or start Ollama service
Write-Host "`n[3/3] Verifying Ollama is running at http://localhost:11434..." -ForegroundColor Yellow

function Test-OllamaService {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

if (Test-OllamaService) {
    Write-Host "  -> Ollama is responsive" -ForegroundColor Green
} else {
    Write-Host "  -> Ollama not responsive. Starting ollama serve..." -ForegroundColor Yellow
    
    # Try to start ollama serve
    try {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -ErrorAction Stop
        Start-Sleep -Seconds 5
        
        # Verify it started
        if (Test-OllamaService) {
            Write-Host "  -> Ollama service started successfully" -ForegroundColor Green
        } else {
            Write-Host "  -> WARNING: Could not auto-start. Run 'ollama serve' manually" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  -> ERROR: Failed to start Ollama: $_" -ForegroundColor Red
    }
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Setup complete! You can now start Corebrum:" -ForegroundColor Green
Write-Host "python main.py" -ForegroundColor White
Write-Host "======================================" -ForegroundColor Cyan