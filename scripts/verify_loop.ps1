# scripts/verify_loop.ps1
$ErrorActionPreference = "Stop"

# Enable PS native command error preference if supported (PS 7.4+)
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $true
}

# Locate uv.exe if not already in PATH
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $candidates = @(
        "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts",
        "$env:USERPROFILE\.cargo\bin",
        "$env:LOCALAPPDATA\Programs\uv"
    )
    foreach ($dir in $candidates) {
        if (Test-Path "$dir\uv.exe") {
            $env:PATH = "$dir;" + $env:PATH
            break
        }
    }
}

# Remove msys64 from PATH to prevent MingW interpreter inspection warnings
$env:PATH = (($env:PATH -split ";") | Where-Object { $_ -notlike "*msys64*" }) -join ";"

function Invoke-Gate {
    param(
        [Parameter(Mandatory = $true)][string]$StageName,
        [Parameter(Mandatory = $true)][scriptblock]$Script
    )
    Write-Host "`n=== $StageName ===" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    try {
        & $Script
    } catch {
        Write-Host "`n[VERIFICATION GATE FAILED] Exception in $StageName" -ForegroundColor Red
        Write-Host $_ -ForegroundColor Red
        exit 1
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[VERIFICATION GATE FAILED] $StageName exited with code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Gate "[1/5] Running Ruff Linter Check" { uv run ruff check src tests scripts }
Invoke-Gate "[1/5] Running Ruff Formatter Check" { uv run ruff format --check src tests scripts }
Invoke-Gate "[2/5] Running Pyright Static Type Checker" { uv run pyright src tests scripts }
Invoke-Gate "[3/5] Running AST Zero-Default & Anti-Hardcode Gate" { uv run python scripts/check_no_defaults_or_hardcode.py }
Invoke-Gate "[4/5] Running Import-Linter Clean Architecture Contracts" { uv run lint-imports }
Invoke-Gate "[5/5] Running Pytest Unit, Property, and Contract Suite" { uv run pytest tests -v }

Write-Host "`n=== ALL 5 VERIFICATION GATES PASSED 100% ===" -ForegroundColor Green
exit 0
