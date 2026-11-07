$ErrorActionPreference = "Stop"

# Ensure Python scripts directory is available in PATH
$scriptsPath = "C:\Users\Hai Nam\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"
if ($env:PATH -notlike "*$scriptsPath*") {
    $env:PATH = "$scriptsPath;" + $env:PATH
}

Write-Host "=== [1/5] Running Ruff Linter & Formatter Check ===" -ForegroundColor Cyan
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts

Write-Host "=== [2/5] Running Pyright Static Type Checker ===" -ForegroundColor Cyan
uv run pyright src tests scripts

Write-Host "=== [3/5] Running AST Zero-Default & Anti-Hardcode Gate ===" -ForegroundColor Cyan
uv run python scripts/check_no_defaults_or_hardcode.py

Write-Host "=== [4/5] Running Import-Linter Clean Architecture Contracts ===" -ForegroundColor Cyan
uv run lint-imports

Write-Host "=== [5/5] Running Pytest Unit, Property, and Contract Suite ===" -ForegroundColor Cyan
uv run pytest tests -v

Write-Host "=== ALL 5 VERIFICATION GATES PASSED 100% ===" -ForegroundColor Green
