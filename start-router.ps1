# Launcher for the local LiteLLM model router.
# Loads .env, then starts the proxy on port 4000.
#
# Usage:  .\start-router.ps1          (foreground; Ctrl+C to stop)
#
# Local runs are for testing with curl and for exercising the config. Cursor
# cannot reach a localhost base URL: it fetches custom base URLs server-side
# and refuses private addresses. Use the VPS endpoint in DEPLOY.md for Cursor.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .env)) {
    Write-Host "No .env found. Copy .env.example to .env and fill it in first." -ForegroundColor Red
    exit 1
}

Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}

# Decision log for the policy plugin. Lives next to the config so a local run
# is inspectable the same way a container run is.
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:POLICY_LOG = Join-Path $logDir "routing.jsonl"

# The policy plugin is resolved relative to the config file, so it must sit
# next to litellm-config.yaml. Fail loudly here rather than at first request.
if (-not (Test-Path (Join-Path $PSScriptRoot "routing_policy.py"))) {
    Write-Host "routing_policy.py is missing: router_settings.plugins would fail to load." -ForegroundColor Red
    exit 1
}

Write-Host "Starting LLM router on http://localhost:4000/v1 ..." -ForegroundColor Green
Write-Host "Decision log: $env:POLICY_LOG" -ForegroundColor DarkGray
# -X utf8 forces UTF-8 regardless of system locale (cp950 etc.) so the
# YAML loader and any provider responses decode correctly.
python -X utf8 -c "import sys; sys.argv = ['litellm', '--config', 'litellm-config.yaml', '--port', '4000']; from litellm import run_server; run_server()"
