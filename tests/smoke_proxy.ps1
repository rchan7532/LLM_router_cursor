# End-to-end smoke test: boot the real proxy against placeholder upstreams and
# assert the policy plugin actually fires and logs a decision.
#
#   .\tests\smoke_proxy.ps1
#
# The unit tests exercise the algorithm in isolation; this exercises the parts
# they cannot see: litellm's plugin loader, the RoutingPlugin Protocol check,
# config acceptance and the routing pipeline. Both bugs found during the build
# of this setup (a dataclass load failure and a plugin path naming a class)
# passed every unit test and failed only here.
#
# The upstream URLs are intentionally unreachable, so every completion ends in
# a 500 after ~30s of DNS retries. That is expected: routing happens before the
# upstream call, and the decision log is the assertion.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

$port = 4099
$logDir = Join-Path $root "logs"
$decisionLog = Join-Path $logDir "smoke.jsonl"
$stdout = Join-Path $logDir "smoke.out"
$stderr = Join-Path $logDir "smoke.err"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Remove-Item $decisionLog, $stdout, $stderr -ErrorAction SilentlyContinue

$env:LITELLM_MASTER_KEY = "sk-smoke-test"
$env:ROUTER_KEY_DEFAULT = "smoke-default"
$env:ROUTER_KEY_BASIC = "smoke-basic"
$env:ROUTER_KEY_DEFAULT_BASE = "https://example.invalid/v1"
$env:ROUTER_KEY_BASIC_BASE = "https://example.invalid/v1"
$env:POLICY_LOG = $decisionLog

# Control plane for the lease assertions: a temp control.json the policy
# reads through the same env-var wiring production uses (POLICY_STATE_DIR).
$controlDir = Join-Path $logDir "smoke-control"
New-Item -ItemType Directory -Force -Path $controlDir | Out-Null
$controlJson = Join-Path $controlDir "control.json"
Remove-Item $controlJson, "$controlJson.lastgood" -ErrorAction SilentlyContinue
$env:POLICY_STATE_DIR = $controlDir

$request = Join-Path $env:TEMP "cursor_smoke_request.json"
# ASCII, not utf8: Windows PowerShell 5.1 writes a BOM for utf8, which makes the
# JSON body invalid and the proxy rejects it with 400 before routing runs.
$body = @'
{
  "model": "cursor-auto",
  "messages": [
    {"role": "system", "content": "You are Cursor."},
    {"role": "user", "content": "seed"},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "fix the failing test in report_parser.py, here is the traceback: Traceback (most recent call last): TypeError"}
  ]
}
'@
[System.IO.File]::WriteAllText($request, $body, [System.Text.Encoding]::ASCII)

Write-Host "Booting proxy on port $port ..." -ForegroundColor Green
$proc = Start-Process -FilePath python `
    -ArgumentList '-X', 'utf8', (Join-Path $PSScriptRoot "_proxy_runner.py"), "$port" `
    -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow

# Per-request stderr snapshots. litellm makes several policy calls per
# request (initial attempt + internal retries), so decision-log assertions
# and the error check below are anchored to what each request produced.
$stderrDir = Join-Path $logDir "smoke-stderr"
New-Item -ItemType Directory -Force -Path $stderrDir | Out-Null

function Send-Ask($tag) {
    $before = (Get-Content $decisionLog -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $stderrHere = Join-Path $stderrDir "$tag.err"
    $code = curl.exe -s -m 60 -o NUL -w "%{http_code}" -X POST "http://127.0.0.1:$port/v1/chat/completions" `
        -H "Authorization: Bearer sk-smoke-test" -H "Content-Type: application/json" `
        --data-binary "@$request" 2> $stderrHere
    Start-Sleep -Seconds 2
    $after = (Get-Content $decisionLog -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $records = @()
    if ($after -gt $before) {
        $records = Get-Content $decisionLog | Select-Object -Skip $before -First ($after - $before) | ForEach-Object { $_ | ConvertFrom-Json }
    }
    return [pscustomobject]@{ Code = $code; Records = @($records); ErrFile = $stderrHere }
}

try {
    $deadline = (Get-Date).AddSeconds(90)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $code = curl.exe -s -o NUL -w "%{http_code}" -m 5 "http://127.0.0.1:$port/health/liveliness"
        if ($code -eq "200") { $ready = $true; break }
    }
    if (-not $ready) {
        Write-Host "FAIL proxy never became healthy. Last stderr:" -ForegroundColor Red
        Get-Content $stderr -Tail 30
        exit 1
    }
    Write-Host "  ok   proxy healthy" -ForegroundColor DarkGray

    $models = curl.exe -s -m 10 "http://127.0.0.1:$port/v1/models" -H "Authorization: Bearer sk-smoke-test"
    if ($models -match '"cursor-auto"') {
        Write-Host "  ok   cursor-auto is served" -ForegroundColor DarkGray
    } else {
        Write-Host "FAIL cursor-auto missing from /v1/models" -ForegroundColor Red
        exit 1
    }

    Write-Host "Sending a debug ask (upstream is unreachable; a 500 is expected) ..."
    $r1 = Send-Ask "req1"
    Write-Host "  request returned http=$($r1.Code)" -ForegroundColor DarkGray
    Start-Sleep -Seconds 1

    $failed = $false
    if ($r1.Code -eq "400") {
        # A malformed body never reaches routing, so say so explicitly rather
        # than letting it read as a routing failure.
        Write-Host "FAIL the proxy rejected the request body (400); routing never ran" -ForegroundColor Red
        Get-Content $stderrHere -ErrorAction SilentlyContinue | Select-Object -Last 5
        exit 1
    }
    if (-not $r1.Records) {
        Write-Host "FAIL the policy plugin never logged a decision" -ForegroundColor Red
        $failed = $true
    } else {
        $first = $r1.Records[0]
        Write-Host "  ok   decision logged: $($first.chosen) (kind=$($first.kind), reason=$($first.reason))" -ForegroundColor DarkGray
        if ($first.chosen -ne "openai/kimi-k2.7-code") {
            Write-Host "FAIL a debug ask should route to kimi-k2.7-code, got $($first.chosen)" -ForegroundColor Red
            $failed = $true
        }
    }

    $bad = Select-String -Path $stderr, $stdout -Pattern 'No deployments left|missing 1 required positional|policy_error|Application startup failed' -ErrorAction SilentlyContinue
    if ($bad) {
        Write-Host "FAIL routing pipeline errors:" -ForegroundColor Red
        $bad | ForEach-Object { $_.Line }
        $failed = $true
    }

    if ($failed) { exit 1 }

    # ---- lease phase (B5): a stateless deadline lease must pin and log ----
    Write-Host "Writing a deadline lease (kimi, 120s) with atomic write discipline ..."
    $leasePython = @'
import json, os, time, sys
path, mode = sys.argv[1], sys.argv[2]
if mode == "live":
    doc = {"lease": {"model": "openai/kimi-k2.7-code", "expires_ts": time.time() + 120, "set_by": "smoke"}}
else:
    doc = {"lease": {"model": "openai/kimi-k2.7-code", "expires_ts": time.time() - 1, "set_by": "smoke"}}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as h:
    json.dump(doc, h)
    h.flush()
    os.fsync(h.fileno())
os.replace(tmp, path)
'@
    $leaseScript = Join-Path $logDir "smoke_lease.py"
    [System.IO.File]::WriteAllText($leaseScript, $leasePython, [System.Text.Encoding]::ASCII)
    python -X utf8 $leaseScript $controlJson live

    Write-Host "Sending a second debug ask under the live lease ..."
    $r2 = Send-Ask "req2"

    $failed = $false
    if (-not $r2.Records) {
        Write-Host "FAIL no decision logged under the live lease" -ForegroundColor Red
        $failed = $true
    } elseif ($r2.Records[0].reason -ne "lease" -or $r2.Records[0].chosen -ne "openai/kimi-k2.7-code") {
        Write-Host "FAIL lease did not pin: reason=$($r2.Records[0].reason) chosen=$($r2.Records[0].chosen)" -ForegroundColor Red
        $failed = $true
    } else {
        Write-Host "  ok   lease pinned the decision (reason=lease, chosen=kimi-k2.7-code)" -ForegroundColor DarkGray
    }

    # ---- corrupt phase (M5): torn control.json must fall to LAST-GOOD, not
    #      defaults. The last-good snapshot IS the live lease, so the next
    #      decision must still be lease-pinned; falling to defaults would
    #      drop the lease and read fresh. ----
    Write-Host "Corrupting control.json (torn write) ..."
    [System.IO.File]::WriteAllText($controlJson, "{ torn", [System.Text.Encoding]::ASCII)

    Write-Host "Sending a third ask with the control file corrupt ..."
    $r3 = Send-Ask "req3"

    if (-not $r3.Records) {
        Write-Host "FAIL no decision logged with corrupt control.json" -ForegroundColor Red
        $failed = $true
    } elseif ($r3.Records[0].reason -ne "lease") {
        Write-Host "FAIL corrupt control.json lost the last-good lease (fell to defaults?): reason=$($r3.Records[0].reason)" -ForegroundColor Red
        $failed = $true
    } else {
        Write-Host "  ok   corrupt control.json fell to LAST-GOOD (lease still pinned, no policy_error)" -ForegroundColor DarkGray
    }

    # ---- expiry phase (B5): the deadline is evaluated statelessly; overwriting
    #      with an EXPIRED lease (also proves a repaired file heals) must
    #      release the pin and re-decide fresh. ----
    Write-Host "Overwriting with an expired lease (repaired file) ..."
    python -X utf8 $leaseScript $controlJson expired

    Write-Host "Sending a fourth ask with the lease expired ..."
    $r4 = Send-Ask "req4"

    if (-not $r4.Records) {
        Write-Host "FAIL no decision logged with expired lease" -ForegroundColor Red
        $failed = $true
    } elseif ($r4.Records[0].reason -eq "lease") {
        Write-Host "FAIL expired lease still pinned: reason=$($r4.Records[0].reason)" -ForegroundColor Red
        $failed = $true
    } else {
        Write-Host "  ok   expired lease released the pin (reason=$($r4.Records[0].reason), fresh decision)" -ForegroundColor DarkGray
    }

    $bad = @()
    foreach ($errFile in (Get-ChildItem $stderrDir -Filter "*.err" -ErrorAction SilentlyContinue)) {
        $bad += Select-String -Path $errFile.FullName -Pattern 'No deployments left|missing 1 required positional|policy_error|Application startup failed' -ErrorAction SilentlyContinue
    }
    $bad += Select-String -Path $stderr, $stdout -Pattern 'Application startup failed|missing 1 required positional' -ErrorAction SilentlyContinue
    if ($bad) {
        Write-Host "FAIL routing pipeline errors:" -ForegroundColor Red
        $bad | ForEach-Object { $_.Line }
        $failed = $true
    }
    if ($failed) { exit 1 }

    Write-Host "smoke test passed" -ForegroundColor Green
}
finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Remove-Item $request, $leaseScript -ErrorAction SilentlyContinue
    Remove-Item Env:\POLICY_STATE_DIR -ErrorAction SilentlyContinue
    # The runner boots litellm from the repo root; clear the runtime artifacts
    # it leaves behind so a fresh run starts clean.
    Remove-Item (Join-Path $root "logs") -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $root "litellm_cost_map.json") -ErrorAction SilentlyContinue
    Pop-Location
}
