<#
.SYNOPSIS
    Serves the studio at a permanent address using Tailscale Funnel.

.DESCRIPTION
    The counterpart to share_link.ps1, and the reason it exists: a Cloudflare
    quick tunnel invents a new hostname every time it starts, so the link has
    to be re-sent after every restart, sleep or crash. Tailscale Funnel gives
    this machine one address that never changes -
    https://<machine>.<tailnet>.ts.net - so the link can be sent once.

    Because the token in .env is also fixed, the whole link is stable:

        https://<machine>.<tailnet>.ts.net/s/<token>

    Funnel exposes this machine to the public internet, so the server is
    started with CONTENT_STUDIO_PUBLIC=1. That forces every request to be
    treated as coming from outside: the key is required, and guests cannot
    delete. Without it the app would decide by sniffing proxy headers, which
    were chosen for Cloudflare and are not guaranteed to match Tailscale's.

.PARAMETER Stop
    Take the funnel down and stop the studio.

.NOTES
    One-time setup, which has to be done by hand because it needs an account:

      1. tailscale login          (opens a browser; Google/GitHub, no card)
      2. Enable HTTPS certificates for the tailnet, and Funnel, in the
         Tailscale admin console. Running this script will print the exact
         link to click if either is missing.
#>

param([switch]$Stop)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8420
$tailscale = "C:\Program Files\Tailscale\tailscale.exe"

function Stop-Everything {
    if (Test-Path $tailscale) {
        # Remove the funnel first, so nothing is briefly published with no
        # server behind it.
        & $tailscale funnel --https=443 off 2>&1 | Out-Null
    }
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conn) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

if ($Stop) {
    Stop-Everything
    Write-Host "Funnel is off. The studio is no longer reachable from outside."
    exit 0
}

if (-not (Test-Path $tailscale)) {
    Write-Host "Tailscale isn't installed." -ForegroundColor Red
    Write-Host "Install it with:  winget install Tailscale.Tailscale"
    exit 1
}

# --- the token -----------------------------------------------------------
$envFile = Join-Path $root ".env"
$token = $null
if (Test-Path $envFile) {
    $line = Select-String -Path $envFile -Pattern '^CONTENT_STUDIO_LINK_TOKEN=(.+)$' |
            Select-Object -First 1
    if ($line) { $token = $line.Matches[0].Groups[1].Value.Trim() }
}
if (-not $token) {
    Write-Host "No CONTENT_STUDIO_LINK_TOKEN in .env." -ForegroundColor Red
    Write-Host "A funnel is public. Without a key anyone who finds the address can use it."
    exit 1
}

# --- signed in? ----------------------------------------------------------
$status = & $tailscale status 2>&1 | Out-String
if ($status -match "Logged out") {
    Write-Host ""
    Write-Host "Tailscale is installed but not signed in." -ForegroundColor Yellow
    Write-Host "Run this once, then run this script again:"
    Write-Host ""
    Write-Host "    & '$tailscale' login" -ForegroundColor White
    Write-Host ""
    Write-Host "It opens a browser. Sign in with Google or GitHub - no card needed."
    exit 1
}

Stop-Everything
Start-Sleep -Seconds 2

# --- the studio ----------------------------------------------------------
# CONTENT_STUDIO_PUBLIC is the whole reason this is safe; see the notes above.
$env:CONTENT_STUDIO_UI_NO_BROWSER = "1"
$env:CONTENT_STUDIO_PUBLIC = "1"
Start-Process -FilePath "$root\venv\Scripts\python.exe" `
              -ArgumentList "-u", "webapp.py" `
              -WorkingDirectory $root -WindowStyle Hidden `
              -RedirectStandardOutput "$env:LOCALAPPDATA\cloudflared\webapp.out" `
              -RedirectStandardError  "$env:LOCALAPPDATA\cloudflared\webapp.err"

$up = $false
foreach ($try in 1..15) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        $up = $true; break
    }
}
if (-not $up) {
    Write-Host "The studio didn't start. Last error output:" -ForegroundColor Red
    Get-Content "$env:LOCALAPPDATA\cloudflared\webapp.err" -Tail 20 -ErrorAction SilentlyContinue
    exit 1
}

# --- the funnel ----------------------------------------------------------
# 127.0.0.1 rather than localhost: the server binds IPv4 only, and "localhost"
# resolves to ::1 first on Windows. Same trap the Cloudflare script hit.
$result = & $tailscale funnel --bg "http://127.0.0.1:$port" 2>&1 | Out-String
Write-Host $result

if ($result -match "Funnel is not enabled|HTTPS is disabled|https://login\.tailscale\.com\S+") {
    Write-Host ""
    Write-Host "Tailscale needs two things switched on for your tailnet." -ForegroundColor Yellow
    Write-Host "The link above (or in the admin console) turns them on:"
    Write-Host "  1. HTTPS certificates"
    Write-Host "  2. Funnel"
    Write-Host "Then run this script again."
    exit 1
}

$host_name = (& $tailscale status --json 2>$null | ConvertFrom-Json).Self.DNSName
if ($host_name) { $host_name = $host_name.TrimEnd(".") }

if (-not $host_name) {
    Write-Host "Funnel started but the hostname couldn't be read." -ForegroundColor Yellow
    Write-Host "Check it with:  & '$tailscale' funnel status"
    exit 1
}

$share = "https://$host_name/s/$token"
$share | Set-Content -Path (Join-Path $root "current_link.txt") -Encoding utf8

Write-Host ""
Write-Host "  Permanent link - this address does not change:" -ForegroundColor Green
Write-Host "  $share" -ForegroundColor White
Write-Host ""
Write-Host "  Send it once. It survives restarts, sleep and reboots."
Write-Host "  The PC still has to be awake and online for it to answer."
Write-Host "  Take it down with:  .\static_link.ps1 -Stop"
Write-Host ""
