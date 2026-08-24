<#
    share_link.ps1 - put Content Studio on a public HTTPS link.

        .\share_link.ps1            # start the server + tunnel, print the link
        .\share_link.ps1 -Stop      # take the link down

    Starts two things: webapp.py on 127.0.0.1:8420, and a Cloudflare tunnel
    pointing at it. The tunnel gives out a free trycloudflare.com address that
    works from anywhere.

    Two things worth knowing before handing the link to anyone:

      * This PC does all the work. If it sleeps or shuts down, the link dies.
      * The free URL is different every time the tunnel starts, so anyone you
        shared the old one with needs the new one.

    Access is controlled by CONTENT_STUDIO_LINK_TOKEN in .env, which is carried
    in the link itself. Whoever you send it to just clicks and starts working -
    no username, no password, nothing to remember - while the bare address
    without the key is useless to anyone who stumbles onto it. The script won't
    open a tunnel without a key set, because an unprotected link means strangers
    spending your API credits.
#>

param([switch]$Stop)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cloudflared = "$env:LOCALAPPDATA\cloudflared\cloudflared.exe"
$log = "$env:LOCALAPPDATA\cloudflared\tunnel.log"

function Stop-Everything {
    Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*webapp.py*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

if ($Stop) {
    Stop-Everything
    Write-Host "Link is down. The studio is no longer reachable from outside." -ForegroundColor Yellow
    exit 0
}

# --- refuse to expose an unprotected server ------------------------------
$envFile = Join-Path $root ".env"
$token = $null
if (Test-Path $envFile) {
    $line = Select-String -Path $envFile -Pattern '^CONTENT_STUDIO_LINK_TOKEN=(.+)$' |
            Select-Object -First 1
    if ($line) { $token = $line.Matches[0].Groups[1].Value.Trim() }
}
if (-not $token) {
    Write-Host "No CONTENT_STUDIO_LINK_TOKEN in .env." -ForegroundColor Red
    Write-Host "Anyone with the link could spend your API credits. Add a key first."
    exit 1
}

if (-not (Test-Path $cloudflared)) {
    Write-Host "cloudflared not found at $cloudflared" -ForegroundColor Red
    Write-Host "Download it from https://github.com/cloudflare/cloudflared/releases/latest"
    exit 1
}

Stop-Everything
Start-Sleep -Seconds 2

# --- the studio itself ---------------------------------------------------
$env:CONTENT_STUDIO_UI_NO_BROWSER = "1"
Start-Process -FilePath "$root\venv\Scripts\python.exe" `
              -ArgumentList "-u", "webapp.py" `
              -WorkingDirectory $root -WindowStyle Hidden `
              -RedirectStandardOutput "$env:LOCALAPPDATA\cloudflared\webapp.out" `
              -RedirectStandardError  "$env:LOCALAPPDATA\cloudflared\webapp.err"

# Wait for it to actually accept connections. Starting the tunnel first just
# produces a link that answers 502 until the server catches up.
$up = $false
foreach ($try in 1..15) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort 8420 -State Listen -ErrorAction SilentlyContinue) {
        $up = $true; break
    }
}
if (-not $up) {
    Write-Host "The studio didn't start. Last error output:" -ForegroundColor Red
    Get-Content "$env:LOCALAPPDATA\cloudflared\webapp.err" -Tail 20 -ErrorAction SilentlyContinue
    exit 1
}

# --- the tunnel ----------------------------------------------------------
# 127.0.0.1 rather than localhost: on Windows "localhost" resolves to ::1
# first, the server listens on IPv4 only, and every request 502s.
Remove-Item $log -ErrorAction SilentlyContinue
# --protocol http2: cloudflared prefers QUIC, which is UDP on port 7844, and
# this network drops it. The symptom is not an error at startup - the tunnel
# registers, prints a perfectly good URL, and then loops forever on "failed to
# dial to edge with quic: timeout: no recent network activity" while every
# visitor gets a Cloudflare 530. http2 rides TCP 443, which goes through
# anywhere a browser does.
Start-Process -FilePath $cloudflared `
              -ArgumentList "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:8420" `
              -RedirectStandardError $log `
              -RedirectStandardOutput "$env:LOCALAPPDATA\cloudflared\tunnel.out" `
              -WindowStyle Hidden

$url = $null
foreach ($try in 1..30) {
    Start-Sleep -Seconds 1
    if (Test-Path $log) {
        $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches
        if ($m) { $url = $m.Matches[0].Value; break }
    }
}

if (-not $url) {
    Write-Host "Tunnel didn't come up. Last lines:" -ForegroundColor Red
    Get-Content $log -Tail 15 -ErrorAction SilentlyContinue
    exit 1
}

# /s/<key> rather than ?k=<key>: a path survives being shared. Query strings
# get trimmed by some link previews, and a trailing ?k=... is easy to lose when
# a long URL wraps onto two lines in a chat app.
$share = "$url/s/$token"
$share | Set-Content -Path (Join-Path $root "current_link.txt") -Encoding utf8

Write-Host ""
Write-Host "  Send this link - clicking it is all she needs to do:" -ForegroundColor Green
Write-Host "  $share" -ForegroundColor White
Write-Host ""
Write-Host "  (also saved to current_link.txt)"
Write-Host "  Keep this PC awake, or the link stops working."
Write-Host "  Take it down with:  .\share_link.ps1 -Stop"
Write-Host ""
