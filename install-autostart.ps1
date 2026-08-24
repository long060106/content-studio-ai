<#
    install-autostart.ps1 - start Content Studio automatically when you log in.

        .\install-autostart.ps1              # turn it on
        .\install-autostart.ps1 -Remove      # turn it off

    Registers a Windows scheduled task that runs share_link.ps1 at logon, so
    the studio and its public link come up on their own after a restart. No
    administrator rights are needed: the task runs as you, not as the system.

    What this does NOT fix: the free tunnel address changes every time it
    starts, so anyone you shared the old link with still needs the new one.
    The current link is always written to current_link.txt.
#>

param([switch]$Remove)

$ErrorActionPreference = "Stop"
$taskName = "ContentStudioLink"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $root "share_link.ps1"

if ($Remove) {
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Auto-start removed. Nothing will run at logon any more." -ForegroundColor Yellow
        Write-Host "The studio is untouched - start it by hand with .\share_link.ps1"
    } catch {
        Write-Host "Nothing to remove; auto-start was not installed." -ForegroundColor Yellow
    }
    exit 0
}

if (-not (Test-Path $target)) {
    Write-Host "Cannot find share_link.ps1 next to this script." -ForegroundColor Red
    Write-Host "Expected it at: $target"
    exit 1
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
          -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$target`""

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# StartWhenAvailable catches the case where the PC was asleep at logon time.
# No execution time limit, because the studio is meant to keep running.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -StartWhenAvailable `
            -ExecutionTimeLimit ([TimeSpan]::Zero)

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings -Force `
        -Description "Starts Content Studio and its public link at logon" | Out-Null
} catch {
    Write-Host "Could not register the task: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "The task did not register." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  Auto-start installed." -ForegroundColor Green
Write-Host ""
Write-Host "  From now on, logging in starts:"
Write-Host "    - the studio at http://127.0.0.1:8420  (no key needed from this PC)"
Write-Host "    - the public link, written to current_link.txt"
Write-Host ""
Write-Host "  Test it now without restarting:"
Write-Host "    Start-ScheduledTask -TaskName $taskName"
Write-Host ""
Write-Host "  Turn it off later:"
Write-Host "    .\install-autostart.ps1 -Remove"
Write-Host ""
