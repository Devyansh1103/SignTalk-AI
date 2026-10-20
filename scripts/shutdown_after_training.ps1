# Shuts the PC down once train_stgcn.py and evaluate_stgcn.py have both finished.
# Usage (detached):  powershell -ExecutionPolicy Bypass -File scripts\shutdown_after_training.ps1
# Cancel a pending shutdown:  shutdown /a     Stop this watcher: close its PowerShell process.

$log = Join-Path $PSScriptRoot "..\logs\auto_shutdown.log"
function Note($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content -Path $log }

function Busy {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -match 'train_stgcn\.py|evaluate_stgcn\.py' }
    return [bool]$p
}

Note "watcher started"
$seen = $false
$idle = 0
$waited = 0
while ($true) {
    if (Busy) {
        if (-not $seen) { Note "training/evaluation detected" }
        $seen = $true
        $idle = 0
    } elseif ($seen) {
        $idle += 30
        # 3 min idle covers the short gap between training and the test evaluation
        if ($idle -ge 180) { break }
    } else {
        $waited += 30
        if ($waited -ge 900) { Note "no training seen in 15 min, exiting without shutdown"; exit 0 }
    }
    Start-Sleep -Seconds 30
}

Note "training + evaluation finished, shutting down in 5 min (cancel: shutdown /a)"
shutdown /s /t 300 /c "SignTalk AI training finished. PC will shut down in 5 minutes. Run 'shutdown /a' to cancel."
