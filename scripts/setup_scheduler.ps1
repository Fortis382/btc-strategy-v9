# BTC Strategy v9.4 - Task Scheduler Setup

$action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\ultimate\btc-v9\scripts\auto_commit.ps1"

$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName "BTC_Strategy_AutoCommit" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "BTC Strategy v9.4 자동 커밋 (30분마다)" `
    -Force

Write-Host "✓ 작업 스케줄러 등록 완료" -ForegroundColor Green
Write-Host "  작업 이름: BTC_Strategy_AutoCommit" -ForegroundColor Cyan
Write-Host "  실행 주기: 30분마다" -ForegroundColor Cyan
