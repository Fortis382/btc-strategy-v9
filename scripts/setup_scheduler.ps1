# scripts\setup_scheduler.ps1
# BTC Strategy v9.4 - Task Scheduler Setup via schtasks.exe (버그 우회판)
$ErrorActionPreference = 'Stop'

try {
    $taskName = "BTC_Strategy_AutoCommit"
    $workDir  = "C:\ultimate\btc-v9"
    $script   = Join-Path $workDir "scripts\auto_commit.ps1"

    if (-not (Test-Path $script)) {
        throw "auto_commit.ps1가 없음: $script"
    }

    # 기존 태스크 제거(있으면)
    schtasks /Query /TN $taskName > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        schtasks /Delete /TN $taskName /F | Out-Null
        Start-Sleep -Milliseconds 200
    }

    # 시작 시간: 지금+1분 (HH:mm, 24h)
    $st = (Get-Date).AddMinutes(1).ToString('HH:mm')

    # 30분마다 반복(무기한). Duration 필드 자체를 쓰지 않음 → XML P9999… 문제 회피
    $tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$script`""

    $args = @(
        "/Create",
        "/TN", $taskName,
        "/SC", "MINUTE",
        "/MO", "30",
        "/ST", $st,
        "/TR", $tr,
        "/RU", $env:USERNAME,
        "/RL", "HIGHEST",
        "/F"
    )

    $null = schtasks @args

    # 첫 실행 트리거
    schtasks /Run /TN $taskName | Out-Null

    Write-Host "✓ 작업 스케줄러 등록 완료" -ForegroundColor Green
    Write-Host ("  작업 이름: {0}" -f $taskName)
    Write-Host ("  첫 실행 : {0}" -f $st)
    Write-Host ("  실행 주기: 30분마다 (schtasks.exe 경로, Duration 없음)")
}
catch {
    Write-Host "✗ 작업 스케줄러 등록 실패" -ForegroundColor Red
    Write-Host $_.Exception.Message
    if ($_.ScriptStackTrace) { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    exit 1
}
