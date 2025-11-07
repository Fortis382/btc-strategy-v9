# BTC Strategy v9.4 - Auto Commit Script
# 30분마다 자동 커밋 + 푸시

$ErrorActionPreference = "SilentlyContinue"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# 작업 디렉토리로 이동
Set-Location -Path "C:\ultimate\btc-v9"

# 변경사항 확인
$status = git status --porcelain

if ($status) {
    Write-Host "[$timestamp] 변경사항 감지" -ForegroundColor Yellow
    
    # Stage all
    git add -A
    
    # Commit
    $commitMsg = "Auto-save: $timestamp"
    git commit -m "$commitMsg"
    
    # Push
    git push origin main
    
    Write-Host "[$timestamp] ✓ GitHub에 푸시 완료" -ForegroundColor Green
    
    # 로그 저장
    Add-Content -Path "logs\system\git_auto_commit.log" -Value "[$timestamp] Success: $commitMsg"
} else {
    Write-Host "[$timestamp] 변경사항 없음" -ForegroundColor Gray
    Add-Content -Path "logs\system\git_auto_commit.log" -Value "[$timestamp] No changes"
}
