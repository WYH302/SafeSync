# One-click script to publish SafeSync to GitHub (https://github.com/WYH302/SafeSync)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Please sign in to GitHub CLI first:" -ForegroundColor Cyan
    gh auth login
}
gh repo view WYH302/SafeSync 2>$null
if ($LASTEXITCODE -ne 0) {
    gh repo create WYH302/SafeSync --public --source=. --remote=origin --push --description "Asymmetric Space-Ground State Synchronization and Evidence-Preserving Co-Construction for Repeated Earth Observation"
} else {
    git push -u origin main
}
