# One-click script to publish SafeSync to GitHub (https://github.com/WYH302/SafeSync)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

& gh auth status 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Please sign in to GitHub CLI first:" -ForegroundColor Cyan
    & gh auth login -h github.com -p https -w
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& gh repo view WYH302/SafeSync 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    & gh repo create WYH302/SafeSync --public --source=. --remote=origin --push --description "Zero-Footprint Certified State Synchronization and Open-World Semantic Routing for Bandwidth-Constrained LEO Earth Observation"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    $remotes = & git remote
    if ($remotes -notcontains "origin") {
        & git remote add origin https://github.com/WYH302/SafeSync.git
    }
    & git push -u origin main
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "Successfully published https://github.com/WYH302/SafeSync" -ForegroundColor Green
