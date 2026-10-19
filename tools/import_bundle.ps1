param(
    [Parameter(Mandatory=$true)][string]$BundlePath,
    [string]$Branch = "main",
    [switch]$Apply
)
$ErrorActionPreference = "Stop"
$repoRoot = (git rev-parse --show-toplevel).Trim()
Set-Location $repoRoot

if (!(Test-Path $BundlePath)) { throw "Bundle not found: $BundlePath" }

if (git status --porcelain) {
    Write-Host "ERROR: internal working tree is not clean." -ForegroundColor Red
    git status --short --branch
    exit 1
}

git bundle verify $BundlePath
$remoteRef = "refs/remotes/bundle/$Branch"
git fetch $BundlePath "$Branch`:$remoteRef"

Write-Host "`nCurrent:"
git log -1 --oneline HEAD
Write-Host "`nIncoming:"
git log -1 --oneline $remoteRef
Write-Host "`nCommits to apply:"
git log --oneline --decorate "HEAD..$remoteRef"
Write-Host "`nDiff summary:"
git diff --stat "HEAD..$remoteRef"

if (-not $Apply) {
    Write-Host "`nDRY RUN ONLY. Re-run with -Apply after review." -ForegroundColor Yellow
    exit 0
}

git merge --ff-only $remoteRef
if ($LASTEXITCODE -ne 0) {
    Write-Host "Fast-forward failed. Internal and external have diverged." -ForegroundColor Red
    exit 1
}

Write-Host "`nInternal repo updated successfully." -ForegroundColor Green
git status --short --branch
git log -1 --oneline
