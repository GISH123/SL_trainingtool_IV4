param(
    [string]$Branch = "main",
    [string]$OutDir = ".\bundles"
)
$ErrorActionPreference = "Stop"
$repoRoot = (git rev-parse --show-toplevel).Trim()
Set-Location $repoRoot

if (git status --porcelain) {
    Write-Host "ERROR: working tree is not clean." -ForegroundColor Red
    git status --short --branch
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$sha = (git rev-parse --short=10 $Branch).Trim()
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$repoName = Split-Path $repoRoot -Leaf
$outFile = Join-Path $OutDir "${repoName}_${Branch}_${stamp}_${sha}.bundle"

git bundle create $outFile $Branch --tags
git bundle verify $outFile

Write-Host "Bundle created: $outFile" -ForegroundColor Green
Write-Host "Commit: $(git rev-parse $Branch)"
