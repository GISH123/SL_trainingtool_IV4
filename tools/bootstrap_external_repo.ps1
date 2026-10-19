param(
    [string]$RepoName = "SL_trainingtool_IV4",
    [string]$Owner = "GISH123"
)
$ErrorActionPreference = "Stop"

if (!(Test-Path ".git")) {
    git init -b main
}

git status --short --branch

Write-Host "`nCreate the GitHub repository as PRIVATE first:"
Write-Host "  $Owner/$RepoName"
Write-Host "`nThen run:"
Write-Host "  git remote add origin https://github.com/$Owner/$RepoName.git"
Write-Host "  git push -u origin main"

