param([string]$RemoteRef = "origin/main")
$ErrorActionPreference = "Stop"
$repoRoot = (git rev-parse --show-toplevel).Trim()
Set-Location $repoRoot

git status --short --branch
Write-Host "`nHEAD:"
git log -1 --oneline HEAD

if (git rev-parse --verify $RemoteRef 2>$null) {
    Write-Host "`n$RemoteRef:"
    git log -1 --oneline $RemoteRef
    $ahead = (git rev-list --count "$RemoteRef..HEAD").Trim()
    $behind = (git rev-list --count "HEAD..$RemoteRef").Trim()
    Write-Host "`nAhead : $ahead"
    Write-Host "Behind: $behind"
    if ($ahead -eq "0" -and $behind -eq "0") {
        Write-Host "SYNC: exact same commit history." -ForegroundColor Green
    } else {
        Write-Host "SYNC: not identical." -ForegroundColor Yellow
    }
}
