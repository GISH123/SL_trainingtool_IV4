param(
    [Parameter(Mandatory=$true)][string]$BundlePath,
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $BundlePath)) {
    throw "Bundle not found: $BundlePath"
}

if (Test-Path $Destination) {
    throw "Destination already exists: $Destination"
}

git bundle verify $BundlePath
if ($LASTEXITCODE -ne 0) {
    throw "Bundle verification failed."
}

git clone -b $Branch $BundlePath $Destination
if ($LASTEXITCODE -ne 0) {
    throw "git clone from bundle failed."
}

Set-Location $Destination

# The bundle path becomes "origin" when cloning from a bundle.
# Remove it because the internal/offline repo intentionally has no network remote.
git remote remove origin 2>$null

Write-Host ""
Write-Host "Internal baseline created successfully." -ForegroundColor Green
git status --short --branch
git log -1 --oneline

Write-Host ""
Write-Host "Now copy ONLY local runtime/data artifacts from the old internal folder"
Write-Host "(datasets, checkpoints, pretrained/models, reports, snapshots, etc.)."
Write-Host "Those paths are ignored by Git and will not affect source synchronization."
