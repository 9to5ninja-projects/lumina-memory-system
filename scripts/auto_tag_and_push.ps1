# PowerShell script for automated version tagging, changelog update, and push
param(
    [string]$version_tag
)

if (-not $version_tag) {
    Write-Host "Usage: .\auto_tag_and_push.ps1 <version_tag>"
    exit 1
}

# Create the tag
& git tag $version_tag

# Update and commit changelog
& python scripts/update_changelog.py $version_tag

# Push all changes and the tag
& git push
& git push origin $version_tag

Write-Host "Pushed changes and tag $version_tag with updated changelog."
