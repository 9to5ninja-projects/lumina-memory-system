# PowerShell script to merge notebook_setup into all target branches and tag each merge
$branches = @(
    "main",
    "xp_core",
    "notebook_dev"
    # Add more branches as needed
)

foreach ($branch in $branches) {
    git checkout $branch
    git merge notebook_setup --strategy=recursive -X theirs
    if ($LASTEXITCODE -eq 0) {
        $tagName = "auto-notebook_setup-$(Get-Date -Format 'yyyyMMdd')-$branch"
        git tag $tagName
        git push
        git push origin $tagName
        Write-Host "Merged notebook_setup into $branch, tagged as $tagName, and pushed."
    } else {
        Write-Host "Merge conflict in $branch. Please resolve manually."
        break
    }
}
