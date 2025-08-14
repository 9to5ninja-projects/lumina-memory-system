# PowerShell script to merge the current branch into one or more target branches and push
# Usage: powershell scripts/merge_to_branches.ps1 main xp_core notebook_dev

param([string[]]$branches)

$currentBranch = git rev-parse --abbrev-ref HEAD

foreach ($branch in $branches) {
    git checkout $branch
    git merge $currentBranch
    git push
}

git checkout $currentBranch
