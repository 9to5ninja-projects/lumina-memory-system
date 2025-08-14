import os
import subprocess
import sys

def update_changelog(version_tag):
    # Get last tag before this one
    last_tag = subprocess.getoutput("git describe --tags --abbrev=0 @^")
    changes = subprocess.getoutput(f"git log --oneline --no-decorate {last_tag}..@")
    
    # Filter out recursive changelog update commits
    filtered_changes = []
    if changes.strip():
        for line in changes.split('\n'):
            if line.strip() and not 'Update changelog for' in line:
                filtered_changes.append(f"- {line}")
    
    with open("CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Version {version_tag}\n")
        if filtered_changes:
            f.write('\n'.join(filtered_changes) + "\n")
        else:
            f.write("- No meaningful changes since previous version.\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_changelog.py <version_tag>")
        sys.exit(1)
    version_tag = sys.argv[1]
    update_changelog(version_tag)
    # Don't automatically push to prevent recursive changelog updates
    os.system(f"git add \"CHANGELOG.md\" && git commit -m \"Update changelog for {version_tag}\"")
