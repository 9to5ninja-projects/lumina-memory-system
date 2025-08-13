import os
import subprocess
import sys

def update_changelog(version_tag):
    # Get last tag before this one
    last_tag = subprocess.getoutput("git describe --tags --abbrev=0 @^")
    changes = subprocess.getoutput(f"git log --oneline --no-decorate {last_tag}..@")
    with open("CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Version {version_tag}\n")
        if changes.strip():
            f.write(changes + "\n")
        else:
            f.write("No changes since previous version.\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_changelog.py <version_tag>")
        sys.exit(1)
    version_tag = sys.argv[1]
    update_changelog(version_tag)
    os.system(f"git add CHANGELOG.md && git commit -m 'Update changelog for {version_tag}' && git push")
