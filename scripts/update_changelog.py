import os
import subprocess
import sys
import re

def update_changelog(version_tag):
    # Read existing changelog to check if this version already exists
    changelog_path = "CHANGELOG.md"
    existing_changelog = ""
    if os.path.exists(changelog_path):
        with open(changelog_path, "r", encoding="utf-8") as f:
            existing_changelog = f.read()
    
    # Check if this version tag already exists in changelog
    if f"## Version {version_tag}" in existing_changelog:
        print(f"Version {version_tag} already exists in changelog. Skipping duplicate entry.")
        return
    
    # Get commits since last tag (excluding this version if it exists)
    try:
        # Get the last tag that's different from current version
        all_tags = subprocess.getoutput("git tag --sort=-version:refname").split('\n')
        last_different_tag = None
        for tag in all_tags:
            if tag.strip() and tag.strip() != version_tag:
                last_different_tag = tag.strip()
                break
        
        if last_different_tag:
            changes = subprocess.getoutput(f"git log --oneline --no-decorate {last_different_tag}..HEAD")
        else:
            # If no previous tags, get recent commits
            changes = subprocess.getoutput("git log --oneline --no-decorate -10")
        
    except Exception:
        # Fallback: get recent commits
        changes = subprocess.getoutput("git log --oneline --no-decorate -5")
    
    # Filter out recursive changelog update commits and format
    filtered_changes = []
    if changes.strip():
        for line in changes.split('\n'):
            if line.strip() and not 'Update changelog for' in line:
                # Remove any leading commit hash artifacts
                clean_line = re.sub(r'^[a-f0-9]+\s*', '', line.strip())
                if clean_line:
                    filtered_changes.append(f"- {line.strip()}")
    
    # Only add entry if we have meaningful changes
    if filtered_changes:
        # Prepend new version to changelog (most recent first)
        new_entry = f"## Version {version_tag}\n" + '\n'.join(filtered_changes) + "\n\n"
        
        if existing_changelog:
            # Insert after the first line (# Changelog)
            lines = existing_changelog.split('\n')
            if lines and lines[0].startswith('# Changelog'):
                updated_changelog = lines[0] + '\n\n' + new_entry + '\n'.join(lines[1:])
            else:
                updated_changelog = new_entry + existing_changelog
        else:
            updated_changelog = "# Changelog\n\n" + new_entry
        
        with open(changelog_path, "w", encoding="utf-8") as f:
            f.write(updated_changelog)
            
        print(f"Added changelog entry for {version_tag}")
    else:
        print(f"No meaningful changes found for {version_tag}. Skipping changelog update.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_changelog.py <version_tag>")
        sys.exit(1)
    version_tag = sys.argv[1]
    update_changelog(version_tag)
    # Only commit if changelog was actually modified
    if os.system("git diff --quiet CHANGELOG.md") != 0:
        os.system(f"git add \"CHANGELOG.md\" && git commit -m \"Update changelog for {version_tag}\"")
    else:
        print("No changelog changes to commit.")