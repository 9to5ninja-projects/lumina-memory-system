#!/usr/bin/env python3
"""
DEPRECATED: Automatic version bumping is disabled.

This script was used for automatic version bumping but is now deprecated
in favor of manual versioning strategy. See VERSIONING_STRATEGY.md for details.

Automated versioning caused issues with:
- Incorrect chronological ordering
- Mixed versioning schemes
- Redundant changelog entries
- Wrong dates and metadata

Use manual git tagging instead:
  git tag -a v0.2.0-alpha -m "Release message"
  git push origin v0.2.0-alpha
"""
import subprocess
import sys

def main():
    print("⚠️  DEPRECATED: Automatic version bumping is disabled")
    print("📖 See VERSIONING_STRATEGY.md for manual versioning process")
    print("🏷️  Create tags manually: git tag -a v0.2.0-alpha -m 'message'")
    sys.exit(1)

# Legacy code preserved for reference but disabled
def get_latest_tag():
    """Get the latest git tag or default to v0.0.0"""
    try:
        result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "v0.0.0"

def bump_version(tag):
    """Bump the patch version of a semantic version tag"""
    if not tag.startswith('v'):
        tag = 'v' + tag
    
    # Remove 'v' prefix and split version parts
    version = tag[1:]
    parts = version.split('.')
    
    if len(parts) != 3:
        # Default to v0.0.1 if format is unexpected
        return "v0.0.1"
    
    # Increment patch version
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    new_patch = patch + 1
    
    return f"v{major}.{minor}.{new_patch}"

def create_tag(new_tag):
    """Create and return the new git tag"""
    subprocess.run(['git', 'tag', new_tag], check=True)
    return new_tag

if __name__ == "__main__":
    main()  # This will exit with error message
