#!/usr/bin/env python3
"""Simple version bumping script that increments patch version"""
import subprocess
import sys

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
    latest_tag = get_latest_tag()
    new_tag = bump_version(latest_tag)
    created_tag = create_tag(new_tag)
    print(created_tag)
