"""
DEPRECATED: Automatic changelog updating is disabled.

This script was used for automatic changelog generation but is now deprecated
in favor of manual changelog management. See VERSIONING_STRATEGY.md for details.

Automated changelog generation caused issues with:
- Incorrect chronological ordering  
- Mixed versioning schemes
- Redundant and confusing entries
- Wrong dates and metadata

Update CHANGELOG.md manually following Keep a Changelog format:
https://keepachangelog.com/en/1.0.0/
"""
import os
import subprocess
import sys
import re

def main():
    print("⚠️  DEPRECATED: Automatic changelog updating is disabled")
    print("📖 See VERSIONING_STRATEGY.md for manual changelog process")
    print("✏️  Update CHANGELOG.md manually following Keep a Changelog format")
    sys.exit(1)

# Legacy code preserved for reference but disabled
def update_changelog(version_tag):
    """DEPRECATED - Do not use"""
    print("This function is deprecated. Update CHANGELOG.md manually.")
    return

if __name__ == "__main__":
    main()  # This will exit with error message