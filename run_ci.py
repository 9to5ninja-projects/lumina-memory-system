#!/usr/bin/env python3
"""
Local CI script to run the same checks as GitHub Actions.
Usage: python run_ci.py
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n {description}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f" {description} passed")
        if result.stdout.strip():
            print(result.stdout)
        return True
    else:
        print(f" {description} failed")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return False

def main():
    """Run all CI checks."""
    print(" Running local CI checks...")
    
    checks = [
        ("ruff format --check .", "Ruff format check"),
        ("black --check .", "Black format check"),
        ("isort --check .", "isort import check"),
        ("ruff check .", "Ruff linting"),
        ("pytest -q --tb=short", "Pytest tests"),
    ]
    
    failed_checks = []
    
    for cmd, description in checks:
        if not run_command(cmd, description):
            failed_checks.append(description)
    
    print(f"\n{'='*60}")
    if failed_checks:
        print(f" CI checks failed: {len(failed_checks)}/{len(checks)}")
        for check in failed_checks:
            print(f"  - {check}")
        print(f"\nTo fix formatting issues, run:")
        print(f"  ruff format .")
        print(f"  black .")
        print(f"  isort .")
        return 1
    else:
        print(f" All CI checks passed: {len(checks)}/{len(checks)}")
        print(" Ready for commit and push!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
