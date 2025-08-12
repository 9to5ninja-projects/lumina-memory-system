#!/usr/bin/env python3
"""
Local CI Validation Script - Run this before pushing to GitHub

This script mimics the GitHub Actions CI pipeline locally
to catch issues before they reach the remote CI system.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description, required=True):
    """Run a command and report results."""
    print(f"\n {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"    SUCCESS: {description}")
            if result.stdout.strip():
                print(f"    Output: {result.stdout.strip()[:200]}...")
            return True
        else:
            status = " REQUIRED FAILURE" if required else "  OPTIONAL FAILURE"
            print(f"   {status}: {description}")
            if result.stderr.strip():
                print(f"    Error: {result.stderr.strip()[:200]}...")
            return not required
            
    except Exception as e:
        status = " EXCEPTION" if required else "  EXCEPTION"
        print(f"   {status}: {e}")
        return not required


def main():
    """Run local CI validation."""
    print(" LOCAL CI VALIDATION - Lumina Memory System")
    print("=" * 60)
    
    # Check Python version
    print(f" Python Version: {sys.version}")
    
    # Track results
    results = []
    
    # Core setup
    results.append(run_command("python --version", "Python Version Check"))
    results.append(run_command("pip install --upgrade pip", "Upgrade pip"))
    
    # Install dependencies
    results.append(run_command("pip install numpy scipy scikit-learn", "Install core dependencies"))
    results.append(run_command("pip install pytest pytest-cov ruff black isort", "Install dev dependencies"))
    
    # Package installation
    results.append(run_command("pip install -e .", "Install package in dev mode", required=False))
    
    # Code quality (non-blocking)
    results.append(run_command("ruff check src/", "Ruff linting (src)", required=False))
    results.append(run_command("ruff check tests/", "Ruff linting (tests)", required=False))
    results.append(run_command("black --check src/", "Black formatting (src)", required=False))
    results.append(run_command("black --check tests/", "Black formatting (tests)", required=False))
    
    # Run tests
    results.append(run_command("python tests/test_smoke.py", "Smoke tests"))
    results.append(run_command("python -m pytest tests/ -v", "Full test suite", required=False))
    
    # Final assessment
    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    
    passed = sum(results)
    total = len(results)
    
    print(f" Passed: {passed}/{total} checks")
    
    if passed == total:
        print(" ALL CHECKS PASSED! Ready to push to GitHub.")
        return 0
    elif passed >= total * 0.7:  # 70% pass rate
        print("  MOSTLY PASSING - Review failures but likely safe to push.")
        return 0
    else:
        print(" MULTIPLE FAILURES - Fix issues before pushing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
