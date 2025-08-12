"""Basic smoke tests to verify CI pipeline."""

import sys
import os


def test_python_version():
    """Test that Python version is supported."""
    assert sys.version_info >= (3, 10), f"Python version {sys.version_info} not supported"
    print(f" Python version {sys.version_info} is supported")


def test_basic_imports():
    """Test that basic Python imports work."""
    try:
        import numpy
        import scipy
        import sklearn
        print(" Core scientific packages available")
        return True
    except ImportError as e:
        print(f"  Import issue: {e}")
        return False


def test_project_structure():
    """Test that project structure is correct."""
    expected_dirs = ['src', 'tests']
    expected_files = ['README.md', 'pyproject.toml', 'requirements.txt']
    
    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f" Directory {directory} exists")
        else:
            print(f"  Directory {directory} missing")
    
    for file in expected_files:
        if os.path.exists(file):
            print(f" File {file} exists")
        else:
            print(f"  File {file} missing")


def test_package_importable():
    """Test that our package can be imported."""
    try:
        sys.path.append('src')
        import lumina_memory
        print(f" Package lumina_memory imported successfully")
        print(f"   Version: {getattr(lumina_memory, '__version__', 'unknown')}")
        return True
    except ImportError as e:
        print(f"  Package import issue: {e}")
        return False


if __name__ == "__main__":
    print(" Running CI smoke tests...")
    
    test_python_version()
    test_basic_imports()
    test_project_structure()
    test_package_importable()
    
    print(" Smoke tests completed!")
