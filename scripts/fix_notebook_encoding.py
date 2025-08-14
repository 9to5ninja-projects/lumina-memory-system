#!/usr/bin/env python3
"""
Script to fix BOM and encoding issues in Jupyter notebooks.
"""

import os
import json
import codecs
from pathlib import Path

def remove_bom_and_fix_encoding(file_path):
    """Remove BOM and fix encoding for a single file."""
    try:
        # Read the file in binary mode to handle BOM
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try different encodings
        text_content = None
        encodings_to_try = ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1252', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                if encoding == 'utf-8-sig':
                    # Handle BOM manually for UTF-8
                    test_content = content
                    if test_content.startswith(codecs.BOM_UTF8):
                        test_content = test_content[len(codecs.BOM_UTF8):]
                        print(f"Removed UTF-8 BOM from {file_path}")
                    text_content = test_content.decode('utf-8')
                elif encoding.startswith('utf-16'):
                    # Handle UTF-16 BOMs
                    if content.startswith(codecs.BOM_UTF16_LE):
                        text_content = content[len(codecs.BOM_UTF16_LE):].decode('utf-16-le')
                        print(f"Removed UTF-16-LE BOM from {file_path}")
                    elif content.startswith(codecs.BOM_UTF16_BE):
                        text_content = content[len(codecs.BOM_UTF16_BE):].decode('utf-16-be')
                        print(f"Removed UTF-16-BE BOM from {file_path}")
                    elif content.startswith(codecs.BOM_UTF16):
                        text_content = content[len(codecs.BOM_UTF16):].decode('utf-16')
                        print(f"Removed UTF-16 BOM from {file_path}")
                    else:
                        text_content = content.decode(encoding)
                else:
                    text_content = content.decode(encoding)
                
                print(f"Successfully decoded {file_path} using {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if text_content is None:
            print(f"ERROR: Could not decode {file_path} with any known encoding")
            return False
        
        # Try to parse as JSON to validate it's a proper notebook
        try:
            nb_data = json.loads(text_content)
            # Ensure it has basic notebook structure
            if not isinstance(nb_data, dict) or 'cells' not in nb_data:
                print(f"WARNING: {file_path} doesn't appear to be a valid notebook")
                return False
        except json.JSONDecodeError as e:
            print(f"ERROR: {file_path} contains invalid JSON: {e}")
            return False
        
        # Write back without BOM, using UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(nb_data, f, indent=1, ensure_ascii=False)
        
        print(f"Fixed encoding for {file_path}")
        return True
        
    except Exception as e:
        print(f"ERROR processing {file_path}: {e}")
        return False

def fix_notebooks_in_directory(directory):
    """Fix all .ipynb files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return
    
    notebook_files = list(directory.glob("*.ipynb"))
    if not notebook_files:
        print(f"No notebook files found in {directory}")
        return
    
    print(f"Found {len(notebook_files)} notebook files in {directory}")
    
    fixed_count = 0
    for notebook_file in notebook_files:
        if remove_bom_and_fix_encoding(notebook_file):
            fixed_count += 1
    
    print(f"Successfully fixed {fixed_count} out of {len(notebook_files)} notebook files")

if __name__ == "__main__":
    # Fix notebooks in the notebooks directory
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    fix_notebooks_in_directory(notebooks_dir)
