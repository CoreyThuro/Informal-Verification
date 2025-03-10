#!/usr/bin/env python3
"""
Test script to verify that backends are properly registered.
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import backend registry
from backends.backend_interface import BackendRegistry

def test_backends():
    """Test if backends are properly registered."""
    
    # List available backends
    available_backends = BackendRegistry.list_backends()
    print(f"Available backends: {available_backends}")
    
    # Try to get the Coq backend
    if "coq" in available_backends:
        try:
            coq_backend = BackendRegistry.get_backend("coq")
            print(f"Successfully created Coq backend: {coq_backend.name}")
        except Exception as e:
            print(f"Error creating Coq backend: {e}")
    else:
        print("Coq backend is not registered")
    
    # Try to get the Lean backend
    if "lean" in available_backends:
        try:
            lean_backend = BackendRegistry.get_backend("lean")
            print(f"Successfully created Lean backend: {lean_backend.name}")
        except Exception as e:
            print(f"Error creating Lean backend: {e}")
    else:
        print("Lean backend is not registered")

if __name__ == "__main__":
    test_backends()