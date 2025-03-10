"""
Backend package for theorem provers.
"""

# Import backend interface
from backends.backend_interface import BackendRegistry, ProverBackend

# Import all backends to ensure they're registered
# Import Coq backend
try:
    from backends.coq.coq_formatter import CoqFormatter
except ImportError as e:
    print(f"Warning: Coq backend could not be imported: {e}")

# Import Lean backend
try:
    from backends.lean.lean_formatter import LeanFormatter
except ImportError as e:
    print(f"Warning: Lean backend could not be imported: {e}")

# List available backends
available_backends = BackendRegistry.list_backends()