"""
Coq verification module.
"""

import os
import subprocess
import tempfile
from typing import Tuple, Optional

def verify_coq_proof(proof_script: str) -> Tuple[bool, Optional[str]]:
    """
    Verify a Coq proof script.
    
    Args:
        proof_script: The Coq proof script
        
    Returns:
        Tuple of (success, error_message)
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
        temp_file.write(proof_script.encode())
        file_path = temp_file.name
    
    try:
        # Run Coq compiler on the file
        result = subprocess.run(
            ["coqc", file_path],
            capture_output=True,
            text=True,
            timeout=10  # Timeout after 10 seconds
        )
        
        # Check if compilation was successful
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Verification timed out."
    except FileNotFoundError:
        return False, "Coq compiler not found. Please make sure Coq is installed."
    except Exception as e:
        return False, f"Error during verification: {str(e)}"
    finally:
        # Clean up the temporary file
        try:
            os.unlink(file_path)
        except:
            pass