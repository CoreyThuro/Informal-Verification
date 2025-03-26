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
    # Save a permanent copy for inspection
    with open('/tmp/debug_coq_proof.v', 'w') as debug_file:
        proof_script = proof_script.replace('\r\n', '\n')  # Normalize line endings
        debug_file.write(proof_script)
        print(f"Debug file saved to /tmp/debug_coq_proof.v")
    
    # Create a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
        file_path = temp_file.name
        print(f"Temporary file created at: {file_path}")
    
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

  
    
