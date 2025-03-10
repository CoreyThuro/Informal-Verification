"""
Coq verifier for validating translated proofs.
Handles verification of Coq proofs and error reporting.
"""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional, Union

class CoqVerifier:
    """
    Verifies Coq proofs and processes error messages.
    """
    
    def __init__(self):
        """Initialize the Coq verifier."""
        self.coq_executable = self._find_coq_executable()
    
    def verify_proof(self, proof_script: str, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify a Coq proof script.
        
        Args:
            proof_script: The Coq proof script to verify
            filename: Optional filename to save the script to
            
        Returns:
            Tuple of (verification success, error message if any)
        """
        if not self.coq_executable:
            return False, "Coq executable not found. Please ensure Coq is installed."
        
        # Create a temporary file if no filename provided
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(proof_script.encode())
            filename = temp_filename
        else:
            # Write to the specified file
            with open(filename, 'w') as f:
                f.write(proof_script)
        
        try:
            # Run coqc on the file
            result = subprocess.run(
                [self.coq_executable, filename],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Successfully verified
                return True, None
            else:
                # Verification failed
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Coq verification timed out."
        except Exception as e:
            return False, f"Error during verification: {str(e)}"
        finally:
            # Clean up temporary files
            self._cleanup_coq_artifacts(filename)
            
            # Remove temporary file if we created one
            if filename == temp_filename:
                try:
                    os.remove(temp_filename)
                except:
                    pass
    
    def interactive_session(self, proof_script: str) -> str:
        """
        Set up an interactive Coq session.
        
        Args:
            proof_script: The Coq proof script
            
        Returns:
            Path to the temporary file containing the script
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(proof_script.encode())
        
        coqtop_path = self._find_coqtop_executable()
        
        print(f"\nInteractive Coq Session:")
        print(f"1. A temporary file has been created at: {temp_filename}")
        
        if coqtop_path:
            print(f"2. You can run an interactive Coq session with:")
            print(f"   {coqtop_path} -l {temp_filename}")
            print(f"3. Inside coqtop, you can step through the proof with commands like:")
            print(f"   - 'Next.' to advance to the next step")
            print(f"   - 'Show.' to see the current goal")
            print(f"   - 'Undo.' to go back one step")
        else:
            print(f"2. Install Coq and use coqtop to interact with this file")
        
        return temp_filename
    
    def process_error(self, error_message: str) -> Dict[str, Any]:
        """
        Process a Coq error message into structured information.
        
        Args:
            error_message: The Coq error message
            
        Returns:
            Dictionary with structured error information
        """
        error_info = {
            "type": "unknown",
            "line": None,
            "col": None,
            "message": error_message,
            "suggestion": None
        }
        
        # Extract line and column information
        location_match = re.search(r'File ".*?", line (\d+), characters (\d+)-(\d+):', error_message)
        if location_match:
            error_info["line"] = int(location_match.group(1))
            error_info["col_start"] = int(location_match.group(2))
            error_info["col_end"] = int(location_match.group(3))
        
        # Categorize common error types
        if "Syntax error" in error_message:
            error_info["type"] = "syntax"
            error_info["suggestion"] = "Check for typos or incorrect Coq syntax"
            
        elif "The reference" in error_message and "was not found" in error_message:
            error_info["type"] = "missing_reference"
            ref_match = re.search(r'The reference ([^ ]+) was not found', error_message)
            if ref_match:
                error_info["reference"] = ref_match.group(1)
                error_info["suggestion"] = f"The identifier '{ref_match.group(1)}' is not defined. Check spelling or add necessary imports."
            
        elif "Unable to unify" in error_message:
            error_info["type"] = "type_mismatch"
            error_info["suggestion"] = "Type mismatch. Check the types of expressions."
            
        elif "No focused proof" in error_message:
            error_info["type"] = "no_proof"
            error_info["suggestion"] = "You're using proof tactics outside of a proof environment."
            
        elif "No such assumption" in error_message:
            error_info["type"] = "missing_assumption"
            assumption_match = re.search(r'No such assumption: ([^ ]+)', error_message)
            if assumption_match:
                error_info["assumption"] = assumption_match.group(1)
                error_info["suggestion"] = f"The assumption '{assumption_match.group(1)}' does not exist in the current context."
            
        elif "Attempt to save an incomplete proof" in error_message:
            error_info["type"] = "incomplete_proof"
            error_info["suggestion"] = "The proof is not complete. There are still goals to be proven."
        
        # Generate generic suggestions for unrecognized errors
        if error_info["type"] == "unknown" and not error_info["suggestion"]:
            error_info["suggestion"] = "Check Coq documentation or try simplifying your proof."
        
        return error_info
    
    def suggest_fixes(self, error_info: Dict[str, Any], proof_script: str) -> str:
        """
        Suggest fixes for common Coq errors.
        
        Args:
            error_info: Structured error information
            proof_script: The original proof script
            
        Returns:
            Modified proof script with suggested fixes
        """
        lines = proof_script.split('\n')
        error_type = error_info.get("type", "unknown")
        line_num = error_info.get("line")
        
        # No fixes if no line number
        if line_num is None or line_num <= 0 or line_num > len(lines):
            return proof_script
        
        # Adjust for 0-based indexing
        line_index = line_num - 1
        
        if error_type == "syntax":
            # For syntax errors, add a comment with the suggestion
            lines[line_index] = f"{lines[line_index]} (* {error_info['suggestion']} *)"
            
        elif error_type == "missing_reference":
            reference = error_info.get("reference", "")
            
            # Try to add common imports for standard libraries
            if reference in ["nat", "Nat"]:
                lines.insert(0, "Require Import Arith.")
            elif reference in ["Z", "Int"]:
                lines.insert(0, "Require Import ZArith.")
            elif reference in ["R", "Real"]:
                lines.insert(0, "Require Import Reals.")
            elif reference in ["ring", "Ring"]:
                lines.insert(0, "Require Import Ring.")
            
        elif error_type == "incomplete_proof":
            # Try to find where the proof ends
            for i in range(len(lines) - 1, -1, -1):
                if re.search(r'\bQed\b|\bAdmitted\b', lines[i]):
                    # Replace Qed with Admitted
                    lines[i] = lines[i].replace("Qed", "Admitted")
                    break
        
        return '\n'.join(lines)
    
    def _find_coq_executable(self) -> Optional[str]:
        """
        Find the Coq compiler executable.
        
        Returns:
            Path to coqc or None if not found
        """
        # Check in standard locations
        standard_paths = [
            "coqc",  # If in PATH
            "/usr/bin/coqc",
            "/usr/local/bin/coqc",
            "C:\\Program Files\\Coq\\bin\\coqc.exe",
            "C:\\Coq\\bin\\coqc.exe"
        ]
        
        for path in standard_paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return path
            except:
                continue
        
        return None
    
    def _find_coqtop_executable(self) -> Optional[str]:
        """
        Find the Coq interactive executable.
        
        Returns:
            Path to coqtop or None if not found
        """
        # Check in standard locations
        standard_paths = [
            "coqtop",  # If in PATH
            "/usr/bin/coqtop",
            "/usr/local/bin/coqtop",
            "C:\\Program Files\\Coq\\bin\\coqtop.exe",
            "C:\\Coq\\bin\\coqtop.exe"
        ]
        
        for path in standard_paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return path
            except:
                continue
        
        return None
    
    def _cleanup_coq_artifacts(self, filename: str) -> None:
        """
        Clean up Coq compilation artifacts.
        
        Args:
            filename: The Coq file path
        """
        # Get the base name without extension
        base = os.path.splitext(filename)[0]
        
        # Delete various Coq artifacts
        artifacts = [
            f"{base}.glob",
            f"{base}.vo",
            f"{base}.vok",
            f"{base}.vos"
        ]
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                try:
                    os.remove(artifact)
                except:
                    pass


# Standalone functions for use in other modules

def verify_coq_proof(proof_script: str, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Verify a Coq proof script.
    
    Args:
        proof_script: The Coq proof script to verify
        filename: Optional filename to save the script to
        
    Returns:
        Tuple of (verification success, error message if any)
    """
    verifier = CoqVerifier()
    return verifier.verify_proof(proof_script, filename)

def analyze_coq_error(error_message: str) -> Dict[str, Any]:
    """
    Analyze a Coq error message.
    
    Args:
        error_message: The Coq error message
        
    Returns:
        Dictionary with structured error information
    """
    verifier = CoqVerifier()
    return verifier.process_error(error_message)

def start_interactive_coq_session(proof_script: str) -> str:
    """
    Start an interactive Coq session.
    
    Args:
        proof_script: The Coq proof script
        
    Returns:
        Path to the temporary file
    """
    verifier = CoqVerifier()
    return verifier.interactive_session(proof_script)