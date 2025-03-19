"""
Lean verifier for validating translated proofs.
Handles verification of Lean proofs and error reporting.
"""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional, Union

class LeanVerifier:
    """
    Verifies Lean proofs and processes error messages.
    """
    
    def __init__(self):
        """Initialize the Lean verifier."""
        self.lean_executable = self._find_lean_executable()
    
    def verify_proof(self, proof_script: str, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify a Lean proof script.
        
        Args:
            proof_script: The Lean proof script to verify
            filename: Optional filename to save the script to
            
        Returns:
            Tuple of (verification success, error message if any)
        """
        if not self.lean_executable:
            return False, "Lean executable not found. Please ensure Lean is installed."
        
        # Create a temporary file if no filename provided
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix='.lean', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(proof_script.encode())
            filename = temp_filename
        else:
            # Write to the specified file
            with open(filename, 'w') as f:
                f.write(proof_script)
        
        try:
            # Run lean on the file
            result = subprocess.run(
                [self.lean_executable, filename],
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
            return False, "Lean verification timed out."
        except Exception as e:
            return False, f"Error during verification: {str(e)}"
        finally:
            # Remove temporary file if we created one
            if filename == temp_filename:
                try:
                    os.remove(temp_filename)
                except:
                    pass
    
    def interactive_session(self, proof_script: str) -> str:
        """
        Set up an interactive Lean session.
        
        Args:
            proof_script: The Lean proof script
            
        Returns:
            Path to the temporary file containing the script
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.lean', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(proof_script.encode())
        
        print(f"\nInteractive Lean Session:")
        print(f"1. A temporary file has been created at: {temp_filename}")
        print(f"2. To work with this file interactively, open it in VS Code with the Lean extension installed.")
        print(f"3. You can also run Lean from the command line with:")
        print(f"   lean {temp_filename}")
        
        return temp_filename
    
    def process_error(self, error_message: str) -> Dict[str, Any]:
        """
        Process a Lean error message into structured information.
        
        Args:
            error_message: The Lean error message
            
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
        location_match = re.search(r'(\w+\.lean):(\d+):(\d+):', error_message)
        if location_match:
            error_info["file"] = location_match.group(1)
            error_info["line"] = int(location_match.group(2))
            error_info["col"] = int(location_match.group(3))
        
        # Categorize common error types
        if "unknown identifier" in error_message:
            error_info["type"] = "unknown_identifier"
            # Fixed regex pattern with proper escaping
            identifier_match = re.search(r'unknown identifier [\'"`]([^\'"]+)[\'"`]', error_message)
            if identifier_match:
                error_info["identifier"] = identifier_match.group(1)
                error_info["suggestion"] = f"The identifier '{identifier_match.group(1)}' is not defined. Check spelling or add necessary imports."
        
        elif "type mismatch" in error_message:
            error_info["type"] = "type_mismatch"
            error_info["suggestion"] = "Type mismatch. Check the types of expressions and ensure they match."
            
        elif "tactic failed" in error_message:
            error_info["type"] = "tactic_failure"
            error_info["suggestion"] = "The tactic failed. Try a different approach or more specific tactics."
            
        elif "unexpected token" in error_message:
            error_info["type"] = "syntax_error"
            error_info["suggestion"] = "Syntax error. Check for typos or incorrect Lean syntax."
            
        elif "declaration has metavariables" in error_message:
            error_info["type"] = "incomplete_proof"
            error_info["suggestion"] = "The proof is incomplete. There are still goals to be solved."
            
        elif "invalid expression" in error_message:
            error_info["type"] = "invalid_expression"
            error_info["suggestion"] = "The expression is invalid. Check the syntax and ensure it follows Lean rules."
            
        elif "unknown universe" in error_message:
            error_info["type"] = "universe_error"
            error_info["suggestion"] = "Universe level error. This might be related to Type constraints."
        
        # Generate generic suggestions for unrecognized errors
        if error_info["type"] == "unknown" and not error_info["suggestion"]:
            error_info["suggestion"] = "Check Lean documentation or try simplifying your proof."
        
        return error_info
    
    def suggest_fixes(self, error_info: Dict[str, Any], proof_script: str) -> str:
        """
        Suggest fixes for common Lean errors.
        
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
        
        if error_type == "syntax_error":
            # For syntax errors, add a comment with the suggestion
            lines[line_index] = f"{lines[line_index]} -- {error_info['suggestion']}"
            
        elif error_type == "unknown_identifier":
            identifier = error_info.get("identifier", "")
            
            # Try to add common imports for standard libraries
            if identifier in ["Nat", "nat"]:
                lines.insert(0, "import Mathlib.Data.Nat.Basic")
            elif identifier in ["Int", "int"]:
                lines.insert(0, "import Mathlib.Data.Int.Basic")
            elif identifier in ["Real", "real"]:
                lines.insert(0, "import Mathlib.Data.Real.Basic")
            elif identifier in ["Ring", "ring"]:
                lines.insert(0, "import Mathlib.Algebra.Ring.Basic")
            
        elif error_type == "incomplete_proof":
            # Try to find where the proof ends
            for i in range(len(lines) - 1, -1, -1):
                if re.search(r'\bQed\b|\bEnd\b|\bexact\b', lines[i]):
                    # Replace with sorry
                    lines[i] = lines[i].replace("Qed", "sorry").replace("End", "sorry")
                    break
        
        return '\n'.join(lines)
    
    def _find_lean_executable(self) -> Optional[str]:
        """
        Find the Lean executable.
        
        Returns:
            Path to lean or None if not found
        """
        # Check in standard locations
        standard_paths = [
            "lean",  # If in PATH
            "/usr/bin/lean",
            "/usr/local/bin/lean",
            "C:\\Program Files\\Lean\\bin\\lean.exe",
            "C:\\Lean\\bin\\lean.exe",
            os.path.expanduser("~/.elan/bin/lean")
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
    
    def _process_feedback_impl(self, error_message: str) -> Dict[str, Any]:
        """
        Process error feedback from Lean.
        
        Args:
            error_message: The error message from Lean
            
        Returns:
            Dictionary with structured error information
        """
        return self.process_error(error_message)


# Standalone functions for use in other modules

def verify_lean_proof(proof_script: str, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Verify a Lean proof script.
    
    Args:
        proof_script: The Lean proof script to verify
        filename: Optional filename to save the script to
        
    Returns:
        Tuple of (verification success, error message if any)
    """
    verifier = LeanVerifier()
    return verifier.verify_proof(proof_script, filename)

def analyze_lean_error(error_message: str) -> Dict[str, Any]:
    """
    Analyze a Lean error message.
    
    Args:
        error_message: The Lean error message
        
    Returns:
        Dictionary with structured error information
    """
    verifier = LeanVerifier()
    return verifier.process_error(error_message)

def start_interactive_lean_session(proof_script: str) -> str:
    """
    Start an interactive Lean session.
    
    Args:
        proof_script: The Lean proof script
        
    Returns:
        Path to the temporary file
    """
    verifier = LeanVerifier()
    return verifier.interactive_session(proof_script)