"""
Enhanced feedback processing for Coq errors.
"""

import re
from typing import Dict, Any, List, Tuple

def analyze_error(error_message: str) -> Dict[str, Any]:
    """
    Analyze a Coq error message.
    
    Args:
        error_message: The error message
        
    Returns:
        Dictionary with error analysis
    """
    # Initialize default analysis
    analysis = {
        "type": "unknown",
        "message": error_message,
        "fixes": []
    }
    
    # Check for missing reference
    ref_match = re.search(r"The reference ([^ ]+) was not found", error_message)
    if ref_match:
        missing_ref = ref_match.group(1)
        analysis["type"] = "missing_reference"
        analysis["missing_reference"] = missing_ref
        
        # Map common references to required imports
        ref_imports = {
            "ring": "Require Import Ring.",
            "Z": "Require Import ZArith.",
            "R": "Require Import Reals.",
            "Nat": "Require Import Arith.",
            "field": "Require Import Field.",
            "lia": "Require Import Lia.",
            "omega": "Require Import Omega."
        }
        
        if missing_ref in ref_imports:
            analysis["fixes"].append(f"add_import:{ref_imports[missing_ref]}")
        else:
            analysis["fixes"].append("add_import:generic")
        
        return analysis
    
    # Check for syntax error
    if "Syntax error" in error_message:
        analysis["type"] = "syntax_error"
        analysis["fixes"].append("check_syntax")
        
        # Look for specific syntax errors
        if "The term" in error_message and "has type" in error_message:
            analysis["fixes"].append("fix_type_mismatch")
        
        return analysis
    
    # Check for type mismatch
    if "Unable to unify" in error_message:
        analysis["type"] = "type_mismatch"
        analysis["fixes"].append("check_types")
        
        # Extract the types
        types_match = re.search(r'Unable to unify "([^"]+)" with "([^"]+)"', error_message)
        if types_match:
            analysis["expected_type"] = types_match.group(1)
            analysis["actual_type"] = types_match.group(2)
        
        return analysis
    
    # Check for incomplete proof
    if "Attempt to save an incomplete proof" in error_message:
        analysis["type"] = "incomplete_proof"
        analysis["fixes"].append("add_qed")
        return analysis
    
    return analysis

def apply_feedback(proof_script: str, error_message: str) -> str:
    """
    Apply feedback to fix common errors.
    
    Args:
        proof_script: The original proof script
        error_message: The error message
        
    Returns:
        Fixed proof script
    """
    # Analyze the error
    analysis = analyze_error(error_message)
    fixes = analysis.get("fixes", [])
    
    # Apply fixes based on analysis
    for fix in fixes:
        if fix.startswith("add_import:"):
            # Add import if not already present
            import_stmt = fix.split(":", 1)[1]
            if import_stmt not in proof_script:
                proof_script = import_stmt + "\n" + proof_script
        
        elif fix == "check_syntax":
            # Try to fix common syntax errors
            proof_script = fix_syntax_errors(proof_script)
        
        elif fix == "check_types":
            # Try to fix type mismatches
            if "expected_type" in analysis and "actual_type" in analysis:
                proof_script = fix_type_mismatch(
                    proof_script, analysis["expected_type"], analysis["actual_type"]
                )
        
        elif fix == "add_qed":
            # Ensure the proof ends with Qed
            if not proof_script.rstrip().endswith("Qed."):
                proof_script = proof_script.rstrip() + "\nQed."
    
    return proof_script

def fix_syntax_errors(proof_script: str) -> str:
    """Fix common syntax errors in Coq proofs."""
    lines = proof_script.split("\n")
    fixed_lines = []
    
    for line in lines:
        # Ensure tactics end with a period
        if re.search(r'^\s*(intro|intros|simpl|auto|ring|apply|destruct|induction|exists)\s', line) and not line.rstrip().endswith("."):
            line = line.rstrip() + "."
        
        # Fix common issues with exists
        line = re.sub(r'exists\s+([a-zA-Z0-9_]+)\s*$', r'exists \1.', line)
        
        fixed_lines.append(line)
    
    return "\n".join(fixed_lines)

def fix_type_mismatch(proof_script: str, expected_type: str, actual_type: str) -> str:
    """Attempt to fix type mismatches."""
    # Add coercion for nat/Z mismatches
    if ("nat" in expected_type and "Z" in actual_type) or ("Z" in expected_type and "nat" in actual_type):
        # Add conversion function
        if "Z.of_nat" not in proof_script and "nat_of_Z" not in proof_script:
            # Add ZArith import if needed
            if "Require Import ZArith" not in proof_script:
                proof_script = "Require Import ZArith.\n" + proof_script
            
            # We don't modify the proof directly as it's hard to know where to add the coercion
            # Instead add a comment suggesting the fix
            proof_script += "\n(* Type mismatch detected: Try using Z.of_nat or nat_of_Z for conversion *)"
    
    return proof_script