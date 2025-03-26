"""
Pattern recognition for mathematical proofs.
Identifies common proof patterns through simple text analysis.
"""

import re
from typing import Dict, List, Tuple, Optional, Any

def recognize_pattern(theorem_text: str, proof_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Recognize the pattern in a mathematical proof.
    
    Args:
        theorem_text: The theorem statement
        proof_text: The proof text
        
    Returns:
        Tuple of (pattern_name, pattern_info)
    """
    pattern_info = {}
    
    # Convert to lowercase for case-insensitive matching
    theorem_lower = theorem_text.lower()
    proof_lower = proof_text.lower()
    combined = f"{theorem_lower} {proof_lower}"
    
    # Extract variables first (used by all patterns)
    variables = _extract_variables(combined)
    pattern_info["variables"] = variables
    
    # Check for evenness proofs (highest priority)
    if _is_evenness_proof(theorem_lower, proof_lower):
        variable = _extract_evenness_variable(theorem_lower, proof_lower)
        pattern_info["variable"] = variable
        return "evenness", pattern_info
    
    # Check for induction proofs
    if _is_induction_proof(proof_lower):
        variable = _extract_induction_variable(proof_lower)
        pattern_info["variable"] = variable
        return "induction", pattern_info
    
    # Check for case analysis proofs
    if _is_cases_proof(proof_lower):
        case_var = _extract_case_variable(proof_lower, variables)
        pattern_info["case_var"] = case_var
        return "cases", pattern_info
    
    # Check for contradiction proofs
    if _is_contradiction_proof(proof_lower):
        return "contradiction", pattern_info
    
    # Fall back to direct proof
    return "direct", pattern_info

def _is_evenness_proof(theorem_text: str, proof_text: str) -> bool:
    """Check if this is an evenness proof about x+x."""
    # Check for patterns like "x+x is even" or "n+n = 2*n"
    combined = f"{theorem_text} {proof_text}"
    
    # Check for key indicators
    has_evenness = re.search(r'\b(even|divisible by 2|multiple of 2)\b', combined)
    has_x_plus_x = re.search(r'\b([a-z])\s*\+\s*\1\b', combined)
    has_2x = re.search(r'2\s*\*\s*([a-z])', combined)
    
    return has_evenness and (has_x_plus_x or has_2x)

def _extract_evenness_variable(theorem_text: str, proof_text: str) -> str:
    """Extract the variable from an evenness proof."""
    combined = f"{theorem_text} {proof_text}"
    
    # Look for x+x pattern
    match = re.search(r'\b([a-z])\s*\+\s*\1\b', combined)
    if match:
        return match.group(1)
    
    # Look for 2*x pattern
    match = re.search(r'2\s*\*\s*([a-z])', combined)
    if match:
        return match.group(1)
    
    # Default variable if no match
    return "n"

def _is_induction_proof(proof_text: str) -> bool:
    """Check if this is a proof by induction."""
    induction_markers = [
        r'\b(induction|inductive)\b',
        r'\bbase case\b',
        r'\binductive step\b',
        r'\bhypothesis\b.*\bk\b',
        r'\b([a-z])\s*=\s*0\b.*\b\1\s*=\s*k\s*\+\s*1\b'
    ]
    
    for marker in induction_markers:
        if re.search(marker, proof_text):
            return True
    
    return False

def _extract_induction_variable(proof_text: str) -> str:
    """Extract the induction variable."""
    # Common induction variables
    common_vars = ['n', 'k', 'm', 'i']
    
    # Look for explicit mentions
    for var in common_vars:
        if re.search(rf'induction on \b{var}\b', proof_text):
            return var
    
    # Look for base case patterns like "n = 0"
    for var in common_vars:
        if re.search(rf'\b{var}\s*=\s*0\b', proof_text):
            return var
    
    # Default to 'n' if no clear variable found
    return "n"

def _is_contradiction_proof(proof_text: str) -> bool:
    """Check if this is a proof by contradiction."""
    contradiction_markers = [
        r'\b(contradiction|contrary|absurd)\b',
        r'\bsuppose not\b',
        r'\bassume.*not\b'
    ]
    
    for marker in contradiction_markers:
        if re.search(marker, proof_text):
            return True
    
    return False

def _extract_variables(text: str) -> List[str]:
    """Extract potential variables from the text."""
    # Extract single letter variables
    variables = set(re.findall(r'\b([a-z])\b', text))
    
    # If no variables found, provide common defaults
    if not variables:
        variables = {'n', 'x'}
    
    return sorted(list(variables))


def _is_cases_proof(proof_text: str) -> bool:
    """Check if this is a proof by case analysis."""
    cases_markers = [
        r'\bcase\b',
        r'\bcases\b',
        r'\bfirst case\b',
        r'\bsecond case\b',
        r'\beither\b.*\bor\b',
        r'\bsplit into\b'
    ]
    
    for marker in cases_markers:
        if re.search(marker, proof_text):
            return True
    
    return False

def _extract_case_variable(proof_text: str, variables: List[str]) -> str:
    """Extract the variable used for case analysis."""
    # Look for explicit mentions
    for var in variables:
        if re.search(rf'case.*?{var}\b', proof_text):
            return var
    
    # Check for cases on variables
    for var in variables:
        if re.search(rf'cases.*?{var}\b', proof_text):
            return var
    
    # Default to first variable or 'n'
    return variables[0] if variables else "n"