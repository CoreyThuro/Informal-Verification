"""
Enhanced translator for evenness proofs with deeper semantic understanding.
"""

from typing import Dict, Any, List, Optional
import re

def translate_evenness_proof(variable: str, domain: str = "11", theorem_text: str = None, proof_text: str = None, 
                            structure_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a Coq proof for the evenness of expressions, with enhanced semantic understanding.
    
    Args:
        variable: The primary variable used in the proof
        domain: The mathematical domain
        theorem_text: Optional theorem text for more context-aware translation
        proof_text: Optional proof text for more context-aware translation
        structure_info: Optional structural information from enhanced pattern recognition
        
    Returns:
        Coq proof script
    """
    # Determine the appropriate imports based on domain
    imports = [
        "Require Import Arith.",
        "Require Import Lia."
    ]
    
    if domain == "11":  # Number theory
        imports.extend([
            "Require Import ZArith.",
            "Require Import Znumtheory."
        ])
    
    # Extract the expression being proven even
    expression = f"{variable} + {variable}"
    if theorem_text:
        # Try to extract a more specific expression from the theorem
        expr_match = re.search(r'([a-zA-Z0-9\s\+\-\*/\^]+) is even', theorem_text, re.IGNORECASE)
        if expr_match:
            candidate = expr_match.group(1).strip()
            if variable in candidate:
                expression = candidate
    
    # Determine if we're proving divisibility by 2 or existence of k where expr = 2k
    use_divisibility = False
    if proof_text and ("divisible" in proof_text.lower() or "divides" in proof_text.lower()):
        use_divisibility = True
    
    # Generate appropriate theorem statement
    if use_divisibility:
        theorem = f"Theorem evenness_proof: forall {variable}: nat, 2 | ({expression})."
    else:
        theorem = f"Theorem evenness_proof: forall {variable}: nat, exists k: nat, {expression} = 2 * k."
    
    # Generate the proof based on the expression and approach
    if expression == f"{variable} + {variable}" or expression == f"2 * {variable}":
        # Simple case: n+n or 2*n
        if use_divisibility:
            proof = f"""Proof.
  intros {variable}.
  exists {variable}.
  simpl.
  reflexivity.
Qed."""
        else:
            proof = f"""Proof.
  intros {variable}.
  exists {variable}.
  ring.
Qed."""
    else:
        # More complex expression
        if use_divisibility:
            proof = f"""Proof.
  intros {variable}.
  unfold "|" in *.
  exists {variable}.
  (* Simplify the expression *)
  simpl.
  (* Use algebraic manipulation to show it equals 2k *)
  ring.
Qed."""
        else:
            proof = f"""Proof.
  intros {variable}.
  exists {variable}.
  (* Simplify the expression *)
  simpl.
  (* Use algebraic manipulation to show it equals 2k *)
  ring.
Qed."""
    
    # Add comments explaining the proof approach
    header_comment = """(* 
 * Proof of evenness using the definition that a number n is even
 * if and only if there exists an integer k such that n = 2k.
 *
 * Strategy: Directly provide the witness k and prove the equality.
 *)
"""
    
    return "\n".join([header_comment] + imports + ["", theorem, proof])