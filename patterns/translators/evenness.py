"""
Specialized translator for evenness proofs.
"""

def translate_evenness_proof(variable: str, domain: str = "11") -> str:
    """
    Generate a Coq proof for the evenness of x+x.
    
    Args:
        variable: The variable used in the proof
        domain: The mathematical domain
        
    Returns:
        Coq proof script
    """
    imports = [
        "Require Import Arith.",
        "Require Import Lia."
    ]
    
    theorem = f"Theorem example: forall {variable}: nat, exists k: nat, {variable} + {variable} = 2 * k."
    
    proof = f"""Proof.
  intros {variable}.
  exists {variable}.
  ring.
Qed."""
    
    return "\n".join(imports + ["", theorem, proof])