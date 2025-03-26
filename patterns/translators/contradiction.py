"""
Specialized translator for contradiction proofs.
"""

from knowledge.kb import KnowledgeBase

def translate_contradiction_proof(theorem_text: str, proof_text: str, variables: list, domain: str = "00") -> str:
    """
    Generate a Coq proof by contradiction.
    
    Args:
        theorem_text: The theorem text
        proof_text: The proof text
        variables: List of variables in the proof
        domain: The mathematical domain
        
    Returns:
        Coq proof script
    """
    # Get imports from knowledge base
    kb = KnowledgeBase()
    imports = kb.get_imports_for_domain(domain)
    
    # Format variables for the theorem
    if variables:
        vars_str = ", ".join([f"{v}: nat" for v in variables])
        theorem = f"Theorem example: forall {vars_str}, P({', '.join(variables)})."
    else:
        theorem = "Theorem example: P."
    
    # Generate proof
    proof = """Proof.
  (* Proof by contradiction *)
  intros.
  assert (H : ~P).
  {
    intro contra.
    (* Derive contradiction *)
    
    (* ... *)
    
    contradiction.
  }
  contradiction.
Qed."""
    
    return "\n".join(imports + ["", theorem, proof])