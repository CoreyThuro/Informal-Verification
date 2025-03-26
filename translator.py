"""
Enhanced translator module for proof translation.
"""

from typing import Dict, Any, List, Optional
import re

from patterns.recognizer import recognize_pattern
from patterns.translators.evenness import translate_evenness_proof
from patterns.translators.induction import translate_induction_proof
from patterns.translators.contradiction import translate_contradiction_proof
from patterns.translators.cases import translate_cases_proof
from patterns.translators.direct import translate_direct_proof
from coq.verifier import verify_coq_proof
from coq.feedback import apply_feedback
from knowledge.kb import KnowledgeBase

class ProofTranslator:
    """
    Main translator class for converting informal proofs to Coq.
    """
    
    def __init__(self):
        """Initialize the translator."""
        self.kb = KnowledgeBase()
    
    def translate(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Translate an informal proof to a formal Coq proof.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            Dictionary with translation results
        """
        # Detect pattern and domain
        pattern, pattern_info = recognize_pattern(theorem_text, proof_text)
        domain = self._detect_domain(theorem_text, proof_text)
        
        print(f"Detected pattern: {pattern}, domain: {domain}")
        
        # Generate formal proof based on pattern
        if pattern == "evenness":
            variable = pattern_info.get("variable", "n")
            formal_proof = translate_evenness_proof(variable, domain)
        elif pattern == "induction":
            variable = pattern_info.get("variable", "n")
            formal_proof = translate_induction_proof(variable, theorem_text, proof_text, domain)
        elif pattern == "contradiction":
            variables = pattern_info.get("variables", ["n"])
            formal_proof = translate_contradiction_proof(theorem_text, proof_text, variables, domain)
        elif pattern == "cases":
            variables = pattern_info.get("variables", ["n"])
            case_var = pattern_info.get("case_var", variables[0] if variables else "n")
            formal_proof = translate_cases_proof(theorem_text, proof_text, variables, case_var, domain)
        else:
            variables = pattern_info.get("variables", ["n"])
            formal_proof = translate_direct_proof(theorem_text, proof_text, variables, domain)
        
        # Verify the proof
        verified, error = verify_coq_proof(formal_proof)
        
        # Apply feedback if verification failed
        if not verified and error:
            fixed_proof = apply_feedback(formal_proof, error)
            reverified, new_error = verify_coq_proof(fixed_proof)
            
            if reverified:
                formal_proof = fixed_proof
                verified = True
                error = None
            else:
                # Try additional domain-specific fixes
                domain_fixed_proof = self._apply_domain_fixes(fixed_proof, new_error, domain)
                if domain_fixed_proof != fixed_proof:
                    domain_verified, domain_error = verify_coq_proof(domain_fixed_proof)
                    if domain_verified:
                        formal_proof = domain_fixed_proof
                        verified = True
                        error = None
                    else:
                        error = domain_error
        
        return {
            "formal_proof": formal_proof,
            "verified": verified,
            "error_message": error,
            "pattern": pattern,
            "domain": domain
        }
    
    def _detect_domain(self, theorem_text: str, proof_text: str) -> str:
        """
        Detect the mathematical domain.
        
        Args:
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            Domain code (e.g., "11" for number theory)
        """
        combined = f"{theorem_text} {proof_text}".lower()
        
        # Domain keywords from knowledge base
        domain_keywords = {
            "11": ["prime", "number", "integer", "divisible", "even", "odd", "gcd"],
            "12-20": ["group", "ring", "field", "algebra", "vector", "matrix"],
            "26-42": ["limit", "continuous", "derivative", "integral", "function"],
            "54-55": ["topology", "open", "closed", "compact", "connected"]
        }
        
        # Count keywords for each domain
        counts = {domain: 0 for domain in domain_keywords}
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in combined:
                    counts[domain] += 1
        
        # Return domain with highest count, or default
        if max(counts.values()) > 0:
            return max(counts.items(), key=lambda x: x[1])[0]
        
        return "00"  # General mathematics
    
    def _apply_domain_fixes(self, proof_script: str, error_message: str, domain: str) -> str:
        """
        Apply domain-specific fixes to the proof.
        
        Args:
            proof_script: The proof script
            error_message: The error message
            domain: The mathematical domain
            
        Returns:
            Fixed proof script
        """
        # Add domain-specific imports if needed
        imports = self.kb.get_imports_for_domain(domain)
        
        # Check if imports are already in the script
        for import_stmt in imports:
            if import_stmt not in proof_script:
                proof_script = import_stmt + "\n" + proof_script
        
        # Apply domain-specific fixes
        if domain == "11":  # Number theory
            # Add ring tactic if missing
            if "ring" not in proof_script and "Error: The reference ring" in error_message:
                proof_script = re.sub(r'Proof\.', 'Proof.\n  ring.', proof_script)
        
        return proof_script